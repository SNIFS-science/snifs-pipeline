import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from scipy.ndimage import shift
from scipy.optimize import minimize
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.optim import LBFGS

# Snifs constants
from pipeline.common.model_params import (
    A0_PARAMS,
    B0_PARAMS,
    QUARTIC_LINEAR_PARAMS,
    Z_1ST,
    Z_2ND,
    Z_4TH,
    default_shift_offsets,
)

# full transparency I did make use of AI so if the comments or wording look wierd it's because it might be Ai
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
READOUT_NOISE = 3.0
NUM_SPAXELS = 225
NUM_SPEC_ELEMENTS = 1400
CCD_SHAPE = (4096, 2048)

# Saving relaxation
MIN_DELTA_IMPROVEMENT = 250.0
MAX_PATIENCE = 3


def pseudo_voigt_torch(x, sigma_g, sigma_l, amplitude=1.0, eta=0.5):
    """Fully differentiable and normalized peak amplitude."""
    g = torch.exp(-(x**2) / (2 * sigma_g**2))
    l = sigma_l**2 / (x**2 + sigma_l**2)
    return amplitude * (eta * g + (1 - eta) * l)


def gaussian_blur_tensor(tensor, kernel_size=21, sigma=3.0):
    """Applies a fast, mathematically separable 1D-by-1D Gaussian blur
    to a 2D tensor to widen the gradient capture basin.
    """
    x = torch.arange(kernel_size, dtype=torch.float64, device=DEVICE) - (kernel_size - 1) / 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    kernel1d = (gauss / gauss.sum()).view(1, 1, -1)  # Shape: [1, 1, kernel_size]

    img = tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    padding = kernel_size // 2

    # Separable passes: vertical 1D convolution followed by horizontal 1D convolution
    blurred = F.conv2d(img, kernel1d.unsqueeze(2), padding=(padding, 0))
    blurred = F.conv2d(blurred, kernel1d.unsqueeze(3), padding=(0, padding))
    return blurred.squeeze()


class SNIFSForwardModel(torch.nn.Module):
    def __init__(self, params_dict, initial_spectrum):
        super().__init__()
        # Freeze physical hardware constants as non-differentiable float64 buffers
        self.register_buffer("A0", torch.tensor(params_dict["A0"], dtype=torch.float64))
        self.register_buffer("B0", torch.tensor(params_dict["B0"], dtype=torch.float64))
        self.register_buffer("QUARTIC", torch.tensor(params_dict["QUARTIC"], dtype=torch.float64))
        self.register_buffer("Z1", torch.tensor(params_dict["Z1"], dtype=torch.float64))
        self.register_buffer("Z2", torch.tensor(params_dict["Z2"], dtype=torch.float64))
        self.register_buffer("Z4", torch.tensor(params_dict["Z4"], dtype=torch.float64))

        # Initialize offset tracking parameter using param dict
        init_offsets = torch.tensor(default_shift_offsets, dtype=torch.float64, device=DEVICE).flatten()
        self.offsets = torch.nn.Parameter(init_offsets)
        self.widths = torch.nn.Parameter(torch.ones(NUM_SPAXELS, dtype=torch.float64, device=DEVICE))

        # Initialize spectrum
        if isinstance(initial_spectrum, torch.Tensor):
            spec_tensor = initial_spectrum.clone().detach().to(dtype=torch.float64, device=DEVICE)
        else:
            spec_tensor = torch.tensor(initial_spectrum, dtype=torch.float64, device=DEVICE)
        self.spectrum = torch.nn.Parameter(spec_tensor)

    def forward(self, target_spaxel_range, override_offsets=None):
        img = torch.zeros(CCD_SHAPE, dtype=torch.float64, device=DEVICE)
        all_x = torch.arange(CCD_SHAPE[1], device=DEVICE, dtype=torch.float64)
        current_offsets = override_offsets if override_offsets is not None else self.offsets

        for idx in target_spaxel_range:
            # Locate true spatial start baseline for this trace segment on the physical CCD grid
            a0 = int(self.A0[idx].item())
            row_start = max(0, a0)
            row_end = min(CCD_SHAPE[0], a0 + NUM_SPEC_ELEMENTS)

            # Map parameters strictly inside bounds if cross-dispersions fall inside bounds
            if row_end > row_start:
                # Calculate relative indexing boundaries (0 to 1400 slice space)
                slice_start = row_start - a0
                slice_end = row_end - a0

                # do the polinomial coordinates (input)
                lam_slice = torch.arange(slice_start, slice_end, device=DEVICE, dtype=torch.float64)

                # quarticc polynomial
                curve_x_slice = (
                    self.B0[idx]
                    + self.Z1[idx] * lam_slice
                    + self.Z2[idx] * (lam_slice**2)
                    + self.Z4[idx] * (lam_slice**4)
                    + current_offsets[idx] * 0.1
                )

                dx = all_x - curve_x_slice.reshape(-1, 1)
                sigma_g = 0.6 * self.widths[idx]
                sigma_l = 1.4 * self.widths[idx]

                amplitude_slice = self.spectrum[idx][slice_start:slice_end].reshape(-1, 1)

                # Directly accumulate trace lines to their actual absolute row indexes
                img[row_start:row_end] += pseudo_voigt_torch(dx, sigma_g, sigma_l, amplitude=amplitude_slice)
        return img


def calculate_chi2_loss(model_img, data_img, weight_source="model"):
    """Calculates physical Chi2 discrepancy using inverse variance weighting."""
    diff = data_img - model_img
    if weight_source == "data":
        weight = 1.0 / (data_img + READOUT_NOISE**2)
    else:
        weight = 1.0 / (model_img + READOUT_NOISE**2)
    return torch.sum((diff**2) * weight)


# the optimization basically has 2 stages.
#STAGE 1 the warmup #STAGE 2 refinement
# the warmup blurrs the image to artificially create more similarity/overlap between the images. THis gives the
# gradient something to look at (this is important because we have a sparse image dominated by a few bright hotspots)
# In the warmup step the hope is to get into the ballpark so that int he next step (refinement)
# we can get a significant (nonzero) gradient without blurring. If it doesn't work I will implement multiple stages of
# blurring slowly decreasing the sigma for gaussian blurr in each step.
# I made it so refinement can happen multiple times for thsi reason. Right now it shoul dbe set to 1 for just the wramup
# or 2 for warm up and refinement once.
# antoher reason for refinement passes is maybe the approx hessian built up is bad


def optimize_calibration(target_img, initial_spectrum, target_spaxel_range, params_dict, num_refinement_passes=5):
    solver_model = SNIFSForwardModel(params_dict, initial_spectrum).to(DEVICE)

    solver_model.offsets.requires_grad = False
    solver_model.widths.requires_grad = False
    solver_model.spectrum.requires_grad = False

    active_offsets = torch.nn.Parameter(solver_model.offsets.data[target_spaxel_range].clone().double())
    mask_indices = torch.tensor(target_spaxel_range, device=DEVICE, dtype=torch.long)

    #
    target_img_blurred = gaussian_blur_tensor(target_img, kernel_size=21, sigma=0.2)

    state = {"eval_count": 0, "current_refinement_pass": 0, "force_stop": False, "phase": "WARMUP"}
    best_loss = float("inf")
    best_offsets_state = active_offsets.data.clone()
    previous_loss = float("inf")
    patience_counter = 0
    best_history = []
    run_saved = False

    def closure():
        nonlocal best_loss, best_offsets_state, previous_loss, patience_counter, run_saved
        optimizer.zero_grad()

        # Enforce parameter constraint mapping boundaries (+- 5.0 pixel range)
        active_offsets.data.clamp_(min=-5.0, max=5.0)

        base_offsets = solver_model.offsets.detach()
        current_offsets = base_offsets.scatter(0, mask_indices, active_offsets)

        full_predicted_img = torch.zeros(CCD_SHAPE, dtype=torch.float64, device=DEVICE)
        batch_size = 15
        for i in range(0, len(target_spaxel_range), batch_size):
            batch = target_spaxel_range[i : i + batch_size]
            full_predicted_img += solver_model(target_spaxel_range=batch, override_offsets=current_offsets)

        # --- PHASE DEPENDENT SYMMETRIC LOSS FUNCTION ---
        if state["phase"] == "WARMUP":
            eval_model = gaussian_blur_tensor(full_predicted_img, kernel_size=21, sigma=0.2)
            eval_target = target_img_blurred
            total_loss = calculate_chi2_loss(eval_model, eval_target, weight_source="data")
        else:
            eval_model = full_predicted_img
            eval_target = target_img
            total_loss = calculate_chi2_loss(eval_model, eval_target, weight_source="model")

        current_loss_val = total_loss.item()
        delta_chi2 = abs(previous_loss - current_loss_val) if previous_loss != float("inf") else float("inf")

        # Save historical best metrics on convergence milestones (THIs is very out of date)
        # Work on this last to determine
        if delta_chi2 < MIN_DELTA_IMPROVEMENT and not run_saved:
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_offsets_state = active_offsets.data.clone()
                best_history.append(active_offsets.mean().item())
                run_saved = True
                print(f"    [{state['phase']}] [SAVED] New Best Pass Chi2: {best_loss:.4f}")

        # stop trying to fit if the loss is changing by less than a certain amount right now that's 1.0 arbitrarily
        if state["phase"] == "REFINEMENT" and delta_chi2 < 1.0:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print("    [STOP] Noise floor hit. Halting loop execution.")
                state["force_stop"] = True
        elif state["phase"] == "REFINEMENT":
            patience_counter = 0

        total_loss.backward()
        previous_loss = current_loss_val
        state["eval_count"] += 1
        print(
            f"  [{state['phase']} Pass {state['current_refinement_pass']}] Eval {state['eval_count']} -> Loss: {current_loss_val:.4f}"
        )
        return total_loss

    print("\n>>> Launching GPU L-BFGS (Scale-Space Warmup) Calibration Engine...")
    for refinement_pass in range(num_refinement_passes):
        state["current_refinement_pass"] = refinement_pass
        state["eval_count"] = 0
        run_saved = False
        state["phase"] = "WARMUP" if refinement_pass == 0 else "REFINEMENT"

        # Rebuild optimizer to wipe out old curvature history when switching landscapes
        optimizer = LBFGS([active_offsets], lr=1.0, max_iter=10, history_size=10, line_search_fn="strong_wolfe")

        optimizer.step(closure)

        # Safe reinforcement parameter clamp checks post-step
        active_offsets.data.clamp_(min=-5.0, max=5.0)

        if state["force_stop"]:
            break

        avg_offset = active_offsets.mean().item()
        with torch.no_grad():
            curr_off = solver_model.offsets.detach().scatter(0, mask_indices, active_offsets)
            current_loss = calculate_chi2_loss(
                solver_model(target_spaxel_range, override_offsets=curr_off), target_img
            ).item()

        print(
            f"*** Pass {refinement_pass} ({state['phase']}) Complete! True Real Loss: {current_loss:.4f} | Avg Offset: {avg_offset:.4f}\n"
        )

    with torch.no_grad():
        best_offsets_state.clamp_(min=-5.0, max=5.0)
        solver_model.offsets[target_spaxel_range] = best_offsets_state

    return solver_model, best_history


# MSE loss for the translational fit
def global_translation_loss(shift_params, recon_img, target_img):
    """Calculates continuous image space tracking error under sub-pixel shifting."""
    dy, dx = shift_params
    # order=1 makes LERP for subpixel transformations
    shifted_img = shift(recon_img, shift=[dy, dx], order=1, mode="constant", cval=0.0)
    return np.sum((target_img - shifted_img) ** 2)


if __name__ == "__main__":
    Path("quicksaves_test").mkdir(parents=True, exist_ok=True)
    fits_path = "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/P25_194_024_004_03_B.fits"

    # Load fits file
    try:
        with fits.open(fits_path) as hdul:
            real_data = np.array(hdul[0].data, dtype=np.float64)
            real_data = np.nan_to_num(real_data, nan=0.0, posinf=0.0, neginf=0.0)
            real_data_tensor = torch.from_numpy(real_data).to(DEVICE).double()
            print(f"Successfully loaded FITS frame. Array Matrix: {real_data_tensor.shape}")
    except FileNotFoundError:
        print(f"FITS file not found at {fits_path}.")
        quit()

    # load spectrum
    try:
        with open("C:/Users/gibis/URAP/snifs-pipeline/src/pipeline/tasks/processing/arc_frame_spectrum.json", "r") as f:
            spec_data = np.array(json.load(f))
    except FileNotFoundError:
        print("JSON spectrum file not found.")
        quit()

    if spec_data.ndim == 1:
        spec_data = spec_data.reshape(NUM_SPAXELS, NUM_SPEC_ELEMENTS)

    # Pack official hardware baseline parameter profiles into dictionary mapping
    params_dict = {
        "A0": A0_PARAMS,
        "B0": B0_PARAMS,
        "QUARTIC": QUARTIC_LINEAR_PARAMS,
        "Z1": Z_1ST,
        "Z2": Z_2ND,
        "Z4": Z_4TH,
    }

    # make the total flux equal each other so the forard model doesn't have to fit for this
    print("\n>>> Executing Initial Coarse Flux Scale Calibration Pass...")
    with torch.no_grad():
        test_model = SNIFSForwardModel(params_dict, spec_data).to(DEVICE)
        initial_rendered_flux = torch.sum(test_model(target_spaxel_range=list(range(NUM_SPAXELS)))).item()

    # Compute the scaling ratio between the FITS and JSON
    real_target_flux = real_data_tensor.sum().item()
    flux_scale_factor = real_target_flux / (initial_rendered_flux + 1e-8)

    print(f"--> [FLUX SCALE] Real FITS Flux: {real_target_flux:.2f} | Model Base Flux: {initial_rendered_flux:.2f}")
    print(f"--> [FLUX SCALE] Pre-scaling JSON spectrum matrix by factor of: {flux_scale_factor:.4f}")

    # rescale the spectrum
    spec_data = spec_data * flux_scale_factor

    # this is the forward model or big matrix A
    master_model = SNIFSForwardModel(params_dict, spec_data).to(DEVICE)

    BATCH_SIZE = 15
    print(f"\n>>> Commencing Outside Batch-by-Batch Optimization Across {NUM_SPAXELS} Spaxels...")

    for start_idx in range(0, NUM_SPAXELS, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, NUM_SPAXELS)
        batch_range = list(range(start_idx, end_idx))

        print("\n" + "=" * 60)
        print(f" RUNNING OPTIMIZER: Spaxels {start_idx} to {end_idx - 1}")
        print("=" * 60)

        # Execute L-BFGS loop focused strictly on the shallow 15-spaxel calculation graph
        solved_batch_model, batch_history = optimize_calibration(
            target_img=real_data_tensor,
            initial_spectrum=spec_data,
            target_spaxel_range=batch_range,
            params_dict=params_dict,
            num_refinement_passes=1,
        )

        # Save the resulting optimized batch parameters safely back to the master tracker context
        with torch.no_grad():
            master_model.offsets.data[batch_range] = solved_batch_model.offsets.data[batch_range]

    print("\nRendering final optimized full-field forward model frame mapping...")
    full_range = list(range(0, NUM_SPAXELS))
    with torch.no_grad():
        # Generate the master image tensor directly on the GPU, avoiding CPU conversion to keep torch functions working downstream
        reconstructed_tensor = master_model(target_spaxel_range=full_range)

    # =========================================================================
    # --- NEW: GPU-ACCELERATED TWO-STAGE GLOBAL TRANSLATION ALIGNMENT ---
    # =========================================================================

    # -------------------------------------------------------------------------
    # testing movint he image into what I think by eye is the correct amount
    # I found out this was wrong and that each spaxel is shifted differently, but I think there's a pattern to the batches

    print("\n[TEST] Artificially perturbing reconstructed image on GPU: 6px right, 1px up...")
    reconstructed_tensor = torch.roll(reconstructed_tensor, shifts=(-1, 6), dims=(0, 1))
    # -------------------------------------------------------------------------

    print("\n>>> Launching Two-Stage Global 2D Detector Space Shift Optimization...")

    # 1COARSE ALIGNMENT. Since the image is sparse, there is no overlap so the gradient is zero. Use gaussian blurring to artifically create overlap and change loss landscape
    with torch.no_grad():
        real_data_blurred_tensor = gaussian_blur_tensor(real_data_tensor, kernel_size=31, sigma=4.0)
        recon_blurred_tensor = gaussian_blur_tensor(reconstructed_tensor, kernel_size=31, sigma=4.0)

    # transfer blurred tensors from GPU to cpu
    real_data_blurred = real_data_blurred_tensor.cpu().numpy()
    recon_blurred = recon_blurred_tensor.cpu().numpy()
    # transfer the real ones aswell
    real_data_np = real_data_tensor.cpu().numpy()
    reconstructed = reconstructed_tensor.cpu().numpy()

    # use neadler mead on cpu to fit for a global shift
    initial_translation_guess = [0.0, 0.0]  # by eye it looks like 6, -1 (x,y)

    coarse_result = minimize(
        global_translation_loss,
        initial_translation_guess,
        args=(recon_blurred, real_data_blurred),
        method="Nelder-Mead",
        options={"xatol": 1e-2, "fatol": 1e-1},  # Loose tolerances for fast alignment
    )

    coarse_dy, coarse_dx = coarse_result.x
    print(f"--> [STAGE 1 DONE] Coarse Alignment Found: dy = {coarse_dy:.4f}, dx = {coarse_dx:.4f}")

    # 2Fine alignment. We are now close enough that we don't have to use blurring for gradient to exist
    print("--> [STAGE 2] Refining alignment on sharp, unblurred images...")

    fine_result = minimize(
        global_translation_loss,
        [coarse_dy, coarse_dx],  # initialize with results from coarse alignment
        args=(reconstructed, real_data_np),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-3},  # Tight tolerances for precision
    )

    opt_dy, opt_dx = fine_result.x
    print(f"--> [GLOBAL SHIFT FOUND] Optimal Translation: dy = {opt_dy:.4f} pixels, dx = {opt_dx:.4f} pixels")

    # shift the reconstructed image by the calculated amount
    reconstructed = shift(reconstructed, shift=[opt_dy, opt_dx], order=1, mode="constant", cval=0.0)

    testnum = 6

    # save diagnostic plots
    np.save(f"quicksaves_test/real_data_{testnum}.npy", real_data_np)
    np.save(f"quicksaves_test/reconstructed_{testnum}.npy", reconstructed)

    error_map = np.abs(reconstructed - real_data_np)
    plt.imsave(f"quicksaves_test/error_map_{testnum}.png", error_map, cmap="hot")
    mse_map = np.square(reconstructed - real_data_np)
    plt.imsave(f"quicksaves_test/mse_map_{testnum}.png", mse_map, cmap="viridis")

    # TEST 3 FIX: Enforce vmin=0 to guarantee 0.0 maps strictly to black
    recon_vmax = np.percentile(reconstructed, 99.5)
    real_vmax = np.percentile(real_data_np, 99.5)
    plt.imsave(
        f"quicksaves_test/reconstructed_image_{testnum}.png", reconstructed, cmap="gray", vmin=0, vmax=recon_vmax
    )
    plt.imsave(f"quicksaves_test/true_image_{testnum}.png", real_data_np, cmap="gray", vmin=0, vmax=real_vmax)

    dr_noisy = max(1.0, real_data_np.max() - real_data_np.min())

    print("\n================ SYSTEM CALIBRATION METRICS ================")
    print(f"MSE:  {mse(real_data_np, reconstructed):.6f}")
    print(f"PSNR: {psnr(real_data_np, reconstructed, data_range=dr_noisy):.2f} dB")
    print(f"TEST NUMBER {testnum} HAS BEEN COMPLETED")
