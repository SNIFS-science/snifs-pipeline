import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import json
from pathlib import Path



#this is naive fit
from pipeline.tasks.processing.parallel_parameter_matrix_drafting import (
    SNIFSForwardModel, DEVICE, NUM_SPAXELS, NUM_SPEC_ELEMENTS, CCD_SHAPE
)

from pipeline.common.model_params import (
    A0_PARAMS, B0_PARAMS, ELL_A, ELL_B, ELL_C0, ELL_C1,
    LIN_CROSS, LIN_SPEC, QUAD_SPEC, QUARTIC_LINEAR_PARAMS, Z_1ST, Z_2ND, Z_4TH
)

def run_diagnostic_render():
    Path("diagnostics").mkdir(parents=True, exist_ok=True)
    fits_path = "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/P25_194_024_004_03_B.fits"
    json_path = "C:/Users/gibis/URAP/snifs-pipeline/src/pipeline/tasks/processing/arc_frame_spectrum.json"
    
    #load data
    print("Loading FITS and JSON data...")
    try:
        with fits.open(fits_path) as hdul:
            real_data = np.array(hdul[0].data, dtype=np.float64)
            real_data = np.nan_to_num(real_data, nan=0.0, posinf=0.0, neginf=0.0)
            real_data_tensor = torch.from_numpy(real_data).to(DEVICE).double()
    except FileNotFoundError:
        print(f"FITS file not found at {fits_path}.")
        return

    try:
        with open(json_path, "r") as f:
            spec_data = np.array(json.load(f))
            if spec_data.ndim == 1:
                spec_data = spec_data.reshape(NUM_SPAXELS, NUM_SPEC_ELEMENTS)
    except FileNotFoundError:
        print(f"JSON spectrum file not found at {json_path}.")
        return

    params_dict = {
        'A0': A0_PARAMS, 'B0': B0_PARAMS, 'QUARTIC': QUARTIC_LINEAR_PARAMS,
        'Z1': Z_1ST, 'Z2': Z_2ND, 'Z4': Z_4TH, 'LIN_SPEC': LIN_SPEC,
        'QUAD_SPEC': QUAD_SPEC, 'LIN_CROSS': LIN_CROSS, 'ELL_A': ELL_A,
        'ELL_B': ELL_B, 'ELL_C0': ELL_C0, 'ELL_C1': ELL_C1
    }

    # scale flux
    print("Calculating flux scale factor...")
    with torch.no_grad():
        try:
            test_model = SNIFSForwardModel(params_dict, initial_spectrum=spec_data, patch_size=100).to(DEVICE)
        except:
            test_model = SNIFSForwardModel(params_dict, initial_spectrum=spec_data).to(DEVICE)
        initial_rendered_flux = torch.sum(test_model(target_spaxel_range=list(range(NUM_SPAXELS)))).item()

    real_target_flux = real_data_tensor.sum().item()
    flux_scale_factor = real_target_flux / (initial_rendered_flux + 1e-8)
    spec_data_scaled = spec_data * flux_scale_factor

    print(f"Base Model Flux: {initial_rendered_flux:.2f} | Real FITS Flux: {real_target_flux:.2f}")
    print(f"Applied Scale Factor: {flux_scale_factor:.4f}")

    # make ideal image
    print("\nRendering the full ideal unfitted model (No Optimization)...")
    try:
        master_model = SNIFSForwardModel(params_dict, initial_spectrum=spec_data_scaled, patch_size=100).to(DEVICE)
    except:
        master_model = SNIFSForwardModel(params_dict, initial_spectrum=spec_data_scaled).to(DEVICE)
    
    with torch.no_grad():
        # Do a single forward pass over all spaxels
        unfitted_model_tensor = master_model(target_spaxel_range=list(range(NUM_SPAXELS)))
        unfitted_model_np = unfitted_model_tensor.cpu().numpy()

    print("Generating diagnostic plots...")
    diff_map = np.abs(real_data - unfitted_model_np)
    
    # Use percentiles for plotting so dead pixels or extreme bright spots don't wash out the image
    vmax_real = np.percentile(real_data, 99.5)
    vmax_model = np.percentile(unfitted_model_np, 99.5)
    vmax_diff = np.percentile(diff_map, 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    ax1 = axes[0]
    im1 = ax1.imshow(real_data, cmap='gray', origin='lower', vmin=0, vmax=vmax_real)
    ax1.set_title("True FITS Image")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(unfitted_model_np, cmap='gray', origin='lower', vmin=0, vmax=vmax_model)
    ax2.set_title("Ideal Unfitted Model")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = axes[2]
    im3 = ax3.imshow(diff_map, cmap='hot', origin='lower', vmin=0, vmax=vmax_diff)
    ax3.set_title("Absolute Difference Map")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    model_name = input("What is the model's name?")
    plt.tight_layout()
    plt.savefig(f"diagnostics/{model_name}_initial_render_comparison.png", dpi=300)
    print(f"Diagnostic image saved to 'diagnostics/{model_name}_initial_render_comparison.png'")
    
    np.save(f"diagnostics/{model_name}_unfitted_model.npy", unfitted_model_np)

if __name__ == "__main__":
    run_diagnostic_render()