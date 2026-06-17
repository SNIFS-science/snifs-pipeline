import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cps
import numpy as np
from cupyx.scipy.sparse import coo_matrix
from numpy.polynomial import Polynomial
from scipy import sparse

from pipeline.common.fitting_math__utils import pseudo_voigt
from pipeline.common.log import get_logger
from pipeline.common.model_params import (
    A0_PARAMS,
    A1_PARAMS,
    B0_PARAMS,
    B1_PARAMS,
    ELL_A,
    ELL_B,
    ELL_C0,
    ELL_C1,
    LIN_CROSS,
    LIN_SPEC,
    QUAD_SPEC,
    QUARTIC_LINEAR_PARAMS,
    Z_1ST,
    Z_2ND,
    Z_4TH,
    default_shift_offsets,
    default_width_offsets,
)
from pipeline.resolver.resolver import get_run_id
from pipeline.tasks.loaders import load_images_from_file

# load the parameterization of the properties of the optics system
SPAXEL_BIG_CACHE = []

# global offsets to the spectrum taken from the cross-correlation of the arc with a reference arc.
xoff = 0
yoff = 0

# make a grid as basis of the per spaxel and per wavelength element
x = np.arange(-50, 50, 1)
xv, yv = np.meshgrid(x, x)

n_cross = (np.log(0.0032) - np.log(6.90e-4)) / (np.log(4.917) - np.log(26.939))
n_spec = (np.log(0.0040) - np.log(8.447e-4)) / (np.log(4.52) - np.log(21.026))

start_time = time.time()

rowindex, columnindex = np.meshgrid(np.arange(0, 2048, 1), np.arange(0, 4096, 1))

READOUT_NOISE = 3.0


def pseudo_voigt_gpu(
    x: cp.ndarray,
    xo: cp.ndarray | float,
    wg: cp.ndarray | float,
    wl: cp.ndarray | float,
    n: float,
    eta: float,
    beta: float | None = None,
    l_off: float = 1.0,
) -> cp.ndarray:
    """VRAM-optimized pseudo-Voigt engine supporting clean 4D broadcasting
    without in-place shape mismatch errors.
    """
    if beta is None:
        beta = 1.0 - eta

    PV = None

    # --- GAUSSIAN STEP ---
    if beta != 0:
        # Standard assignment handles different shapes flawlessly
        PV = beta * cp.exp(-cp.log(2.0) * (x - xo) ** 2 / wg**2)
        cp.get_default_memory_pool().free_all_blocks()

    # --- LORENTZIAN STEP ---
    if eta != 0:
        base = cp.abs(x - xo)
        if n < 0:
            base = cp.maximum(base, 1e-8)

        # Standard division allows shape (15, 1400, 1, 1) to broadcast over (15, 1400, 202, 202)
        denom = (base**n) / (wl**n)
        del base

        # Add the scalar offset in-place (Safe because l_off is a single scalar)
        denom += l_off

        # Invert denom in-place using its own memory space
        cp.reciprocal(denom, out=denom)

        # Multiply by a scalar in-place (Safe because eta is a single scalar)
        denom *= eta

        if PV is None:
            PV = denom
        else:
            # Safe because both PV and denom are now confirmed (15, 1400, 202, 202) shapes
            PV += denom
            del denom

        cp.get_default_memory_pool().free_all_blocks()

    if PV is None:
        PV = cp.zeros_like(x, dtype=cp.float32)

    return PV


def compare_matrices_fast_sparse(gpu_sparse_matrix, cpu_sparse_matrix):
    print("=========================================")
    print("RUNNING SPARSE Comparison")
    print("=========================================")

    # 1. Ensure the GPU matrix is in CSR format on the device
    if hasattr(gpu_sparse_matrix, "tocsr"):
        gpu_csr = gpu_sparse_matrix.tocsr()
    else:
        gpu_csr = gpu_sparse_matrix

    # 2. Compute the squared Frobenius Norm of the CPU matrix directly from its sparse data array
    # (Frobenius norm of a sparse matrix is just the sum of its squared non-zero elements!)
    cpu_csr = cpu_sparse_matrix.tocsr()
    sq_norm_cpu = np.sum(cpu_csr.data**2)

    # 4. Calculate the difference matrix *as a sparse matrix*
    # Moving the CPU matrix pointers to the GPU is incredibly lightweight
    print("Computing sparse matrix difference on GPU...")
    cpu_csr_gpu = cps.csr_matrix(cpu_csr)

    # Direct sparse subtraction (only computes elements where data exists)
    diff_sparse = gpu_csr - cpu_csr_gpu

    # 5. Calculate final metrics
    sq_norm_diff = cp.sum(diff_sparse.data**2).item()
    max_pixel_drift = cp.max(cp.abs(diff_sparse.data)).item()

    total_frobenius_diff = np.sqrt(sq_norm_diff)
    total_frobenius_cpu = np.sqrt(sq_norm_cpu)

    rel_error = (total_frobenius_diff / total_frobenius_cpu) if total_frobenius_cpu > 0 else 0.0

    print("FINAL MATRIX VERIFICATION REPORT")
    print("=========================================")
    print(f"Relative Structural Error: {rel_error:.6e} ({rel_error * 100:.5f}%)")
    print(f"Worst-case Pixel Drift:    {max_pixel_drift:.6e}")
    print("=========================================")
    # --- Add these lines inside your fast diagnostic script ---
    print("CPU Sparse Matrix Non-Zero Row/Col layout:")
    print("Row indices sample (first 10):", cpu_csr.indices[:10])
    print("Col indices sample (first 10):", cpu_csr.has_canonical_format)

    print("\nGPU Sparse Matrix Non-Zero Row/Col layout:")
    print("Row indices sample (first 10):", gpu_csr.indices.get()[:10])
    return rel_error


def psf_calculation(sum_along_cross_disp: np.ndarray) -> np.ndarray:
    """Calculate the psf.

    Args:
        sum_along_cross_disp: numpy array of summed cross-dispersion profile for one spec element

    Returns:
        line profile function: adjusted cross-dispersion profile of length 100
    """
    if len(sum_along_cross_disp) != 100:
        toAppend = np.zeros(100)
        toAppend[: len(sum_along_cross_disp)] = sum_along_cross_disp
        return toAppend
    return sum_along_cross_disp


def save_results(data: np.ndarray, filename: str) -> None:
    """Wrapper save results function.

    Args:
        data: numpy array of residuals to save
        filename: name of the file to save the data to
    """
    logger = get_logger()
    flow_run_id = get_run_id()

    # REVERSE
    # output_location = (PUBLIC_PATH_MAP[flow_run_id] / filename).resolve()
    output_location = Path("./cpu_code").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving data to {output_location}")

    fullpath = output_location / filename
    try:
        np.save(fullpath, data)
    except:
        np.save(f"./cpu_code/{filename}", data)


def save_shift_results(data: np.ndarray, spaxel: int, is_translational_shift: bool) -> None:
    """Save the results of the shift we wanted to apply to the big matrix.

    Args:
        data: numpy array of residuals to save
        spaxel: spaxel number
        is_translational_shift: whether the data corresponds to translational shifts or width shifts
    """
    if is_translational_shift:
        shift_type = "translational"
    else:
        shift_type = "width"

    # REVERT
    # flow_run_id = get_run_id()
    flow_run_id = 0
    save_results(data, f"{shift_type}_shift_{spaxel}_{flow_run_id}.npy")


def make_matrix(
    spaxel: int,
    offsets: np.ndarray = default_shift_offsets,
    widths: np.ndarray = default_width_offsets,
    partial: bool = True,
    oversample_factor: int = 1,
) -> sparse.csr_matrix:
    """Makes the big matrix.

    Args:
        spaxel: spaxel number that will be adjusted (shifted or widened)
        offsets: array of extra offsets in cross-dispersion direction for all spaxels (225,)
            defaults to zeros
        widths: array of width scaling factors for all spaxels (225,)
            defaults to a multiplier of 1
        partial (optional): whether to compute only a subset of the matrix. Defaults to False.
        oversample_factor (optional): oversampling factor for the model. Defaults to 1.

    Returns:
        sparse.csr_matrix: sparse matrix representing the shifted model
    """
    logger = get_logger()
    flow_run_id = get_run_id()

    list_huge_matrix = []
    line_profile = []
    print(offsets[5])

    if partial:
        grouping = spaxel // 15
        spaxel_range = range(grouping * 15, (grouping + 1) * 15)  # list 0 to 14 right now
    else:
        spaxel_range = range(225)

    # [spaxel index] [x] [y]
    num_groups = 1
    # this is just the old loop but now we do 15 spaxels at a time

    # for the future when you do mor ethan 1 group we can store the cp.linspace thing outside

    gpu_start = time.time()
    group_size = 15
    for group in range(num_groups):
        print(f"evaluating {group}")
        # get starting and ending indices for the spaxelIDs
        start_ind = group * group_size
        end_ind = start_ind + group_size

        # find the place in the image where to put the spectrum per spaxel
        a0 = cp.array(A0_PARAMS[start_ind:end_ind] + yoff, dtype=cp.int32)
        a1 = cp.array(A1_PARAMS[start_ind:end_ind] + yoff + 1, dtype=cp.int32)
        b0 = cp.array(B0_PARAMS[start_ind:end_ind] + xoff - 50, dtype=cp.int32)

        # make sure left side is not off the edge for te artificial bounding box
        off = cp.full(group_size, 50, dtype=cp.int32)
        mask = b0 < 0
        off[mask] += b0[mask]
        b0[mask] = 0

        b1 = cp.array(B1_PARAMS[start_ind:end_ind] + xoff + 51, dtype=cp.int32)
        cp.clip(b1, a_min=0, a_max=2048, out=b1)

        # find biggest box so we can make all boxes the same size so they can just use 1 GPU instance
        # we know a1-a0 finna be 1400 so we already have that
        vert_span = 1400
        cross_span = cp.max(b1 - b0).item()
        # now the cp.full mane like [1400,1400,1400...] but reshape makes it a column vector 15 row 1 col matrix

        vert_vec = cp.full(group_size, vert_span, dtype=cp.int16).reshape(-1, 1)
        cross_vec = cp.full(group_size, cross_span, dtype=cp.int16).reshape(-1, 1)

        # row vectors with the scaling that you want so that you can make meshgrid with no loop
        vert_scale_row = cp.linspace(0, 1, oversample_factor * vert_span, dtype=cp.float64)
        cross_scale_row = cp.linspace(0, 1, oversample_factor * cross_span, dtype=cp.float64)

        # now take the tensor product of these to make the coord grid remmeber to go up in dim we do column row
        # cupy broadcasting will automatically extend the a0 [15,1] column vector so that it matches the dim of matrix
        # [15,1] + [15, 1400*oversample_factor]

        disp_grid = a0.reshape(-1, 1) + (vert_vec * vert_scale_row)
        cross_grid = b0.reshape(-1, 1) + (cross_vec * cross_scale_row)
        del vert_scale_row
        del cross_scale_row
        # the grids for exampel disp_grid takes in 2 indices [spaxel] [oversample index along dispersion axis]
        # it gives you CCD subpixel that you're on

        # now we make the polynomial change so it can handle matrices and vectors instead on GPU
        # First we need to get spaxel dependent coefficients into cupy arrays
        P0 = cp.broadcast_to(cp.array(QUARTIC_LINEAR_PARAMS[start_ind:end_ind])[:, None], (group_size, 1400))
        P1 = cp.broadcast_to(cp.array(Z_1ST[start_ind:end_ind])[:, None], (group_size, 1400))
        P2 = cp.broadcast_to(cp.array(Z_2ND[start_ind:end_ind])[:, None], (group_size, 1400))
        P4 = cp.broadcast_to(cp.array(Z_4TH[start_ind:end_ind])[:, None], (group_size, 1400))

        # evaluate_batch_polynomial(vertcoord,P0,P1,P2,P4)
        # print(cp.array(offsets[start_ind:end_ind]).shape)
        # offset_col = cp.broadcast(cp.array(offsets[start_ind:end_ind].T)[:,:,None],(5,15,1400))

        O1 = cp.array(offsets[start_ind:end_ind].T)[0].T[:, None]
        O2 = cp.array(offsets[start_ind:end_ind].T)[1].T[:, None]
        O3 = cp.array(offsets[start_ind:end_ind].T)[2].T[:, None]
        O4 = cp.array(offsets[start_ind:end_ind].T)[3].T[:, None]
        O5 = cp.array(offsets[start_ind:end_ind].T)[4].T[:, None]

        W1 = cp.array(widths[start_ind:end_ind].T)[0].T[:, None]
        W2 = cp.array(widths[start_ind:end_ind].T)[1].T[:, None]
        W3 = cp.array(widths[start_ind:end_ind].T)[2].T[:, None]
        W4 = cp.array(widths[start_ind:end_ind].T)[3].T[:, None]
        W5 = cp.array(widths[start_ind:end_ind].T)[4].T[:, None]

        # offset is 5 numbers fsr? then put that into polynomial then broadcast across the spectrum then add to wavelength solution
        # adjustmentP = cp.broadcast_to(cp.array(offsets[start_ind:end_ind])[:,None], (15,1400))
        x0 = cp.arange(0, 1400, 1, dtype=cp.int16)
        grid_view = cp.broadcast_to(x0[None, :], (group_size, 1400))

        # am not including global stuff right now
        curve = (
            P0
            + O1
            + ((P1 + O3) * grid_view)
            + ((P2 + O2) * (grid_view**2))
            + (O4 * (grid_view**3))
            + ((P4 + O5) * (grid_view**4))
            # DID NOT INCLUDE GLOBAL OFFSETS YET
        )
        # evaluate_batch_polynomial(x0,P0,P1,P2,P4)

        width_polynomial = W1 + (W2 * grid_view) + (W3 * (grid_view**2)) + (W4 * (grid_view**3)) + (W5 * (grid_view**4))

        # prolly faster and better for memory to make x0 from 1 list an dgrid view it
        # [15,1400] [spax] [spec]

        # #make wavelength spectrum box
        # c0 = (x0 - 50) * oversample_factor
        # cp.clip(c0, a_min=0, a_max=None, out = c0)
        # c1 = (x0 + 50) * oversample_factor
        # cp.clip(c1, a_min=0, a_max=1400*oversample_factor, out = c1)

        # c0 = cp.broadcast_to(c0[None, :], (15, 1400))
        # c1 = cp.broadcast_to(c1[None, :], (15, 1400))

        # # cp.rint().astype(cp.int16)

        # d0 = (curve - 50) * oversample_factor
        # cp.rint(d0).astype(cp.int16)
        # cp.clip(d0, 0, None, out=d0)

        # d1 = (curve + 50) * oversample_factor
        # cp.rint(d1).astype(cp.int16)
        # cp.clip(d1, 0, 1400 * oversample_factor - 1, out=d1)

        print(disp_grid.shape, cross_grid.shape)
        # disp_grid [15,1400 *oversample] [spaxel] [y_coords] output is CCD pixel coords subpixel
        # cross_grid is [15, max leftright spacing *oversample]  [spaxel] [x_coords] output is CCD leftright subpixel coords
        # Create local relative steps from -50 to +50 scaled by the oversample factor
        # For oversample=2, this creates 202 elements

        # To make curve work on all simultaneous we need to also make it 4D renamed and done later
        # Take curve and width which are [15,1400] => [spax] [spectrum] [x] [y]
        # curve_4D = curve[:, :, None, None]
        # width_4D = width_polynomial[:, :, None, None]

        # Do the same thing for the grid distortion constants
        QUAD_SPEC_4D = cp.array(QUAD_SPEC[start_ind:end_ind])[:, None, None, None]
        LIN_SPEC_4D = cp.array(LIN_SPEC[start_ind:end_ind])[:, None, None, None]
        LIN_CROSS_4D = cp.array(LIN_CROSS[start_ind:end_ind])[:, None, None, None]

        ELL_A_4D = cp.array(ELL_A[start_ind:end_ind])[:, None, None, None]
        ELL_B_4D = cp.array(ELL_B[start_ind:end_ind])[:, None, None, None]
        ELL_C0_4D = cp.array(ELL_C0[start_ind:end_ind])[:, None, None, None]
        ELL_C1_4D = cp.array(ELL_C1[start_ind:end_ind])[:, None, None, None]

        # I realized we artificially choose 50 as the size of the box, but in each batch to save memory and time we should find the
        # biggest box and set the size to that.

        # get max of semi and major axis
        max_b = cp.max(ELL_B_4D).item()  # spectrum
        max_a = cp.max(ELL_A_4D).item()  # crossdisp

        # add padding
        dynamic_half_box = int(np.ceil(max(max_b, max_a))) + 3

        c0 = (x0 - dynamic_half_box) * oversample_factor
        c0 = cp.rint(c0)  # Correctly round to nearest pixel integer
        cp.clip(c0, a_min=0, a_max=None, out=c0)
        c0 = c0.astype(cp.int16)  # Convert to integer type for indexing

        c1 = (x0 + dynamic_half_box) * oversample_factor
        c1 = cp.rint(c1)
        cp.clip(c1, a_min=0, a_max=1400 * oversample_factor, out=c1)
        c1 = c1.astype(cp.int16)

        # Broadcast spectrum boundaries across spaxel matrix blocks
        c0 = cp.broadcast_to(c0[None, :], (15, 1400))
        c1 = cp.broadcast_to(c1[None, :], (15, 1400))

        # 2. Generate vertical cross-dispersion trace bounds using dynamic box size
        d0 = (curve - dynamic_half_box) * oversample_factor
        d0 = cp.rint(d0)
        cp.clip(d0, a_min=0, a_max=None, out=d0)
        d0 = d0.astype(cp.int16)

        d1 = (curve + dynamic_half_box) * oversample_factor
        d1 = cp.rint(d1)
        cp.clip(d1, a_min=0, a_max=1400 * oversample_factor - 1, out=d1)
        d1 = d1.astype(cp.int16)

        # use arange because sizing with linspace can cause resolution to change
        step_size = 1.0 / oversample_factor

        y_local = cp.arange(-dynamic_half_box, dynamic_half_box + step_size, step_size, dtype=cp.float32)
        x_local = cp.arange(-dynamic_half_box, dynamic_half_box + step_size, step_size, dtype=cp.float32)

        # Shape: (1, 1, 202, 1) -> [spax] [spectrum] [x] [y]
        Y_local_4D = y_local[None, None, :, None]  # Shape: (1, 1, window_size_y, 1)
        # Shape: (1, 1, 1, 202) -> [spax] [spectrum] [x] [y]
        X_local_4D = x_local[None, None, None, :]  # Shape: (1, 1, 1, window_size_x)

        # STATIC VERSION OF CODE
        # half_box_len = 50
        # window_size = 2*half_box_len * oversample_factor + 2

        # # make the oversampled boxes
        # y_local = cp.linspace(-half_box_len, half_box_len, window_size, dtype=cp.float32)
        # x_local = cp.linspace(-half_box_len, half_box_len, window_size, dtype=cp.float32)

        # Y_local_4D = y_local[None, None, :, None]
        # X_local_4D = x_local[None, None, None, :]

        # the spec_trace tells you how much does x move as you change y it makes the smile
        # cross trace tells you how much does y change as you move accross the cross dispersion direction
        spec_trace = QUAD_SPEC_4D * (Y_local_4D**2) + LIN_SPEC_4D * Y_local_4D
        cross_trace = LIN_CROSS_4D * X_local_4D

        # make the warped coordinate matrices
        # xiii is the same 4D shape, but it outputs the distorted oversampled grid position
        # which has a unique warping for every single spaxel and wavelength
        xiii = X_local_4D - spec_trace
        yiii = Y_local_4D - cross_trace

        # making the elliptical mask

        # print("ELL C1 4d")
        # print(cp.mean(ELL_C1_4D))
        # print(cp.std(ELL_C1_4D))
        # print("ELL C0 4d")
        # print(cp.mean(ELL_C0_4D))
        # print(cp.std(ELL_C0_4D))
        # print("ELL B 4d")
        # print(cp.mean(ELL_B_4D))
        # print(cp.std(ELL_B_4D))
        # print("ELL A 4d")
        # print(cp.mean(ELL_A_4D))
        # print(cp.std(ELL_A_4D))

        mask_footprint = (
            cp.sqrt(((Y_local_4D - ELL_C1_4D) ** 2 / (ELL_B_4D**2)) + ((X_local_4D - ELL_C0_4D) ** 2 / (ELL_A_4D**2)))
            < 1
        )
        # for group 1 this used to be 15 spaxels, 1400, spectra, 201 grid by 201 grid now for group 1 it is 149 by 149

        # delete uneeded variables bc memory on gpu not enough
        del spec_trace, cross_trace, x_local, y_local
        del QUAD_SPEC_4D, LIN_SPEC_4D, LIN_CROSS_4D
        del ELL_A_4D, ELL_B_4D, ELL_C0_4D, ELL_C1_4D
        del Y_local_4D, X_local_4D

        cp.get_default_memory_pool().free_all_blocks()

        # reference code
        # wavelength_dep_width = float(width_polynomial(spec_element))
        # spectral = 0.8 * pseudo_voigt(
        #     np.abs(xiii), 0, 0.6 * wavelength_dep_width, 1.3 * wavelength_dep_width, 5.4, 0.6
        # ) + pseudo_voigt(np.abs(xiii), 0, 1.2, 0.2, -n_spec, 0.1, beta=0)
        # crossdis = 0.99 * pseudo_voigt(
        #     np.abs(yiii), 0, 0.6 * wavelength_dep_width, 1.4 * wavelength_dep_width, 5.2, 0.6
        # ) + pseudo_voigt(np.abs(yiii), 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10)

        # make all the things so it's spaxel, wavelength, minigrid
        # Width polynomial shape: (15, 1400) -> Reshape to (15, 1400, 1, 1)
        w_4d = width_polynomial[:, :, None, None]

        # eval pseudo voigt along the xiii (15 x 1400 x 202 x 202)

        # delete variables as they're not needed

        # swap this to inplace later
        abs_xiii = cp.abs(xiii)
        del xiii
        cp.get_default_memory_pool().free_all_blocks()
        # same inputs as the spectral for cpu
        # run unit tests to see if pseudo voigt gpu and pseudo voigt gpu give the same numbers
        spectral = 0.8 * pseudo_voigt_gpu(abs_xiii, 0, 0.6 * w_4d, 1.3 * w_4d, 5.4, 0.6)

        cp.get_default_memory_pool().free_all_blocks()

        # use inplace addition to avoid using more memory also same inputs as the reference code
        spectral += pseudo_voigt_gpu(abs_xiii, 0, 1.2, 0.2, -n_spec, 0.1, beta=0)

        # get rid of xiii
        del abs_xiii
        cp.get_default_memory_pool().free_all_blocks()

        # repeat the same thing as above but for y
        abs_yiii = cp.abs(yiii)
        del yiii
        cp.get_default_memory_pool().free_all_blocks()

        crossdis = 0.99 * pseudo_voigt_gpu(abs_yiii, 0, 0.6 * w_4d, 1.4 * w_4d, 5.2, 0.6)

        crossdis += pseudo_voigt_gpu(abs_yiii, 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10)
        del abs_yiii
        cp.get_default_memory_pool().free_all_blocks()

        # spectral should be  (15, 1400, 202, 202)

        # RENABLE
        # for i in range(group_size):
        #     # Calculate the true global spaxel ID matching your dataset
        #     global_spaxel_id = start_ind + i

        #     # Check against your exclusion condition:
        #     # Skip saving if this global index matches the offset 'spaxel' variable
        #     if 'spaxel' in locals() and global_spaxel_id == spaxel:
        #         print(f"Skipping cache for offset spaxel: {global_spaxel_id}")
        #         continue

        #     # Double-check that our target index fits within your 15-slot global cache bounds
        #     if global_spaxel_id < len(SPAXEL_BIG_CACHE):
        #         # 1. Slice out the 3D data slice on the GPU
        #         gpu_spaxel_slice = spectral[i]

        #         # 2. Pull the data from GPU VRAM to Host CPU RAM using .get()
        #         # We save it as standard 32-bit float array
        #         SPAXEL_BIG_CACHE[global_spaxel_id] = gpu_spaxel_slice.get()

        # Clean up any temporary indexing references
        if "gpu_spaxel_slice" in locals():
            del gpu_spaxel_slice
        cp.get_default_memory_pool().free_all_blocks()
        # =====================================================================
        # combine axes and apply the mask  (15 x 1400 x 202 x 202)

        # have not gotten to check things below here yet
        # reference
        # model = spectral * crossdis.T * (mask_footprint.T * 1.0)
        # model = model / np.max(model)
        # doing transpose on cross dis flips the last two axis because it's the small grid on cpu code
        # on gpu code it's swapping 3 and 2
        if "crossdis" in locals() and crossdis is not None:
            spectral *= cp.transpose(crossdis, (0, 1, 3, 2))
            try:
                del crossdis
            except NameError:
                pass
        else:
            raise RuntimeError("Cross dispersion is not defined")

        cp.get_default_memory_pool().free_all_blocks()

        # multiply with the mask same reasoning for transpose as before
        if "mask_footprint" in locals() and mask_footprint is not None:
            spectral *= cp.transpose(mask_footprint, (0, 1, 3, 2))
            try:
                del mask_footprint
            except NameError:
                pass
        else:
            raise RuntimeError("check why maskfootprintn is NONE or out of scope")
        cp.get_default_memory_pool().free_all_blocks()

        # 3. Final model assignment
        model_4d = spectral
        # =====================================================================

        # now we don't need spectral
        del spectral

        # reference
        # testModel = (
        #     model.reshape((model.shape[0] // oversample_factor, oversample_factor, -1, oversample_factor))
        #     .sum(axis=3)
        #     .sum(axis=1)
        # )

        # # image[a0:a1,b0:b1][c0:c1,d0:d1] = model.T
        # sum_cross_disp_axis = np.sum(testModel, axis=1)
        # line_profile.append(psf_calculation(sum_cross_disp_axis))

        # now we have to downsample back to ccd coords
        # I think the shape needs to go from (15, 1400, 202, 202) -> (15, 1400, 101, 2, 101, 2)
        # so the 101 defines the actual pixels then within that the 2s give subpixel arrangment
        # Hsub is model.shape[0] W_sub is tehcnically should be the same but what
        #  if we want to make boxes different like rectangles with dynamic ranging
        H_sub, W_sub = model_4d.shape[2], model_4d.shape[3]

        # # calculate the boundaries I forgot why I added this
        # # make H and W even numbers so when
        # H_even = (H_sub // oversample_factor) * oversample_factor
        # W_even = (W_sub // oversample_factor) * oversample_factor

        # 2. Calculate the final real CCD pixel dimensions
        H_real = H_sub // oversample_factor
        W_real = W_sub // oversample_factor

        # (15, 1400, 101, 2, 101, 2)
        testModel_4d = model_4d_even.reshape(
            group_size, 1400, H_real, oversample_factor, W_real, oversample_factor
        ).sum(axis=(3, 5))
        # testModel_4d (15, 1400, H_real, W_real) check if H_real and W_real are 101

        # 5. Clean up old references
        del model_4d, model_4d_even
        if "mask_footprint" in locals():
            del mask_footprint

        # 3 don't use slicing

        # create a local grid tracking the downsampled indices within the 101x101 patch
        # Shapes: (101, 1) and (1, 101)
        # assuming H_real and W_real are both 101 so if above step is wrong this one is too
        local_y_1d = cp.arange(H_real, dtype=cp.int32)
        local_x_1d = cp.arange(W_real, dtype=cp.int32)

        # local_Y counts vertically down rows, local_X counts horizontally across columns
        local_Y, local_X = cp.meshgrid(local_y_1d, local_x_1d, indexing="ij")

        # reshape so they can be broadcast to fit with the spaxel and wavelength for first 2 coords (1, 1, 101, 101)
        local_Y_4d = local_Y[None, None, :, :]
        local_X_4d = local_X[None, None, :, :]

        #
        # c0 = c0 // oversample_factor
        # c1 = c1 // oversample_factor
        # d0 = d0 // oversample_factor
        # d1 = d1 // oversample_factor
        #             row = rowind[mask_val[: len(rowind), : len(rowind.T)]]
        # colind = columnindex[a0:a1, b0:b1][c0:c1, d0:d1]
        # col = colind[mask_val[: len(rowind), : len(rowind.T)]]

        # data = testModel.T[: len(rowind), : len(rowind.T)][mask_val[: len(rowind), : len(rowind.T)]]
        # s_image = sparse.csr_matrix((data, (col, row)), shape=(4096, 2048))
        # The addition will now naturally produce the perfect (15, 1400, 101, 101) shape
        global_rows = (d0 // oversample_factor)[:, :, None, None] + local_Y_4d
        global_cols = (c0 // oversample_factor)[:, :, None, None] + local_X_4d

        # below this is noise? happens in cpu code so
        mask_val = testModel_4d > 1e-4

        # apply mask
        sparse_data = testModel_4d[mask_val]
        sparse_rows = global_rows[mask_val]
        sparse_cols = global_cols[mask_val]

        # start witht he flattened coordinates at the end (0-4095, 0-2047) to a flattened 1D index
        flat_ccd_indices = sparse_rows * 2048 + sparse_cols

        # Map each spaxel and wavelength to a unique row index in your final design matrix
        # Equation: row_idx = spaxel_ID * 1400 + wavelength_bin
        spaxel_indices = (
            cp.arange(group_size)[:, None] + start_ind
        )  # (15, tbd) add start_ind because if we're not in group 0
        wavelength_indices = cp.arange(1400)[None, :]  # (tbd but should become 15,1400 )

        #  each spectrum needs 1400 indices of space the adding wavelength indices will make each spectrum count up normally
        # so now we have [0-1399, 1400 - 2799, ...] 15 of these
        global_design_rows_2d = spaxel_indices * 1400 + wavelength_indices

        # now add the oversampled subgrid (15, 1400, 1, 1)
        global_design_rows_4d_base = global_design_rows_2d[:, :, None, None]

        # (assuming H_real and W_real are both 101)
        global_design_rows_4d = cp.broadcast_to(global_design_rows_4d_base, (group_size, 1400, H_real, W_real))

        # Now apply the mask
        sparse_design_rows = global_design_rows_4d[mask_val]

        # free memory
        del global_design_rows_4d_base, global_design_rows_4d
        cp.get_default_memory_pool().free_all_blocks()

        # Build the final massive Sparse Matrix directly in GPU Memory!
        batch_sparse_matrix = coo_matrix(
            (sparse_data, (sparse_design_rows, flat_ccd_indices)), shape=(group_size * 1400, 4096 * 2048)
        )
        gpu_done = time.time()

        cpu_start = time.time()

    for spaxel_ID in spaxel_range:  # looping thru them rn
        logger.info(f"starting spaxel {spaxel_ID}, total time = {time.time() - start_time}")
        # find the place in the image where to put the spectrum per spaxel
        a0 = int(A0_PARAMS[spaxel_ID] + yoff)
        a1 = int(A1_PARAMS[spaxel_ID] + yoff) + 1
        b0 = int(B0_PARAMS[spaxel_ID] + xoff - 50)

        # make sure left side is not off the edge for te artificial bounding box
        off = 50
        if b0 < 0:
            off += b0
            b0 = 0

        b1 = int(B1_PARAMS[spaxel_ID] + xoff + 50) + 1

        # make a grid in the bounding box range that is finer mesh by the oversample factor
        xsub = np.linspace(0, a1 - a0 - 1, (a1 - a0) * oversample_factor)
        ysub = np.linspace(0, b1 - b0 - 1, (b1 - b0) * oversample_factor)

        """print("a0", a0)
        print("a1",a1)
        print("b0",b0)
        print("b1",b1)
        print("xsub",xsub)
        print("ysub",ysub)"""
        # MAKE coordinate grid
        xv_sub, yv_sub = np.meshgrid(ysub, xsub)

        ##########################################################################
        # the spectrum for spaxel with number spaxel_ID can be found in this box.
        # image[a0:a1,b0:b1]
        ##########################################################################

        # we will model 1400 spectral elements for each spaxel in the blue cube
        x0 = np.arange(0, 1400, 1)

        p = Polynomial([QUARTIC_LINEAR_PARAMS[spaxel_ID], Z_1ST[spaxel_ID], Z_2ND[spaxel_ID], 0, Z_4TH[spaxel_ID]])
        # print(p)
        adjustmentP = Polynomial(offsets[spaxel_ID])
        # print(adjustmentP)

        # TODO: make sure the parameters are saved so they cover 0,1400 rather than 50 something
        curve = p(x0) + adjustmentP(x0) + off
        # print("curve",curve)
        width_polynomial = Polynomial(widths[spaxel_ID])

        for spec_element in range(0, 1400):
            # the per spaxel per wavelength model will be in this box:
            c0 = int(spec_element - 50) * oversample_factor
            if c0 < 0:
                c0 = 0
            c1 = int(spec_element + 50) * oversample_factor  # (c0+100*factor-(factor-1))
            if c1 >= 1400 * oversample_factor:
                c1 = int(1400 * oversample_factor)

            ##########################################################################
            # the monochromatic image for spaxel with number spaxel_ID can be found in this box.
            # image[a0:a1,b0:b1][c0:c1,int(curve[spec_element]-50):int(curve[spec_element]+50)]

            d0 = int(round((curve[spec_element] - 50))) * oversample_factor
            if d0 < 0:
                d0 = 0
            d1 = int(round((curve[spec_element] + 50))) * oversample_factor
            if d1 >= 1400 * oversample_factor:
                d1 = int(1400 * oversample_factor) - 1

            xv_sub_mono = xv_sub[c0:c1, d0:d1]
            yv_sub_mono = yv_sub[c0:c1, d0:d1]

            # make the model ###############
            popt = [0, 0]  # a way to allow for a shift.
            # redefine x and y
            y = yv_sub_mono.T[0] - spec_element
            x = xv_sub_mono[0] - curve[spec_element]

            spec_trace = QUAD_SPEC[spaxel_ID] * y**2 + LIN_SPEC[spaxel_ID] * y + popt[0]
            cross_trace = LIN_CROSS[spaxel_ID] * x + popt[1]

            xiii = (xv_sub_mono - curve[spec_element]).T - spec_trace
            yiii = (yv_sub_mono - spec_element) - cross_trace

            mask_footprint = (
                np.sqrt(
                    (yv_sub_mono - spec_element - ELL_C1[spaxel_ID] - popt[1]) ** 2 / ELL_B[spaxel_ID] ** 2
                    + (xv_sub_mono - curve[spec_element] - ELL_C0[spaxel_ID] - popt[0]) ** 2 / ELL_A[spaxel_ID] ** 2
                )
                < 1
            )

            # adjusting the width
            wavelength_dep_width = float(width_polynomial(spec_element))
            spectral = 0.8 * pseudo_voigt(
                np.abs(xiii), 0, 0.6 * wavelength_dep_width, 1.3 * wavelength_dep_width, 5.4, 0.6
            ) + pseudo_voigt(np.abs(xiii), 0, 1.2, 0.2, -n_spec, 0.1, beta=0)
            crossdis = 0.99 * pseudo_voigt(
                np.abs(yiii), 0, 0.6 * wavelength_dep_width, 1.4 * wavelength_dep_width, 5.2, 0.6
            ) + pseudo_voigt(np.abs(yiii), 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10)

            model = spectral * crossdis.T * (mask_footprint.T * 1.0)
            model = model / np.max(model)

            testModel = (
                model.reshape((model.shape[0] // oversample_factor, oversample_factor, -1, oversample_factor))
                .sum(axis=3)
                .sum(axis=1)
            )

            # image[a0:a1,b0:b1][c0:c1,d0:d1] = model.T
            sum_cross_disp_axis = np.sum(testModel, axis=1)
            line_profile.append(psf_calculation(sum_cross_disp_axis))

            c0 = c0 // oversample_factor
            c1 = c1 // oversample_factor
            d0 = d0 // oversample_factor
            d1 = d1 // oversample_factor

            # get the indicies for the sparse matrix
            mask_val = testModel.T > 1e-4

            rowind = rowindex[a0:a1, b0:b1][c0:c1, d0:d1]

            row = rowind[mask_val[: len(rowind), : len(rowind.T)]]
            colind = columnindex[a0:a1, b0:b1][c0:c1, d0:d1]
            col = colind[mask_val[: len(rowind), : len(rowind.T)]]

            data = testModel.T[: len(rowind), : len(rowind.T)][mask_val[: len(rowind), : len(rowind.T)]]
            s_image = sparse.csr_matrix((data, (col, row)), shape=(4096, 2048))

            s_image = s_image.reshape((1, int(2048 * 4096)))
            list_huge_matrix.append(s_image)

    if not partial:
        flow_run_id = 0  # REVERSE
        save_results(np.vstack(line_profile), f"line_spread_{flow_run_id}.npy")
    pass

    huge_matrix = sparse.vstack(list_huge_matrix)
    cpu_done = time.time()

    print(f"GPU time = {gpu_done - gpu_start}")
    print(f"between time = {cpu_start - gpu_done}")
    print(f"CPU time = {cpu_done - cpu_start}")

    print(compare_matrices_fast_sparse(batch_sparse_matrix, huge_matrix))

    quit()
    return huge_matrix  # type: ignore


def calculate_residuals(fitModel: sparse.csr_matrix, imagea: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Calculate the residuals for the fit array compared to the data.

    Args:
        fitModel: sparse matrix of fitted model data
        imagea: numpy array of actual image data
        heights: numpy array defining height bins
    Returns:
        numpy.ndarray: array of normalized chi-squared values per height bin including readout noise.
    """  # noqa: D205, D212
    logger = get_logger()
    norms = []
    notbadmodel = fitModel + READOUT_NOISE**2  # also avoids division by zero
    difference = imagea - fitModel
    chi2 = np.square(difference) / notbadmodel
    logger.info(f"Total chi2 value: {np.nansum(chi2)}")

    for i in range(len(heights) - 1):  # for every height bin
        chi2Sub = chi2[int(heights[i]) : int(heights[i + 1]), :]
        numpix = chi2Sub.shape[0] * chi2Sub.shape[1]
        norms.append(np.sum(chi2Sub) / (numpix + 1))
    return np.array(norms)


# TODO: make this accept a path rather than a dataImage directly
def fit(
    matrix: sparse.csr_matrix,
    data_image: Path,
    spectrum: np.ndarray,
    spaxel: int,
    partial: bool = True,
    num_height_bins: int = 256,
) -> np.ndarray:
    """Fit the matrix and get the resulting data vector.

    Args:
        matrix: sparse matrix representing the model
        data_image: Path to the preprocessed data image file
        spectrum: full spectrum, needs to be concatenated
        spaxel: spaxel number being processed
        partial: whether the matrix is partial or full
        num_height_bins: number of height bins to calculate residuals for
    Returns:
        numpy.ndarray: array of normalized chi-squared values per height bin.
    """  # noqa: D205
    assert 4096 % num_height_bins == 0, "num_height_bins must be a factor of 4096"
    data_to_fit_to = load_images_from_file(data_image)[0].data.T  # Maybe this was me REVERSE
    if partial:
        spectrum = np.concatenate(spectrum[15 * (spaxel // 15) : 15 * (spaxel // 15 + 1)])
    else:
        spectrum = np.concatenate(spectrum[:])

    logger = get_logger()
    flow_run_id = get_run_id()

    heights = np.linspace(0, 4095, num_height_bins)  # must be a factor of 4096

    start = time.time()

    matrix = matrix.transpose()  # type: ignore
    spectrum = sparse.csr_matrix(spectrum).transpose()  # type: ignore

    # calculate the product of matrix and vector, then reshape it to the CCD size of 4096 x 2048 pixels
    sparse_model_image = matrix.dot(spectrum)
    sparse_model_image = sparse_model_image.reshape((4096, 2048))
    model_image = sparse_model_image.todense()

    # load an example SNIFS file, the file should be preprocessed
    # make a mask for all pixels containing signal from the model
    flag = (model_image > 0.0) & np.isfinite(data_to_fit_to).T  # REVERSE
    flag = np.array(flag.astype(float))
    masked_image_to_fit_to = np.where(flag, data_to_fit_to.T, 0.0)  # REVERSE

    # bring the image into the right shape for fitting
    flat_image = masked_image_to_fit_to.flatten()
    fl = np.array(flat_image.transpose().flat)

    # otherwise fit will run for hours and just return garbage
    assert np.all(np.isfinite(fl)), "b contains NaN or Inf!"

    # do the final fit using scipy
    from scipy.sparse.linalg import lsqr

    lsqr_start = time.time()
    fit_vector, _, _, _ = lsqr(matrix, fl)[:4]

    lsqr_end = time.time()

    print(f"LSQR TOOK {(lsqr_start - lsqr_end):.4f} seconds")
    if not partial:
        flow_run_id = 0  # REVERSE
        save_results(fit_vector, f"fit_vector_{flow_run_id}.npy")

    stop = time.time()

    fitModel = matrix.dot(fit_vector)
    fitModel = fitModel.reshape((4096, 2048))

    # plot_fitting_check(fitModel, masked_image_to_fit_to)
    norms = calculate_residuals(fitModel, masked_image_to_fit_to, heights)

    stop = time.time()
    logger.info(stop - start)

    return np.array(norms)


def shifting_spaxel(
    spaxel: int,
    shift: float,
    isTranslationalShift: bool,
    translational_params: np.ndarray = default_shift_offsets,
    width_params: np.ndarray = default_width_offsets,
    oversample_factor: int = 1,
    is_partial: bool = True,
) -> sparse.csr_matrix:
    """Args:
        spaxel: spaxel number to shift
        shift: amount to shift (either translational or width)
        isTranslationalShift: whether to apply translational shifts or width shifts
        translational_params: array of translational shift parameters. Defaults to default_shift_offsets.
        width_params: array of width shift parameters. Defaults to default_width_offsets.
        oversample_factor: oversampling factor for the model. Defaults to 1.

    Returns:
        sparse.csr_matrix: shifted sparse matrix.
    """  # noqa: D205
    os = translational_params
    ws = width_params
    if isTranslationalShift:
        os[spaxel][0] = shift
    else:
        ws[spaxel][0] = shift
    return make_matrix(spaxel, os, widths=ws, oversample_factor=oversample_factor, partial=is_partial)


def repeat_shift_fit(
    spaxels: list[int],
    shifts: list[float],
    is_translational_shift: bool,
    processed_data_path: Path,
    spectrum: np.ndarray,
    translational_params: np.ndarray = default_shift_offsets,
    width_params: np.ndarray = default_width_offsets,
    oversample_factor: int = 1,
) -> None:
    """Args:
    spaxels: list of spaxel numbers to process, must be between 0 and 224
        shifts: list of shifts to apply (either translational or widths)
        is_translational_shift: whether to apply translational shifts or width shifts
        translational_params: array of translational shift parameters. Defaults to default_shift_offsets.
        width_params: array of width shift parameters. Defaults to default_width_offsets.
        oversample_factor: oversampling factor for the model. Defaults to 1.
    """  # noqa: D205
    assert all((s < 225 and s >= 0) for s in spaxels), "spaxel numbers must be between 0 and 224"
    if not is_translational_shift:
        assert all(shift > 0 for shift in shifts), "all width shifts must be positive"

    for spaxel in spaxels:
        errors = []
        for shift in shifts:
            shifted_matrix = shifting_spaxel(
                spaxel,
                shift,
                is_translational_shift,
                translational_params=translational_params,
                width_params=width_params,
                oversample_factor=oversample_factor,
            )
            errs = fit(shifted_matrix, processed_data_path, spectrum, spaxel)
            errors.append(errs)
        errors = np.array(errors)
        offsets = np.array(shifts)
        offsets = np.reshape(offsets, (-1, 1))
        data = np.concatenate((offsets, errors), axis=1)
        save_shift_results(data, spaxel, is_translational_shift)


if __name__ == "__main__":
    offsets = [-2.0, -1, 0, 1, 2]
    # offsets = list(np.arange(-1.7, 1.7, 0.2125, dtype=float))

    # TODO: use Sam's Preprocess Summary class to identify the type of image
    # TODO: check which file to use and make this a path argument? (idk if that works with pkls)
    # C:\Users\gibis\URAP\snifs-pipeline\src\pipeline\tasks\processing\arc_frame_spectrum.json
    with open("C:/Users/gibis/URAP/snifs-pipeline/src/pipeline/tasks/processing/arc_frame_spectrum.json", "r") as f:
        data = json.load(f)
        spec = np.array(data)

    ws = list(np.linspace(0.9, 1.3, 1))

    base_dir = Path("output/level=preprocessed")
    files = [
        base_dir
        / (
            "run_id=25_291_003/type=OBJECT/channel=B/object=2025wny/"
            "observation=25_291_003_001/flow_run_id=unknown/1_PREPROCESSED.fits"
        ),
    ]
    # files = [
    #     base_dir
    #     / (
    #         "run_id=25_291_003/type=OBJECT/channel=B/object=2025wny/"
    #         "observation=25_291_003_001/flow_run_id=unknown/1_PREPROCESSED.fits"
    #     ),
    # ]
    fits_path = "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/P25_194_024_004_03_B.fits"
    files = [fits_path]
    for file in files:
        data_path = Path(file)
        assert Path(file).exists(), f"File {file} does not exist."

        repeat_shift_fit(list(range(5, 8)), offsets, True, data_path, spec, oversample_factor=2)
