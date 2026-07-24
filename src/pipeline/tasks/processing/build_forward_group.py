import cupy as cp
import numpy as np
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
    default_width_offsets
)
xoff = 0
yoff = 0
n_cross = np.float64(-0.9020140646117615)
n_spec = np.float64(-1.0115923616013527)

pv_preamble = '''
__device__ float pv_core(float x, float xo, float wg, float wl, float n, float eta, float beta, float l_off, float amp) {
    float val = 0.0f;
    if (beta != 0.0f) {
        float tmp = (x - xo) / wg;
        val += beta * expf(-0.69314718056f * tmp * tmp);
    }
    if (eta != 0.0f) {
        float base = fabsf(x - xo);
        if (n < 0.0f) {
            base = max(base, 1e-8f);
        }
        val += eta / (powf(base / wl, n) + l_off);
    }
    return amp * val;
}
'''


fused_profile_kernel = cp.ElementwiseKernel(
    in_params='raw float32 curve, raw float32 width, raw float32 quad_spec, raw float32 lin_spec, raw float32 lin_cross, float32 n_spec, float32 n_cross, int32 dim, float32 step_size',
    out_params='float32 spectral',
    operation='''
    long long dim2 = (long long)dim * dim; // Cast to long long to prevent int32 overflow on large oversamples
    int y_idx = i % dim;
    int x_idx = (i / dim) % dim;
    int w     = (i / dim2) % 1400;
    int s     = i / (dim2 * 1400);

    float x_val = -50.0f + ((float)x_idx * step_size);
    float y_val = -50.0f + ((float)y_idx * step_size);

    int sw_idx = s * 1400 + w;
    float c_val = curve[sw_idx];
    float w_val = width[sw_idx];

    // Compute fraction correction & traces inline
    float x_frac = x_val - (c_val - roundf(c_val));
    float spec_tr = quad_spec[s] * y_val * y_val + lin_spec[s] * y_val;
    float cross_tr = lin_cross[s] * x_frac;

    float local_xiii = fabsf(x_frac - spec_tr);
    float local_yiii = fabsf(y_val - cross_tr);

    // Evaluate Dispersion components
    float val_spec = pv_core(local_xiii, 0.0f, 0.6f * w_val, 1.3f * w_val, 5.4f, 0.6f, 0.4f, 1.0f, 0.8f)
                   + pv_core(local_xiii, 0.0f, 1.2f, 0.2f, -n_spec, 0.1f, 0.0f, 1.0f, 1.0f);

    // Evaluate Cross-Dispersion components
    float val_cross = pv_core(local_yiii, 0.0f, 0.6f * w_val, 1.4f * w_val, 5.2f, 0.6f, 0.4f, 1.0f, 0.99f)
                    + pv_core(local_yiii, 0.0f, 1.2f, 0.1f, -n_cross, 0.1f, 0.0f, 10.0f, 1.0f);

    spectral = val_spec * val_cross;
    ''',
    name='fused_profile_kernel',
    preamble=pv_preamble
)

#normalization and the elliptical footprint
normalize_and_mask_kernel = cp.ElementwiseKernel(
    in_params='float32 in_spectral, raw float32 maxes, raw float32 curve, raw float32 ell_c0, raw float32 ell_c1, raw float32 ell_a, raw float32 ell_b, int32 dim, float32 step_size',
    out_params='float32 out_spectral',
    operation='''
    long long dim2 = (long long)dim * dim;
    int y_idx = i % dim;
    int x_idx = (i / dim) % dim;
    int w     = (i / dim2) % 1400;
    int s     = i / (dim2 * 1400);

    int sw_idx = s * 1400 + w;
    float m_val = maxes[sw_idx];
    
    float spec_val = 0.0f;
    if (m_val > 0.0f) {
        spec_val = in_spectral / m_val;
    }

    float x_val = -50.0f + ((float)x_idx * step_size);
    float y_val = -50.0f + ((float)y_idx * step_size);
    float c_val = curve[sw_idx];
    float x_frac = x_val - (c_val - roundf(c_val));

    // Elliptical footprint math
    float dx = (x_frac - ell_c0[s]) / ell_a[s];
    float dy = (y_val - ell_c1[s]) / ell_b[s];

    if ((dx * dx + dy * dy) >= 1.0f) {
        out_spectral = 0.0f;
    } else {
        out_spectral = spec_val;
    }
    ''',
    name='normalize_and_mask_kernel'
)
# -------------------------------------------------------------
# 1. BUILD NEIGHBORS (14 Spaxels)
# -------------------------------------------------------------
def build_neighbor_matrix(
    target_spaxel: int,
    offsets: np.ndarray,
    widths: np.ndarray,
    oversample_factor: int = 1,
    group_size: int = 15
) -> tuple:
    """Computes the 14 static neighbor spaxels within the group.
    Offsets should be in a 14 by 5 numpy array and widths
    technically groupsize - 1 by 5 numpy array"""
    group = target_spaxel // group_size
    start_ind = group * group_size
    
    # Identify the 14 neighbor indices (both local 0-14, and global)
    local_indices = np.array([i for i in range(group_size) if i != target_spaxel])
    global_indices = start_ind + local_indices
    n_active = len(local_indices) # 14
    
    # 1. & 2. ARRAYS & POLYNOMIALS (Subset to 14)
    a0 = cp.array(A0_PARAMS[global_indices] + yoff, dtype=cp.int16) 
    b0 = cp.array(B0_PARAMS[global_indices] + xoff - 50, dtype=cp.int16) 

    off = cp.full(n_active, 50, dtype=cp.int16)
    mask = b0 < 0
    off[mask] += b0[mask]
    b0[mask] = 0
    del mask

    P0 = cp.array(QUARTIC_LINEAR_PARAMS[global_indices], dtype=cp.float32)[:, None]
    P1 = cp.array(Z_1ST[global_indices], dtype=cp.float32)[:, None]
    P2 = cp.array(Z_2ND[global_indices], dtype=cp.float32)[:, None]
    P4 = cp.array(Z_4TH[global_indices], dtype=cp.float32)[:, None]

    O_arr = cp.array(offsets[global_indices].T, dtype=cp.float32)
    O1, O2, O3, O4, O5 = [O_arr[i].T[:, None] for i in range(5)]
    
    W_arr = cp.array(widths[global_indices].T, dtype=cp.float32)
    W1, W2, W3, W4, W5 = [W_arr[i].T[:, None] for i in range(5)]
    
    x0_f64 = cp.arange(0, 1400, 1, dtype=cp.float64)
    grid_view = cp.broadcast_to(x0_f64[None, :], (n_active, 1400))
    gpu_off = cp.full(n_active, off, dtype=cp.float64)[:, None]

    curve = (
        (P0 + O1) +
        ((P1 + O2) * grid_view) +
        ((P2 + O3)* (grid_view**2)) +
        (O4 * (grid_view**3)) +
        ((P4 + O5)* (grid_view**4)) + gpu_off
    ).astype(cp.float32)

    width_polynomial = (W1 + (W2 * grid_view) + (W3 * (grid_view**2)) + (W4 * (grid_view**3)) + (W5 * (grid_view**4))).astype(cp.float32)
    
    del P0, P1, P2, P4, O_arr, W_arr, grid_view, gpu_off, x0_f64

    # 3, 4, 5. FUSED PROFILE GENERATION & MASKING
    dim = int(100 * oversample_factor)
    step_size = 1.0 / oversample_factor

    QUAD_SPEC_1D = cp.array(QUAD_SPEC[global_indices], dtype=cp.float32)
    LIN_SPEC_1D = cp.array(LIN_SPEC[global_indices], dtype=cp.float32)
    LIN_CROSS_1D = cp.array(LIN_CROSS[global_indices], dtype=cp.float32)

    spectral = cp.empty((n_active, 1400, dim, dim), dtype=cp.float32)

    fused_profile_kernel(
        curve, width_polynomial, QUAD_SPEC_1D, LIN_SPEC_1D, LIN_CROSS_1D,
        float(n_spec), float(n_cross), dim, step_size, spectral
    )
    del QUAD_SPEC_1D, LIN_SPEC_1D, LIN_CROSS_1D, width_polynomial

    maxes = cp.max(spectral, axis=(2, 3))

    ELL_C0_1D = cp.array(ELL_C0[global_indices], dtype=cp.float32)
    ELL_C1_1D = cp.array(ELL_C1[global_indices], dtype=cp.float32)
    ELL_A_1D  = cp.array(ELL_A[global_indices], dtype=cp.float32)
    ELL_B_1D  = cp.array(ELL_B[global_indices], dtype=cp.float32)

    normalize_and_mask_kernel(
        spectral, maxes, curve, 
        ELL_C0_1D, ELL_C1_1D, ELL_A_1D, ELL_B_1D, 
        dim, step_size, spectral
    )
    del maxes, ELL_C0_1D, ELL_C1_1D, ELL_A_1D, ELL_B_1D

    # 6. DOWNSAMPLING
    W_sub, H_sub = spectral.shape[2], spectral.shape[3]
    W_real = (W_sub // oversample_factor)
    H_real = (H_sub // oversample_factor)

    testModel_4d = spectral.reshape(
        n_active, 1400, W_real, oversample_factor, H_real, oversample_factor
    ).sum(axis=(3, 5))
    del spectral

    testModel_4d = cp.transpose(testModel_4d, (0, 1, 3, 2))

    # 7. BOUNDING BOX & INDEX MAPPING
    x0_int = cp.arange(0, 1400, 1, dtype=cp.int16)
    
    c0 = cp.clip((x0_int - 50), 0, None)
    c1 = cp.clip((x0_int + 50), 0, 1400)
    c0 = cp.broadcast_to(c0[None, :], (n_active, 1400))
    c1 = cp.broadcast_to(c1[None, :], (n_active, 1400))

    d0 = cp.rint(cp.maximum((curve - 50), 0)).astype(cp.int16)
    d1 = cp.rint(cp.clip((curve + 50), 0, 1399)).astype(cp.int16)

    local_grid = cp.arange(100, dtype=cp.int16)
    abs_c_4d = (x0_int - 50)[None, :, None, None] + local_grid[None, None, :, None]
    abs_d_4d = cp.rint(curve - 50).astype(cp.int32)[:, :, None, None] + local_grid[None, None, None, :]

    in_bounds_mask = (
        (abs_c_4d >= c0[:, :, None, None]) & 
        (abs_c_4d <  c1[:, :, None, None]) &
        (abs_d_4d >= d0[:, :, None, None]) & 
        (abs_d_4d <  d1[:, :, None, None])
    )
    
    mask_val = (testModel_4d > 1e-4) & in_bounds_mask 
    del in_bounds_mask

    sparse_data = testModel_4d[mask_val]
    del testModel_4d

    spaxel_idx, spec_idx, local_row_idx, local_col_idx = cp.nonzero(mask_val)
    
    # CRITICAL: Map the internal 0-13 index back to the true 0-14 group index
    # so the rows sit exactly where they should in the final combined matrix.
    cp_local_indices = cp.array(local_indices)
    actual_local_spaxel = cp_local_indices[spaxel_idx]
    
    sparse_design_rows = (actual_local_spaxel * 1400) + spec_idx

    abs_c = a0[spaxel_idx] + spec_idx - 50 + local_row_idx 
    curve_vals = cp.round(curve[spaxel_idx, spec_idx]).astype(cp.int32)
    abs_d = b0[spaxel_idx] + curve_vals - 50 + local_col_idx 
    flat_ccd_indices = (abs_c * 2048) + abs_d

    return sparse_data, sparse_design_rows, flat_ccd_indices
    # Return as CSR to optimize memory and prepare for instant addition
    # return coo_matrix(
    #     (sparse_data, (sparse_design_rows, flat_ccd_indices)),
    #     shape=(group_size * 1400, 4096 * 2048)
    # ).tocsr()

def build_target_matrix(
    target_spaxel: int,
    offsets: np.ndarray,
    widths: np.ndarray,
    o_pert: np.ndarray = None,
    w_pert: np.ndarray = None,
    oversample_factor: int = 1,
    group_size: int = 15,
) -> tuple:
    """Computes the 1 target spaxel with highly perturbed parameters."""
    group = target_spaxel // group_size
    start_ind = group * group_size

    is_offset = o_pert is not None

    n_active = len(o_pert) if is_offset else len(w_pert)
    
    # 1. & 2. ARRAYS & POLYNOMIALS (Subset to 1)
    a0 = cp.full(n_active, A0_PARAMS[target_spaxel] + yoff, dtype=cp.int16)
    b0 = cp.full(n_active, B0_PARAMS[target_spaxel] + xoff - 50, dtype=cp.int16) 

    off = cp.full(n_active, 50, dtype=cp.int16)
    mask = b0 < 0
    off[mask] += b0[mask]
    b0[mask] = 0
    del mask

    P0 = cp.full(n_active, QUARTIC_LINEAR_PARAMS[target_spaxel], dtype=cp.float32)[:, None]
    P1 = cp.full(n_active, Z_1ST[target_spaxel], dtype=cp.float32)[:, None]
    P2 = cp.full(n_active, Z_2ND[target_spaxel], dtype=cp.float32)[:, None]
    P4 = cp.full(n_active, Z_4TH[target_spaxel], dtype=cp.float32)[:, None]

    O_arr = cp.broadcast_to(cp.array(offsets[target_spaxel][:,None], dtype=cp.float32), (5,n_active))
    O1, O2, O3, O4, O5 = [O_arr[i].T[:, None] for i in range(5)]
    
    W_arr = cp.broadcast_to(cp.array(widths[target_spaxel][:,None], dtype=cp.float32), (5,n_active))
    W1, W2, W3, W4, W5 = [W_arr[i].T[:, None] for i in range(5)]
    
    x0_f64 = cp.arange(0, 1400, 1, dtype=cp.float64)
    grid_view = cp.broadcast_to(x0_f64[None, :], (n_active, 1400))
    gpu_off = cp.full(n_active, off, dtype=cp.float64)[:, None]

    curve = (
        (P0 + O1) +
        ((P1 + O2) * grid_view) +
        ((P2 + O3)* (grid_view**2)) +
        (O4 * (grid_view**3)) +
        ((P4 + O5)* (grid_view**4)) + gpu_off
    ).astype(cp.float32)
    print(curve.shape)
    width_polynomial = (W1 + (W2 * grid_view) + (W3 * (grid_view**2)) + (W4 * (grid_view**3)) + (W5 * (grid_view**4))).astype(cp.float32)

    if is_offset:
        curve += cp.array(o_pert,dtype=cp.float32)[:,None]
    else:
        width_polynomial += cp.array(w_pert,dtype=cp.float32)[:,None]

    del P0, P1, P2, P4, O_arr, W_arr, grid_view, gpu_off, x0_f64, o_pert, w_pert

    # 3, 4, 5. FUSED PROFILE GENERATION & MASKING
    dim = int(100 * oversample_factor)
    step_size = 1.0 / oversample_factor

    QUAD_SPEC_1D = cp.array(QUAD_SPEC[target_spaxel], dtype=cp.float32)
    LIN_SPEC_1D = cp.array(LIN_SPEC[target_spaxel], dtype=cp.float32)
    LIN_CROSS_1D = cp.array(LIN_CROSS[target_spaxel], dtype=cp.float32)

    spectral = cp.empty((n_active, 1400, dim, dim), dtype=cp.float32)

    fused_profile_kernel(
        curve, width_polynomial, QUAD_SPEC_1D, LIN_SPEC_1D, LIN_CROSS_1D,
        float(n_spec), float(n_cross), dim, step_size, spectral
    )
    del QUAD_SPEC_1D, LIN_SPEC_1D, LIN_CROSS_1D, width_polynomial

    maxes = cp.max(spectral, axis=(2, 3))

    ELL_C0_1D = cp.array(ELL_C0[target_spaxel], dtype=cp.float32)
    ELL_C1_1D = cp.array(ELL_C1[target_spaxel], dtype=cp.float32)
    ELL_A_1D  = cp.array(ELL_A[target_spaxel], dtype=cp.float32)
    ELL_B_1D  = cp.array(ELL_B[target_spaxel], dtype=cp.float32)

    normalize_and_mask_kernel(
        spectral, maxes, curve, 
        ELL_C0_1D, ELL_C1_1D, ELL_A_1D, ELL_B_1D, 
        dim, step_size, spectral
    )
    del maxes, ELL_C0_1D, ELL_C1_1D, ELL_A_1D, ELL_B_1D

    # 6. DOWNSAMPLING
    W_sub, H_sub = spectral.shape[2], spectral.shape[3]
    W_real = (W_sub // oversample_factor)
    H_real = (H_sub // oversample_factor)

    testModel_4d = spectral.reshape(
        n_active, 1400, W_real, oversample_factor, H_real, oversample_factor
    ).sum(axis=(3, 5))
    del spectral

    testModel_4d = cp.transpose(testModel_4d, (0, 1, 3, 2))

    # 7. BOUNDING BOX & INDEX MAPPING
    x0_int = cp.arange(0, 1400, 1, dtype=cp.int16)
    
    c0 = cp.clip((x0_int - 50), 0, None)
    c1 = cp.clip((x0_int + 50), 0, 1400)
    c0 = cp.broadcast_to(c0[None, :], (n_active, 1400))
    c1 = cp.broadcast_to(c1[None, :], (n_active, 1400))

    d0 = cp.rint(cp.maximum((curve - 50), 0)).astype(cp.int16)
    d1 = cp.rint(cp.clip((curve + 50), 0, 1399)).astype(cp.int16)

    local_grid = cp.arange(100, dtype=cp.int16)
    abs_c_4d = (x0_int - 50)[None, :, None, None] + local_grid[None, None, :, None]
    abs_d_4d = cp.rint(curve - 50).astype(cp.int32)[:, :, None, None] + local_grid[None, None, None, :]

    in_bounds_mask = (
        (abs_c_4d >= c0[:, :, None, None]) & 
        (abs_c_4d <  c1[:, :, None, None]) &
        (abs_d_4d >= d0[:, :, None, None]) & 
        (abs_d_4d <  d1[:, :, None, None])
    )
    
    mask_val = (testModel_4d > 1e-4) & in_bounds_mask 
    del in_bounds_mask

    pert_out = []
    sta = 0
    for i in range(n_active):
        sparse_data = testModel_4d[i, mask_val[i] ]

        spaxel_idx, spec_idx, local_row_idx, local_col_idx = cp.nonzero(mask_val)
        spec_idx, local_row_idx, local_col_idx = cp.nonzero(mask_val[i])

        spaxel_idx = spaxel_idx[sta:sta + len(local_col_idx)]
        sta += len(local_col_idx)

        # CRITICAL: Even though spaxel_idx is entirely 0, map it to the actual target slot
        actual_local_spaxel = cp.full_like(spaxel_idx, target_spaxel)
        sparse_design_rows = (actual_local_spaxel * 1400) + spec_idx

        abs_c = a0[spaxel_idx] + spec_idx - 50 + local_row_idx 
        curve_vals = cp.round(curve[spaxel_idx, spec_idx]).astype(cp.int32)
        abs_d = b0[spaxel_idx] + curve_vals - 50 + local_col_idx 
        flat_ccd_indices = (abs_c * 2048) + abs_d

        pert_out.append((sparse_data, sparse_design_rows, flat_ccd_indices))
    return pert_out