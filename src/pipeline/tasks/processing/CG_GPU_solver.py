
import cupy as cp
import cupyx.scipy.sparse as sps

# 1. KERNELS
cg_update_xr = cp.ElementwiseKernel(
    'T alpha, T p, T Mp', 'T x, T r',
    '''
    x = x + alpha * p;
    r = r - alpha * Mp;
    ''',
    'cg_update_xr'
)

cg_update_p = cp.ElementwiseKernel(
    'T beta, T r', 'T p',
    'p = r + beta * p;',
    'cg_update_p'
)

# Fused reduction kernel: computes the squared norm of the UNSCALED residual
calc_unscaled_rs = cp.ReductionKernel(
    'T r, T scale',
    'T out',
    '(r / scale) * (r / scale)',
    'a + b',
    'out = a',
    '0',
    'calc_unscaled_rs'
)

# 2. SOLVER FUNCTION
def cg_solve_chunked(M, c, scale_vec=None, max_iter=1000, atol=1e-6, check_interval=20):
    tiny = cp.finfo(c.dtype).tiny 
    
    # --- Matrix Format Check & Conversion ---
    if not sps.isspmatrix_csr(M):
        # print("Converting matrix to CSR format...")
        M = M.tocsr()
        
    # --- Auto-Assemble L2 Jacobi Preconditioner ---
    if scale_vec is None:
        # print("Assembling L2 Jacobi scaling vector...")
        # Efficiently compute L2 norm of each row: sqrt(sum(M_ij^2))
        M_sq = sps.csr_matrix((M.data ** 2, M.indices, M.indptr), shape=M.shape)
        
        # .ravel() ensures we get a flat 1D array instead of a matrix object
        row_l2_norms = cp.sqrt(M_sq.sum(axis=1).ravel())
        
        # S_i = 1 / sqrt(N_i). We use maximum to prevent div-by-zero on empty rows.
        scale_vec = 1.0 / cp.sqrt(cp.maximum(row_l2_norms, tiny))

    # Apply scaling to RHS
    c_scaled = c * scale_vec

    x = cp.zeros_like(c_scaled)
    r = c_scaled.copy()
    p = c_scaled.copy()
    
    # Calculate initial unscaled residual squared to anchor `atol` in physical space
    rs0_unscaled = float(cp.dot(c, c)) 
    stop_threshold = (atol ** 2) * rs0_unscaled
    
    # print(f"Target Unscaled Squared Residual: < {stop_threshold:.4e}")
    
    # rsold tracks the SCALED residual for alpha/beta momentum
    rsold = cp.dot(r, r) 
    
    max_blocks = max_iter // check_interval
    total_iters = 0
    
    for block in range(max_blocks):
        for _ in range(check_interval):
            # Apply symmetric scaling: M_scaled @ p = S * (M @ (S * p))
            Mp = scale_vec * M.dot(scale_vec * p)
            
            p_dot_Mp = cp.dot(p, Mp)
            alpha = rsold / cp.maximum(p_dot_Mp, tiny)
            
            cg_update_xr(alpha, p, Mp, x, r)
            
            rsnew = cp.dot(r, r)
            
            beta = rsnew / cp.maximum(rsold, tiny)
            
            cg_update_p(beta, r, p)
            rsold = rsnew
            
        total_iters += check_interval
        
        # Calculate true unscaled residual for strict mathematical tolerance checking
        current_unscaled_rsnew = float(calc_unscaled_rs(r, scale_vec))
            
        # print(f"Batch {block+1:02d} | Iters: {total_iters:04d} | Unscaled Sq Residual: {current_unscaled_rsnew:.4e}")
        
        if current_unscaled_rsnew < stop_threshold:
            # Re-map y back to original unconditioned coordinates: x_true = S * y
            return x * scale_vec

    print(f"\n[WARNING] Reached {max_iter} max iterations without full convergence.")
    return x * scale_vec