    # from pipeline.tasks.processing.assemble import solve_system_cpu, solve_system_gpu, pcgls_gpu
if __name__ == "__main__":
    # from pipeline.tasks.processing.build_forward_group import build_neighbor_matrix, build_target_matrix
    # import cupy as cp
    # import time
    # from scipy import sparse

    # from scipy.sparse.linalg import spsolve
    from astropy.io import fits
    # import cupyx.scipy.sparse as cpsparse
    import numpy as np
    # from pipeline.common.model_params import default_shift_offsets, default_width_offsets
    import matplotlib.pyplot as plt

    DETECTOR_SHAPE = DETECTOR_SHAPE = (4096, 2048)

    # neighbor_data = build_neighbor_matrix(
    #     target_spaxel=0, offsets=default_shift_offsets, widths=default_width_offsets, oversample_factor=4
    # )

    # pert_data = build_target_matrix(
    #     target_spaxel=0, widths=default_width_offsets,
    #     offsets=default_shift_offsets, o_pert=[0,0,0]
    # )
    # cp.get_default_memory_pool().free_all_blocks()
    # ndata, nrow, ncol = neighbor_data
    # pdata, prow, pcol = pert_data[0]
    # data = cp.concatenate((ndata, pdata))
    # rows = cp.concatenate((nrow, prow))
    # cols = cp.concatenate((ncol, pcol))

    # cdata = data.get()
    # crows = rows.get()
    # ccols = cols.get()

    fits_path = "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/P25_194_024_004_03_B.fits"

    try:
        with fits.open(fits_path) as hdul:
            SCIENCE_IMAGE_2D = np.array(hdul[0].data, dtype=np.float64)
            SCIENCE_IMAGE_2D = np.nan_to_num(SCIENCE_IMAGE_2D, nan=0.0, posinf=0.0, neginf=0.0)
            # Flattened Science Image
            # SCIENCE_IMAGE = SCIENCE_IMAGE_2D.ravel()
    except Exception:
        # Fallback to zeros for testing if missing
        SCIENCE_IMAGE_2D = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
        # SCIENCE_IMAGE = SCIENCE_IMAGE_2D.ravel()


    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    # Create sample 2D data with a wide range of values
    data = SCIENCE_IMAGE_2D * (SCIENCE_IMAGE_2D > 1e-4)

    # Plot the 2D array using imshow with log norm and a fire-like colormap
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='inferno', norm=LogNorm())

    # Add a colorbar to show the z-scale mapping
    plt.colorbar(im, ax=ax, label='Z-value (Log Scale)')
    plt.title('2D Array with Fire Cmap and Log Z-Scale')

    plt.show()


    # G_SCIENCE = cp.asarray(SCIENCE_IMAGE)
    # cpu_times = []
    # gpu_times = []
    # # Selects 15 spaxels (15 * 1400 = 21000 length array)
    # spec: np.ndarray = np.ones((225, 1400))
    # spax = 0
    # spectra = np.concatenate(spec[15 * (spax // 15) : 15 * (spax // 15 + 1)])

    # gspec = cp.array(spectra)
    # for i in range(10):
    #     _, _, elapsed_cpu = solve_system_cpu(cdata, crows, ccols, (4096*2048,21000),spectra,SCIENCE_IMAGE)
    #     cpu_times.append(elapsed_cpu)
    #     print(f"{i} CPU DONE! {elapsed_cpu}")
    #     _, _, elapsed_gpu = solve_system_gpu(data, rows, cols, (4096*2048,21000),gspec, G_SCIENCE)
    #     gpu_times.append(elapsed_gpu)
    #     print(f"{i} GPU DONE! {elapsed_gpu}")
    #     cp.get_default_memory_pool().free_all_blocks()
    # print(cpu_times)
    # print(gpu_times)

    # def compare_solves(data, rows, cols, shape, group_spectra, science_flat):
    #     # 1. Run CPU Solve
    #     loss_cpu, fit_cpu, time_cpu = solve_system_cpu(
    #         data, rows, cols, shape, group_spectra, science_flat
    #     )
        
    #     # Extract x_cpu directly to compare solution vectors
    #     A_cpu = sparse.csr_matrix((data, (cols, rows)), shape=shape)
    #     b_cpu = np.asarray(science_flat, dtype=np.float64).ravel()
    #     if group_spectra is not None and group_spectra.shape[0] == shape[1]:
    #         shifted = A_cpu.dot(group_spectra)
    #         flag = (shifted > 0.0) & np.isfinite(b_cpu)
    #         b_cpu = np.where(flag, b_cpu, 0.0)
    #     AtA = A_cpu.T.dot(A_cpu).tocsc() + 1e-10 * sparse.eye(shape[1], format="csc")
    #     Atb = A_cpu.T.dot(b_cpu)
    #     x_cpu = spsolve(AtA, Atb)

    #     # 2. Run GPU Solve (with explicit stream sync for accurate timing)
    #     cp.cuda.Stream.null.synchronize()
    #     data = cp.array(data)
    #     rows = cp.array(rows)
    #     cols = cp.array(cols)
    #     group_spectra = cp.array(group_spectra)
    #     science_flat = cp.array(science_flat)

    #     loss_gpu, fit_gpu, time_gpu = solve_system_gpu(
    #         data, rows, cols, shape, group_spectra, science_flat
    #     )
    #     cp.cuda.Stream.null.synchronize()

    #     # Re-extract x_gpu array from PCGLS for direct comparison
    #     A_gpu = cpsparse.csr_matrix((data, (cols, rows)), shape=shape, dtype=cp.float64)
    #     b_gpu = cp.asarray(b_cpu, dtype=cp.float64)
    #     x_gpu_vec = pcgls_gpu(A_gpu, b_gpu, tol=1e-5, maxiter=50, damp=1e-5)
    #     x_gpu = cp.asnumpy(x_gpu_vec)

    #     # 3. Compute Metrics
    #     speedup = time_cpu / time_gpu if time_gpu > 0 else 0.0
        
    #     # Relative difference in reconstructed spectrum (x)
    #     rel_err_x = np.linalg.norm(x_cpu - x_gpu) / np.linalg.norm(x_cpu)
    #     max_err_x = np.max(np.abs(x_cpu - x_gpu))
        
    #     # Relative difference in detector model image
    #     rel_err_fit = np.linalg.norm(fit_cpu - fit_gpu) / np.linalg.norm(fit_cpu)

    #     print("=== SOLVER COMPARISON REPORT ===")
    #     print(f"Runtime CPU:      {time_cpu:.4f} s")
    #     print(f"Runtime GPU:      {time_gpu:.4f} s")
    #     print(f"GPU Speedup:      {speedup:.2f}x")
    #     print(f"Loss CPU:         {loss_cpu:.6e}")
    #     print(f"Loss GPU:         {loss_gpu:.6e}")
    #     print(f"Relative Diff x:  {rel_err_x:.6e}")
    #     print(f"Max Abs Diff x:   {max_err_x:.6e}")
    #     print(f"Relative Fit Err: {rel_err_fit:.6e}")

    #     return {
    #         "speedup": speedup,
    #         "rel_err_x": rel_err_x,
    #         "x_cpu": x_cpu,
    #         "x_gpu": x_gpu
    #     }

    # print(compare_solves(cdata,crows,ccols,(4096*2048,21000),spectra,SCIENCE_IMAGE))