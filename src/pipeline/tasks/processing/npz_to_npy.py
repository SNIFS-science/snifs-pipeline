

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import sparse
from scipy.sparse.linalg import spsolve
from multiprocessing import Pool, shared_memory
import gc
import cupy as cp
import cupyx.scipy.sparse as cpsp
# from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve
# from scipy import sparse
import numpy as np
import time






# from pipeline.tasks.processing.CG_GPU_solver import cg_solve_chunked
from pipeline.tasks.processing.build_model import build_neighbor_matrix, build_target_matrix

#ARCS VS CONTINUUM
# fits_file_name = "24_028_001_001_03_B"

#CHANGE HERE TODO
fits_file_name = "24_028_001_001_03_B" #3 things to change
IS_CONTINUUM = False


#UR DONE! TODO
savename = f"{fits_file_name}_reconstructed"
if IS_CONTINUUM:
    savename += "_cont.npy"
else:
    savename += "_arc.npy"

downloads_dir = r"C:/Users/gibis/URAP/downloads from nersc"
folder_dir = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/spaxel_fits"
fits_path = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/{fits_file_name}.fits"



savepath = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/{savename}"

NUM_SPAXELS = 225
print("WARNING PLEASE MAKE SURE THE FOLDER DIRECTOR FITS PATH DIRECTORY AND SAVE PATH ARE CORRECT")
print("="*20)
print("\nCHECK WHEHTER YOU ARE DOING CONTINUUM OR ARC\n")
print("="*20)
#CONTINUUM
def get_spec_list():
#     s = [np.ones(SPEC_LEN, dtype=np.float64) for _ in range(NUM_SPAXELS)]
#     return s
#ARC
    if IS_CONTINUUM:
        s = [np.ones(SPEC_LEN, dtype=np.float64) for _ in range(NUM_SPAXELS)]
        print("using continuum initialization")
    else:
        import pickle
        print("using arc initialization")
        with open("./spectra_arc.pkl", 'rb') as file:
            loaded_data = pickle.load(file)
        s = loaded_data
    return s

SPAXELS_PER_GROUP = 15
DETECTOR_SHAPE = (4096, 2048)
N_DET_PIX = DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]
SPEC_LEN = 1400
MATRIX_ROWS = N_DET_PIX
MATRIX_COLS = SPEC_LEN * SPAXELS_PER_GROUP
MATRIX_SHAPE = (MATRIX_ROWS, MATRIX_COLS)


def _to_ascending(c):
    return np.ascontiguousarray(np.asarray(c, dtype=np.float64)[..., ::-1])


def get_group_spectra(spax_id: int) -> np.ndarray:
    spec = get_spec_list()
    lo = SPAXELS_PER_GROUP * (spax_id // SPAXELS_PER_GROUP)
    hi = min(lo + SPAXELS_PER_GROUP, len(spec))
    out = np.zeros(MATRIX_COLS, dtype=np.float64)
    if hi > lo:
        cat = np.concatenate([spec[i] for i in range(lo, hi)])
        n = min(cat.size, MATRIX_COLS)
        out[:n] = cat[:n]
        del cat
    return out

def get_science_image() -> np.ndarray:
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape != DETECTOR_SHAPE:
            arr = arr.T
            if arr.shape != DETECTOR_SHAPE:
                raise ValueError(f"FITS shape {arr.shape} != {DETECTOR_SHAPE}")
    except Exception:
        arr = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
        raise ValueError("FITS shape error")

    img = arr.ravel()
    return img


def process_item(tup):
    # Unpack the shared memory descriptors instead of the heavy arrays
    group_id, shm_names, shapes, dtypes = tup
    
    spax_id = group_id * 15
    base_col = group_id * SPAXELS_PER_GROUP * SPEC_LEN

    # Attach to the existing shared memory blocks using their names
    shm_data = shared_memory.SharedMemory(name=shm_names[0])
    shm_cols = shared_memory.SharedMemory(name=shm_names[1])
    shm_rows = shared_memory.SharedMemory(name=shm_names[2])

    # Reconstruct numpy arrays backed by shared memory
    data = np.ndarray(shapes[0], dtype=dtypes[0], buffer=shm_data.buf)
    cols = np.ndarray(shapes[1], dtype=dtypes[1], buffer=shm_cols.buf)
    rows = np.ndarray(shapes[2], dtype=dtypes[2], buffer=shm_rows.buf)


    cols -= base_col

    A = sparse.csr_matrix(
        (data, (rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False))),
        shape=MATRIX_SHAPE,
    )
    del data, rows, cols

    science = get_science_image()
    group_spec = get_group_spectra(spax_id)

    if group_spec.shape[0] == A.shape[1]:
        shifted = A.dot(group_spec)
        b = np.where((shifted > 0.0) & np.isfinite(science), science, 0.0)
        del shifted
    else:
        b = np.array(science, dtype=np.float64)
        raise RuntimeError(
            f"Shape mismatch: group_spec length ({group_spec.shape[0]}) does not match matrix columns ({A.shape[1]})"
        )

    At = A.T.tocsr()
    AtA = (At.dot(A)).tocsc() + 1e-10 * sparse.eye(MATRIX_SHAPE[1], format="csc")
    Atb = At.dot(b)
    x = spsolve(AtA, Atb)
    del AtA, Atb, At


    b = A.dot(x).reshape(DETECTOR_SHAPE)
    return b


if __name__ == '__main__':
    offsets = []
    widths = []
    for i in range(0, 225):
        with np.load(f"{folder_dir}/spaxel_{i:03d}.npz") as info:
            offsets.append(info["off_poly"][-1])
            widths.append(info["wid_poly"][-1])

    offsets = _to_ascending(np.array(offsets))
    widths = _to_ascending(np.array(widths))

    jobs = []
    allocated_shms = []  # Keep track so we can clean up RAM later

    for k in range(15):
        print(f"STARTING GROUP {k}")
        off = offsets[k*15:(k+1)*15]
        wid = widths[k*15:(k+1)*15]
        spax_id = k * 15
        
        try:
            neigh = build_neighbor_matrix(target_spaxel=spax_id, offsets=off, widths=wid, oversample_factor=4)
        except Exception:
            raise RuntimeError("comment out tasks or start a prefect server for build_forward_group.py")

        targ = build_target_matrix(
            target_spaxel=spax_id, offsets=off, widths=wid, o_pert=[0.0], oversample_factor=4
        )

        data = np.concatenate((neigh[0], targ[0][0]))
        cols = np.concatenate((neigh[1], targ[0][1]))
        rows = np.concatenate((neigh[2], targ[0][2]))

        # Create Shared Memory blocks sized specifically for these arrays
        shm_data = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_cols = shared_memory.SharedMemory(create=True, size=cols.nbytes)
        shm_rows = shared_memory.SharedMemory(create=True, size=rows.nbytes)

        # Map numpy arrays to the shared memory buffer and copy the data over
        np.ndarray(data.shape, dtype=data.dtype, buffer=shm_data.buf)[:] = data[:]
        np.ndarray(cols.shape, dtype=cols.dtype, buffer=shm_cols.buf)[:] = cols[:]
        np.ndarray(rows.shape, dtype=rows.dtype, buffer=shm_rows.buf)[:] = rows[:]

        # Append the job configuration containing strictly lightweight data
        shm_names = (shm_data.name, shm_cols.name, shm_rows.name)
        shapes = (data.shape, cols.shape, rows.shape)
        dtypes = (data.dtype, cols.dtype, rows.dtype)
        
        jobs.append((k, shm_names, shapes, dtypes))
        
        # Save reference for cleanup
        allocated_shms.extend([shm_data, shm_cols, shm_rows])


    # Run the Pool
    with Pool(processes=6) as pool:
        results = pool.map(process_item, jobs)

    # ---------------------------------------------------------
    # CRITICAL: Clean up shared memory to prevent RAM leaks
    # ---------------------------------------------------------
    for shm in allocated_shms:
        shm.close()
        shm.unlink()

    total = None
    for i in results:
        if total is None:
            total = i
        else:
            total += i

    plt.imshow(total)
    plt.show()

    np.save(savepath, total)