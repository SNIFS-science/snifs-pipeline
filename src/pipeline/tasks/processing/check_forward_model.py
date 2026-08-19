import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy import sparse
from pipeline.tasks.processing.build_forward_group import build_neighbor_matrix, build_target_matrix
#135/149

from astropy.io import fits
folder_dir = "./spaxel_fits/shrinking_span"
def _to_ascending(c):
    """np.polyfit order (x^4 first) -> builder order (constant first).

    build_neighbor_matrix / build_target_matrix unpack O1..O5 as the
    coefficients of x^0..x^4, matching Polynomial([...]) in the reference.
    Everything on the Python side is np.polyfit order, so reverse here and
    only here. Works for both (5,) and (15, 5).
    """
    return np.ascontiguousarray(np.asarray(c, dtype=np.float64)[..., ::-1])

NUM_SPAXELS = 225
def get_spec_list():
    s = [
        np.ones(SPEC_LEN, dtype=np.float64)
        for _ in range(NUM_SPAXELS)
    ]
    return s
fits_path = os.environ.get(
    "SNIFS_TEST_FITS",
    "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/deep_skyflat_coadd.fits",
)
SPAXELS_PER_GROUP = 15
DETECTOR_SHAPE = (4096, 2048)
N_DET_PIX = DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]
SPEC_LEN = 1400
MATRIX_ROWS = N_DET_PIX
MATRIX_COLS = SPEC_LEN * SPAXELS_PER_GROUP
MATRIX_SHAPE = (MATRIX_ROWS, MATRIX_COLS)
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
            raise ValueError(f"FITS shape {arr.shape} != {DETECTOR_SHAPE}")
    except Exception:
        arr = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
        raise ValueError(f"FITS shape error")
    
    img = arr.ravel()

    return img
# data = {}

offsets = []
widths = []
for i in range(135,150):
    with np.load(f"{folder_dir}/spaxel_{i}.npz") as info:
        offsets.append(info['off_poly'][-1])

        widths.append(info['wid_poly'][-1])

offsets = _to_ascending(np.array(offsets))
widths = _to_ascending(np.array(widths))

spax_id = 136
total = None
solved = False
for k in range(15):
    print(f"STARTING GROUP {k}")
    spax_id = k*15
    if k == 136 // 15:
        try:
            neigh = build_neighbor_matrix(
                target_spaxel=spax_id,
                offsets=offsets,
                widths=widths,
                oversample_factor=4
            )
        except Exception:
            raise("comment out tasks or start a prefect server for build_forward_group.py")

        targ = build_target_matrix(
            target_spaxel=spax_id,
            offsets=offsets,
            widths=widths,
            o_pert=[0.0,0.0],
            oversample_factor=4
        )
    else:
        try:
            neigh = build_neighbor_matrix(
                    target_spaxel=spax_id,
                    offsets=np.zeros_like(offsets),
                    widths=np.zeros_like(widths),
                    oversample_factor=4
                )
        except Exception:
            raise("comment out tasks or start a prefect server for build_forward_group.py")
        targ = build_target_matrix(
            target_spaxel=spax_id,
            offsets=np.zeros_like(offsets),
            widths=np.zeros_like(widths),
            o_pert=[0.0,0.0],
            oversample_factor=4
        )


    data = np.concatenate((neigh[0], targ[0][0]))
    cols = np.concatenate((neigh[1], targ[0][1]))
    rows = np.concatenate((neigh[2], targ[0][2]))

    SPAXELS_PER_GROUP = 15
    SPEC_LEN = 1400
    group_id = spax_id // SPAXELS_PER_GROUP
    base_col = group_id * SPAXELS_PER_GROUP * SPEC_LEN
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
            f"Shape mismatch: group_spec length ({group_spec.shape[0]}) "
            f"does not match matrix columns ({A.shape[1]})"
        )

    At = A.T.tocsr()
    AtA = (At.dot(A)).tocsc() + 1e-10 * sparse.eye(MATRIX_SHAPE[1], format="csc")
    Atb = At.dot(b)
    x = spsolve(AtA, Atb)

    b = (A * x).reshape(DETECTOR_SHAPE)
    if solved:
        total += b
    else:
        total = b
        solved = True

plt.imshow(total)
plt.show()

np.save("science_recreation.npy",total)