#!/usr/bin/env python3
"""SNIFS wavelength forward-model fit -- heterogeneous CPU/GPU Dask + Prefect."""

from __future__ import annotations

import atexit
import importlib
import os

# Limit MKL to 1 thread to prevent exponential memory buffer allocations
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import deque
from contextlib import contextmanager, nullcontext
from multiprocessing.shared_memory import SharedMemory

import dask

dask.config.set(
    {
        # Prevent Dask from over-submitting tasks to workers beyond capacity (default was 1.1)
        "distributed.diagnostics.nvml": False,
        "distributed.scheduler.worker-saturation": 0.8,
        # Memory management limits relative to worker memory_limit (e.g., CPU_MEM_LIMIT)
        "distributed.worker.memory.target": 0.65,  # Start spilling/clearing unneeded data at 60%
        "distributed.worker.memory.spill": 0.75,  # Aggressive memory management at 70%
        "distributed.worker.memory.pause": 0.80,  # Stop starting NEW tasks when worker reaches 80%
        "distributed.worker.memory.terminate": 0.95,  # Terminate worker if it hits 95% to prevent OS crash
    }
)
import dask.config
import numpy as np
from astropy.io import fits
from dask.distributed import LocalCluster, get_worker
from prefect import flow, get_run_logger, task
from prefect_dask.task_runners import DaskTaskRunner
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:  # Prefect 3
    from prefect.cache_policies import NO_CACHE

    _TASK_KW = {"cache_policy": NO_CACHE, "retries": 3, "retry_delay_seconds": 2}
except ImportError:  # Prefect 2
    _TASK_KW = {"retries": 0}

# ==============================================================================
# ENVIRONMENT & CONFIGURATION
# ==============================================================================
CUDA_DIR = os.environ.get("CUDA_HOME")
if CUDA_DIR:
    os.environ["CUDA_PATH"] = CUDA_DIR

    _cuda_bin = os.path.join(CUDA_DIR, "bin")
    if _cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_cuda_bin}{os.pathsep}{os.environ.get('PATH', '')}"
else:
    get_run_logger().warning("CUDA_HOME is not set. Did you forget to run 'module load cudatoolkit'?")
os.environ.setdefault("DASK_DISTRIBUTED__DIAGNOSTICS__NVML", "False")
# dask.config.set({"distributed.scheduler.worker-saturation": 1.1})


def _envint(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


NUM_SPAXELS = 225
SPAXELS_PER_GROUP = 15
NUM_GROUPS = NUM_SPAXELS // SPAXELS_PER_GROUP
MAX_ITERS = 10
N_OFFSET = 5
N_WIDTH = 6

NUM_GPUS = 4
NUM_CPU_WORKERS = _envint("MODEL_CPU_WORKERS", 32)
NUM_WORKERS_PER_GPU = 4
CPU_MEM_LIMIT = os.environ.get("MODEL_CPU_MEM", "3.5GiB")
GPU_MEM_LIMIT = os.environ.get("MODEL_GPU_MEM", "16GiB")

MAX_SPAXELS_IN_FLIGHT = _envint("MODEL_INFLIGHT", 12)
BUILD_PERTS_ONE_AT_A_TIME = False

DETECTOR_SHAPE = (4096, 2048)
N_DET_PIX = DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]
SPEC_LEN = 1400
MATRIX_ROWS = N_DET_PIX
MATRIX_COLS = SPEC_LEN * SPAXELS_PER_GROUP
MATRIX_SHAPE = (MATRIX_ROWS, MATRIX_COLS)

POLY_DEGREE = 4
POLY_NCOEF = POLY_DEGREE + 1
IDX_DTYPE = np.int32

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--group", type=int, required=True, help="Specify which group to forward model (0-14).")

parser.add_argument("--fits-path", type=str, required=True, help="The file path to the target FITS file.")

parser.add_argument("--output-dir", type=str, required=True, help="The file path to the save folder.")

args = parser.parse_args()


GROUP_MODELED = args.group

print(args.output_dir)
# TODO tasks 1, 2, and 3 the bin_saves, outdir and fits file path
from pathlib import Path

save_path = args.output_dir

folder_path = f"{save_path}/bin_saves"
out_dir = f"{save_path}/spaxel_fits"

Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(folder_path).mkdir(parents=True, exist_ok=True)
fits_path = args.fits_path

yoff = 0
n_bins = 255
heights = np.linspace(0, 4095, 256)


from pipeline.common.model_params import A0_PARAMS, A1_PARAMS
from pipeline.tasks.processing.build_forward_group import (
    build_neighbor_matrix,
    build_target_matrix,
)

HAS_REAL_PIPELINE = True

try:
    import cupy as cp

    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    HAS_GPU = False


def _to_ascending(c):
    """np.polyfit order (x^4 first) -> builder order (constant first).

    build_neighbor_matrix / build_target_matrix unpack O1..O5 as the
    coefficients of x^0..x^4, matching Polynomial([...]) in the reference.
    Everything on the Python side is np.polyfit order, so reverse here and
    only here. Works for both (5,) and (15, 5).
    """
    return np.ascontiguousarray(np.asarray(c, dtype=np.float64)[..., ::-1])


# ==============================================================================
# PER-PROCESS STATE & SHARED MEMORY
# ==============================================================================
def _holder():
    try:
        return get_worker()
    except (ValueError, ImportError, AttributeError):
        return sys.modules[__name__]


def _worker_store() -> dict:
    h = _holder()
    st = getattr(h, "_snifs_store", None)
    if st is None:
        st = {}
        h._snifs_store = st
    return st


def _shm_registry() -> dict:
    h = _holder()
    reg = getattr(h, "_snifs_shm", None)
    if reg is None:
        reg = {}
        h._snifs_shm = reg
    return reg


def _token_index() -> dict:
    h = _holder()
    tk = getattr(h, "_snifs_tokens", None)
    if tk is None:
        tk = {}
        h._snifs_tokens = tk
    return tk


def dask_setup(worker):
    worker._snifs_store = {}
    worker._snifs_shm = {}
    worker._snifs_tokens = {}


def dask_teardown(worker):
    for shm in list(getattr(worker, "_snifs_shm", {}).values()):
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
    getattr(worker, "_snifs_shm", {}).clear()


def _shm_put(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    meta = {"name": "", "shape": [int(s) for s in arr.shape], "dtype": arr.dtype.str, "nbytes": int(arr.nbytes)}
    if arr.nbytes == 0:
        return meta
    shm = SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    del view
    _shm_registry()[shm.name] = shm
    meta["name"] = shm.name
    return meta


def _put_triplet(trip) -> list:
    trip = list(trip)
    metas = []
    while trip:
        a = trip.pop(0)
        if a is not None and getattr(a, "dtype", None) is not None and a.dtype.kind in "iu":
            a = a.astype(IDX_DTYPE, copy=False)
        metas.append(_shm_put(a))
        del a
    return metas


def _open_shm(meta):
    if not meta["name"]:
        return None, np.empty(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]))
    shm = SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    return shm, arr


def assemble_triplets(neighbor_meta, target_meta):
    metas = list(neighbor_meta) + list(target_meta)
    handles, views = [], []
    data = rows = cols = None
    try:
        for m in metas:
            h, v = _open_shm(m)
            handles.append(h)
            views.append(v)
            del v
        data = np.concatenate((views[0], views[3]))
        rows = np.concatenate((views[1], views[4]))
        cols = np.concatenate((views[2], views[5]))
    finally:
        del views[:]
        for h in handles:
            if h is not None:
                try:
                    h.close()
                except BufferError:
                    pass
        handles.clear()
        import gc

        gc.collect()

    return data, rows, cols


def get_science_image() -> np.ndarray:
    st = _worker_store()
    img = st.get("science_flat")
    if img is not None:
        return img
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape != DETECTOR_SHAPE:
            raise ValueError(f"FITS shape {arr.shape} != {DETECTOR_SHAPE}")
    except Exception:
        arr = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
        raise ValueError("FITS shape error")

    img = arr.ravel()
    img.flags.writeable = False
    st["science_flat"] = img
    return img


def get_bin_index() -> np.ndarray:
    st = _worker_store()
    idx = st.get("bin_idx")
    if idx is not None:
        return idx
    y = np.repeat(np.arange(DETECTOR_SHAPE[0], dtype=np.int32), DETECTOR_SHAPE[1])
    b = np.digitize(y, np.floor(heights).astype(int)) - 1
    b = np.where((b >= 0) & (b < n_bins), b, n_bins).astype(np.int32)
    b.flags.writeable = False
    del y
    st["bin_idx"] = b
    return b


def get_spec_list():
    st = _worker_store()
    s = st.get("spec")
    if s is None:
        s = [np.ones(SPEC_LEN, dtype=np.float64) for _ in range(NUM_SPAXELS)]
        st["spec"] = s
    return s


def get_group_spectra(spax_id: int) -> np.ndarray:
    st = _worker_store()
    key = ("group_spec", spax_id // SPAXELS_PER_GROUP)
    out = st.get(key)
    if out is not None:
        return out
    spec = get_spec_list()
    lo = SPAXELS_PER_GROUP * (spax_id // SPAXELS_PER_GROUP)
    hi = min(lo + SPAXELS_PER_GROUP, len(spec))
    out = np.zeros(MATRIX_COLS, dtype=np.float64)
    if hi > lo:
        cat = np.concatenate([spec[i] for i in range(lo, hi)])
        n = min(cat.size, MATRIX_COLS)
        out[:n] = cat[:n]
        del cat
    out.flags.writeable = False
    st[key] = out
    return out


def compute_bin_stats(model_1d, science_1d) -> np.ndarray:
    bin_idx = get_bin_index()
    w = np.abs(np.subtract(science_1d, model_1d))
    finite = np.isfinite(w)
    if not finite.all():
        np.copyto(w, 0.0, where=~finite)
    sums = np.bincount(bin_idx, weights=w, minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(bin_idx, weights=finite.astype(np.float64), minlength=n_bins + 1)[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.where(counts > 0, sums, np.nan)  # / np.maximum(counts, 1.0)
    del w, finite, sums, counts
    return stats.astype(np.float64)


def fit_best_shift(shifts, values, degree=POLY_DEGREE):
    shifts = np.asarray(shifts, dtype=float)
    values = np.asarray(values, dtype=float)
    m = np.isfinite(shifts) & np.isfinite(values)
    k = int(m.sum())
    if k < 3:
        return np.nan
    deg = int(min(degree, k - 1))
    if deg < 1:
        return np.nan
    try:
        poly = np.poly1d(np.polyfit(shifts[m], values[m], deg))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    x_fine = np.linspace(shifts[m].min(), shifts[m].max(), 2000)
    return float(x_fine[np.argmin(poly(x_fine))])


def make_next_scalar_shifts(prev_shifts, whole_losses, n_params):
    return np.asarray(prev_shifts, dtype=float)
    # prev = np.asarray(prev_shifts, dtype=float)
    # whole = np.asarray(whole_losses, dtype=float)
    # if whole.size == 0 or np.all(np.isnan(whole)):
    #     return prev.astype(np.float32).copy()
    # # best = prev[int(np.nanargmin(whole))]
    # span = max((prev.max() - prev.min()) * 0.5, 1e-3)
    # return np.linspace(- span , span , n_params).astype(np.float32)


def np_interp_nearest(x_new, x, y):
    x, y, x_new = np.asarray(x), np.asarray(y), np.asarray(x_new)
    if x.size == 0 or x_new.size == 0:
        return np.full(x_new.shape, np.nan, dtype=float)
    if x.size == 1:
        return np.full(x_new.shape, y[0], dtype=float)
    idx = np.clip(np.searchsorted(x, x_new, side="left"), 1, x.size - 1)
    return np.where((x_new - x[idx - 1]) < (x[idx] - x_new), y[idx - 1], y[idx])


@task(name="Build_Matrices_GPU", **_TASK_KW)
def build_matrices_task(spax_id, is_offset, shifts, off_poly, wid_poly):
    t0 = time.perf_counter()
    shifts = np.asarray(shifts, dtype=np.float64).ravel()
    off_poly = np.asarray(off_poly, dtype=np.float64)
    wid_poly = np.asarray(wid_poly, dtype=np.float64)
    n_pert = int(shifts.size)
    token = f"{spax_id}-{uuid.uuid4().hex[:12]}"

    neighbor = build_neighbor_matrix(
        target_spaxel=spax_id, offsets=_to_ascending(off_poly), widths=_to_ascending(wid_poly), oversample_factor=4
    )
    neighbor_meta = _put_triplet(neighbor)
    del neighbor

    pert_metas = []
    if BUILD_PERTS_ONE_AT_A_TIME:
        for k in range(n_pert):
            one = shifts[k : k + 1]
            res = build_target_matrix(
                target_spaxel=spax_id,
                widths=_to_ascending(wid_poly),
                offsets=_to_ascending(off_poly),
                o_pert=one if is_offset else None,
                w_pert=None if is_offset else one,
                oversample_factor=4,
            )
            trip = res[0] if len(res) == 1 else res[k]
            pert_metas.append(_put_triplet(trip))
            del res, trip
    else:
        res = build_target_matrix(
            target_spaxel=spax_id,
            widths=_to_ascending(wid_poly),
            offsets=_to_ascending(off_poly),
            o_pert=shifts if is_offset else None,
            w_pert=None if is_offset else shifts,
            oversample_factor=4,
        )
        while res:
            pert_metas.append(_put_triplet(res.pop(0)))

    if HAS_GPU and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()

    names = [m["name"] for m in neighbor_meta if m["name"]]
    for grp in pert_metas:
        names += [m["name"] for m in grp if m["name"]]
    _token_index()[token] = names

    try:
        origin = get_worker().address
    except Exception:
        origin = ""

    nbytes = sum(m["nbytes"] for m in neighbor_meta) + sum(m["nbytes"] for g in pert_metas for m in g)

    return {
        "token": token,
        "origin": origin,
        "spax_id": int(spax_id),
        "n_pert": n_pert,
        "neighbor": neighbor_meta,
        "perts": pert_metas,
        "nbytes": int(nbytes),
        "build_s": round(time.perf_counter() - t0, 3),
    }


@task(name="Solve_Perturbation_CPU", **_TASK_KW)
def solve_perturbation_task(payload, p_idx):
    import gc

    t0 = time.perf_counter()
    spax_id = payload["spax_id"]

    data, cols, rows = assemble_triplets(payload["neighbor"], payload["perts"][p_idx])

    group_id = spax_id // SPAXELS_PER_GROUP
    base_col = group_id * SPAXELS_PER_GROUP * SPEC_LEN
    cols -= base_col

    A = sparse.csr_matrix(
        (data, (rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False))),
        shape=MATRIX_SHAPE,
    )
    del data, rows, cols
    gc.collect()  # Force cleanup of shared memory views

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

    try:
        At = A.T.tocsr()
        AtA = (At.dot(A)).tocsc() + 1e-10 * sparse.eye(MATRIX_SHAPE[1], format="csc")
        Atb = At.dot(b)
        x = spsolve(AtA, Atb)
        del AtA, Atb, At

        # del AtA, Atb, At, AtA_upper

    except (MemoryError, OSError) as e:
        # If OOM hits during the solve or matrix mult, catch it and fail gracefully
        get_run_logger().error(f"OOM caught on Spaxel {spax_id}: {str(e)} performing perturbation: {p_idx}")
        raise RuntimeError("Task failed due to out-of-memory error.") from e

    finally:
        # Guarantee PARDISO frees memory even if the solve fails
        if "solver" in locals():
            solver.free_memory()
            del solver
        # Clean up all heavy intermediates
        for var in ["AtA_upper", "Atb", "At", "AtA"]:
            if var in locals():
                del locals()[var]
        gc.collect()

    fit_model = A.dot(x)
    loss = float(np.sum(np.abs(get_science_image() - fit_model)))
    bin_stats = compute_bin_stats(fit_model, science)
    del A, x, fit_model

    return {
        "spax_id": int(spax_id),
        "p_idx": int(p_idx),
        "loss": loss,
        "bin_stats": bin_stats,
        "solve_s": round(time.perf_counter() - t0, 2),
    }


@task(name="Aggregate_Spaxel_CPU", **_TASK_KW)
def aggregate_spaxel_task(spax_id, prev_state, results, is_offset, curr_shifts, curr_iter):
    logger = get_run_logger()
    results = sorted(results, key=lambda r: r["p_idx"])
    losses = np.array([r["loss"] for r in results], dtype=float)
    bin_stats_arr = np.vstack([r["bin_stats"] for r in results]).astype(np.float64)

    prefix = "offsets" if is_offset else "widths"
    for r in results:
        np.save(f"{folder_path}/{prefix}_it{curr_iter}_sp{spax_id}_p{r['p_idx']}.npy", r["bin_stats"])

    curr_shifts = np.asarray(curr_shifts, dtype=float)
    best_bin = np.array([fit_best_shift(curr_shifts, bin_stats_arr[:, b]) for b in range(n_bins)], dtype=float)

    prev_coeffs = np.asarray(prev_state["off_poly"] if is_offset else prev_state["wid_poly"], dtype=float)
    coeffs = prev_coeffs.copy()

    a0 = int(A0_PARAMS[spax_id] + yoff)
    a1 = int(A1_PARAMS[spax_id] + yoff) + 1
    lo = max(0, min(n_bins, a0 // 16 - 2))
    hi = max(lo + 1, min(n_bins, a1 // 16 + 2))
    x_dense = np.arange(a0, max(a0, a1 - 1))

    if x_dense.size and hi > lo:
        # 1. Recreate interpolation nodes in detector-row coordinates
        bin_centres = 0.5 * (heights[:-1] + heights[1:])
        active_centres = bin_centres[lo:hi]
        active_best_bins = best_bin[lo:hi]

        # 2. Hard assertions to prevent silent extrapolation
        assert x_dense.min() >= active_centres[0], f"Query min {x_dense.min()} < Node min {active_centres[0]}"
        assert x_dense.max() <= active_centres[-1], f"Query max {x_dense.max()} > Node max {active_centres[-1]}"

        # 3. Perform the interpolation in matching units
        dense = np_interp_nearest(x_dense, active_centres, active_best_bins)

        fittable = np.array(dense, dtype=float, copy=True)
        s0, s1 = min(220, fittable.size), min(540, fittable.size)
        fittable[s0:s1] = np.nan
        mask = np.isfinite(fittable)
        if mask.sum() > POLY_NCOEF:
            try:
                coeffs = np.polyfit(np.arange(fittable.size)[mask], fittable[mask], POLY_DEGREE)
            except (np.linalg.LinAlgError, ValueError):
                raise RuntimeError(f"spaxel {spax_id}: polyfit failed, keeping previous coeffs")

    coeffs = np.asarray(coeffs, dtype=float).ravel()
    if coeffs.size != POLY_NCOEF:
        coeffs = np.resize(coeffs, POLY_NCOEF)

    next_shifts = make_next_scalar_shifts(curr_shifts, losses, curr_shifts.size)

    new_state = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prev_state.items()}
    if is_offset:
        new_state["offset_shifts"] = next_shifts
        new_state["off_poly"] = prev_state["off_poly"] + coeffs
    else:
        new_state["width_shifts"] = next_shifts
        new_state["wid_poly"] = prev_state["wid_poly"] + coeffs
    
    zero_idx = int(np.argmin(np.abs(curr_shifts)))
    new_state["last_loss"] = float(losses[zero_idx])  # loss at current accumulated state
    # new_state["loss_frac"] = float(losses[zero_idx] / results[zero_idx]["b_abs_sum"])

    return new_state


@task(name="Release_SHM_GPU", **_TASK_KW)
def release_matrices_task(payload):
    reg = _shm_registry()
    names = _token_index().pop(payload["token"], None)
    if names is None:
        names = [m["name"] for m in payload["neighbor"] if m["name"]]
        for g in payload["perts"]:
            names += [m["name"] for m in g if m["name"]]
    freed = 0
    for name in names:
        shm = reg.pop(name, None)
        if shm is None:
            try:
                shm = SharedMemory(name=name)
            except FileNotFoundError:
                continue
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except (FileNotFoundError, OSError):
            pass
        freed += 1
    return freed


# ==============================================================================
# SUBMISSION HELPERS
# ==============================================================================
@contextmanager
def _annot(**kw):
    kw = {k: v for k, v in kw.items() if v}
    try:
        cm = dask.annotate(**kw) if kw else nullcontext()
    except Exception:
        cm = nullcontext()
    with cm:
        yield


def _submit(task_obj, addrs, resource, *args, **kwargs):
    with _annot(resources={resource: 1}, workers=list(addrs) if addrs else None, allow_other_workers=False):
        return task_obj.submit(*args, **kwargs)


def _submit_pinned(task_obj, address, resource, *args, **kwargs):
    with _annot(resources={resource: 1}, workers=[address] if address else None, allow_other_workers=False):
        return task_obj.submit(*args, **kwargs)


# ==============================================================================
# FLOW
# ==============================================================================
@flow(name="Wavelength_Forward_Model")
def fit_forward_model(n_cpu_workers: int, n_gpu_workers: int, sched_address: str):
    logger = get_run_logger()
    from dask.distributed import Client

    wait_timeout = _envint("WORKER_WAIT_TIMEOUT", 600)
    poll_s = 15
    n_want = n_cpu_workers + n_gpu_workers
    with Client(sched_address, timeout="120s") as client:
        waited = 0
        while len(client.scheduler_info()["workers"]) < n_want and waited < wait_timeout:
            n_have = len(client.scheduler_info()["workers"])
            logger.info("waiting for workers: %d/%d registered (%ds/%ds)", n_have, n_want, waited, wait_timeout)
            time.sleep(poll_s)
            waited += poll_s
        # Raises WorkerStartTimeoutError with a clear final count if still short.
        client.wait_for_workers(n_want, timeout=1)
        info = client.scheduler_info()["workers"]
        gpu_addrs = [a for a, i in info.items() if "GPU" in (i.get("resources") or {})]
        cpu_addrs = [a for a, i in info.items() if "CPU" in (i.get("resources") or {})]
        logger.info("workers ready: %d CPU, %d GPU", len(cpu_addrs), len(gpu_addrs))

    # Added history tracking arrays for the polynomial outputs
    spaxel_states = {
        i: {
            "offset_shifts": np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32),
            "width_shifts": np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=np.float32),
            "off_poly": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "wid_poly": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            "off_poly_history": [],
            "wid_poly_history": [],
            "loss_history": [],
            "last_loss": float("nan"),
        }
        for i in range(NUM_SPAXELS)
    }

    for curr_iter in range(MAX_ITERS):
        is_offset = curr_iter % 2 == 0
        t_iter = time.time()

        group_off = {
            g: np.vstack(
                [spaxel_states[s]["off_poly"] for s in range(g * SPAXELS_PER_GROUP, (g + 1) * SPAXELS_PER_GROUP)]
            )
            for g in range(NUM_GROUPS)
        }
        group_wid = {
            g: np.vstack(
                [spaxel_states[s]["wid_poly"] for s in range(g * SPAXELS_PER_GROUP, (g + 1) * SPAXELS_PER_GROUP)]
            )
            for g in range(NUM_GROUPS)
        }

        inflight = deque()
        releases = []

        def retire_oldest():
            rec = inflight.popleft()
            spax = rec["spax"]

            # Fetch the previous history so we can append to it
            hist_off = spaxel_states[spax].get("off_poly_history", [])
            hist_wid = spaxel_states[spax].get("wid_poly_history", [])
            hist_loss = spaxel_states[spax].get("loss_history", [])

            # This blocks until the CPU aggregation (and implicitly the GPU tasks) are done
            new_st = rec["agg"].result()

            # Append newly processed polynomials to history lists
            new_st["off_poly_history"] = hist_off + [new_st["off_poly"]]
            new_st["wid_poly_history"] = hist_wid + [new_st["wid_poly"]]
            new_st["loss_history"] = hist_loss + [new_st["last_loss"]]

            spaxel_states[spax] = new_st

            # NOW we resolve the GPU payload future to get the origin address for cleanup
            resolved_payload = rec["payload_fut"].result()

            releases.append(_submit_pinned(release_matrices_task, resolved_payload["origin"], "GPU", resolved_payload))

        for group_id in range(NUM_GROUPS):
            if group_id != GROUP_MODELED:
                continue
            for spax_id in range(group_id * SPAXELS_PER_GROUP, (group_id + 1) * SPAXELS_PER_GROUP):
                while len(inflight) >= MAX_SPAXELS_IN_FLIGHT:
                    retire_oldest()

                st = spaxel_states[spax_id]
                shifts = st["offset_shifts"] if is_offset else st["width_shifts"]
                n_pert = int(np.size(shifts))

                # 1. REMOVE .result() from the end of this call!
                # We want to return a Future, not block waiting for the answer.
                payload_fut = _submit(
                    build_matrices_task,
                    gpu_addrs,
                    "GPU",
                    spax_id,
                    is_offset,
                    shifts,
                    group_off[group_id],
                    group_wid[group_id],
                )

                # 2. Pass the future (payload_fut) directly into the CPU task
                sol_futs = [_submit(solve_perturbation_task, cpu_addrs, "CPU", payload_fut, p) for p in range(n_pert)]

                agg = _submit(
                    aggregate_spaxel_task, cpu_addrs, "CPU", spax_id, st, sol_futs, is_offset, shifts, curr_iter
                )

                # 3. Store the payload future in your tracker instead of the resolved payload
                inflight.append({"spax": spax_id, "payload_fut": payload_fut, "agg": agg})

        while inflight:
            retire_oldest()
        for r in releases:
            r.result()

        logger.info(
            "iteration %d (%s) done in %.1f s", curr_iter, "offsets" if is_offset else "widths", time.time() - t_iter
        )

    # Save the 2D arrays containing history to the NPZ file
    for spax_id, st in spaxel_states.items():
        np.savez(
            f"{out_dir}/spaxel_{spax_id:03d}.npz",
            off_poly=np.array(st["off_poly_history"]),
            wid_poly=np.array(st["wid_poly_history"]),
            offset_shifts=st["offset_shifts"],
            width_shifts=st["width_shifts"],
            losses=np.array(st["loss_history"]),
        )

    return {k: {"off_poly": v["off_poly"], "wid_poly": v["wid_poly"]} for k, v in spaxel_states.items()}


# ==============================================================================
# PREFECT API & ENTRY POINT
# ==============================================================================
def _api_healthy(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_prefect_api(host="127.0.0.1", port=4200, timeout=180.0):
    api = os.environ.get("PREFECT_API_URL")
    if api and _api_healthy(api):
        return None
    api = f"http://{host}:{port}/api"
    if _api_healthy(api):
        os.environ["PREFECT_API_URL"] = api
        return None
    proc = subprocess.Popen(
        [sys.executable, "-m", "prefect", "server", "start", "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("prefect server exited")
        if _api_healthy(api):
            os.environ["PREFECT_API_URL"] = api
            return proc
        time.sleep(1.0)
    proc.terminate()
    raise TimeoutError("Prefect server timeout")


def _module_identity(path):
    d = os.path.dirname(os.path.abspath(path))
    parts = [os.path.splitext(os.path.basename(path))[0]]
    while os.path.isfile(os.path.join(d, "__init__.py")):
        parts.append(os.path.basename(d))
        d = os.path.dirname(d)
    return ".".join(reversed(parts)), d


def main():
    module_name, root = _module_identity(__file__)
    if root not in sys.path:
        sys.path.insert(0, root)

    server_proc = ensure_prefect_api()
    if server_proc is not None:
        atexit.register(lambda: (server_proc.terminate(), server_proc.wait()))

    pythonpath = os.pathsep.join([p for p in (root, os.environ.get("PYTHONPATH", "")) if p])
    worker_env = {
        "PYTHONPATH": pythonpath,
        "PREFECT_API_URL": os.environ["PREFECT_API_URL"],
        "PREFECT_LOGGING_LEVEL": os.environ.get("PREFECT_LOGGING_LEVEL", "INFO"),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
        "PATH": os.environ.get("PATH", ""),
        "DASK_DISTRIBUTED__DIAGNOSTICS__NVML": "False",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    os.environ["PYTHONPATH"] = pythonpath

    with LocalCluster(
        n_workers=NUM_CPU_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=CPU_MEM_LIMIT,
        resources={"CPU": 1},
        preload=[module_name],
        env=worker_env,
        dashboard_address=os.environ.get("SNIFS_DASHBOARD", ":0"),
    ) as cluster:
        sched = cluster.scheduler_address

        # 1. Setup the environment for the workers
        merged = os.environ.copy()
        merged.update(worker_env)
        # Force all simulated workers to use your single laptop GPU (GPU 0)
        merged["CUDA_VISIBLE_DEVICES"] = "0"

        # 2. Launch 3 separate Dask workers, pinning them all to GPU 0
        num_gpu_workers = NUM_GPUS * NUM_WORKERS_PER_GPU
        gpu_procs = []
        for i in range(num_gpu_workers):
            gpu_id = str(i % NUM_GPUS)  # Cycles through 0, 1, 2, 3
            worker_env = merged.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id

            gpu_cmd = [
                sys.executable,
                "-m",
                "dask",
                "worker",
                sched,
                "--nworkers",
                "1",
                "--nthreads",
                "1",
                "--resources",
                "GPU=1",
                "--memory-limit",
                GPU_MEM_LIMIT,
                "--no-nanny",
                "--preload",
                module_name,
                "--name",
                f"gpu-worker-gpu{gpu_id}-{i}",
            ]
            proc = subprocess.Popen(gpu_cmd, env=worker_env)
            gpu_procs.append(proc)

        # 3. Execute the flow and clean up
        try:
            fit_forward_model.with_options(task_runner=DaskTaskRunner(address=sched))(
                n_cpu_workers=NUM_CPU_WORKERS, n_gpu_workers=num_gpu_workers, sched_address=sched
            )
        finally:
            # Safely terminate all GPU worker processes
            for proc in gpu_procs:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    _name, _root = _module_identity(__file__)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        _mod = importlib.import_module(_name)
    except Exception as _exc:
        _mod = sys.modules["__main__"]
    _mod.main()

# 105:120
