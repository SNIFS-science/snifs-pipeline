#!/usr/bin/env python3
"""
SNIFS wavelength forward-model fit -- heterogeneous CPU/GPU Dask + Prefect.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import atexit
import importlib
import subprocess
import urllib.request
from collections import deque
from contextlib import contextmanager, nullcontext
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from astropy.io import fits
from scipy import sparse
from scipy.sparse.linalg import spsolve

import dask
import dask.config
from dask.distributed import LocalCluster, get_worker

from prefect import flow, task, get_run_logger
from prefect_dask.task_runners import DaskTaskRunner

try:                                    # Prefect 3
    from prefect.cache_policies import NO_CACHE
    _TASK_KW = {"cache_policy": NO_CACHE, "retries": 0}
except ImportError:                     # Prefect 2
    _TASK_KW = {"retries": 0}

# ==============================================================================
# ENVIRONMENT & CONFIGURATION
# ==============================================================================
CUDA_DIR = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3")
_cuda_bin = os.path.join(CUDA_DIR, "bin")
if os.path.isdir(_cuda_bin):
    os.environ["CUDA_HOME"] = CUDA_DIR
    os.environ["CUDA_PATH"] = CUDA_DIR
    if _cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_cuda_bin}{os.pathsep}{os.environ.get('PATH','')}"
os.environ.setdefault("DASK_DISTRIBUTED__DIAGNOSTICS__NVML", "False")
dask.config.set({"distributed.scheduler.worker-saturation": 1.1})

def _envint(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

NUM_SPAXELS        = 15
SPAXELS_PER_GROUP  = 15
NUM_GROUPS         = NUM_SPAXELS // SPAXELS_PER_GROUP
MAX_ITERS          = 3
N_OFFSET           = 5
N_WIDTH            = 6

NUM_CPU_WORKERS    = _envint("SNIFS_CPU_WORKERS", max(1, 8))
NUM_GPU_WORKERS    = _envint("SNIFS_GPU_WORKERS", 1)
CPU_MEM_LIMIT      = os.environ.get("SNIFS_CPU_MEM", "14GiB")
GPU_MEM_LIMIT      = os.environ.get("SNIFS_GPU_MEM", "6GiB")

MAX_SPAXELS_IN_FLIGHT = _envint("SNIFS_INFLIGHT", 0) or max(2, NUM_CPU_WORKERS // N_WIDTH + 2)
BUILD_PERTS_ONE_AT_A_TIME = os.environ.get("SNIFS_BUILD_ONE", "1") == "1"

DETECTOR_SHAPE  = (4096, 2048)
N_DET_PIX       = DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]
SPEC_LEN        = 1400
MATRIX_ROWS     = N_DET_PIX
MATRIX_COLS     = SPEC_LEN * SPAXELS_PER_GROUP
MATRIX_SHAPE    = (MATRIX_ROWS, MATRIX_COLS)

POLY_DEGREE = 4
POLY_NCOEF  = POLY_DEGREE + 1
IDX_DTYPE = np.int32

folder_path = os.environ.get("SNIFS_BIN_DIR", "./bin_saves")
out_dir     = os.environ.get("SNIFS_OUT_DIR", "./spaxel_fits")
fits_path   = os.environ.get(
    "SNIFS_TEST_FITS",
    "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/P25_194_024_004_03_B.fits",
)

yoff    = 0
n_bins  = 255
heights = np.linspace(0, 4095, 256)

os.makedirs(folder_path, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

from pipeline.tasks.processing.build_forward_group import (
    build_neighbor_matrix,
    build_target_matrix,
)
from pipeline.common.model_params import A0_PARAMS, A1_PARAMS
HAS_REAL_PIPELINE = True

try:
    import cupy as cp
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    HAS_GPU = False

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
        setattr(h, "_snifs_store", st)
    return st

def _shm_registry() -> dict:
    h = _holder()
    reg = getattr(h, "_snifs_shm", None)
    if reg is None:
        reg = {}
        setattr(h, "_snifs_shm", reg)
    return reg

def _token_index() -> dict:
    h = _holder()
    tk = getattr(h, "_snifs_tokens", None)
    if tk is None:
        tk = {}
        setattr(h, "_snifs_tokens", tk)
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
    meta = {"name": "", "shape": [int(s) for s in arr.shape], "dtype": arr.dtype.str,
            "nbytes": int(arr.nbytes)}
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
    except Exception as exc:
        arr = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
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
    b = np.digitize(y, heights) - 1
    b = np.where((b >= 0) & (b < n_bins), b, n_bins).astype(np.int32)
    b.flags.writeable = False
    del y
    st["bin_idx"] = b
    return b

def get_spec_list():
    st = _worker_store()
    s = st.get("spec")
    if s is None:
        s = [np.abs(np.random.default_rng(1000 + i).standard_normal(SPEC_LEN)).astype(np.float64)
             for i in range(NUM_SPAXELS)]
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

# ==============================================================================
# NUMERICS
# ==============================================================================
def compute_bin_stats(model_1d, science_1d) -> np.ndarray:
    bin_idx = get_bin_index()
    w = np.abs(np.subtract(science_1d, model_1d))
    finite = np.isfinite(w)
    if not finite.all():
        np.copyto(w, 0.0, where=~finite)
    sums = np.bincount(bin_idx, weights=w, minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(bin_idx, weights=finite.astype(np.float64), minlength=n_bins + 1)[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    del w, finite, sums, counts
    return stats.astype(np.float32)

def fit_best_shift(shifts, values, degree=POLY_DEGREE):
    shifts = np.asarray(shifts, dtype=float)
    values = np.asarray(values, dtype=float)
    m = np.isfinite(shifts) & np.isfinite(values)
    k = int(m.sum())
    if k < 3:
        return np.nan
    deg = int(min(degree, k - 2))
    if deg < 1:
        return np.nan
    try:
        poly = np.poly1d(np.polyfit(shifts[m], values[m], deg))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    x_fine = np.linspace(shifts[m].min(), shifts[m].max(), 2000)
    return float(x_fine[np.argmin(poly(x_fine))])

def make_next_scalar_shifts(prev_shifts, whole_losses, n_params):
    prev = np.asarray(prev_shifts, dtype=float)
    whole = np.asarray(whole_losses, dtype=float)
    if whole.size == 0 or np.all(np.isnan(whole)):
        return prev.astype(np.float32).copy()
    best = prev[int(np.nanargmin(whole))]
    span = max((prev.max() - prev.min()) * 0.5, 1e-3)
    return np.linspace(best - span / 2, best + span / 2, n_params).astype(np.float32)

def np_interp_nearest(x_new, x, y):
    x, y, x_new = np.asarray(x), np.asarray(y), np.asarray(x_new)
    if x.size == 0 or x_new.size == 0:
        return np.full(x_new.shape, np.nan, dtype=float)
    if x.size == 1:
        return np.full(x_new.shape, y[0], dtype=float)
    idx = np.clip(np.searchsorted(x, x_new, side="left"), 1, x.size - 1)
    return np.where((x_new - x[idx - 1]) < (x[idx] - x_new), y[idx - 1], y[idx])

# ==============================================================================
# TASKS
# ==============================================================================
@task(name="Build_Matrices_GPU", **_TASK_KW)
def build_matrices_task(spax_id, is_offset, shifts, off_poly, wid_poly):
    t0 = time.perf_counter()
    shifts = np.asarray(shifts, dtype=np.float64).ravel()
    off_poly = np.asarray(off_poly, dtype=np.float64)
    wid_poly = np.asarray(wid_poly, dtype=np.float64)
    n_pert = int(shifts.size)
    token = f"{spax_id}-{uuid.uuid4().hex[:12]}"

    neighbor = build_neighbor_matrix(target_spaxel=spax_id, offsets=off_poly,
                                     widths=wid_poly, oversample_factor=1)
    neighbor_meta = _put_triplet(neighbor)
    del neighbor

    pert_metas = []
    if BUILD_PERTS_ONE_AT_A_TIME:
        for k in range(n_pert):
            one = shifts[k:k + 1]
            res = build_target_matrix(
                target_spaxel=spax_id, widths=wid_poly, offsets=off_poly,
                o_pert=one if is_offset else None,
                w_pert=None if is_offset else one,
            )
            trip = res[0] if len(res) == 1 else res[k]
            pert_metas.append(_put_triplet(trip))
            del res, trip
    else:
        res = build_target_matrix(
            target_spaxel=spax_id, widths=wid_poly, offsets=off_poly,
            o_pert=shifts if is_offset else None,
            w_pert=None if is_offset else shifts,
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

    nbytes = sum(m["nbytes"] for m in neighbor_meta) + \
        sum(m["nbytes"] for g in pert_metas for m in g)

    return {"token": token, "origin": origin, "spax_id": int(spax_id),
            "n_pert": n_pert, "neighbor": neighbor_meta, "perts": pert_metas,
            "nbytes": int(nbytes), "build_s": round(time.perf_counter() - t0, 3)}

@task(name="Solve_Perturbation_CPU", **_TASK_KW)
def solve_perturbation_task(payload, p_idx):
    t0 = time.perf_counter()
    spax_id = payload["spax_id"]

    data, cols, rows = assemble_triplets(payload["neighbor"], payload["perts"][p_idx])

    A = sparse.csr_matrix(
        (data, (rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False))),
        shape=MATRIX_SHAPE,
    )
    del data, rows, cols

    science = get_science_image()
    group_spec = get_group_spectra(spax_id)

    if group_spec.shape[0] == MATRIX_SHAPE[1]:
        shifted = A.dot(group_spec)
        b = np.where((shifted > 0.0) & np.isfinite(science), science, 0.0)
        del shifted
    else:
        b = np.array(science, dtype=np.float64)

    At = A.T.tocsr()
    AtA = (At.dot(A)).tocsc() + 1e-10 * sparse.eye(MATRIX_SHAPE[1], format="csc")
    Atb = At.dot(b)
    x = spsolve(AtA, Atb)
    del AtA, Atb, At

    fit_model = A.dot(x)
    loss = float(np.sum(np.abs(b - fit_model)))
    bin_stats = compute_bin_stats(fit_model, science)
    del A, x, fit_model, b

    return {"spax_id": int(spax_id), "p_idx": int(p_idx), "loss": loss,
            "bin_stats": bin_stats, "solve_s": round(time.perf_counter() - t0, 2)}

@task(name="Aggregate_Spaxel_CPU", **_TASK_KW)
def aggregate_spaxel_task(spax_id, prev_state, results, is_offset, curr_shifts, curr_iter):
    logger = get_run_logger()
    results = sorted(results, key=lambda r: r["p_idx"])
    losses = np.array([r["loss"] for r in results], dtype=float)
    bin_stats_arr = np.vstack([r["bin_stats"] for r in results]).astype(np.float64)

    prefix = "offsets" if is_offset else "widths"
    for r in results:
        np.save(f"{folder_path}/{prefix}_it{curr_iter}_sp{spax_id}_p{r['p_idx']}.npy",
                r["bin_stats"])

    curr_shifts = np.asarray(curr_shifts, dtype=float)
    best_bin = np.array([fit_best_shift(curr_shifts, bin_stats_arr[:, b]) for b in range(n_bins)],
                        dtype=float)

    prev_coeffs = np.asarray(prev_state["off_poly"] if is_offset else prev_state["wid_poly"],
                             dtype=float)
    coeffs = prev_coeffs.copy()

    a0 = int(A0_PARAMS[spax_id] + yoff)
    a1 = int(A1_PARAMS[spax_id] + yoff) + 1
    lo = max(0, min(n_bins, a0 // 16 - 2))
    hi = max(lo + 1, min(n_bins, a1 // 16 + 2))
    x_dense = np.arange(a0, max(a0, a1 - 1))

    if x_dense.size and hi > lo:
        dense = np_interp_nearest(x_dense, np.arange(n_bins)[lo:hi], best_bin[lo:hi])
        fittable = np.array(dense, dtype=float, copy=True)
        s0, s1 = min(220, fittable.size), min(540, fittable.size)
        fittable[s0:s1] = np.nan
        mask = np.isfinite(fittable)
        if mask.sum() > POLY_NCOEF:
            try:
                coeffs = np.polyfit(np.arange(fittable.size)[mask], fittable[mask], POLY_DEGREE)
            except (np.linalg.LinAlgError, ValueError):
                logger.warning("spaxel %d: polyfit failed, keeping previous coeffs", spax_id)
    
    coeffs = np.asarray(coeffs, dtype=float).ravel()
    if coeffs.size != POLY_NCOEF:
        coeffs = np.resize(coeffs, POLY_NCOEF)

    next_shifts = make_next_scalar_shifts(curr_shifts, losses, curr_shifts.size)

    new_state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in prev_state.items()}
    if is_offset:
        new_state["offset_shifts"], new_state["off_poly"] = next_shifts, coeffs
    else:
        new_state["width_shifts"], new_state["wid_poly"] = next_shifts, coeffs
    new_state["last_loss"] = float(np.nanmin(losses)) if losses.size else float("nan")

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
    with _annot(resources={resource: 1}, workers=list(addrs) if addrs else None,
                allow_other_workers=False):
        return task_obj.submit(*args, **kwargs)

def _submit_pinned(task_obj, address, resource, *args, **kwargs):
    with _annot(resources={resource: 1},
                workers=[address] if address else None,
                allow_other_workers=False):
        return task_obj.submit(*args, **kwargs)

# ==============================================================================
# FLOW
# ==============================================================================
@flow(name="Wavelength_Forward_Model")
def fit_forward_model(n_cpu_workers: int, n_gpu_workers: int, sched_address: str):
    logger = get_run_logger()
    from dask.distributed import Client

    with Client(sched_address, timeout="120s") as client:
        client.wait_for_workers(n_cpu_workers + n_gpu_workers, timeout=300)
        info = client.scheduler_info()["workers"]
        gpu_addrs = [a for a, i in info.items() if "GPU" in (i.get("resources") or {})]
        cpu_addrs = [a for a, i in info.items() if "CPU" in (i.get("resources") or {})]

    # Added history tracking arrays for the polynomial outputs
    spaxel_states = {
        i: {"offset_shifts": np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32),
            "width_shifts":  np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=np.float32),
            "off_poly": np.zeros(POLY_NCOEF),
            "wid_poly": np.zeros(POLY_NCOEF),
            "off_poly_history": [],
            "wid_poly_history": [],
            "last_loss": float("nan")}
        for i in range(NUM_SPAXELS)
    }

    for curr_iter in range(MAX_ITERS):
        is_offset = (curr_iter % 2 == 0)
        t_iter = time.time()

        group_off = {g: np.vstack([spaxel_states[s]["off_poly"]
                                   for s in range(g * SPAXELS_PER_GROUP,
                                                  (g + 1) * SPAXELS_PER_GROUP)])
                     for g in range(NUM_GROUPS)}
        group_wid = {g: np.vstack([spaxel_states[s]["wid_poly"]
                                   for s in range(g * SPAXELS_PER_GROUP,
                                                  (g + 1) * SPAXELS_PER_GROUP)])
                     for g in range(NUM_GROUPS)}

        inflight = deque()
        releases = []

        def retire_oldest():
            rec = inflight.popleft()
            spax = rec["spax"]
            
            # Fetch the previous history so we can append to it
            hist_off = spaxel_states[spax].get("off_poly_history", [])
            hist_wid = spaxel_states[spax].get("wid_poly_history", [])
            
            new_st = rec["agg"].result()
            
            # Append newly processed polynomials to history lists
            new_st["off_poly_history"] = hist_off + [new_st["off_poly"]]
            new_st["wid_poly_history"] = hist_wid + [new_st["wid_poly"]]
            
            spaxel_states[spax] = new_st
            releases.append(_submit_pinned(release_matrices_task, rec["origin"],
                                           "GPU", rec["payload"]))

        for group_id in range(NUM_GROUPS):
            for spax_id in range(group_id * SPAXELS_PER_GROUP,
                                 (group_id + 1) * SPAXELS_PER_GROUP):
                while len(inflight) >= MAX_SPAXELS_IN_FLIGHT:
                    retire_oldest()

                st = spaxel_states[spax_id]
                shifts = st["offset_shifts"] if is_offset else st["width_shifts"]
                n_pert = int(np.size(shifts))

                payload = _submit(build_matrices_task, gpu_addrs, "GPU",
                                  spax_id, is_offset, shifts,
                                  group_off[group_id], group_wid[group_id]).result()

                sol_futs = [_submit(solve_perturbation_task, cpu_addrs, "CPU", payload, p)
                            for p in range(n_pert)]
                agg = _submit(aggregate_spaxel_task, cpu_addrs, "CPU",
                              spax_id, st, sol_futs, is_offset, shifts, curr_iter)

                inflight.append({"spax": spax_id, "payload": payload,
                                 "origin": payload["origin"], "agg": agg})

        while inflight:
            retire_oldest()
        for r in releases:
            r.result()

        logger.info("iteration %d (%s) done in %.1f s", curr_iter,
                    "offsets" if is_offset else "widths", time.time() - t_iter)

    # Save the 2D arrays containing history to the NPZ file
    for spax_id, st in spaxel_states.items():
        np.savez(f"{out_dir}/spaxel_{spax_id:03d}.npz",
                 off_poly=np.array(st["off_poly_history"]),
                 wid_poly=np.array(st["wid_poly_history"]),
                 offset_shifts=st["offset_shifts"], width_shifts=st["width_shifts"])

    return {k: {"off_poly": v["off_poly"], "wid_poly": v["wid_poly"]}
            for k, v in spaxel_states.items()}

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
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
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
        dashboard_address=os.environ.get("SNIFS_DASHBOARD", ":8787"),
    ) as cluster:
        sched = cluster.scheduler_address

        gpu_cmd = [
            sys.executable, "-m", "dask", "worker", sched,
            "--nworkers", str(NUM_GPU_WORKERS),
            "--nthreads", "1",
            "--resources", "GPU=1",
            "--memory-limit", GPU_MEM_LIMIT,
            "--preload", module_name,
            "--name", "gpu-worker",
        ]
        merged = os.environ.copy()
        merged.update(worker_env)
        gpu_proc = subprocess.Popen(gpu_cmd, env=merged)

        try:
            fit_forward_model.with_options(
                task_runner=DaskTaskRunner(address=sched)
            )(n_cpu_workers=NUM_CPU_WORKERS, n_gpu_workers=NUM_GPU_WORKERS, sched_address=sched)
        finally:
            gpu_proc.terminate()
            try:
                gpu_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                gpu_proc.kill()

if __name__ == "__main__":
    _name, _root = _module_identity(__file__)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        _mod = importlib.import_module(_name)
    except Exception as _exc:
        _mod = sys.modules["__main__"]
    _mod.main()