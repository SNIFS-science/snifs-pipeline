#!/usr/bin/env python3

import queue
import threading
import time
import webbrowser
from multiprocessing.shared_memory import SharedMemory
from contextlib import ExitStack
import numpy as np
import psutil
from astropy.io import fits

# ------------------------------------------------------------------------------
# Prefect Imports
# ------------------------------------------------------------------------------
from prefect import flow, get_run_logger, task, get_client
from prefect.artifacts import create_markdown_artifact
from prefect.client.schemas.objects import State, StateType

from scipy import sparse
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------------------------
# Pipeline Imports
# ------------------------------------------------------------------------------
from pipeline.tasks.processing.build_forward_group import build_neighbor_matrix, build_target_matrix
from pipeline.common.model_params import A0_PARAMS, A1_PARAMS
# ------------------------------------------------------------------------------
# Optional GPU backend
# ------------------------------------------------------------------------------
import os

folder_path = "./bin_saves"

# Checks if the folder does not exist, then creates it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("bin_saves folder created!")
else:
    print("bin_saves folder already exists.")
    raise RuntimeError("rename the old folder or delete it.\n Program exiting")


try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cpsolve
    import pynvml

    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    raise RuntimeError("There is no GPUs availible")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_SPAXELS = 30
SPAXELS_PER_GROUP = 15
NUM_GROUPS = NUM_SPAXELS // SPAXELS_PER_GROUP
MAX_ITERS = 3

N_OFFSET = 5
N_WIDTH = 6

NUM_CPU_WORKERS = 3
NUM_GPU_WORKERS = 2
NUM_GPUS = 1

LOCAL_RAM_REQ_GB = 4.0  # check I think we should do 2 GB per GPU controller and 216/44 per worker
LOCAL_VRAM_REQ_GB = 5.0  # 13.8 min in real run

QUEUE_BOUND = 30
QUEUE_OFFLOAD_RATIO = 4

DETECTOR_SHAPE = (4096, 2048)
N_DET_PIX = DETECTOR_SHAPE[0] * DETECTOR_SHAPE[1]

# MATRIX SHAPE: 1400 (spectra length) * 15 (spaxels per group) rows and N_DET_PIX cols
SPEC_LEN = 1400
MATRIX_ROWS = N_DET_PIX
MATRIX_COLS = SPEC_LEN * SPAXELS_PER_GROUP
MATRIX_SHAPE = (MATRIX_ROWS, MATRIX_COLS)

fits_path = "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/P25_194_024_004_03_B.fits"

try:
    with fits.open(fits_path) as hdul:
        SCIENCE_IMAGE_2D = np.array(hdul[0].data, dtype=np.float64)
        SCIENCE_IMAGE_2D = np.nan_to_num(SCIENCE_IMAGE_2D, nan=0.0, posinf=0.0, neginf=0.0)
        # Flattened Science Image
        SCIENCE_IMAGE = SCIENCE_IMAGE_2D.ravel()
except Exception:
    # Fallback to zeros for testing if missing
    SCIENCE_IMAGE_2D = np.zeros(DETECTOR_SHAPE, dtype=np.float64)
    SCIENCE_IMAGE = SCIENCE_IMAGE_2D.ravel()

spec = [
    np.abs(np.random.default_rng(1000 + i).standard_normal(SPEC_LEN)).astype(np.float64) for i in range(NUM_SPAXELS)
]

#TESTING MODE 
yoff = 0

def kill_entire_flow(task, task_run, state):
    """
    Hook that runs automatically if the task fails. 
    It instructs Prefect to cancel the parent flow immediately.
    """
    flow_run_id = task_run.flow_run_id
    print(f"Task failed! Actively cancelling parent flow: {flow_run_id}")
    
    # Connect to the Prefect API Client to cancel the execution
    with get_client(sync_client=True) as client:
        client.set_flow_run_state(
            flow_run_id=flow_run_id,
            state=State(type=StateType.CANCELLING, name="CancellingDueToTaskFailure")
        )




def build_group_spectra(spax_id: int) -> np.ndarray:
    lo = SPAXELS_PER_GROUP * (spax_id // SPAXELS_PER_GROUP)
    hi = lo + SPAXELS_PER_GROUP
    group = [spec[i] for i in range(lo, min(hi, len(spec)))]
    if not group:
        return np.zeros(MATRIX_ROWS, dtype=np.float64)
    cat = np.concatenate(group)

    if cat.shape[0] >= MATRIX_ROWS:
        return cat[:MATRIX_ROWS]
    out = np.zeros(MATRIX_ROWS, dtype=np.float64)
    out[: cat.shape[0]] = cat
    return out


spec: np.ndarray = np.ones((225, 1400))
heights: np.ndarray = np.linspace(0, 4095, 256)
n_bins: int = len(heights) - 1
x_sparse: np.ndarray = np.linspace(0, 4095, 256) + 8


def stat_l1(science: np.ndarray, model: np.ndarray, sl) -> float:
    """Per-bin L1 over a detector row-slice."""
    s = np.asarray(science)[sl]
    m = np.asarray(model)[sl]
    flag = np.isfinite(s) & np.isfinite(m)
    if not np.any(flag):
        return np.nan
    return float(np.sum(np.abs(s[flag] - m[flag])))


def compute_bin_stats(model_2d: np.ndarray, science_2d: np.ndarray) -> np.ndarray:
    """Per-bin L1 vector for one (spaxel, param) model."""
    bin_stats = np.full(n_bins, np.nan, dtype=np.float32)
    for bin_idx in range(n_bins):
        sl = (slice(int(heights[bin_idx]), int(heights[bin_idx + 1])), slice(None))
        try:
            bin_stats[bin_idx] = stat_l1(science_2d, model_2d, sl)
        except Exception:
            print(Exception)
    return bin_stats


def fit_best_shift(shifts, values, degree=4):
    """Fit poly to (shift, value); return shift at minimum."""
    shifts = np.asarray(shifts, dtype=float) #[-.2, -.1,0,.1,.2]
    values = np.asarray(values, dtype=float) #loss of each p_idx in a bin (bin done by thing outside)

    m = np.isfinite(shifts) & np.isfinite(values)
    if m.sum() <= degree:
        raise RuntimeError(f"Not enough values in shifts to fit a {degree} polynomial.")
        return np.nan
    coeffs = np.polyfit(shifts[m], values[m], degree)
    poly = np.poly1d(coeffs)
    x_fine = np.linspace(np.min(shifts[m]), np.max(shifts[m]), 2000)
    return x_fine[np.argmin(poly(x_fine))]


def make_next_scalar_shifts(prev_shifts, whole_losses, n_params):
    """Recenter the coarse grid on the best whole-model loss and shrink span."""
    prev_shifts = np.asarray(prev_shifts, dtype=float)
    whole = np.asarray(whole_losses, dtype=float)
    if np.all(np.isnan(whole)):
        return prev_shifts.copy()
    best = prev_shifts[np.nanargmin(whole)]
    span = (prev_shifts.max() - prev_shifts.min()) * 0.5
    if span <= 0:
        span = 1.0
    return np.linspace(best - span / 2, best + span / 2, n_params).astype(np.float32)


# ==============================================================================
# EVENT-DRIVEN RESOURCE GUARDS
# ==============================================================================
class CPUMemoryGuard:
    def __init__(self, required_gb: float = 1.0):
        self.cond = threading.Condition()
        self.required_gb = required_gb

    def check_ram(self) -> bool:
        avail_gb = psutil.virtual_memory().available / (1024**3)
        return avail_gb >= self.required_gb

    def acquire(self):
        with self.cond:
            while not self.check_ram():
                self.cond.wait(timeout=5.0)

    def release(self):
        with self.cond:
            self.cond.notify_all()


class GPUMemoryGuard:
    def __init__(self, num_gpus: int = 1, required_gb: float = 1.0):
        self.required_gb = required_gb
        self.conditions = {gpu_id: threading.Condition() for gpu_id in range(num_gpus)}

    def check_vram(self, gpu_id: int) -> bool:
        if not HAS_GPU:
            return True
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.free / (1024**3)) >= self.required_gb
        except Exception:
            try:
                dev = cp.cuda.Device(gpu_id)
                return (dev.mem_info[0] / (1024**3)) >= self.required_gb
            except Exception:
                return True

    def acquire(self, gpu_id: int):
        cond = self.conditions[gpu_id]
        with cond:
            while not self.check_vram(gpu_id):
                cond.wait(timeout=5.0)

    def release(self, gpu_id: int):
        cond = self.conditions[gpu_id]
        with cond:
            cond.notify_all()


cpu_mem_guard = CPUMemoryGuard(required_gb=LOCAL_RAM_REQ_GB)
gpu_mem_guard = GPUMemoryGuard(num_gpus=NUM_GPUS, required_gb=LOCAL_VRAM_REQ_GB)


# ==============================================================================
# GLOBAL STATE & DIAGNOSTIC TRACKING
# ==============================================================================
# maybe change global state so that different semantic steps call different gloabals. like retrieve loss is indep of
# retrieving offset solution hist
class GlobalState:
    def __init__(self):
        self.lock = threading.Lock()
        self.iter_count = np.zeros(dtype=int)
        self.stopped = np.zeros(dtype=bool)

        self.offset_loss = np.zeros((MAX_ITERS, N_OFFSET), dtype=np.float32)
        self.width_loss = np.zeros((MAX_ITERS, N_WIDTH), dtype=np.float32)
        self.width_loss[:,-1] = 1  # CHECK

        self.offset_loss_ready = np.zeros((N_OFFSET), dtype=bool)
        self.width_loss_ready = np.zeros((N_WIDTH), dtype=bool)

        # Reformatted to explicitly define correct dimension counts
        self.offset_perturbations = np.broadcast_to(
            np.array([-0.2, -0.1, 0, 0.1, 0.2])[None, :], (MAX_ITERS // 2, N_OFFSET)
        ).astype(np.float32)
        self.width_perturbations = np.broadcast_to(
            np.array([-0.2, -0.1, 0, 0.1, 0.2, 0.3])[None, :], (MAX_ITERS // 2, N_WIDTH)
        ).astype(np.float32)

        self.offset_solution_hist = np.zeros((MAX_ITERS // 2, 5), dtype=np.float32)
        self.width_solution_hist = np.zeros((MAX_ITERS // 2, 5), dtype=np.float32)

        # # Prefect Diagnostic Timers
        # self.cpu_solve_times = []
        # self.gpu_solve_times = []
        # self.matrix_build_times = []

    # def reset_diagnostics(self):
    #     with self.lock:
    #         self.cpu_solve_times.clear()
    #         self.gpu_solve_times.clear()
    #         self.matrix_build_times.clear()


state = {i: GlobalState() for i in range(NUM_SPAXELS)}
def all_states_stopped():
    for i in range(NUM_SPAXELS):
        if not state[i].stopped:
            return False

    return True

gpu_queue = queue.Queue(maxsize=QUEUE_BOUND)
cpu_queue = queue.Queue(maxsize=QUEUE_BOUND)


# ==============================================================================
# SOLVERS & WORKERS
# ==============================================================================
def solve_system_cpu(data, rows, cols, shape, group_spectra, science_flat):
    t0 = time.perf_counter()
    A = sparse.csr_matrix((data, (rows, cols)), shape=shape)
    b = np.asarray(science_flat, dtype=np.float64).ravel()

    # Apply the mask based on model support
    if group_spectra is not None and group_spectra.shape[0] == shape[1]:
        shifted = A.dot(group_spectra)
        flag = (shifted > 0.0) & np.isfinite(b)
        b = np.where(flag, b, 0.0)

    AtA = A.T.dot(A).tocsc()
    Atb = A.T.dot(b)
    AtA = AtA + 1e-10 * sparse.eye(AtA.shape[0], format="csc")
    x = spsolve(AtA, Atb)

    fit_model = A.dot(x)
    loss = float(np.sum(np.abs(b - fit_model)))

    elapsed = time.perf_counter() - t0
    # with state.lock:
    #     state.cpu_solve_times.append(elapsed)
    return loss, fit_model


def solve_system_gpu(data, rows, cols, shape, group_spectra, science_flat, gpu_id=0):
    if not HAS_GPU:
        return solve_system_cpu(data, rows, cols, shape, group_spectra, science_flat)

    t0 = time.perf_counter()
    with cp.cuda.Device(gpu_id):
        A_g = cpsparse.csr_matrix((cp.asarray(data), (cp.asarray(rows), cp.asarray(cols))), shape=shape)
        b_g = cp.asarray(np.asarray(science_flat, dtype=np.float64).ravel())

        # Apply the mask based on model support (GPU side)
        if group_spectra is not None and group_spectra.shape[0] == shape[1]:
            spec_g = cp.asarray(group_spectra)
            shifted_g = A_g.dot(spec_g)
            flag_g = (shifted_g > 0.0) & cp.isfinite(b_g)
            b_g = cp.where(flag_g, b_g, 0.0)

        AtA_g = A_g.T.dot(A_g).tocsc()
        Atb_g = A_g.T.dot(b_g)
        x_g = cpsolve.lsqr(AtA_g, Atb_g)[0]
        fit_model_g = A_g.dot(x_g)
        loss = float(cp.sum(cp.abs(b_g - fit_model_g)).get())
        fit_model = cp.asnumpy(fit_model_g)

    elapsed = time.perf_counter() - t0
    # with state.lock:
    #     state.gpu_solve_times.append(elapsed)
    return loss, fit_model

def np_interp_nearest(x_new, x, y):
    # Find the insertion indices to keep x sorted
    idx = np.searchsorted(x, x_new, side='left')
    
    # Clip the indices to prevent out-of-bounds errors at the boundaries
    idx = np.clip(idx, 1, len(x) - 1)
    
    # Check if the point is closer to the left or right neighbor
    left_closer = (x_new - x[idx - 1]) < (x[idx] - x_new)
    
    # Return the y value of the closest neighbor
    return np.where(left_closer, y[idx - 1], y[idx])

def update_spaxel_loss(spax_id, p_idx, loss, model_1d, is_offset, poly_degree=4):
    # Reshape model to 2D for spatial binning analysis
    try:
        model_2d = np.asarray(model_1d).reshape(DETECTOR_SHAPE)
        bin_stats = compute_bin_stats(model_2d, SCIENCE_IMAGE_2D)
        if is_offset:
            np.save(bin_stats, f"{folder_path}/offsets_{spax_id}_{p_idx}")
        else:
            np.save(bin_stats, f"{folder_path}/widths_{spax_id}_{p_idx}")
    except Exception:
        raise RuntimeError("Bin stats not computed correctly instead of update_spaxel_loss")
        # bin_stats = np.full(n_bins, np.nan, np.float32)

    with state[spax_id].lock:
        curr_iter = state[spax_id].iter_count

        # to enforce asynchronus synchronization, we oculd make the compute condition
        # for the widths to be np.all without the spax_id ebcause then
        # the cpu workers in the queue would be forced to finish the
        # widths before moving to iter 3. problem is idle time
        if is_offset:
            state[spax_id].offset_loss[p_idx] = loss
            state[spax_id].offset_loss_ready[p_idx] = True
            all_ready = np.all(state[spax_id].offset_loss_ready)
            shifts = state[spax_id].offset_solutions.copy()
            whole_loss = state[spax_id].offset_loss.copy()
        else:
            state[spax_id].width_loss[p_idx] = loss
            state[spax_id].width_loss_ready[p_idx] = True
            all_ready = np.all(state[spax_id].width_loss_ready)
            shifts = state[spax_id].width_solutions.copy()
            whole_loss = state[spax_id].width_loss.copy()

        # Exit early if this spaxel is still waiting on other shift evaluations
        if not all_ready:
            return


    # 2. Compute Spatial Interpolation (A0/A1 Generation) | not necessary
    best_bin_shifts = np.zeros(n_bins, dtype=np.float32)

    bin_stats = []
    if is_offset:
        for i in range(N_OFFSET):
            with np.load(f"{folder_path}/offsets_{spax_id}_{i}") as data:
                bin_stats.append(data)
    else:
        for i in range(N_WIDTH):
            with np.load(f"{folder_path}/widths_{spax_id}_{i}") as data:
                bin_stats.append(data)

    # For each spatial bin, fit the L1 loss curve to find the optimal shift
    for b in range(n_bins):
        # Extract the loss values for this specific bin across all tested shifts
        best_bin_shifts[b] = fit_best_shift(shifts, bin_stats[:][b]) #check if this is right

    # Fit a spatial polynomial across the detector bins to get A0/A1 coefficients
    x_sparse = np.arange(n_bins)
    valid_mask = np.isfinite(best_bin_shifts)
    a0 = int(A0_PARAMS[spax_id] + yoff)
    a1 = int(A1_PARAMS[spax_id] + yoff) + 1

    
    x_dense = np.arange(a0, a1 - 1)
    best_bin_shifts = np_interp_nearest(x_dense, x_sparse[a0 // 16 - 2 : a1 // 16 + 2], best_bin_shifts[a0 // 16 - 2 : a1 // 16 + 2])

    #spectrum is dominated by other spaxels in this range and has a wierd unexpected shape so we mask it out during calibration
    fittable = np.copy(best_bin_shifts)
    fittable[220:540] = np.nan

    if np.sum(valid_mask) > poly_degree:
        # We have enough valid bins to fit the polynomial trace

        x_fittable = np.arange(len(fittable))
        mask_fin = np.isfinite(fittable)
        if mask_fin.sum() < 5:
            print(f"Spaxel {spax_id}: insufficient finite values for polyfit, skipping", flush=True)
            raise RuntimeError("Not enough valid bins to perform wavelength solution offset / width fitting after masking 220-540")
        coeffs = np.polyfit(x_fittable[mask_fin], fittable[mask_fin], poly_degree)

    else:
        raise RuntimeError(f"Not enough best bin shifts (less than {poly_degree})")
        # Fallback to the global best shift if the matrix is heavily masked/corrupted
        best_scalar = shifts[np.nanargmin(whole_loss)] if np.any(np.isfinite(whole_loss)) else 0.0
        coeffs = np.zeros(poly_degree + 1)
        coeffs[-1] = best_scalar

    # ======================================================================
    # 3. Advance the parameters for the next iteration
    # ======================================================================
    next_iter = curr_iter + 1

    # Store History
    with state[spax_id].lock:
        if is_offset:
            state[spax_id].offset_solution_hist[curr_iter] = np.asarray(coeffs)
            state[spax_id].offset_loss_ready[:] = False
        else:
            state[spax_id].width_solution_hist[curr_iter] = np.asarray(coeffs)
            state[spax_id].width_loss_ready[:] = False

        if next_iter < MAX_ITERS:
            # Prepare next grid centered around the best global loss
            n_params = N_OFFSET if is_offset else N_WIDTH
            next_shifts = make_next_scalar_shifts(shifts, whole_loss, n_params)

            if is_offset:
                state[spax_id].offset_solutions = next_shifts
            else:
                state[spax_id].width_solutions = next_shifts
            gpu_queue.put(spax_id)

        else:
            state[spax_id].stopped = True
        state[spax_id].iter_count += 1



def write_mat_to_shm(spax_id, p_idx, data, rows, cols):
    data_np = cp.asnumpy(data)  # Example extraction
    rows_np = np.asarray(rows)
    cols_np = np.asarray(cols)

    # 3. Allocate Shared Memory blocks for the large arrays
    shm_data = SharedMemory(create=True, size=data_np.nbytes)
    shm_rows = SharedMemory(create=True, size=rows_np.nbytes)
    shm_cols = SharedMemory(create=True, size=cols_np.nbytes)

    # 4. Copy the data into the shared memory buffers
    np.ndarray(data_np.shape, dtype=data_np.dtype, buffer=shm_data.buf)[:] = data_np[:]
    np.ndarray(rows_np.shape, dtype=rows_np.dtype, buffer=shm_rows.buf)[:] = rows_np[:]
    np.ndarray(cols_np.shape, dtype=cols_np.dtype, buffer=shm_cols.buf)[:] = cols_np[:]

    # 5. Close the GPU's access to the handles (DO NOT unlink here)
    shm_data.close()
    shm_rows.close()
    shm_cols.close()

    # 6. Pass lightweight metadata to the CPU queue
    # Using a dict is safer than a massive tuple to prevent unpack errors
    return {
        "spax_id": spax_id,
        "shape_tuple": data.shape,  # Pass small scalars normally
        "p_idx": p_idx,
        # SHM Pointers and shapes required to reconstruct the arrays
        "shm_info": {
            "data": {"name": shm_data.name, "shape": data_np.shape, "dtype": data_np.dtype},
            "rows": {"name": shm_rows.name, "shape": rows_np.shape, "dtype": rows_np.dtype},
            "cols": {"name": shm_cols.name, "shape": cols_np.shape, "dtype": cols_np.dtype},
        },
    }

@task(on_failure=[kill_entire_flow])
def gpu_worker_task(gpu_id=0):
    while True:
        try:
            spax_id = gpu_queue.get(timeout=1.0)
            group_id = spax_id // SPAXELS_PER_GROUP
            group_start = group_id * SPAXELS_PER_GROUP
            group_end = group_start + SPAXELS_PER_GROUP
        except queue.Empty:
            if all_states_stopped():
                return
            continue

        gpu_mem_guard.acquire(gpu_id)
        try:
            with state[spax_id].lock:
                if state[spax_id].stopped:
                    gpu_queue.task_done()
                    continue

                curr_iter = state[spax_id].iter_count
                offset_ind = curr_iter // 2 - (1 if (curr_iter != 0) and is_offset else 0)
                width_ind = curr_iter // 2 - (1 if (curr_iter > 1) else 0)
                is_offset = (curr_iter % 2) == 0

                off_pert = None
                wid_pert = None
                if is_offset:
                    off_pert = state[spax_id].offset_perturbations[offset_ind].copy()
                else:
                    wid_pert = state[spax_id].width_perturbations[width_ind].copy()

            group_states = [state[i] for i in range(group_start, group_end)]
            acquired_all = False

            while not acquired_all:
                with ExitStack() as stack:
                    for s in group_states:
                        # Try to grab the lock immediately without waiting
                        success = s.lock.acquire(blocking=False)
                        
                        if not success:
                            # SOMEONE ELSE HAS A LOCK! 
                            # Breaking out of the 'with' block forces ExitStack to 
                            # instantly release all locks we gathered up to this point.
                            break 
                        
                        # If successful, tell ExitStack to manage its release later
                        stack.register(s.lock.release)
                    else:
                        # The 'else' block only runs if the 'for' loop finished completely 
                        # without hitting a 'break' (meaning we successfully got all 15 locks)
                        acquired_all = True
                        
                        # combine the polynomial solutions accross spaxels. V stacking means [spax_id][poly_solu_id]
                        off_poly = np.vstack([s.offset_solution_hist[offset_ind] for s in group_states])
                        wid_poly = np.vstack([s.width_solution_hist[width_ind] for s in group_states])
                        
                if not acquired_all:
                    # We failed to get all 15 locks. Take a micro-nap to let the other 
                    # thread finish with spaxel 1, then the 'while' loop will try again.
                    time.sleep(0.001) 


            # TODO make mps work here and so that HAS GPU actually does something
            if HAS_GPU:
                neighbor_data = build_neighbor_matrix(
                    target_spaxel=spax_id, offsets=off_poly, widths=wid_poly, oversample_factor=1
                )

                pert_data = build_target_matrix(
                    target_spaxel=spax_id, widths=wid_poly, offsets=off_poly, o_pert=off_pert, w_pert=wid_pert
                )

                for i, pert_tup in enumerate(pert_data):
                    ndata, nrow, ncol = neighbor_data
                    pdata, prow, pcol = pert_tup

                    data = cp.concatenate(ndata, pdata)
                    rows = cp.concatenate(nrow, prow)
                    cols = cp.concatenate(ncol, pcol)

                    item_meta = write_mat_to_shm(spax_id, i, data, rows, cols)
                    cpu_queue.put(item_meta)
                # IMPLEMENT SHM HERE SO THAT CPU WORKER THAT PICKS UP THIS QUEUE
                # maybe put some information int he queue so the
                # cpu worker that picks up this item can findthe info generated by the GPU worker

        finally:
            gpu_mem_guard.release(gpu_id)
            gpu_queue.task_done()

@task(on_failure=[kill_entire_flow])
def cpu_worker_task():
    while True:
        try:
            item = cpu_queue.get(timeout=1.0)
        except queue.Empty:
            if all_states_stopped() and cpu_queue.empty():
                return
            continue

        cpu_mem_guard.acquire()
        shm_info = item["shm_info"]
        shm_data = SharedMemory(name=shm_info["data"]["name"])
        shm_rows = SharedMemory(name=shm_info["rows"]["name"])
        shm_cols = SharedMemory(name=shm_info["cols"]["name"])
        spax_id = SharedMemory(name=shm_info["spax_id"])
        p_idx = SharedMemory(name=shm_info["p_idx"])
        try:
            # 2. Reconstruct zero-copy numpy arrays from the memory buffers
            data = np.ndarray(shm_info["data"]["shape"], dtype=shm_info["data"]["dtype"], buffer=shm_data.buf)
            rows = np.ndarray(shm_info["rows"]["shape"], dtype=shm_info["rows"]["dtype"], buffer=shm_rows.buf)
            cols = np.ndarray(shm_info["cols"]["shape"], dtype=shm_info["cols"]["dtype"], buffer=shm_cols.buf)

            # 3. Execute your CPU workload
            loss, model = solve_system_cpu(
                data,
                rows,
                cols,
                item["shape_tuple"],
                None,  # missing group spectra
                None,  # missing science image
            )

            with state[spax_id].lock:
                curr_iter = state[spax_id].iter_count
                is_offset = (curr_iter % 2) == 0
                state[spax_id].stopped = curr_iter == MAX_ITERS

            update_spaxel_loss(
                spax_id,
                p_idx,
                loss,
                model,
                is_offset
            )

        finally:
            shm_data.close()
            shm_rows.close()
            shm_cols.close()

            shm_data.unlink()
            shm_rows.unlink()
            shm_cols.unlink()
            cpu_mem_guard.release()
            cpu_queue.task_done()


# ==============================================================================
# PREFECT WORKFLOW
# ==============================================================================
@task
def process_group_step(k: int):
    logger = get_run_logger()
    # state.reset_diagnostics() TODO replace

    pert_spaxel_ids = [k + SPAXELS_PER_GROUP * n for n in range(NUM_GROUPS)]
    logger.info(f"--- Starting Step k={k} | Testing Spaxels: {pert_spaxel_ids} ---")

    for sid in pert_spaxel_ids:
        gpu_queue.put(sid)

    # Spawn GPU Workers
    gpu_threads = []
    for _ in range(NUM_GPU_WORKERS):
        t = threading.Thread(target=gpu_worker_task, args=(0,))
        t.start()
        gpu_threads.append(t)

    # Spawn CPU Workers
    cpu_threads = []
    for _ in range(NUM_CPU_WORKERS):
        t = threading.Thread(target=cpu_worker_task)
        t.start()
        cpu_threads.append(t)

    # Wait for completion
    gpu_queue.join()
    cpu_queue.join()

    for t in gpu_threads + cpu_threads:
        t.join()

    # # Calculate Diagnostics Metrics TODO REPLACE
    # avg_cpu = np.mean(state.cpu_solve_times) if state.cpu_solve_times else 0.0
    # avg_gpu = np.mean(state.gpu_solve_times) if state.gpu_solve_times else 0.0
    # avg_build = np.mean(state.matrix_build_times) if state.matrix_build_times else 0.0

    # logger.info(
    #     f"[Diagnostics Step k={k}] CPU Solves: {len(state.cpu_solve_times)} (Avg {avg_cpu * 1000:.2f}ms) | GPU Solves: {len(state.gpu_solve_times)} (Avg {avg_gpu * 1000:.2f}ms)"
    # )

    # # Send Prefect Dashboard Artifact
    # markdown_report = f"""
    # ### 📊 Performance Diagnostics — Step k={k}

    # | Metric | Output Value |
    # | :--- | :--- |
    # | **Matrix Assembly Time (Avg)** | `{avg_build * 1000:.2f} ms` |
    # | **CPU Solve Count** | `{len(state.cpu_solve_times)}` |
    # | **CPU Solve Duration (Avg)** | `{avg_cpu * 1000:.2f} ms` |
    # | **GPU Solve Count** | `{len(state.gpu_solve_times)}` |
    # | **GPU Solve Duration (Avg)** | `{avg_gpu * 1000:.2f} ms` |
    # | **Active CPU Workers** | `{NUM_CPU_WORKERS}` |
    # | **Active GPU Workers** | `{NUM_GPU_WORKERS}` |
    # """
    # create_markdown_artifact(
    #     key=f"step-k-{k}-diagnostics", markdown=markdown_report, description=f"Step k={k} Performance Metrics"
    # )


@flow(name="Wavelength_Forward_Model_Local_Test")
def calibration_flow():
    for k in range(NUM_GROUPS):
        process_group_step(k)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:4200")
    calibration_flow()

    save_dict = {name: value for name, value in vars(state).items() if name != "lock"}
    np.savez("./test_offset_results.npz", **save_dict)
