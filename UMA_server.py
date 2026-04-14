#!/usr/bin/env python3
"""
Unified UMA FAIRChem inference server — CPU & GPU.

Fast startup via load-once-fork-many (CPU mode)
────────────────────────────────────────────────
The model is loaded ONCE in the master process, tensors are moved to POSIX
shared memory (/dev/shm), then workers are forked.  Each worker accesses the
SAME tensor data — no copies.  For GPU mode, workers use spawn and load
independently (CUDA requirement).

Adaptive thread scaling (CPU mode)
──────────────────────────────────
Threads per worker adapt dynamically: 1 active → all cores, N active → 1 each.

Launch examples
───────────────
  # Local checkpoint (fastest startup — recommended)
  python UMA_server.py --device cpu --model /path/to/uma-s-1p2.pt --workers 40 --ncores 40

  # HuggingFace model name (downloads/caches on first run)
  python UMA_server.py --device cpu --model uma-s-1p2 --workers 40 --ncores 40

  # GPU, 1 worker, local checkpoint
  python UMA_server.py --device cuda --model /path/to/uma-s-1p2.pt --workers 1

Protocol
────────
  Request :  b"UMA1" + 4-byte big-endian length + JSON
  Response:  b"UMA1" + 4-byte big-endian length + JSON
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import socket
import socketserver
import sys
import tempfile
import threading
import time
import traceback
import warnings

import torch
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

# ── Constants ───────────────────────────────────────────────────────────
EV_TO_HA               = 1 / 27.21138602
EV_ANGSTROM_TO_HA_BOHR = EV_TO_HA * 0.52917721067
MAGIC                  = b"UMA1"
MAX_PAYLOAD            = 50_000_000          # 50 MB sanity cap

log = logging.getLogger("uma-server")


# ── Inference ───────────────────────────────────────────────────────────
def run_inference(
    xyz_text: str,
    charge: int,
    mult: int,
    dograd: bool,
    predictor,
) -> tuple[float, list[float]]:
    """Pure computation — no file I/O, no side-effects."""
    atoms = read(io.StringIO(xyz_text), format="extxyz")
    atoms.info["charge"] = charge
    atoms.info["spin"]   = mult
    atoms.calc            = FAIRChemCalculator(predictor, task_name="omol")

    energy_eV   = atoms.get_potential_energy()
    gradient_eV = -atoms.get_forces() if dograd else []

    energy_ha = energy_eV * EV_TO_HA
    grad_ha_bohr = (
        (gradient_eV * EV_ANGSTROM_TO_HA_BOHR).flatten().tolist()
        if dograd else []
    )
    return energy_ha, grad_ha_bohr


# ── TCP helpers ─────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes or raise."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def send_response(sock: socket.socket, payload: dict) -> None:
    raw = json.dumps(payload).encode()
    sock.sendall(MAGIC + len(raw).to_bytes(4, "big") + raw)


# ── Threaded TCP server (accept-fast, infer-serial) ────────────────────
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    ThreadingMixIn ensures incoming connections are accepted immediately in
    their own thread, even while another thread holds the inference lock.
    This prevents ORCA clients from getting connection-refused errors during
    NumFreq bursts.
    """
    allow_reuse_address = True
    request_queue_size  = 256
    daemon_threads      = True          # threads die with the process

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.socket.bind(self.server_address)


# ── Adaptive thread-count helper ───────────────────────────────────────
class AdaptiveThreads:
    """
    Shared across all worker processes via multiprocessing.Value.
    Each worker increments the counter before inference and decrements after,
    then sets torch threads = total_cores / active_count.

    Result:
      1 active worker  → all cores  (fast single-step Opt)
      N active workers → ~1 core each (max throughput NumFreq)
    """

    def __init__(self, total_cores: int, active_counter: mp.Value):
        self._total   = total_cores
        self._counter = active_counter     # mp.Value('i', 0)

    def acquire(self) -> int:
        """Mark this worker as active; return thread count to use."""
        with self._counter.get_lock():
            self._counter.value += 1
            active = self._counter.value
        threads = max(1, self._total // active)
        torch.set_num_threads(threads)
        return threads

    def release(self) -> None:
        """Mark this worker as idle."""
        with self._counter.get_lock():
            self._counter.value = max(0, self._counter.value - 1)


def make_handler(
    predictor,
    inference_lock: threading.Lock,
    worker_id: int,
    adaptive: AdaptiveThreads | None,
):
    """Factory — returns a BaseRequestHandler class bound to this worker."""

    class _Handler(socketserver.BaseRequestHandler):

        def handle(self):
            t0 = time.monotonic()
            nthreads = 0
            try:
                # ── read magic + length-prefixed JSON ──────────────────
                try:
                    magic = recv_exact(self.request, 4)
                except ConnectionError:
                    return          # health-check probe — ignore silently
                if magic != MAGIC:
                    raise ValueError(
                        f"bad magic {magic!r} (expected {MAGIC!r}) — "
                        "stray connection?"
                    )
                size = int.from_bytes(recv_exact(self.request, 4), "big")
                if size > MAX_PAYLOAD:
                    raise ValueError(f"payload too large ({size} bytes)")

                params = json.loads(recv_exact(self.request, size))

                # ── validate ───────────────────────────────────────────
                missing = [k for k in ("xyzfile", "charge", "mult", "dograd")
                           if k not in params]
                if missing:
                    raise KeyError(f"missing field(s): {', '.join(missing)}")

                with open(params["xyzfile"]) as fh:
                    xyz_text = fh.read()

                # ── serialise inference inside this worker ─────────────
                with inference_lock:
                    if adaptive is not None:
                        nthreads = adaptive.acquire()
                    try:
                        energy, grad = run_inference(
                            xyz_text,
                            params["charge"],
                            params["mult"],
                            params["dograd"],
                            predictor,
                        )
                    finally:
                        if adaptive is not None:
                            adaptive.release()

                send_response(self.request, {"energy": energy, "gradient": grad})
                dt = time.monotonic() - t0
                thr_info = f" ({nthreads}T)" if adaptive is not None else ""
                log.info("[W%d] OK  %.3fs%s  %s",
                         worker_id, dt, thr_info, params["xyzfile"])

            except Exception as exc:
                tb = traceback.format_exc()
                log.error("[W%d] FAIL\n%s", worker_id, tb)
                try:
                    send_response(self.request, {"error": str(exc)})
                except Exception:
                    pass

    return _Handler


# ── Model loading (called once in master for CPU, per-worker for GPU) ──
def load_model(model: str, device: str, total_cores: int, gpu_id: str = None):
    """Load the FAIRChem model on the given device. Returns the predictor."""
    is_local = os.path.isfile(model)

    if device == "cpu":
        torch.set_num_threads(total_cores)
        settings = InferenceSettings(
            tf32=True,
            external_graph_gen=False,
            internal_graph_gen_version=2,
        )
        if is_local:
            return load_predict_unit(model, device="cpu", inference_settings=settings)
        else:
            return pretrained_mlip.get_predict_unit(model, device="cpu", inference_settings=settings)
    else:
        target = gpu_id or "cuda"
        if is_local:
            return load_predict_unit(model, device=target)
        else:
            return pretrained_mlip.get_predict_unit(model, device=target)


# ── Worker process entry point ──────────────────────────────────────────
def worker_main(
    worker_id: int,
    host: str,
    port: int,
    device: str,
    predictor,
    total_cores: int,
    active_counter,
):
    """
    For CPU mode:  predictor is inherited from the master via fork (fast).
    For GPU mode:  predictor is None; worker loads its own copy (CUDA requirement).
    """
    # Re-enable GC for new allocations; frozen (model) objects stay untouched
    gc.enable()
    # ── configure logging ──────────────────────────────────────────────
    logging.basicConfig(
        format=f"%(asctime)s [W{worker_id}] %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    logging.getLogger().setLevel(logging.ERROR)  # silence FAIRChem root warnings
    log.setLevel(logging.INFO)                   # keep our logger verbose

    # ── adaptive threads (CPU only) ────────────────────────────────────
    adaptive = None
    if device == "cpu" and active_counter is not None:
        adaptive = AdaptiveThreads(total_cores, active_counter)

    # ── start TCP server ───────────────────────────────────────────────
    inference_lock = threading.Lock()
    handler_cls    = make_handler(predictor, inference_lock, worker_id, adaptive)

    log.info("ready on %s:%d (core budget: %d)", host, port, total_cores)
    with ThreadedTCPServer((host, port), handler_cls) as srv:
        srv.serve_forever()


def gpu_worker_main(
    worker_id: int,
    host: str,
    port: int,
    model: str,
    total_cores: int,
):
    """GPU workers must load the model themselves (spawn, no shared memory)."""
    # ── suppress noise ─────────────────────────────────────────────────
    os.environ["WANDB_MODE"]     = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    warnings.filterwarnings("ignore", message=".*dataset_list.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

    logging.basicConfig(
        format=f"%(asctime)s [W{worker_id}] %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    logging.getLogger().setLevel(logging.ERROR)
    log.setLevel(logging.INFO)

    ngpu       = torch.cuda.device_count()
    gpu_device = f"cuda:{worker_id % ngpu}" if ngpu > 1 else "cuda"
    log.info("loading FAIRChem model on %s …", gpu_device)
    predictor = load_model(model, "cuda", total_cores, gpu_id=gpu_device)
    log.info("assigned to %s", gpu_device)

    inference_lock = threading.Lock()
    handler_cls    = make_handler(predictor, inference_lock, worker_id, None)

    log.info("ready on %s:%d", host, port)
    with ThreadedTCPServer((host, port), handler_cls) as srv:
        srv.serve_forever()


# ── Master process ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="UMA FAIRChem inference server (CPU & GPU)",
    )
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--port",    type=int, default=9099)
    ap.add_argument("--device",  default="cpu", choices=["cpu", "cuda"],
                    help="Inference device (default: cpu)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker processes "
                         "(default: cpu_count for cpu, gpu_count for cuda)")
    ap.add_argument("--ncores",  type=int, default=None,
                    help="Total CPU core budget shared across all workers. "
                         "Threads per worker adapt dynamically based on load. "
                         "(default: cpu_count)")
    ap.add_argument("--model",   type=str, default="uma-s-1p2",
                    help="Model name (HuggingFace) or path to local .pt file "
                         "(default: uma-s-1p2)")
    args = ap.parse_args()

    # ── master temp dir ────────────────────────────────────────────────
    master_tmp = tempfile.mkdtemp(prefix="uma_server_")
    os.environ["TMPDIR"]         = master_tmp
    os.environ["WANDB_MODE"]     = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    tempfile.tempdir             = master_tmp

    # ── suppress warnings before model load ────────────────────────────
    warnings.filterwarnings("ignore", message=".*dataset_list.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    logging.basicConfig(
        format="%(asctime)s [Master] %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    logging.getLogger().setLevel(logging.ERROR)
    log.setLevel(logging.INFO)

    # ── sensible defaults ──────────────────────────────────────────────
    if args.workers is None:
        if args.device == "cuda":
            args.workers = torch.cuda.device_count() or 1
        else:
            args.workers = os.cpu_count() or 4

    if args.ncores is None:
        args.ncores = os.cpu_count() or args.workers

    supports_reuseport = hasattr(socket, "SO_REUSEPORT")

    # ── shared active-worker counter (CPU only) ────────────────────────
    active_counter = mp.Value("i", 0) if args.device == "cpu" else None

    # ==================================================================
    #  CPU path: load model ONCE in master, then fork workers.
    #  Workers inherit the predictor via copy-on-write — near-instant.
    # ==================================================================
    if args.device == "cpu":
        is_local = os.path.isfile(args.model)
        log.info("loading model '%s' (%s) …",
                 args.model, "local" if is_local else "HuggingFace")
        t0 = time.monotonic()
        predictor = load_model(args.model, "cpu", args.ncores)
        dt = time.monotonic() - t0
        log.info("model loaded in %.1fs", dt)

        # Move ALL model tensors to POSIX shared memory (/dev/shm).
        # The predictor is an nn.Module — share_memory() recursively
        # moves every parameter and buffer to /dev/shm.
        # This is TRULY shared between forked processes — no COW copies.
        predictor.share_memory()
        n_params  = sum(1 for _ in predictor.parameters())
        n_buffers = sum(1 for _ in predictor.buffers())
        mem_mb    = (sum(t.nelement() * t.element_size() for t in predictor.parameters())
                   + sum(t.nelement() * t.element_size() for t in predictor.buffers())) / 1e6
        log.info("shared %d params + %d buffers (%.0f MB) via /dev/shm",
                 n_params, n_buffers, mem_mb)

        # Freeze GC to minimize Python-level COW on wrapper objects
        gc.collect()
        gc.freeze()
        torch.set_num_threads(1)

        # Fork AFTER loading, BEFORE any threads exist → safe
        processes: list[mp.Process] = []
        for w in range(args.workers):
            p = mp.Process(
                target=worker_main,
                args=(
                    w,
                    args.host,
                    args.port if supports_reuseport else args.port + w,
                    "cpu",
                    predictor,          # tensors in /dev/shm, shared across workers
                    args.ncores,
                    active_counter,
                ),
                daemon=False,
            )
            p.start()
            processes.append(p)

    # ==================================================================
    #  GPU path: must use spawn (CUDA requirement), each worker loads
    #  its own model copy.
    # ==================================================================
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for w in range(args.workers):
            p = mp.Process(
                target=gpu_worker_main,
                args=(
                    w,
                    args.host,
                    args.port if supports_reuseport else args.port + w,
                    args.model,
                    args.ncores,
                ),
                daemon=False,
            )
            p.start()
            processes.append(p)

    print(
        f"[Master] {args.workers} worker(s) on {args.device}, "
        f"model {args.model}, core budget {args.ncores}, port {args.port}",
        flush=True,
    )

    # ── graceful shutdown on signal ────────────────────────────────────
    def _shutdown(signum, frame):
        print("[Master] shutting down …", flush=True)
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
        shutil.rmtree(master_tmp, ignore_errors=True)
        print(f"[Master] cleaned up {master_tmp}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # ── wait for workers ───────────────────────────────────────────────
    try:
        for p in processes:
            p.join()
    finally:
        shutil.rmtree(master_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
