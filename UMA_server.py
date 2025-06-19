#!/usr/bin/env python3

from __future__ import annotations
import argparse
import io
import json
import multiprocessing as mp
import os
import socket
import socketserver
import sys
import traceback

import numpy as np
import torch
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ── Constants ──────────────────────────────────────────────────────────────
EV_TO_HA              = 1 / 27.21138602
EV_ANGSTROM_TO_HA_BOHR = EV_TO_HA * 0.52917721067
FAIRCHEM_MODEL        = "uma-s-1"

# ── Helpers ────────────────────────────────────────────────────────────────
def run_fairchem_from_string(
    xyz_text: str,
    charge: int,
    mult: int,
    dograd: bool,
    ncores: int,
    predictor,
):
    """Core inference (no file I/O)."""
    torch.set_num_threads(ncores)
    atoms = read(io.StringIO(xyz_text), format="extxyz")
    atoms.info["charge"] = charge
    atoms.info["spin"]   = mult
    atoms.calc           = FAIRChemCalculator(predictor, task_name="omol")

    energy_eV   = atoms.get_potential_energy()
    gradient_eV = -atoms.get_forces() if dograd else []

    energy_ha = energy_eV * EV_TO_HA
    gradient_ha_bohr = (
        (gradient_eV * EV_ANGSTROM_TO_HA_BOHR).flatten().tolist() if dograd else []
    )
    return energy_ha, gradient_ha_bohr


# ── Custom single-thread TCP server class that supports SO_REUSEPORT ──────
class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True  # SO_REUSEADDR
    request_queue_size = 128

    def server_bind(self):
        # Enable both REUSEADDR and (if available) REUSEPORT
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            # SO_REUSEPORT not available (e.g. non-Linux) – workers must pick
            # different ports in that case.
            pass
        self.socket.bind(self.server_address)


# ── Worker process main loop ───────────────────────────────────────────────
def worker_main(
    worker_id: int,
    host: str,
    port: int,
    ncores: int,
):
    print(f"[Worker {worker_id}] loading FAIRChem model …", flush=True)
    predictor = pretrained_mlip.get_predict_unit(FAIRCHEM_MODEL, device="cpu")
    print(f"[Worker {worker_id}] ready (listening on {host}:{port})", flush=True)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                # ── receive length-prefixed JSON request ───────────────────
                hdr = self.request.recv(4)
                if not hdr:
                    return
                size = int.from_bytes(hdr, "big")
                data = b''
                while len(data) < size:
                    chunk = self.request.recv(size - len(data))
                    if not chunk:
                        return
                    data += chunk
                params = json.loads(data.decode())

                # ── read XYZ once, run model ───────────────────────────────
                with open(params["xyzfile"], "r") as fh:
                    xyz_txt = fh.read()

                energy, grad = run_fairchem_from_string(
                    xyz_txt,
                    params["charge"],
                    params["mult"],
                    params["dograd"],
                    params.get("ncores", 1),
                    predictor,
                )

                resp = json.dumps({"energy": energy, "gradient": grad}).encode()
                self.request.sendall(len(resp).to_bytes(4, "big") + resp)

            except Exception as exc:
                tb = traceback.format_exc()
                err = json.dumps({"error": str(exc)}).encode()
                # best-effort reply
                try:
                    self.request.sendall(len(err).to_bytes(4, "big") + err)
                except Exception:
                    pass
                print(f"[Worker {worker_id}] ERROR:\n{tb}", file=sys.stderr, flush=True)

    with ReuseTCPServer((host, port), Handler) as srv:
        # no threading mixin → single request at a time inside this process
        srv.serve_forever()


# ── Master process – launch pool and wait ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="Number of *processes* (= model instances)")
    ap.add_argument("--ncores",  type=int, default=1,
                    help="torch.set_num_threads per worker")
    args = ap.parse_args()

    # On non-Linux systems SO_REUSEPORT may be missing; in that case we let
    # each worker use port+worker_id instead.
    supports_reuseport = hasattr(socket, "SO_REUSEPORT")

    processes = []
    for w in range(args.workers):
        p = mp.Process(
            target=worker_main,
            args=(
                w,
                args.host,
                args.port if supports_reuseport else args.port + w,
                args.ncores,
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    print(f"[Master] spawned {len(processes)} workers.", flush=True)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("[Master] terminating …")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == "__main__":
    mp.set_start_method("fork")  # safest for model weights on Linux
    main()

