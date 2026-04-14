#!/usr/bin/env python3
"""
UMA client — ORCA otool_external interface for FAIRChem + optional XTB solvation.

Sends inference requests to a running UMA_server.py instance, optionally adds
an XTB solvation correction, and writes an ORCA-compatible .engrad file.

Usage:
    python UMA_client.py  inputfile  [--solvent SOLVENT] [--port PORT]

The input file is the ORCA external-tool control file with format:
    xyz_filename       # line 1
    charge             # line 2
    multiplicity       # line 3
    ncores             # line 4
    dograd  (0 or 1)   # line 5
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

# ── Constants ───────────────────────────────────────────────────────────
EV_TO_HA               = 1 / 27.21138602
EV_ANGSTROM_TO_HA_BOHR = EV_TO_HA * 0.52917721067
MAGIC                  = b"UMA1"
MAX_RETRY_DELAY        = 120.0               # cap exponential backoff


# ── Input parsing ───────────────────────────────────────────────────────
def _strip_comments(s: str) -> str:
    return s.split("#")[0].strip()


def read_input(inpfile: str | Path) -> tuple[str, int, int, int, bool]:
    """Parse the ORCA external-tool control file."""
    with Path(inpfile).open() as f:
        xyzname = _strip_comments(f.readline())
        charge  = int(_strip_comments(f.readline()))
        mult    = int(_strip_comments(f.readline()))
        ncores  = int(_strip_comments(f.readline()))
        dograd  = bool(int(_strip_comments(f.readline())))
    return xyzname, charge, mult, ncores, dograd


# ── XTB solvation helper ───────────────────────────────────────────────
def _read_engrad(path: str | Path, natoms: int) -> tuple[float | None, list[float]]:
    """Parse an XTB .engrad file → (energy_Eh, flat_gradient_Eh_bohr)."""
    lines = Path(path).read_text().splitlines()
    energy: float | None = None
    gradient: list[float] = []
    n = len(lines)
    i = 0

    # find energy
    while i < n:
        if "The current total energy in Eh" in lines[i]:
            i += 1
            while i < n and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
                i += 1
            if i < n:
                energy = float(lines[i].strip())
            break
        i += 1

    # find gradient
    for idx, line in enumerate(lines):
        if "The current gradient in Eh/bohr" in line:
            j = idx + 1
            while j < n and (lines[j].strip() == "" or lines[j].strip().startswith("#")):
                j += 1
            for k in range(j, min(j + 3 * natoms, n)):
                val = lines[k].strip()
                if val and not val.startswith("#"):
                    gradient.append(float(val))
            break

    return energy, gradient


def run_xtb(
    xyzname: str,
    charge: int,
    mult: int,
    solvent: str | None,
    ncores: int,
) -> tuple[float, list[float]]:
    """Run XTB, return (energy_Eh, flat_gradient_Eh_bohr)."""
    basename    = Path(xyzname).stem
    engrad_file = Path(xyzname).with_name(f"{basename}.engrad")

    cmd = ["xtb", xyzname, "--grad", "--acc", "0.2", "--norestart"]
    if solvent:
        cmd += ["--alpb", solvent]
    if charge is not None:
        cmd += ["--chrg", str(charge)]
    if mult is not None:
        cmd += ["--uhf", str(max(0, mult - 1))]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ncores)

    print(f"[XTB] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)

    with open(xyzname) as fh:
        natoms = int(fh.readline().strip())

    energy, gradient = _read_engrad(engrad_file, natoms)

    # clean up XTB scratch
    for fname in ("charges", "energy", "gradient", "wbo",
                  "xtbrestart", "xtbtopo.mol"):
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

    if energy is None:
        raise RuntimeError(f"XTB produced no energy in {engrad_file}")

    return energy, gradient


# ── FAIRChem server communication ───────────────────────────────────────
def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server disconnected")
        buf.extend(chunk)
    return bytes(buf)


def request_fairchem(
    xyzfile: str,
    charge: int,
    mult: int,
    dograd: bool,
    ncores: int,
    host: str  = "127.0.0.1",
    port: int  = 9099,
    retries: int   = 20,
    backoff_base: float = 2.0,
) -> dict:
    """Send an inference request to the UMA server; retry with backoff."""
    payload = json.dumps({
        "xyzfile": str(Path(xyzfile).resolve()),
        "charge":  charge,
        "mult":    mult,
        "dograd":  dograd,
        "ncores":  ncores,
    }).encode()

    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=10.0) as sock:
                sock.settimeout(300.0)     # generous read timeout

                # send: MAGIC + 4-byte length + JSON
                sock.sendall(MAGIC + len(payload).to_bytes(4, "big") + payload)

                # receive: MAGIC + 4-byte length + JSON
                magic = _recv_exact(sock, 4)
                if magic != MAGIC:
                    raise RuntimeError(f"bad magic in reply: {magic!r}")

                nbytes = int.from_bytes(_recv_exact(sock, 4), "big")
                data   = _recv_exact(sock, nbytes)

            result = json.loads(data)
            if "error" in result:
                raise RuntimeError(f"server error: {result['error']}")
            return result

        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(backoff_base ** (attempt - 1) + random.uniform(1, 5),
                        MAX_RETRY_DELAY)
            print(
                f"[FAIRChem client] attempt {attempt}/{retries} failed "
                f"({exc}). Retrying in {delay:.1f}s …",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay)

    # unreachable, but keeps type-checkers happy
    raise RuntimeError("all retries exhausted")


# ── ORCA .engrad writer ────────────────────────────────────────────────
def write_engrad(
    outfile: str | Path,
    natoms: int,
    energy: float,
    dograd: bool,
    gradient: list[float] | None = None,
) -> None:
    with Path(outfile).open("w") as f:
        f.write("#\n# Number of atoms\n#\n")
        f.write(f"{natoms:>5}\n")
        f.write("#\n# The current total energy in Eh\n#\n")
        f.write(f"{energy:>15.12f}\n")
        f.write("#\n# The current gradient in Eh/bohr\n#\n")
        if dograd and gradient:
            for g in gradient:
                f.write(f"{g:>15.12f}\n")


# ── Main ────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> None:
    default_port = int(os.getenv("UMA_PORT", "9099"))

    ap = ArgumentParser(
        prog=argv[0],
        allow_abbrev=False,
        description="ORCA external-tool client: FAIRChem + optional XTB solvation",
    )
    ap.add_argument("inputfile")
    ap.add_argument("--solvent", type=str, default=None,
                    help="ALPB solvent name (omit or 'none' to skip XTB)")
    ap.add_argument("--port", type=int, default=default_port,
                    help=f"FAIRChem server port (default $UMA_PORT or {default_port})")
    args = ap.parse_args(argv[1:])

    # ── parse control file ─────────────────────────────────────────────
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)

    # ── optional XTB solvation correction ──────────────────────────────
    skip_xtb = (args.solvent is None
                or args.solvent.lower() == "none")

    if skip_xtb:
        solvation_energy   = 0.0
        solvation_gradient: list[float] = []
    else:
        vacuum_e, vacuum_g = run_xtb(xyzname, charge, mult, None, ncores)
        sol_e, sol_g       = run_xtb(xyzname, charge, mult, args.solvent, ncores)

        solvation_energy   = sol_e - vacuum_e
        solvation_gradient = [s - v for s, v in zip(sol_g, vacuum_g)]

    # ── FAIRChem inference ─────────────────────────────────────────────
    result   = request_fairchem(
        xyzname, charge, mult, dograd, ncores,
        host="127.0.0.1", port=args.port,
    )
    energy   = result["energy"] + solvation_energy
    gradient = result["gradient"]          # already flat from server

    if dograd and solvation_gradient and len(solvation_gradient) == len(gradient):
        gradient = [g + s for g, s in zip(gradient, solvation_gradient)]

    # ── write ORCA .engrad ─────────────────────────────────────────────
    with open(xyzname) as fh:
        natoms = int(fh.readline())

    write_engrad(
        Path(xyzname).with_suffix(".engrad"),
        natoms, energy, dograd, gradient,
    )


if __name__ == "__main__":
    main(sys.argv)
