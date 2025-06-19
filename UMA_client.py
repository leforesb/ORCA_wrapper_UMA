#!/usr/bin/env python3
import sys
import socket
import json
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import socket as pysocket
import os
import subprocess
import time
import random



# --- Constants ---
EV_TO_HA = 1 / 27.21138602
EV_ANGSTROM_TO_HA_BOHR = 1 / 27.21138602 * 0.52917721067

def strip_comments(s):
    return s.split("#")[0].strip()

def enforce_path_object(fname):
    if isinstance(fname, str):
        return Path(fname)
    elif isinstance(fname, Path):
        return fname
    else:
        raise TypeError("Input must be a string or a Path object.")

def read_input(inpfile):
    inpfile = enforce_path_object(inpfile)
    with inpfile.open() as f:
        xyzname = strip_comments(f.readline())
        charge = int(strip_comments(f.readline()))
        mult = int(strip_comments(f.readline()))
        ncores = int(strip_comments(f.readline()))
        dograd = bool(int(strip_comments(f.readline())))
    return xyzname, charge, mult, ncores, dograd

def extract_nprocs_from_inpfile():
    inp_files = list(Path(".").glob("*.inp"))
    if not inp_files:
        print("ERROR: No .inp file found to extract nprocs.", file=sys.stderr)
        sys.exit(1)
    target_file = None
    if len(inp_files) == 1:
        target_file = inp_files[0]
    else:
        compound_file = [f for f in inp_files if "_Compound_0.inp" in f.name]
        if len(compound_file) == 1:
            target_file = compound_file[0]
        else:
            print("ERROR: Multiple .inp files found, and could not uniquely identify _Compound_0.inp.", file=sys.stderr)
            sys.exit(1)
    with target_file.open() as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith("nprocs"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
                else:
                    break
    print(f"ERROR: 'nprocs' not found or malformed in the file '{target_file.name}'.", file=sys.stderr)
    sys.exit(1)


def flatten_gradient(gradient):
    flat = []
    for g in gradient:
        if isinstance(g, (list, tuple)):
            flat.extend(g)
        else:
            flat.append(g)
    return flat

def create_xtb_cmd(xyzname, charge, mult, solvent=None, ncores=1):
    cmd = [
        "xtb", xyzname,
        "--grad",
        "--acc", "0.2", "--norestart"
    ]
    if solvent:
        cmd += ["--alpb", solvent]
    if charge is not None:
        cmd += ["--chrg", str(charge)]
    if mult is not None:
        n_unpaired = max(0, mult - 1)
        cmd += ["--uhf", str(n_unpaired)]
    return cmd

def read_engrad_file(engrad_file, natoms):
    energy = None
    gradient = []
    try:
        with open(engrad_file, "r") as f:
            lines = f.readlines()
        n = len(lines)
        i = 0
        # Find energy
        while i < n:
            if "The current total energy in Eh" in lines[i]:
                i += 1
                while i < n and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
                    i += 1
                if i < n:
                    energy = float(lines[i].strip())
                break
            i += 1
        # Find gradient block
        grad_start = None
        for idx, line in enumerate(lines):
            if "The current gradient in Eh/bohr" in line:
                grad_start = idx + 1
                while grad_start < n and (lines[grad_start].strip() == "" or lines[grad_start].strip().startswith("#")):
                    grad_start += 1
                break
        if grad_start is not None:
            for j in range(grad_start, grad_start + 3 * natoms):
                if j >= n:
                    break
                val = lines[j].strip()
                if val != "" and not val.startswith("#"):
                    try:
                        gradient.append(float(val))
                    except Exception:
                        pass
    except Exception as e:
        print(f"ERROR reading {engrad_file}: {e}", file=sys.stderr)
    return energy, gradient

def run_xtb(
    xyzname: str,
    charge: int,
    mult: int,
    solvent: str = None,
    ncores: int = 1
) -> tuple[float, list[float]]:
    from pathlib import Path
    import os
    import subprocess

    xyzname_path = Path(xyzname)
    basename = xyzname_path.stem
    engrad_file = xyzname_path.with_name(f"{basename}.engrad")

    # Prepare xtb command
    cmd = [
        "xtb", xyzname,
        "--grad",
        "--acc", "0.2", "--norestart"
    ]
    if solvent:
        cmd += ["--alpb", solvent]
    if charge is not None:
        cmd += ["--chrg", str(charge)]
    if mult is not None:
        # uhf = number of unpaired electrons = multiplicity - 1
        n_unpaired = max(0, mult - 1)
        cmd += ["--uhf", str(n_unpaired)]

    # Set number of threads
    env = os.environ.copy()
#    env["OMP_NUM_THREADS"] = str(1)
    env["OMP_NUM_THREADS"] = str(ncores)

    # Run xtb
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # Read number of atoms
    with open(xyzname, "r") as f:
        natoms = int(f.readline().strip())
    
    # Read energy and gradient from engrad file (same as before)
    energy, gradient = read_engrad_file(engrad_file, natoms)

    # Clean up xtb extra files
    for fname in ["charges", "energy", "gradient", "wbo", "xtbrestart", "xtbtopo.mol"]:
        try:
            os.remove(fname)
        except Exception:
            pass  # Ignore missing files

    if energy is None:
        print(f"ERROR: No energy found in {engrad_file}", file=sys.stderr)
    if not gradient:
        print(f"WARNING: No gradient found in {engrad_file}", file=sys.stderr)
    return energy, gradient


def _open_socket(host: str, port: int,
                 connect_timeout: float = 5.0,
                 read_timeout: float = 120.0) -> socket.socket:
    """
    Return a connected socket with separate time-outs:
    * connect_timeout  – maximum time to establish TCP connection
    * read_timeout     – maximum idle time while waiting for data
    """
    sock = socket.create_connection((host, port), timeout=connect_timeout)
    sock.settimeout(read_timeout)
    return sock



def request_fairchem_server(
        xyzfile: str,
        charge: int,
        mult: int,
        dograd: bool,
        ncores: int,
        host: str,
        port: int,
        retries: int = 10,
        backoff_base: float = 1.3,
) -> dict:
    params = {
        "xyzfile": str(Path(xyzfile).resolve()),
        "charge":  charge,
        "mult":    mult,
        "dograd":  dograd,
        "ncores":  ncores,
    }
    payload = json.dumps(params).encode()

    for attempt in range(1, retries + 1):
        try:
            with _open_socket(host, port) as sock:
                sock.sendall(len(payload).to_bytes(4, "big") + payload)

                # receive 4-byte length header
                hdr = sock.recv(4)
                if len(hdr) != 4:
                    raise RuntimeError("Incomplete length header from server")
                nbytes = int.from_bytes(hdr, "big")

                # receive full payload
                data = b''
                while len(data) < nbytes:
                    chunk = sock.recv(nbytes - len(data))
                    if not chunk:
                        raise RuntimeError("Connection closed mid-reply")
                    data += chunk

            result = json.loads(data.decode())

            if "error" in result:
                raise RuntimeError(f"Server error: {result['error']}")

            return result                       # ← SUCCESS

        except Exception as exc:
            if attempt == retries:
                raise                          # bubble up final failure
            delay = (backoff_base ** (attempt - 1)) + random.uniform(1, 5)
            print(
                f"[FAIRChem client] attempt {attempt}/{retries} failed "
                f"({exc}). Retrying in {delay:.2f}s …",
                file=sys.stderr,
            )
            time.sleep(delay)


def write_engrad(outfile, natoms, energy, dograd, gradient=None):
    outfile = enforce_path_object(outfile)
    with outfile.open("w") as f:
        output = "#\n# Number of atoms\n#\n"
        output += f"{natoms:>5}\n#\n# The current total energy in Eh\n#\n"
        output += f"{energy:>15.12f}\n#\n# The current gradient in Eh/bohr\n#\n"
        if dograd and gradient is not None:
            for g in gradient:
                output += f"{g:>15.12f}\n"
        f.write(output)

def main(argv):
    # ──── default port from env ────────────────────────────
    default_port = int(os.getenv("UMA_PORT", "9099"))

    parser = ArgumentParser(
        prog=argv[0],
        allow_abbrev=False,
        description="Client for UMA calculation + optional XTB solvation",
    )
    parser.add_argument("inputfile")
    parser.add_argument("--solvent", type=str, default=None,
                        help="ALPB solvent name (*None* to skip XTB)")
    # expose --port, default from $UMA_PORT
    parser.add_argument("--port", type=int, default=default_port,
                        help="FAIRChem server port "
                             "(default from $UMA_PORT or 9099)")
    args = parser.parse_args(argv[1:])

    # ── Read control file ─────────────────────────────────────
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)
    nprocs_from_inp = extract_nprocs_from_inpfile()
    input_basename  = Path(args.inputfile).stem

    # ── Optional XTB solvent correction ───────────────────────
    skip_xtb = args.solvent is not None and args.solvent.lower() == "none"

    if skip_xtb:
        solvation_energy   = 0.0
        solvation_gradient = []
    else:
        vacuum_e, vacuum_g = run_xtb(xyzname, charge, mult, None, ncores)
        if args.solvent:
            sol_e, sol_g = run_xtb(xyzname, charge, mult, args.solvent, ncores)
        else:
            sol_e, sol_g = vacuum_e, vacuum_g

        solvation_energy   = sol_e - vacuum_e
        vacuum_g           = flatten_gradient(vacuum_g)
        sol_g              = flatten_gradient(sol_g)
        solvation_gradient = [s - v for s, v in zip(sol_g, vacuum_g)]

    # ── FAIRChem inference ────────────────────────────────────
    result   = request_fairchem_server(
        xyzname, charge, mult, dograd, ncores,
        host="127.0.0.1", port=args.port
    )
    energy   = result["energy"] + solvation_energy
    gradient = flatten_gradient(result["gradient"])

    if dograd and solvation_gradient and len(solvation_gradient) == len(gradient):
        gradient = [g + s for g, s in zip(gradient, solvation_gradient)]

    # ── Write ORCA .engrad ────────────────────────────────────
    with open(xyzname) as fh:
        natoms = int(fh.readline())
    out_file = Path(xyzname).with_suffix(".engrad")
    write_engrad(out_file, natoms, energy, dograd, gradient)

if __name__ == "__main__":
    main(sys.argv)

