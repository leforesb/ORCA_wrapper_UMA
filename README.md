# ORCA_wrapper_UMA

In this repository are wrapper scripts to interface Fairchem's Universal Model for Atoms (UMA) with ORCA 6 through the "ExtOpt" option, taking advantage of ORCA's algorithms for geometry optimizations, transition state search and conformational sampling (GOAT). The server supports both CPU and GPU inference. The default UMA model is "uma-s-1p2", configurable via the `--model` flag (accepts either a HuggingFace model name or a path to a local `.pt` checkpoint).

Implicit solvation is implemented as an additive corrective term for both energy and gradients, calculated using the Grimme group's ALPB solvation scheme through [xtb](https://github.com/grimme-lab/xtb).

The two scripts work together as server and client. The server part needs to be started before the subsequent ORCA calculation. The server takes care of the heavy import steps of torch and fairchem once, before the start of the ORCA calculation, with all subsequent UMA calculations being handled through requests by the client side.

### CPU mode (default)
The model is loaded **once** in the master process, then its tensors are moved to POSIX shared memory (`/dev/shm`). Worker processes are forked and share the same model data with zero copies — startup is near-instant regardless of the number of workers. The `--ncores` flag sets the **total CPU core budget** shared across all workers. Threads per worker adapt dynamically based on load: when a single worker is active (e.g. during an optimization step) it uses all cores; when many workers are active simultaneously (e.g. during a numerical frequency calculation) each gets roughly one core. This maximizes throughput in both scenarios automatically.

### GPU mode (`--device cuda`)
Each worker spawns independently and loads its own model copy (required by CUDA). Workers are distributed across available GPUs in round-robin fashion. The number of workers defaults to the number of detected GPUs.

### Reliability
The server uses a `ThreadingMixIn` so that incoming connections are accepted immediately even while another request is being processed. This prevents ORCA clients from getting connection-refused errors during NumFreq bursts. On the client side, requests are retried with exponential backoff (up to 20 attempts), so transient failures do not crash the ORCA calculation.

## Requirements
- A working environment (conda or other) for both ORCA 6 (i.e. OpenMPI) and Fairchem's UMA model, see requirements in the [fairchem](https://github.com/facebookresearch/fairchem) repository. The script uses the implementation of the UMA model as a calculator in ASE.
- If using a HuggingFace model name (e.g. `uma-s-1p2`), an account on HuggingFace with access to the UMA model through an access token is required. You will be required to log in on your local machine to access it. Refer to the documentation on the [HuggingFace page](https://huggingface.co/facebook/UMA). Alternatively, you can pass a local `.pt` checkpoint path via `--model` to skip the HuggingFace dependency entirely.
- ORCA 6.1.0. The script as it is will work with ORCA 6.0.1, as long as you do not try to do numerical frequencies in parallel with it, as ORCA fails for not being able to read a valid ".hostnames" file. This was corrected in 6.1.0, see Forum post [here](https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=13416).
- [xtb](https://github.com/grimme-lab/xtb) for the implicit solvation component (only needed if `--solvent` is used).

## Example ORCA input
ORCA will need to call the client script through the ExtOpt option and the ProgExt keyword.
If implicit solvation is required, it can be requested as an argument through the Ext_Params keyword with the `--solvent SOLVENT` flag. This will trigger two xtb calculations: one in vacuum and one in the requested solvent with the ALPB solvation scheme. The corrective solvation terms for the potential energy and for gradients are calculated as the difference of the corresponding terms in the two calculations, and are added to the UMA energy and gradient outputs. If the `--solvent` flag is omitted (or set to `none`), the xtb calculations are skipped entirely and you get the pure UMA energy and gradient. Here is a simple ORCA input calling the client script:
```
! ExtOpt Opt

%method
      ProgExt "/path/to/your/wrapper/scripts/directory/UMA_client.py"
      Ext_Params "--solvent thf"  # omit entirely or "--solvent none" to skip xtb
end

%pal
        nprocs 16
end

*xyzfile 0 1 input.xyz

```

## Example Slurm script excerpt
The server script needs to be launched before the start of the ORCA calculation.
The UMA server will also expect to be given a port to listen to for client requests. That port needs to be different for every job to avoid several jobs contacting the same server if running simultaneously on the same cluster node. A suggestion to do that based on the Slurm job id is shown below. The UMA_PORT environment variable is of course also needed for the client script to make requests.

```
[... your slurm job script with environment etc ...]

export UMA_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# --- Start FAIRChem server in background ---
# CPU mode (recommended): model loaded once, workers share memory
python3 /path/to/UMA_server.py --device cpu --model /path/to/uma-s-1p2.pt --workers ${SLURM_NPROCS} --ncores ${SLURM_NPROCS} --port "$UMA_PORT" > fairchem_server.log 2>&1 &

# GPU mode alternative:
# python3 /path/to/UMA_server.py --device cuda --model /path/to/uma-s-1p2.pt --port "$UMA_PORT" > fairchem_server.log 2>&1 &

SERVER_PID=$!

# --- Ensure server is killed on exit ---
cleanup() {
    echo "Killing FAIRChem server (PID $SERVER_PID)"
    kill $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

# Give server a moment to start up
sleep 12

# Run ORCA calculation
$ORCAROOT/orca ${MY_INPUT_FILE} > ${MY_OUTPUT_FILE}

[... rest of slurm job script ...]
``` 

Depending on your machine and how slow/fast it is to start the server (mainly time taken by python imports and model loading), adjust the "sleep" command to delay the start of the ORCA calculation accordingly. In CPU mode with a local checkpoint, startup is typically fast since the model is loaded only once. Should the calculation start before the server is fully ready, the retry logic on the client side should prevent the calculation from crashing immediately.
