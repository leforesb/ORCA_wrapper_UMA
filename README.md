# ORCA_wrapper_UMA

In this repository are wrapper scripts to interface Fairchem's Universal Model for Atoms (UMA) with ORCA 6 through the "ExtOpt" option, taking advantage of ORCA's algorithms for geometry optimizations, transition state search and conformational sampling (GOAT). For now, the script is setup to do the UMA calculation on CPU, but could be easily edited to work an GPU devices. The UMA model used at the time of writing is "uma-s-1".

Implicit solvation is implemented as an additive corrective term for both energy and gradients, calculated using the Grimme group's ALPB solvation scheme through [xtb](https://github.com/grimme-lab/xtb).

The two scripts work together as server and client. The server part needs to be started before the subsequent ORCA calculation. The server takes care of the heavy import steps of torch and fairchem once, before the start of the ORCA calculation, with all subsequent UMA calculations being handled through requests by the client side. The server will spawn a number of workers corresponding to the number of cores requested in the ORCA input file through the explicit %pal block, for instance:
```
%pal
  nprocs 8
end
```
The presence of these workers is mainly to avoid thread fights or the server getting clogged occasionally when receiving multiple requests simultaneously (for instance during a numfreq calculation). In case that still happens, and a request from the client side gets lost, the client script features a "retry logic" to submit a request again a few seconds later and therefore prevents the ORCA calculation from failing there.

Parallelization is taken care of both at the UMA level through "torch.set_num_threads(ncores)" and for xtb by setting the OMP_NUM_THREADS variable as instructed by ORCA when the script is called. Note that while the UMA calculation is done on the server side, the client script handles the xtb calculations.
In this way, if 16 cores are requested, say, during optimization, only one of the workers will do the UMA calculation but with Torch parallelization. When switching to a numfreq calculation, all 16 workers will work in parallel with 1 torch thread each.


## Requirements
- A working environment (conda or other) for both ORCA 6 (i.e. OpenMPI) and Fairchem's UMA model, see requirements in the [fairchem](https://github.com/facebookresearch/fairchem) repository. The script uses the implementation of the UMA model as a calculator in ASE.
- An account on HuggingFace with access to the UMA model through an access token. You will be required to log in on your local machine to access it. Refer to the documentation on the [HuggingFace page](https://huggingface.co/facebook/UMA).
- ORCA 6.1.0. The script as it is will work with ORCA 6.0.1, as long as you do not try to do numerical frequencies in parallel with it, as ORCA fails for not being able to read a valid ".hostnames" file. This was corrected in 6.1.0, see Forum post [here](https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=13416).
- [xtb](https://github.com/grimme-lab/xtb) for the implicit solvation component.

## Example ORCA input
ORCA will need to call the client script through the ExtOpt option and the ProgExt keyword.
If implicit solvation is required, it can be requested as an argument in the same way as is implemented in xtb, through the Ext_Params keyword with the "--solvent SOLVENT" flag. This will trigger two xtb calculations: one in vacuum and one in the requested solvent with the ALPB solvation scheme. The corrective solvation terms for the potential energy and for gradients are calculated as the difference of the correponding terms in the two calculations, and are added to the UMA energy and gradient outputs. If not desired, "--solvent none" will entirely skip the xtb calculations and you will be left with the pure UMA energy and gradient. Here is a simple ORCA input calling the client script:
```
! ExtOpt Opt

%method
      ProgExt "/path/to/your/wrapper/scripts/directory/UMA_client.py"
      Ext_Params "--solvent thf"  # "--solvent none" skips the xtb calculations
end

%pal
        nprocs 16
end

*xyzfile 0 1 input.xyz

```

## Example Slurm script excerpt
The server script needs to be launched before the start of the ORCA calculation. In its current implementation, the server script needs to be copied and executed in the calculation's scratch directory.
The UMA server will also expect to be given a port to listen to for client requests. That port needs to be different for every job to avoid several jobs contacting the same server if running simultaneously on the same cluster node. A suggestion to do that based on the Slurm job id is shown below. The UMA_PORT environment variable is of course also needed for the client script to make requests.

```
[... your slurm job script with environment etc ...]

export UMA_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# --- Copy FAIRChem server script ---
cp "/path/to/your/wrapper/scripts/directory/UMA_server.py" "${SCRATCH_DIR}/"

# --- Start FAIRChem server in background ---
python3 UMA_server.py --workers CORES --ncores=1 --port "$UMA_PORT" > fairchem_server.log 2>&1 &
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

Depending on your machine and how slow/fast it is to start the server (mainly time taken by python imports), adjust the "sleep" command to delay the start of the ORCA calculation accordingly. Should the calculation start before the server is fully ready, the retry logic on the client side should prevent the calculation from crashing immediately.

