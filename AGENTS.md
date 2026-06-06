# UCCL Development and Build Guide (Isambard-Specific)

This file contains instructions for AI agents and developers working on the `uccl` repository (specifically the `cxi-ep` transport / `ep` subdirectory) on Isambard (NVIDIA Grace Hopper node architecture).

## Environment Setup

### 1. Dev Container Mounting Rule
When spawning the container using `run-dev-container.sh`, always set `WORK` to the repository root directory `~/src/uccl` rather than `~/src/uccl/ep`.
* **Why:** The `ep` compilation depends on top-level headers (e.g., `../include/util/gpu_rt.h`). If `WORK` is set to `ep`, only the `ep` folder is bind-mounted inside the container, hiding parent headers.

### 2. Virtual Environment Configuration
The virtual environment in `~/src/uccl/ep/.venv` is configured to inherit the pre-compiled container packages (like `torch` and vLLM dependencies) instead of reinstalling them on top of Python.
* In `~/src/uccl/ep/.venv/pyvenv.cfg`, ensure the following is set:
  ```ini
  include-system-site-packages = true
  ```
* This ensures that activating `.venv` exposes the system `torch` (e.g., `2.11.0+cu129`) instantly.

---

## Build and Compilation Workflow

### 1. Compiling `uccl_ep` for Grace Hopper (SM90a)
The base vLLM image exports target architectures like `8.0 8.7 8.9 9.0 10.0 12.0`. However, the codebase contains Hopper-specific inline PTX assembly (like TMA and `cp.async.bulk`) in `internode.cu` and `dispatch.cu` that is incompatible with older targets (like SM80/SM89), causing `ptxas` to crash.

To build, force the compilation strictly to the Grace Hopper architecture (`9.0a`) inside the `ep` subdirectory:
```bash
# Inside the container, activated venv:
cd ep
TORCH_CUDA_ARCH_LIST="9.0a" USE_CXI=1 pip install -e . --no-build-isolation
```

### 2. Namespace Integration
The python wrapper packages import elements via `import uccl.ep`. To make this resolve correctly in the development virtual environment:
1. Copy the compiled shared library binary from the build output to the main package directory:
   ```bash
   cp ~/src/uccl/ep/ep.abi3.so ~/src/uccl/uccl/
   ```
2. Install the top-level repository in editable mode inside the virtual environment:
   ```bash
   cd ~/src/uccl
   pip install -e . --no-build-isolation
   ```

---

## Correctness & Verification Testing

### Distributed Execution Rule
Never use the raw system `torchrun` script (usually located at `/usr/local/bin/torchrun`), as it starts PyTorch distributed children using the system Python interpreter instead of the virtual environment interpreter, triggering `ModuleNotFoundError: No module named 'uccl'`.

Instead, always run the distributed tests programmatically using the virtual environment's Python launcher:
```bash
# Verify top-level import
python3 -c "import torch; import uccl.ep; print(uccl.ep.__file__)"

# Run Intranode correctness validation on 4 GPUs
python3 -m torch.distributed.run --nproc_per_node=4 bench/test_intranode.py --num-processes 4
```
