# Troubleshooting the learning projects

Start with the CPU path: `cargo test` and `cargo run --example walkthrough`.
It works without CUDA and gives a known mathematical baseline.

## GPU commands belong in the Linux project environment

Enter `wsl -d Ubuntu`, then run:

```bash
cd /mnt/d/CUDA_RUST
source .tools/env.sh
nvidia-smi
command -v cargo
command -v cargo-oxide
bash scripts/run_gpu.sh --help
```

The README describes preparation if `.tools/env.sh` or a tool is missing.
Preparation downloads tooling; it is not needed for the CPU lessons.

| Symptom | What to inspect |
| --- | --- |
| `cargo: command not found` | Linux Rust installation and shell PATH; Windows Rust is separate |
| `cargo-oxide` missing | Install the exact CLI revision/toolchain from the README |
| CUDA headers/toolkit missing | Source `.tools/env.sh`; inspect `CUDA_TOOLKIT_PATH` |
| bindgen cannot find libclang | Inspect `LIBCLANG_PATH` and the setup script's generated environment |
| Tile compiler fails | Confirm the prepared Tile IR tools and the pinned dependency revision |
| Kernel artifact/module load fails | Keep host binary and compiled device artifact from the same build |
| Wrong results only at edges | Inspect logical lengths, row strides, zero padding, and final bounds checks |
| Kernel hangs after an edit | Inspect divergent/early-return paths around every block barrier |
| A very slow first run | Distinguish compilation/specialization from measured warmed-up execution |
| Timing changes between runs | Compare equivalent scopes, warmup, input sizes, and repetition counts |

## Debug correctness in layers

Check the CPU result on a small hand-calculated input. Then inspect launch
geometry and flattened addresses, followed by output ownership. For a reduction
or tiled matrix product, trace shared writes, barriers, and reads separately.
Finally verify every output, including padding where the example stores it.

A nonzero exit means the runner stops at the failed command. It reports success
only after all selected programs finish successfully. `VERIFICATION.md` records
what was actually checked; a successful CPU test alone is not a GPU result.

For deeper tooling, cuda-oxide's [debugging guide](https://nvidia.github.io/cuda-rust/gpu-programming/error-handling-and-debugging.html)
introduces artifact inspection and NVIDIA diagnostic tools. Follow their Linux
and driver requirements when adding profiling or sanitizer checks; those checks
are separate from this repository's reference comparisons.
