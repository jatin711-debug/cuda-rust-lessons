# Learn CUDA with Rust

Start with [the foundations](lessons/00-foundations.md) and the CPU walkthrough,
then read and run the GPU implementations side by side.
If you know Rust and CUDA basics, start with the
[advanced GPU track](lessons/09-advanced-gpu.md): device-only multi-pass reduction,
a padded shared-memory transpose, and split-versus-fused CUDA event timing.
The examples use **the same input data and mathematical operation** in both libraries.
Each GPU program copies its result back and checks **every element** against a CPU reference.

| Lesson | Problem solved | What cuda-oxide teaches | What cuTile Rust teaches |
| --- | --- | --- | --- |
| Easy | Add 1,003 pairs of numbers | Thread indices, `DisjointSlice`, checked launch geometry | Tensor ownership, partitioning, lazy execution |
| SAXPY | Scale and add vectors with configurable lengths | Scalar parameters, edge checks, fused pointwise work | Scalar broadcasting, complete tiles, specialization |
| Reduction | Sum chunks, then finish the small sum on the CPU | Shared-memory trees, neutral tail values, one writer per block | Axis reductions, explicit input/output shape mapping |
| Medium | Convert large logits into stable probabilities | Shared memory, maximum/sum reductions, barriers | Tile reductions, broadcasting, fused operations |
| Transpose | Rearrange an uneven rectangular matrix | Row strides, 2D output ownership, strided reads | Swapped tile coordinates and transposed fragments |
| Hard | Multiply uneven rectangular matrices with data reuse | 2D blocks, cooperative loads, two barriers per tile, edge conditions | Matrix tiles, `mma`, specialization, zero padding |
| Pipeline | Connect two kernels without an intermediate CPU round trip | Resident buffers, launch order, synchronized host timing | Operation ownership, device intermediates, end-to-end timing |

There are **17 runnable GPU programs**: seven paired SIMT/Tile problems and three
advanced oxide programs. The **ten learning chapters** include CPU labs and a
CPU classifier capstone. The capstone's GPU port remains a guided exercise.

These are learning kernels. The small matrix size makes it practical to inspect
all results; it does not measure the GPU's peak performance.

## Get the project

```bash
git clone https://github.com/jatin711-debug/cuda-rust-lessons.git
cd cuda-rust-lessons
```

The CPU commands work from any clone location. The GPU setup examples below use
this machine's `D:\CUDA_RUST` / `/mnt/d/CUDA_RUST` paths; adjust them to your clone.
Downloaded toolchains, generated GPU artifacts, and local run logs are excluded
from Git. Cargo lockfiles are included to preserve the pinned builds.

## Run something immediately on Windows

From `D:\CUDA_RUST`:

```powershell
cargo run --example walkthrough
cargo run --example lab -- help
cargo run --example lab -- launch 257 256
cargo run --example classifier
cargo test
```

This root package uses **ordinary CPU Rust and no GPU dependencies**. It shows the
inputs, answers, and work division. Passing these tests validates the CPU
references and padding logic; it does not establish that a GPU kernel compiled or ran.
GPU verification is recorded separately in [VERIFICATION.md](VERIFICATION.md).

## Follow this learning path

| Step | Explanation | Run on the CPU | GPU lesson name |
| --- | --- | --- | --- |
| 1 | [Host/device, Rust ownership, grids, warps, memory](lessons/00-foundations.md) | `cargo run --example lab -- launch 257 256` | CPU indexing model |
| 2 | [Vector addition](lessons/01-vector-add.md) | `cargo run --example walkthrough` | `easy-vector-add` |
| 3 | [SAXPY and fusion](lessons/04-saxpy-and-fusion.md) | `cargo run --example lab -- saxpy 257` | `saxpy` |
| 4 | [Parallel reduction](lessons/05-parallel-reduction.md) | `cargo run --example lab -- reduction 1003 128` | `block-reduction` |
| 5 | [Stable softmax](lessons/02-stable-softmax.md) | `cargo run --example walkthrough` | `medium-softmax` |
| 6 | [Transpose and memory access](lessons/06-transpose-and-memory.md) | `cargo run --example lab -- transpose 3 5` | `transpose` |
| 7 | [Tiled matrix multiplication](lessons/03-tiled-matmul.md) | `cargo run --example walkthrough` | `hard-matmul` |
| 8 | [Streams, ownership, and timing](lessons/07-streams-and-timing.md) | `cargo run --example lab -- pipeline 257` | `pipeline` |
| 9 | [Classifier capstone and GPU port exercise](lessons/08-capstone.md) | `cargo run --example classifier` | Guided extension |
| Advanced | [Multi-pass reduction, padded transpose, and event timing](lessons/09-advanced-gpu.md) | Trace indices and pass sizes in the chapter | `device-reduction`, `tiled-transpose`, `saxpy-events` (oxide) |

The original lesson filenames stay stable; follow the order above rather than
sorting filenames. Each paired core GPU lesson has a file with the same name in
[gpu/oxide/src/bin](gpu/oxide/src/bin) and [gpu/tile/src/bin](gpu/tile/src/bin).
Predict the indices/shapes before running, inspect the actual code, then answer
the chapter's checkpoint. [Practice exercises](lessons/exercises.md) supply
expected answers, and [troubleshooting](lessons/troubleshooting.md) covers setup
and common correctness failures.

Every GPU file contains both a `kernels` module and `main`:

1. The kernel describes the GPU's computation.
2. `main` creates a device/context and stream, supplies inputs, launches work,
   copies outputs back, and checks the answer.
3. The CPU functions in [src/lib.rs](src/lib.rs) and
   [src/learning.rs](src/learning.rs) define the reference computation.

The CPU `lab` accepts indexing, SAXPY, reduction, transpose, and pipeline
experiments. Its diagrams/printed stages illustrate work assignment; they do
not emulate GPU scheduling or predict GPU performance.

The kernels do not run on the CPU. In particular, cuda-oxide's device intrinsics
are compiler-recognized operations, not functions you can meaningfully call in a CPU walkthrough.

## GPU setup on this machine

The inspected machine has an RTX 3050 Laptop GPU (compute capability 8.6), a
current NVIDIA Windows driver, and WSL2 Ubuntu. The existing Windows toolkit is
CUDA 12.6 and the existing Ubuntu toolkit is CUDA 12.0.

These pinned library revisions need newer tools. GPU examples run **inside Linux**;
the Windows CPU walkthrough remains independent. No driver installation is needed
if your existing WSL GPU access works.

From PowerShell, enter Ubuntu and the project:

```powershell
wsl -d Ubuntu
```

Then run these in Ubuntu Bash:

```bash
cd /mnt/d/CUDA_RUST
nvidia-smi
python3 scripts/prepare_linux.py
source .tools/env.sh
```

The preparation helper downloads a selected CUDA 13.3 toolset and a libclang
parser. It verifies publisher-provided SHA-256 hashes. Bulk files go into
`~/.cache/cuda-rust-lessons/tools` to avoid slow WSL filesystem operations on the
Windows drive; `.tools/env.sh` sets paths for the current shell. It does not use
sudo, replace the installed toolkits, change shell startup files, or install a driver.
The first preparation downloads roughly 250 MB plus libclang and uses additional
space when unpacked. The Linux Rust/compiler caches use additional space.

If you already have a suitable Linux CUDA installation, you may use it instead:

```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
# bindgen also needs libclang and standard C headers available on your system.
```

For cuTile Rust, stable Rust 1.89 or newer is sufficient:

```bash
cd /mnt/d/CUDA_RUST/gpu/tile
cargo +stable run --bin easy-vector-add
cargo +stable run --bin medium-softmax
cargo +stable run --bin hard-matmul
```

For cuda-oxide, install the exact nightly and CLI matching the pinned revision:

```bash
rustup toolchain install nightly-2026-08-28 --profile minimal \
  --component rust-src,rustc-dev,llvm-tools,rustfmt
cargo +nightly-2026-08-28 install --locked \
  --git https://github.com/NVIDIA/cuda-rust.git \
  --rev 55fa80345539beebb59a657c2c4498cb591bf224 \
  --root "$HOME/.cache/cuda-rust-lessons/cli" cargo-oxide

cd /mnt/d/CUDA_RUST/gpu/oxide
cargo oxide run --bin easy-vector-add
cargo oxide run --bin medium-softmax
cargo oxide run --bin hard-matmul
```

The first oxide invocation builds its code-generation backend. The first launch
of a cuTile specialization compiles it through Tile IR. Neither first-run time
is a steady-state kernel benchmark.

After setup, the convenience runner executes a selection:

```bash
cd /mnt/d/CUDA_RUST
bash scripts/run_gpu.sh tile
bash scripts/run_gpu.sh oxide
bash scripts/run_gpu.sh all
bash scripts/run_gpu.sh all saxpy 257 2
bash scripts/run_gpu.sh all pipeline 257 2
bash scripts/run_gpu.sh tile transpose
bash scripts/run_gpu.sh --help
bash scripts/verify_gpu_edges.sh all
bash scripts/run_gpu.sh oxide device-reduction 1048576 100
bash scripts/run_gpu.sh oxide tiled-transpose 1024 1024 100
bash scripts/run_gpu.sh oxide saxpy-events 1048576 100
bash scripts/verify_advanced.sh
```

Each program returns an error on a failed comparison. The runner stops at the
first failed command rather than displaying a success message for unverified work.

`saxpy`, `pipeline`, `device-reduction`, and `saxpy-events` accept optional
length/repetition arguments. Length
is bounded to 1..=1048576 and repetitions to 1..=10000; other programs use the
fixed lesson data. Advanced `tiled-transpose` accepts rows/columns/repetitions,
up to 4096 per dimension and 4194304 total elements. The advanced programs are
oxide implementations; selecting `all` runs them on oxide alongside the paired
core examples. SIMT SAXPY/pipeline time repeated resident-buffer launches with a final
wait; Tile times repeated end-to-end chains including allocation and copies.
Both perform a completed warmup first. Their timing scopes differ; see
[lesson 07](lessons/07-streams-and-timing.md) before comparing numbers.
The advanced programs use enabled CUDA events; their intervals exclude
allocation and copies but include any gaps between submitted kernels.

## What to notice in the APIs

**cuda-oxide:** `&[f32]` expresses shared inputs; `DisjointSlice` restricts mutable
output access to typed thread indices. `prepare_*` validates the launch contract
and produces the token required by the generated safe launch method. Module
loading remains an explicit `unsafe` trust boundary. Shared memory in the medium
and hard examples also requires documented `unsafe` blocks and correct barriers.

**cuTile Rust:** mutable output tensors are partitioned before launch. The kernel
sees its own writable sub-tensor; read-only tensors can be shared. Tile operations
describe whole-array computations. Generated launchers return lazy operations
that preserve input ownership through execution and synchronization.

`-1` in a device tensor shape means a dynamic dimension. A const generic such as
`WIDTH`, `BM`, or `BK` affects specialization. A tile width is a data dimension,
not a physical GPU thread count.

Both libraries can still reject a shape, fail compilation, or report a runtime
GPU error. Safety does not establish mathematical correctness, so these examples
always compare the actual GPU output with a CPU answer.

## Sources and exact versions

- [NVIDIA blog](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [cuda-oxide source revision](https://github.com/NVIDIA/cuda-rust/tree/55fa80345539beebb59a657c2c4498cb591bf224)
- [cuTile Rust source revision](https://github.com/NVlabs/cutile-rs/tree/cc720f182f38bf46527753caa340e78d6d5fa3cc)
- [cuda-oxide safety model](https://nvidia.github.io/cuda-rust/gpu-safety/the-safety-model.html)
- [cuTile Rust documentation](https://nvlabs.github.io/cutile-rs/main/)
- [Fearless Concurrency on the GPU](https://arxiv.org/abs/2606.15991)

GPU manifests pin source revisions because these projects are evolving. The
cuda-oxide project uses its revision's shared host runtime, `cuda-core = 0.3.1`;
the cuTile project uses the runtime shipped with its own revision. Each GPU
folder is a separate Cargo project to keep those versions and toolchains apart.
