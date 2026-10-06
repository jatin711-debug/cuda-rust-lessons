# Verification

Checked on **2026-10-06** after expanding the learning project.

## CPU and source checks

Windows Rust: `rustc 1.99.0 (b940084d7 2026-09-28)`.

| Check | Result |
| --- | --- |
| `cargo test` | All 12 integration tests passed |
| `cargo clippy --all-targets -- -D warnings` | Passed for the dependency-free root package |
| Root and Tile `cargo fmt --all -- --check` | Passed |
| Oxide source `rustfmt --check --edition 2024` | Passed using Windows rustfmt; no Windows GPU build implied |
| CPU walkthrough | Completed with normalized softmax rows and known matrix outputs |
| CPU labs: launch, SAXPY, reduction, transpose, pipeline | All five lab modes completed using lesson sample commands |
| CPU classifier | Known hidden activations/logits matched; probability checks passed; classes 0 and 1 |
| Invalid CPU lab arguments | Rejected zero length, non-power-of-two tree width, oversized matrix, nonnumeric length, unknown command |
| Local Markdown links | All links resolved across the 18 project documents |
| Bash syntax for the GPU runners | Passed |

The tests cover hand-calculated answers, softmax properties, shape rejection,
matrix padding/cropping, launch coverage at boundaries, reduction tails,
transpose inversion, exact exclusive-scan answers/differences and overflow,
classifier tensor shapes/probabilities and hand-calculated layer outputs, and
validation errors. Comparison arithmetic uses f64;
the regression test rejects a mismatch that previously hid behind f32 overflow.

## Actual GPU execution

Environment: WSL Ubuntu, NVIDIA GeForce RTX 3050 Laptop GPU, compute capability
8.6 (`sm_86`), driver 616.92, prepared CUDA 13.3 tools.

- Tile used Linux stable Rust 1.95.0 and source revision
  `cc720f182f38bf46527753caa340e78d6d5fa3cc`.
- Oxide used `nightly-2026-08-28` (rustc 1.100.0-nightly) and the matching
  cargo-oxide/source revision `55fa80345539beebb59a657c2c4498cb591bf224`.

`bash scripts/run_gpu.sh all` built and ran **all 20 GPU programs** after adding
the deeper track (seven Tile programs and thirteen oxide programs). Each
program copied actual device results back and compared every output against its
CPU reference. Tile examples also checked explicitly stored padded outputs.

| Lesson | cuda-oxide / SIMT | cuTile Rust / Tile |
| --- | --- | --- |
| Vector addition, 1003 values | Passed, max error 0 | Passed, max error 0, including 21 padding values |
| SAXPY, 1003 values | Passed, max error 0 | Passed, max error 0, including padding |
| Block reduction, 8 partials | Passed, max error 0, final sum -27.5 | Passed, max error 0, final sum -27.5 |
| Stable softmax, 4x64 | Passed, max error about 3.73e-9; probability checks passed | Passed, max error about 3.73e-9; probability checks passed |
| Transpose, 35x19 to 19x35 | Passed, max error 0 | Passed, max error 0, including padded matrix |
| Matrix multiplication, 35x23 times 23x19 | Passed, max error 0 | Passed, max error 0, including padded matrix |
| Two-stage pipeline, 1003 values | Passed, max error 0 | Passed, max error 0, including padding |

`bash scripts/verify_gpu_edges.sh all` also passed **20 GPU boundary runs**:
SAXPY and pipeline, both libraries, lengths **1, 127, 128, 129, 257**, and two
measured repetitions each. Every comparison had maximum error 0. The local run
log is in `.tools/gpu-edge-verification.log` (generated tooling/logs are ignored).

## Scope of these results

These are correctness checks on the stated inputs and hardware, not a benchmark
of GPU peak performance or a proof for arbitrary shapes/inputs. Reduction's
final small sum in the original block-reduction lesson runs on the CPU. The
advanced device-reduction example performs all reduction passes on the GPU.
The complete GPU classifier performs five ordered forward-pass kernels and
checks hidden activations, logits, and probabilities after timing. Its weights
are deterministic lesson data, with no training or model-accuracy claim.

SAXPY/pipeline timings use host clocks after warmup. SIMT excludes allocation
and copies from its timed interval; Tile includes them. The numbers have
different scopes and do not establish a relative library speedup.

## Advanced track verification

The three additional oxide programs were compiled and run on the same WSL/GPU
environment. `bash scripts/verify_advanced.sh` passed **19 additional GPU cases**:

| Program | Verified inputs | Result |
| --- | --- | --- |
| `device-reduction` | Lengths 1, 127, 128, 129, 16384, 16385, 1048576 | Final GPU scalar matched the CPU reference; max error 0 |
| `tiled-transpose` | Shapes 1x1, 1x33, 33x1, 31x33, 32x32, 33x31, 35x19, 1024x1024 | Both strided baseline and padded shared-memory variant matched every element exactly |
| `saxpy-events` | Lengths 1, 129, 257, 1048576 | Both split and fused variants matched every CPU reference output |

Each case used two timed repetitions after warmup. The local log is
`.tools/advanced-verification.log`. CUDA timing events were enabled, recorded,
synchronized, and queried successfully. The event intervals exclude allocation
and transfers but may include inter-launch gaps; these checks do not establish
peak bandwidth or a universal speedup. The transpose variants use matching
32x8 geometry, and SAXPY variants use matching 128-thread blocks.

## Warp, scan, and application verification

`bash scripts/verify_deep_core.sh` compiled and ran the three new oxide binaries
on the same WSL/GPU environment. All **26 additional GPU cases** passed:

| Program | Verified inputs | Result |
| --- | --- | --- |
| `warp-reduction` | Lengths 1, 31, 32, 33, 127, 128, 129, 16384, 16385, 1048576 | Final GPU scalar matched the CPU reference; maximum error 0 |
| `prefix-scan` | The same ten lengths | Every u32 exclusive prefix matched exactly, including three-level hierarchies |
| `gpu-classifier` | Batches 1, 15, 16, 17, 33, 1024; fixed features/hidden/classes 23/35/19 | Every hidden/logit/probability value matched within tolerance; hidden/logit error 0, probability error at most about 2.98e-8; row properties passed |

Each case used two measured repetitions after a completed warmup. Allocations
and transfers were outside the enabled CUDA event interval. The classifier's
three diagnostic tensor readbacks occurred after timing. The scan uses a
read-completion barrier before each shared-memory overwrite and a write-completion
barrier afterward; its offset launches propagate from the highest level downward.
The warp reduction launches complete physical warps with neutral input tails.
Length 1 requires no warp-reduction kernel.

The local log is `.tools/deep-core-verification.log`. These runs establish the
stated comparisons on the stated cases; no sanitizer, profiler, throughput
speedup, training, or multi-GPU verification is implied. Later projects in
[the roadmap](lessons/ROADMAP.md) are future specifications rather than implemented
or verified examples.
