# SAXPY: turn a formula into a kernel

**Goal:** add a scalar parameter, handle arbitrary positive vector lengths, and
understand how fusion reduces intermediate array traffic.

SAXPY is the operation `z[i] = alpha*x[i] + y[i]` on single-precision vectors.
We use `alpha=2`; the first three default answers are `-11.5, -10.5, -9.5`.

```powershell
cargo run --example lab -- saxpy 257
```

After Linux setup, run in WSL Bash:

```bash
bash scripts/run_gpu.sh all saxpy 257 2
```

This runs both GPU implementations with 257 real elements and two measured
repetitions. The fixed block/tile width is 128; each binary verifies every result.
Read [SIMT SAXPY](../gpu/oxide/src/bin/saxpy.rs) and
[Tile SAXPY](../gpu/tile/src/bin/saxpy.rs) next to the
[CPU reference](../src/learning.rs).

## Trace one answer through the host

1. `pointwise_inputs` constructs bounded, nonconstant CPU arrays.
2. `saxpy` computes a reference using f64 intermediate arithmetic.
3. The GPU inputs and output are allocated or described as lazy operations.
4. The generated launcher receives input ownership/access, output access, and `alpha`.
5. Readback completes before `check_close` inspects the actual answer.

The finite checks and f64 tolerance arithmetic in `check_close` prevent NaNs or
comparison overflow from hiding a bad output. Tolerances allow legitimate
floating-point differences; they do not compensate for missing writes or indices.

## Trace one answer inside each kernel

SIMT thread `(block=2, lane=1)` owns `2*128+1=257`. For length 257, it is an
excess thread. The checked output access fails, and no input is read by that worker.
For length 258, the same worker computes one legitimate answer.

Tile program 2 owns positions `256..384`. The host zero-pads to complete
128-element tiles. `alpha.broadcast(shape![WIDTH])` represents the scalar over
the tile's shape; loads and arithmetic operate on that whole fragment. The last
tile's padded outputs are also checked against zero.

## Why combine multiply and add?

Consider splitting the formula into `temporary=alpha*x`, then `z=temporary+y`.
Counting ideal f32 array traffic for N elements:

| Implementation | Array reads | Array writes | Bytes |
| --- | --- | --- | --- |
| Two kernels with a global temporary | x, temporary, y | temporary, z | 20*N |
| One SAXPY kernel | x, y | z | 12*N |

The fused implementation also needs one launch instead of two. These counts
exclude cache behavior, allocations, scalar arguments, padding, and transfers.
They explain a potential advantage; they are not a measured speedup.

**Kernel fusion** here means combining operations without a global intermediate.
It is separate from a hardware **fused multiply-add** instruction, which can
round multiplication and addition together. The chosen binary-fraction data
keeps arithmetic easy to inspect; other inputs may show rounding differences.

## Experiments

Predict the last valid thread/tile for lengths 1, 127, 128, 129, and 1003. Run
those lengths through the runner. Change `alpha` in both the GPU launch and CPU
reference, retaining the full comparison. Try larger lengths before interpreting
timings, then read [streams and timing](07-streams-and-timing.md) for their scope.

**Checkpoint:** name the temporary array fusion removes, and explain why Tile's
padding and SIMT's bounds check preserve the same real answers.
Continue to [reduction](05-parallel-reduction.md).
