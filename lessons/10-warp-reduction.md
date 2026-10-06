# Warp cooperation and grid-stride reduction

**Goal:** separate logical input coverage from physical lane participation, then
combine register, warp, block, and grid-level work.

Read [warp-reduction.rs](../gpu/oxide/src/bin/warp-reduction.rs) alongside the
[shared-memory reduction](09-advanced-gpu.md). Run in WSL Bash:

```bash
bash scripts/run_gpu.sh oxide warp-reduction 129 2
bash scripts/run_gpu.sh oxide warp-reduction 1048576 100
```

## Four levels of cooperation

Each block launches exactly 128 threads: four complete 32-lane warps. A thread
first accumulates several inputs into a private register. Each warp then reduces
its 32 register values with `warp::reduce_sum_f32`. Lane zero writes the warp's
sum into one of four shared cells. A block barrier makes those four values
available to warp zero, which performs one more warp reduction. Its lane zero
writes the block's partial sum.

```text
thread registers -> 32-lane warp sums -> 4 shared values -> block sum
block sums       -> next kernel pass -> final GPU scalar
```

Only 16 bytes of explicitly declared shared storage and one block barrier are
needed per pass. Compare that mechanism with the 128-value shared tree; the
smaller storage/barrier count alone does not establish a speedup. The pinned
[warp helper implementation](https://github.com/NVIDIA/cuda-rust/blob/55fa80345539beebb59a657c2c4498cb591bf224/crates/cuda-device/src/warp.rs)
uses full-warp shuffle operations. Read its participation contract before using
it in another kernel.

## An input tail is not a partial physical warp

At length 129, most threads in the second block have no input. They still exist,
contribute zero, and participate in their warp reduction and block barrier.

Do not move the first warp reduction inside `if index < input.len()`. That would
leave named lanes absent from a collective. The second reduction is inside
`if warp_id == 0`: this is uniform within each warp, and all 32 lanes of warp zero
execute it. Only four lanes contribute shared values; the other 28 supply zero.
No block barrier occurs inside that conditional.

This program always launches whole warps. If you introduce a physically partial
warp or a divergent subgroup, you need a collective whose mask and shuffle
sources match the participating lanes. Zero-padding data cannot repair an
incorrect participation mask.

## Grid-stride coverage

The host caps each pass at 128 blocks. A thread visits:

```text
start  = block * 128 + local_thread
stride = grid_blocks * 128
indices = start, start + stride, start + 2*stride, ... below input.len()
```

The initial starts cover distinct residues modulo `stride`; adding that stride
preserves each thread's residue. Therefore every input has exactly one reader
in this pass. Neighbouring lanes still read neighbouring elements in each loop
iteration. The cap deliberately demonstrates grid-stride work; it is not tuned
to this GPU's occupancy.

For 1,048,576 inputs the pass sizes are `[1048576,128,1]`. Each thread in the
first pass visits 64 inputs. For 16,385 inputs they are `[16385,128,1]`, while the
original one-value-per-thread reduction needs `[16385,129,2,1]`. These programs
change both the block-level reduction and work assignment, so their event
intervals do not isolate the effect of shuffles.

## Ownership and measurement

Every block has one output slot. The final unchecked store is justified by a
unique block ID, a bounds guard, and a single lane-zero writer. Shared cells
have one writer each, and the barrier precedes their inter-warp reads. Separate
pass buffers avoid overwriting inputs that other blocks still need.

All passes use one stream. Buffers/contracts are prepared before warmup and
timing; repetitions reuse them. Only the scalar is copied back. Length 1 is
already a scalar and launches no reduction kernel, so its event interval
measures no reduction work. Arbitrary floating-point inputs may differ from
the CPU because additions are grouped differently; the supplied binary
fractions are deliberately easy to verify.

## Checkpoints and next experiment

1. Why do 31 logical values still use 128 physical threads? The launch contract
   is fixed; unused threads contribute zero and preserve collective participation.
2. Why is a block barrier still needed? Warp-local shuffles do not order the
   shared-memory handoff between four different warps.
3. Why can one thread read several inputs but only one thread write each partial?
   Grid-stride reads partition the input; lane zero owns the block output slot.
4. Build a shuffle variant with the original pass geometry before comparing
   against the shared tree. Then vary grid size separately. Keep timing scope,
   input, warmup, and full CPU validation identical.

Continue with [hierarchical scan](11-hierarchical-scan.md), which preserves all
prefixes rather than discarding them into one scalar.
