# Complete GPU classifier inference

**Goal:** manage a multi-kernel application's shapes, buffer lifetimes, numerical
contracts, and diagnostic checks.

Read [gpu-classifier.rs](../gpu/oxide/src/bin/gpu-classifier.rs), then compare it
with the tiny, hand-calculated [CPU capstone](08-capstone.md). Run in WSL Bash:

```bash
bash scripts/run_gpu.sh oxide gpu-classifier 17 2
bash scripts/run_gpu.sh oxide gpu-classifier 1024 100
bash scripts/verify_deep_core.sh
```

Arguments are batch size 1..=1024 and repetitions 1..=10000. Defaults are 17 and 1.
Inputs and weights are deterministic lesson data; no training or accuracy claim
is involved. This is a complete forward inference pipeline.

## Five ordered launches

Let B be the selected batch:

| Tensor | Shape | Producer | Consumer |
| --- | --- | --- | --- |
| Input X | B x 23 | Host upload | Dense layer 1 |
| W1, bias1 | 23 x 35, 35 | Host upload | First dense/bias kernels |
| First raw product | B x 35 | Dense layer 1 | Bias + ReLU |
| Hidden H | B x 35 | Bias + ReLU | Dense layer 2 |
| W2, bias2 | 35 x 19, 19 | Host upload | Second dense/bias kernels |
| Second raw product | B x 19 | Dense layer 2 | Bias kernel |
| Logits | B x 19 | Bias kernel | Row softmax |
| Probabilities | B x 19 | Row softmax | Host validation |

```text
H = ReLU(X * W1 + bias1)
logits = H * W2 + bias2
probabilities = stable_row_softmax(logits)
```

Weights, inputs, and all intermediate buffers stay allocated on the GPU across
repetitions. The five launches use one stream, so consumers follow producers
without an intermediate host wait or readback. Each output has a separate
allocation, allowing safe shared input borrows and mutable output borrows.
Drop and synchronization behavior still belongs to the runtime's ownership
contract; do not assume a Rust function returning means an asynchronous kernel
has completed on the device.

## Deliberately awkward dimensions

The dense kernel uses 16x16 blocks and two shared 16x16 tiles. Dimensions 23,
35, and 19 exercise both K-tile tails and output edges. Invalid loads contribute
zero. Every thread reaches the tile-ready barrier and the barrier before the
next overwrite, including threads whose final output lies outside the matrix.

`RowWidth` binds the real output stride. No padded global storage is used in
this oxide implementation. The host launches ceil(N/16) by ceil(B/16) blocks;
the typed 2D index and row guard restrict the final store to the real matrix.

Bias kernels use `flat_index % columns` to select the row-broadcast bias. They
apply ReLU only after the first layer. Applying ReLU to logits would change the
model's computation.

## Nineteen classes in a 32-thread softmax block

One block owns one sample's row. Nineteen lanes load logits; thirteen inactive
class lanes contribute negative infinity to the maximum reduction and zero
to the exponential-sum reduction. All 32 threads participate in every barrier.
The resulting denominator includes only the real classes.

Padded zero logits would incorrectly introduce thirteen extra exponentials.
Simply refusing to store their outputs would not fix that denominator. Real
lanes compute `exp(logit - row_max)` and divide by the real sum. Separate shared
arrays hold maximum and sum reductions, so a lane cannot overwrite the maximum
while another lane still reads it. All generated logits are finite.

The output's runtime row width is 19; inactive lanes cannot obtain a destination.
The kernel is specialized by its 32-thread geometry to at most 32 classes. The
host supplies the fixed 19-class lesson shape. Supporting arbitrary class counts
requires a new cooperative mapping and corresponding launch/shape validation.

## Validate intermediate tensors

The independent CPU pipeline uses the root f64-accumulating matrix reference,
row-broadcast biases, ReLU, and stable softmax. After timing, the GPU program
copies back hidden activations, logits, and probabilities and compares **every
element of all three tensors**. It also checks finite probabilities, range,
and row sums. A normalized row alone cannot establish correct classification.

The diagnostic readbacks are intentionally outside the measured interval.
An inference-only application can return only probabilities; retaining optional
layer checks is valuable while developing fused kernels. Repetitions overwrite
all outputs and reuse allocations. Warmup completes before recording the start
event, and the stop event is synchronized before querying its elapsed time.
The interval covers five launches per repetition and may include submission
gaps; it excludes allocation, input transfers, and diagnostic readbacks.

## Checkpoints and deeper experiments

1. With batch 17, the first product launches 3x2 blocks and the second 2x2 blocks.
   Explain why edge threads cannot return early before the dense barriers.
2. Why must the second bias kernel omit ReLU? Negative logits are valid scores
   and belong in the stable softmax computation.
3. What does checking hidden/logits detect that checking row sums misses?
   Incorrect matrix strides, biases, and activations can still normalize.
4. Fuse bias/ReLU into the first dense epilogue and bias into the second. Remove
   the two raw-product allocations. Preserve the original five-kernel version
   and compare the full tensors before measuring the three-kernel pipeline.
5. Implement a cuTile counterpart. Distinguish storage padding from real shapes,
   and mask padded classes out of both maximum and denominator reductions.
6. Add a CPU cross-entropy loss and derive gradients before porting a training
   step. Use finite-difference checks on tiny inputs to validate derivatives;
   inference correctness does not establish backward-pass correctness.

The [continuing roadmap](ROADMAP.md) takes this baseline toward graphs, profiling,
attention, mixed precision, and reusable Rust GPU APIs.
