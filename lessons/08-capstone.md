# Capstone: a tiny two-layer classifier

**Goal:** connect the earlier operations into an application and design its GPU
memory/ownership flow before writing additional kernels.

```powershell
cargo run --example classifier
```

The supplied [classifier](../examples/classifier.rs) is a complete **CPU**
walkthrough with hand-selected weights. It checks known hidden activations and
logits, then validates stable softmax probabilities. It is not a trained model
and serves as an arithmetic reference. After tracing these values, study the
[complete GPU classifier](12-gpu-classifier.md), which uses larger, deliberately
uneven dimensions and validates each layer. Both use deterministic lesson
weights rather than a trained model.

## Follow the shapes and values

```text
X              [2,3]   two samples, three features each
W1             [3,2]   first dense layer
H = ReLU(X*W1 + bias1)
H              [2,2]   [2,0, 0.5,1]
W2             [2,3]   second dense layer
logits         [2,3]   [2,0.25,-2.25, 0.5,1.25,-0.25]
softmax rows   [2,3]   one probability vector per sample
selected class          0 for sample 0, 1 for sample 1
```

The matrix products reuse the root `matmul` reference; biases broadcast across
rows, ReLU clamps negative hidden values to zero, and softmax normalizes each
sample's scores. Output classes are zero-based indices of maximum probability.

## Connect the computation to the GPU

1. Build a correct baseline for a single matrix product, checking every output.
2. Add a pointwise bias/ReLU kernel. Use `column = flat_index % hidden_width`
   to select the bias for row-major output.
3. Retain hidden activations on the device and launch the second matrix product.
4. Add its bias, then row softmax, and copy back only the final probabilities.
5. Compare with the CPU pipeline and independently check row sums and ranges.

The current softmax lesson is specialized to 64 columns. The capstone has three
classes, so reusing it unchanged would violate its contract. Write a small
shape-aware kernel or an explicitly padded variant. Padded softmax logits must
exclude padded classes from the denominator; padding logits with zero is
generally incorrect. This is different from zero-padding a matrix product.

The supplied GPU application demonstrates this with 19 real classes and 32
physical lanes: inactive lanes contribute negative infinity to the maximum
and zero to the exponential sum. Read the implementation before trying a
three-class port of the tiny example. Its diagnostic hidden/logit readbacks
occur after timing; its five-kernel forward computation stays on the device.

For the tiled matrix products, retain each tensor's real dimensions alongside
its padded storage dimensions. Bias and activation kernels should handle the
real region deliberately. Pad the tiny matrices using the tested root helpers,
and make each launcher contract match the chosen storage layout.

## Completion checklist

- Every dimension and row stride is stated on the host.
- Every writable output element has one owner or explicit coordination.
- Producer and consumer launches have an established ordering.
- Padded data cannot change the real probabilities.
- Every actual output is compared with an independent reference.
- Timing states whether it includes allocations, transfers, and compilation.

After the baseline passes, try fusing bias/ReLU with the first product's output
stage. Keep the unfused version as a comparison. The tiny default sizes teach
correctness and ownership; scale the workload before investigating throughput.
Use [lesson 12](12-gpu-classifier.md) for the supplied GPU baseline and
[the continuing roadmap](ROADMAP.md) to choose a larger capstone.
