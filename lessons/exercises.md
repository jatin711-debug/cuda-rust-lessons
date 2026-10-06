# Practice, predictions, and answer hints

Run the CPU labs first. Then execute the GPU version in WSL using the runner.
Make a prediction before each experiment and preserve the complete output check.

| Exercise | Run or edit | Expected checkpoint |
| --- | --- | --- |
| Count excess workers | `cargo run --example lab -- launch 257 256` | 2 blocks, 512 workers, 255 excess |
| Find a boundary thread | `cargo run --example lab -- launch 129 128` | Block 1, lane 0 owns index 128; lane 1 has none |
| Fuse two operations | `cargo run --example lab -- saxpy 257` | Ideal array traffic 20*N becomes 12*N |
| Trace a reduction tail | `cargo run --example lab -- reduction 129 128` | Two partials; last chunk has one real value |
| Track transpose strides | `cargo run --example lab -- transpose 3 5` | Input (1,3) at address 8 becomes output (3,1) at address 10 |
| Check another coordinate | Transpose input (0,3) in [3,5] | Input address 3 becomes output address 9 |
| Remove a CPU round trip | `cargo run --example lab -- pipeline 257` | Device intermediate is consumed by stage 2 |
| Compose a classifier | `cargo run --example classifier` | Hidden [2,0,0.5,1]; classes 0 and 1 |

## GPU experiments with no source edits

```bash
bash scripts/run_gpu.sh all saxpy 1 1
bash scripts/run_gpu.sh all saxpy 127 2
bash scripts/run_gpu.sh all saxpy 128 2
bash scripts/run_gpu.sh all saxpy 129 2
bash scripts/run_gpu.sh all pipeline 257 2
```

SAXPY and pipeline accept length/repetition arguments. Other binaries retain
their lesson's fixed data shape. The runner rejects extra arguments for those
lessons instead of pretending that a shape changed.

## Code exercises

1. Change the SAXPY scalar in its CPU reference and GPU launch together. Predict
   the first result, then compare the entire output.
2. Trace the multi-pass GPU reduction in [the advanced track](09-advanced-gpu.md),
   then add a warp-shuffle variant with correct lane participation masks.
3. Compare the supplied padded shared-memory SIMT transpose with an unpadded
   variant. Keep geometry and timing scope identical and inspect bank accesses.
4. Fuse the two-stage pipeline. Algebra gives `x + 1.5*y`; arbitrary f32 inputs
   may round differently than the original intermediate-buffer computation.
5. Port the classifier using the checkpoints in [lesson 08](08-capstone.md).

## Explain these without running code

- Why does a Tile width not specify a physical thread count?
- Why can zero pad a sum, yet change a softmax denominator?
- Why is a block barrier insufficient for communication across an ordinary grid?
- Why can safe ownership coexist with a mathematically incorrect kernel?
- Why would transferring an intermediate back to the CPU slow a pipeline?

Answers are in [foundations](00-foundations.md), [reduction](05-parallel-reduction.md),
[softmax](02-stable-softmax.md), and [streams](07-streams-and-timing.md).
