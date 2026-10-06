# Reduction: many contributions become a few answers

**Goal:** understand shared memory, neutral padding, stage barriers, and why a
global sum needs coordination beyond one ordinary block.

```powershell
cargo run --example lab -- reduction 1003 128
```

In WSL Bash:

```bash
bash scripts/run_gpu.sh all block-reduction
```

Read [SIMT reduction](../gpu/oxide/src/bin/block-reduction.rs) and
[Tile reduction](../gpu/tile/src/bin/block-reduction.rs). Both compute eight
partial sums from 1003 inputs. The **CPU** finishes the final sum, `-27.5`.
This is deliberately not advertised as a complete device-only reduction.

## Why independent `output += value` is wrong

Two workers can read the same old output, each add its own contribution, and
both write a replacement. One update can be lost. An atomic operation could
coordinate a shared destination, but atomics are not needed for this lesson:
each block writes a separate output element instead.

The last block contains 107 real values and 21 excess workers. Zero is the
neutral element for addition: adding padded zero values preserves the answer.
Using zero as padding for a maximum reduction would fail on all-negative inputs;
the neutral padding value depends on the operation.

## Work through the tree before reading the GPU code

For eight values, the first stage has four writers:

```text
[1,2,3,4,5,6,7,8]
  writers 0..3 add cells 4..7 -> prefix [6,8,10,12]
  barrier
  writers 0..1 add cells 2..3 -> prefix [16,20]
  barrier
  writer 0 adds cell 1       -> prefix [36]
  barrier
```

At each stage, writers have distinct cells and read an upper region that nobody
changes during that stage. A barrier completes those writes before the next
stage reads them. Inactive writers still reach every barrier. The CPU lab prints
the equivalent stages for the first input chunk.

In the SIMT kernel, each of 128 lanes initializes one shared cell, including
zero for an excess input. Seven reduction stages leave the block sum in cell 0.
Only lane 0 writes it to `output[block]`.

That store uses a small documented `unsafe` operation: a normal per-thread
index would be `block*128+lane`, while our output has only one element per block.
The contract is unique block slots, one writer per slot, and checked bounds.
The launcher checks geometry; the programmer still supplies this store's proof.
See cuda-oxide's [shared-memory explanation](https://nvidia.github.io/cuda-rust/advanced/shared-memory-and-synchronization.html)
for the memory and synchronization contracts.

## Tile expresses the same cooperation as a reduction

The host reshapes the padded input into `[8,128]` and partitions output as
`[1,1]`. Program r loads input tile `[r,0]` with shape `[1,128]`, reduces axis 1
to `[1]`, reshapes to `[1,1]`, and stores its owned result. The compiler supplies
the physical cooperation. `WIDTH` is explicitly specialized to 128.

## Why finish on the CPU?

A block barrier cannot make all blocks in an ordinary grid reach a global
checkpoint. For a device-only result, a straightforward extension is a second
kernel reading the partial sums after the first kernel completes in the same
stream. It may need further passes for very large inputs. This avoids assuming
which block runs last or spinning until unscheduled blocks produce values.

Reduction order also changes floating-point rounding. Summing in f64 for the
CPU reference gives a useful independent answer; it does not make the GPU tree
bitwise equivalent for arbitrary data.

The [advanced track](09-advanced-gpu.md) implements the complete multi-pass GPU
version, with reusable pass allocations and only a final scalar readback.

**Checkpoint:** identify every writer, every barrier, and the tail padding.
Continue to [softmax](02-stable-softmax.md), whose maximum and sum reductions now
build directly on this tree.
