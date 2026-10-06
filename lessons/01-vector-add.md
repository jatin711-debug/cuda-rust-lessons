# Easy: vector addition

**Problem:** given arrays A and B, calculate C[i] = A[i] + B[i]. There is no
dependency between different output elements, so the work is easy to parallelize.

Read the [SIMT implementation](../gpu/oxide/src/bin/easy-vector-add.rs), then the
[Tile implementation](../gpu/tile/src/bin/easy-vector-add.rs).

## The answer is small enough to inspect

The first five values are:

```text
A = [-17.0, -16.75, -16.5, -16.25, -16.0]
B = [ -3.0,  -2.50, -2.0,  -1.50,  -1.0]
C = [-20.0, -19.25, -18.5, -17.75, -17.0]
```

We use 1,003 elements, deliberately avoiding a multiple of the block/tile width.
This exposes a common bug: assuming every launched worker has a valid element.

## What cuda-oxide does

The host launches four blocks with 256 threads each. The logical index of a
thread is `block_x * 256 + thread_x`.

```text
block 0 -> elements   0..255
block 1 -> elements 256..511
block 2 -> elements 512..767
block 3 -> elements 768..1002, then 21 excess threads
```

The excess threads receive `None` from `output.get_mut(index)` and skip their
write. The inputs have the same logical length, so a valid output position also
has valid input positions. The host-generated CPU reference checks that shape
relationship before the buffers are allocated.

`DisjointSlice` is the key output abstraction. A plain integer cannot authorize
a mutable output borrow. The trusted thread index and checked one-dimensional
launch geometry establish which thread owns which output element.

## What cuTile Rust does

The host pads both arrays to 1,024 elements with zero, partitions output into
128-element regions, and gets eight logical tile programs. Each program adds a
whole pair of input fragments and stores its own output fragment.

```text
tile 0 -> elements   0..127
tile 1 -> elements 128..255
...
tile 7 -> elements 896..1023
```

The last 21 values are padding. The code verifies those values are also zero.
Padding keeps this first kernel easy to inspect. Current cuTile input views can
also handle partial edge tiles with padding; this example makes the extra
elements explicit rather than introducing view-padding rules immediately.

`partition([128])` establishes ownership, supplies the static tile width, and
determines the logical grid. It does not request 128 physical GPU threads.

## Follow the host chain

`copy_host_vec_to_device` and `zeros` construct lazy operations. The kernel call
owns those operations. `first` selects output from the returned tuple;
`unpartition` removes the host partition wrapper; `to_host_vec` schedules a copy;
`sync_on` executes the work and waits before returning readable CPU data.

The SIMT example instead prepares and immediately enqueues the launch on its
stream. Its host readback synchronizes before the CPU checks the answer.

## A useful experiment

Change `VECTOR_LEN` in `src/lib.rs` to 1, 255, 256, or 257 and rerun both programs.
The actual GPU comparison must still pass. The host prints the computed padding
and truncates the preview for small inputs. Next compare the two host interfaces: one borrows buffers, the
other moves operations/tensors through a chain.

For experiments without changing the shared constant, use the next lesson's
configurable SAXPY binary: `bash scripts/run_gpu.sh all saxpy 257 2`.

**Checkpoint:** identify the thread that owns index 1002 and the Tile partition
that contains it. Explain why the following SIMT thread skips input reads.
Continue to [SAXPY and fusion](04-saxpy-and-fusion.md).
