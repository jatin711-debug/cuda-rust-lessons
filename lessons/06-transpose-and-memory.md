# Transpose: correct indices can still access memory inefficiently

**Goal:** reason about row-major strides, output ownership, and how tiles can
rearrange data. This lesson adds memory layout without extra arithmetic.

```powershell
cargo run --example lab -- transpose 3 5
```

In WSL Bash:

```bash
bash scripts/run_gpu.sh all transpose
```

The GPU examples use `[35,19]` input and `[19,35]` output, with exact checks of
all 665 real elements. Read [SIMT transpose](../gpu/oxide/src/bin/transpose.rs)
and [Tile transpose](../gpu/tile/src/bin/transpose.rs).

## Write the indices on paper

```text
input  [2,3] = [1,2,3, 4,5,6]
output [3,2] = [1,4, 2,5, 3,6]

input(row,col)  is input[row*cols + col]
output(col,row) is output[col*rows + row]
```

Transpose changes coordinates and dimensions. Simply swapping shape metadata
on a contiguous tensor does not produce the same contiguous output array.

## The SIMT baseline chooses contiguous stores

The grid covers output coordinates. Adjacent X threads increment output column,
so their output stores use consecutive row-major positions. Those same threads
read input rows at one fixed input column; their reads are separated by 19 f32
elements for the default shape.

Warps access global memory through transactions. Nearby, suitably aligned
addresses can require fewer transactions than scattered addresses; this is the
idea behind **coalescing**. The exact traffic depends on alignment, access size,
hardware, and caches. This example intentionally shows the indexing tradeoff,
not a performance result. See NVIDIA's [global-memory discussion](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

`RowWidth::new(&mut output, rows)` is important: output has `rows` columns after
transposition. Binding the original input column count would describe the wrong
output layout. Coordinate checks and the typed output index handle edge threads.

## The Tile version transposes each fragment

The host pads the input to `[48,32]`. The output therefore has padded shape
`[32,48]`; its 16x16 output tile at `[r,c]` reads input tile `[c,r]` and stores
`values.transpose()`. It checks the entire padded result before reporting success.

The two swaps solve different problems:

1. Swap **tile coordinates** to select the correct source region.
2. Transpose **within the tile** to rearrange that region's elements.

Omitting either swap fails the complete reference comparison. Zero padding is
preserved by transposition and can be cropped after readback.

## The shared-memory optimization exercise

A SIMT tiled transpose can load a contiguous input region into shared memory,
synchronize, and read it in a transposed order for contiguous output stores.
The thread that writes an output element may read a shared cell populated by a
different thread. Ownership, initialized cells, barriers, and edge conditions
must all be reasoned through before replacing the baseline.

Sketch a 4x4 tile and mark each thread's load and store. Then extend to 16x16
and an uneven shape. Some layouts add a padding column to alter shared-memory
bank access patterns; that is a separate optimization to measure after correctness.
The [advanced track](09-advanced-gpu.md) includes a complete 32x8-thread,
32x33-shared-memory SIMT version and an equivalent-geometry baseline. Read it
after tracing this simpler implementation.

**Checkpoint:** compute both flattened addresses for input `(row=2,col=4)` in a
`[3,5]` matrix, and identify which accesses are strided in the baseline.
Continue to [matrix multiplication](03-tiled-matmul.md), where shared tiles are
reused for arithmetic instead of rearrangement.
