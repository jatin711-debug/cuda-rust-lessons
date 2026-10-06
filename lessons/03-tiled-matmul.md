# Hard: tiled rectangular matrix multiplication

**Problem:** calculate C = A x B while reusing input data efficiently. Matrix
multiplication is a core operation in neural networks and scientific computing.

Read the [SIMT implementation](../gpu/oxide/src/bin/hard-matmul.rs) and
[Tile implementation](../gpu/tile/src/bin/hard-matmul.rs).

## Indexing comes before optimization

For row-major matrices:

```text
A has shape [M,K]; A[row,inner] lives at row*K + inner.
B has shape [K,N]; B[inner,col] lives at inner*N + col.
C has shape [M,N]; C[row,col] lives at row*N + col.

C[row,col] = sum over inner of A[row,inner] * B[inner,col]
```

This lesson uses M=35, K=23, N=19. All three dimensions have incomplete 16-wide
tiles. The reference checks all 665 real output elements, not a random sample.

## SIMT: explicitly stage data in shared memory

Each block contains 16x16 physical threads and computes one 16x16 output region.
Each thread accumulates one output element.

For an inner step:

1. Each thread loads one A value and one B value into distinct shared cells.
2. A barrier makes the loaded tiles available to the whole block.
3. Threads accumulate 16 multiply-add contributions from those shared tiles.
4. A second barrier prevents the next loads from overwriting data still being read.

```text
global A/B -> shared 16x16 tiles -> reuse during dot products -> output C
```

The two shared tiles take 2 x 256 x 4 = 2,048 bytes per block. Their allocation
is static, so the host requests zero *dynamic* shared-memory bytes.

An edge thread may have no real output, yet it still participates in tile loads
and barriers. Invalid input positions contribute zero. Only after the complete
inner loop does it skip its output write. An early return before the barriers
could leave valid threads waiting forever.

Output uses `Runtime2DIndex`. `RowWidth::new` binds N on the host so every thread
resolves its output coordinates against the same row width. Choosing a different
row width independently in each thread could make two coordinates alias.

## Tile: reuse through matrix operations

One logical program owns an output tile `[BM,BN]`. At inner step p, it loads:

```text
A tile [BM,BK] at [output_tile_row, p]
B tile [BK,BN] at [p, output_tile_column]

accumulator = mma(A_tile, B_tile, accumulator)
```

`mma` describes multiply-and-accumulate over those matrix fragments. The Tile IR
compiler chooses its hardware mapping; supported shapes/types can use Tensor
Cores. This is fundamentally different from spelling out one thread's scalar
dot product.

The teaching code sets BM=BN=BK=16 and pads M/K/N to 48/32/32 with zero. In
particular, padding K matters: looping over `K / BK` with unpadded K=23 would
silently omit seven inner terms. The padded loop has two complete inner steps.
It copies the padded result back, verifies its edge region, then crops out the
original 35x19 answer.

Const generics control specialization. BM/BN are output tile sizes; BK is an
inner blocking choice; K here is the padded inner extent. Changing any one can
produce a different compiled kernel. This is a teaching choice; production
kernels often keep full tensor dimensions dynamic to reuse compiled variants.

## Precision and performance

Both examples accept f32 inputs. The data uses small binary fractions, which
are exactly representable even in TF32, to isolate indexing and tiling from
precision effects. This does not prove arbitrary f32 MMA matches scalar f32
arithmetic bit for bit; use application-specific tolerances and precision modes
for other inputs.

The SIMT kernel is a conventional shared-memory teaching GEMM. It does not
implement the tensor-core instruction paths of a tuned CUDA GEMM. Comparing its
timing directly against Tile's compiler-managed `mma` would conflate the
algorithm, hardware instructions, and library interfaces.

Use these small inputs to understand correctness. After that, experiment with
larger matrices, tile shapes, data types, and profiling. Retain the CPU check and
the edge handling as you make those changes.

**Checkpoint:** trace one A load, one B load, and one C store at an edge block.
Explain the different purposes of the two barriers and why padding K is required.
Continue to [streams, ownership, and timing](07-streams-and-timing.md).
