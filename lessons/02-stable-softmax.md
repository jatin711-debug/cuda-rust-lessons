# Medium: numerically stable softmax

**Problem:** turn a row of model scores, or logits, into positive probabilities
that sum to one. Softmax appears in classification and attention.

Read the [SIMT implementation](../gpu/oxide/src/bin/medium-softmax.rs) alongside the
[Tile implementation](../gpu/tile/src/bin/medium-softmax.rs).

## Solve the numerical problem first

The naive formula is `exp(x[i]) / sum(exp(x[j]))`. In f32, `exp(1000)` overflows.
The formula can then produce an infinity divided by infinity, which is NaN.

Use this mathematically equivalent form:

```text
maximum = max(x)
numerator[i] = exp(x[i] - maximum)
probability[i] = numerator[i] / sum(numerator)
```

Every exponent is nonpositive, so positive overflow is avoided. At least one
numerator equals one, making the denominator positive for these finite inputs.
Very small probabilities may underflow to zero, which is acceptable here.

The four test rows contain large positive values, large negative values, equal
values, and one dominant value. We verify every probability against an f64 CPU
reference and independently check the finite/range/row-sum properties.

## SIMT: a block cooperates on one row

This teaching kernel is specialized for **64 columns**. One 64-thread block
owns one row; each thread loads one logit. Two shared arrays hold intermediates.

The maximum reduction follows a tree:

```text
64 candidates -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 maximum
```

At each step, the active lower-half threads combine their value with a distinct
upper-half value. A barrier makes the stage complete before the next stage.
The sum reduction uses the same work division after exponentiation.

All threads reach every barrier, including threads that are inactive writers
at a particular stage. Putting a barrier inside `if lane < offset` would make
only some threads arrive, which is invalid.

The two separate shared arrays prevent another subtle race: thread zero must
not overwrite the maximum with its numerator while other threads still need to
read the maximum. MAXIMA remains unchanged while SUMS is populated and reduced.

The shared arrays require `unsafe` raw access. Each comment states the disjoint
writes and synchronization that make those accesses valid. The final global
store goes through `DisjointSlice`.

## Tile: one program operates on a whole row

The output partition has shape `[1, 64]`. The kernel loads that complete row,
reduces along axis 1, reshapes the result to `[1, 1]`, then broadcasts it back to
`[1, 64]` for elementwise subtraction/division.

```text
row                     [1,64]
reduce_max(row, axis=1)  [1]
reshape                 [1,1]
broadcast               [1,64]
```

The compiler implements thread cooperation and internal synchronization. The
kernel never names a shared array or a physical lane.

The full row must fit into one output partition for this simple formula. If you
split one row across multiple tile programs, each would compute a local maximum
and local denominator; those would not normalize the entire row. Larger-row or
distributed softmax needs another algorithm and coordination.

## Why fusion helps

Both kernels perform the maximum, exponentiation, sum, and normalization inside
one kernel. They avoid writing several full intermediate arrays to global
memory. On the Tile side, those intermediates are computational tile values.
On the SIMT side, some state is per-thread and some is in block-shared memory.

Do not change the SIMT column count alone: its block shape and two shared arrays
are deliberately fixed at 64. The host assertion tells you when that lesson's
contract no longer matches your input.

**Checkpoint:** explain why subtracting the maximum preserves the distribution,
why the maximum and sum use separate shared arrays, and why a partial-row Tile
partition would normalize the wrong set of values.
Continue to [transpose and memory access](06-transpose-and-memory.md).
