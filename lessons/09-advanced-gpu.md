# Advanced track: coordination, memory transactions, and measurement

This track assumes you already know Rust borrowing, CUDA grids, and basic
shared-memory/barrier rules. Start here if the earlier chapters are familiar.
The examples add explicit CUDA mechanisms through cuda-oxide; the original
paired SIMT/Tile programs remain useful baselines.

Run in WSL Bash after sourcing the prepared environment:

```bash
cd /mnt/d/CUDA_RUST
bash scripts/run_gpu.sh oxide device-reduction 1048576 100
bash scripts/run_gpu.sh oxide tiled-transpose 1024 1024 100
bash scripts/run_gpu.sh oxide saxpy-events 1048576 100
bash scripts/verify_advanced.sh
```

## 1. Device-only multi-pass reduction

Read [device-reduction.rs](../gpu/oxide/src/bin/device-reduction.rs).
Unlike `block-reduction`, the final sum is computed on the device. There is one
initial input transfer and only a single scalar result readback. No host reads
the intermediate partial sums.

```text
1003 input values -> 8 partials -> 1 result
1048576 values    -> 8192 partials -> 64 partials -> 1 result
```

Each pass reduces chunks of 128 values using the same kernel. The host allocates
all pass buffers and prepares their launch contracts before timing. Repetitions
reuse those allocations. Every pass writes its own output allocation; attempting
to reduce in place could let a block overwrite data another block still needs.

The host is still responsible for launching each pass. “Device-only reduction”
describes where the mathematical intermediate/final reductions happen, not an
absence of CPU orchestration.

### Borrowing expresses the dataflow

```rust
let (before, after) = buffers.split_at_mut(pass + 1);
module.reduce_pass(&stream, contract, &before[pass], &mut after[0])?;
```

`split_at_mut` proves the buffer containers lie in nonoverlapping regions. The
input borrow and mutable output borrow therefore coexist without an aliasing
workaround. At the device level, separate allocations and one block writer per
partial provide the corresponding memory-access proof.

Each subsequent launch uses the same stream, establishing the producer/consumer
ordering. No block assumes another block has already run. Excess lanes contribute
zero and still reach the barriers. For length 1, the existing singleton is already
the answer: the program submits no reduction kernel.

**Investigate:** lengths 16384 and 16385 change the pass graph from `[16384,128,1]`
to `[16385,129,2,1]`. Explain why a single additional input can require another
pass. Then read the supplied [warp reduction](10-warp-reduction.md) and compare
synchronization, shared-memory traffic, work assignment, and event intervals.
A partial physical warp requires correct participation masks; the supplied
variant instead launches complete warps and zero-fills logical input tails.

## 2. Shared-memory transpose with padded bank layout

Read [tiled-transpose.rs](../gpu/oxide/src/bin/tiled-transpose.rs). It contains
both a strided-read baseline and a shared-memory version, using the **same**
32x8 block shape and output-tile mapping.

A block owns a 32x32 output tile. Each of 256 threads handles four cells:

```text
local x = threadIdx.x                  0..31
local y = threadIdx.y                  0..7
local rows visited = y, y+8, y+16, y+24
```

The tiled version first loads four input cells per thread with contiguous X
addresses into `shared[(y+offset)*33 + x]`. All 1024 real tile cells are initialized,
including zero for invalid edge coordinates. One barrier makes the tile ready.
The same threads then store output cells from `shared[x*33 + y+offset]`.

Input tile axes and within-tile coordinates both transpose. This arrangement
allows contiguous global reads and contiguous global writes while exchanging
data through shared memory. Blocks launch over output tiles, so the input tile
base reverses the two block coordinates.

### Why 33 columns?

For 32-bit shared values and the conventional 32-bank model, a word address maps
to its bank modulo 32. A column access with row stride 32 maps different rows to
the same bank. Stride 33 maps `x*33+column` to `(x+column) mod 32`, distributing
the warp's rows across banks. The padding column itself is never read or written.
This is the classic padded-tile technique described in NVIDIA's
[shared-memory bank discussion](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks).

Storage is `32*33*4 = 4224` bytes per block. The extra column addresses a bank
layout issue; the global matrix still retains its original uneven dimensions.
There is no second barrier because the shared tile is not reused for another
load phase within this kernel. Compare this to matmul's repeated tile loop.

### Why explicitly unsafe output stores?

One thread writes four rows of a logical tile. Its standard runtime 2D index
describes a single physical grid position; it does not automatically authorize
this four-element mapping. The kernel uses documented unchecked output access
and supplies a stronger manual argument:

- Blocks own disjoint 32x32 output regions.
- `(x,y,offset)` is a bijection over each region's 1024 cells.
- Shape guards restrict writes to actual output cells.
- Shared reads occur after initialization and the block barrier.

The output still declares the 2D index space required by the checked launch
contract, with `RowWidth` bound on the host. That declaration does not by itself
prove unchecked multi-cell writes safe; the mapping above does.

**Investigate:** test 1x33, 33x1, 31x33, 32x32, and 33x31 before larger matrices.
Add an unpadded 32x32 shared variant to isolate bank-layout effects. Use the same
geometry and timing scope, and inspect profiler metrics before attributing a
speed change to bank conflicts. A tiny matrix may favor the simpler baseline.

## 3. Split versus fused SAXPY with CUDA events

Read [saxpy-events.rs](../gpu/oxide/src/bin/saxpy-events.rs). This comparison uses
the same inputs, 128-thread block geometry, resident allocations, warmup, and
timing scope for both implementations.

```text
split: temporary = 2*x; output = temporary+y  (two launches)
fused: output = 2*x+y                        (one launch)
```

All input/output/intermediate allocations precede timing. Readback and full CPU
comparisons follow it. The chosen input fractions make both variants easy to
compare; arbitrary f32 values may expose multiply-add rounding differences.

Timing events are explicitly enabled with `context.new_event(Some(0))`. In this
pinned runtime, `None` creates an event with timing disabled. Record the start
event, submit repeated work, record the stop event, then synchronize the stop
before querying `elapsed_ms`.

These intervals describe the stream timeline between events. They can include
idle gaps while the host submits launches; they are not automatically the sum
of individual kernel durations. The examples batch repetitions under one event
pair rather than placing event calls around every tiny kernel. NVIDIA's
[event API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
documents the timing and synchronization semantics.

The ideal array traffic is 20*N bytes for split and 12*N for fused. Repeated
arrays may remain in caches, so those counts are not measured DRAM traffic.
Small workloads may expose launch costs; larger workloads may expose memory
traffic. Collect several trials and reverse measurement order to investigate
clock, thermal, cache, and ordering effects. Avoid drawing a peak-bandwidth
conclusion from a single event interval.

## Deeper follow-on work

Continue with supplied examples for [warp-level reduction](10-warp-reduction.md),
[hierarchical prefix scan](11-hierarchical-scan.md), and a
[complete GPU classifier](12-gpu-classifier.md). Change one mechanism at a time
and retain shape-edge cases and full reference comparisons. The
[continuing roadmap](ROADMAP.md) defines future projects for atomics, sorting,
CUDA Graph replay, cross-stream dependencies, profiling, and attention.
