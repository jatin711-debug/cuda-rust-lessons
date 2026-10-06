# Beyond the current lessons

The project has runnable baselines through warp cooperation, hierarchical scan,
and complete GPU inference. Those are foundations for a much deeper path.
**The projects below are specifications for future extensions, not supplied
binaries.** Each stage has a concrete result and a correctness checkpoint.

## Choose a route through the existing code

If Rust and CUDA basics are familiar, follow this order:

1. [Advanced coordination and timing](09-advanced-gpu.md): multi-pass reduction,
   padded transpose, split/fused work, and event intervals.
2. [Warp reduction](10-warp-reduction.md): collectives, grid-stride coverage,
   and the difference between data tails and lane participation.
3. [Hierarchical scan](11-hierarchical-scan.md): global offsets and ordered
   producer/consumer launches across several levels.
4. [GPU classifier](12-gpu-classifier.md): complete resident-buffer dataflow
   with independent checks of intermediate tensors.

Before progressing, explain each output ownership mapping, every barrier, and
the timing scope. Run `bash scripts/verify_deep_core.sh` on supported NVIDIA
hardware. The pinned library revisions and verified machine are recorded in
[verification](../VERIFICATION.md); future capabilities must be checked against
the chosen runtime and GPU.

## Stage A: irregular parallel algorithms

| Project | Build in order | Graduation checkpoint |
| --- | --- | --- |
| Histogram | Atomic global bins, then block-private bins and a merge | Every bin matches an integer CPU count; sum of bins equals input length |
| Stable compaction | Boolean flags, exclusive scan, scatter, device output count | Output equals the CPU filter in original order, including all-kept/all-dropped cases |
| Stable radix sort | One-bit partition using flags/scan, then multiple digit passes | Keys sorted; attached original indices retain order among equal keys |
| Sparse matrix-vector product | CSR representation and row-owned baseline, then cooperative long rows | Compare every row against a CPU sparse product, including empty/skewed rows |

Start histogram tests with uniform, single-bin, and highly skewed data. Atomic
updates prevent lost increments but contention can serialize hot addresses.
Block-private storage adds initialization and merge costs; measure the tradeoff.
Do not use ordinary `&mut` writes to implement overlapping updates. Inspect the
pinned runtime's atomic types and host/device contracts before choosing an API.

Compaction connects scan to changing output sizes. For radix sort, retain an
original-index payload so stability can be tested rather than inferred from
sorted keys. A one-bit stable partition is already a substantial project;
complete it before increasing the radix.

## Stage B: asynchronous host/device systems

| Project | Core mechanism | Graduation checkpoint |
| --- | --- | --- |
| Chunked transfer/compute pipeline | Pinned host staging, multiple streams, completion events | Same output as a serial baseline; no buffer reuse before its transfer/consumer finishes |
| Cross-stream DAG | Separate producers and consumers with event waits | Correctness remains when independent work is reordered; dependencies shown on a timeline |
| CUDA Graph classifier replay | Capture or explicitly build the fixed inference launch graph | Same layer outputs; separate graph creation, first execution, and repeated replay measurements |
| Reusable Rust execution object | Owned buffers, prepared contracts, explicit submit/wait interface | Invalid shapes rejected; repeated calls cannot alias mutable in-flight storage |

Draw buffer lifetimes first: the host must preserve transfer sources and
destinations until asynchronous operations complete. Pinned memory and multiple
streams enable overlap under the relevant device/runtime conditions; a profiler
timeline must demonstrate whether overlap actually occurs. An event dependency
provides ordering; elapsed event time is a separate measurement concern. NVIDIA's
[asynchronous execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
is the reference for stream and event behavior.

Keep a serial baseline and measure completed end-to-end work as well as kernel
intervals. Graph support may require extending the pinned host runtime; do not
invent launcher APIs or assume a generated Rust launcher is capture-compatible.

## Stage C: architecture and evidence-driven optimization

Use the existing transpose, reduction, and classifier as profiling targets.
For each change, record an explicit hypothesis, one changed mechanism,
correctness results, several timing trials, and the relevant profiler evidence.

| Investigation | Change | Evidence to collect |
| --- | --- | --- |
| Coalescing and bank layout | Strided/cooperative global loads; padded/unpadded shared tile | Memory transactions and shared bank-conflict metrics |
| Occupancy and register pressure | Block shape, elements per thread, unrolling | Registers, spills, shared storage, achieved occupancy, elapsed time |
| Roofline reasoning | Tile reuse and fusion | Operation/byte estimates with the stated memory level, measured bandwidth/throughput |
| Launch overhead | Batch size, fused epilogues, graph replay | CPU/GPU timeline and completed latency across sizes |

Learn Nsight Systems for timelines, Nsight Compute for kernel behavior, and
Compute Sanitizer for invalid accesses and synchronization hazards. Check tool
availability and platform support before prescribing commands; the current
verification does not claim these tools were run. Preserve CPU comparisons:
a profiler or sanitizer passing does not establish the right mathematical result.
NVIDIA's [best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
connects memory, resource usage, and measurement.

An optimization can lower occupancy and still improve performance. More active
warps can hide latency, while reuse and instruction efficiency can reduce work.
Derive a hypothesis from the workload, then measure; occupancy alone is not the
objective. Include sizes beyond cache residency before calling a repeated-array
result a DRAM-bandwidth measurement.

## Stage D: matrix engines and numerical contracts

Build a correctness-first GEMM benchmark with arbitrary M/N/K, then add fused
epilogues, supported Tensor Core operations, and a small tuning search over tile
shapes. Retain scalar/tiled f32 and CPU f64 references. Report input, accumulator,
and output precision separately, with absolute/relative error and problem scale.

Stress K tails, output edges, cancellation, large magnitudes, and mixed signs.
Reduced-precision inputs change the numerical contract; increasing tolerance
without explaining the observed error is not validation. Compare against a
supported vendor implementation with the same shapes and precision once Rust
bindings are available. Exclude autotuning and compilation from steady-state
timing, and report them separately for cold-start use cases.

Check instruction availability for the GPU's compute capability. The verified
RTX 3050 Laptop is `sm_86`; examples tied to newer architecture features need
separate hardware-specific tracks. Source-level Rust syntax does not make every
CUDA instruction available on every GPU.

## Stage E: attention and training

Build attention in stages so each new dependency has an independent reference:

1. CPU reference for `softmax(Q * K^T / sqrt(d)) * V` with explicit tensor shapes.
2. GPU baseline that materializes scores, masks them, computes row softmax, and
   multiplies by V. Verify causal masking and awkward sequence lengths.
3. Tiled online softmax. Maintain running row maximum `m`, denominator `l`, and
   weighted numerator `o`. For a new score tile S, use:

   ```text
   m_new = max(m, max(S))
   alpha = exp(m - m_new)
   P = exp(S - m_new)
   l_new = alpha * l + sum(P)
   o_new = alpha * o + P * V_tile
   output = o / l after the final tile
   ```

   Masked entries contribute zero. Handle fully masked tiles/rows deliberately:
   subtracting negative infinity from negative infinity produces NaN. State the
   chosen all-masked-row behavior in both CPU and GPU references.
4. Compare every output against the materialized baseline, then measure temporary
   storage and transfers. The online form can avoid a full sequence-by-sequence
   score matrix; it adds rescaling and more involved tiled dataflow.
5. Implement backward formulas and gradient checks before mixed-precision
   training, loss scaling, or fused optimizer updates.

Read the [original FlashAttention paper](https://arxiv.org/abs/2205.14135) for its
IO-aware tiling approach. A teaching implementation using online softmax is not
automatically an equivalent optimized FlashAttention implementation. Tie any
performance claim to a measured implementation, shape, precision, and GPU.

## Stage F: library and hardware scale

Turn validated kernels into a small Rust library with explicit shapes, strides,
dtype/device metadata, fallible launch preparation, and ownership of in-flight
work. Support contiguous tensors first; add views only after deriving aliasing
and lifetime rules. Keep device error propagation and synchronization observable.

Optional multi-GPU work needs additional hardware: peer-transfer validation,
partitioning, collectives, and topology-aware measurement. Start with partitioned
vector work before distributed matrix products. The current single-GPU machine
can develop interfaces and CPU models but cannot verify a multi-GPU execution.

Choose one substantial capstone: a stable sorting library, a sparse graph
workload, a tiled attention kernel, or a training step. A finished capstone should
ship its reference, edge cases, documented memory/lifetime proof, reproducible
measurements, and an explanation of where the design stops scaling.
