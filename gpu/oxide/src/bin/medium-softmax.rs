//! Problem: cooperate within a block to normalize a complete row of logits.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::*;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn stable_softmax(logits: &[f32], mut output: DisjointSlice<f32>) {
        // CUDA gives each block its own shared arrays. The 64 threads in this
        // block process the 64 columns of a single row.
        static mut MAXIMA: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, 64> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let value = logits[row * 64 + lane];

        // SAFETY: &raw mut creates no overlapping whole-array Rust references.
        // The exact 64-thread contract ensures lane < 64; each lane initially
        // writes only its own cell. Later reads/writes are ordered by barriers.
        let maximum_cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
        let sum_cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe { maximum_cells.add(lane).write(value) };
        thread::sync_threads();

        // Tree reduction: 64 values -> 32 -> 16 -> ... -> one maximum.
        // At each stage the active writers own distinct cells and read cells
        // in the upper half, which nobody writes during that stage.
        let mut offset = 32usize;
        while offset > 0 {
            if lane < offset {
                unsafe {
                    let left = maximum_cells.add(lane).read();
                    let right = maximum_cells.add(lane + offset).read();
                    maximum_cells.add(lane).write(left.max(right));
                }
            }
            thread::sync_threads(); // All 64 threads reach every barrier.
            offset /= 2;
        }
        let maximum = unsafe { maximum_cells.read() };
        let numerator = (value - maximum).exp();

        // Separate arrays matter: overwriting MAXIMA[0] before all threads
        // have read it would create a race. SUMS leaves the maximum unchanged.
        unsafe { sum_cells.add(lane).write(numerator) };
        thread::sync_threads();
        offset = 32;
        while offset > 0 {
            if lane < offset {
                unsafe {
                    let total = sum_cells.add(lane).read() + sum_cells.add(lane + offset).read();
                    sum_cells.add(lane).write(total);
                }
            }
            thread::sync_threads();
            offset /= 2;
        }
        let denominator = unsafe { sum_cells.read() };
        let index = thread::index_1d();
        if let Some(destination) = output.get_mut(index) {
            *destination = numerator / denominator;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let logits = softmax_inputs();
    let expected = softmax_rows(&logits, SOFTMAX_ROWS, SOFTMAX_COLS)?;
    assert_eq!(
        SOFTMAX_COLS, 64,
        "this teaching kernel specializes on 64 columns"
    );
    let input = DeviceBuffer::from_host(&stream, &logits)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, logits.len())?;
    // SAFETY: this binary embeds its own matching device artifact.
    let module = unsafe { kernels::load(&context)? };
    let prepared =
        module.prepare_stable_softmax(LaunchConfig1D::new(SOFTMAX_ROWS as u32, 64, 0))?;
    module.stable_softmax(&stream, &prepared, &input, &mut output)?;

    let actual = output.to_host_vec(&stream)?;
    let error = check_close(&actual, &expected, 2e-6, 2e-5)?;
    check_probabilities(&actual, SOFTMAX_ROWS, SOFTMAX_COLS)?;
    println!("SIMT MEDIUM: all {SOFTMAX_ROWS} rows match CPU; max error {error}");
    for (index, row) in actual.chunks_exact(SOFTMAX_COLS).enumerate() {
        println!("Row {index}: sum = {:.6}", row.iter().sum::<f32>());
    }
    Ok(())
}
