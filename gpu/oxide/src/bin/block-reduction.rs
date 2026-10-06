//! Each block produces one partial sum; the CPU finishes the small final sum.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::{VECTOR_LEN, check_close, learning::*};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn partial_sums(input: &[f32], mut output: DisjointSlice<f32>) {
        static mut VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let index = block * 128 + lane;
        let value = if index < input.len() {
            input[index]
        } else {
            0.0
        };
        // SAFETY: the launch contract supplies exactly 128 distinct lanes.
        // Each lane initializes its own cell, and all lanes reach each barrier.
        let cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        unsafe { cells.add(lane).write(value) };
        thread::sync_threads();
        let mut offset = 64usize;
        while offset > 0 {
            if lane < offset {
                // SAFETY: distinct lower-half writers read an upper half that
                // no thread changes in this stage; the previous barrier completed.
                unsafe {
                    let sum = cells.add(lane).read() + cells.add(lane + offset).read();
                    cells.add(lane).write(sum);
                }
            }
            thread::sync_threads();
            offset /= 2;
        }
        if lane == 0 && block < output.len() {
            // SAFETY: only lane zero writes; each block owns a unique output
            // slot. The bounds check covers that slot. A per-thread index would
            // be wrong here because output contains one element per block.
            unsafe { *output.get_unchecked_mut(block) = cells.read() };
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, _) = pointwise_inputs(VECTOR_LEN);
    let expected = block_sums(&input, 128)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let input_device = DeviceBuffer::from_host(&stream, &input)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, expected.len())?;
    // SAFETY: the embedded artifact belongs to this kernels module.
    let module = unsafe { kernels::load(&context)? };
    let prepared =
        module.prepare_partial_sums(LaunchConfig1D::new(expected.len() as u32, 128, 0))?;
    module.partial_sums(&stream, &prepared, &input_device, &mut output)?;
    let actual = output.to_host_vec(&stream)?;
    let error = check_close(&actual, &expected, 1e-5, 1e-5)?;
    let total: f64 = actual.iter().map(|&v| f64::from(v)).sum();
    let reference: f64 = input.iter().map(|&v| f64::from(v)).sum();
    if (total - reference).abs() > 1e-5 {
        return Err(format!("final sum differs: {total} versus {reference}").into());
    }
    println!(
        "SIMT REDUCTION: checked {} partials; total {total}; max error {error}",
        actual.len()
    );
    println!("Partials: {actual:?}; the final sum runs on the CPU.");
    Ok(())
}
