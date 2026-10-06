//! Multi-pass reduction: every intermediate and the final sum stay on the GPU.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::{check_close, learning::*};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn reduce_pass(input: &[f32], mut output: DisjointSlice<f32>) {
        static mut VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let index = block * 128 + lane;
        // Excess lanes contribute zero but still participate in every barrier.
        let value = if index < input.len() {
            input[index]
        } else {
            0.0
        };
        // SAFETY: exactly 128 lanes initialize distinct cells in 0..128.
        let cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        unsafe { cells.add(lane).write(value) };
        thread::sync_threads();
        let mut offset = 64usize;
        while offset > 0 {
            if lane < offset {
                // SAFETY: lower-half writers are distinct; upper-half cells
                // are read-only in this stage. Barriers separate stages.
                unsafe {
                    cells
                        .add(lane)
                        .write(cells.add(lane).read() + cells.add(lane + offset).read());
                }
            }
            thread::sync_threads();
            offset /= 2;
        }
        if lane == 0 && block < output.len() {
            // SAFETY: one writer per block, unique block slots, checked bounds.
            unsafe { *output.get_unchecked_mut(block) = cells.read() };
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ExperimentOptions::from_env()?;
    let (input, _) = pointwise_inputs(options.len);
    let expected = input.iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // Each pass owns a separate allocation: no in-place inter-block races.
    let mut buffers = vec![DeviceBuffer::from_host(&stream, &input)?];
    let mut sizes = vec![options.len];
    let mut len = options.len;
    while len > 1 {
        len = len.div_ceil(128);
        buffers.push(DeviceBuffer::<f32>::zeroed(&stream, len)?);
        sizes.push(len);
    }
    // SAFETY: this executable owns the matching compiled device module.
    let module = unsafe { kernels::load(&context)? };
    let prepared = sizes
        .iter()
        .skip(1)
        .map(|&len| module.prepare_reduce_pass(LaunchConfig1D::new(len as u32, 128, 0)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut enqueue = || -> Result<(), Box<dyn std::error::Error>> {
        for (pass, contract) in prepared.iter().enumerate() {
            // Rust proves these buffers are separate: borrow an input from the
            // prefix and an exclusive output from the nonoverlapping suffix.
            let (before, after) = buffers.split_at_mut(pass + 1);
            module.reduce_pass(&stream, contract, &before[pass], &mut after[0])?;
        }
        Ok(())
    };
    enqueue()?;
    stream.synchronize()?;
    // CUDA event flag 0 is CU_EVENT_DEFAULT: timing enabled (None disables it).
    let start = context.new_event(Some(0))?;
    let stop = context.new_event(Some(0))?;
    start.record(&stream)?;
    for _ in 0..options.repeats {
        enqueue()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let device_ms = start.elapsed_ms(&stop)?;
    // The only result readback is the final single-element buffer.
    let actual = buffers.last().unwrap().to_host_vec(&stream)?;
    let error = check_close(&actual, &[expected], 1e-5, 1e-5)?;
    println!(
        "DEVICE REDUCTION: lengths {sizes:?}; {} passes; sum {}; max error {error}",
        prepared.len(),
        actual[0]
    );
    println!(
        "{} repetitions: {device_ms:.6} ms CUDA event interval; allocation/transfers excluded",
        options.repeats
    );
    println!("Length 1 needs no reduction launch; it reads back the original singleton.");
    Ok(())
}
