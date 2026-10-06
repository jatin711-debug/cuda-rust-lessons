//! Grid-stride accumulation plus full-warp shuffles and one block barrier/pass.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use cuda_rust_lessons::{check_close, learning::*};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn warp_pass(input: &[f32], mut output: DisjointSlice<f32>) {
        static mut WARP_SUMS: SharedArray<f32, 4> = SharedArray::UNINIT;
        let local = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let lane = local % 32;
        let warp_id = local / 32;
        let stride = thread::gridDim_x() as usize * 128;
        let mut index = block * 128 + local;
        let mut sum = 0.0f32;
        while index < input.len() {
            sum += input[index];
            index += stride;
        }
        // All 32 physical lanes in each warp reach this call, including lanes
        // with no logical input. Never put it inside `if index < input.len()`.
        let warp_sum = warp::reduce_sum_f32(sum);
        // SAFETY: exactly four warps, one lane-zero writer per shared cell.
        let cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_SUMS) };
        if lane == 0 {
            unsafe { cells.add(warp_id).write(warp_sum) };
        }
        thread::sync_threads(); // Inter-warp handoff requires a block barrier.
        if warp_id == 0 {
            // SAFETY: the barrier completed all four writes; only indices 0..4
            // are read. The other 28 lanes supply neutral zeros.
            let contribution = if lane < 4 {
                unsafe { cells.add(lane).read() }
            } else {
                0.0
            };
            let block_sum = warp::reduce_sum_f32(contribution);
            if lane == 0 && block < output.len() {
                // SAFETY: one writer per block, unique output slot, checked bounds.
                unsafe { *output.get_unchecked_mut(block) = block_sum };
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ExperimentOptions::from_env()?;
    let (input, _) = pointwise_inputs(options.len);
    let expected = input.iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let mut buffers = vec![DeviceBuffer::from_host(&stream, &input)?];
    let mut sizes = vec![options.len];
    let mut len = options.len;
    while len > 1 {
        // A bounded grid forces grid-stride work for large inputs. This is a
        // teaching choice, not a device-specific occupancy optimum.
        len = len.div_ceil(128).min(128);
        sizes.push(len);
        buffers.push(DeviceBuffer::<f32>::zeroed(&stream, len)?);
    }
    // SAFETY: this executable owns its matching compiled kernels artifact.
    let module = unsafe { kernels::load(&context)? };
    let contracts = sizes
        .iter()
        .skip(1)
        .map(|&blocks| module.prepare_warp_pass(LaunchConfig1D::new(blocks as u32, 128, 0)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut enqueue = || -> Result<(), Box<dyn std::error::Error>> {
        for (pass, contract) in contracts.iter().enumerate() {
            let (before, after) = buffers.split_at_mut(pass + 1);
            module.warp_pass(&stream, contract, &before[pass], &mut after[0])?;
        }
        Ok(())
    };
    enqueue()?;
    stream.synchronize()?;
    let start = context.new_event(Some(0))?;
    let stop = context.new_event(Some(0))?;
    start.record(&stream)?;
    for _ in 0..options.repeats {
        enqueue()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let ms = start.elapsed_ms(&stop)?;
    let actual = buffers.last().unwrap().to_host_vec(&stream)?;
    let error = check_close(&actual, &[expected], 1e-5, 1e-5)?;
    println!(
        "WARP REDUCTION: lengths {sizes:?}; sum {}; max error {error}",
        actual[0]
    );
    println!(
        "{} repetitions: {ms:.6} ms CUDA event interval; 16 shared bytes/block, one block barrier/pass",
        options.repeats
    );
    Ok(())
}
