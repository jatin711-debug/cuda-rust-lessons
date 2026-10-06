//! Compare split and fused SAXPY with identical resident buffers and event timing.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::{check_close, learning::*};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn scale(input: &[f32], mut output: DisjointSlice<f32>, alpha: f32) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(destination) = output.get_mut(index) {
            *destination = alpha * input[position];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn add(x: &[f32], y: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(destination) = output.get_mut(index) {
            *destination = x[position] + y[position];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn fused(x: &[f32], y: &[f32], mut output: DisjointSlice<f32>, alpha: f32) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(destination) = output.get_mut(index) {
            *destination = alpha * x[position] + y[position];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ExperimentOptions::from_env()?;
    let (x, y) = pointwise_inputs(options.len);
    let expected = saxpy(2.0, &x, &y)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let x_device = DeviceBuffer::from_host(&stream, &x)?;
    let y_device = DeviceBuffer::from_host(&stream, &y)?;
    let mut intermediate = DeviceBuffer::<f32>::zeroed(&stream, options.len)?;
    let mut split_output = DeviceBuffer::<f32>::zeroed(&stream, options.len)?;
    let mut fused_output = DeviceBuffer::<f32>::zeroed(&stream, options.len)?;
    // SAFETY: the executable embeds the artifact compiled from this module.
    let module = unsafe { kernels::load(&context)? };
    let geometry = LaunchConfig1D::new((options.len as u32).div_ceil(128), 128, 0);
    let scale_contract = module.prepare_scale(geometry)?;
    let add_contract = module.prepare_add(geometry)?;
    let fused_contract = module.prepare_fused(geometry)?;
    let mut split = || -> Result<(), Box<dyn std::error::Error>> {
        module.scale(&stream, &scale_contract, &x_device, &mut intermediate, 2.0)?;
        module.add(
            &stream,
            &add_contract,
            &intermediate,
            &y_device,
            &mut split_output,
        )?;
        Ok(())
    };
    let mut fused = || -> Result<(), Box<dyn std::error::Error>> {
        module.fused(
            &stream,
            &fused_contract,
            &x_device,
            &y_device,
            &mut fused_output,
            2.0,
        )?;
        Ok(())
    };
    split()?;
    fused()?;
    stream.synchronize()?;
    // Some(0) enables timing; the runtime's default None disables timing.
    let start = context.new_event(Some(0))?;
    let stop = context.new_event(Some(0))?;
    start.record(&stream)?;
    for _ in 0..options.repeats {
        split()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let split_ms = start.elapsed_ms(&stop)?;
    start.record(&stream)?;
    for _ in 0..options.repeats {
        fused()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let fused_ms = start.elapsed_ms(&stop)?;
    // Readback/verification are outside both measured intervals.
    let split_actual = split_output.to_host_vec(&stream)?;
    let fused_actual = fused_output.to_host_vec(&stream)?;
    check_close(&split_actual, &expected, 1e-5, 1e-5)?;
    check_close(&fused_actual, &expected, 1e-5, 1e-5)?;
    println!(
        "SAXPY EVENTS: both variants checked all {} values",
        options.len
    );
    println!(
        "{} repetitions, CUDA event interval: split {split_ms:.6} ms, fused {fused_ms:.6} ms",
        options.repeats
    );
    println!(
        "Ideal f32 array traffic/iteration: split {} bytes, fused {} bytes",
        20 * options.len,
        12 * options.len
    );
    println!(
        "Same 128-thread geometry and inputs; allocation/copies excluded. Inter-launch gaps remain included."
    );
    Ok(())
}
