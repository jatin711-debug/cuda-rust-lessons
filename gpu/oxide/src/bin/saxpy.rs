//! Fused elementwise scale-and-add, with configurable lengths.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::{check_close, learning::*};
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn scale_add(x: &[f32], y: &[f32], mut output: DisjointSlice<f32>, alpha: f32) {
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
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, options.len)?;

    // SAFETY: this program loads the artifact built from its own kernels module.
    let module = unsafe { kernels::load(&context)? };
    let geometry = Launch1D::new(options.len, 128)?;
    let prepared = module.prepare_scale_add(LaunchConfig1D::new(geometry.blocks as u32, 128, 0))?;
    let mut enqueue = || -> Result<(), Box<dyn std::error::Error>> {
        module.scale_add(&stream, &prepared, &x_device, &y_device, &mut output, 2.0)?;
        Ok(())
    };
    enqueue()?; // Warm up before starting the host clock.
    stream.synchronize()?;
    let started = Instant::now();
    for _ in 0..options.repeats {
        enqueue()?;
    }
    stream.synchronize()?; // Timing submission alone would miss execution.
    let elapsed = started.elapsed();
    let actual = output.to_host_vec(&stream)?;
    let error = check_close(&actual, &expected, 1e-5, 1e-5)?;
    println!(
        "SIMT SAXPY: checked {} elements; max error {error}",
        options.len
    );
    println!(
        "{} repetitions: {:.3} ms host wall time including launches + final synchronization; transfers excluded",
        options.repeats,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("First values: {:?}", &actual[..actual.len().min(5)]);
    Ok(())
}
