//! Two ordered kernels keep their intermediate on the GPU.
use cuda_rust_lessons::{check_close, learning::*, round_up};
use cutile::prelude::*;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale_add<const WIDTH: i32>(
        output: &mut Tensor<f32, { [WIDTH] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
        alpha: f32,
    ) {
        let scale = alpha.broadcast(shape![WIDTH]);
        output.store(scale * x.load_like(output) + y.load_like(output));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ExperimentOptions::from_env()?;
    let (mut x, mut y) = pointwise_inputs(options.len);
    let intermediate_expected = saxpy(2.0, &x, &y)?;
    let mut expected = saxpy(0.5, &intermediate_expected, &y)?;
    let padded_len = round_up(options.len, 128);
    x.resize(padded_len, 0.0);
    y.resize(padded_len, 0.0);
    expected.resize(padded_len, 0.0);
    let x = Arc::new(x);
    let y = Arc::new(y);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let execute = || -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let left = api::copy_host_vec_to_device(&x);
        let right = api::copy_host_vec_to_device(&y);
        let intermediate = api::zeros::<f32>(&[padded_len]).partition([128]);
        let first = kernels::scale_add(intermediate, left, right, 2.0)
            .first()
            .unpartition(); // Device tensor/operation; no intermediate readback.
        let output = api::zeros::<f32>(&[padded_len]).partition([128]);
        let second_y = api::copy_host_vec_to_device(&y);
        Ok(kernels::scale_add(output, first, second_y, 0.5)
            .first()
            .unpartition()
            .to_host_vec()
            .sync_on(&stream)?)
    };
    let _warmup = execute()?; // Compile the specialization before measuring.
    let started = Instant::now();
    let mut actual = Vec::new();
    for _ in 0..options.repeats {
        actual = execute()?;
    }
    let elapsed = started.elapsed();
    let error = check_close(&actual, &expected, 1e-5, 1e-5)?;
    println!(
        "Tile PIPELINE: checked {} real + {} padded elements; max error {error}",
        options.len,
        padded_len - options.len
    );
    println!("{} repetitions: {:.3} ms end-to-end host wall time including allocation + copies + kernels + synchronization", options.repeats, elapsed.as_secs_f64() * 1000.0);
    println!("First values: {:?}", &actual[..options.len.min(5)]);
    Ok(())
}
