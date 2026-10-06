//! One logical tile program computes each row sum; the CPU sums the partials.
use cuda_rust_lessons::{check_close, learning::*, round_up, VECTOR_LEN};
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn partial_sums<const WIDTH: i32>(
        output: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        // Output has one row per partial. That row selects the corresponding
        // WIDTH-element input partition; the column coordinate is always zero.
        let rows = input.partition(shape![1, WIDTH]);
        let values = rows.load([program_id(0), 0i32]);
        let sum: Tile<f32, { [1] }> = reduce_sum(values, 1i32);
        output.store(sum.reshape(shape![1, 1]));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut input, _) = pointwise_inputs(VECTOR_LEN);
    let expected = block_sums(&input, 128)?;
    let reference: f64 = input.iter().map(|&v| f64::from(v)).sum();
    input.resize(round_up(VECTOR_LEN, 128), 0.0);
    let host = Arc::new(input);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let input = api::copy_host_vec_to_device(&host)
        .sync_on(&stream)?
        .reshape(&[expected.len(), 128])?;
    let output = api::zeros::<f32>(&[expected.len(), 1]).partition([1, 1]);
    let actual: Vec<f32> = kernels::partial_sums(output, input)
        .generics(vec!["128".into()])
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;
    let error = check_close(&actual, &expected, 1e-5, 1e-5)?;
    let total: f64 = actual.iter().map(|&v| f64::from(v)).sum();
    if (total - reference).abs() > 1e-5 {
        return Err(format!("final sum differs: {total} versus {reference}").into());
    }
    println!(
        "Tile REDUCTION: checked {} partials; total {total}; max error {error}",
        actual.len()
    );
    println!("Partials: {actual:?}; the final sum runs on the CPU.");
    Ok(())
}
