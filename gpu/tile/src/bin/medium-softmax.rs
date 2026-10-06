//! Problem: turn each row of logits into probabilities without exp overflow.
use cuda_rust_lessons::*;
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn stable_softmax<const COLS: i32>(
        output: &mut Tensor<f32, { [1, COLS] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
    ) {
        // One logical program owns one complete row. Splitting a row into
        // separate programs would require combining their max/sum reductions.
        let row = logits.load_like(output);
        // Reducing axis 1 removes the column dimension: [1, COLS] -> [1].
        // Give the reduction its result type before reshaping for broadcasting.
        let row_maximum: Tile<f32, { [1] }> = reduce_max(row, 1i32);
        let maximum = row_maximum.reshape(shape![1, 1]).broadcast(shape![1, COLS]);
        let numerator = exp(row - maximum);
        let row_sum: Tile<f32, { [1] }> = reduce_sum(numerator, 1i32);
        let denominator = row_sum.reshape(shape![1, 1]).broadcast(shape![1, COLS]);
        output.store(numerator / denominator);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let logits = softmax_inputs();
    let expected = softmax_rows(&logits, SOFTMAX_ROWS, SOFTMAX_COLS)?;
    let host = Arc::new(logits);
    let input = api::copy_host_vec_to_device(&host)
        .sync_on(&stream)?
        .reshape(&[SOFTMAX_ROWS, SOFTMAX_COLS])?;
    let output = api::zeros::<f32>(&[SOFTMAX_ROWS, SOFTMAX_COLS]).partition([1, SOFTMAX_COLS]);

    let actual: Vec<f32> = kernels::stable_softmax(output, input)
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;

    let error = check_close(&actual, &expected, 2e-6, 2e-5)?;
    check_probabilities(&actual, SOFTMAX_ROWS, SOFTMAX_COLS)?;
    println!("Tile MEDIUM: all {SOFTMAX_ROWS} rows match CPU; max error {error}");
    for (index, row) in actual.chunks_exact(SOFTMAX_COLS).enumerate() {
        println!("Row {index}: sum = {:.6}", row.iter().sum::<f32>());
    }
    Ok(())
}
