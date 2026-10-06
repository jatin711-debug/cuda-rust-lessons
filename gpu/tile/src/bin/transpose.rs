//! A transposed tile is stored in its exclusively owned output region.
use cuda_rust_lessons::{
    check_close, crop_matrix, learning::transpose, pad_matrix, round_up, MATRIX_M, MATRIX_N,
};
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn transpose_matrix<const WIDTH: i32>(
        output: &mut Tensor<f32, { [WIDTH, WIDTH] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tiles = input.partition(shape![WIDTH, WIDTH]);
        // Output tile [r,c] corresponds to input tile [c,r].
        let values = tiles.load([program_id(1), program_id(0)]);
        output.store(values.transpose());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = (MATRIX_M, MATRIX_N);
    let input: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25).collect();
    let expected = transpose(&input, rows, cols)?;
    let padded_rows = round_up(rows, 16);
    let padded_cols = round_up(cols, 16);
    let host = Arc::new(pad_matrix(&input, rows, cols, padded_rows, padded_cols)?);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let input = api::copy_host_vec_to_device(&host)
        .sync_on(&stream)?
        .reshape(&[padded_rows, padded_cols])?;
    let output = api::zeros::<f32>(&[padded_cols, padded_rows]).partition([16, 16]);
    let padded: Vec<f32> = kernels::transpose_matrix(output, input)
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;
    let actual = crop_matrix(&padded, padded_cols, padded_rows, cols, rows)?;
    let error = check_close(&actual, &expected, 0.0, 0.0)?;
    let padded_expected = pad_matrix(&expected, cols, rows, padded_cols, padded_rows)?;
    check_close(&padded, &padded_expected, 0.0, 0.0)?;
    println!("Tile TRANSPOSE: [{rows},{cols}] -> [{cols},{rows}], checked real + padded outputs; max error {error}");
    Ok(())
}
