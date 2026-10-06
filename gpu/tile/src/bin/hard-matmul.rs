//! Problem: reuse loaded matrix tiles instead of reloading per output element.
use cuda_rust_lessons::*;
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn tiled_matmul<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        output: &mut Tensor<f32, { [BM, BN] }>,
        left: &Tensor<f32, { [-1, K] }>,
        right: &Tensor<f32, { [K, -1] }>,
    ) {
        // This program owns C's tile at logical coordinates (tile_m, tile_n).
        let tile_m = program_id(0);
        let tile_n = program_id(1);
        let left_tiles = left.partition(shape![BM, BK]);
        let right_tiles = right.partition(shape![BK, BN]);
        let mut accumulator = constant(0.0f32, shape![BM, BN]);

        // K is padded to a multiple of BK on the host, so no K tail is lost.
        for step in 0i32..(K / BK) {
            let a = left_tiles.load([tile_m, step]);
            let b = right_tiles.load([step, tile_n]);
            accumulator = mma(a, b, accumulator);
        }
        output.store(accumulator);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let (a, b) = matrix_inputs();
    let expected = matmul(&a, &b, MATRIX_M, MATRIX_K, MATRIX_N)?;
    let (m, k, n) = (
        round_up(MATRIX_M, 16),
        round_up(MATRIX_K, 16),
        round_up(MATRIX_N, 16),
    );

    // Zero padding preserves the real product while keeping the device kernel
    // simple. SIMT's version handles these edges with per-thread conditions.
    let host_a = Arc::new(pad_matrix(&a, MATRIX_M, MATRIX_K, m, k)?);
    let host_b = Arc::new(pad_matrix(&b, MATRIX_K, MATRIX_N, k, n)?);
    let left = api::copy_host_vec_to_device(&host_a)
        .sync_on(&stream)?
        .reshape(&[m, k])?;
    let right = api::copy_host_vec_to_device(&host_b)
        .sync_on(&stream)?
        .reshape(&[k, n])?;
    let output = api::zeros::<f32>(&[m, n]).partition([16, 16]);

    let padded: Vec<f32> = kernels::tiled_matmul(output, left, right)
        // BM/BN describe the output; BK is the inner tile width. Explicitly
        // specialize all four constants so that the compilation choices are visible.
        .generics(vec!["16".into(), "16".into(), "16".into(), k.to_string()])
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;
    let actual = crop_matrix(&padded, m, n, MATRIX_M, MATRIX_N)?;
    let error = check_close(&actual, &expected, 1e-4, 1e-4)?;
    // Also verify the padded output: unintended stores cannot hide in the edges.
    let padded_expected = pad_matrix(&expected, MATRIX_M, MATRIX_N, m, n)?;
    check_close(&padded, &padded_expected, 1e-4, 1e-4)?;
    println!(
        "Tile HARD: checked all {} real outputs; max error {error}",
        actual.len()
    );
    println!("Logical shape M/K/N = {MATRIX_M}/{MATRIX_K}/{MATRIX_N}; padded = {m}/{k}/{n}");
    println!("First eight outputs: {:?}", &actual[..8]);
    Ok(())
}
