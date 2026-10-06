use cuda_rust_lessons::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CPU walkthrough: no GPU library is loaded by this program.\n");

    let (a, b) = vector_inputs();
    let sum = vector_add(&a, &b)?;
    println!("EASY: add {VECTOR_LEN} pairs of numbers");
    let shown = VECTOR_LEN.min(5);
    println!("  A preview = {:?}", &a[..shown]);
    println!("  B preview = {:?}", &b[..shown]);
    println!("  C preview = {:?}", &sum[..shown]);
    let launch = learning::Launch1D::new(VECTOR_LEN, 256)?;
    let padded = round_up(VECTOR_LEN, 128);
    println!(
        "  SIMT: {} blocks x 256 threads; {} excess threads skip writes.",
        launch.blocks,
        launch.excess_threads()
    );
    println!(
        "  Tile: pad to {padded} elements, then partition into {} tiles of 128.\n",
        padded / 128
    );

    let logits = softmax_inputs();
    let probabilities = softmax_rows(&logits, SOFTMAX_ROWS, SOFTMAX_COLS)?;
    check_probabilities(&probabilities, SOFTMAX_ROWS, SOFTMAX_COLS)?;
    println!("MEDIUM: normalize {SOFTMAX_ROWS} rows of {SOFTMAX_COLS} logits");
    println!(
        "  Naive exp(1000) overflows f32: {}",
        1000.0_f32.exp().is_infinite()
    );
    println!("  Subtract the row maximum before exponentiating.");
    for (row_index, row) in probabilities
        .as_chunks::<SOFTMAX_COLS>()
        .0
        .iter()
        .enumerate()
    {
        println!(
            "  Row {row_index}: sum={:.6}, first 4={:?}",
            row.iter().sum::<f32>(),
            &row[..4]
        );
    }
    println!("  SIMT: threads cooperate through shared memory and barriers.");
    println!("  Tile: reductions operate on an entire row tile.\n");

    let (a, b) = matrix_inputs();
    let product = matmul(&a, &b, MATRIX_M, MATRIX_K, MATRIX_N)?;
    println!("HARD: A[{MATRIX_M},{MATRIX_K}] x B[{MATRIX_K},{MATRIX_N}]");
    println!(
        "  C shape: [{MATRIX_M},{MATRIX_N}], first row: {:?}",
        &product[..8]
    );
    println!("  SIMT: 16x16 thread blocks load shared tiles; edge loads become zero.");
    println!(
        "  Tile: pad M/K/N to {}/{}/{}, then use 16x16 tile operations.",
        round_up(MATRIX_M, 16),
        round_up(MATRIX_K, 16),
        round_up(MATRIX_N, 16)
    );
    println!("\nNext: read lessons/01-vector-add.md, then open both matching GPU files.");
    Ok(())
}
