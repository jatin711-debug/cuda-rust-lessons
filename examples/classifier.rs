//! CPU capstone: two dense layers with ReLU and stable softmax.
//! These are hand-selected weights for tracing arithmetic, not a trained model.
use cuda_rust_lessons::{check_close, check_probabilities, matmul, softmax_rows};

fn add_bias(values: &mut [f32], cols: usize, bias: &[f32]) {
    assert_eq!(cols, bias.len());
    for row in values.chunks_exact_mut(cols) {
        for (value, bias) in row.iter_mut().zip(bias) {
            *value += bias;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = [1.0, 2.0, -1.0, 0.5, 0.0, 1.0]; // [batch=2, features=3]
    let first_weights = [1.0, -1.0, 0.5, 0.0, 0.0, 1.0]; // [3, hidden=2]
    let second_weights = [1.0, 0.0, -1.0, 0.0, 1.0, 0.5]; // [2, classes=3]
    let mut hidden = matmul(&inputs, &first_weights, 2, 3, 2)?;
    add_bias(&mut hidden, 2, &[0.0, 0.5]);
    for value in &mut hidden {
        *value = value.max(0.0); // ReLU
    }
    check_close(&hidden, &[2.0, 0.0, 0.5, 1.0], 0.0, 0.0)?;
    let mut logits = matmul(&hidden, &second_weights, 2, 2, 3)?;
    add_bias(&mut logits, 3, &[0.0, 0.25, -0.25]);
    check_close(&logits, &[2.0, 0.25, -2.25, 0.5, 1.25, -0.25], 0.0, 0.0)?;
    let probabilities = softmax_rows(&logits, 2, 3)?;
    check_probabilities(&probabilities, 2, 3)?;
    println!("CPU capstone with hand-selected weights (no GPU execution):");
    println!("X[2,3] -> dense[2,2] -> bias + ReLU -> dense[2,3] -> bias -> softmax");
    println!("hidden: {hidden:?}\nlogits: {logits:?}");
    for (sample, row) in probabilities.as_chunks::<3>().0.iter().enumerate() {
        let class = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        println!("sample {sample}: probabilities {row:?}, selected class {class}");
    }
    println!("See lessons/08-capstone.md to trace the math, then lessons/12-gpu-classifier.md for complete GPU inference.");
    Ok(())
}
