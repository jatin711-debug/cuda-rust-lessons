//! Small, dependency-free CPU references used to check both GPU implementations.
//! These functions are ordinary CPU Rust; they do not emulate either GPU library.

pub mod learning;

pub const VECTOR_LEN: usize = 1003;
pub const SOFTMAX_ROWS: usize = 4;
pub const SOFTMAX_COLS: usize = 64;
pub const MATRIX_M: usize = 35;
pub const MATRIX_K: usize = 23;
pub const MATRIX_N: usize = 19;

/// Nonconstant inputs make missing writes and mistaken indices visible.
pub fn vector_inputs() -> (Vec<f32>, Vec<f32>) {
    let a = (0..VECTOR_LEN).map(|i| i as f32 * 0.25 - 17.0).collect();
    let b = (0..VECTOR_LEN)
        .map(|i| (i % 13) as f32 * 0.5 - 3.0)
        .collect();
    (a, b)
}

pub fn vector_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err("vector inputs must have the same length".into());
    }
    Ok(a.iter().zip(b).map(|(x, y)| x + y).collect())
}

/// Rows exercise positive overflow, negative underflow, a uniform distribution,
/// and a sharply peaked distribution. All inputs are finite.
pub fn softmax_inputs() -> Vec<f32> {
    let mut input = Vec::with_capacity(SOFTMAX_ROWS * SOFTMAX_COLS);
    input.extend((0..SOFTMAX_COLS).map(|i| 1000.0 + (i % 9) as f32 * 0.25));
    input.extend((0..SOFTMAX_COLS).map(|i| -1000.0 + (i % 7) as f32 * 0.5));
    input.extend(std::iter::repeat_n(42.0, SOFTMAX_COLS));
    input.extend((0..SOFTMAX_COLS).map(|i| if i == 17 { 25.0 } else { -25.0 }));
    input
}

/// Accumulate in f64 so the reference does not repeat the GPU's f32 rounding.
pub fn softmax_rows(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    require_shape(input, rows, cols, "softmax input")?;
    if !input.iter().all(|x| x.is_finite()) {
        return Err("these softmax lessons require finite logits".into());
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(cols) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let numerators: Vec<f64> = row.iter().map(|&x| (x as f64 - maximum).exp()).collect();
        let denominator: f64 = numerators.iter().sum();
        output.extend(numerators.into_iter().map(|x| (x / denominator) as f32));
    }
    Ok(output)
}

pub fn matrix_inputs() -> (Vec<f32>, Vec<f32>) {
    // Binary fractions remain exactly representable in f32 and TF32. This
    // keeps the hard lesson focused on indexing, tiling, and boundary handling.
    let a = (0..MATRIX_M * MATRIX_K)
        .map(|i| ((i * 7 % 17) as f32 - 8.0) * 0.125)
        .collect();
    let b = (0..MATRIX_K * MATRIX_N)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.125)
        .collect();
    (a, b)
}

/// Row-major C[M,N] = A[M,K] * B[K,N], using independent f64 accumulation.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String> {
    require_shape(a, m, k, "matrix A")?;
    require_shape(b, k, n, "matrix B")?;
    let output_len = m.checked_mul(n).ok_or("output matrix size overflow")?;
    let mut output = vec![0.0; output_len];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0_f64;
            for inner in 0..k {
                sum += a[row * k + inner] as f64 * b[inner * n + col] as f64;
            }
            output[row * n + col] = sum as f32;
        }
    }
    Ok(output)
}

pub fn round_up(value: usize, tile: usize) -> usize {
    assert!(tile > 0, "tile width must be positive");
    value
        .div_ceil(tile)
        .checked_mul(tile)
        .expect("rounded size overflow")
}

/// Zero padding preserves matrix multiplication in the original output region.
pub fn pad_matrix(
    input: &[f32],
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
) -> Result<Vec<f32>, String> {
    require_shape(input, rows, cols, "matrix to pad")?;
    if padded_rows < rows || padded_cols < cols {
        return Err("padding cannot shrink a matrix".into());
    }
    let len = padded_rows
        .checked_mul(padded_cols)
        .ok_or("padded matrix size overflow")?;
    let mut output = vec![0.0; len];
    for row in 0..rows {
        output[row * padded_cols..row * padded_cols + cols]
            .copy_from_slice(&input[row * cols..(row + 1) * cols]);
    }
    Ok(output)
}

pub fn crop_matrix(
    input: &[f32],
    padded_rows: usize,
    padded_cols: usize,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    require_shape(input, padded_rows, padded_cols, "matrix to crop")?;
    if rows > padded_rows || cols > padded_cols {
        return Err("crop dimensions exceed the stored matrix".into());
    }
    let mut output = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        output.extend_from_slice(&input[row * padded_cols..row * padded_cols + cols]);
    }
    Ok(output)
}

/// A failed comparison is an error, so each GPU binary exits unsuccessfully.
/// Explicit finite checks prevent a NaN from passing an ordinary tolerance test.
/// Error arithmetic and the returned maximum use f64, so finite f32 inputs
/// cannot overflow the comparison itself.
pub fn check_close(
    actual: &[f32],
    expected: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: {} versus {}",
            actual.len(),
            expected.len()
        ));
    }
    if !absolute_tolerance.is_finite()
        || !relative_tolerance.is_finite()
        || absolute_tolerance < 0.0
        || relative_tolerance < 0.0
    {
        return Err("comparison tolerances must be finite and nonnegative".into());
    }
    let mut largest_error = 0.0_f64;
    for (index, (&value, &reference)) in actual.iter().zip(expected).enumerate() {
        if !value.is_finite() || !reference.is_finite() {
            return Err(format!(
                "non-finite value at index {index}: {value} versus {reference}"
            ));
        }
        let error = (f64::from(value) - f64::from(reference)).abs();
        let allowed = f64::from(absolute_tolerance)
            + f64::from(relative_tolerance) * f64::from(reference).abs();
        if error > allowed {
            return Err(format!(
                "index {index}: GPU={value}, CPU={reference}, error={error}, allowed={allowed}"
            ));
        }
        largest_error = largest_error.max(error);
    }
    Ok(largest_error)
}

pub fn check_probabilities(output: &[f32], rows: usize, cols: usize) -> Result<(), String> {
    require_shape(output, rows, cols, "softmax output")?;
    for (index, row) in output.chunks_exact(cols).enumerate() {
        if row.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0) {
            return Err(format!("row {index} contains invalid probabilities"));
        }
        let sum: f64 = row.iter().map(|&x| x as f64).sum();
        if (sum - 1.0).abs() > 1e-5 {
            return Err(format!("row {index} sums to {sum}, rather than 1"));
        }
    }
    Ok(())
}

fn require_shape(input: &[f32], rows: usize, cols: usize, label: &str) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err(format!("{label} dimensions must be positive"));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{label} size overflow"))?;
    if input.len() != len {
        return Err(format!("{label} needs {len} elements, got {}", input.len()));
    }
    Ok(())
}
