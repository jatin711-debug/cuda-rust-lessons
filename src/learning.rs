//! CPU mathematics and work-assignment models for the extended lessons.
//! A work-assignment model illustrates indices; it does not emulate CUDA.

/// A one-dimensional grid with enough threads to cover a logical vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch1D {
    pub len: usize,
    pub blocks: usize,
    pub threads_per_block: usize,
    pub launched_threads: usize,
}

impl Launch1D {
    pub fn new(len: usize, threads_per_block: usize) -> Result<Self, String> {
        if threads_per_block == 0 || threads_per_block > 1024 {
            return Err("this indexing lab requires 1..=1024 threads per block".into());
        }
        let blocks = len.div_ceil(threads_per_block);
        let launched_threads = blocks
            .checked_mul(threads_per_block)
            .ok_or("launch size overflow")?;
        Ok(Self {
            len,
            blocks,
            threads_per_block,
            launched_threads,
        })
    }

    pub fn excess_threads(self) -> usize {
        self.launched_threads - self.len
    }

    /// Return the element owned by a worker, or None for an excess/invalid worker.
    pub fn element(self, block: usize, thread: usize) -> Option<usize> {
        if block >= self.blocks || thread >= self.threads_per_block {
            return None;
        }
        let index = block * self.threads_per_block + thread;
        (index < self.len).then_some(index)
    }
}

/// Small binary fractions keep the first numerical experiments easy to check.
pub fn pointwise_inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..len).map(|i| ((i % 31) as f32 - 15.0) * 0.25).collect();
    let y = (0..len).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    (x, y)
}

/// SAXPY: single-precision a*x + y. The CPU reference accumulates in f64.
pub fn saxpy(alpha: f32, x: &[f32], y: &[f32]) -> Result<Vec<f32>, String> {
    if x.len() != y.len() {
        return Err("SAXPY inputs must have equal lengths".into());
    }
    if !alpha.is_finite() || x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err("SAXPY lessons require finite inputs and a finite scalar".into());
    }
    Ok(x.iter()
        .zip(y)
        .map(|(&x, &y)| (f64::from(alpha) * f64::from(x) + f64::from(y)) as f32)
        .collect())
}

/// One independent reference sum per chunk, including the final partial chunk.
pub fn block_sums(input: &[f32], width: usize) -> Result<Vec<f32>, String> {
    if width == 0 {
        return Err("reduction width must be positive".into());
    }
    if input.iter().any(|v| !v.is_finite()) {
        return Err("reduction lessons require finite inputs".into());
    }
    Ok(input
        .chunks(width)
        .map(|chunk| chunk.iter().map(|&v| f64::from(v)).sum::<f64>() as f32)
        .collect())
}

/// Row-major transpose: input[r,c] becomes output[c,r].
pub fn transpose(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    super::require_shape(input, rows, cols, "transpose input")?;
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    Ok(output)
}

/// Shared bounded CLI for GPU SAXPY and pipeline experiments.
/// Bounds keep accidental allocations/repetition manageable on a laptop.
#[derive(Debug, Clone, Copy)]
pub struct ExperimentOptions {
    pub len: usize,
    pub repeats: usize,
}

impl ExperimentOptions {
    pub fn from_env() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() > 2 || args.iter().any(|arg| arg == "--help" || arg == "-h") {
            return Err("usage: <binary> [length: 1..=1048576] [repeats: 1..=10000]".into());
        }
        fn number(
            arg: Option<&String>,
            default: usize,
            maximum: usize,
            label: &str,
        ) -> Result<usize, String> {
            let value = match arg {
                Some(arg) => arg
                    .parse::<usize>()
                    .map_err(|_| format!("{label} must be an integer"))?,
                None => default,
            };
            if value == 0 || value > maximum {
                return Err(format!("{label} must be in 1..={maximum}"));
            }
            Ok(value)
        }
        Ok(Self {
            len: number(args.first(), super::VECTOR_LEN, 1_048_576, "length")?,
            repeats: number(args.get(1), 1, 10_000, "repeats")?,
        })
    }
}
