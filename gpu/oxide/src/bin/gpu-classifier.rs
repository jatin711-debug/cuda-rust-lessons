//! Complete device inference: tiled dense -> bias/ReLU -> dense -> bias -> softmax.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::{RowWidth, cuda_module};
use cuda_rust_lessons::{check_close, check_probabilities, learning::*};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn dense(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        m: u32,
        k: u32,
        n: u32,
    ) {
        static mut LEFT_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut RIGHT_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
        let local_col = thread::threadIdx_x() as usize;
        let local_row = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * 16 + local_row;
        let col = thread::blockIdx_x() as usize * 16 + local_col;
        let slot = local_row * 16 + local_col;

        // SAFETY: the 16x16 launch contract makes slots unique and in 0..256.
        // Shared pointers stay within this block, and barriers separate writes
        // from cooperative reads. No whole-array mutable references are formed.
        let tile_a = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT_TILE) };
        let tile_b = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT_TILE) };
        let mut accumulator = 0.0f32;
        let mut inner_start = 0usize;
        while inner_start < k as usize {
            let a_col = inner_start + local_col;
            let b_row = inner_start + local_row;
            let a = if row < m as usize && a_col < k as usize {
                left[row * k as usize + a_col]
            } else {
                0.0
            };
            let b = if b_row < k as usize && col < n as usize {
                right[b_row * n as usize + col]
            } else {
                0.0
            };
            unsafe {
                tile_a.add(slot).write(a);
                tile_b.add(slot).write(b);
            }
            thread::sync_threads(); // Entire A/B tile is ready to read.

            let mut inner = 0usize;
            while inner < 16 {
                unsafe {
                    accumulator += tile_a.add(local_row * 16 + inner).read()
                        * tile_b.add(inner * 16 + local_col).read();
                }
                inner += 1;
            }
            thread::sync_threads(); // Readers finish before the next overwrite.
            inner_start += 16;
        }

        // Edge threads participated in every barrier. They skip only the final
        // store. Returning early above could deadlock their block.
        if row < m as usize {
            if let Some(index) = thread::index_2d_runtime(&output) {
                if let Some(destination) = output.get_mut(index) {
                    *destination = accumulator;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn bias_activation(
        values: &[f32],
        bias: &[f32],
        mut output: DisjointSlice<f32>,
        columns: u32,
        relu: u32,
    ) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(destination) = output.get_mut(index) {
            let value = values[position] + bias[position % columns as usize];
            *destination = if relu != 0 { value.max(0.0) } else { value };
        }
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 2, block = (32, 1, 1))]
    pub fn row_softmax(
        logits: &[f32],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        classes: u32,
    ) {
        static mut MAXIMA: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_y() as usize;
        let value = if lane < classes as usize {
            logits[row * classes as usize + lane]
        } else {
            f32::NEG_INFINITY
        };
        // SAFETY: the launch has exactly 32 lanes; each initializes its own cell.
        // Inactive class lanes still join all barriers with neutral contributions.
        let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe { maxima.add(lane).write(value) };
        thread::sync_threads();
        let mut offset = 16usize;
        while offset > 0 {
            if lane < offset {
                // SAFETY: distinct lower-half writers read an unchanged upper
                // half; barriers order the stages.
                unsafe {
                    maxima.add(lane).write(
                        maxima
                            .add(lane)
                            .read()
                            .max(maxima.add(lane + offset).read()),
                    )
                };
            }
            thread::sync_threads();
            offset /= 2;
        }
        let maximum = unsafe { maxima.read() };
        let numerator = if lane < classes as usize {
            (value - maximum).exp()
        } else {
            0.0
        };
        unsafe { sums.add(lane).write(numerator) };
        thread::sync_threads();
        offset = 16;
        while offset > 0 {
            if lane < offset {
                unsafe {
                    sums.add(lane)
                        .write(sums.add(lane).read() + sums.add(lane + offset).read())
                };
            }
            thread::sync_threads();
            offset /= 2;
        }
        let denominator = unsafe { sums.read() };
        // Grid=(1,batch), block=(32,1). RowWidth binds the actual 19 columns,
        // so the 13 inactive class lanes cannot obtain an output cell.
        if let Some(index) = thread::index_2d_runtime(&output) {
            if let Some(destination) = output.get_mut(index) {
                *destination = numerator / denominator;
            }
        }
    }
}

fn number(args: &[String], index: usize, default: usize, maximum: usize) -> Result<usize, String> {
    let value = match args.get(index) {
        Some(arg) => arg
            .parse::<usize>()
            .map_err(|_| "classifier arguments must be integers")?,
        None => default,
    };
    if value == 0 || value > maximum {
        return Err(format!("argument must be in 1..={maximum}"));
    }
    Ok(value)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: gpu-classifier [batch:1..=1024] [repeats:1..=10000]".into());
    }
    let batch = number(&args, 0, 17, 1024)?;
    let repeats = number(&args, 1, 1, 10_000)?;
    let problem = classifier_problem(batch)?;
    let expected = classifier_reference(&problem)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let x = DeviceBuffer::from_host(&stream, &problem.input)?;
    let w1 = DeviceBuffer::from_host(&stream, &problem.first_weights)?;
    let b1 = DeviceBuffer::from_host(&stream, &problem.first_bias)?;
    let w2 = DeviceBuffer::from_host(&stream, &problem.second_weights)?;
    let b2 = DeviceBuffer::from_host(&stream, &problem.second_bias)?;
    let hidden_len = batch * CLASSIFIER_HIDDEN;
    let logits_len = batch * CLASSIFIER_CLASSES;
    let mut first_raw = DeviceBuffer::<f32>::zeroed(&stream, hidden_len)?;
    let mut hidden = DeviceBuffer::<f32>::zeroed(&stream, hidden_len)?;
    let mut second_raw = DeviceBuffer::<f32>::zeroed(&stream, logits_len)?;
    let mut logits = DeviceBuffer::<f32>::zeroed(&stream, logits_len)?;
    let mut probabilities = DeviceBuffer::<f32>::zeroed(&stream, logits_len)?;
    // SAFETY: module and compiled artifact are produced together for this binary.
    let module = unsafe { kernels::load(&context)? };
    let first_dense = module.prepare_dense(LaunchConfig2D::new(
        (
            (CLASSIFIER_HIDDEN as u32).div_ceil(16),
            (batch as u32).div_ceil(16),
        ),
        (16, 16),
        0,
    ))?;
    let second_dense = module.prepare_dense(LaunchConfig2D::new(
        (
            (CLASSIFIER_CLASSES as u32).div_ceil(16),
            (batch as u32).div_ceil(16),
        ),
        (16, 16),
        0,
    ))?;
    let hidden_bias = module.prepare_bias_activation(LaunchConfig1D::new(
        (hidden_len as u32).div_ceil(128),
        128,
        0,
    ))?;
    let logits_bias = module.prepare_bias_activation(LaunchConfig1D::new(
        (logits_len as u32).div_ceil(128),
        128,
        0,
    ))?;
    let softmax = module.prepare_row_softmax(LaunchConfig2D::new((1, batch as u32), (32, 1), 0))?;
    let mut enqueue = || -> Result<(), Box<dyn std::error::Error>> {
        module.dense(
            &stream,
            &first_dense,
            &x,
            &w1,
            RowWidth::new(&mut first_raw, CLASSIFIER_HIDDEN as u32),
            batch as u32,
            CLASSIFIER_FEATURES as u32,
            CLASSIFIER_HIDDEN as u32,
        )?;
        module.bias_activation(
            &stream,
            &hidden_bias,
            &first_raw,
            &b1,
            &mut hidden,
            CLASSIFIER_HIDDEN as u32,
            1,
        )?;
        module.dense(
            &stream,
            &second_dense,
            &hidden,
            &w2,
            RowWidth::new(&mut second_raw, CLASSIFIER_CLASSES as u32),
            batch as u32,
            CLASSIFIER_HIDDEN as u32,
            CLASSIFIER_CLASSES as u32,
        )?;
        module.bias_activation(
            &stream,
            &logits_bias,
            &second_raw,
            &b2,
            &mut logits,
            CLASSIFIER_CLASSES as u32,
            0,
        )?;
        module.row_softmax(
            &stream,
            &softmax,
            &logits,
            RowWidth::new(&mut probabilities, CLASSIFIER_CLASSES as u32),
            CLASSIFIER_CLASSES as u32,
        )?;
        Ok(())
    };
    enqueue()?;
    stream.synchronize()?;
    let start = context.new_event(Some(0))?;
    let stop = context.new_event(Some(0))?;
    start.record(&stream)?;
    for _ in 0..repeats {
        enqueue()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let ms = start.elapsed_ms(&stop)?;
    // Diagnostic readbacks happen AFTER timing. They check each layer, not just
    // whether the final row sums happen to be one.
    let actual_hidden = hidden.to_host_vec(&stream)?;
    let actual_logits = logits.to_host_vec(&stream)?;
    let actual = probabilities.to_host_vec(&stream)?;
    let hidden_error = check_close(&actual_hidden, &expected.hidden, 1e-5, 1e-5)?;
    let logits_error = check_close(&actual_logits, &expected.logits, 1e-5, 1e-5)?;
    let probability_error = check_close(&actual, &expected.probabilities, 2e-6, 2e-5)?;
    check_probabilities(&actual, batch, CLASSIFIER_CLASSES)?;
    println!(
        "GPU CLASSIFIER: batch {batch}, features 23, hidden 35, classes 19; checked every hidden/logit/probability value"
    );
    println!(
        "Max errors: hidden {hidden_error}, logits {logits_error}, probabilities {probability_error}"
    );
    println!(
        "{repeats} repetitions: {ms:.6} ms CUDA event interval; five ordered kernels/iteration, resident weights/intermediates"
    );
    println!(
        "First probabilities: {:?}; deterministic weights, not a trained model",
        &actual[..5]
    );
    Ok(())
}
