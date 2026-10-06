//! Baseline transpose: contiguous output stores, strided input reads.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::{RowWidth, cuda_module};
use cuda_rust_lessons::{MATRIX_M, MATRIX_N, check_close, learning::transpose};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn transpose_matrix(
        input: &[f32],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        input_rows: u32,
        input_cols: u32,
    ) {
        // Launch over OUTPUT coordinates. Output row is input column;
        // output column is input row. Adjacent X threads store adjacent cells.
        let out_row = thread::index_2d_row();
        let out_col = thread::index_2d_col();
        if out_row < input_cols as usize && out_col < input_rows as usize {
            if let Some(index) = thread::index_2d_runtime(&output) {
                if let Some(destination) = output.get_mut(index) {
                    *destination = input[out_col * input_cols as usize + out_row];
                }
            }
        }
        // The host-bound output row width equals input_rows. Safe indexing
        // rejects out_col >= input_rows before any input read.
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = (MATRIX_M, MATRIX_N);
    let input: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25).collect();
    let expected = transpose(&input, rows, cols)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let input_device = DeviceBuffer::from_host(&stream, &input)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    // SAFETY: the loaded artifact is generated from this module.
    let module = unsafe { kernels::load(&context)? };
    let geometry = LaunchConfig2D::new(
        ((rows as u32).div_ceil(16), (cols as u32).div_ceil(16)),
        (16, 16),
        0,
    );
    let prepared = module.prepare_transpose_matrix(geometry)?;
    module.transpose_matrix(
        &stream,
        &prepared,
        &input_device,
        RowWidth::new(&mut output, rows as u32),
        rows as u32,
        cols as u32,
    )?;
    let actual = output.to_host_vec(&stream)?;
    let error = check_close(&actual, &expected, 0.0, 0.0)?;
    println!(
        "SIMT TRANSPOSE: [{rows},{cols}] -> [{cols},{rows}], checked {} outputs; max error {error}",
        actual.len()
    );
    println!("Baseline uses contiguous stores and strided reads; see lesson 06.");
    Ok(())
}
