//! Problem: reuse matrix tiles in shared memory, including all three edge axes.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::{RowWidth, cuda_module};
use cuda_rust_lessons::*;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (16, 16, 1))]
    pub fn tiled_matmul(
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let (a, b) = matrix_inputs();
    let expected = matmul(&a, &b, MATRIX_M, MATRIX_K, MATRIX_N)?;
    let left = DeviceBuffer::from_host(&stream, &a)?;
    let right = DeviceBuffer::from_host(&stream, &b)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, MATRIX_M * MATRIX_N)?;
    // SAFETY: this binary embeds its own matching device artifact.
    let module = unsafe { kernels::load(&context)? };
    let geometry = LaunchConfig2D::new(
        (
            (MATRIX_N as u32).div_ceil(16),
            (MATRIX_M as u32).div_ceil(16),
        ),
        (16, 16),
        0, // Both shared arrays have compile-time sizes; no dynamic allocation.
    );
    let prepared = module.prepare_tiled_matmul(geometry)?;
    module.tiled_matmul(
        &stream,
        &prepared,
        &left,
        &right,
        // Bind output's row width once on the host. Device-side safe indexing
        // must use the same width in every thread to keep writes disjoint.
        RowWidth::new(&mut output, MATRIX_N as u32),
        MATRIX_M as u32,
        MATRIX_K as u32,
        MATRIX_N as u32,
    )?;

    let actual = output.to_host_vec(&stream)?;
    let error = check_close(&actual, &expected, 1e-4, 1e-4)?;
    println!(
        "SIMT HARD: checked all {} outputs; max error {error}",
        actual.len()
    );
    println!("M/K/N = {MATRIX_M}/{MATRIX_K}/{MATRIX_N}; 3x2 blocks, each 16x16 threads");
    println!("First eight outputs: {:?}", &actual[..8]);
    Ok(())
}
