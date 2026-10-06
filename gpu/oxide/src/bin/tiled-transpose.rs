//! Compare a strided-read baseline with a 32x33 shared-memory transpose.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::{RowWidth, cuda_module};
use cuda_rust_lessons::{check_close, learning::transpose};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn baseline(
        input: &[f32],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        rows: u32,
        cols: u32,
    ) {
        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        let out_col = thread::blockIdx_x() as usize * 32 + x;
        let row_base = thread::blockIdx_y() as usize * 32 + y;
        let mut offset = 0usize;
        while offset < 32 {
            let out_row = row_base + offset;
            if out_row < cols as usize && out_col < rows as usize {
                // SAFETY: each block owns one 32x32 output region. Within it,
                // (x,y,offset=0/8/16/24) identifies one unique guarded cell.
                unsafe {
                    *output.get_unchecked_mut(out_row * rows as usize + out_col) =
                        input[out_col * cols as usize + out_row];
                }
            }
            offset += 8;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (32, 8, 1))]
    pub fn tiled(
        input: &[f32],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        rows: u32,
        cols: u32,
    ) {
        static mut TILE: SharedArray<f32, 1056> = SharedArray::UNINIT; // 32*33
        let x = thread::threadIdx_x() as usize;
        let y = thread::threadIdx_y() as usize;
        // Launch coordinates describe OUTPUT tiles, so input tile axes swap.
        let input_row_base = thread::blockIdx_x() as usize * 32;
        let input_col_base = thread::blockIdx_y() as usize * 32;
        // SAFETY: 256 lanes each initialize four distinct cells. Column 32 is
        // padding and is never read. Valid cell offsets are within 32*33.
        let cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let mut offset = 0usize;
        while offset < 32 {
            let input_row = input_row_base + y + offset;
            let input_col = input_col_base + x;
            let value = if input_row < rows as usize && input_col < cols as usize {
                input[input_row * cols as usize + input_col]
            } else {
                0.0
            };
            unsafe { cells.add((y + offset) * 33 + x).write(value) };
            offset += 8;
        }
        thread::sync_threads(); // All cells are initialized before transposed reads.
        let out_col = input_row_base + x;
        offset = 0;
        while offset < 32 {
            let out_row = input_col_base + y + offset;
            if out_row < cols as usize && out_col < rows as usize {
                // SAFETY: transposed shared reads access initialized cells.
                // Output ownership is the same bijection as in baseline.
                unsafe {
                    *output.get_unchecked_mut(out_row * rows as usize + out_col) =
                        cells.add(x * 33 + y + offset).read();
                }
            }
            offset += 8;
        }
        // No second barrier: shared storage is not overwritten within this launch.
    }
}

fn number(args: &[String], index: usize, default: usize, maximum: usize) -> Result<usize, String> {
    let n = match args.get(index) {
        Some(arg) => arg
            .parse::<usize>()
            .map_err(|_| "arguments must be integers")?,
        None => default,
    };
    if n == 0 || n > maximum {
        return Err(format!("argument must be in 1..={maximum}"));
    }
    Ok(n)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 3 {
        return Err("usage: tiled-transpose [rows] [cols] [repeats]".into());
    }
    let rows = number(&args, 0, 35, 4096)?;
    let cols = number(&args, 1, 19, 4096)?;
    let repeats = number(&args, 2, 100, 10_000)?;
    let len = rows
        .checked_mul(cols)
        .filter(|&n| n <= 4_194_304)
        .ok_or("matrix permits at most 4194304 elements")?;
    let input: Vec<f32> = (0..len).map(|i| (i % 1021) as f32 * 0.25).collect();
    let expected = transpose(&input, rows, cols)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let input_device = DeviceBuffer::from_host(&stream, &input)?;
    let mut baseline_output = DeviceBuffer::<f32>::zeroed(&stream, len)?;
    let mut tiled_output = DeviceBuffer::<f32>::zeroed(&stream, len)?;
    // SAFETY: the artifact is built from this exact kernels module.
    let module = unsafe { kernels::load(&context)? };
    let geometry = LaunchConfig2D::new(
        ((rows as u32).div_ceil(32), (cols as u32).div_ceil(32)),
        (32, 8),
        0,
    );
    let baseline_contract = module.prepare_baseline(geometry)?;
    let tiled_contract = module.prepare_tiled(geometry)?;
    module.baseline(
        &stream,
        &baseline_contract,
        &input_device,
        RowWidth::new(&mut baseline_output, rows as u32),
        rows as u32,
        cols as u32,
    )?;
    module.tiled(
        &stream,
        &tiled_contract,
        &input_device,
        RowWidth::new(&mut tiled_output, rows as u32),
        rows as u32,
        cols as u32,
    )?;
    stream.synchronize()?;
    let start = context.new_event(Some(0))?;
    let stop = context.new_event(Some(0))?;
    start.record(&stream)?;
    for _ in 0..repeats {
        module.baseline(
            &stream,
            &baseline_contract,
            &input_device,
            RowWidth::new(&mut baseline_output, rows as u32),
            rows as u32,
            cols as u32,
        )?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let baseline_ms = start.elapsed_ms(&stop)?;
    start.record(&stream)?;
    for _ in 0..repeats {
        module.tiled(
            &stream,
            &tiled_contract,
            &input_device,
            RowWidth::new(&mut tiled_output, rows as u32),
            rows as u32,
            cols as u32,
        )?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let tiled_ms = start.elapsed_ms(&stop)?;
    let baseline_actual = baseline_output.to_host_vec(&stream)?;
    let tiled_actual = tiled_output.to_host_vec(&stream)?;
    check_close(&baseline_actual, &expected, 0.0, 0.0)?;
    check_close(&tiled_actual, &expected, 0.0, 0.0)?;
    println!(
        "TILED TRANSPOSE: [{rows},{cols}] -> [{cols},{rows}]; both variants checked all {len} values exactly"
    );
    println!(
        "{repeats} repetitions, CUDA event interval: baseline {baseline_ms:.6} ms, tiled {tiled_ms:.6} ms"
    );
    println!(
        "Identical geometry: 32x8 threads, four outputs/thread; tiled shared memory: 4224 bytes/block."
    );
    println!(
        "Intervals include gaps between submitted kernels; they are not peak-bandwidth measurements."
    );
    Ok(())
}
