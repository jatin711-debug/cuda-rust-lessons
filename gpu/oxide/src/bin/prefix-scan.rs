/*!
Hierarchical exclusive scan: local scans, scan block totals, propagate offsets.
*/
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::learning::*;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn local_scan(
        input: &[u32],
        mut output: DisjointSlice<u32>,
        mut totals: DisjointSlice<u32>,
    ) {
        static mut VALUES: SharedArray<u32, 128> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let position = block * 128 + lane;
        let value = if position < input.len() {
            input[position]
        } else {
            0
        };
        // SAFETY: one distinct cell per lane, all 128 lanes initialize a cell.
        let cells = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        unsafe { cells.add(lane).write(value) };
        thread::sync_threads();
        let mut offset = 1usize;
        while offset < 128 {
            // SAFETY: all reads use initialized cells, before any stage writes.
            let current = unsafe { cells.add(lane).read() };
            let previous = if lane >= offset {
                unsafe { cells.add(lane - offset).read() }
            } else {
                0
            };
            thread::sync_threads(); // Readers finish before writers overwrite.
            unsafe { cells.add(lane).write(current + previous) };
            thread::sync_threads(); // Writers finish before the next stage reads.
            offset *= 2;
        }
        let inclusive = unsafe { cells.add(lane).read() };
        if let Some(destination) = output.get_mut(thread::index_1d()) {
            *destination = inclusive - value;
        }
        if lane == 127 && block < totals.len() {
            // SAFETY: lane 127 alone writes each unique guarded block-total slot.
            unsafe { *totals.get_unchecked_mut(block) = inclusive };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn add_offsets(block_prefixes: &[u32], mut values: DisjointSlice<u32>) {
        let block = thread::blockIdx_x() as usize;
        if let Some(destination) = values.get_mut(thread::index_1d()) {
            *destination += block_prefixes[block];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ExperimentOptions::from_env()?;
    let input = scan_inputs(options.len);
    let expected = exclusive_scan_u32(&input)?; // Rejects a u32-overflowing total.
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let mut inputs = vec![DeviceBuffer::from_host(&stream, &input)?];
    let mut scans = Vec::new();
    let mut sizes = Vec::new();
    let mut len = options.len;
    loop {
        sizes.push(len);
        scans.push(DeviceBuffer::<u32>::zeroed(&stream, len)?);
        let blocks = len.div_ceil(128);
        inputs.push(DeviceBuffer::<u32>::zeroed(&stream, blocks)?);
        if blocks == 1 {
            break;
        }
        len = blocks;
    }
    // SAFETY: the loaded device artifact matches this kernels module.
    let module = unsafe { kernels::load(&context)? };
    let scan_contracts = sizes
        .iter()
        .map(|&len| {
            module.prepare_local_scan(LaunchConfig1D::new(len.div_ceil(128) as u32, 128, 0))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offset_contracts = sizes
        .iter()
        .take(sizes.len() - 1)
        .map(|&len| {
            module.prepare_add_offsets(LaunchConfig1D::new(len.div_ceil(128) as u32, 128, 0))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut enqueue = || -> Result<(), Box<dyn std::error::Error>> {
        for (level, contract) in scan_contracts.iter().enumerate() {
            let (before, after) = inputs.split_at_mut(level + 1);
            module.local_scan(
                &stream,
                contract,
                &before[level],
                &mut scans[level],
                &mut after[0],
            )?;
        }
        // A lower level needs GLOBAL prefixes of its block totals. Propagate
        // from the top down; forward propagation would leave missing offsets.
        for level in (0..offset_contracts.len()).rev() {
            let (lower, upper) = scans.split_at_mut(level + 1);
            module.add_offsets(
                &stream,
                &offset_contracts[level],
                &upper[0],
                &mut lower[level],
            )?;
        }
        Ok(())
    };
    enqueue()?;
    stream.synchronize()?;
    let start = context.new_event(Some(0))?;
    let stop = context.new_event(Some(0))?;
    start.record(&stream)?;
    for _ in 0..options.repeats {
        enqueue()?;
    }
    stop.record(&stream)?;
    stop.synchronize()?;
    let ms = start.elapsed_ms(&stop)?;
    let actual = scans[0].to_host_vec(&stream)?;
    if let Some((index, (value, reference))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        return Err(format!("scan index {index}: {value} versus {reference}").into());
    }
    if actual.len() != expected.len() {
        return Err("scan length mismatch".into());
    }
    println!(
        "PREFIX SCAN: levels {sizes:?}; checked {} u32 prefixes exactly; first {:?}",
        actual.len(),
        &actual[..actual.len().min(8)]
    );
    println!(
        "{} repetitions: {ms:.6} ms CUDA event interval; {} scan launches + {} offset launches/iteration",
        options.repeats,
        sizes.len(),
        offset_contracts.len()
    );
    Ok(())
}
