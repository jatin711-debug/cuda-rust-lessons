//! Problem: add arrays, with one exclusive output element per GPU thread.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use cuda_rust_lessons::*;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vector_add(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        // 1003 is not divisible by 256. The last 21 threads get None and skip.
        // Only a trusted ThreadIndex can obtain a mutable output element.
        if let Some(destination) = output.get_mut(index) {
            *destination = left[position] + right[position];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let (a, b) = vector_inputs();
    let expected = vector_add(&a, &b)?;
    let left = DeviceBuffer::from_host(&stream, &a)?;
    let right = DeviceBuffer::from_host(&stream, &b)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, VECTOR_LEN)?;

    // SAFETY: this binary owns the bundle compiled from the kernels module.
    let module = unsafe { kernels::load(&context)? };
    let geometry = LaunchConfig1D::new((VECTOR_LEN as u32).div_ceil(256), 256, 0);
    let prepared = module.prepare_vector_add(geometry)?;
    // Shared borrows for inputs, an exclusive borrow for output, and a checked
    // geometry token associated specifically with vector_add.
    module.vector_add(&stream, &prepared, &left, &right, &mut output)?;

    let actual = output.to_host_vec(&stream)?; // Copy back and synchronize.
    let error = check_close(&actual, &expected, 1e-6, 0.0)?;
    println!("SIMT EASY: checked all {VECTOR_LEN} outputs; max error {error}");
    println!("First sums: {:?}", &actual[..VECTOR_LEN.min(5)]);
    Ok(())
}
