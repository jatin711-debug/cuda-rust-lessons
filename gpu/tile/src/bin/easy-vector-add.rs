//! Problem: add two independent arrays. Read lessons/01-vector-add.md first.
use cuda_rust_lessons::*;
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn vector_add<const WIDTH: i32>(
        output: &mut Tensor<f32, { [WIDTH] }>,
        left: &Tensor<f32, { [-1] }>,
        right: &Tensor<f32, { [-1] }>,
    ) {
        // The launcher gives this program one exclusive WIDTH-element output.
        // Input loads use that same region. No individual thread indices appear.
        let left_values = left.load_like(output);
        let right_values = right.load_like(output);
        output.store(left_values + right_values);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let (mut a, mut b) = vector_inputs();
    let mut expected = vector_add(&a, &b)?;

    // Keep this first lesson on complete tiles. Only 1003 elements are real;
    // the remaining 21 elements are zero. This padding is on the CPU.
    let padded_len = round_up(VECTOR_LEN, 128);
    a.resize(padded_len, 0.0);
    b.resize(padded_len, 0.0);
    expected.resize(padded_len, 0.0);
    let a = Arc::new(a);
    let b = Arc::new(b);
    let left = api::copy_host_vec_to_device(&a);
    let right = api::copy_host_vec_to_device(&b);
    let output = api::zeros::<f32>(&[padded_len]).partition([128]);

    // Each value above is a lazy operation. The generated launcher moves these
    // operations into a chain, which owns their inputs until completion.
    let actual: Vec<f32> = kernels::vector_add(output, left, right)
        .first() // Retrieve output from (output, left, right).
        .unpartition() // Metadata change; no GPU copy.
        .to_host_vec() // Record the copy back.
        .sync_on(&stream)?; // Execute, synchronize, then give the Vec to the CPU.

    let error = check_close(&actual, &expected, 1e-6, 0.0)?;
    println!(
        "Tile EASY: checked {VECTOR_LEN} real + {} padded elements; max error {error}",
        padded_len - VECTOR_LEN
    );
    println!("First sums: {:?}", &actual[..VECTOR_LEN.min(5)]);
    Ok(())
}
