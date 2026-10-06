//! Run `cargo run --example lab -- help` for experiments that need no GPU.
use cuda_rust_lessons::{check_close, learning::*, round_up};

fn dimension(args: &[String], index: usize, default: usize) -> Result<usize, String> {
    let value = args.get(index).map_or(Ok(default), |arg| {
        arg.parse::<usize>()
            .map_err(|_| format!("'{arg}' is not a positive integer"))
    })?;
    if value == 0 || value > 1_048_576 {
        return Err("lab dimensions must be in 1..=1048576".into());
    }
    Ok(value)
}

fn preview(values: &[f32]) -> &[f32] {
    &values[..values.len().min(8)]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map_or("help", String::as_str);
    if matches!(command, "help" | "--help" | "-h") {
        println!(
            "CPU learning labs (no CUDA execution):\n\
            cargo run --example lab -- launch [length=1003] [block=256]\n\
            cargo run --example lab -- saxpy [length=1003]\n\
            cargo run --example lab -- reduction [length=1003] [width=128]\n\
            cargo run --example lab -- transpose [rows=3] [cols=5]\n\
            cargo run --example lab -- pipeline [length=1003]"
        );
        return Ok(());
    }
    let maximum_args = if matches!(command, "launch" | "reduction" | "transpose") {
        3
    } else {
        2
    };
    if args.len() > maximum_args {
        return Err("too many arguments; run the lab with 'help'".into());
    }
    match command {
        "launch" => {
            let len = dimension(&args, 1, 1003)?;
            let width = dimension(&args, 2, 256)?;
            let launch = Launch1D::new(len, width)?;
            println!("CPU model: index = block * {width} + thread");
            println!(
                "{len} elements -> {} blocks -> {} workers -> {} excess workers",
                launch.blocks,
                launch.launched_threads,
                launch.excess_threads()
            );
            for block in 0..launch.blocks.min(4) {
                let first = block * width;
                let end = ((block + 1) * width).min(len);
                println!("block {block}: elements {first}..{end} (end excluded)");
            }
            let last = launch.blocks - 1;
            for thread in [0, width - 1] {
                println!(
                    "last block, thread {thread}: {:?}",
                    launch.element(last, thread)
                );
            }
            println!("Try lengths 255, 256, 257. Predict the grid before running.");
        }
        "saxpy" => {
            let len = dimension(&args, 1, 1003)?;
            let (x, y) = pointwise_inputs(len);
            let output = saxpy(2.0, &x, &y)?;
            println!("CPU reference: z[i] = 2*x[i] + y[i]");
            println!(
                "x: {:?}\ny: {:?}\nz: {:?}",
                preview(&x),
                preview(&y),
                preview(&output)
            );
            println!("Separate multiply/add kernels move about {} bytes; a fused SAXPY moves about {} bytes (ideal f32 array traffic).", 20 * len, 12 * len);
            println!("This counts array reads/writes, not measured DRAM traffic or GPU time.");
        }
        "reduction" => {
            let len = dimension(&args, 1, 1003)?;
            let width = dimension(&args, 2, 128)?;
            if !width.is_power_of_two() || width > 1024 {
                return Err("the tree lab requires a power-of-two width up to 1024".into());
            }
            let (input, _) = pointwise_inputs(len);
            let partials = block_sums(&input, width)?;
            let total: f64 = input.iter().map(|&v| f64::from(v)).sum();
            println!(
                "{len} inputs -> {} partial sums -> one CPU final sum {total}",
                partials.len()
            );
            println!("partials: {:?}", preview(&partials));
            let mut shared = vec![0.0; width];
            let live = len.min(width);
            shared[..live].copy_from_slice(&input[..live]);
            let mut offset = width / 2;
            while offset > 0 {
                for lane in 0..offset {
                    shared[lane] += shared[lane + offset];
                }
                println!(
                    "CPU tree stage: {offset} writers, barrier, prefix {:?}",
                    preview(&shared[..offset])
                );
                offset /= 2;
            }
            check_close(&shared[..1], &partials[..1], 1e-5, 1e-5)?;
            println!(
                "GPU blocks need neutral zeros for the tail and every thread at every barrier."
            );
        }
        "transpose" => {
            let rows = dimension(&args, 1, 3)?;
            let cols = dimension(&args, 2, 5)?;
            let len = rows
                .checked_mul(cols)
                .filter(|&len| len <= 1_048_576)
                .ok_or("transpose lab permits at most 1048576 elements")?;
            let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let output = transpose(&input, rows, cols)?;
            println!("CPU transpose: [{rows},{cols}] -> [{cols},{rows}]");
            for row in 0..rows.min(4) {
                println!(
                    "input row {row}: {:?}",
                    preview(&input[row * cols..(row + 1) * cols])
                );
            }
            for row in 0..cols.min(4) {
                println!(
                    "output row {row}: {:?}",
                    preview(&output[row * rows..(row + 1) * rows])
                );
            }
            println!(
                "Consecutive input columns become output addresses separated by {rows} elements."
            );
            println!(
                "16-wide GPU tiles would pad the shape to [{},{}].",
                round_up(rows, 16),
                round_up(cols, 16)
            );
        }
        "pipeline" => {
            let len = dimension(&args, 1, 1003)?;
            let (x, y) = pointwise_inputs(len);
            let intermediate = saxpy(2.0, &x, &y)?;
            let output = saxpy(0.5, &intermediate, &y)?;
            println!("stage 1: intermediate = 2*x + y\nstage 2: output = 0.5*intermediate + y");
            println!(
                "intermediate: {:?}\noutput: {:?}",
                preview(&intermediate),
                preview(&output)
            );
            println!("GPU versions retain the intermediate on the device and use one stream.");
            println!("One final readback supplies the CPU verification data.");
        }
        _ => return Err(format!("unknown lab '{command}'; run with 'help'").into()),
    }
    Ok(())
}
