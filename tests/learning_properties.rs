use cuda_rust_lessons::{check_close, learning::*};

#[test]
fn grids_cover_each_element_once_at_block_boundaries() {
    for len in [0, 1, 31, 32, 127, 128, 129, 255, 256, 257, 1003] {
        for width in [1, 32, 128, 256] {
            let launch = Launch1D::new(len, width).unwrap();
            let elements: Vec<_> = (0..launch.blocks)
                .flat_map(|block| {
                    (0..width).filter_map(move |thread| launch.element(block, thread))
                })
                .collect();
            assert_eq!(elements, (0..len).collect::<Vec<_>>());
            assert_eq!(launch.launched_threads, len + launch.excess_threads());
            assert_eq!(launch.element(launch.blocks, 0), None);
        }
    }
    assert!(Launch1D::new(1, 0).is_err());
    assert!(Launch1D::new(1, 1025).is_err());
    assert!(Launch1D::new(usize::MAX, 256).is_err());
}

#[test]
fn saxpy_and_two_stage_pipeline_match_known_answers() {
    let x = [1.0, -2.0, 0.5];
    let y = [3.0, 4.0, -1.0];
    let intermediate = saxpy(2.0, &x, &y).unwrap();
    assert_eq!(intermediate, [5.0, 0.0, 0.0]);
    assert_eq!(saxpy(0.5, &intermediate, &y).unwrap(), [5.5, 4.0, -1.0]);
    assert!(saxpy(1.0, &x, &y[..2]).is_err());
    assert!(saxpy(f32::NAN, &x, &y).is_err());
}

#[test]
fn reduction_preserves_the_tail_and_rejects_invalid_inputs() {
    assert_eq!(
        block_sums(&[1.0, 2.0, 3.0, 4.0, 5.0], 2).unwrap(),
        [3.0, 7.0, 5.0]
    );
    assert_eq!(block_sums(&[], 128).unwrap(), Vec::<f32>::new());
    assert!(block_sums(&[1.0], 0).is_err());
    assert!(block_sums(&[f32::INFINITY], 1).is_err());
}

#[test]
fn transpose_has_known_layout_and_is_its_own_inverse() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transposed = transpose(&input, 2, 3).unwrap();
    assert_eq!(transposed, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(transpose(&transposed, 3, 2).unwrap(), input);
    assert!(transpose(&input, 3, 3).is_err());
}

#[test]
fn comparison_cannot_accept_overflow_as_infinity_equal_to_infinity() {
    // True error is 2*MAX, but allowed is MAX+1.5*MAX = 2.5*MAX:
    // the comparison may pass, while the returned maximum must remain finite.
    let error = check_close(&[f32::MAX], &[-f32::MAX], f32::MAX, 1.5).unwrap();
    assert!(error.is_finite());
    assert!(error > f64::from(f32::MAX));
    // Both error and tolerance overflowed f32 in the previous implementation;
    // inf > inf was false even though the real error exceeds this tolerance.
    assert!(check_close(&[f32::MAX], &[-f32::MAX], f32::MAX, 0.5).is_err());
}
