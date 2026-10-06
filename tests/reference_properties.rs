use cuda_rust_lessons::*;

#[test]
fn vector_add_matches_a_hand_calculated_result_and_rejects_bad_lengths() {
    assert_eq!(
        vector_add(&[1.0, -2.0, 0.5], &[3.0, 2.0, -1.5]).unwrap(),
        vec![4.0, 0.0, -1.0]
    );
    assert!(vector_add(&[1.0], &[]).is_err());
}

#[test]
fn softmax_is_normalized_shift_invariant_and_stable_for_extreme_logits() {
    let logits = softmax_inputs();
    let output = softmax_rows(&logits, SOFTMAX_ROWS, SOFTMAX_COLS).unwrap();
    check_probabilities(&output, SOFTMAX_ROWS, SOFTMAX_COLS).unwrap();
    let uniform = &output[2 * SOFTMAX_COLS..3 * SOFTMAX_COLS];
    check_close(
        uniform,
        &vec![1.0 / SOFTMAX_COLS as f32; SOFTMAX_COLS],
        1e-7,
        0.0,
    )
    .unwrap();
    assert!(output[3 * SOFTMAX_COLS + 17] > 0.99999);

    let small = softmax_rows(&[0.0, 1.0, 2.0], 1, 3).unwrap();
    let shifted = softmax_rows(&[1000.0, 1001.0, 1002.0], 1, 3).unwrap();
    check_close(&small, &shifted, 1e-7, 0.0).unwrap();
    check_close(&small, &[0.09003057, 0.24472848, 0.66524094], 1e-7, 0.0).unwrap();
    assert!(softmax_rows(&[f32::NAN], 1, 1).is_err());
    assert!(softmax_rows(&[1.0], 0, 1).is_err());
}

#[test]
fn rectangular_matmul_matches_a_known_answer() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    assert_eq!(
        matmul(&a, &b, 2, 3, 2).unwrap(),
        vec![58.0, 64.0, 139.0, 154.0]
    );
    assert!(matmul(&a, &b, 3, 3, 2).is_err());
}

#[test]
fn zero_padding_preserves_the_original_product_for_uneven_shapes() {
    let (a, b) = matrix_inputs();
    let expected = matmul(&a, &b, MATRIX_M, MATRIX_K, MATRIX_N).unwrap();
    let m = round_up(MATRIX_M, 16);
    let k = round_up(MATRIX_K, 16);
    let n = round_up(MATRIX_N, 16);
    let padded_a = pad_matrix(&a, MATRIX_M, MATRIX_K, m, k).unwrap();
    let padded_b = pad_matrix(&b, MATRIX_K, MATRIX_N, k, n).unwrap();
    let padded_c = matmul(&padded_a, &padded_b, m, k, n).unwrap();
    let cropped = crop_matrix(&padded_c, m, n, MATRIX_M, MATRIX_N).unwrap();
    check_close(&cropped, &expected, 0.0, 0.0).unwrap();
    assert!(pad_matrix(&a, MATRIX_M, MATRIX_K, MATRIX_M - 1, MATRIX_K).is_err());
}

#[test]
fn verification_rejects_nan_missing_elements_and_wrong_values() {
    assert!(check_close(&[f32::NAN], &[1.0], 1e-5, 0.0).is_err());
    assert!(check_close(&[f32::INFINITY], &[1.0], 1e-5, 0.0).is_err());
    assert!(check_close(&[1.0], &[1.0, 2.0], 1e-5, 0.0).is_err());
    assert!(check_close(&[0.0], &[1.0], 1e-5, 0.0).is_err());
    assert!(check_close(&[1.0], &[1.0], f32::NAN, 0.0).is_err());
}
