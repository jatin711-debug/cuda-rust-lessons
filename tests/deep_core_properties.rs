use cuda_rust_lessons::{check_probabilities, learning::*};

#[test]
fn exclusive_scan_matches_known_answers_and_rejects_total_overflow() {
    assert_eq!(
        exclusive_scan_u32(&[3, 1, 4, 0, 2]).unwrap(),
        [0, 3, 4, 8, 8]
    );
    assert_eq!(exclusive_scan_u32(&[7]).unwrap(), [0]);
    assert_eq!(exclusive_scan_u32(&[]).unwrap(), Vec::<u32>::new());
    assert!(exclusive_scan_u32(&[u32::MAX, 1]).is_err());
    for len in [127, 128, 129, 16384, 16385] {
        let input = scan_inputs(len);
        let prefix = exclusive_scan_u32(&input).unwrap();
        assert_eq!(prefix[0], 0);
        for index in 1..len {
            assert_eq!(prefix[index] - prefix[index - 1], input[index - 1]);
        }
    }
}

#[test]
fn classifier_reference_preserves_batch_shapes_and_probability_properties() {
    for batch in [1, 17, 33] {
        let problem = classifier_problem(batch).unwrap();
        let reference = classifier_reference(&problem).unwrap();
        assert_eq!(reference.hidden.len(), batch * CLASSIFIER_HIDDEN);
        assert_eq!(reference.logits.len(), batch * CLASSIFIER_CLASSES);
        assert!(reference
            .hidden
            .iter()
            .all(|&value| value.is_finite() && value >= 0.0));
        check_probabilities(&reference.probabilities, batch, CLASSIFIER_CLASSES).unwrap();
    }
    // Hand-calculated two-row case exercises broadcast biases, ReLU, and
    // distinct hidden-to-class weights independently of the generated data.
    let mut known = classifier_problem(2).unwrap();
    known.input.fill(0.0);
    known.first_weights.fill(0.0);
    known.first_bias.fill(-1.0);
    known.first_bias[0] = 2.0;
    known.first_bias[1] = 3.0;
    known.second_weights.fill(0.0);
    known.second_weights[0] = 0.5;
    known.second_weights[CLASSIFIER_CLASSES] = -0.25;
    known.second_weights[1] = -0.5;
    known.second_weights[CLASSIFIER_CLASSES + 1] = 0.25;
    known.second_bias.fill(0.0);
    known.second_bias[0] = 0.5;
    known.second_bias[1] = -0.5;
    let reference = classifier_reference(&known).unwrap();
    let mut hidden_row = [0.0; CLASSIFIER_HIDDEN];
    hidden_row[0] = 2.0;
    hidden_row[1] = 3.0;
    let mut logits_row = [0.0; CLASSIFIER_CLASSES];
    logits_row[0] = 0.75; // 2*0.5 + 3*(-0.25) + 0.5
    logits_row[1] = -0.75; // 2*(-0.5) + 3*0.25 - 0.5
    assert_eq!(reference.hidden, hidden_row.repeat(2));
    assert_eq!(reference.logits, logits_row.repeat(2));
    assert!(reference.probabilities[0] > reference.probabilities[2]);
    assert!(reference.probabilities[2] > reference.probabilities[1]);
    assert!(classifier_problem(0).is_err());
    assert!(classifier_problem(1025).is_err());
    let mut bad = classifier_problem(1).unwrap();
    bad.second_bias.pop();
    assert!(classifier_reference(&bad).is_err());
}
