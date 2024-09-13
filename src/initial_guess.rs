use ndarray::Array2;

pub fn calculate_initial_guess(incomplete_matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = incomplete_matrix.dim();

    // Calculate row averages
    let row_averages: Vec<f64> = (0..rows)
        .map(|i| {
            let row = incomplete_matrix.row(i);
            let known_values: Vec<f64> = row.iter().filter(|&&x| !x.is_nan()).cloned().collect();
            if known_values.is_empty() {
                f64::NAN
            } else {
                known_values.iter().sum::<f64>() / known_values.len() as f64
            }
        })
        .collect();

    // Calculate column averages
    let col_averages: Vec<f64> = (0..cols)
        .map(|j| {
            let col = incomplete_matrix.column(j);
            let known_values: Vec<f64> = col.iter().filter(|&&x| !x.is_nan()).cloned().collect();
            if known_values.is_empty() {
                f64::NAN
            } else {
                known_values.iter().sum::<f64>() / known_values.len() as f64
            }
        })
        .collect();

    // Calculate global average
    let global_average = row_averages.iter().filter(|&&x| !x.is_nan()).sum::<f64>()
        / row_averages.iter().filter(|&&x| !x.is_nan()).count() as f64;

    let mut completed_matrix = incomplete_matrix.clone();
    for ((i, j), value) in completed_matrix.indexed_iter_mut() {
        if value.is_nan() {
            *value = if !row_averages[i].is_nan() {
                row_averages[i]
            } else if !col_averages[j].is_nan() {
                col_averages[j]
            } else {
                global_average
            };
        }
    }

    completed_matrix
}
