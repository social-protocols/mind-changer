use ndarray::{s, Array2};
use ndarray_linalg::SVD;

use std::time::Instant;

/// Performs matrix completion using Singular Value Decomposition (SVD).
///
/// This algorithm completes a matrix with missing values by iteratively applying SVD
/// and updating the missing entries. It follows these steps:
/// 1. Initialization: Missing values are filled with the mean of known values.
/// 2. SVD Decomposition: The matrix is decomposed into singular values and vectors.
/// 3. Rank Reduction: Only the top k singular values and vectors are kept.
/// 4. Update: Missing values are updated with the low-rank approximation.
/// 5. Convergence: The process iterates until the change is below a specified tolerance.
///
/// # Arguments
/// * `incomplete_matrix` - The input matrix with missing values (represented as NaN)
/// * `rank` - The rank of the low-rank approximation
/// * `tolerance` - The convergence tolerance
/// * `max_iterations` - The maximum number of iterations
///
/// # Returns
/// The completed matrix
pub fn matrix_completion_svd(
    incomplete_matrix: Array2<f64>,
    rank: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Array2<f64> {
    let start_time = Instant::now();
    let missing_value_mask = incomplete_matrix.mapv(|x| x.is_nan());

    let known_values_mean = incomplete_matrix
        .iter()
        .filter(|&&x| !x.is_nan())
        .sum::<f64>()
        / incomplete_matrix.iter().filter(|&&x| !x.is_nan()).count() as f64;

    let mut completed_matrix =
        incomplete_matrix.mapv(|x| if x.is_nan() { known_values_mean } else { x });

    for iteration in 0..max_iterations {
        // println!("Iteration {}", iteration + 1);

        let (u, s, vt) = completed_matrix.svd(true, true).unwrap();
        let low_rank_approximation = u
            .unwrap()
            .slice(s![.., ..rank])
            .dot(&Array2::from_diag(&s.slice(s![..rank])))
            .dot(&vt.unwrap().slice(s![..rank, ..]));

        let diff = &low_rank_approximation - &completed_matrix;
        let frobenius_norm_difference = diff
            .iter()
            .zip(missing_value_mask.iter())
            .map(|(&d, &mask)| if mask { d * d } else { 0.0 })
            .sum::<f64>()
            .sqrt();

        completed_matrix
            .iter_mut()
            .zip(low_rank_approximation.iter())
            .zip(missing_value_mask.iter())
            .for_each(|((val, &approx_val), &is_missing)| {
                if is_missing {
                    *val = approx_val;
                }
            });

        // println!("Frobenius norm difference: {}", frobenius_norm_difference);

        if frobenius_norm_difference < tolerance {
            println!("Converged after {} iterations", iteration + 1);
            break;
        }
    }

    println!("Total execution time: {:?}", start_time.elapsed());
    completed_matrix
}
