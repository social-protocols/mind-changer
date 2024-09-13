use ndarray::{s, Array2};
use ndarray_linalg::SVD;
use std::time::Instant;

use crate::initial_guess::calculate_initial_guess;

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
    initial_guess: Option<Array2<f64>>,
) -> Array2<f64> {
    // Start timing the execution to measure performance
    let start_time = Instant::now();
    // Create a mask to identify missing values (NaN) in the matrix
    let missing_value_mask = incomplete_matrix.mapv(|x| x.is_nan());

    // Initialize the completed matrix by filling missing values
    // with row averages, column averages, or global average
    let mut completed_matrix = match initial_guess {
        Some(guess) => guess,
        None => calculate_initial_guess(&incomplete_matrix),
    };

    // Iterate to refine the matrix completion
    for iteration in 0..max_iterations {
        // println!("Iteration {}", iteration + 1);

        // Perform Singular Value Decomposition (SVD)
        // Decompose the matrix into U, S, and V^T
        let svd = completed_matrix.svd(true, true).unwrap();
        let u = svd.0.unwrap();
        let s = svd.1;
        let vt = svd.2.unwrap();

        // Reconstruct the matrix using only the top 'rank' singular values
        // This creates a low-rank approximation of the matrix
        let low_rank_approximation = u
            .slice(s![.., ..rank])
            .dot(&Array2::from_diag(&s.slice(s![..rank])))
            .dot(&vt.slice(s![..rank, ..]));

        // Calculate the difference between the low-rank approximation and the current matrix
        let diff = &low_rank_approximation - &completed_matrix;
        // Compute the Frobenius norm of the difference, considering only missing values
        // This measures how much the missing values have changed
        let frobenius_norm_difference = diff
            .iter()
            .zip(missing_value_mask.iter())
            .map(|(&d, &mask)| if mask { d * d } else { 0.0 })
            .sum::<f64>()
            .sqrt();

        // Update the missing values in the completed matrix with the low-rank approximation
        completed_matrix
            .iter_mut()
            .zip(low_rank_approximation.iter())
            .zip(missing_value_mask.iter())
            .for_each(|((val, &approx_val), &is_missing)| {
                if is_missing {
                    *val = approx_val;
                }
            });

        println!(
            "{} Frobenius norm difference: {}",
            iteration, frobenius_norm_difference
        );

        // Check for convergence: if the change is below the tolerance, stop iterating
        if frobenius_norm_difference < tolerance {
            println!("Converged after {} iterations", iteration + 1);
            break;
        }
    }

    // Print the total execution time for performance analysis
    println!("Total execution time: {:?}", start_time.elapsed());
    completed_matrix
}
