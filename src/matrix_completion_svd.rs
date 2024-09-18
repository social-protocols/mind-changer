use ndarray::{s, Array1, Array2};
use ndarray_linalg::SVD;
use std::io::Write;
use std::{io::stdout, time::Instant};

use crate::initial_guess::calculate_initial_guess;

/// Performs matrix completion using Singular Value Decomposition (SVD).
///
/// This function takes an incomplete matrix and attempts to fill in the missing values
/// using a low-rank approximation method based on SVD.
///
/// # Arguments
/// * `incomplete_matrix` - The input matrix with missing values (represented as NaN)
/// * `target_rank` - The desired rank for the low-rank approximation
/// * `convergence_tolerance` - The tolerance for convergence (based on Frobenius norm)
/// * `max_iterations` - The maximum number of iterations to perform
/// * `initial_guess` - An optional initial guess for the completed matrix
///
/// # Returns
/// The completed matrix with estimated values for previously missing entries
pub fn matrix_completion_svd(
    incomplete_matrix: Array2<f64>,
    energy_threshold: f64,
    convergence_tolerance: f64,
    max_iterations: usize,
    initial_guess: Option<Array2<f64>>,
) -> Array2<f64> {
    let start_time = Instant::now();

    // Initialize the matrix to be completed
    let mut completed_matrix =
        initial_guess.unwrap_or_else(|| calculate_initial_guess(&incomplete_matrix));

    let mut actual_iterations = 0;
    let mut last_frobenius_norm_difference = 0.0;
    for iteration in 0..max_iterations {
        // println!("{}", iteration);
        // print_array(&completed_matrix.slice(s![..20, ..]).to_owned());
        // thread::sleep(Duration::from_secs(1));

        actual_iterations = iteration + 1;
        // Perform SVD on the current completed matrix
        let svd = completed_matrix.svd(true, true).unwrap();
        let (left_singular_vectors, singular_values, right_singular_vectors_t) =
            (svd.0.unwrap(), svd.1, svd.2.unwrap());

        // Determine the adaptive rank based on cumulative singular values
        let adaptive_rank = calculate_rank_from_singular_values(&singular_values, energy_threshold);

        // Store the current matrix for comparison
        let previous_matrix = completed_matrix.clone();

        // Compute the low-rank approximation
        completed_matrix = left_singular_vectors
            .slice(s![.., ..adaptive_rank])
            .dot(&Array2::from_diag(
                &singular_values.slice(s![..adaptive_rank]),
            ))
            .dot(&right_singular_vectors_t.slice(s![..adaptive_rank, ..]));

        // Update only the missing values in the original matrix
        completed_matrix.zip_mut_with(&incomplete_matrix, |completed_value, &original_value| {
            if !original_value.is_nan() {
                *completed_value = original_value;
            }
        });

        // Check for convergence using Frobenius norm
        let matrix_difference = &completed_matrix - &previous_matrix;
        last_frobenius_norm_difference = matrix_difference.mapv(|x| x * x).sum().sqrt();
        print!(
            "\rmatrix completion: [{:>4}] {:.10} (rank {})",
            iteration, last_frobenius_norm_difference, adaptive_rank
        );
        stdout().flush().unwrap();

        if last_frobenius_norm_difference < convergence_tolerance {
            break;
        }
    }

    println!(
        "\rmatrix completion: [{:>4}] {:.10} in {:>6}ms",
        actual_iterations,
        last_frobenius_norm_difference,
        start_time.elapsed().as_millis()
    );
    completed_matrix
}

fn calculate_rank_from_singular_values(
    singular_values: &Array1<f64>,
    energy_threshold: f64,
) -> usize {
    let total_energy: f64 = singular_values.iter().map(|&x| x * x).sum();
    let mut cumulative_energy = 0.0;
    for (i, &value) in singular_values.iter().enumerate() {
        cumulative_energy += value * value;
        if cumulative_energy / total_energy >= energy_threshold {
            return i + 1;
        }
    }
    singular_values.len()
}
