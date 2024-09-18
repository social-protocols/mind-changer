use ndarray::{s, Array1, Array2};
use ndarray_linalg::SVD;
use std::io::Write;
use std::{io::stdout, time::Instant};

// Hyperparameters
const ENERGY_THRESHOLD: f64 = 0.995; // for adaptive rank selection
const MIN_RANK: usize = 2;
const CONVERGENCE_TOLERANCE: f64 = 1e-3;
const MAX_ITERATIONS: usize = 1000;

pub struct Factorization {
    pub u: Array2<f64>,
    pub s: Array1<f64>,
    pub vt: Array2<f64>,
    pub k: usize,
}

pub fn matrix_factorization_svd(
    incomplete_matrix: &Array2<f64>,
    fixed_rank: Option<usize>,
    initial_guess: Option<Array2<f64>>,
) -> Factorization {
    let start_time = Instant::now();

    // Initialize the matrix to be completed
    let mut completed_matrix = initial_guess.unwrap_or_else(|| {
        Array2::from_elem(incomplete_matrix.dim(), observed_mean(incomplete_matrix))
    });

    let mut actual_iterations = 0;
    let mut last_frobenius_norm_difference = 0.0;
    let mut last_k = 1;

    for iteration in 0..MAX_ITERATIONS {
        actual_iterations = iteration + 1;

        // Perform SVD on the current completed matrix
        let svd = completed_matrix.svd(true, true).unwrap();
        let (left_singular_vectors, singular_values, right_singular_vectors_t) =
            (svd.0.unwrap(), svd.1, svd.2.unwrap());

        // adaptive rank selection
        let k: usize = fixed_rank.unwrap_or_else(|| {
            calculate_rank_from_singular_values(&singular_values, ENERGY_THRESHOLD).max(MIN_RANK)
        });
        last_k = k;

        // Store the current matrix for comparison
        let previous_matrix = completed_matrix.clone();

        // Compute the low-rank approximation
        completed_matrix = left_singular_vectors
            .slice(s![.., ..k])
            .dot(&Array2::from_diag(&singular_values.slice(s![..k])))
            .dot(&right_singular_vectors_t.slice(s![..k, ..]));

        // Update only the missing values in the original matrix
        completed_matrix.zip_mut_with(incomplete_matrix, |completed_value, &original_value| {
            if !original_value.is_nan() {
                *completed_value = original_value;
            }
        });

        // Check for convergence using Frobenius norm
        let matrix_difference = &completed_matrix - &previous_matrix;
        last_frobenius_norm_difference = matrix_difference.mapv(|x| x * x).sum().sqrt();
        print!(
            "\rmatrix factorization: [{:>4}] {:.10} (rank {})",
            iteration, last_frobenius_norm_difference, k
        );
        stdout().flush().unwrap();

        if last_frobenius_norm_difference < CONVERGENCE_TOLERANCE {
            break;
        }
    }

    let k = last_k;

    println!(
        "\rmatrix factorization: [{:>4}] {:.10} (rank {}) in {:>6}ms",
        actual_iterations,
        last_frobenius_norm_difference,
        k,
        start_time.elapsed().as_millis()
    );

    // Perform final SVD to get the factorization
    let svd = completed_matrix.svd(true, true).unwrap();
    let (u, s, vt) = (svd.0.unwrap(), svd.1, svd.2.unwrap());

    // Return only the top k factors
    Factorization {
        u: u.slice(s![.., ..k]).to_owned(),
        s: s.slice(s![..k]).to_owned(),
        vt: vt.slice(s![..k, ..]).to_owned(),
        k: last_k,
    }
}

fn observed_mean(matrix: &Array2<f64>) -> f64 {
    let non_nan_values: Vec<f64> = matrix.iter().filter(|&&x| !x.is_nan()).cloned().collect();

    let sum: f64 = non_nan_values.iter().sum();
    let count = non_nan_values.len();

    sum / count as f64
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_matrix_factorization_svd() {
        let incomplete_matrix = arr2(&[
            [1.0, f64::NAN, 3.0],
            [4.0, 5.0, f64::NAN],
            [f64::NAN, 8.0, 9.0],
        ]);

        let factorization = matrix_factorization_svd(&incomplete_matrix, None, None);

        assert_eq!(factorization.u.shape(), &[3, factorization.k]);
        assert_eq!(factorization.s.shape(), &[factorization.k]);
        assert_eq!(factorization.vt.shape(), &[factorization.k, 3]);

        // Reconstruct the matrix
        let reconstructed = factorization
            .u
            .dot(&Array2::from_diag(&factorization.s))
            .dot(&factorization.vt);

        // Check if known values are close to the original
        assert!((reconstructed[[0, 0]] - 1.0).abs() < 0.1);
        assert!((reconstructed[[0, 2]] - 3.0).abs() < 0.1);
        assert!((reconstructed[[1, 0]] - 4.0).abs() < 0.1);
        assert!((reconstructed[[1, 1]] - 5.0).abs() < 0.1);
        assert!((reconstructed[[2, 1]] - 8.0).abs() < 0.1);
        assert!((reconstructed[[2, 2]] - 9.0).abs() < 0.1);

        // Check if all values are filled
        for value in reconstructed.iter() {
            assert!(!value.is_nan());
        }
    }
}
