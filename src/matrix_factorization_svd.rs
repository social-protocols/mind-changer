use ndarray::{s, Array1, Array2};
use ndarray_linalg::SVD;
use std::io::Write;
use std::{io::stdout, time::Instant};

// Hyperparameters
const CONVERGENCE_TOLERANCE: f64 = 1e-3;
const MAX_ITERATIONS: usize = 300;

pub struct Factorization {
    pub u: Array2<f64>,
    pub s: Array1<f64>,
    pub vt: Array2<f64>,
    pub k: usize,
}

pub fn matrix_factorization_svd(
    incomplete_matrix: &Array2<f64>,
    rank: usize,
    initial_guess: Option<Array2<f64>>,
) -> Factorization {
    let start_time = Instant::now();

    let mut completed_matrix = initial_guess.unwrap_or_else(|| {
        Array2::from_elem(incomplete_matrix.dim(), observed_mean(incomplete_matrix))
    });

    let mut actual_iterations = 0;
    let mut last_frobenius_norm_difference = 0.0;
    let mut last_rmse = 0.0;
    let mut final_svd = None;

    for iteration in 0..MAX_ITERATIONS {
        actual_iterations = iteration + 1;

        let svd = completed_matrix.svd(true, true).unwrap();
        let (u, s, vt) = (svd.0.unwrap(), svd.1, svd.2.unwrap());

        let previous_matrix = completed_matrix.clone();

        completed_matrix = u.slice(s![.., ..rank])
            .dot(&Array2::from_diag(&s.slice(s![..rank])))
            .dot(&vt.slice(s![..rank, ..]));

        // Calculate RMSE for observed values
        let (sum_squared_error, count) = incomplete_matrix.indexed_iter()
            .filter(|(_, &val)| !val.is_nan())
            .fold((0.0, 0), |(sum, count), ((i, j), &original_value)| {
                let error = original_value - completed_matrix[[i, j]];
                (sum + error * error, count + 1)
            });
        last_rmse = (sum_squared_error / count as f64).sqrt();

        // Update only the missing values in the original matrix
        completed_matrix.zip_mut_with(incomplete_matrix, |completed_value, &original_value| {
            if !original_value.is_nan() {
                *completed_value = original_value;
            }
        });

        last_frobenius_norm_difference = (&completed_matrix - &previous_matrix).mapv(|x| x * x).sum().sqrt();
        
        print!(
            "\rmatrix factorization: [{:>4}/{}] Frobenius: {:.10}, RMSE: {:.10} (rank {})",
            iteration, MAX_ITERATIONS, last_frobenius_norm_difference, last_rmse, rank
        );
        stdout().flush().unwrap();

        final_svd = Some((u, s, vt));

        if last_frobenius_norm_difference < CONVERGENCE_TOLERANCE {
            break;
        }
    }

    println!(
        "\rmatrix factorization: [{:>4}/{}] Frobenius: {:.10}, RMSE: {:.10} (rank {}) in {:>6}ms",
        actual_iterations,
        MAX_ITERATIONS,
        last_frobenius_norm_difference,
        last_rmse,
        rank,
        start_time.elapsed().as_millis()
    );

    let (u, s, vt) = final_svd.unwrap();

    Factorization {
        u: u.slice(s![.., ..rank]).to_owned(),
        s: s.slice(s![..rank]).to_owned(),
        vt: vt.slice(s![..rank, ..]).to_owned(),
        k: rank,
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

        let factorization = matrix_factorization_svd(&incomplete_matrix, 2, None);

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
