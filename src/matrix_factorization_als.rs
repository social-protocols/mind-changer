use crate::print_array;
use ndarray::Array2;
use ndarray_linalg::Solve;
use rand::prelude::*;

// Hyperparameters
const K: usize = 3; // Number of latent factors
const MAX_ITERATIONS: usize = 1000; // Maximum number of ALS iterations
const LAMBDA: f64 = 0.01; // Regularization parameter
const CONVERGENCE_THRESHOLD: f64 = 1e-4; // Convergence threshold

pub struct Factorization {
    pub u: Array2<f64>,
    pub v: Array2<f64>,
}

pub fn matrix_factorization_als(incomplete_matrix: Array2<f64>) -> Factorization {
    let (users, items) = incomplete_matrix.dim();

    // Initialize U and V with small random values using a fixed seed
    let mut rng = StdRng::seed_from_u64(42);
    let mut u = Array2::from_shape_fn((users, K), |_| rng.gen::<f64>() * 0.1);
    let mut v = Array2::from_shape_fn((items, K), |_| rng.gen::<f64>() * 0.1);

    let mut prev_error = f64::INFINITY;

    for iteration in 0..MAX_ITERATIONS {
        // Fix V and solve for U
        for i in 0..users {
            let v_i = incomplete_matrix
                .row(i)
                .mapv(|x| if x.is_nan() { 0.0 } else { 1.0 });
            let v_i_v = &v.t().dot(&Array2::from_diag(&v_i));
            let v_i_v_reg = v_i_v.dot(&v) + LAMBDA * Array2::eye(K);
            let r_i = incomplete_matrix
                .row(i)
                .mapv(|x| if x.is_nan() { 0.0 } else { x });
            let r_i_v = r_i.dot(&v);
            u.row_mut(i)
                .assign(&v_i_v_reg.solve(&r_i_v.t()).unwrap().t());
        }

        // Fix U and solve for V
        for j in 0..items {
            let u_j = incomplete_matrix
                .column(j)
                .mapv(|x| if x.is_nan() { 0.0 } else { 1.0 });
            let u_j_u = &u.t().dot(&Array2::from_diag(&u_j));
            let u_j_u_reg = u_j_u.dot(&u) + LAMBDA * Array2::eye(K);
            let r_j = incomplete_matrix
                .column(j)
                .mapv(|x| if x.is_nan() { 0.0 } else { x });
            let r_j_u = r_j.dot(&u);
            v.row_mut(j)
                .assign(&u_j_u_reg.solve(&r_j_u.t()).unwrap().t());
        }

        // Check for convergence
        let reconstructed = u.dot(&v.t());
        let error = (&reconstructed - &incomplete_matrix)
            .mapv(|x| if x.is_nan() { 0.0 } else { x * x })
            .sum()
            / (users * items) as f64;

        if (prev_error - error).abs() < CONVERGENCE_THRESHOLD {
            println!("Converged after {} iterations", iteration + 1);
            break;
        }

        prev_error = error;

        if iteration == MAX_ITERATIONS - 1 {
            println!("Reached maximum iterations without converging");
        }
    }

    Factorization { u, v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;
    use rand::Rng;

    #[test]
    fn test_matrix_factorization_als() {
        // Create a simple 3x3 matrix with some missing values (NaN)
        let incomplete_matrix = arr2(&[
            [0.8, f64::NAN, -0.5],
            [-0.6, 0.3, f64::NAN],
            [f64::NAN, -0.2, 0.7],
        ]);

        println!("Incomplete matrix:");
        print_array(&incomplete_matrix);
        println!();

        let factorization = matrix_factorization_als(incomplete_matrix.clone());

        // Check dimensions of factorized matrices
        assert_eq!(factorization.u.dim(), (3, K));
        assert_eq!(factorization.v.dim(), (3, K));

        // Reconstruct the matrix
        let reconstructed = factorization.u.dot(&factorization.v.t());

        println!("Reconstructed matrix:");
        print_array(&reconstructed);
        println!();

        // Check if known values are close to the original
        assert_relative_eq!(reconstructed[[0, 0]], 0.8, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[0, 2]], -0.5, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 0]], -0.6, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 1]], 0.3, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 1]], -0.2, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 2]], 0.7, epsilon = 0.1);

        // Check if the algorithm filled in the missing values
        assert!(!reconstructed[[0, 1]].is_nan());
        assert!(!reconstructed[[1, 2]].is_nan());
        assert!(!reconstructed[[2, 0]].is_nan());
    }

    #[test]
    fn test_larger_matrix() {
        let mut rng = rand::thread_rng();
        let matrix = Array2::from_shape_fn((10, 10), |_| {
            if rng.gen::<f64>() < 0.3 {
                f64::NAN
            } else {
                rng.gen::<f64>() * 2.0 - 1.0
            }
        });

        let factorization = matrix_factorization_als(matrix.clone());
        let reconstructed = factorization.u.dot(&factorization.v.t());

        let mut total_error = 0.0;
        let mut count = 0;

        for i in 0..10 {
            for j in 0..10 {
                if !matrix[[i, j]].is_nan() {
                    total_error += (reconstructed[[i, j]] - matrix[[i, j]]).abs();
                    count += 1;
                }
            }
        }

        let average_error = total_error / count as f64;
        println!("Average error for larger matrix: {}", average_error);
        assert!(
            average_error < 0.3,
            "Average error {} is too high",
            average_error
        );
    }

    #[test]
    fn test_extreme_values() {
        let matrix = arr2(&[
            [0.99, f64::NAN, -0.99],
            [-0.95, 0.98, f64::NAN],
            [f64::NAN, -0.97, 0.96],
        ]);

        let factorization = matrix_factorization_als(matrix.clone());
        let reconstructed = factorization.u.dot(&factorization.v.t());

        assert_relative_eq!(reconstructed[[0, 0]], 0.99, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[0, 2]], -0.99, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 0]], -0.95, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 1]], 0.98, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 1]], -0.97, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 2]], 0.96, epsilon = 0.1);
    }

    #[test]
    fn test_sparse_matrix() {
        let mut rng = rand::thread_rng();
        let matrix = Array2::from_shape_fn((20, 20), |_| {
            if rng.gen::<f64>() < 0.8 {
                f64::NAN
            } else {
                rng.gen::<f64>() * 2.0 - 1.0
            }
        });

        println!("Original sparse matrix:");
        print_array(&matrix);
        println!();

        let factorization = matrix_factorization_als(matrix.clone());
        let reconstructed = factorization.u.dot(&factorization.v.t());

        println!("Reconstructed matrix:");
        print_array(&reconstructed);
        println!();

        let mut total_error = 0.0;
        let mut count = 0;

        for i in 0..20 {
            for j in 0..20 {
                if !matrix[[i, j]].is_nan() {
                    total_error += (reconstructed[[i, j]] - matrix[[i, j]]).abs();
                    count += 1;
                }
            }
        }

        let average_error = total_error / count as f64;
        println!("Average error for sparse matrix: {}", average_error);
        assert!(
            average_error < 0.4,
            "Average error {} is too high",
            average_error
        );
    }

    #[test]
    fn test_consistency() {
        let matrix = arr2(&[
            [0.8, f64::NAN, -0.5],
            [-0.6, 0.3, f64::NAN],
            [f64::NAN, -0.2, 0.7],
        ]);

        let mut results = Vec::new();

        for _ in 0..5 {
            let factorization = matrix_factorization_als(matrix.clone());
            let reconstructed = factorization.u.dot(&factorization.v.t());
            results.push(reconstructed);
        }

        // Check if all results are identical
        for i in 1..results.len() {
            for j in 0..3 {
                for k in 0..3 {
                    assert_relative_eq!(results[0][[j, k]], results[i][[j, k]], epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_matrix_with_missing_row() {
        let matrix = arr2(&[
            [f64::NAN, f64::NAN, f64::NAN],
            [0.5, -0.3, 0.7],
            [-0.2, 0.6, 0.4],
        ]);

        let factorization = matrix_factorization_als(matrix.clone());
        let reconstructed = factorization.u.dot(&factorization.v.t());

        // Check if the missing row has been filled
        assert!(!reconstructed[[0, 0]].is_nan());
        assert!(!reconstructed[[0, 1]].is_nan());
        assert!(!reconstructed[[0, 2]].is_nan());

        // Check if known values are close to the original
        assert_relative_eq!(reconstructed[[1, 0]], 0.5, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 1]], -0.3, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 2]], 0.7, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 0]], -0.2, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 1]], 0.6, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 2]], 0.4, epsilon = 0.1);
    }

    #[test]
    fn test_matrix_with_missing_column() {
        let matrix = arr2(&[
            [0.8, f64::NAN, -0.5],
            [-0.6, f64::NAN, 0.3],
            [0.4, f64::NAN, -0.2],
        ]);

        let factorization = matrix_factorization_als(matrix.clone());
        let reconstructed = factorization.u.dot(&factorization.v.t());

        // Check if the missing column has been filled
        assert!(!reconstructed[[0, 1]].is_nan());
        assert!(!reconstructed[[1, 1]].is_nan());
        assert!(!reconstructed[[2, 1]].is_nan());

        // Check if known values are close to the original
        assert_relative_eq!(reconstructed[[0, 0]], 0.8, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[0, 2]], -0.5, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 0]], -0.6, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[1, 2]], 0.3, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 0]], 0.4, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 2]], -0.2, epsilon = 0.1);
    }

    #[test]
    fn test_matrix_with_few_entries() {
        let matrix = arr2(&[
            [f64::NAN, 0.9, f64::NAN],
            [f64::NAN, f64::NAN, f64::NAN],
            [0.7, f64::NAN, f64::NAN],
        ]);

        let factorization = matrix_factorization_als(matrix.clone());
        let reconstructed = factorization.u.dot(&factorization.v.t());

        // Check if all entries have been filled
        for i in 0..3 {
            for j in 0..3 {
                assert!(!reconstructed[[i, j]].is_nan());
            }
        }

        // Check if known values are close to the original
        assert_relative_eq!(reconstructed[[0, 1]], 0.9, epsilon = 0.1);
        assert_relative_eq!(reconstructed[[2, 0]], 0.7, epsilon = 0.1);
    }
}
