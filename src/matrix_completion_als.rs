use ndarray::{s, Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::{Solve, SVD};
use std::time::Instant;
use crate::print_array::print_array;

/// Performs matrix completion using Alternating Least Squares (ALS).
///
/// This algorithm completes a matrix with missing values by iteratively updating
/// factor matrices U and V. It follows these steps:
/// 1. Initialization: Missing values are filled with the mean of known values.
/// 2. Factor matrices U and V are initialized using SVD of the initial matrix.
/// 3. Alternating updates: U and V are updated alternately while keeping the other fixed.
/// 4. Convergence: The process iterates until the change is below a specified tolerance or max iterations are reached.
///
/// # Arguments
/// * `incomplete_matrix` - The input matrix with missing values (represented as NaN)
/// * `rank` - The rank of the low-rank approximation
/// * `tolerance` - The convergence tolerance
/// * `max_iterations` - The maximum number of iterations
/// * `lambda` - Regularization parameter
///
/// # Returns
/// The completed matrix
pub fn matrix_completion_als(
    observed_matrix: Array2<f64>,
    initial_guess: Option<Array2<f64>>,
    rank: usize,
    tolerance: f64,
    _max_iterations: usize,
    lambda: f64,
) -> Array2<f64> {
    let max_iterations = 2; // Limit max_iterations to 2 for debugging
    let start_time = Instant::now();
    let observed_mask = observed_matrix.mapv(|x| !x.is_nan());
    let (m, n) = observed_matrix.dim();

    // Create a default initial guess if not provided
    let mut completed_matrix = match initial_guess.as_ref() {
        Some(guess) => guess.clone(),
        None => {
            let mut item_averages = vec![0.0; m];
            let mut item_counts = vec![0; m];
            let mut user_averages = vec![0.0; n];
            let mut user_counts = vec![0; n];

            // Calculate item and user averages
            for ((i, j), &value) in observed_matrix.indexed_iter() {
                if !value.is_nan() {
                    item_averages[i] += value;
                    item_counts[i] += 1;
                    user_averages[j] += value;
                    user_counts[j] += 1;
                }
            }

            // Finalize averages
            for i in 0..m {
                item_averages[i] = if item_counts[i] > 0 { item_averages[i] / item_counts[i] as f64 } else { 0.0 };
            }
            for j in 0..n {
                user_averages[j] = if user_counts[j] > 0 { user_averages[j] / user_counts[j] as f64 } else { 0.0 };
            }

            // Create initial guess matrix
            Array2::from_shape_fn((m, n), |(i, j)| {
                if !observed_matrix[[i, j]].is_nan() {
                    observed_matrix[[i, j]]
                } else if item_counts[i] > 0 && user_counts[j] > 0 {
                    (item_averages[i] + user_averages[j]) / 2.0
                } else if item_counts[i] > 0 {
                    item_averages[i]
                } else if user_counts[j] > 0 {
                    user_averages[j]
                } else {
                    0.0 // Fallback for completely new items/users
                }
            })
        }
    };

    let mut u = Array::random((m, rank), Uniform::new(-0.01, 0.01));
    let mut v = Array::random((n, rank), Uniform::new(-0.01, 0.01));

    // Print initial guess matrix (top 20 lines)
    println!("Initial guess matrix (top 20 lines):");
    print_top_part(&completed_matrix);

    // Print first lines of values for the initial guess matrix
    println!("First lines of values for the initial guess matrix:");
    for i in 0..5.min(completed_matrix.nrows()) {
        let row = completed_matrix.row(i);
        println!("Row {}: {:?}", i, &row.as_slice().unwrap()[..20.min(row.len())]);
    }

    // Initialize U and V using SVD if possible
    if let Ok(svd) = completed_matrix.svd(true, true) {
        if let (Some(u_init), Some(v_init)) = (svd.0, svd.2) {
            u.assign(&u_init.slice(s![.., ..rank]));
            v.assign(&v_init.slice(s![.., ..rank]));
        }
    }

    for iteration in 0..max_iterations {
        println!("Iteration {}", iteration + 1);

        // Update U and V matrices
        update_factor_matrix(&mut u, &v, &observed_matrix, &observed_mask, lambda, true);
        update_factor_matrix(&mut v, &u, &observed_matrix.t().to_owned(), &observed_mask.t().to_owned(), lambda, false);

        // Compute low-rank approximation and update completed matrix
        let low_rank_approximation = u.dot(&v.t());
        update_completed_matrix(&mut completed_matrix, &low_rank_approximation, &observed_mask);

        // Check for convergence
        let frobenius_norm_difference = calculate_frobenius_norm_difference(&completed_matrix, &low_rank_approximation, &observed_mask);
        println!("Iteration {}: Frobenius norm difference: {}", iteration + 1, frobenius_norm_difference);

        // Print intermediate matrices (top 20 lines)
        println!("U matrix (top 20 lines):");
        print_top_part(&u);
        println!("V matrix (top 20 lines):");
        print_top_part(&v);
        println!("Low-rank approximation (top 20 lines):");
        print_top_part(&low_rank_approximation);
        println!("Completed matrix (top 20 lines):");
        print_top_part(&completed_matrix);

        if frobenius_norm_difference < tolerance {
            println!("Converged after {} iterations", iteration + 1);
            break;
        } else if frobenius_norm_difference.is_nan() || frobenius_norm_difference.is_infinite() {
            println!("Warning: Invalid Frobenius norm difference. Stopping iterations.");
            break;
        }

        // Update completed_matrix for the next iteration
        completed_matrix = low_rank_approximation.clone();

        // Debug print
        println!("Debug: U shape: {:?}, V shape: {:?}", u.shape(), v.shape());
    }

    // Print final statistics
    print_final_statistics(&completed_matrix, initial_guess.as_ref(), &observed_matrix, &observed_mask, start_time);

    completed_matrix
}

fn update_factor_matrix(factor: &mut Array2<f64>, other: &Array2<f64>, data: &Array2<f64>, mask: &Array2<bool>, lambda: f64, is_u: bool) {
    let (rows, _) = factor.dim();
    for i in 0..rows {
        let (slice, data_slice) = if is_u {
            (mask.slice(s![i, ..]), data.slice(s![i, ..]))
        } else {
            (mask.slice(s![.., i]), data.slice(s![.., i]))
        };
        let other_observed = other.select(Axis(0), &slice.iter().enumerate().filter_map(|(idx, &x)| if x { Some(idx) } else { None }).collect::<Vec<_>>());
        let data_observed = data_slice.select(Axis(0), &slice.iter().enumerate().filter_map(|(idx, &x)| if x { Some(idx) } else { None }).collect::<Vec<_>>());
        
        let a = other_observed.t().dot(&other_observed) + lambda * Array::eye(factor.ncols());
        let b = other_observed.t().dot(&data_observed);
        
        let solution = match a.solve_into(b) {
            Ok(sol) => sol,
            Err(_) => Array::random(factor.ncols(), Uniform::new(-0.01, 0.01)),
        };
        factor.slice_mut(s![i, ..]).assign(&solution);
    }
}

fn update_completed_matrix(completed_matrix: &mut Array2<f64>, low_rank_approximation: &Array2<f64>, observed_mask: &Array2<bool>) {
    completed_matrix
        .iter_mut()
        .zip(low_rank_approximation.iter())
        .zip(observed_mask.iter())
        .for_each(|((val, &approx_val), &is_observed)| {
            if !is_observed {
                *val = approx_val;
            }
        });
}

fn calculate_frobenius_norm_difference(matrix1: &Array2<f64>, matrix2: &Array2<f64>, mask: &Array2<bool>) -> f64 {
    let sum_squared_diff: f64 = matrix1.iter()
        .zip(matrix2.iter())
        .zip(mask.iter())
        .map(|((&a, &b), &m)| if !m { (a - b).powi(2) } else { 0.0 })
        .sum();

    let num_unobserved = mask.iter().filter(|&&x| !x).count() as f64;
    
    if num_unobserved > 0.0 {
        (sum_squared_diff / num_unobserved).sqrt()
    } else {
        0.0
    }
}

fn print_top_part(matrix: &Array2<f64>) {
    let top_rows = 20.min(matrix.nrows());
    let top_cols = 20.min(matrix.ncols());
    print_array(&matrix.slice(s![..top_rows, ..top_cols]).to_owned());
}

fn print_final_statistics(completed_matrix: &Array2<f64>, initial_guess: Option<&Array2<f64>>, observed_matrix: &Array2<f64>, observed_mask: &Array2<bool>, start_time: Instant) {
    println!("Total execution time: {:?}", start_time.elapsed());
    
    let rmse = (completed_matrix - observed_matrix)
        .iter()
        .zip(observed_mask.iter())
        .filter_map(|(&diff, &mask)| if mask { Some(diff.powi(2)) } else { None })
        .sum::<f64>()
        .sqrt() / observed_mask.iter().filter(|&&x| x).count() as f64;
    println!("RMSE: {}", rmse);
}
