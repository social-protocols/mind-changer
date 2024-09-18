use ndarray::{s, Array2, Zip};
use ndarray_linalg::{Norm, SVD};
use std::io::Write;

// Hyperparameters
const LAMBDA_REG: f64 = 0.01;
const MAX_ITER: usize = 3000;
const TOL: f64 = 1e-6;
const LR: f64 = 1.0;

// ADAM hyperparameters
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPSILON: f64 = 1e-8;

/// Performs matrix completion using nuclear norm minimization and gradient descent.
///
/// This function takes an incomplete matrix and attempts to fill in the missing values.
///
/// # Arguments
/// * `incomplete_matrix` - The input matrix with missing values (represented as NaN)
///
/// # Returns
/// The completed matrix with estimated values for previously missing entries, or an error message

pub fn matrix_completion_gd(incomplete_matrix: Array2<f64>) -> Array2<f64> {
    let observation_mask = incomplete_matrix.map(|&v| !v.is_nan());
    let mut completed_matrix = incomplete_matrix.clone();

    let observed_count = observation_mask.iter().filter(|&&m| m).count() as f64;
    let mean_value = completed_matrix
        .iter()
        .filter(|&&v| !v.is_nan())
        .sum::<f64>()
        / observed_count;

    // Initialize missing values with mean
    Zip::from(&mut completed_matrix)
        .and(&observation_mask)
        .for_each(|value, &is_observed| {
            if !is_observed {
                *value = mean_value;
            }
        });

    let mut previous_loss = f64::INFINITY;

    // ADAM variables
    let mut m = Array2::zeros(completed_matrix.dim());
    let mut v = Array2::zeros(completed_matrix.dim());
    let mut t = 0;

    for iteration in 0..MAX_ITER {
        // Compute loss
        let (mse_loss, nuclear_norm) = compute_loss(
            &completed_matrix,
            &incomplete_matrix,
            &observation_mask,
            &observed_count,
        );
        let total_loss = mse_loss + LAMBDA_REG * nuclear_norm;

        // Check convergence
        if (previous_loss - total_loss).abs() / previous_loss < TOL {
            println!("\rMatrix completion: [{:>4}] {:.10}", iteration, total_loss);
            break;
        }
        previous_loss = total_loss;

        // ADAM step
        adam_step(
            &mut completed_matrix,
            &incomplete_matrix,
            &observation_mask,
            &mut m,
            &mut v,
            &mut t,
        );

        // Singular value thresholding
        singular_value_thresholding(&mut completed_matrix);

        // Project back to observed values
        project_to_observed(&mut completed_matrix, &incomplete_matrix, &observation_mask);

        print!("\rMatrix completion: [{:>4}] {:.10}", iteration, total_loss);
        std::io::stdout().flush().expect("Failed to flush stdout");
    }

    println!(); // Print a newline after the loop
    completed_matrix
}

fn compute_loss(
    completed_matrix: &Array2<f64>,
    incomplete_matrix: &Array2<f64>,
    observation_mask: &Array2<bool>,
    observed_count: &f64,
) -> (f64, f64) {
    let mse_loss = Zip::from(completed_matrix)
        .and(incomplete_matrix)
        .and(observation_mask)
        .fold(0.0, |acc, &completed_val, &original_val, &is_observed| {
            if is_observed {
                acc + (completed_val - original_val).powi(2)
            } else {
                acc
            }
        })
        / observed_count;

    let svd = completed_matrix
        .svd(false, false)
        .unwrap_or_else(|_| panic!("SVD computation failed"));
    let nuclear_norm = svd.1.sum();

    (mse_loss, nuclear_norm)
}

fn adam_step(
    completed_matrix: &mut Array2<f64>,
    incomplete_matrix: &Array2<f64>,
    observation_mask: &Array2<bool>,
    m: &mut Array2<f64>,
    v: &mut Array2<f64>,
    t: &mut usize,
) {
    let lr = LR;
    *t += 1;

    // Compute gradient
    let mut gradient = completed_matrix.clone();
    Zip::from(&mut gradient)
        .and(incomplete_matrix)
        .and(observation_mask)
        .for_each(|grad_val, &original_val, &is_observed| {
            if is_observed {
                *grad_val -= original_val;
            }
        });

    // Add regularization term to gradient
    let (u, s, vt) = completed_matrix.svd(true, true).unwrap();
    let u = u.unwrap();
    let vt = vt.unwrap();

    // Use only the top k singular values/vectors
    let k = s.len().min(completed_matrix.nrows().min(completed_matrix.ncols()));
    let u_k = u.slice(s![.., ..k]);
    let vt_k = vt.slice(s![..k, ..]);

    let reg_gradient = u_k.dot(&vt_k);
    gradient += &(LAMBDA_REG * &reg_gradient);

    // Normalize gradient
    let norm = gradient.norm_l2();
    if norm > 1.0 {
        gradient /= norm;
    }

    // Update biased first moment estimate
    *m = &*m * ADAM_BETA1 + &gradient * (1.0 - ADAM_BETA1);

    // Update biased second raw moment estimate
    *v = &*v * ADAM_BETA2 + &(&gradient * &gradient) * (1.0 - ADAM_BETA2);

    // Compute bias-corrected first moment estimate
    let m_hat = m.mapv(|x| x / (1.0 - ADAM_BETA1.powi(*t as i32)));

    // Compute bias-corrected second raw moment estimate
    let v_hat = v.mapv(|x| x / (1.0 - ADAM_BETA2.powi(*t as i32)));

    // Update parameters
    *completed_matrix -= &((&m_hat / &(v_hat.mapv(|x| x.sqrt()) + ADAM_EPSILON)) * lr);
}

fn singular_value_thresholding(matrix: &mut Array2<f64>) {
    let (u, mut singular_values, vt) = matrix
        .svd(true, true)
        .unwrap_or_else(|_| panic!("SVD computation failed"));

    singular_values
        .iter_mut()
        .for_each(|s| *s = (*s - LAMBDA_REG).max(0.0));
    let u = u.unwrap_or_else(|| panic!("Failed to unwrap U matrix"));
    let vt = vt.unwrap_or_else(|| panic!("Failed to unwrap V^T matrix"));
    let singular_values_diag = Array2::from_diag(&singular_values);
    *matrix = u
        .slice(s![.., ..singular_values.len()])
        .dot(&singular_values_diag)
        .dot(&vt.slice(s![..singular_values.len(), ..]));
}

fn project_to_observed(
    completed_matrix: &mut Array2<f64>,
    incomplete_matrix: &Array2<f64>,
    observation_mask: &Array2<bool>,
) {
    Zip::from(completed_matrix)
        .and(incomplete_matrix)
        .and(observation_mask)
        .for_each(|completed_val, &original_val, &is_observed| {
            if is_observed {
                *completed_val = original_val;
            }
        });
}
