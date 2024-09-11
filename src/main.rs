use ndarray::Array2;

use ndarray::*;
use ndarray_linalg::*;

mod print_array;
use crate::print_array::print_array;

// nuclear norm minimization to fill in missing data of a user-item matrix: Singular Value Thresholding (SVT)
// Parameters:
// M (numpy.ndarray): The incomplete matrix
// omega (list of tuples): List of observed entries (i, j)
// tau (float): Threshold parameter
// delta (float): Step size
// max_iter (int): Maximum number of iterations
// epsilon (float): Convergence tolerance
pub fn svt_algorithm(
    m: &Array2<f64>,
    omega: &[(usize, usize)],
    tau: f64,
    delta: f64,
    max_iter: usize,
    epsilon: f64,
) -> Array2<f64> {
    let (rows, cols) = m.dim();
    let mut y = Array2::<f64>::zeros((rows, cols));

    for _ in 0..max_iter {
        // Singular value decomposition
        let svd = y.svd(true, true).unwrap();
        let u = svd.0.as_ref().unwrap();
        let mut s = svd.1;
        let vt = svd.2.as_ref().unwrap();

        // Soft-thresholding operator
        s.mapv_inplace(|x| if x > tau { x - tau } else { 0.0 });

        // Update Y
        let mut error_sum = 0.0;
        for &(i, j) in omega {
            let x_ij = u.row(i).dot(&(&s * &vt.column(j)));
            let r_ij = m[[i, j]] - x_ij;
            y[[i, j]] += delta * r_ij;
            error_sum += r_ij.powi(2);
        }

        // Check for convergence
        let error = (error_sum / omega.len() as f64).sqrt();
        if error < epsilon {
            break;
        }
    }

    // Final reconstruction
    let svd = y.svd(true, true).unwrap();
    let u = svd.0.unwrap();
    let mut s = svd.1;
    let vt = svd.2.unwrap();
    s.mapv_inplace(|x| if x > tau { x - tau } else { 0.0 });

    // Correct multiplication for non-square matrices
    let s_matrix = Array2::from_diag(&s);
    u.dot(&s_matrix).dot(&vt)
}

fn main() {
    use ndarray::array;

    // Example usage
    let m = array![
        [0., 0., 0., 0., 1.],
        [1., 1., 1., 1., 0.],
        [1., 0., 0., 0., 1.],
        [1., 0., 1., 1., 0.],
    ];

    let omega: Vec<(usize, usize)> = m
        .indexed_iter()
        .filter(|&(_, &value)| value == 1.0)
        .map(|((i, j), _)| (i, j))
        .collect();

    let tau = 5.0;
    let delta = 1.2;
    let max_iter = 1000;
    let epsilon = 1e-4;

    let completed_matrix = svt_algorithm(&m, &omega, tau, delta, max_iter, epsilon);

    println!("Original incomplete matrix:");
    print_array(&m);
    println!("\nCompleted matrix:");
    print_array(&completed_matrix);
}
