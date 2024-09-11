use ndarray::Array2;
use ndarray_linalg::SVD;

use std::{error::Error, time::Instant};

pub fn matrix_completion_svt(m: Array2<f64>, observed: Vec<(usize, usize)>) -> Array2<f64> {
    println!("Input matrix M shape: {:?}", m.shape());
    println!("Number of observed entries: {}", observed.len());

    let tau = 5.0;
    let delta = 1.;
    let max_iter = 1000;
    let epsilon = 1e-3;

    println!(
        "svt: tau: {}, delta: {}, max_iter: {}, epsilon: {}",
        tau, delta, max_iter, epsilon
    );

    let start = Instant::now();
    let completed_matrix = svt_algorithm(&m, &observed, tau, delta, max_iter, epsilon);
    let duration = start.elapsed();

    println!("Algorithm execution time: {} ms", duration.as_millis());
    completed_matrix
}

fn svt_algorithm(
    m: &Array2<f64>,
    omega: &[(usize, usize)],
    tau: f64,
    delta: f64,
    max_iter: usize,
    epsilon: f64,
) -> Array2<f64> {
    let (rows, cols) = m.dim();
    let mut y = Array2::<f64>::zeros((rows, cols));
    let mut x = Array2::<f64>::zeros((rows, cols));

    for iter in 0..max_iter {
        let (u, s, vt) = y.svd(true, true).unwrap();
        let s = s.mapv(|x| (x - tau).max(0.0));

        let min_dim = s.len();
        x.fill(0.0);
        for i in 0..min_dim {
            let u_col = u.as_ref().unwrap().column(i);
            let vt_row = vt.as_ref().unwrap().row(i);
            for r in 0..rows {
                for c in 0..cols {
                    x[[r, c]] += s[i] * u_col[r] * vt_row[c];
                }
            }
        }

        let mut error = 0.0;
        for &(i, j) in omega {
            let r_ij = m[[i, j]] - x[[i, j]];
            y[[i, j]] += delta * r_ij;
            error += r_ij * r_ij;
        }

        if (error / omega.len() as f64).sqrt() < epsilon {
            println!("Converged after {} iterations", iter + 1);
            return x;
        }
    }

    println!("Maximum iterations reached");
    x
}
