use std::time::Instant;

mod dataset;
mod print_array;
mod svt;

use crate::dataset::extract_matrix_from_dataset;
use crate::svt::svt_algorithm;

use crate::print_array::print_array;

fn main() {
    let file_path = "dataset/ratings-00001.tsv";
    let max_raters = 80;

    let m = extract_matrix_from_dataset(file_path, max_raters)
        .expect("Failed to extract matrix from dataset");

    println!("Input matrix M shape: {:?}", m.shape());

    let omega: Vec<(usize, usize)> = m
        .indexed_iter()
        .filter(|&(_, &value)| !value.is_nan())
        .map(|((i, j), _)| (i, j))
        .collect();

    println!("Number of observed entries: {}", omega.len());

    let tau = 5.0;
    let delta = 1.2;
    let max_iter = 1000;
    let epsilon = 1e-3;

    println!("Parameters:");
    println!(
        "tau: {}, delta: {}, max_iter: {}, epsilon: {}",
        tau, delta, max_iter, epsilon
    );

    let start = Instant::now();
    let _completed_matrix = svt_algorithm(&m, &omega, tau, delta, max_iter, epsilon);
    let duration = start.elapsed();

    println!("\nAlgorithm execution time: {} ms", duration.as_millis());

    println!("\nOriginal incomplete matrix:");
    print_array(&m);
    println!("\nCompleted matrix:");
    print_array(&_completed_matrix);
}
