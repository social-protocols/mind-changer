use burn::{
    backend::Autodiff,
    config::Config,
    module::{Module, Param},
    nn::{
        loss::{MseLoss, Reduction},
        Initializer,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        activation,
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

use ndarray::{arr2, Array1, Array2, Axis};

use ndarray::*;
use ndarray_linalg::*;
// use ndarray_stats::QuantileExt;
// use statrs::distribution::{Continuous, StudentsT};
use std::result::Result::{Err, Ok};

// https://burn.dev/burn-book/overview.html
//

fn print_array(array: &Array2<f64>) {
    for row in array.outer_iter() {
        for &value in row.iter() {
            // Clamp the value between 0 and 1
            let value = value.max(0.0).min(1.0);

            // Convert the value to a grayscale level
            let level = (255.0 * (1.0 - value)).round() as u8;

            // Print the block with the appropriate color
            print!("\x1b[48;2;{};{};{}m  \x1b[0m", level, level, level);
        }
        println!();
    }
}

fn main() {
    let array_d2 = arr2(&[
        [0., 0., 0., 0., 1.],
        [1., 1., 1., 1., 0.],
        [1., 0., 0., 0., 1.],
        [1., 0., 1., 1., 0.],
    ]);

    let (u, sigma, v_t) = array_d2.svd(true, true).unwrap();

    let u = u.unwrap();
    let v_t = v_t.unwrap();
    // Keep only the two largest singular values
    let k = 2;
    let truncated_sigma = sigma.slice(s![..k]).to_owned();
    let truncated_u = u.slice(s![.., ..k]).to_owned();
    let truncated_v_t = v_t.slice(s![..k, ..]).to_owned();

    // Reconstruct the matrix
    let approx = truncated_u.dot(&Array2::from_diag(&truncated_sigma).dot(&truncated_v_t));

    println!("original:");
    print_array(&array_d2);
    println!("approx:");
    print_array(&approx);

    // Calculate the Frobenius norm of the difference
    let diff = &array_d2 - &approx;
    println!("diff:");
    print_array(&diff);
    let frobenius_norm = diff.mapv(|x| x.powi(2)).sum().sqrt();
    println!(
        "The Frobenius norm of the difference is: {}",
        frobenius_norm
    );

    // let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    // run::<Autodiff<burn::backend::NdArray>>(&device);
}

fn run<B: AutodiffBackend>(device: &B::Device) {
    let config = ModelConfig::new(MatrixFactorizationConfig::new(2), AdamConfig::new());
    B::seed(5);

    let target_matrix: Tensor<B, 2> = Tensor::from_floats(
        [
            [1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0.],
        ],
        device,
    );
    let mut model = config.model.init(&target_matrix, device);
    let mut optim = config.optimizer.init();

    let mse = MseLoss::new();
    for epoch in 1..config.num_epochs + 1 {
        let approximation = model.forward();
        println!("{}", approximation);
        // println!("{}", model.a.val());
        // println!("{}", model.b.val());

        let loss = mse.forward(approximation, target_matrix.clone(), Reduction::Mean);
        println!(
            "[Train - Epoch {}] Loss {:.3}",
            epoch,
            loss.clone().into_scalar(),
        );
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.lr, model, grads);
    }
}

#[derive(Module, Debug)]
pub struct MatrixFactorization<B: Backend> {
    a: Param<Tensor<B, 2>>,
    b: Param<Tensor<B, 2>>,
}

impl<B: Backend> MatrixFactorization<B> {
    pub fn forward(&self) -> Tensor<B, 2> {
        activation::softplus(self.a.val(), 1.).matmul(activation::softplus(self.b.val(), 1.))
    }
}

#[derive(Config)]
pub struct MatrixFactorizationConfig {
    d_latent_factors: usize,
}

impl MatrixFactorizationConfig {
    pub fn init<B: Backend>(
        &self,
        target_matrix: &Tensor<B, 2>,
        device: &B::Device,
    ) -> MatrixFactorization<B> {
        let initializer = Initializer::Normal {
            mean: -0.0,
            std: 0.01,
        };
        let shape = target_matrix.shape();
        MatrixFactorization {
            a: initializer.init([shape.dims[0], self.d_latent_factors], device),
            b: initializer.init([self.d_latent_factors, shape.dims[1]], device),
        }
    }
}

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.1)]
    pub lr: f64,
    pub model: MatrixFactorizationConfig,
    pub optimizer: AdamConfig,
}

// fn calculate_change(
//     before_matrix: &Array2<f64>,
//     after_matrix: &Array2<f64>,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     // Fill missing values using SVD
//     let (before_u, before_s, _) = before_matrix.svd(true, true)?;
//     let (after_u, after_s, _) = after_matrix.svd(true, true)?;
//
//     let n_components = 10;
//     let before_filled =
//         &before_u.slice(s![.., ..n_components]) * &before_s.slice(s![..n_components]);
//     let after_filled = &after_u.slice(s![.., ..n_components]) * &after_s.slice(s![..n_components]);
//
//     // Calculate overall similarity (1 - cosine distance)
//     let similarity =
//         1.0 - cosine_distance(&before_filled.into_raw_vec(), &after_filled.into_raw_vec())?;
//
//     // Calculate individual user changes
//     let user_changes = (&after_filled - &before_filled)
//         .mapv(f64::abs)
//         .mean_axis(Axis(1))
//         .unwrap();
//
//     // Calculate item popularity shifts
//     let item_shifts =
//         after_matrix.mean_axis(Axis(0)).unwrap() - before_matrix.mean_axis(Axis(0)).unwrap();
//
//     // Perform t-test
//     let (t_stat, p_value) =
//         paired_t_test(&before_filled.into_raw_vec(), &after_filled.into_raw_vec())?;
//
//     println!("Overall similarity: {}", similarity);
//     println!("Average user change: {}", user_changes.mean().unwrap());
//     println!("Item popularity shifts: {:?}", item_shifts);
//     println!("T-statistic: {}", t_stat);
//     println!("P-value: {}", p_value);
//
//     Ok(())
// }
//
// fn cosine_distance(a: &[f64], b: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
//     let dot_product: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
//     let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
//     let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
//
//     Ok(1.0 - (dot_product / (norm_a * norm_b)))
// }
//
// fn paired_t_test(a: &[f64], b: &[f64]) -> Result<(f64, f64), Box<dyn std::error::Error>> {
//     let n = a.len() as f64;
//     let diff: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
//     let mean_diff = diff.iter().sum::<f64>() / n;
//     let var_diff = diff.iter().map(|&x| (x - mean_diff).powi(2)).sum::<f64>() / (n - 1.0);
//     let std_error = (var_diff / n).sqrt();
//     let t_stat = mean_diff / std_error;
//     let df = n - 1.0;
//     let t_dist = StudentsT::new(0.0, 1.0, df)?;
//     let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
//
//     Ok((t_stat, p_value))
// }
//
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let before_matrix = Array2::from_shape_vec(
//         (3, 3),
//         vec![
//             1.0,
//             2.0,
//             std::f64::NAN,
//             3.0,
//             std::f64::NAN,
//             4.0,
//             std::f64::NAN,
//             5.0,
//             6.0,
//         ],
//     )?;
//
//     let after_matrix = Array2::from_shape_vec(
//         (3, 3),
//         vec![
//             2.0,
//             1.0,
//             std::f64::NAN,
//             4.0,
//             std::f64::NAN,
//             3.0,
//             std::f64::NAN,
//             6.0,
//             5.0,
//         ],
//     )?;
//
//     calculate_change(&before_matrix, &after_matrix)?;
//
//     Ok(())
// }
