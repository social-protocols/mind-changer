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
use rustc_hash::FxHashMap;
use serde::Deserialize;
// use ndarray_stats::QuantileExt;
// use statrs::distribution::{Continuous, StudentsT};
use std::error::Error;
use std::io;
use std::process;
use std::result::Result::{Err, Ok};

use csv::ReaderBuilder;
// https://burn.dev/burn-book/overview.html

use ndarray_linalg::SVD;

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

fn print_array(array: &Array2<f64>) {
    for row in array.outer_iter() {
        for &value in row.iter() {
            // Clamp the value between -1.0 and 1.0
            let value = value.max(-1.0).min(1.0);

            // Calculate the RGB values based on the value
            let (r, g, b) = if value < 0.0 {
                // Fade from yellow (255, 255, 0) to black (0, 0, 0)
                let level = (255.0 * (-value)).round() as u8;
                (level, level, 0)
            } else {
                // Fade from black (0, 0, 0) to blue (0, 0, 255)
                let level = (255.0 * value).round() as u8;
                (0, 0, level)
            };

            // Print the block with the appropriate color
            print!("\x1b[48;2;{};{};{}m  \x1b[0m", r, g, b);
        }
        println!();
    }
}

#[allow(non_snake_case, dead_code)]
#[derive(Debug, Deserialize)]
struct Record {
    noteId: String,
    raterParticipantId: String,
    createdAtMillis: u64,
    version: u8,
    agree: u8,
    disagree: u8,
    helpful: i8,
    notHelpful: i8,
    helpfulnessLevel: String,
    helpfulOther: u8,
    helpfulInformative: u8,
    helpfulClear: u8,
    helpfulEmpathetic: u8,
    helpfulGoodSources: u8,
    helpfulUniqueContext: u8,
    helpfulAddressesClaim: u8,
    helpfulImportantContext: u8,
    helpfulUnbiasedLanguage: u8,
    notHelpfulOther: u8,
    notHelpfulIncorrect: u8,
    notHelpfulSourcesMissingOrUnreliable: u8,
    notHelpfulOpinionSpeculationOrBias: u8,
    notHelpfulMissingKeyPoints: u8,
    notHelpfulOutdated: u8,
    notHelpfulHardToUnderstand: u8,
    notHelpfulArgumentativeOrBiased: u8,
    notHelpfulOffTopic: u8,
    notHelpfulSpamHarassmentOrAbuse: u8,
    notHelpfulIrrelevantSources: u8,
    notHelpfulOpinionSpeculation: u8,
    notHelpfulNoteNotNeeded: u8,
    ratedOnTweetId: String,
}

fn main_2() -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path("dataset/ratings-00001.tsv")?;
    let mut note_ids = FxHashMap::default();
    let mut rater_ids = FxHashMap::default();
    let mut records: Vec<Record> = Vec::new();
    let mut lineCounter = 0;
    for result in rdr.deserialize() {
        let record: Record = result?;
        if record.helpful != 0 || record.notHelpful != 0 {
            if !note_ids.contains_key(&record.noteId) {
                note_ids.insert(record.noteId.clone(), note_ids.len());
            }
            if !rater_ids.contains_key(&record.raterParticipantId) {
                rater_ids.insert(record.raterParticipantId.clone(), rater_ids.len());
            }
            records.push(record);
        }

        if rater_ids.len() >= 100 {
            break;
        }
        lineCounter += 1;
    }
    println!("scanned {} lines", lineCounter);
    println!("found {} records", records.len());

    let mut matrix: Array2<f64> = Array2::zeros((note_ids.len(), rater_ids.len()));

    for record in records {
        let note_index = *note_ids.get(&record.noteId).unwrap();
        let rater_index = *rater_ids.get(&record.raterParticipantId).unwrap();
        matrix[[note_index, rater_index]] = (record.helpful - record.notHelpful) as f64;
    }

    print_array(&matrix);

    // let array_d2 = arr2(&[
    //     [0., 0., 0., 0., 1.],
    //     [1., 1., 1., 1., 0.],
    //     [1., 0., 0., 0., 1.],
    //     [1., 0., 1., 1., 0.],
    // ]);

    let (u, sigma, v_t) = matrix.svd(true, true).unwrap();

    let u = u.unwrap();
    let v_t = v_t.unwrap();
    // Keep only the two largest singular values
    let k = 20;
    let truncated_sigma = sigma.slice(s![..k]).to_owned();
    let truncated_u = u.slice(s![.., ..k]).to_owned();
    let truncated_v_t = v_t.slice(s![..k, ..]).to_owned();

    // Reconstruct the matrix
    let approx = truncated_u.dot(&Array2::from_diag(&truncated_sigma).dot(&truncated_v_t));

    println!("approx:");
    print_array(&approx);

    // Calculate the Frobenius norm of the difference
    let diff = &matrix - &approx;
    println!("error:");
    print_array(&diff);
    let frobenius_norm = diff.mapv(|x| x.powi(2)).sum().sqrt();
    println!(
        "The Frobenius norm of the difference is: {}",
        frobenius_norm
    );

    // let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    // run::<Autodiff<burn::backend::NdArray>>(&device);
    Ok(())
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
