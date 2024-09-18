use burn::backend::Autodiff;
use burn::config::Config;
use burn::module::Module;
use burn::optim::Optimizer;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use ndarray::Array2;

use burn::module::Param;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::nn::Initializer;
use burn::optim::AdamConfig;
use burn::optim::GradientsParams;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::result::Result;
use std::result::Result::Ok;

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

#[derive(Module, Debug)]
pub struct MatrixFactorization<B: Backend> {
    pub(crate) a: Param<Tensor<B, 2>>,
    pub(crate) b: Param<Tensor<B, 2>>,
}

impl<B: Backend> MatrixFactorization<B> {
    pub fn forward(&self) -> Tensor<B, 2> {
        activation::softplus(self.a.val(), 1.).matmul(activation::softplus(self.b.val(), 1.))
    }
}

#[derive(Config)]
pub struct MatrixFactorizationConfig {
    pub(crate) d_latent_factors: usize,
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

pub fn matrix_completion_gd(incomplete_matrix: Array2<f64>) -> Array2<f64> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let incomplete_matrix = array2_to_tensor(incomplete_matrix, &device);
    let result_tensor = run::<Autodiff<burn::backend::NdArray>>(incomplete_matrix, &device);
    tensor_to_array2(result_tensor)
}

fn run<B: AutodiffBackend>(incomplete_matrix: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2> {
    let config = ModelConfig::new(MatrixFactorizationConfig::new(2), AdamConfig::new());
    B::seed(config.seed);

    let mut model = config.model.init(&incomplete_matrix, device);
    let mut optim = config.optimizer.init();

    let mse = MseLoss::new();
    for epoch in 1..config.num_epochs + 1 {
        let approximation = model.forward();
        println!("{}", approximation);
        // println!("{}", model.a.val());
        // println!("{}", model.b.val());
        //

        let loss = mse.forward(approximation, incomplete_matrix.clone(), Reduction::Mean);
        println!(
            "[Train - Epoch {}] Loss {:.3}",
            epoch,
            loss.clone().into_scalar(),
        );
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.lr, model, grads);
    }

    // return completed matrix
    model.forward()
}

fn array2_to_tensor<B: Backend>(array: Array2<f64>, device: &B::Device) -> Tensor<B, 2> {
    let shape = array.shape();
    let data = array.clone().into_raw_vec();
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]])
}

fn tensor_to_array2<B: Backend>(tensor: Tensor<B, 2>) -> Array2<f64> {
    let values = tensor.to_data().to_vec().unwrap();
    Array2::from_shape_vec(tensor.dims(), values).unwrap()
}
