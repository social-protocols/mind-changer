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
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

fn main() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    run::<Autodiff<burn::backend::NdArray>>(&device);
}

fn run<B: AutodiffBackend>(device: &B::Device) {
    let config = ModelConfig::new(MatrixFactorizationConfig::new(3), AdamConfig::new());
    B::seed(config.seed);

    let target_matrix: Tensor<B, 2> = Tensor::from_floats(
        [
            [1., 0., 1., 0., 1.],
            [1., 1., 0., 0., 0.],
            [1., 0., 0., 0., 1.],
            [1., 1., 0., 0., 0.],
        ],
        device,
    );
    let mut model = config.model.init(&target_matrix, device);
    let mut optim = config.optimizer.init();

    let mse = MseLoss::new();
    for epoch in 1..config.num_epochs + 1 {
        let approximation = model.forward();
        println!("{}", approximation);

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
        self.a.val().matmul(self.b.val())
    }
}

#[derive(Config)]
pub struct MatrixFactorizationConfig {
    d_latent: usize,
}

impl MatrixFactorizationConfig {
    pub fn init<B: Backend>(
        &self,
        target_matrix: &Tensor<B, 2>,
        device: &B::Device,
    ) -> MatrixFactorization<B> {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.1,
        };
        let shape = target_matrix.shape();
        MatrixFactorization {
            a: initializer.init([shape.dims[0], self.d_latent], device),
            b: initializer.init([self.d_latent, shape.dims[1]], device),
        }
    }
}

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-1)]
    pub lr: f64,
    pub model: MatrixFactorizationConfig,
    pub optimizer: AdamConfig,
}
