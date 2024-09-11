mod modname {
    use burn::nn::Initializer;

    use burn::module::Param;

    use burn::tensor::backend::Backend;

    use burn::optim::GradientsParams;

    use burn::nn::loss::Reduction;

    use burn::nn::loss::MseLoss;

    use burn::tensor::Tensor;

    use burn::optim::AdamConfig;

    use std::result::Result::Ok;

    use super::print_array;

    use ndarray::Array2;

    use rustc_hash::FxHashMap;

    use csv::ReaderBuilder;

    use std::error::Error;

    use std::result::Result;

    #[allow(non_snake_case, dead_code)]
    #[derive(Debug, Deserialize)]
    pub(crate) struct Record {
        pub(crate) noteId: String,
        pub(crate) raterParticipantId: String,
        pub(crate) createdAtMillis: u64,
        pub(crate) version: u8,
        pub(crate) agree: u8,
        pub(crate) disagree: u8,
        pub(crate) helpful: i8,
        pub(crate) notHelpful: i8,
        pub(crate) helpfulnessLevel: String,
        pub(crate) helpfulOther: u8,
        pub(crate) helpfulInformative: u8,
        pub(crate) helpfulClear: u8,
        pub(crate) helpfulEmpathetic: u8,
        pub(crate) helpfulGoodSources: u8,
        pub(crate) helpfulUniqueContext: u8,
        pub(crate) helpfulAddressesClaim: u8,
        pub(crate) helpfulImportantContext: u8,
        pub(crate) helpfulUnbiasedLanguage: u8,
        pub(crate) notHelpfulOther: u8,
        pub(crate) notHelpfulIncorrect: u8,
        pub(crate) notHelpfulSourcesMissingOrUnreliable: u8,
        pub(crate) notHelpfulOpinionSpeculationOrBias: u8,
        pub(crate) notHelpfulMissingKeyPoints: u8,
        pub(crate) notHelpfulOutdated: u8,
        pub(crate) notHelpfulHardToUnderstand: u8,
        pub(crate) notHelpfulArgumentativeOrBiased: u8,
        pub(crate) notHelpfulOffTopic: u8,
        pub(crate) notHelpfulSpamHarassmentOrAbuse: u8,
        pub(crate) notHelpfulIrrelevantSources: u8,
        pub(crate) notHelpfulOpinionSpeculation: u8,
        pub(crate) notHelpfulNoteNotNeeded: u8,
        pub(crate) ratedOnTweetId: String,
    }

    pub(crate) fn main_2() -> Result<(), Box<dyn Error>> {
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

    pub(crate) fn run<B: AutodiffBackend>(device: &B::Device) {
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
}
