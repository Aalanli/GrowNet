use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, Tensor, ops::TensorOps}, optim::Optimizer,
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B::IntegerBackend, 1>,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| tensor / 255)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B::IntegerBackend, 1>::from_data(Data::from([item.label as i64])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device).detach();
        let targets = Tensor::cat(targets, 0).to_device(&self.device).detach();

        MNISTBatch { images, targets }
    }
}

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Param<nn::conv::Conv2d<B>>,
    pool: nn::pool::MaxPool2d,
    activation: nn::GELU,
}

#[derive(Config)]
pub struct ConvBlockConfig {
    channels: [usize; 2],
    #[config(default = "[3, 3]")]
    kernel_size: [usize; 2],
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(config: &ConvBlockConfig) -> Self {
        let conv = nn::conv::Conv2d::new(
            &nn::conv::Conv2dConfig::new(config.channels, config.kernel_size)
                .with_padding(nn::conv::Conv2dPaddingConfig::Same),
        );
        let pool = nn::pool::MaxPool2d::new(
            &nn::pool::MaxPool2dConfig::new(config.channels[1], config.kernel_size)
                .with_padding(nn::conv::Conv2dPaddingConfig::Same),
        );
        let activation = nn::GELU::new();

        Self {
            conv: Param::new(conv),
            pool,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input.clone());
        let x = self.pool.forward(x);
        let x = self.activation.forward(x);

        (x + input) / 2.0
    }
}


/// Configuration to create a [Multilayer Perceptron](Mlp) layer.
#[derive(Config)]
pub struct MlpConfig {
    /// The number of layers.
    #[config(default = 3)]
    pub num_layers: usize,
    /// The dropout rate.
    #[config(default = 0.5)]
    pub dropout: f64,
    /// The size of each layer.
    #[config(default = 256)]
    pub d_model: usize,
}

/// Multilayer Perceptron module.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linears: Param<Vec<nn::Linear<B>>>,
    dropout: nn::Dropout,
    activation: nn::ReLU,
}

impl<B: Backend> Mlp<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &MlpConfig) -> Self {
        let mut linears = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            let linear = nn::Linear::new(&nn::LinearConfig::new(config.d_model, config.d_model));
            linears.push(linear);
        }

        Self {
            linears: Param::new(linears),
            dropout: nn::Dropout::new(&nn::DropoutConfig::new(0.3)),
            activation: nn::ReLU::new(),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, d_model]`
    /// - output: `[batch_size, d_model]`
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = self.dropout.forward(x);
            x = self.activation.forward(x);
        }

        x
    }
}


use burn::{
    nn::{loss::CrossEntropyLoss},
    optim::AdamConfig,
    tensor::{
        backend::{ADBackend},
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct MnistConfig {
    #[config(default = 6)]
    pub num_epochs: usize,
    #[config(default = 12)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub optimizer: AdamConfig,
    pub mlp: MlpConfig,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    conv: Param<ConvBlock<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
    num_classes: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &MnistConfig, d_input: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let output = nn::Linear::new(&nn::LinearConfig::new(config.mlp.d_model, num_classes));
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, config.mlp.d_model));
        let conv = ConvBlock::new(&ConvBlockConfig::new([1, 1]));

        Self {
            mlp: Param::new(mlp),
            conv: Param::new(conv),
            output: Param::new(output),
            input: Param::new(input),
            num_classes,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, heigth, width] = input.dims();

        let x = input.reshape([batch_size, 1, heigth, width]).detach();
        let x = self.conv.forward(x);
        let x = x.reshape([batch_size, heigth * width]);

        let x = self.input.forward(x);
        let x = self.mlp.forward(x);

        self.output.forward(x)
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLoss::new(self.num_classes, None);
        let loss = loss.forward(&output, &targets);

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<B, MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<B, ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

use burn::optim::decay::WeightDecayConfig;
use burn::optim::{Adam};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

pub fn run<B: ADBackend>(device: B::Device) {
    // Config
    let config_optimizer =
        AdamConfig::new(1e-4).with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config_mlp = MlpConfig::new();
    let config = MnistConfig::new(config_optimizer, config_mlp);
    B::seed(config.seed);

    // Data
    let batcher_train = Arc::new(MNISTBatcher::<B>::new(device.clone()));
    let batcher_valid = Arc::new(MNISTBatcher::<B::InnerBackend>::new(device.clone()));
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::train()));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::test()));

    // Model
    let mut optim = Adam::new(&config.optimizer);
    let mut model = Model::new(&config, 784, 10);

    for i in dataloader_train.iter() {
        let output = <Model<B> as TrainStep<B, MNISTBatch<B>, ClassificationOutput<B>>>::step(&model, i);
        optim.update_module(&mut model, output.grads);
    }

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer::<f32>(2)
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        
        .build(model, optim);

    let _model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();
}

#[test]
fn test_run() {
    use burn_ndarray::{NdArrayBackend, NdArrayDevice};
    use burn_autodiff::ADBackendDecorator;

    let dev = NdArrayDevice::Cpu;
    run::<ADBackendDecorator<NdArrayBackend::<f32>>>(dev);
    
}