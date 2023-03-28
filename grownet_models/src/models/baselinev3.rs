use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor, Shape}, module::ADModule, optim::Optimizer,
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
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
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MNISTBatch { images, targets }
    }
}

use burn::{
    module::{Module, Param},
    nn::{self, conv::Conv2dPaddingConfig, loss::CrossEntropyLoss, BatchNorm2d},
    tensor::{
        backend::{ADBackend},
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use num_traits::ToPrimitive;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Param<ConvBlock<B>>,
    conv2: Param<ConvBlock<B>>,
    conv3: Param<ConvBlock<B>>,
    dropout: nn::Dropout,
    fc1: Param<nn::Linear<B>>,
    fc2: Param<nn::Linear<B>>,
    activation: nn::GELU,
}

const NUM_CLASSES: usize = 10;

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        let conv1 = ConvBlock::new([1, 8], [3, 3]); // out: [Batch,8,26,26]
        let conv2 = ConvBlock::new([8, 16], [3, 3]); // out: [Batch,16,24x24]
        let conv3 = ConvBlock::new([16, 24], [3, 3]); // out: [Batch,24,22x22]
        let hidden_size = 24 * 22 * 22;
        let fc1 = nn::LinearConfig::new(hidden_size, 32)
            .with_bias(false);
        let fc1 = nn::Linear::new(&fc1);
        let fc2 = nn::LinearConfig::new(32, NUM_CLASSES)
            .with_bias(false);
        let fc2 = nn::Linear::new(&fc2);

        let dropout = nn::Dropout::new(&nn::DropoutConfig::new(0.3));

        Self {
            conv1: Param::from(conv1),
            conv2: Param::from(conv2),
            conv3: Param::from(conv3),
            fc1: Param::from(fc1),
            fc2: Param::from(fc2),
            dropout,
            activation: nn::GELU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, heigth, width] = input.dims();

        let x = input.reshape([batch_size, 1, heigth, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let [batch_size, channels, heigth, width] = x.dims();
        let x = x.reshape([batch_size, channels * heigth * width]);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.fc2.forward(x)
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLoss::new(None);
        let loss = loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Param<nn::conv::Conv2d<B>>,
    norm: Param<BatchNorm2d<B>>,
    activation: nn::GELU,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2]) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(Conv2dPaddingConfig::Valid);
        let conv = nn::conv::Conv2d::new(&conv);
        let norm = nn::BatchNorm2dConfig::new(channels[1]);
        let norm = nn::BatchNorm2d::new(&norm);

        Self {
            conv: Param::from(conv),
            norm: Param::from(norm),
            activation: nn::GELU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
}

impl<B: ADBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

use std::sync::Arc;

use burn::optim::decay::WeightDecayConfig;
use burn::optim::{Adam, AdamConfig};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 4)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 0)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}


use super::Config as MConfig;
use super::{config, TrainProcess};
use anyhow::Result;

pub fn baseline_config() -> MConfig {
    use crate::{Options, Config, opt};
    config!(
        ("lr", 1e-4),
        ("weight_decay", 5e-5),
        ("batch_size", 4),
        ("epochs", 4),
        ("train_log_steps", 100)
    )
}

fn run_v2<B: ADBackend>(device: B::Device, config: &MConfig) -> Result<TrainProcess> {
    use super::{PlotPoint, TrainRecv, TrainSend, RunStats};
    use crossbeam::channel::unbounded;

    let lr: f64 = config.uget("lr").into();
    let decay: f64 = config.uget("weight_decay").into();
    let batch_size: isize = config.uget("batch_size").into();
    let epochs: isize = config.uget("epochs").into();
    let train_log_steps: isize = config.uget("train_log_steps").into();

    let (command_sender, command_recv) = unbounded::<TrainSend>();
    let (log_sender, log_recv) = unbounded::<TrainRecv>();

    let sender = log_sender;
    let recv = command_recv;

    let handle = std::thread::spawn(move || {
        let config_optimizer =
            AdamConfig::new(lr).with_weight_decay(Some(WeightDecayConfig::new(decay)));
        let config = MnistTrainingConfig::new(config_optimizer);
        B::seed(config.seed);
    
        let batcher_train = Arc::new(MNISTBatcher::<B>::new(device.clone()));
        let batcher_valid = Arc::new(MNISTBatcher::<B::InnerBackend>::new(device.clone()));
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(batch_size as usize)
            .shuffle(config.seed)
            //.num_workers(config.num_workers)
            .build(Arc::new(MNISTDataset::train()));
        let dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(batch_size as usize)
            .shuffle(config.seed)
            //.num_workers(config.num_workers)
            .build(Arc::new(MNISTDataset::test()));
    
        // Model
        let mut optim = Adam::<B>::new(&config.optimizer);
        let mut model = Model::<B>::new();
    
        let mut steps = 0;
        let mut running_train_loss = 0.0;
        let mut running_train_acc = 0.0;
        let mut steps_since_last_log = 0;
    
        for _epoch in 0..epochs {
            let mut train_iter = dataloader_train.iter();
            let mut _test_iter = dataloader_test.iter();
    
            while let Some(item) = train_iter.next() {
                let item = <Model<B> as TrainStep<_, _>>::step(&model, item);
                model = optim.update_module(model, item.grads);
                let item = item.item;
                running_train_loss += f64::from_elem(item.loss.to_data().value[0]);
                running_train_acc += compute_accuracy(item);
                steps += 1;
                steps_since_last_log += 1;

                if steps % train_log_steps == 0 {
                    sender
                        .send(TrainRecv::PLOT(super::PlotPoint { 
                            title: "train loss", 
                            x_title: "step", 
                            y_title: "cross entropy", 
                            x: steps as f64, 
                            y: (running_train_loss / steps_since_last_log as f64)
                        }))
                        .unwrap();
                    sender
                        .send(TrainRecv::PLOT(super::PlotPoint { 
                            title: "train accuracy", 
                            x_title: "step", 
                            y_title: "accuracy", 
                            x: steps as f64, 
                            y: (running_train_acc / steps_since_last_log as f64)
                        })).unwrap();
                    steps_since_last_log = 1;
                    running_train_acc = 0.0;
                    running_train_loss = 0.0;
                }

                if let Ok(TrainSend::KILL) = recv.try_recv() {
                    return;
                }
            }
    
        }
    });

    Ok(TrainProcess {
        send: command_sender,
        recv: log_recv,
        handle: Some(handle),
    })
}

fn compute_accuracy<B: Backend>(input: ClassificationOutput<B>) -> f64 {
    let [batch_size, _n_classes] = input.output.dims();

    let targets = input.targets.clone().to_device(&B::Device::default());
    let outputs = input
        .output
        .clone()
        .argmax(1)
        .to_device(&B::Device::default())
        .reshape([batch_size]);

    let total_current =
        Into::<i64>::into(outputs.equal(targets).into_int().sum().to_data().value[0]) as usize;
    let accuracy = 100.0 * total_current as f64 / batch_size as f64;
    accuracy
}

pub fn run_train_loop(config: &MConfig) -> Result<TrainProcess> {
    use burn_ndarray::NdArrayBackend;
    use burn_autodiff::ADBackendDecorator;
    let dev = burn_ndarray::NdArrayDevice::Cpu;

    run_v2::<ADBackendDecorator<NdArrayBackend<f32>>>(dev, config)
}

#[test]
fn test_train_loop() {
    let config = baseline_config();
    let mut handle = run_train_loop(&config).unwrap();
    handle.send.send(super::TrainSend::KILL).unwrap();
    handle.kill_blocking().unwrap();

}