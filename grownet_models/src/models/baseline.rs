
/// This code is taken from https://github.com/LaurentMazare/tch-rs/blob/main/examples/cifar/main.rs
/// with minor adjustments

use anyhow::Result;
use crossbeam::channel::unbounded;
use tch::nn::{FuncT, ModuleT, OptimizerConfig, SequentialT};
use tch::{nn, Device};
use derivative::Derivative;
use serde::{Serialize, Deserialize};


use super::{TrainProcess, Train, Log, TrainCommand};

fn conv_bn(vs: &nn::Path, c_in: i64, c_out: i64) -> SequentialT {
    let conv2d_cfg = nn::ConvConfig { padding: 1, bias: false, ..Default::default() };
    nn::seq_t()
        .add(nn::conv2d(vs, c_in, c_out, 3, conv2d_cfg))
        .add(nn::batch_norm2d(vs, c_out, Default::default()))
        .add_fn(|x| x.relu())
}

fn layer<'a>(vs: &nn::Path, c_in: i64, c_out: i64) -> FuncT<'a> {
    let pre = conv_bn(&vs.sub("pre"), c_in, c_out);
    let block1 = conv_bn(&vs.sub("b1"), c_out, c_out);
    let block2 = conv_bn(&vs.sub("b2"), c_out, c_out);
    nn::func_t(move |xs, train| {
        let pre = xs.apply_t(&pre, train).max_pool2d_default(2);
        let ys = pre.apply_t(&block1, train).apply_t(&block2, train);
        pre + ys
    })
}

fn fast_resnet(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(conv_bn(&vs.sub("pre"), 3, 64))
        .add(layer(&vs.sub("layer1"), 64, 128))
        .add(conv_bn(&vs.sub("inter"), 128, 256))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(layer(&vs.sub("layer2"), 256, 512))
        .add_fn(|x| x.max_pool2d_default(4).flat_view())
        .add(nn::linear(&vs.sub("linear"), 512, 10, Default::default()))
        .add_fn(|x| x * 0.125)
}

fn learning_rate(epoch: i64) -> f64 {
    if epoch < 50 {
        0.1
    } else if epoch < 100 {
        0.01
    } else {
        0.001
    }
}

#[derive(Derivative, Serialize, Deserialize, Clone, Debug)]
#[derivative(Default)]
pub struct SGD {
    #[derivative(Default(value="0.9"))]
    pub momentum: f64,
    #[derivative(Default(value="0.0"))]
    pub dampening: f64,
    #[derivative(Default(value="5e-4"))]
    pub wd: f64,
    #[derivative(Default(value="true"))]
    pub nesterov: bool 
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct ImTransform {
    #[derivative(Default(value="true"))]
    pub flip: bool,
    #[derivative(Default(value="4"))]
    pub crop: i64,
    #[derivative(Default(value="8"))]
    pub cutout: i64
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct BaselineParams {
    pub sgd: SGD,
    pub transform: ImTransform,
    #[derivative(Default(value="1.0"))]
    pub lr: f64,
    #[derivative(Default(value="100"))]
    pub epochs: u32,
    #[derivative(Default(value="4"))]
    pub batch_size: u32,
    pub data_path: String
}

impl Train for BaselineParams {
    fn build(&self) -> TrainProcess {
        let (command_sender, command_recv) = unbounded::<TrainCommand>();
        let (log_sender, log_recv) = unbounded::<Log>();

        let params = self.clone();
        let handle = std::thread::spawn(move || {
            let sender = log_sender;
            let recv = command_recv;
            let m = tch::vision::cifar::load_dir(params.data_path).unwrap();
            let vs = nn::VarStore::new(Device::cuda_if_available());
            let net = fast_resnet(&vs.root());
            let mut opt =
                nn::Sgd { momentum: params.sgd.momentum, 
                          dampening: params.sgd.dampening, 
                          wd: params.sgd.wd, 
                          nesterov: params.sgd.nesterov }.build(&vs, params.lr).unwrap();
            for epoch in 1..(params.epochs as i64) {
                opt.set_lr(learning_rate(epoch));
                for (bimages, blabels) in m.train_iter(params.batch_size as i64).shuffle().to_device(vs.device()) {
                    let bimages = tch::vision::dataset::augmentation(&bimages, params.transform.flip, params.transform.crop, params.transform.cutout);
                    let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
                    opt.backward_step(&loss);

                    match recv.recv().unwrap() {
                        TrainCommand::KILL => {
                            return ();
                        },
                        _ => {}
                    }
                        
                }
                let test_accuracy =
                net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
                sender.send(Log::PLOT("test accuracy".to_string(), epoch as f32, 100. * test_accuracy as f32)).unwrap();
            }
        });
        
        TrainProcess { send: command_sender, recv: log_recv, handle: handle }
    }
}


#[test]
fn cifar_test() {
    use std::path;
    println!("cur dir {:?}", std::env::current_dir().unwrap());
    let p: path::PathBuf = "assets/ml_datasets/cifar10".into();
    let p1 = std::env::current_dir().unwrap().parent().unwrap().join(p);
    assert!(p1.exists(), "path does not exist");
    let data = tch::vision::cifar::load_dir(p1).unwrap();
    let mut iter = data.train_iter(4);
    let (im, _label) = iter.next().unwrap();
    println!("im shape {:?}", im.size());
    println!("im stats: max {}, min {}, type {:?}", im.max(), im.min(), im.kind());
}