use std::fmt::Display;
use std::time::{Duration, Instant};

/// This code is taken from https://github.com/LaurentMazare/tch-rs/blob/main/examples/cifar/main.rs
/// with adjustments
use anyhow::{Error, Result};
use crossbeam::channel::unbounded;
use derivative::Derivative;
use derive_more::{Deref, DerefMut};
use serde::{Deserialize, Serialize};
use tch::nn::{FuncT, ModuleT, OptimizerConfig, SequentialT};
use tch::{nn, Device, Tensor};

use crate::config;
use crate::configs::Config;

use super::{TrainProcess, TrainRecv, TrainSend, RunStats};

pub fn check_default(org: &Config, y: &Config) -> Result<()> {
    if !org.subset(&y) {
        return Err(Error::msg(format!(
            "Given config is not subset of expected config\ngiven:\n{y}\nexpected:\n{org}"
        )));
    }
    Ok(())
}

fn conv_bn(vs: &nn::Path, c_in: i64, c_out: i64) -> SequentialT {
    let conv2d_cfg = nn::ConvConfig {
        padding: 1,
        bias: false,
        ..Default::default()
    };
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
        //.add_fn(|x| x.max_pool2d_default(4).flat_view())
        //.add(nn::linear(&vs.sub("linear"), 512, 10, Default::default()))
        //.add_fn(|x| x * 0.125)
        
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


use crate::{opt, Options};

pub fn sgd_config() -> Config {
    config!(
        ("momentum", 0.9),
        ("dampening", 0.0),
        ("wd", 5e-4),
        ("nesterov", true)
    )
}

pub fn checkpoint_config() -> Config {
    config!(
        ("steps_per_checkpoint", 500),
        ("checkpoint_path", Path("")),
        ("checkpoint_basename", "")
    )
}

pub fn imtransform_config() -> Config {
    config!(("flip", true), ("crop", 4), ("cutout", 8))
}

pub fn baseline_config() -> Config {
    use crate::{opt, Options};
    let mut config = config!(
        ("epochs", 100),
        ("batch_size", 4),
        ("lr", 1.0),
        ("steps_per_log", 100),
        ("steps_per_checkpoint", 500),
        ("data_path", Path(""))
    );

    config.add("sgd", sgd_config()).unwrap();
    config
        .add("transform", imtransform_config())
        .unwrap();
    // config
    //     .add("checkpoint", checkpoint_config())
    //     .unwrap();

    config
}

#[test]
fn test() {
    tch::maybe_init_cuda();
    let vs = nn::VarStore::new(Device::Cuda(0));
    println!("{:?}", Device::cuda_if_available());
    let net = fast_resnet(&vs.root());
    let no_grad = tch::no_grad_guard();
    let t = std::time::Instant::now();
    let x = Tensor::rand(&[4, 3, 256, 256], (tch::Kind::Float, Device::Cuda(0)));
    let mut y;
    for i in 0..25 {
        y = net.forward_t(&x, false);
    }

    println!("{}", t.elapsed().as_secs_f32());
}

pub fn build(config: &Config) -> Result<TrainProcess> {
    check_default(&baseline_config(), config)?;
    let params = config.clone();

    let (command_sender, command_recv) = unbounded::<TrainSend>();
    let (log_sender, log_recv) = unbounded::<TrainRecv>();

    let handle = std::thread::spawn(move || {
        let params_ = params;
        let params = &params_;
        let sender = log_sender;
        let recv = command_recv;
        let m =
            tch::vision::cifar::load_dir(std::path::PathBuf::from(params.uget("data_path")))
                .unwrap();
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let net = fast_resnet(&vs.root());
        let mut opt = nn::Sgd {
            momentum: params.uget("sgd/momentum").into(),
            dampening: params.uget("sgd/dampening").into(),
            wd: params.uget("sgd/wd").into(),
            nesterov: params.uget("sgd/nesterov").into(),
        }
        .build(&vs, params.uget("lr").into())
        .unwrap();

        let mut steps: usize = 0;

        let flip = params.uget("transform/flip").into();
        let crop = params.uget("transform/crop").into();
        let cutout = params.uget("transform/cutout").into();

        let steps_per_log: usize = params.uget("steps_per_log").into();
        let epochs = params.uget("epochs").into();
        let batch_size = params.uget("batch_size").into();

        // let steps_per_checkpoint: usize = params.uget("checkpoint/steps_per_checkpoint").into();
        // let checkpoint_path: std::path::PathBuf =
            // params.uget("checkpoint/checkpoint_path").into();
        // let checkpoint_basename: String = params.uget("checkpoint/checkpoint_basename").into();
        let mut avg_time = 0.0;
        let mut last_time;
        let now = Instant::now();

        for epoch in 1..epochs {
            opt.set_lr(learning_rate(epoch));

            for (bimages, blabels) in m.train_iter(batch_size).shuffle().to_device(vs.device())
            {
                last_time = now.elapsed().as_secs_f32();
                let bimages = tch::vision::dataset::augmentation(&bimages, flip, crop, cutout);
                let loss = net
                    .forward_t(&bimages, true)
                    .cross_entropy_for_logits(&blabels);
                opt.backward_step(&loss);
                steps += 1;
                if steps % steps_per_log == 0 {
                    let loss = f64::from(loss.to_device(tch::Device::Cpu)) / batch_size as f64;
                    let acc = net.batch_accuracy_for_logits(
                        &bimages,
                        &blabels,
                        vs.device(),
                        batch_size.into(),
                    );
                    sender
                        .send(TrainRecv::PLOT(super::PlotPoint { 
                            title: "train loss", 
                            x_title: "step", 
                            y_title: "cross entropy", 
                            x: steps as f64, 
                            y: loss 
                        }))
                        .unwrap();
                    sender
                        .send(TrainRecv::PLOT(super::PlotPoint { 
                            title: "train accuracy", 
                            x_title: "step", 
                            y_title: "accuracy", 
                            x: steps as f64, 
                            y: acc 
                        })).unwrap();
                    sender.send(TrainRecv::STATS(RunStats { step_time: Some(avg_time / steps_per_log as f32) })).unwrap();
                    avg_time = 0.0;
                }

                // if steps % steps_per_checkpoint == 0 {
                //     let run_path =
                //         checkpoint_path.join(format!("{}-step{steps}", checkpoint_basename));
                //     if let Err(e) = vs.save(&run_path) {
                //         sender
                //             .send(TrainRecv::FAILED(format!(
                //                 "failed to write checkpoint, {}",
                //                 e
                //             )))
                //             .unwrap();
                //         return;
                //     } else {
                //         sender
                //             .send(TrainRecv::CHECKPOINT(steps as f32, run_path))
                //             .unwrap();
                //     }
                // }

                if let Ok(TrainSend::KILL) = recv.try_recv() {
                    return;
                }
                avg_time += now.elapsed().as_secs_f32() - last_time;
            }
            let test_accuracy =
                net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
            sender
                .send(TrainRecv::PLOT(super::PlotPoint { 
                    title: "test accuracy", 
                    x_title: "epoch", 
                    y_title: "accuracy", 
                    x: epoch as f64, 
                    y: test_accuracy as f64 
                })).unwrap();
        }
    });

    Ok(TrainProcess {
        send: command_sender,
        recv: log_recv,
        handle: Some(handle),
    })
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
    println!(
        "im stats: max {}, min {}, type {:?}",
        im.max(),
        im.min(),
        im.kind()
    );
}

#[test]
fn baseline_config_test() {
    let c = baseline_config();
    println!("{}", c);
}