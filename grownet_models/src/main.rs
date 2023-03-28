#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use arrayfire::*;
use model_lib::Options;
use model_lib::models::baselinev2::SimpleResnet;
use model_lib::nn::af_ops::conv::Conv2d;
use model_lib::models::baselinev2::{run_on_main, baseline_config};

fn main() {
    let mut config = baseline_config();
    config.update_key("epochs", &Options::INT(1)).unwrap();
    config.insert("dataset_path", &Options::STR("assets/ml_datasets".into())).unwrap();
    config.insert("train_log_steps", &Options::INT(500)).unwrap();
    set_device(0);
    let mut a = randn!(1);
    info();
    let _config1 = config.clone();
    let handle = std::thread::spawn(move || {
        set_device(0);
        a += randn!(1);
        // run_on_main(&config);
        // let model = SimpleResnet::<f32>::new(10);
        let a = randn!(28, 28, 3, 1);
        let conv = Conv2d::<f32>::new(3, 3, [3, 3], [1, 1], [1, 1], false);
        let y = conv.forward2(&a);
        // let (_y, _df) = model.forward(&a);
        y.eval();
    });
    handle.join().expect("first job");
    let mut a = randn!(1);

    println!("first job finished");
    let handle = std::thread::spawn(move || {
        set_device(0);
        a += randn!(1);
        println!("starting second job");

        // run_on_main(&config1);
        let model = SimpleResnet::<f32>::new(10);
        let a = randn!(28, 28, 3, 1);
        let (_y, _df) = model.forward(&a);
    });
    handle.join().expect("second job");
    println!("second job finished");
    
    // use std::thread;
    // set_device(0);
    // info();
    // let mut a = constant(1, dim4!(3, 3));

    // let handle = thread::spawn(move || {
    //     //set_device to appropriate device id is required in each thread
    //     set_device(0);

    //     println!("\nFrom thread {:?}", thread::current().id());

    //     a += constant(2, dim4!(3, 3));
    //     // print(&a);

    //     let w = randn!(3, 3, 3, 3);
    //     let x = randn!(28, 28, 3, 3);
    //     convolve2_nn(&x, &w, dim4!(1, 1), dim4!(1, 1), dim4!(1, 1));
        
    // });

    // //Need to join other threads as main thread holds arrayfire context
    // handle.join().unwrap();

    // let mut a = constant(1, dim4!(3, 3));
    // let handle = thread::spawn(move || {
    //     //set_device to appropriate device id is required in each thread
    //     set_device(0);

    //     println!("\nFrom thread {:?}", thread::current().id());

    //     let w = randn!(3, 3, 3, 3);
    //     let x = randn!(28, 28, 3, 3);
    //     convolve2_nn(&x, &w, dim4!(1, 1), dim4!(1, 1), dim4!(1, 1));
    //     a += constant(2, dim4!(3, 3));
    //     print(&a);
    // });

    // //Need to join other threads as main thread holds arrayfire context
    // handle.join().unwrap();

}