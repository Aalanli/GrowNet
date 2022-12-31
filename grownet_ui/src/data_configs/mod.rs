use crate::UI;
use grownet_macros::derive_ui;
use model_lib::{datasets, Config};
use datasets::Transform;

use datasets::mnist::MnistParams;
derive_ui!(
    struct MnistParams {
        pub path: PathBuf,
        pub train_batch_size: usize,
        pub test_batch_size: usize,
    }
);

use datasets::cifar::Cifar10Params;
derive_ui!(
    pub struct Cifar10Params {
        pub path: path::PathBuf,
        pub train_batch_size: usize,
        pub test_batch_size: usize,
    }
);

use datasets::transforms::Normalize;
derive_ui!(
    pub struct Normalize {
        pub mu: f32,
        pub range: f32,
    }
);

use datasets::transforms::BasicImAugumentation;
derive_ui!(
    pub struct BasicImAugumentation {
        pub flip: bool,
        pub crop: i64,
        pub cutout: i64,
    }
);

