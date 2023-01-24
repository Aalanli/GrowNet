use crate::UI;
use grownet_macros::derive_ui;

use model_lib::models::baseline as md;

use md::SGD;
derive_ui!(
    #[derive(Derivative, Serialize, Deserialize, Clone, Debug)]
    #[derivative(Default)]
    pub struct SGD {
        #[derivative(Default(value = "0.9"))]
        pub momentum: f64,
        #[derivative(Default(value = "0.0"))]
        pub dampening: f64,
        #[derivative(Default(value = "5e-4"))]
        pub wd: f64,
        #[derivative(Default(value = "true"))]
        pub nesterov: bool,
    }
);

use md::ImTransform;
derive_ui!(
    #[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
    #[derivative(Default)]
    pub struct ImTransform {
        #[derivative(Default(value = "true"))]
        pub flip: bool,
        #[derivative(Default(value = "4"))]
        pub crop: i64,
        #[derivative(Default(value = "8"))]
        pub cutout: i64,
    }
);

pub use md::BaselineParams;
derive_ui!(
    #[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
    #[derivative(Default)]
    pub struct BaselineParams {
        pub sgd: SGD,
        pub transform: ImTransform,
        #[derivative(Default(value = "1.0"))]
        pub lr: f64,
        #[derivative(Default(value = "100"))]
        pub epochs: u32,
        #[derivative(Default(value = "4"))]
        pub batch_size: u32,
        pub data_path: String,
    }
);
