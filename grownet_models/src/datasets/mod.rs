use serde::{Serialize, Deserialize};
use tch;
use ndarray::prelude::*;
use anyhow::Result;

pub mod transforms {
    use serde::{Serialize, Deserialize};
    use tch::Tensor;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct BasicImAugumentation {
        pub flip: bool,
        pub crop: i64,
        pub cutout: i64,
    }

    impl BasicImAugumentation {
        pub fn transform(&self, x: &Tensor) -> Tensor {
            tch::vision::dataset::augmentation(x, self.flip, self.crop, self.cutout)
        } 
    }

    impl Default for BasicImAugumentation {
        fn default() -> Self {
            Self { flip: true, crop: 4, cutout: 8 }
        }
    }

}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TorchCifar10Params {
    pub path: std::path::PathBuf,
    pub batch_sz: i64,
    pub aug: transforms::BasicImAugumentation,
}

impl TorchCifar10Params {
    pub fn build(&self) -> Result<tch::vision::dataset::Dataset> {
        Ok(tch::vision::cifar::load_dir(&self.path)?)
    }
}