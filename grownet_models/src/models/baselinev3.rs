use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor, Shape},
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

#[test]
fn test_run() {
    use burn_ndarray::{NdArrayBackend, NdArrayDevice};
    use burn_autodiff::ADBackendDecorator;
    let device = NdArrayDevice::Cpu;
    type B = NdArrayBackend<f32>;
    use std::sync::Arc;
    use burn::{
        config::Config,
        data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
        tensor::backend::ADBackend,
        train::{
            metric::{AccuracyMetric, LossMetric},
            LearnerBuilder,
        },
    };

    let a = ndarray::Array2::<f32>::zeros([3, 4]);
    let vec = vec![0.0f32; 12];
    let b = Data::<f32, 2>::new(vec, Shape::new([3, 4]));
    let c = Tensor::<B, 2>::from_data(b.convert());


    let batcher_train = Arc::new(MNISTBatcher::<B>::new(device.clone()));
    let batcher_valid = Arc::new(MNISTBatcher::<B>::new(device.clone()));
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(5)
        .shuffle(3)
        .num_workers(1)
        .build(Arc::new(MNISTDataset::train()));
    
    for i in dataloader_train.iter() {
        
    }
}