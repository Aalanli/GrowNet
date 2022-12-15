use std::sync::Arc;
use anyhow::{Result, Error};

use bevy_egui::egui::mutex::Mutex;
use model_lib::datasets::TorchCifar10Params;
use model_lib::{datasets};
use crate::ui::data_ui::Viewer;

use super::{UI, Config, ConfigWrapper};
use super::ui::data_ui::ClassificationViewer;

use ndarray::prelude::*;
use bevy_egui::egui;
use tch::{self, Tensor};
use tch::vision::dataset::Dataset;
use tch::data::Iter2;


impl UI for datasets::transforms::BasicImAugumentation {
    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
        ui.vertical(|ui| {
            ui.checkbox(&mut self.flip, "flip");
            ui.label("crop");
            ui.add(egui::DragValue::new(&mut self.crop));
            ui.label("cutout");
            ui.add(egui::DragValue::new(&mut self.cutout));
        });
    }
}

impl UI for datasets::TorchCifar10Params {
    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
        let mut text = self.path.to_str().unwrap().to_owned();
        ui.label("dataset path");
        ui.text_edit_singleline(&mut text);
        ui.label("batch size");
        ui.add(egui::DragValue::new(&mut self.batch_sz));
        ui.label("transform parameters");
        self.aug.ui(ui);
    }
}

struct Cifar10ViewerState {
    dataset: Mutex<Dataset>,
    train_iter: Mutex<Iter2>,
    test_iter: Mutex<Iter2>
}

impl Cifar10ViewerState {
    fn new(config: &TorchCifar10Params) -> Result<Self> {
        let dataset = config.build()?;
        let train_iter = Mutex::new(dataset.test_iter(config.batch_sz));
        let test_iter = Mutex::new(dataset.train_iter(config.batch_sz));
        Ok(Self { dataset: Mutex::new(dataset), train_iter, test_iter })
    }
}

pub fn convert_image_tensor(t: &Tensor) -> Result<Array<f32, Ix4>> {
    let dims: Vec<_> = t.size().iter().map(|x| *x as usize).collect();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::msg("tensor shape, expect NCWH"));
    }
    let t = t.to_device(tch::Device::Cpu).to_dtype(tch::Kind::Float, false, true);
    let ptr = t.as_ptr() as *const f32;
    let arr = unsafe{ ArrayView4::<f32>::from_shape_ptr((dims[0], dims[1], dims[2], dims[3]), ptr).to_owned() };
    Ok(arr)
}

pub fn build_viewer() -> impl Viewer {
    let mut viewer: ClassificationViewer<ConfigWrapper<_, Cifar10ViewerState>> = ClassificationViewer::new(
        ConfigWrapper::new(datasets::TorchCifar10Params::default()));
    
    viewer.set_next_train_fn(|d| {
        if let Some(s) = &mut d.state {
            let titer = &mut *s.train_iter.lock();
            
            let ts = titer.next();
            let ts = if let Some(x) = ts {
                x
            } else {
                let data = s.dataset.lock().train_iter(d.config.batch_sz);
                *titer = data;
                titer.next().unwrap()
            };
            convert_image_tensor(&ts.0)
        } else {
            Err(Error::msg("failed to load dataset"))
        }
    });

    viewer.set_next_test_fn(|d| {
        if let Some(s) = &mut d.state {
            let titer = &mut *s.test_iter.lock();
            
            let ts = titer.next();
            let ts = if let Some(x) = ts {
                x
            } else {
                let data = s.dataset.lock().train_iter(d.config.batch_sz);
                *titer = data;
                titer.next().unwrap()
            };
            convert_image_tensor(&ts.0)
        } else {
            Err(Error::msg("failed to load dataset"))
        }
    });

    viewer.set_shuffle_train_fn(|d| {
        if let Some(s) = &mut d.state {
            let titer = &mut *s.train_iter.lock();
            titer.shuffle();
        }
    });

    viewer.set_shuffle_test_fn(|d| {
        if let Some(s) = &mut d.state {
            let titer = &mut *s.test_iter.lock();
            titer.shuffle();
        }
    });

    viewer.set_drop_fn(|d| {
        d.state = None;
    });

    viewer.set_load_fn(|d| {
        match Cifar10ViewerState::new(&d.config) {
            Ok(s) => {d.state = Some(s); Ok(())}
            Err(e) => { Err(e) }
        }
    });
    viewer
}