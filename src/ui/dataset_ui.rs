use std::{borrow::Cow, mem::MaybeUninit};
use std::collections::HashMap;
use std::path;
use std::fs;

use ndarray::{s, Axis};
use bevy::prelude::*;
use bevy::app::AppExit;
use bevy_egui::{egui, EguiContext};
use serde::{Deserialize, Serialize};
use strum::{IntoEnumIterator, EnumIter};
use anyhow::{Result, Error};

use image::{ImageBuffer, RgbImage};

use super::{Param, serialize_hashmap, deserialize_hashmap};
use crate::datasets::{DatasetEnum, DatasetTypes, DatasetBuilder, mnist, self};


pub trait ViewerUI: Send + Sync + Param {
    fn load_dataset(&mut self, dataset: DatasetTypes) -> Result<()>;
}


/// Main state for the dataset section of the ui
/// 
/// cur_data: The current active dataset in the viewer
/// viewer: The current active viewer, err if no viewer is active
/// 
/// dataset_configs: 
///     A hash map containing serialized versions of dataset parameters (captured by dataset_ui) 
///     on app startup, so that upon app shutdown, if the serializations differ, then save the new
///     serialization
/// 
/// dataset_ui:
///     A hash map containing the DatasetUI object for each variant of the dataset, which gets set
///     through the ui
pub struct DatasetState {
    pub cur_data: DatasetEnum,
    viewer_is_active: bool,
    viewers: HashMap<DatasetEnum, Box<dyn ViewerUI>>,
    dataset_ui: HashMap<DatasetEnum, Box<dyn DatasetBuilder>>
}

impl Default for DatasetState {
    fn default() -> Self {
        DatasetState { 
            cur_data: DatasetEnum::MNIST,
            viewer_is_active: false,
            viewers: DatasetEnum::iter().map(|k| (k, match_viewers(&k))).collect(),
            dataset_ui: DatasetEnum::iter().map(|k| (k, k.get_param())).collect() 
        }
    }
}

impl Param for DatasetState {
    fn ui(&mut self, ui: &mut egui::Ui) {
        let past_data = self.cur_data;
        ui.horizontal(|ui| {
            // Select which dataset to use
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("Datasets");
                    for opt in DatasetEnum::iter() {
                        ui.selectable_value(&mut self.cur_data, opt, ron::to_string(&opt).unwrap());
                    }
                });
            });
            // display the ui for the selected dataset
            self.dataset_ui.get_mut(&self.cur_data).unwrap().ui(ui);
            // if dataset changed, then load the new dataset into the viewer
            if self.cur_data != past_data || !self.viewer_is_active {
                ui.label(format!("building {}", self.dataset_ui.get(&self.cur_data).unwrap().config()));
                let built = self.dataset_ui.get(&self.cur_data).unwrap().build();
                match built {
                    Ok(data) => {
                        match data {
                            DatasetTypes::Classification(a) => {
                                if !self.viewers.contains_key(&self.cur_data) {
                                    self.viewers.insert(self.cur_data, Box::new(ClassificationViewer::default()));
                                }
                                self.viewers.get_mut(&self.cur_data).unwrap().load_dataset(DatasetTypes::Classification(a)).unwrap();
                            }
                        }
                    }
                    Err(err) => {
                        ui.label(egui::RichText::new(err.to_string()).underline());
                    }
                }
                self.viewer_is_active = true;
            }

            // display the current active viewer
            ui.vertical(|ui| {
                if self.viewer_is_active {
                    let viewer = self.viewers.get_mut(&self.cur_data).unwrap();
                    viewer.ui(ui);
                }

                if ui.button("reset viewer").clicked() {
                    self.viewer_is_active = false;
                }
            });
        });
    }

    fn config(&self) -> String {
        let data_ui_configs: HashMap<DatasetEnum, String> = self.dataset_ui
            .iter().map(|(k, v)| (k.clone(), v.config())).collect();

        let viewer_configs: HashMap<DatasetEnum, String> = self.viewers
            .iter().map(|(k, v)| (k.clone(), v.config())).collect();

        let configs = (data_ui_configs, viewer_configs);
        ron::to_string(&configs).unwrap()
    }

    fn load_config(&mut self, config: &str) {
        type PartialConfig = HashMap<DatasetEnum, String>;
        let configs: (PartialConfig, PartialConfig) = ron::from_str(config).unwrap();
        let keys: Vec<_> = self.dataset_ui.keys().cloned().collect();
        for k in &keys {
            self.dataset_ui.get_mut(k).unwrap().load_config(&configs.0[k]);
        }
        let keys: Vec<_> = self.viewers.keys().cloned().collect();
        for k in &keys {
            self.viewers.get_mut(k).unwrap().load_config(&configs.1[k]);
        }
    }
}


struct EmptyViewer {}

impl Param for EmptyViewer {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("No viewer implemented for this dataset.");
    }
    fn config(&self) -> String {
        "".to_string()
    }
    fn load_config(&mut self, _config: &str) {}
}

impl ViewerUI for EmptyViewer {
    fn load_dataset(&mut self, _dataset: DatasetTypes) -> Result<()> {
        Ok(())
    }
}

/// Viewer for the Classification dataset type
struct ClassificationViewer {
    data: Option<datasets::ClassificationType>,
    texture: Option<egui::TextureHandle>,
    im_scale: f32
}

impl ViewerUI for ClassificationViewer {
    fn load_dataset(&mut self, dataset: datasets::DatasetTypes) -> Result<()> {
        let DatasetTypes::Classification(d) = dataset;
        self.data = Some(d);
        Ok(())
    }
}

impl Param for ClassificationViewer {
    fn ui(&mut self, ui: &mut egui::Ui) {
        if let Some(data) = &mut self.data {
            ui.group(|ui| {
                let texture = self.texture.get_or_insert_with(|| {
                    let mut data_point = data.next();
                    let data_point = data_point.get_or_insert_with(|| {
                        data.reset();
                        data.next().unwrap()
                    });
                    let pixels: Vec<_> = data_point.image.image
                        .index_axis(Axis(0), 0)
                        .as_slice()
                        .unwrap()
                        .chunks_exact(3)
                        .map(|x| {
                            egui::Color32::from_rgb((x[0] * 255.0) as u8, (x[0] * 255.0) as u8, (x[0] * 255.0) as u8)
                        }).collect();
                    let size = data_point.image.size();
                    let color_image = egui::ColorImage { size, pixels };
    
                    ui.ctx().load_texture("im sample", color_image, egui::TextureFilter::Nearest)
                });

                let mut size = texture.size_vec2();
                size *= self.im_scale;
                ui.image(texture, size);

                ui.horizontal_centered(|ui| {
                    if ui.button("next").clicked() {
                        self.texture = None;
                    }
        
                    if ui.button("shuffle").clicked() {
                        data.shuffle();
                    }
                });

                ui.label("image scale");
                ui.add(egui::Slider::new(&mut self.im_scale, 0.1..=10.0));
            });
        } else {
            ui.label("No dataset loaded");
        }
    }
    fn config(&self) -> String {
        ron::to_string(&self.im_scale).unwrap()
    }

    fn load_config(&mut self, config: &str) {
        let scale: f32 = ron::from_str(config).unwrap();
        self.im_scale = scale;
    }
}

impl Default for ClassificationViewer {
    fn default() -> Self {
        ClassificationViewer { data: None, texture: None, im_scale: 1.0 }
    }
}


fn match_viewers(dataset: &DatasetEnum) -> Box<dyn ViewerUI> {
    match dataset {
        DatasetEnum::MNIST => Box::new(ClassificationViewer::default()),
        //DatasetEnum::CIFAR10 => Box::new(ClassificationViewer::default())
    }
}