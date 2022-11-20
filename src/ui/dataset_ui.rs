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

use super::Config;
use crate::datasets::{DatasetEnum, DatasetTypes, DatasetUI, mnist, self};


pub trait ViewerUI: Send + Sync + Config {
    fn load_dataset(&mut self, dataset: DatasetTypes) -> Result<()>;
    fn ui(&mut self, ui: &mut egui::Ui) -> Result<()>;
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
    pub viewers: HashMap<DatasetEnum, Box<dyn ViewerUI>>,
    viewer_is_active: bool,
    dataset_configs: HashMap<DatasetEnum, String>,
    viewer_configs: HashMap<DatasetEnum, String>,
    dataset_ui: HashMap<DatasetEnum, Box<dyn DatasetUI>>
}

impl Default for DatasetState {
    fn default() -> Self {
        DatasetState { 
            cur_data: DatasetEnum::MNIST,
            viewers: HashMap::new(),
            viewer_is_active: false,
            dataset_configs: HashMap::new(),
            viewer_configs: HashMap::new(),
            dataset_ui: HashMap::new() 
        }
    }
}

impl DatasetState {
    pub fn update(&mut self, ui: &mut egui::Ui) {
        let past_data = self.cur_data;
        ui.horizontal(|ui| {
            // Select which dataset to use
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("Datasets");
                    for opt in DatasetEnum::iter() {
                        ui.selectable_value(&mut self.cur_data, opt, opt.name());
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
                    match viewer.ui(ui) {
                        Err(err) => {
                            ui.group(|ui| {
                                let label = egui::RichText::new(err.to_string()).underline();
                                ui.label("could not load dataset:");
                                ui.label(label);
                            });
                        }
                        _ => {}
                    }
                }

                if ui.button("reset viewer").clicked() {
                    self.viewer_is_active = false;
                }
            });


    
        });
    }

    /// Bevy startup system for setting up the dataset viewer datasets
    pub fn setup(&mut self, root_path: &path::Path) {
        let data_subdir = root_path.join("datasets");
        let d_enum: Vec<_> = DatasetEnum::iter().collect();

        for s in d_enum {
            // getting the config path, specialized on the name of the dataset
            let user_path = data_subdir.join(s.name()).with_extension("ron");
            let mut viewer_config_name = s.name().to_string();
            viewer_config_name.push_str("config");
            let viewer_path = data_subdir.join(viewer_config_name).with_extension("ron");

            // if the user path exists, then load config from the user path
            // else use the default
            let mut param = s.get_param();
            let mut viewer_param = match_viewers(&s);
            let serialized: String;
            let serialized_viewer: String;
            if user_path.exists() {
                serialized = fs::read_to_string(&user_path).unwrap();
            } else {
                serialized = param.config();
            }
            if viewer_path.exists() {
                serialized_viewer = fs::read_to_string(&viewer_path).unwrap();
            } else {
                serialized_viewer = viewer_param.config();
            }
            // Each configuration should store two maps, ones which contains the 
            // original state, and one which contains the mutable state
            // subject to change throughout the app
            param.load_config(&serialized);
            viewer_param.load_config(&serialized_viewer);

            self.dataset_configs.insert(s, serialized);
            self.viewer_configs.insert(s, serialized_viewer);
            self.dataset_ui.insert(s, param);
            self.viewers.insert(s, viewer_param);
        }
        
    }

    pub fn save_params(&self, root_path: &path::Path) {
        let data_subdir = root_path.join("datasets");
        if !data_subdir.exists() {
            fs::create_dir_all(&data_subdir).unwrap();
        }

        for (dataset, param) in self.dataset_ui.iter() {
            // if the config has changed throughout the app, then save the new config
            // in the user store.
            if self.dataset_configs.contains_key(dataset) && self.dataset_configs[dataset] != param.config() {
                let user_path = data_subdir.join(dataset.name()).with_extension("ron");                
                fs::write(&user_path, param.config()).unwrap();
            }
        }

        for (dataset, viewer) in self.viewers.iter() {
            if self.viewer_configs.contains_key(&dataset) && self.viewer_configs[dataset] != viewer.config() {
                let mut viewer_config_name = dataset.name().to_string();
                viewer_config_name.push_str("config");
                let viewer_path = data_subdir.join(viewer_config_name).with_extension("ron");
                fs::write(&viewer_path, viewer.config()).unwrap();
            }
        }
    
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

    fn ui(&mut self, ui: &mut egui::Ui) -> Result<()> {
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

        Ok(())
    }
}

impl Config for ClassificationViewer {
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
        DatasetEnum::MNIST => Box::new(ClassificationViewer::default())
    }
}