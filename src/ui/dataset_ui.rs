use std::{borrow::Cow, mem::MaybeUninit};
use std::collections::HashMap;
use std::path;
use std::fs;

use ndarray::{s, Axis};
use bevy::prelude::*;
use bevy::app::AppExit;
use bevy_egui::{egui, EguiContext};
use strum::IntoEnumIterator;
use anyhow::{Result, Error};

use super::ViewerUI;
use crate::datasets::{DatasetEnum, DatasetTypes, DatasetUI, mnist, self};


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
    pub viewer: Result<Box<dyn ViewerUI>>,
    dataset_configs: HashMap<DatasetEnum, String>,
    dataset_ui: HashMap<DatasetEnum, Box<dyn DatasetUI>>
}

impl Default for DatasetState {
    fn default() -> Self {
        DatasetState { 
            cur_data: DatasetEnum::MNIST, 
            viewer: Err(Error::msg("no dataset configured")),
            dataset_configs: HashMap::new(),
            dataset_ui: HashMap::new() 
        }
    }
}

impl DatasetState {
    pub fn update(&mut self, ui: &mut egui::Ui) {
        let past_data = self.cur_data;
        ui.horizontal(|ui| {
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("Datasets");
                    for opt in DatasetEnum::iter() {
                        ui.selectable_value(&mut self.cur_data, opt, opt.name());
                    }
                });
            });
            self.dataset_ui.get_mut(&self.cur_data).unwrap().ui(ui);
            if self.cur_data != past_data || self.viewer.is_err() {
                eprintln!("building {}", self.dataset_ui.get(&self.cur_data).unwrap().config());
                let built = self.dataset_ui.get(&self.cur_data).unwrap().build();
                match built {
                    Ok(data) => {
                        match data {
                            DatasetTypes::Classification(a) => {
                                self.viewer = Ok(Box::new(ClassificationViewer { data: a, texture: None }));
                            }
                        }
                    }
                    Err(err) => {self.viewer = Err(err); }
                }
            }
    
            match &mut self.viewer {
                Ok(data) => {data.ui(ui);}
                Err(err) => {
                    ui.group(|ui| {
                        let label = egui::RichText::new(err.to_string()).underline();
                        ui.label("could not load dataset:");
                        ui.label(label);
                    });
                }
            }
    
        });
    }

    /// Bevy startup system for setting up the dataset viewer datasets
    pub fn setup(&mut self, root_path: &path::Path) {
        let data_subdir = root_path.join("datasets");
        let d_enum: Vec<_> = DatasetEnum::iter().collect();

        for s in d_enum {
            // getting the config path, specialized on the name of the dataset
            let user_path = data_subdir.join(s.name()).with_extension("ron");

            // if the user path exists, then load config from the user path
            // else use the default
            let mut param = s.get_param();
            let serialized: String;
            if user_path.exists() {
                serialized = fs::read_to_string(&user_path).unwrap();
            } else {
                serialized = param.config();
            }
            eprintln!("{}", serialized);
            self.dataset_configs.insert(s, serialized.clone());
            // Each configuration should store two maps, ones which contains the 
            // original state, and one which contains the mutable state
            // subject to change throughout the app
            param.load_config(&serialized);
            self.dataset_ui.insert(s, param);
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
    
    }
}


/// Viewer for the Classification dataset type
struct ClassificationViewer {
    data: datasets::ClassificationType,
    texture: Option<egui::TextureHandle>
}

impl ViewerUI for ClassificationViewer {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            let texture = self.texture.get_or_insert_with(|| {
                let mut data_point = self.data.next();
                let data_point = data_point.get_or_insert_with(|| {
                    self.data.reset();
                    self.data.next().unwrap()
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
            let size = texture.size_vec2();
            ui.image(texture, size);
            if ui.button("next").clicked() {
                self.texture = None;
            }
        });
    }
}