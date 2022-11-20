use std::collections::HashMap;

use bevy_egui::egui;
use anyhow::Result;

mod viewers;
use viewers::ViewerUI;

mod mnist;
pub use mnist::MNIST;

use super::Param;
use crate::datasets::{DatasetTypes, TransformTypes};

/// To integrate with the ui, each dataset defines this trait
pub trait DatasetSetup {
    fn parameters() -> Box<dyn DatasetBuilder>;
    fn viewer() -> Box<dyn ViewerUI>;
    fn transforms() -> Vec<TransformTypes>;
    fn name() -> &'static str;
}

/// Each Dataset has two structs, that of the parameters it holds
/// and the data that it holds, this trait is implemented on the parameters for the dataset
/// this separation is made as the parameters are usually light and copyible, while
/// the data is not light, and require some non-negliable compute for the setup.
/// This trait adjusts the parameters, and builds the dataset on the parameters it holds.
pub trait DatasetBuilder: Param {
    fn build(&self) -> Result<DatasetTypes>;
}

/// Main state for the dataset section of the ui
/// 
/// cur_data: The current active dataset in the viewer
/// viewer_is_active: If there is a currently active viewer, useful for reloading logic
/// 
/// viewers: 
///     A hash map containing the ViewerUI object for each variant of the dataset, each viewer
///     handles dataset visualization, as characterized by the DatasetType
/// 
/// dataset_ui:
///     A hash map containing the DatasetUI object for each variant of the dataset, which gets set
///     through the ui
pub struct DatasetState {
    pub cur_data: &'static str,
    viewer_is_active: bool,
    viewers: HashMap<&'static str, Box<dyn ViewerUI>>,
    dataset_ui: HashMap<&'static str, Box<dyn DatasetBuilder>>,
    transforms: HashMap<&'static str, Vec<TransformTypes>>
}

impl DatasetState {
    pub fn new() -> Self {
        DatasetState { 
            cur_data: "mnist",
            viewer_is_active: false,
            viewers: HashMap::new(),
            dataset_ui: HashMap::new(),
            transforms: HashMap::new()
        }
    }
    pub fn insert_dataset<D: DatasetSetup>(&mut self) {
        self.dataset_ui.insert(D::name(), D::parameters());
        self.viewers.insert(D::name(), D::viewer());
        self.transforms.insert(D::name(), D::transforms());
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
                    for opt in self.dataset_ui.keys() {
                        ui.selectable_value(&mut self.cur_data, opt, *opt);
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
                                    self.viewers.insert(self.cur_data, Box::new(viewers::ClassificationViewer::default()));
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
        let data_ui_configs: HashMap<String, String> = self.dataset_ui
            .iter().map(|(k, v)| (k.to_string(), v.config())).collect();

        let viewer_configs: HashMap<String, String> = self.viewers
            .iter().map(|(k, v)| (k.to_string(), v.config())).collect();

        let transform_configs: HashMap<String, String> = self.transforms
            .iter().map(|(k, v)| {
                let temp: Vec<String> = v.iter().map(|x| {
                    match x {
                        TransformTypes::Classification(x) => x.config()
                    }
                }).collect();
                (k.to_string(), ron::to_string(&temp).unwrap())
            }).collect();

        let configs = (data_ui_configs, viewer_configs, transform_configs);
        ron::to_string(&configs).unwrap()
    }

    fn load_config(&mut self, config: &str) {
        type PartialConfig = HashMap<String, String>;
        let configs: (PartialConfig, PartialConfig, PartialConfig) = ron::from_str(config).unwrap();
        let keys: Vec<_> = self.dataset_ui.keys().cloned().collect();
        for k in &keys {
            self.dataset_ui.get_mut(k).unwrap().load_config(&configs.0[*k]);
        }
        let keys: Vec<_> = self.viewers.keys().cloned().collect();
        for k in &keys {
            self.viewers.get_mut(k).unwrap().load_config(&configs.1[*k]);
        }
        let keys: Vec<_> = self.transforms.keys().cloned().collect();
        for k in &keys {
            let temp: Vec<String> = ron::from_str(&configs.2[*k]).unwrap();
            self.transforms.get_mut(k).unwrap().iter_mut().zip(temp.iter()).for_each(|(x, st)| match x {
                TransformTypes::Classification(x) => x.load_config(st)
            });
        }
    }
}


