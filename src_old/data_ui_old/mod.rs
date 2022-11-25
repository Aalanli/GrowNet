use std::collections::HashMap;

use bevy_egui::egui;
use anyhow::{Result, Context};

mod viewers;
use viewers::ViewerUI;

mod mnist;
pub use mnist::MNIST;

use super::{Param, Config, UI};
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
pub trait DatasetBuilder: Config + UI {
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
    transforms: HashMap<&'static str, Vec<TransformTypes>>,
    selected_transform: HashMap<&'static str, (usize, bool)>
}

impl DatasetState {
    pub fn new() -> Self {
        DatasetState { 
            cur_data: "mnist",
            viewer_is_active: false,
            viewers: HashMap::new(),
            dataset_ui: HashMap::new(),
            transforms: HashMap::new(),
            selected_transform: HashMap::new(),
        }
    }
    pub fn insert_dataset<D: DatasetSetup>(&mut self) {
        self.dataset_ui.insert(D::name(), D::parameters());
        self.viewers.insert(D::name(), D::viewer());
        let mut transforms = D::transforms();
        transforms.push(TransformTypes::Identity);
        self.transforms.insert(D::name(), transforms);
        self.selected_transform.insert(D::name(), (0, false));
    }
}


impl UI for DatasetState {
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
            // get the past transform value, for later comparison
            let cur_transform = self.selected_transform.get(self.cur_data).unwrap().0;
            // display the ui for the selected dataset
            ui.vertical(|ui| {
                self.dataset_ui.get_mut(&self.cur_data).map(|x| {x.ui(ui);});
                // display any potential transforms
                let t = self.transforms.get_mut(self.cur_data).unwrap();
                // get the transform index, which points at the position in the list of transforms to load
                let transform_idx = &mut self.selected_transform.get_mut(self.cur_data).unwrap().0;
                for (i, h) in (0..t.len()).zip(t.iter_mut()) {
                    ui.group(|ui| {
                        let mut checked = i == *transform_idx;
                        ui.checkbox(&mut checked, format!("transform {}", i));
                        // update the index to the selected transform
                        if checked {
                            *transform_idx = i;
                        }

                        match h {
                            TransformTypes::Classification(h) => h.ui(ui),
                            TransformTypes::Identity => {
                                ui.label("identity transform");
                            }
                        };
                    });
                    ui.end_row();
                }
            });

            ui.separator();
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
                                let cur_viewer = self.viewers.get_mut(&self.cur_data).unwrap();
                                cur_viewer.load_dataset(DatasetTypes::Classification(a)).unwrap();
                            }
                        }
                    }
                    Err(err) => {
                        ui.label(egui::RichText::new(err.to_string()).underline());
                    }
                }
                self.viewer_is_active = true;
            }

            // if the transform changed, then load the new transform into the viewer
            let (new_transform_idx, is_loaded) = self.selected_transform.get_mut(self.cur_data).unwrap();
            if *new_transform_idx != cur_transform || !*is_loaded {
                *is_loaded = true;
                // get and clone transform
                let transform = &self.transforms.get(self.cur_data).unwrap()[*new_transform_idx];
                let transform = transform.clone();
                // then load into viewer
                eprintln!("loading transform");
                self.viewers.get_mut(&self.cur_data).unwrap().load_transform(transform).unwrap();
            }
            

            // display the current active viewer
            ui.vertical(|ui| {
                if self.viewer_is_active {
                    let viewer = self.viewers.get_mut(&self.cur_data).unwrap();
                    viewer.ui(ui);
                }

                ui.horizontal(|ui| {
                    if ui.button("reset viewer").clicked() {
                        self.viewer_is_active = false;
                    }
                    if ui.button("reset transform").clicked() {
                        *is_loaded = false;
                    }
                });
            });
        });
    }
}

impl Config for DatasetState {
    fn config(&self) -> String {
        let data_ui_configs: HashMap<String, String> = self.dataset_ui
            .iter().map(|(k, v)| (k.to_string(), v.config())).collect();

        let viewer_configs: HashMap<String, String> = self.viewers
            .iter().map(|(k, v)| (k.to_string(), v.config())).collect();

        let transform_configs: HashMap<String, String> = self.transforms
            .iter().map(|(k, v)| {
                let temp: Vec<String> = v.iter().map(|x| {
                    match x {
                        TransformTypes::Classification(x) => x.config(),
                        _ => "".to_string()
                    }
                }).collect();
                (k.to_string(), ron::to_string(&temp).unwrap())
            }).collect();
        
        let configs = (data_ui_configs, viewer_configs, transform_configs, self.selected_transform.clone());
        ron::to_string(&configs).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        type PartialConfig = HashMap<String, String>;
        let configs: (PartialConfig, PartialConfig, PartialConfig, HashMap<String, (usize, bool)>) = ron::from_str(config).context("Dataset State")?;
        let keys: Vec<_> = self.dataset_ui.keys().cloned().collect();
        for k in &keys {
            self.dataset_ui.get_mut(k).unwrap().load_config(&configs.0[*k])?;
        }
        let keys: Vec<_> = self.viewers.keys().cloned().collect();
        for k in &keys {
            self.viewers.get_mut(k).unwrap().load_config(&configs.1[*k])?;
        }
        let keys: Vec<_> = self.transforms.keys().cloned().collect();
        for k in &keys {
            let temp: Vec<String> = ron::from_str(&configs.2[*k]).unwrap();
            for (x, st) in self.transforms.get_mut(k).unwrap().iter_mut().zip(temp.iter()) {
                match x {
                    TransformTypes::Classification(x) => {x.load_config(st)?;},
                    _ => {}
                }
            }
        }

        self.selected_transform.iter_mut().for_each(|(k, v)| {
            if let Some(s) = configs.3.get(*k) {
                *v = *s;
            }
        });
        Ok(())
    }
}


