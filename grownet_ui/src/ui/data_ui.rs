use std::sync::Mutex;

use anyhow::{Context, Error, Result};
use bevy::prelude::*;
use bevy_egui::egui;
use itertools::Itertools;
use ndarray::{Array, Array3, Ix4};

use super::{AppState, OperatingState, UIParams};
use crate::{Configure, UI, CONFIG_PATH};
use model_lib::datasets::{data, Dataset, DatasetBuilder};

pub struct DatasetUIPlugin;
impl Plugin for DatasetUIPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(setup_dataset_ui)
            .add_system_set(SystemSet::on_update(OperatingState::Close).with_system(save_dataset_ui));
    }
}

fn setup_dataset_ui(mut commands: Commands, params: Res<UIParams>) {
    use model_lib::datasets as data;

    let mut dataset_ui = DatasetUI::default();

    let cifar10 = data::cifar::Cifar10Params::default();
    let cifar_viewer = ClassificationViewer::new(cifar10);
    dataset_ui.push_viewer(cifar_viewer, "cifar10");

    let mnist = data::mnist::MnistParams::default();
    let mnist_viewer = ClassificationViewer::new(mnist);
    dataset_ui.push_viewer(mnist_viewer, "mnist");

    // load configurations from disk
    let root_path: std::path::PathBuf = CONFIG_PATH.into();
    let config_file = root_path.join("data_ui_config").with_extension("ron");

    if config_file.exists() {
        eprintln!("loading from config file {}", config_file.to_str().unwrap());
        let config: String =
            ron::from_str(&std::fs::read_to_string(&config_file).unwrap()).unwrap();
        if let Err(e) = dataset_ui.load_config(&config) {
            eprintln!("failed to load data config\n{}", e);
        }
    }

    commands.insert_resource(dataset_ui);
}

fn save_dataset_ui(params: Res<UIParams>, dataset_params: Res<DatasetUI>) {
    eprintln!("saving data_ui params");
    let data_ui_config = dataset_params.config();
    eprintln!("data ui params {}", data_ui_config);

    let root_path: std::path::PathBuf = CONFIG_PATH.into();
    if !root_path.exists() {
        std::fs::create_dir_all(&root_path).unwrap();
    }

    let config_file = root_path.join("data_ui_config").with_extension("ron");

    let serialized = ron::to_string(&data_ui_config).unwrap();
    std::fs::write(&config_file, serialized).unwrap();
}

#[derive(Default, Resource)]
pub struct DatasetUI {
    cur_active: usize,
    viewers: Vec<Box<dyn Viewer>>,
    names: Vec<&'static str>,
}

impl DatasetUI {
    pub fn push_viewer<T: Viewer + 'static>(&mut self, viewer: T, name: &'static str) {
        self.viewers.push(Box::new(viewer));
        self.names.push(name);
    }
}

impl UI for DatasetUI {
    fn ui(&mut self, ui: &mut egui::Ui) {
        //ui.spacing_mut().interact_size.y = ui.available_size().y;
        ui.horizontal(|ui| {
            // Select which dataset to use
            let last_active = self.cur_active;
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("Datasets");
                    for (i, name) in self.names.iter().enumerate() {
                        ui.selectable_value(&mut self.cur_active, i, *name);
                    }
                });
            });
            // if dataset changed, drop the dataset if there was any loaded, to save memory
            if last_active != self.cur_active {
                self.viewers[last_active].drop_dataset();
            }

            self.viewers[self.cur_active].ui(ui);
        });
    }
}

impl Configure for DatasetUI {
    fn config(&self) -> String {
        let viewers: Vec<_> = self.viewers.iter().map(|v| v.config()).collect_vec();
        ron::to_string(&viewers).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let config: Vec<String> = ron::from_str(config)?;
        for (i, s) in config.iter().enumerate() {
            self.viewers[i].load_config(s)?;
        }
        Ok(())
    }
}

pub trait Viewer: Configure + UI {
    fn drop_dataset(&mut self);
}

/// Viewer for the Classification dataset type
pub struct ClassificationViewer<D: DatasetBuilder> {
    train_data: Option<Mutex<D::Dataset>>,
    test_data: Option<Mutex<D::Dataset>>,
    params: D,
    train_texture: Option<Vec<egui::TextureHandle>>,
    test_texture: Option<Vec<egui::TextureHandle>>,
    im_scale: f32,
}

unsafe impl<D: DatasetBuilder> Send for ClassificationViewer<D> {}
unsafe impl<D: DatasetBuilder> Sync for ClassificationViewer<D> {}

impl<D, B> ClassificationViewer<B>
where
    D: Dataset<DataPoint = data::ImClassify>,
    B: DatasetBuilder<Dataset = D>,
{
    pub fn new(builder: B) -> Self {
        Self {
            train_data: None,
            test_data: None,
            params: builder,
            train_texture: None,
            test_texture: None,
            im_scale: 1.0,
        }
    }

    fn load_texture(data: &mut Mutex<D>, ui: &mut egui::Ui) -> Vec<egui::TextureHandle> {
        use ndarray::Axis;
        let data = &mut *data.lock().unwrap();
        let data_point = if let Some(x) = data.next() {
            x
        } else {
            data.reset();
            data.next().unwrap()
        };

        let batch_size = data_point.image.image.dim().0;

        let mut pixels: Vec<Vec<_>> = (0..batch_size)
            .map(|batch| {
                data_point
                    .image
                    .image
                    .index_axis(Axis(0), batch)
                    .as_slice()
                    .unwrap()
                    .chunks_exact(3)
                    .map(|x| {
                        egui::Color32::from_rgb(
                            (x[0] * 255.0) as u8,
                            (x[1] * 255.0) as u8,
                            (x[2] * 255.0) as u8,
                        )
                    })
                    .collect()
            })
            .collect();

        let size = data_point.image.size();
        let handles = (0..batch_size)
            .map(|_| {
                let color_image = egui::ColorImage {
                    size,
                    pixels: pixels.pop().unwrap(),
                };
                ui.ctx()
                    .load_texture("im sample", color_image, egui::TextureOptions::NEAREST)
            })
            .collect();
        handles
    }

    fn loading_logic(&mut self, ui: &mut egui::Ui) {
        let err_ui = |err: Error, ui: &mut egui::Ui| -> Option<D> {
            ui.label(format!("Error loading dataset {}", err.to_string()));
            None
        };
        // load the datasets if not loaded already
        if let None = self.train_data {
            let train_data = self
                .params
                .build_train()
                .map_or_else(|e| err_ui(e, ui), |x| Some(x));
            self.train_data = train_data.map(|x| Mutex::new(x));
        }
        if let None = self.test_data {
            if let Some(x) = self.params.build_test() {
                let test_data = x.map_or_else(|e| err_ui(e, ui), |x| Some(x));
                self.test_data = test_data.map(|x| Mutex::new(x));
            }
        }

        // load a data point if not loaded already
        if let Some(data) = &mut self.train_data {
            if let None = self.train_texture {
                self.train_texture = Some(Self::load_texture(data, ui));
            }
        }
        if let Some(data) = &mut self.test_data {
            if let None = self.test_texture {
                self.test_texture = Some(Self::load_texture(data, ui));
            }
        }
    }

    fn display_im_if_any(&self, textures: &Option<Vec<egui::TextureHandle>>, ui: &mut egui::Ui) {
        let im_scale = self.im_scale;
        if let Some(images) = &textures {
            for im in images {
                let mut size = im.size_vec2();
                size *= im_scale;
                ui.image(im, size);
            }
        }
    }
}

impl<D, B> UI for ClassificationViewer<B>
where
    D: Dataset<DataPoint = data::ImClassify>,
    B: DatasetBuilder<Dataset = D> + UI,
{
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            self.params.ui(ui);
            self.loading_logic(ui);
        });
        ui.vertical(|ui| {
            egui::containers::panel::SidePanel::right("test images").show(ui.ctx(), |ui| {
                ui.vertical(|ui| {
                    egui::containers::ScrollArea::vertical()
                        .id_source("test images")
                        .show(ui, |ui| {
                            self.display_im_if_any(&self.test_texture, ui);
                            if let Some(_) = &self.test_texture {
                                if ui.button("next test").clicked() {
                                    self.test_texture = None;
                                }
                            }
                            if let Some(data) = &mut self.test_data {
                                if ui.button("shuffle test").clicked() {
                                    (*data.lock().unwrap()).shuffle();
                                }
                            }
                            if ui.button("reset test").clicked() {
                                self.test_data = None;
                                self.test_texture = None;
                            }
                        });
                    //egui::CollapsingHeader::new("test images")
                    //    .default_open(true)
                    //    .show(ui, |ui| {
                    //    self.display_im_if_any(&self.test_texture, ui);
                    //});
                });
            });

            egui::containers::panel::SidePanel::right("train images").show(ui.ctx(), |ui| {
                ui.vertical(|ui| {
                    egui::containers::ScrollArea::vertical()
                        .id_source("train images")
                        .show(ui, |ui| {
                            self.display_im_if_any(&self.train_texture, ui);
                            if let Some(_) = &self.train_texture {
                                if ui.button("next train").clicked() {
                                    self.train_texture = None;
                                }
                            }
                            if let Some(data) = &mut self.train_data {
                                if ui.button("shuffle train").clicked() {
                                    (*data.lock().unwrap()).shuffle();
                                }
                            }
                            if ui.button("reset train").clicked() {
                                self.train_data = None;
                                self.train_texture = None;
                            }
                        });

                    //egui::CollapsingHeader::new("train images")
                    //    .default_open(true)
                    //    .show(ui, |ui| {
                    //    self.display_im_if_any(&self.train_texture, ui);
                    //});
                });
            });

            ui.label("image scale");
            ui.add(egui::Slider::new(&mut self.im_scale, 0.1..=10.0));
        });
    }
}

impl<D, B> Configure for ClassificationViewer<B>
where
    D: Dataset<DataPoint = data::ImClassify>,
    B: DatasetBuilder<Dataset = D>,
{
    fn config(&self) -> String {
        let self_params = ron::to_string(&self.im_scale).unwrap();
        let data_params = self.params.config();
        ron::to_string(&(self_params, data_params)).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let (self_params, data_params): (String, String) =
            ron::from_str(config).context("Classification Viewer")?;
        let scale: f32 = ron::from_str(&self_params).context("Classification Viewer")?;
        self.params
            .load_config(&data_params)
            .context("Classification viewer dataset parameters")?;
        self.im_scale = scale;
        Ok(())
    }
}

impl<D, B> Viewer for ClassificationViewer<B>
where
    D: Dataset<DataPoint = data::ImClassify>,
    B: DatasetBuilder<Dataset = D> + UI,
{
    fn drop_dataset(&mut self) {
        self.train_data = None;
        self.test_data = None;
    }
}
