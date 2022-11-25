use anyhow::{Result, Error, Context};
use bevy_egui::egui;
use ndarray::Axis;

use crate::{Param, Config, UI};
use crate::datasets::{self, TransformTypes, transforms};
use crate::datasets::DatasetTypes;

/// Each viewer maintains its own internal state, and manipulates the dataset
/// fed to it
pub trait ViewerUI: Config + UI {
    fn load_dataset(&mut self, dataset: DatasetTypes) -> Result<()>;
    fn load_transform(&mut self, transform: TransformTypes) -> Result<()>;
}


pub struct EmptyViewer {}

impl UI for EmptyViewer {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("No viewer implemented for this dataset.");
    }
}

impl Config for EmptyViewer {
    fn config(&self) -> String {
        "".to_string()
    }
    fn load_config(&mut self, _config: &str) -> Result<()> {Ok(())}
}

impl ViewerUI for EmptyViewer {
    fn load_dataset(&mut self, _dataset: DatasetTypes) -> Result<()> {
        Ok(())
    }
    fn load_transform(&mut self, _transform: TransformTypes) -> Result<()> {
        Ok(())
    }
}

/// Viewer for the Classification dataset type
pub struct ClassificationViewer {
    data: Option<datasets::ClassificationType>,
    transform: Option<datasets::ClassificationTransform>,
    texture: Option<egui::TextureHandle>,
    im_scale: f32
}

impl ViewerUI for ClassificationViewer {
    fn load_dataset(&mut self, dataset: datasets::DatasetTypes) -> Result<()> {
        let DatasetTypes::Classification(d) = dataset;
        self.data = Some(d);
        Ok(())
    }
    fn load_transform(&mut self, transform: TransformTypes) -> Result<()> {
        match transform {
            TransformTypes::Classification(t) => {
                self.transform = Some(t);
                Ok(())
            }
            TransformTypes::Identity => {
                self.transform = None;
                Ok(())
            }
        }
    }
}

impl UI for ClassificationViewer {
    fn ui(&mut self, ui: &mut egui::Ui) {
        if let Some(data) = &mut self.data {
            let texture = self.texture.get_or_insert_with(|| {
                let data_point = if let Some(x) = data.next() {
                    x
                } else {
                    data.reset();
                    data.next().unwrap()
                };
    
                let data_point = if let Some(t) = &self.transform {
                    t.transform(data_point)
                } else {
                    data_point
                };
    
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
        } else {
            ui.label("No dataset loaded");
        }
    }
}

impl Config for ClassificationViewer {
    fn config(&self) -> String {
        ron::to_string(&self.im_scale).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let scale: f32 = ron::from_str(config).context("Classification Viewer")?;
        self.im_scale = scale;
        Ok(())
    }
}

impl Default for ClassificationViewer {
    fn default() -> Self {
        ClassificationViewer { data: None, transform: None, texture: None, im_scale: 1.0 }
    }
}
