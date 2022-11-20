use anyhow::Result;
use bevy_egui::egui;
use ndarray::Axis;

use super::Param;
use crate::datasets;
use crate::datasets::DatasetTypes;

/// Each viewer maintains its own internal state, and manipulates the dataset
/// fed to it
pub trait ViewerUI: Param {
    fn load_dataset(&mut self, dataset: DatasetTypes) -> Result<()>;
}


pub struct EmptyViewer {}

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
pub struct ClassificationViewer {
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
