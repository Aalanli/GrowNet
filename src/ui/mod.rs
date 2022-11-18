use std::{borrow::Cow, mem::MaybeUninit};

use ndarray::{s, Axis};
use strum::IntoEnumIterator;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};

use crate::datasets::{DatasetParams, DatasetEnum, DatasetTypes, mnist::{MnistParams, Mnist}, self};

/// The state for the entire app, which characterizes the two main modes of operation
/// Menu involves only light ui tasks, while Trainer may involve some heavy compute, 
/// (which may run on another thread), and rendering.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
enum State {
    Menu,
    Trainer
}

pub struct UI;
impl Plugin for UI {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(UIParams::default())
            .insert_resource(DatasetState::default())
            .add_system(menu_ui);
    }
}

fn menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut params: ResMut<UIParams>,
    mut dataset_state: ResMut<DatasetState>,
    mut dataset_params: ResMut<DatasetParams>
) {
    
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Label::new("Data Explorer"));

        ui.horizontal(|ui| {
            // The four possible states for the ui to be in,
            // selecting "Train" switches to the Trainer app state
            ui.selectable_value(&mut params.open_panel, OpenPanel::Models, "Models");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Datasets, "Datasets");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Misc, "Misc");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Train, "Train");
        });
        ui.separator();

        match params.open_panel {
            OpenPanel::Models => {},
            OpenPanel::Datasets => {update_dataset(&mut dataset_state, &mut *dataset_params, ui)},
            OpenPanel::Misc => {update_misc(&mut params.misc, ui)},
            OpenPanel::Train => {},
        }


    });
}

fn update_dataset(dataset: &mut DatasetState, dataset_params: &mut DatasetParams, ui: &mut egui::Ui) {
    let past_data = dataset.cur_data;
    ui.horizontal(|ui| {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label("Datasets");
                for opt in DatasetEnum::iter() {
                    ui.selectable_value(&mut dataset.cur_data, opt, opt.name());
                }
            });
        });
        dataset_params.0.get_mut(&dataset.cur_data).unwrap().ui(ui);
        if dataset.cur_data != past_data || dataset.viewer.is_none() {
            eprintln!("built {}", dataset_params.0.get(&dataset.cur_data).unwrap().config());
            let built = dataset_params.0.get(&dataset.cur_data).unwrap().build();
            match built {
                DatasetTypes::Classification(a) => {
                    dataset.viewer = Some(Box::new(ClassificationViewer { data: a, texture: None }));
                }
            }
        }

        if let Some(viewer) = &mut dataset.viewer {
            viewer.ui(ui);
        }

    });
}

trait ViewerUI: Send + Sync {
    fn ui(&mut self, ui: &mut egui::Ui);
}

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


fn update_misc(misc: &mut Misc, ui: &mut egui::Ui) {
    let mut local_font_delta = misc.font_delta;
    // stylistic changes
    ui.collapsing("styling", |ui| {
        ui.label("font size delta");
        ui.add(egui::Slider::new(&mut local_font_delta, -9.0..=12.0));
    });

    if local_font_delta != misc.font_delta {
        change_font_size(local_font_delta, ui.ctx());
        misc.font_delta = local_font_delta;
    }
}


struct UIParams {
    open_panel: OpenPanel,
    misc: Misc
}

impl Default for UIParams {
    fn default() -> Self {
        UIParams { misc: Misc::default(), open_panel: OpenPanel::Models }
    }
}

#[derive(PartialEq, Eq)]
enum OpenPanel {
    Models,
    Datasets,
    Misc,
    Train
}

struct Misc {
    font_delta: f32
}

impl Default for Misc {
    fn default() -> Self {
        Misc { font_delta: 4.0 }
    }
}

struct DatasetState {
    pub cur_data: DatasetEnum,
    pub viewer: Option<Box<dyn ViewerUI>>
}

impl Default for DatasetState {
    fn default() -> Self {
        DatasetState { cur_data: DatasetEnum::MNIST, viewer: None }
    }
}

fn change_font_size(font_delta: f32, ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(egui::TextStyle::Body, egui::FontId::new(18.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Monospace, egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Button, egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Small, egui::FontId::new(10.0 + font_delta, egui::FontFamily::Proportional));
    ctx.set_style(style);

}