use std::{borrow::Cow, mem::MaybeUninit};

use strum::IntoEnumIterator;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};

use crate::datasets::{DatasetParams, PossibleDatasets, mnist::{MnistParams, Mnist}};

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
    ui.horizontal(|ui| {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label("Datasets");
                for opt in PossibleDatasets::iter() {
                    ui.selectable_value(&mut dataset.cur_data, opt, opt.name());
                }
            });
        });

        match dataset.cur_data {
            PossibleDatasets::MNIST => update_mnist(dataset, &mut dataset_params.mnist, ui)
        }

    });
}

fn update_mnist(dataset: &mut DatasetState, params: &mut MnistParams, ui: &mut egui::Ui) {
    ui.group(|ui| {
        ui.vertical(|ui| {
            ui.label("batch size");
            ui.add(egui::DragValue::new(&mut params.batch_size));
            
        });
    });
}

fn get_transform() {

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
    pub cur_data: PossibleDatasets,
    pub mnist: Option<Mnist>
}

impl Default for DatasetState {
    fn default() -> Self {
        DatasetState { cur_data: PossibleDatasets::MNIST, mnist: None }
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