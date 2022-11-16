use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use std::{borrow::Cow};

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
            .add_system(menu_ui);
    }
}

struct UIParams {
    font_delta: f32,
    open_panel: OpenPanel,
    training: bool
}


impl Default for UIParams {
    fn default() -> Self {
        UIParams { font_delta: 4.0, open_panel: OpenPanel::Models, training: false }
    }
}


#[derive(PartialEq, Eq)]
enum OpenPanel {
    Data,
    Models
}


fn menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut params: ResMut<UIParams>,
) {
    let mut local_font_delta = params.font_delta;
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Label::new("Data Explorer"));
        
        // stylistic changes
        ui.collapsing("styling", |ui| {
            ui.label("font size delta");
            ui.add(egui::Slider::new(&mut local_font_delta, -9.0..=12.0));
        });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut params.open_panel, OpenPanel::Models, "Models");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Data, "Datasets");
        });
        ui.separator();

        match params.open_panel {
            OpenPanel::Models => {},
            OpenPanel::Data => {},
        }

        ui.separator();
        let train = ui.button("training").clicked();
    });

    if local_font_delta != params.font_delta {
        change_font_size(local_font_delta, egui_context.ctx_mut());
        params.font_delta = local_font_delta;
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