use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
enum State {
    Menu,
    DataExplorer,
    Trainer
}

pub struct UI;
impl Plugin for UI {
    fn build(&self, app: &mut App) {
        app.add_system(data_explorer);
    }
}

fn data_explorer(
    mut egui_context: ResMut<EguiContext>
) {
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Label::new("Data Explorer"));

    });
}