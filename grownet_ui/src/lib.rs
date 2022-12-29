#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]


pub mod ui;
pub mod visualizations;
pub mod model_configs;
pub mod data_configs;

use bevy_egui::egui;
use anyhow::{Result, Context};
use serde::{Serialize, de::DeserializeOwned, Deserialize};
use num::Num;
use model_lib::Config;
use grownet_macros::{Config, UI};

pub trait UI: Config {
    fn ui(&mut self, ui: &mut egui::Ui);
}


#[derive(Default, Serialize, Deserialize)]
pub struct DragConfig<T> {
    pub val: T,
    pub speed: f64
}

impl<T: Sync + Send + Serialize + DeserializeOwned + egui::emath::Numeric> UI for DragConfig<T> {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::DragValue::new(&mut self.val).speed(self.speed));
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct TextBox {
    pub val: String,
    pub msg: String,
}

impl UI for TextBox {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::TextEdit::singleline(&mut self.val).hint_text(&self.msg));
    }
}

#[test]
fn test_derive_ui_macro() {
    #[derive(Default, Config, UI)]
    struct Test {
        batch_size: DragConfig<u32>,
        eps: DragConfig<f32>,
        path: TextBox
    }

    let t = Test::default();
    let _c = t.config();
    let _f = <Test as UI>::ui;
}

pub struct ConfigWrapper<C, D> {
    pub config: C,
    pub state: Option<D>
}

impl<C, D> ConfigWrapper<C, D> {
    pub fn new(config: C) -> Self {
        Self { config, state: None }
    }

    pub fn drop_state(&mut self) {
        self.state = None;
    }
}

impl<C: Config, D: Send + Sync> Config for ConfigWrapper<C, D> {
    fn config(&self) -> String {
        self.config.config()
    }
    fn load_config(&mut self, config: &str) -> Result<()> {
        self.config.load_config(config)
    }
}

impl<C: UI, D: Send + Sync> UI for ConfigWrapper<C, D> {
    fn ui(&mut self, ui: &mut egui::Ui) {
        self.config.ui(ui);
    }
}

