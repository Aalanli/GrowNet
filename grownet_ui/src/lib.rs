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

macro_rules! derive_numeric_ui {
    ($a:ty) => {
        impl UI for $a {
            fn ui(&mut self, ui: &mut egui::Ui) {
                ui.add(egui::DragValue::new(self).speed(0.1));
            }
        }
    };
}

derive_numeric_ui!(i32);
derive_numeric_ui!(i64);
derive_numeric_ui!(f32);
derive_numeric_ui!(f64);
derive_numeric_ui!(u32);
derive_numeric_ui!(u64);
derive_numeric_ui!(isize);
derive_numeric_ui!(usize);

impl UI for bool {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(self, "");
    }
}

impl UI for String {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.text_edit_singleline(self);
    }
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
        batch_size: u32,
        eps: f32,
        path: String
    }

    let t = Test::default();
    let _c = t.config();
    let _f = <Test as UI>::ui;
}