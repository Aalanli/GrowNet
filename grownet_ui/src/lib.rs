#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

pub mod data_configs;
pub mod model_configs;
pub mod ui;
pub mod visualizations;

use anyhow::{Context, Result};
use bevy_egui::egui;
use grownet_macros::{Config, UI};
use model_lib::Configure;
use num::Num;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// TODO: Get rid of Config, it simply isn't necessary
pub trait UI: Configure {
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

impl UI for std::path::PathBuf {
    fn ui(&mut self, ui: &mut egui::Ui) {
        let mut str = self.to_str().unwrap().to_string();
        ui.text_edit_singleline(&mut str);
        *self = str.into();
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct DragConfig<T> {
    pub val: T,
    pub speed: f64,
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
        path: String,
    }

    let t = Test::default();
    let _c = t.config();
    let _f = <Test as UI>::ui;
}
