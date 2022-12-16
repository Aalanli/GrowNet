#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]


pub mod ui;
pub mod visualizations;
pub mod model_configs;
pub mod data_configs;

use bevy_egui::egui;
use anyhow::{Result, Context};
use serde::{Serialize, de::DeserializeOwned};

use model_lib::Config;

pub trait UI: Config {
    fn ui(&mut self, ui: &mut egui::Ui);
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

