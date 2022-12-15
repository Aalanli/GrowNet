#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

pub mod models_configs;
pub mod data_configs;
pub mod datasets;
pub mod ui;
pub mod visualizations;

use bevy_egui::egui;
use anyhow::{Result, Context};
use serde::{Serialize, de::DeserializeOwned};

/// Param trait captures the various parameters settings that needs to be saved
/// to disk and modified through the ui
pub trait Param: Config + UI {}

pub trait Config: Send + Sync {
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str) -> Result<()>;
}

pub trait UI {
    fn ui(&mut self, ui: &mut egui::Ui);
}

impl<T: Serialize + DeserializeOwned + Send + Sync> Config for T {
    fn config(&self) -> String {
        ron::to_string(self).unwrap()
    }
    fn load_config(&mut self, config: &str) -> Result<()> {
        *self = ron::from_str(config).context(format!("Failed to load context {}", config))?;
        Ok(())
    }
}

impl<T: Config + UI> Param for T {}

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

