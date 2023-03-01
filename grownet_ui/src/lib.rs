#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

pub mod ops;
pub mod run_systems;
pub mod ui;

use anyhow::{Context, Result};
use bevy_egui::egui;
use num::Num;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const CONFIG_PATH: &'static str = "assets/config";
const RUN_DATA_PATH: &'static str = "assets/model_runs";

pub use run_systems::{config_ui_adjust, config_ui_show};