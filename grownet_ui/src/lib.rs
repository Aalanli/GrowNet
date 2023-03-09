#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use std::path::PathBuf;
use std::collections::HashSet;
use bevy::prelude::Resource;

pub mod ops;
pub mod run_systems;
pub mod ui;

use anyhow::{Context, Result};
use bevy_egui::egui;
use num::Num;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const ROOT_CONFIG_PATH: &'static str = "assets/config";

pub use run_systems::{config_ui_adjust, config_ui_show};


#[derive(Resource)]
pub struct Serializer {
    root_path: PathBuf, // the source folder to save all app state
    saved_paths: HashSet<PathBuf>, // path already saved
}

impl Default for Serializer {
    fn default() -> Self {
        let root_path = PathBuf::from(ROOT_CONFIG_PATH);
        if !root_path.exists() {
            std::fs::create_dir_all(&root_path).expect("unable to setup path manager");
        }
        Self { root_path, saved_paths: HashSet::new() }
    }
}

impl Serializer {
    pub fn serialize<T: Serialize>(&mut self, path: &str, x: &T) {
        let qualifed_path = self.root_path.join(path);
        if self.saved_paths.contains(&qualifed_path) {
            panic!("path {} already exists", qualifed_path.display());
        } else {
            if !qualifed_path.parent().expect(&format!("path {} does not have a parent", qualifed_path.display())).exists() {
                std::fs::create_dir_all(qualifed_path.parent().unwrap()).expect("failed to create directory for serialize");
            }
            let train_data_writer = std::fs::File::create(&qualifed_path).unwrap();
            println!("serializing to {}", qualifed_path.display());
            bincode::serialize_into(train_data_writer, x).expect("unable to serialize");
            self.saved_paths.insert(qualifed_path);
        }
    }

    pub fn deserialize<T: DeserializeOwned>(&self, path: &str, x: &mut T) {
        let qualifed_path = self.root_path.join(path);
        if qualifed_path.exists() {
            println!("deserializing from {}", qualifed_path.display());
            let reader = std::fs::File::open(&qualifed_path).expect("unable to open file");
            match bincode::deserialize_from::<std::fs::File, T>(reader) {
                Ok(de) => { *x = de; },
                Err(e) => {
                    eprintln!("failed to deserializing from {} due to {e}", qualifed_path.display());
                }
            }
        }
    }
}