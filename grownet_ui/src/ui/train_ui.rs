use std::collections::HashMap;

use anyhow::{Result, Error};
use bevy::prelude::*;
use bevy_egui::egui;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use model_lib::models::{Train, TrainProgress};
use super::super::model_configs::baseline;
use crate::{Config, UI};

pub trait TrainConfig: Train + Config + UI {}
impl<T: Train + Config + UI> TrainConfig for T {}



#[derive(Serialize, Deserialize, Default)]
struct TrainingUI {
    baseline: ModelEnviron<baseline::BaselineParams>,
    model: Models
}

#[derive(Default, Serialize, Deserialize)]
pub struct TrainLogs {
    models: HashMap<String, ModelLogInfo>
}

impl TrainingUI {
    fn ui(&mut self, ui: &mut egui::Ui) {

    }
}

#[derive(Serialize, Deserialize)]
pub enum Models {
    BASELINE
}

impl Default for Models {
    fn default() -> Self {
        Models::BASELINE
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct ModelEnviron<Config> {
    model_name: String,
    config: Config,
    runs: Vec<Config>,
    version_num: u32,
    #[serde(skip)]
    cur_run: Option<TrainProgress>,
    run_info: Option<RunInfo>,
    run_finished: bool,
}

impl<C: TrainConfig + Default> ModelEnviron<C> {
    fn new(name: String) -> Self {
        Self { model_name: name, config: C::default(), runs: Vec::new(), version_num: 0, cur_run: None, run_info: None, run_finished: false }
    }

    fn end_training(&mut self) -> Result<RunInfo> {
        todo!()
    }

    fn try_finish(&self) -> Option<RunInfo> {
        todo!()
    }

    fn ui(&mut self, ui: &mut egui::Ui, logs: &mut TrainLogs) {

    }

}


#[derive(Serialize, Deserialize)]
pub struct ModelLogInfo {
    name: String,
    runs: Vec<RunInfo>,   
}

#[derive(Serialize, Deserialize)]
pub struct RunInfo {
    pub version: u32,
    pub name: String,
    pub dataset: String,
    pub plots: HashMap<String, Vec<(f32, f32)>>,
    pub config: HashMap<String, String>
}


impl TrainLogs {
    fn new_model(&mut self, name: String) -> Result<()> {
        if self.models.contains_key(&name) {
            return Err(Error::msg(format!("already contains model {name}")));
        }
        self.models.insert(name.clone(), ModelLogInfo { name: name, runs: Vec::new() });
        Ok(())
    }

    fn insert_run(&mut self, model: String, run: RunInfo) -> Result<()> {
        if let Some(x) = self.models.get_mut(&model) {
            x.runs.push(run);
        } else {
            return Err(Error::msg(format!("Does not contain model {}", model)));
        }
        Ok(())
    }
}

#[test]
fn serialize_test() {
    #[derive(Default, Serialize, Deserialize)]
    struct TestParams {
        a: usize,
        b: String,
        c: f32
    }

    let a = TestParams::default();
    let json = ron::to_string(&a).unwrap();
    println!("{}", json);
    let out: HashMap<String, String> = ron::from_str(&json).expect("unable to convert to hashmap");
    println!("{:?}", out);
}