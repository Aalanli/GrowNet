use crate::Config;
use anyhow::{Error, Result};
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread::{spawn, JoinHandle};

pub mod baseline;
mod m1;
mod m2;

pub enum TrainCommand {
    KILL,
    OTHER(usize),
}

#[derive(Clone)]
pub enum Log {
    PLOT(String, f32, f32), // key, x, y
    KILLED,
}

/// The handle to the process running the training, interact with that process
/// through this struct by sending commands and receiving logs
pub struct TrainProcess {
    pub send: Sender<TrainCommand>,
    pub recv: Receiver<Log>,
    pub handle: JoinHandle<()>,
}

impl TrainProcess {
    pub fn kill(&mut self) -> Result<()> {
        self.send.send(TrainCommand::KILL)?;
        Ok(())
    }
}

/// Struct containing all the logs associated with every run
/// configs gets 'collapsed' into a common dictionary representation for displaying purposes
#[derive(Default, Serialize, Deserialize)]
pub struct TrainLogs {
    pub models: Vec<RunInfo>, // flattened representation, small enough number of runs to justify not using hashmap
}

/// This struct represents an individual training run
#[derive(Serialize, Deserialize, Default)]
pub struct RunInfo {
    pub model_class: String, // name for the class of models this falls under
    pub version: u32,        // id for this run
    pub comments: String,
    pub dataset: String,
    pub plots: HashMap<String, Vec<(f32, f32)>>,
    pub config: Option<HashMap<String, String>>, // TODO: convert types to this more easily
}

impl RunInfo {
    pub fn run_name(&self) -> String {
        format!("{}-v{}", self.model_class, self.version)
    }

    pub fn log(&mut self, log: Log) {}

    pub fn reset(&mut self) {}
}

pub trait Train {
    fn build(&self) -> TrainProcess;
}
