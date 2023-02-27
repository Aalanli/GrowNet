use std::collections::VecDeque;
use std::ops::DerefMut;
use std::path::PathBuf;

use anyhow::{Error, Result};
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread::{spawn, JoinHandle};

use crate::{Config, Configure};
pub mod baseline;
pub mod baselinev3;
// pub mod baselinev2;
mod m1;
mod m2;

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct RunStats {
    pub step_time: Option<f32>,
}

#[derive(Clone)]
pub struct PlotPoint {
    pub title: &'static str,
    pub x_title: &'static str,
    pub y_title: &'static str,
    pub x: f64,
    pub y: f64
}

pub enum TrainSend {
    KILL,
    OTHER(usize),
}

/// The reason there is a TrainRecv and a Log, with the two being nearly identical
/// is that the TrainRecv is the direct output of the training process, which does not
/// have information, as it does not concern itself, with various details such as model version
/// etc. As its only responsibility is to train the model and output raw data. This raw data
/// is TrainRecv, and is processed by TrainEnviron, which does concern itself with details such as
/// model version number, etc, and converts/integrates this information to Log.
#[derive(Clone)]
pub enum TrainRecv {
    PLOT(PlotPoint), // key, x, y
    FAILED(String),
    STATS(RunStats),
    // CHECKPOINT(f32, std::path::PathBuf),
}

/// The handle to the process running the training, interact with that process
/// through this struct by sending commands and receiving logs
pub struct TrainProcess {
    send: Sender<TrainSend>,
    recv: Receiver<TrainRecv>,
    handle: Option<JoinHandle<()>>,
}

impl TrainProcess {
    pub fn is_running(&self) -> bool {
        self.handle.is_some() && !self.handle.as_ref().unwrap().is_finished()
    }

    pub fn send_command(&mut self, command: TrainSend) {
        self.send.send(command).expect("unable to send train command");
    }

    pub fn try_recv(&mut self) -> Vec<TrainRecv> {
        self.recv.try_iter().collect()
    }

    pub fn try_kill(&mut self) {
        if self.is_running() {
            self.send
                .send(TrainSend::KILL)
                .expect("unable to send kill msg");
        }
    }

    /// blocks until process is killed
    pub fn kill_blocking(&mut self) -> Result<()> {
        self.try_kill();
        let handle = std::mem::replace(&mut self.handle, None).unwrap();
        handle.join().map_err(|x| Error::msg(format!("thread error {:?}", x.downcast_ref::<&str>())))
    }
}


#[derive(Serialize, Deserialize)]
pub struct CachedInfo {
    checkpoints: VecDeque<Config>,
    keep: usize
}


pub struct CheckpointManager {
    pub folder: PathBuf,
    pub max_checkpoints: usize,
    nver: usize
}

impl CheckpointManager {
    pub fn new(folder: PathBuf, max_checkpoints: usize) -> Self {
        if !folder.exists() {
            std::fs::create_dir(&folder).expect(&format!("failed to create folder {}", folder.display()));
        }
        Self { folder, max_checkpoints, nver: 0 }
    }

    pub fn new_path(&mut self, step: usize) -> PathBuf {
        self.nver += 1;
        self.folder.join(format!("ckpt-{:0>10}-{}", self.nver, step)).with_extension("ckpt")
    }

    /// returns a sorted list of checkpoints paths, full paths, youngest checkpoints first
    pub fn checkpoints(&self) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.folder).expect(&format!("failed to read checkpoint dir {}", self.folder.display()))
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap().path())
            .filter(|x| x.is_file() && x.extension().unwrap() == "ckpt")
            .collect();
        files.sort();
        files
    }

    pub fn remove_old_checkpoints(&mut self) {
        let mut checkpoints = self.checkpoints();
        checkpoints.reverse();
        while self.max_checkpoints < checkpoints.len() {
            std::fs::remove_file(checkpoints.pop().unwrap()).expect("unable to remove old checkpoints");
        } 
    }
}

// #[derive(Serialize, Deserialize, Default)]
// pub struct CheckpointManagerClient {
//     checkpoints: HashMap<String, Checkpoints>
// }

// impl CheckpointManagerClient {
//     fn get(&self, run_info: &RunInfo, i: usize) -> Option<std::path::PathBuf> {
//         let name = run_info.run_name();
//         self.checkpoints.get(&name).and_then(|x| { 
//             x.checkpoints.get(i)
//                 .and_then(|x| Some(x.1.clone())) 
//         })
//     }

//     fn get_latest(&self, run_info: &RunInfo) -> Option<std::path::PathBuf> {
//         let name = run_info.run_name();
//         self.checkpoints.get(&name).and_then(|x| {
//             if x.checkpoints.len() > 1 {
//                 x.checkpoints.get(x.checkpoints.len() - 1 )
//                     .and_then(|x| Some(x.1.clone())) 
//             } else { None }
//         })
//     }

//     fn add(&mut self, run_info: &RunInfo, checkpoint: (usize, std::path::PathBuf)) {
//         todo!()
//     }
// }


