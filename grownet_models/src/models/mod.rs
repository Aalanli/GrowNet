use std::collections::VecDeque;
use std::ops::DerefMut;

use anyhow::{Error, Result};
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread::{spawn, JoinHandle};

use crate::{Config, Configure};
pub mod baseline;
mod m1;
mod m2;

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
    PLOT(String, f32, f32), // key, x, y
    CHECKPOINT(f32, std::path::PathBuf),
    FAILED(String),
    KILLED,
}

/// The handle to the process running the training, interact with that process
/// through this struct by sending commands and receiving logs
pub struct TrainProcess {
    pub send: Sender<TrainSend>,
    pub recv: Receiver<TrainRecv>,
    handle: Option<JoinHandle<()>>,
}

impl TrainProcess {
    pub fn is_running(&self) -> bool {
        self.handle.is_some() && self.handle.as_ref().unwrap().is_finished()
    }

    /// blocks until process is killed
    pub fn kill_blocking(&mut self) {
        self.send
            .send(TrainSend::KILL)
            .expect("unable to send kill msg");
        let handle = std::mem::replace(&mut self.handle, None).unwrap();
        handle.join().expect("trouble joining thread");
    }
}
