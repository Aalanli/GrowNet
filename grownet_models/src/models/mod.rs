use crate::Config;
use crossbeam::channel::{Sender, Receiver};
use std::thread::{JoinHandle, spawn};
use anyhow::{Result, Error};

pub mod baseline;
mod m1;
mod m2;

pub enum TrainCommand {
    KILL,
    OTHER(usize)
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
    pub handle: JoinHandle<()>
}

impl TrainProcess {
    pub fn kill(&mut self) -> Result<()> {
        self.send.send(TrainCommand::KILL)?;
        Ok(())
    }
}

pub trait Train {
    fn build(&self) -> TrainProcess;
}