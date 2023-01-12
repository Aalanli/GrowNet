use crate::Config;
use crossbeam::channel::{Sender, Receiver};
use std::thread::{JoinHandle, spawn};
use anyhow::{Result, Error};

pub mod baseline;
mod m1;
mod m2;

pub enum TrainCommand {
    STOP,
    OTHER(usize)
}

pub enum Log {
    PLOT(String, f32, f32), // key, x, y
}


pub struct TrainProgress {
    pub send: Sender<TrainCommand>,
    pub recv: Receiver<Log>,
    pub handle: JoinHandle<()>
}

impl TrainProgress {
    fn kill(&mut self) -> Result<()> {
        self.send.send(TrainCommand::STOP)?;
        Ok(())
    }
}

pub trait Train {
    fn build(&self) -> TrainProgress;
}