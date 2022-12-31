use crate::Config;
use crossbeam::channel::{Sender, Receiver};
use std::thread::{JoinHandle, spawn};

mod baseline;
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

pub trait Train {
    fn build(&self) -> TrainProgress;
}