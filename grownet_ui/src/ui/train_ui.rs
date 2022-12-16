use crossbeam::channel::{Sender, Receiver};

pub enum Log {
    PLOT(String, f32, f32),
}

pub enum Commands {
    STOP
}

pub trait Trainer {
    fn build(&mut self) -> (Sender<Commands>, Receiver<Log>);

}

pub struct TrainBaseline {
    
}