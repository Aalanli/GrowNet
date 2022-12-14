mod baseline;
mod m1;
mod m2;

pub enum TrainCommand {
    STOP,
    OTHER(usize)
}

/// Assume that the model and dataset is not Send or Sync, so send
/// parameters instead, which are Send and Sync
/// 
/// A training run needs:
/// 1. Model configuration parameters
/// 2. Dataset configuration parameters
/// 3. Optimizer parameters
/// 4. Logger
/// 5. other processing which affects training dynamics (ex. lr-scheduler)
pub struct ModelConfig<M, D, Opt, Log, Misc> {
    model: M,
    data: D,
    opt: Opt,
    log: Log,
    misc: Misc
}



pub trait Logger {

}

