use std::collections::VecDeque;

use anyhow::{Error, Result};
use crossbeam::channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread::{spawn, JoinHandle};

use crate::Config;
pub mod baseline;
mod m1;
mod m2;

#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy)]
pub enum Models {
    BASELINE,
}

impl std::fmt::Display for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Models::BASELINE => write!(f, "baseline"),
        }
    }
}

impl Default for Models {
    fn default() -> Self {
        Models::BASELINE
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct TrainData {
    pub plots: ModelPlots,
    pub console: Console,
    pub past_runs: ModelRuns,
}

impl TrainData {
    pub fn handle_baseline_logs(&mut self, logs: &[Log]) {
        for log in logs {
            match log.clone() {
                Log::PLOT(name, run_name, x, y) => {
                    self.plots.add_plot(&name, &run_name, x, y);
                }
                Log::CONSOLE(msg) => {
                    self.console.log(msg);
                }
                Log::RUNINFO(info) => {
                    self.past_runs.insert(info);
                }
            }
        }
    }    
}

/// TrainEnviron is responsible for training the various models given a config,
/// it knows all the low level implementation details of the config, and can access the config fields.
/// It converts these low level detains and outputs a (unified?) Log, which the various UI elements
/// consume. The UI is not responsible for knowing configs, only adjusting the parameters and giving it
/// to TrainEnviron to train, receiving Logs and updating visualizations.
/// This is made this way for future work in distributed computing, where TrainEnviron may run on a 
/// headless server, without the bevy stuff.
#[derive(Default, Serialize, Deserialize)]
pub struct TrainEnviron {
    pub baseline: SimpleEnviron<baseline::BaselineParams>,
    selected: Models
}

impl TrainEnviron {
    pub fn selected(&self) -> Models { return self.selected; }

    pub fn is_running(&self) -> bool {
        match self.selected {
            Models::BASELINE => { self.baseline.is_running() }
        }
    }

    pub fn run(&mut self, model: Models) -> Result<()> {
        self.selected = model.clone();
        match model {
            Models::BASELINE => {
                if self.baseline.is_running() {
                    return Err(Error::msg("baseline is already running"))
                }
                self.baseline.run = Some(self.baseline.config.build());
                self.baseline.run_info = Some(RunInfo { 
                    model_class: "baseline".into(), 
                    version: self.baseline.runs, 
                    dataset: "cifar10".into(),
                    err_status: true,
                    ..Default::default() 
                });
                self.baseline.runs += 1;
                Ok(())
            }
        }
    }

}

#[derive(Default, Serialize, Deserialize)]
pub struct SimpleEnviron<T> {
    config: T,
    runs: usize,
    #[serde(skip)]
    run: Option<TrainProcess>,
    run_info: Option<RunInfo>,
}

impl<T> SimpleEnviron<T> {
    pub fn is_running(&self) -> bool { self.run.is_some() }

    pub fn run_name(&self) -> Option<String> {
        self.run_info.as_ref().and_then(|x| Some(x.run_name()) )
    }

    pub fn set_config(&mut self, config: T) {
        self.config = config;
    } 

    pub fn kill_blocking(&mut self) -> Result<Vec<Log>> {
        if let Some(t) = &mut self.run {
            t.kill().and_then(|x| Ok(self.convert_recv(&x)) )
        } else {
            Err(Error::msg("Nothing to kill"))
        }
    }

    pub fn run_data(&mut self) -> Vec<Log> {
        let mut logs = Vec::new();
        if self.is_running() { 
            while let Ok(log) = self.run.as_mut().unwrap().recv.try_recv() {
                logs.extend(self.convert_recv(&[log]));
            }
        }
        logs
    }

    pub fn convert_recv(&mut self, recv: &[TrainRecv]) -> Vec<Log> {
        let mut logs = Vec::new();
        for log in recv {
            match log.clone() {
                TrainRecv::KILLED => {
                    logs.push(Log::CONSOLE(format!("{} finished training", &self.run_info.as_ref().unwrap().run_name())));
                    let mut run_info = std::mem::replace(&mut self.run_info, None).unwrap();
                    run_info.err_status = true;
                    logs.push(Log::RUNINFO(run_info));
                    self.run = None;
                }
                TrainRecv::PLOT(name, x, y) => {
                    logs.push(Log::CONSOLE(format!("Logged plot name: {}, x: {}, y: {}", &name, x, y)));
                    logs.push(Log::PLOT(name, self.run_name().unwrap(), x, y));
                }
            }
        }
        logs
    }
}


/// ModelPlots contains the various plots of the model
#[derive(Default, Serialize, Deserialize)]
pub struct ModelPlots {
    graphs: HashMap<String, PlotGraph> // each key represents the title of the plot
}

/// Each PlotGraph signifies a graph which contains multiple lines, from multiple runs
#[derive(Deserialize, Serialize, Default)]
pub struct PlotGraph {
    title: String,
    x_title: String,
    y_title: String,
    plots: HashMap<String, Vec<(f32, f32)>>,
}

impl ModelPlots {
    pub fn add_plot(&mut self, title: &str, run_name: &str, x: f32, y: f32) {
        if let Some(graph) = self.graphs.get_mut(title) {
            if let Some(run) = graph.plots.get_mut(run_name) {
                run.push((x, y));
            } else {
                graph.plots.insert(run_name.into(), vec![(x, y)]);
            }
        } else {
            // TODO: kind of ad-hoc
            let (x_title, y_title) = if title.contains("accuracy") {
                ("steps", "accuracy")
            }
            else if title.contains("loss") {
                ("steps", "loss")
            } else {
                ("", "")
            };

            self.graphs.insert(title.into(), PlotGraph { 
                title: title.into(), 
                x_title: x_title.into(), 
                y_title: y_title.into(), 
                plots: HashMap::from([(run_name.to_string(), vec![(x, y)])]) 
            });
        }
    }
}

#[derive(Deserialize, Serialize, Default)]
pub struct ModelRuns {
    runs: HashMap<String, Vec<RunInfo>>
}

impl ModelRuns {
    pub fn insert(&mut self, run_info: RunInfo) {
        if let Some(runs) = self.runs.get_mut(&run_info.model_class) {
            runs.push(run_info);
        } else {
            self.runs.insert(run_info.model_class.clone(), vec![run_info]);
        }
    }
}

/// This struct represents an individual training run
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct RunInfo {
    pub model_class: String, // name for the class of models this falls under
    pub version: usize,        // id for this run
    pub comments: String,
    pub dataset: String,
    pub config: HashMap<String, String>, // TODO: convert types to this more easily
    pub err_status: bool, // True is returned successfully, false if Killed mid-run
}

impl RunInfo {
    pub fn run_name(&self) -> String {
        format!("{}-v{}", self.model_class, self.version)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Console {
    pub console_msgs: VecDeque<String>,
    pub max_console_msgs: usize,
}

impl Console {
    fn new(n_logs: usize) -> Self {
        Console {
            console_msgs: VecDeque::new(),
            max_console_msgs: n_logs,
        }
    }

    fn log(&mut self, msg: String) {
        self.console_msgs.push_back(msg);
        if self.console_msgs.len() > self.max_console_msgs {
            self.console_msgs.pop_front();
        }
    }
}

impl Default for Console {
    fn default() -> Self {
        Self {
            console_msgs: VecDeque::new(),
            max_console_msgs: 50,
        }
    }
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
    PLOT(String, f32, f32), // key, x, y
    KILLED,
}

/// The (unified?) Log which visualization utilities consume
#[derive(Clone)]
pub enum Log {
    PLOT(String, String, f32, f32),
    RUNINFO(RunInfo),
    CONSOLE(String)
}


/// The handle to the process running the training, interact with that process
/// through this struct by sending commands and receiving logs
pub struct TrainProcess {
    pub send: Sender<TrainSend>,
    pub recv: Receiver<TrainRecv>,
    pub handle: JoinHandle<()>,
}

impl TrainProcess {
    /// blocks until process is killed
    pub fn kill(&mut self) -> Result<Vec<TrainRecv>> {
        self.send.send(TrainSend::KILL)?;
        let mut log_msgs = Vec::new();
        loop {
            let msg = self.recv.try_recv()?;
            match msg {
                TrainRecv::KILLED => { return Ok(log_msgs); }
                x => {
                    log_msgs.push(x);
                }
            }
        }
    }
}
