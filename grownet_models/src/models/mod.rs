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

#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone)]
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


#[derive(Default, Serialize, Deserialize)]
pub struct TrainEnviron {
    pub baseline: SimpleEnviron<baseline::BaselineParams>,
    pub selected: Models,
    pub plots: ModelPlots,
    pub console: Console,
    pub past_runs: ModelRuns,
}

impl TrainEnviron {
    pub fn is_running(&self) -> bool {
        match self.selected {
            Models::BASELINE => { self.baseline.is_running() }
        }
    }

    pub fn kill(&mut self) -> Result<()> {
        match self.selected {
            Models::BASELINE => { 
                if let Err(e) = self.baseline.kill() {
                    self.console.log(format!("unable to kill baseline due to {}", e.to_string()));
                    Err(e)
                } else {
                    self.console.log(format!("successfully killed baseline"));
                    let mut run = std::mem::replace(&mut self.baseline.run_info, None).unwrap();
                    run.err_status = false;
                    self.past_runs.insert(run);
                    Ok(()) // TODO: can be better, since we could be throwing away some logs
                }
            }
        }
    }

    pub fn run(&mut self) -> Result<()> {
        match self.selected {
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

    pub fn update_data(&mut self) {
        if !self.is_running() { return; }
        match self.selected {
            Models::BASELINE => {
                let recv = self.baseline.run.as_mut().unwrap();
                while let Ok(log) = recv.recv.try_recv() {
                    match log {
                        Log::PLOT(name, x, y) => {
                            self.console.log(format!("Logged plot name: {}, x: {}, y: {}", &name, x, y));
                            self.plots.add_plot(&name, &self.baseline.run_info.as_ref().unwrap().run_name(), x, y);
                        }
                        Log::KILLED => {
                            self.baseline.run_info = None;
                            self.console.log(format!("{} finished training", &self.baseline.run_info.as_ref().unwrap().run_name()));
                            let mut run = std::mem::replace(&mut self.baseline.run_info, None).unwrap();
                            run.err_status = true;
                            self.past_runs.insert(run);
                        }
                    }
                }
            }
        }
    }    
}

#[derive(Default, Serialize, Deserialize)]
pub struct SimpleEnviron<T> {
    pub config: T,
    pub past_configs: Vec<T>,
    pub past_configs_cache: Vec<T>,
    pub runs: usize,
    #[serde(skip)]
    pub run: Option<TrainProcess>,
    pub run_info: Option<RunInfo>
}

impl<T> SimpleEnviron<T> {
    pub fn is_running(&self) -> bool { self.run.is_some() }

    pub fn kill(&mut self) -> Result<Vec<Log>> {
        if let Some(t) = &mut self.run {
            t.kill()
        } else {
            Ok(Vec::new())
        }
    }

    pub fn run_name(&self) -> Option<String> {
        self.run_info.as_ref().and_then(|x| Some(x.run_name()) )
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
#[derive(Serialize, Deserialize, Default)]
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
    /// blocks until process is killed
    pub fn kill(&mut self) -> Result<Vec<Log>> {
        self.send.send(TrainCommand::KILL)?;
        let mut log_msgs = Vec::new();
        loop {
            let msg = self.recv.try_recv()?;
            match msg {
                Log::KILLED => { return Ok(log_msgs); }
                x => {
                    log_msgs.push(x);
                }
            }
        }
    }
}
