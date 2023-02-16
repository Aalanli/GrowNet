use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy::window::WindowCloseRequested;
use bevy_egui::{egui, EguiContext};
use bevy::ecs::schedule::ShouldRun;
use crossbeam::channel::Receiver;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{AppState, OperatingState, OpenPanel, UIParams, handle_pane_options};
use crate::{Configure, UI, ops, CONFIG_PATH};
use model_lib::models::{self, TrainRecv};
use model_lib::Config;
use crate::model_configs::{self as M, run_data as run, immutable_show};

mod train_systems;

pub struct TrainUIPlugin;
impl Plugin for TrainUIPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugin(run::RunDataPlugin)
            .add_event::<Despawn>()
            .add_event::<Kill>()
            .insert_resource(RunQueue::default())
            .insert_resource(TrainingUI::default())
            .add_startup_system(setup_train_ui)
            .add_system_set(
                SystemSet::on_update(AppState::Models)
                    .label("train_menu")
                    .with_system(training_menu)
            )
            .add_system_set(SystemSet::on_update(AppState::Trainer)
                .with_system(training_system)
                .with_system(queue_ui))
            .add_system_set(
                SystemSet::on_update(OperatingState::Active).with_system(run_queue))
            .add_system_set(
                SystemSet::on_update(OperatingState::Cleanup).with_system(cleanup_queue))
            .add_system_set(
                SystemSet::on_update(OperatingState::Close).with_system(save_train_ui))
            .add_plugin(M::baseline::BaselinePlugin);
    }
}

/// possibly load any training state from disk
fn setup_train_ui(
    mut train_ui: ResMut<TrainingUI>,
) {
    let root_path: std::path::PathBuf = CONFIG_PATH.into();
    ops::try_deserialize(&mut *train_ui, &root_path.join("train_ui").with_extension("config"));
}

/// write train state to disk
fn save_train_ui(
    train_ui: Res<TrainingUI>,
) {
    // load configurations from disk
    let root_path: std::path::PathBuf = CONFIG_PATH.into();
    eprintln!("serializing train_ui");
    // save config files to disk
    ops::serialize(&*train_ui, &root_path.join("train_ui").with_extension("config"));
}

/// TrainingUI is the menu in which one adjusts configurations and launches training processes
/// It contains a list of past configurations, and options to kill tasks and restart tasks
#[derive(Serialize, Deserialize, Resource)]
pub struct TrainingUI {
    baseline: ConfigEnviron,
    model: run::Models,
}

impl Default for TrainingUI {
    fn default() -> Self {
        Self { 
            baseline: ConfigEnviron::new("baseline",models::baseline::baseline_config()), 
            model: run::Models::BASELINE 
        }
    }
}

fn training_menu(
    mut egui_context: ResMut<EguiContext>,
    mut app_state: ResMut<State<AppState>>,
    op_state: Res<State<OperatingState>>,
    mut params: ResMut<UIParams>,
    mut train_ui: ResMut<TrainingUI>,
    mut run_queue: ResMut<RunQueue>,
    killer: EventWriter<Kill>,
) {
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        let prev_panel = params.open_panel;
        handle_pane_options(ui, &mut params.open_panel);


        if std::mem::discriminant(&params.open_panel) == std::mem::discriminant(&OpenPanel::Trainer) {
            // stupid hack, to prevent infinite cycling between AppState::Trainer and AppState::Menu
            params.open_panel = prev_panel;
            app_state.set(AppState::Trainer).unwrap();
        } else if std::mem::discriminant(&params.open_panel) != std::mem::discriminant(&OpenPanel::Models) {
            app_state.set(AppState::Menu).unwrap(); // should be fine to not return here
        }

        let space = ui.available_size();
        ui.horizontal(|ui| {
            // to this to prevent that
            ui.allocate_ui(space, |ui| {
                // the left most panel showing a list of model options
                ui.vertical(|ui| {
                    ui.selectable_value(&mut train_ui.model, run::Models::BASELINE, "baseline");
                    
                });

                // update any configurations using the ui
                ui.vertical(|ui| match train_ui.model {
                    run::Models::BASELINE => {
                        train_ui.baseline.ui(ui);
                    }
                });

                // Launching training
                ui.vertical(|ui| {
                    // TODO: add some keybindings to certain buttons
                    // entry point for launching training
                    // only launch things if the operating state is active
                    if *op_state.current() == OperatingState::Active && ui.button("Launch Training").clicked() {
                        match train_ui.model {
                            run::Models::BASELINE => {
                                let (spawn_fn, runinfo) = 
                                    M::baseline::baseline_spawn_fn(train_ui.baseline.version_num as usize, train_ui.baseline.get_config());
                                app_state.set(AppState::Trainer).unwrap();
                                train_ui.baseline.version_num += 1;
                                run_queue.add_run(runinfo, spawn_fn);
                            }
                        }
                    }
                    if *op_state.current() == OperatingState::Cleanup {
                        ui.label("killing any active tasks");
                    }
                    run_queue.ui(ui, killer);
                });
            });
        });
    });
}



/// Since each run is identified with an Entity, sending a Kill event for a particular entity
/// should kill it. Listeners for each run type should listen for this event, and kill their
/// respective runs when this event is heard.
#[derive(Deref)]
pub struct Kill(pub Entity);

/// Once the listener kills the task, this Event is sent back to RunQueue to confirm that
/// it is alright to free its resources.
#[derive(Deref)]
pub struct Despawn(pub Entity);

pub type SpawnRun = Box<dyn FnOnce(&mut Commands) -> Result<Entity> + Send + Sync>;
/// A wrapper with all of the required information to spawn a new run
pub struct Spawn(run::RunInfo, SpawnRun);

/// RunQueue keeps track of runs waiting to be spawned, and current active runs
/// it has a system which takes care of spawning new tasks and killing tasks
#[derive(Resource, Default)]
pub struct RunQueue {
    queued_runs: VecDeque<Spawn>,
    active_runs: VecDeque<(run::RunInfo, Entity)>,
    spawn_errors: VecDeque<String>,
}

impl RunQueue {
    fn add_run(&mut self, info: run::RunInfo, run_fn: SpawnRun) {
        self.queued_runs.push_back(Spawn(info, run_fn));
    }

    fn ui(&mut self, ui: &mut egui::Ui, mut kill: EventWriter<Kill>) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            // show errors
            if self.spawn_errors.len() == 0 {
                ui.horizontal(|ui| {
                    ui.label("launch errors");
                    if ui.button("clear").clicked() {
                        self.spawn_errors.clear();
                    }
                });
                for msg in self.spawn_errors.iter() {
                    ui.label(egui::RichText::new(msg).color(egui::Color32::RED));
                }
                ui.separator();
            }
            // show a list of queued runs, with option to remove a run
            ui.label("queued runs");
            let mut i = 0;
            while i < self.queued_runs.len() {
                if ui.button("remove").clicked() {
                    self.queued_runs.remove(i);
                    continue;
                }
                ui.collapsing(self.queued_runs[i].0.run_name(), |ui| {
                    self.queued_runs[i].0.show_basic(ui);
                });
                i += 1;
            }
            // show a list of active runs, with option to kill a run, albeit indirectly
            ui.label("active runs");
            for i in 0..self.active_runs.len() {
                if ui.button(format!("kill {}", self.active_runs[i].0.run_name())).clicked() {
                    kill.send(Kill(self.active_runs[i].1));
                }
            }
        });
    }
}

/// the queue ui on the training side
fn queue_ui(
    mut egui_context: ResMut<EguiContext>,
    mut queue: ResMut<RunQueue>,
    killer: EventWriter<Kill>,
) {
    egui::Window::new("run queue").show(egui_context.ctx_mut(), |ui| {
        queue.ui(ui, killer);
    });
}

/// send kill signals for all active runs in the queue
/// after all active tasks are killed, 
fn cleanup_queue(
    mut queue: ResMut<RunQueue>,
    mut killer: EventWriter<Kill>,
    mut killed: EventReader<Despawn>,
    mut app_state: ResMut<State<OperatingState>>
) {
    queue.queued_runs.clear();
    for i in queue.active_runs.iter() {
        killer.send(Kill(i.1));
    }

    for i in killed.iter() {
        remove_once_if_any(&mut queue.active_runs, |x| { x.1 == i.0 });
    }

    // if there are no more active runs, signal to close app
    if queue.active_runs.len() == 0 {
        app_state.set(OperatingState::Close).expect("failed to set close state");
    }
}

/// run_queue takes care of spawning and despawning various training runs
/// and only runs when the OperatingState is active
fn run_queue(
    mut commands: Commands,
    mut queue: ResMut<RunQueue>,
    mut killed: EventReader<Despawn>,
    params: Res<UIParams>
) {
    // remove any entities already killed or despawned
    // TODO: could have various error handling policies here, but for the sake of simplicity, just ignore for now
    // which implies that training runs that are unkillable will just collect in the active_runs queue
    for k in killed.iter() {
        let id = k.0;
        if remove_once_if_any(&mut queue.active_runs, |x| {x.1 == id}) {
            eprintln!("removed id {:?}", id);
            commands.entity(id).despawn();
        }
    }
    // spawn new things
    for _ in 0..(params.run_queue_max_active - queue.active_runs.len()) {
        if let Some(x) = queue.queued_runs.pop_front() {
            let (info, spawn_fn) = (x.0, x.1);
            let id = spawn_fn(&mut commands);
            match id {
                Ok(id) => { queue.active_runs.push_back((info, id)); },
                Err(msg) => {
                    queue.spawn_errors.push_back(msg.to_string());
                    if queue.spawn_errors.len() >= params.run_queue_num_errs {
                        queue.spawn_errors.pop_front();
                    }
                },
            }
        } else {
            break;
        }
    }
}


#[derive(Deref, DerefMut, Component)]
pub struct ConfigComponent(Config);

/// Environment responsible for manipulating various configs, and passing them to TrainEnviron to train,
/// this does not know any low-level details about the configs.
#[derive(Serialize, Deserialize)]
pub struct ConfigEnviron {
    name: String,
    config: Config,
    default: Config,
    saved_configs: CheckedList<Config>,
    version_num: u32,
}

impl ConfigEnviron {
    pub fn new(name: &str, config: Config) -> Self {
        Self {
            name: name.to_string(),
            config: config.clone(),
            default: config,
            saved_configs: CheckedList { header: name.to_string() + " saved configs", deletion: true, ..default() },
            version_num: 0,
        }
    }

    pub fn get_config(&self) -> Config {
        self.config.clone()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let space = ui.available_size();
        ui.horizontal(|ui| {
            ui.allocate_ui(space, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        // reset current config logic
                        if self.saved_configs.checked.is_none() {
                            // nothing is checked, reset to default
                            if ui.button("reset config").clicked() {
                                self.config.update(&self.default).unwrap();
                            }
                        } else {
                            // something is checked, default to past config
                            let checked = self.saved_configs.checked.unwrap();
                            if ui
                                .button(format!("reset config with past config {}", checked))
                                .clicked()
                            {
                                if let Some(a) = self.saved_configs.get_checked() {
                                    self.config.update(a).unwrap();
                                }
                            }
                        }
                        // save current config logic
                        if ui.button("save config").clicked() {
                            self.saved_configs.push(self.config.clone());
                        }
                    });
                    egui::ScrollArea::vertical().id_source("configs").show(ui, |ui| {
                        self.config.ui(ui);
                    });
                });
                
                // implement adding and deletion from config stack
                self.saved_configs.ui(ui, |x, y| { x.update(&y).unwrap(); });

                // TODO: show past training runs
            });
        });
    }
}

#[derive(Default, Serialize, Deserialize)]
struct CheckedList<T> {
    header: String,
    pub_ui: Vec<T>,
    saved: Vec<T>,
    item_open: Vec<bool>,
    open: bool,
    deletion: bool,
    checked: Option<usize>,
}

impl<T: UI + Clone> CheckedList<T> {
    fn get_checked(&self) -> Option<&T> {
        self.checked.map(|x| &self.saved[x])
    }

    fn push(&mut self, x: T) {
        self.item_open.push(false);
        self.saved.push(x.clone());
        self.pub_ui.push(x);
    }

    fn close_all(&mut self) {
        self.item_open.iter_mut().for_each(|x| *x = false);
    }

    fn ui(&mut self, ui: &mut egui::Ui, update_fn: impl Fn(&mut T, &T)) {
        let is_open = egui::CollapsingHeader::new(&self.header).default_open(self.open).show(ui, |ui| {
            egui::ScrollArea::vertical().id_source("past configs").show(ui, |ui| {
                // use pub_runs as dummy display
                let mut i = 0;
                while i < self.pub_ui.len() {
                    // allow checked to be negative so it becomes possible for no
                    // option to be checked
                    let mut cur_check = self.checked.is_some() && i == self.checked.unwrap();
                    let mut removed_run = false;
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut cur_check, format!("config {}", i));
                        if self.deletion && ui.button("delete config").clicked() {
                            self.pub_ui.remove(i);
                            self.saved.remove(i);
                            removed_run = true;
                            if cur_check {
                                self.checked = None;
                            }
                        }
                        
                    });
                    if removed_run {
                        continue;
                    }
                    ui.push_id(format!("config panel checked {}", i), |ui| {
                        let is_open = egui::CollapsingHeader::new("").default_open(self.item_open[i]).show(ui, |ui| {
                            self.pub_ui[i].ui(ui);
                        }); 
                        self.item_open[i] = is_open.fully_open();
                    });

                    // we don't want past configs to change, so we have an immutable copy
                    update_fn(&mut self.pub_ui[i], &self.saved[i]);
                    // only one option can be checked at a time
                    let checked = self.checked.is_some() && i == self.checked.unwrap();
                    if cur_check {
                        self.checked = Some(i);
                    } else if checked {
                        self.checked = None;
                    }
                    i += 1;
                }
            });
        });
        self.open = is_open.fully_open();
    }
}


/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    plots: Res<run::ModelPlots>,
    console: Res<run::Console>,    
    running: Query<&run::RunInfo>,
) {
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("back to menu").clicked() {
                    state.set(AppState::Menu).unwrap();
                }

            });

            // the console and log graphs are part of the fore-ground egui panel
            // while any background rendering stuff is happening in a separate system, taking TrainResource as a parameter
            ui.collapsing("console", |ui| {
                console.console_ui(ui);
            });

            ui.collapsing("currently running", |ui| {
                for run in running.iter() {
                    run.show_basic(ui);
                }
            });

            // TODO: Add plotting utilites

        });
    });
}

// removes the first instance where f evaluates true, returns true is anything is removed, false otherwise
fn remove_once_if_any<T>(queue: &mut VecDeque<T>, mut f: impl FnMut(&T) -> bool) -> bool {
    let idx = {
        let mut u = -1;
        for (i, r) in queue.iter().enumerate() {
            if f(r) {
                u = i as isize;
                break;
            }
        }
        u
    };
    if idx != -1 {
        queue.remove(idx as usize);
        true
    } else {
        false
    }
}
