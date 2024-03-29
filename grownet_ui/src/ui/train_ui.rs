use std::collections::{HashMap, VecDeque, HashSet};
use std::path::PathBuf;

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy::window::WindowCloseRequested;
use bevy_egui::{egui, EguiContext};
use bevy::ecs::schedule::ShouldRun;
use crossbeam::channel::{Receiver, Sender};
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use model_lib::models::{self, TrainRecv};
use model_lib::Config;

use crate::{ops, config_ui_adjust};
use crate::run_systems::{self as run, config_ui_show, ModelPlots, PlotViewerV1, PlotViewerV2};
use run::{Models, Despawn, Kill, Spawn, SpawnRun};
use super::{Serializer, AppState, OperatingState, OpenPanel, UIParams, handle_pane_options};


pub struct TrainUIPlugin;
impl Plugin for TrainUIPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugin(run::RunDataPlugin) // setup any running data, such as plot tracking, etc.
            .insert_resource(RunQueue::default())
            .insert_resource(TrainingUI::default())
            .add_startup_system(setup_train_ui) // load UIParams from disk
            .add_system_set(
                SystemSet::on_update(AppState::Models)
                    .label("train_menu")
                    .with_system(train_menu_ui)
            )
            .add_system_set(SystemSet::on_update(AppState::Trainer)
                .with_system(train_env_ui)
                // .with_system(queue_ui)
            )
            .add_system_set(
                SystemSet::on_update(OperatingState::Active).with_system(run_queue))
            .add_system_set(
                SystemSet::on_update(OperatingState::Cleanup).with_system(cleanup_queue))
            .add_system_set(
                SystemSet::on_update(OperatingState::Close).with_system(save_train_ui))
            .add_plugin(run::baseline::BaselinePlugin);
    }
}

/// Main menu to launch and configure training tasks
fn train_menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut app_state: ResMut<State<AppState>>,
    op_state: Res<State<OperatingState>>,
    mut params: ResMut<UIParams>,
    mut train_ui: ResMut<TrainingUI>,
    mut run_queue: ResMut<RunQueue>,
    mut plot_viewer: ResMut<PlotViewerV2>,
    plots: Res<ModelPlots>,
    stats: Res<run::RunStats>,
    run_recv: ResMut<run::RunRecv>,
    killer: EventWriter<Kill>,
    mut config_width_delta: Local<f32>
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

        let height = ui.available_height();
        ui.horizontal(|ui| {
            let original_width = ui.available_width();
            let width = original_width * *config_width_delta;
            // do this to let the inner config environments to set their preferred width. Otherwise, elements
            // before the widest element is squished.
            let mut needed_width = 1.0;
            // update any configurations using the ui, we need to set height this way, as otherwise inner scrolling
            // widgets will not work
            ui.allocate_ui(egui::Vec2::new(width, height), |ui| {
                ui.vertical(|ui| {
                    // which model configuration to show
                    egui::ComboBox::from_label("models")
                        .selected_text(format!("{}", train_ui.model))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut train_ui.model, Models::BASELINE, "baseline");
                        });
        
                    // load any runinfos sent from training processes
                    while let Ok(run) = run_recv.try_recv() {
                        if !train_ui.run_ids.contains(&run.2) {
                            train_ui.run_ids.insert(run.2);
                            match run.0 {
                                Models::BASELINE => { train_ui.baseline.add_run(run.1); }
                            }
                        }
                    }

                    // The config environments of each specific model type
                    match train_ui.model {
                        run::Models::BASELINE => {
                            needed_width = train_ui.baseline.ui(ui).width();
                        }
                    }

                    // TODO: add some keybindings to launch training tasks
                    // TODO: make this section stick to the bottom
                    ui.with_layout(egui::Layout::top_down(egui::Align::BOTTOM), |ui| {
                        // entry point for launching training
                        // only launch things if the operating state is active
                        if *op_state.current() == OperatingState::Active && ui.button("Launch Training").clicked() {
                            match train_ui.model {
                                run::Models::BASELINE => {
                                    let (spawn_fn, runinfo) = 
                                        run::baseline::baseline_spawn_fn(train_ui.baseline.version_num as usize, train_ui.baseline.get_config(), train_ui.baseline.get_global_config());
                                    //app_state.set(AppState::Trainer).unwrap();
                                    train_ui.baseline.version_num += 1;
                                    run_queue.add_run(runinfo, spawn_fn);
                                }
                            }
                        }
                        if *op_state.current() == OperatingState::Cleanup {
                            ui.label("killing any active tasks");
                        }
                        // the running queue displays the status of running tasks
                        run_queue.ui(ui, killer, &*stats);
                    });

                });

            });

            // set the preferred width of the inner config environs
            *config_width_delta = needed_width / original_width;

            let width = ui.available_width();
            // plot viewer
            // once again, we set the maximum height so that inner scolling widgets are not squished
            // plot_viewer expects horizontal to be the layout
            ui.allocate_ui(egui::Vec2::new(width, height), |ui| {
                plot_viewer.ui(ui, &*plots);
            });
            
        });
    });
}


/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
fn train_env_ui(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    mut queue: ResMut<RunQueue>,
    stats: Res<run::RunStats>,
    killer: EventWriter<Kill>,
    // mut viewer: ResMut<PlotViewerV1>,
    // plots: Res<ModelPlots>, 
    console: Res<run::Console>,    
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

            // ui.separator();
            // ui.heading("plots");
            // viewer.ui(ui, &*plots);
            queue.ui(ui, killer, &*stats);
        });
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
        ops::remove_once_if_any(&mut queue.active_runs, |x| { x.1 == i.0 });
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
        if ops::remove_once_if_any(&mut queue.active_runs, |x| {x.1 == id}) {
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

/// possibly load any training state from disk
/// Startup System
fn setup_train_ui(
    mut train_ui: ResMut<TrainingUI>,
    serializer: Res<Serializer>
) {
    serializer.deserialize("train_ui", &mut *train_ui);
}

/// write train state to disk
/// Shutdown system
fn save_train_ui(
    train_ui: Res<TrainingUI>,
    mut serializer: ResMut<Serializer>
) {
    serializer.serialize("train_ui", &*train_ui)
}



/// TrainingUI is the menu in which one adjusts configurations and launches training processes
/// It contains a list of past configurations, and options to kill tasks and restart tasks
#[derive(Serialize, Deserialize, Resource)]
pub struct TrainingUI {
    baseline: ConfigEnviron,
    model: run::Models,
    #[serde(skip)]
    run_ids: std::collections::HashSet<Entity>,
}

impl Default for TrainingUI {
    fn default() -> Self {
        use model_lib::*;
        Self { 
            baseline: ConfigEnviron::new(
                "baseline",
                models::baselinev3::baseline_config(),
                config!()
            ), 
            model: run::Models::BASELINE,
            run_ids: HashSet::new()
        }
    }
}

/// Environment responsible for manipulating various configs, and passing them to TrainEnviron to train,
/// this does not know any low-level details about the configs.
#[derive(Serialize, Deserialize)]
pub struct ConfigEnviron {
    name: String,
    config: Config,
    default: Config,
    // saved_configs: CheckedList<Config>,
    saved_runs: CheckedList<run::RunInfo>,
    version_num: u32,
    global_config: Config,
    // checkpoint configs
    // checkpoint_folder: PathBuf,
    // num_kept_checkpoints: u32,
}

impl ConfigEnviron {
    pub fn new(name: &str, config: Config, global_config: Config) -> Self {
        // let checkpoint_folder = PathBuf::from(RUN_DATA_PATH).join(name);
        // if !checkpoint_folder.exists() {
        //     std::fs::create_dir(&checkpoint_folder).expect(&format!("unable to create checkpoint folder for {}", name));
        // }
        Self {
            name: name.to_string(),
            config: config.clone(),
            default: config,
            // saved_configs: CheckedList { header: name.to_string() + " saved configs", deletion: true, ..default() },
            saved_runs: CheckedList { title: name.to_string() + " saved runs", default_open: false, deletion: true, ..default()},
            version_num: 0,
            global_config
            // num_kept_checkpoints: 3,
            // checkpoint_folder,
        }
    }

    pub fn get_config(&self) -> Config {
        self.config.clone()
    }

    pub fn get_global_config(&self) -> Config {
        self.global_config.clone()
    }

    pub fn add_run(&mut self, run: run::RunInfo) {
        self.saved_runs.add(run);
    }

    pub fn get_run(&self) -> Option<&run::RunInfo> {
        self.saved_runs.get_checked().or_else(|| self.saved_runs.get_latest())
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let response = ui.group(|ui| {
            egui::ScrollArea::vertical().id_source("global config").show(ui, |ui| {
                ui.label(egui::RichText::new("global config").heading().underline());
                ui.separator();
                config_ui_adjust(&mut self.global_config, ui);
                ui.separator();
                ui.label(egui::RichText::new("local config").heading().underline());
                ui.separator();
                ui.horizontal(|ui| {
                    // reset current config logic
                    if !self.saved_runs.is_checked() {
                        if ui.button("reset local config").clicked() {
                            self.config.update(&self.default).unwrap();
                        }
                    } else {
                        // something is checked, default to past config
                        let checked = self.saved_runs.get_checked_num().unwrap();
                        if ui
                            .button(format!("reset local with past config {}", checked))
                            .clicked()
                        {
                            if let Some(a) = self.saved_runs.get_checked() {
                                self.config.update(&a.config).unwrap();
                            }
                        }
                    }
                });
                
                config_ui_adjust(&mut self.config, ui);
                ui.separator();
                
                ui.collapsing("past configs", |ui| {
                    self.saved_runs.ui(ui, |ui, run| { run.show_basic(ui); });
                });
            });
        });
        response.response.rect
        
        
        // implement adding and deletion from config stack
        // self.saved_configs.ui(ui, |x, y| { x.update(&y).unwrap(); });
        
        // TODO: show past training runs

    }
}

/// A wrapper struct owning a list of values, providing a ui method which allows insertion and deletion from that list
#[derive(Serialize, Deserialize, Default)]
struct CheckedList<T> {
    title: String,
    saved: VecDeque<T>,      // the saved items
    is_open: VecDeque<bool>, // the collapsing header is open
    default_open: bool,      // whether each new addition is open on default
    deletion: bool,          // support deletion
    checked: Option<usize>   // current checked position
}

impl<T> CheckedList<T> {
    pub fn is_checked(&self) -> bool {
        self.checked.is_some()
    }

    pub fn get_latest(&self) -> Option<&T> {
        self.saved.get(0)
    }

    pub fn get_checked_num(&self) -> Option<usize> {
        self.checked
    }

    pub fn get_checked(&self) -> Option<&T> {
        self.checked.map(|x| &self.saved[x])
    }

    pub fn add(&mut self, v: T) {
        self.saved.push_front(v);
        self.is_open.push_front(self.default_open);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, mut f: impl FnMut(&mut egui::Ui, &T)) {
            // use pub_runs as dummy display
            let mut i = 0;
            while i < self.saved.len() {
                // allow checked to be negative so it becomes possible for no
                // option to be checked
                let mut cur_check = self.checked.is_some() && i == self.checked.unwrap();
                let mut removed_run = false;
                // heading for each collapsing header
                ui.horizontal(|ui| {
                    ui.checkbox(&mut cur_check, format!("{}", i));
                    if self.deletion && ui.button("delete").clicked() {
                        self.saved.remove(i);
                        self.is_open.remove(i);
                        if cur_check {
                            self.checked = None;
                        }
                        removed_run = true;
                    }                    
                });
                if removed_run {
                    continue;
                }
                ui.push_id(format!("checked box panel open {}", i), |ui| {
                    let is_open = egui::CollapsingHeader::new("").default_open(self.is_open[i]).show(ui, |ui| {
                        f(ui, &self.saved[i]);
                    }); 
                    self.is_open[i] = is_open.fully_open();
                });

                // only one option can be checked at a time
                let checked = self.checked.is_some() && i == self.checked.unwrap();
                if cur_check {
                    self.checked = Some(i);
                } else if checked {
                    self.checked = None;
                }
                i += 1;
            }
    }
    
}

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

    fn ui(&mut self, ui: &mut egui::Ui, mut kill: EventWriter<Kill>, stats: &run::RunStats) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            // show errors
            if self.spawn_errors.len() > 0 {
                ui.horizontal(|ui| {
                    ui.label("launch errors");
                    if ui.button("clear").clicked() {
                        self.spawn_errors.clear();
                    }
                });
                for msg in self.spawn_errors.iter() {
                    ui.label(egui::RichText::new(msg).color(egui::Color32::RED));
                }
                //ui.separator();
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
                let cur_run = &mut self.active_runs[i];
                ui.horizontal(|ui| {
                    if ui.button("kill").clicked() {
                        kill.send(Kill(cur_run.1));
                    }
                    ui.vertical(|ui| {
                        ui.collapsing(cur_run.0.run_name(), |ui| {
                            if stats.has_stat(cur_run.1) {
                                ui.vertical(|ui| {
                                    stats.show_basic_stats(cur_run.1, ui);
                                });
                            }
                            cur_run.0.show_basic(ui);
                        });
                    });
                });
            }
        });
    }
}

