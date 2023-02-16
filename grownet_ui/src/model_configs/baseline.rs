use bevy::prelude::*;
use anyhow::Result;

use crate::ui::train_ui::{self as ui, Despawn, Kill};
use super::run_data as run;
use super::{Config, SpawnRun};

pub struct BaselinePlugin;
impl Plugin for BaselinePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_system(run_baseline);
    }
}

#[derive(Component, Deref, DerefMut)]
struct BaseTrainProcess(run::models::TrainProcess);


/// Add plots and run data contributions to their respective containers
fn run_baseline(
    mut despawner: EventWriter<Despawn>,
    mut killer: EventReader<Kill>,
    
    mut plots: ResMut<run::ModelPlots>,
    mut console: ResMut<run::Console>,
    mut model_runinfos: ResMut<run::ModelRunInfo>,
    mut runs: Query<(Entity, &mut run::RunInfo, &mut BaseTrainProcess)>,
) {
    use run::{TrainRecv};
    for (id, mut info, mut train_proc) in runs.iter_mut() {
        if train_proc.is_running() {
            let msgs = train_proc.try_recv();
            for msg in msgs {
                match msg {
                    TrainRecv::PLOT(name, x, y) => {
                        console.log(format!("Logged plot name: {}, x: {}, y: {}", &name, x, y));
                        plots.add_plot(&name, &info.run_name(), x, y);
                    }
                    TrainRecv::FAILED(err_msg) => {
                        console.log(format!("Error {} while training {}", err_msg, info.run_name()));
                        // the training run has failed => thread exited => free resources
                        despawner.send(Despawn(id));
                        let mut info = info.clone();
                        info.err_status = Some(err_msg);
                        model_runinfos.add_info(run::Models::BASELINE, info.run_name(), info).unwrap();
                    },
                    TrainRecv::CHECKPOINT(step, path) => {
                        console.log(format!("saving checkpoint for {} at step {}", info.run_name(), step));
                        console.log(format!("saving to {}", path.to_str().unwrap()));
                        info.add_checkpoint(step, path);
                    },
                }
            }
        } else {
            console.log(format!("{} finished training", info.run_name()));
            let info = info.clone();
            model_runinfos.add_info(run::Models::BASELINE, info.run_name(), info).unwrap(); // todo: better runinfo and checkpoint managers
            despawner.send(Despawn(id));
        }
    }
    // detects if any needs to be killed
    // not the most efficient, but there aren't that many runs
    for i in killer.iter() {
        for (id, _, mut run) in runs.iter_mut() {
            if i.0 == id {
                run.try_kill();
                break;
            }
        }
    }
}

fn cleanup_baselines(
    mut runs: Query<(&run::RunInfo, &mut BaseTrainProcess)>,
    mut model_runinfos: ResMut<run::ModelRunInfo>,
) {
    for (info, mut proc) in runs.iter_mut() {
        proc.try_kill();
        model_runinfos.add_info(run::Models::BASELINE, info.run_name(), info.clone()).unwrap();
    }
}

pub fn baseline_spawn_fn(version_num: usize, config: Config) -> (Box<dyn FnOnce(&mut Commands) -> Result<Entity> + Send + Sync>, run::RunInfo) {
    let runinfo = run::RunInfo {
        model_class: "baseline".into(),
        version: version_num,
        dataset: "cifar10".into(),
        config: config.clone(),
        ..Default::default()
    };
    let run_info = runinfo.clone();
    let spawn_fn = Box::new(move |commands: &mut Commands| -> Result<Entity> {
        let config = config;
        run::models::baseline::build(&config).map(|x| {
            let env = BaseTrainProcess(x);
            let id = commands.spawn((run_info, env)).id();
            id
        })
    });
    (spawn_fn, runinfo)
}