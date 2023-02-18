use bevy::prelude::*;
use anyhow::Result;

use crate::ui::train_ui::{self as ui};
use super::run_data::{self as run, Despawn, Kill, SpawnRun};
use super::{Config};

pub struct BaselinePlugin;
impl Plugin for BaselinePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_startup_system(setup_run)
            .add_system(run_baseline);
    }
}

#[derive(Component, Deref, DerefMut)]
struct BaseTrainProcess(run::models::TrainProcess);

#[derive(Resource)]
struct BaselineProcess {
    run_sender: run::RunSend
}

fn setup_run(mut commands: Commands, sender: ResMut<run::RunSend>) {
    commands.insert_resource(BaselineProcess { run_sender: sender.clone() });
}

/// Add plots and run data contributions to their respective containers
fn run_baseline(
    mut despawner: EventWriter<Despawn>,
    mut killer: EventReader<Kill>,
    mut plots: ResMut<run::ModelPlots>,
    mut console: ResMut<run::Console>,
    mut run_stats: ResMut<run::RunStats>,
    mut runs: Query<(Entity, &mut run::RunInfo, &mut BaseTrainProcess)>,
    run_sender: ResMut<BaselineProcess>,
) {
    use run::{TrainRecv};
    for (id, info, mut train_proc) in runs.iter_mut() {
        if train_proc.is_running() {
            let msgs = train_proc.try_recv();
            for msg in msgs {
                match msg {
                    TrainRecv::PLOT(point) => {
                        console.log(format!("Logged {}, {}: {}, {}: {}", point.title, point.x_title, point.x, point.y_title, point.y));
                        plots.add_point(&run::PlotId { 
                            model: run::Models::BASELINE, 
                            run_name: info.run_name(), 
                            title: point.title.into(),
                            x_title: point.x_title.into(),
                            y_title: point.y_title.into(),
                         }, (point.x, point.y));
                    }
                    TrainRecv::FAILED(err_msg) => {
                        console.log(format!("Error {} while training {}", err_msg, info.run_name()));
                        // the training run has failed => thread exited => free resources
                        despawner.send(Despawn(id));
                        let mut info = info.clone();
                        info.err_status = Some(err_msg);
                        run_sender.run_sender.send(run::RunId(run::Models::BASELINE, info, id)).expect("unable to send baseline run info");
                    },
                    TrainRecv::STATS(stats) => {
                        run_stats.update(id, stats);
                    }
                    // TrainRecv::CHECKPOINT(step, path) => {
                    //     console.log(format!("saving checkpoint for {} at step {}", info.run_name(), step));
                    //     console.log(format!("saving to {}", path.to_str().unwrap()));
                    //     info.add_checkpoint(step, path);
                    // },
                }
            }
        } else {
            console.log(format!("{} finished training", info.run_name()));
            let mut info = info.clone();
            info.err_status = None;
            run_sender.run_sender.send(run::RunId(run::Models::BASELINE, info, id)).expect("unable to send baseline run info");
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