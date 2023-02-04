/*
fn handle_baseline_logs(
    mut model_runs: Res<run::ModelRuns>,
    mut plots: Res<run::ModelPlots>,
    mut console: Res<run::Console>,
    mut err_ev: EventWriter<TrainError>,
) {
    // if train_env.is_running() {
    //     match train_env.selected() {
    //         Models::BASELINE => {
    //             let logs = train_env.baseline.run_data();
    //             if let Err(e) = train_data.handle_baseline_logs(&logs) {
    //                 err_ev.send(TrainError(format!("{e}")));
    //             }
    //         }
    //     }
    // }
}

/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    mut train_env: ResMut<TrainEnviron>,
    mut err_ev: EventReader<TrainError>,
    mut local_errors: Local<Vec<String>>, // some local error messages possibly relayed by err_ev
    train_data: Mut<TrainData>,
) {
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        egui::ScrollArea::vertical().show(ui, |ui| {
            if ui.button("back to menu").clicked() {
                state.set(AppState::Menu).unwrap();
            }
            if ui.button("stop training").clicked() {
                match train_env.selected() {
                    run::Models::BASELINE => {
                        train_env.baseline.kill_blocking();
                    }
                }
            }
            for msg in err_ev.iter() {
                local_errors.push((*msg).clone());
            }

            if local_errors.len() > 0 {
                for i in &*local_errors {
                    ui.label(i);
                }
            }

            // the console and log graphs are part of the fore-ground egui panel
            // while any background rendering stuff is happening in a separate system, taking TrainResource as a parameter
            ui.collapsing("console", |ui| {
                console_ui(&train_data.console, ui);
            });

            // TODO: Add plotting utilites

        });
    });
}
*/
