use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, Range};
use bevy::utils::HashSet;
use bevy_inspector_egui::egui::TextureHandle;
use crossbeam::channel::{Sender, Receiver};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::egui;
use itertools::Itertools;
use model_lib::models::PlotPoint;
use plotters::coord::types::RangedCoordf64;
use plotters::style::Color;
use serde::{Deserialize, Serialize};
use plotters::prelude::*;

pub use model_lib::{models, Config};
pub use models::{TrainProcess, TrainRecv, TrainSend};
pub use crate::ui::OperatingState;


use crate::ops;
use crate::CONFIG_PATH;

/// Plugin to instantiate all run data resources, and saving/loading logic
pub struct RunDataPlugin;
impl Plugin for RunDataPlugin {
    fn build(&self, app: &mut App) {
        let (send, recv) = crossbeam::channel::unbounded();
        let run_sender = RunSend(send);
        let run_recv = RunRecv(recv);
        app
            .add_event::<Despawn>()
            .add_event::<Kill>()
            .insert_resource(run_sender)
            .insert_resource(run_recv)
            .insert_resource(ModelPlots::default())
            .insert_resource(Console::default())
            .insert_resource(RunStats::default())
            .add_startup_system(setup_run_data)
            .add_system_set(
                SystemSet::on_update(OperatingState::Close).with_system(save_run_data));
    }
}

/// possibly load run data from disk
fn setup_run_data(
    mut plots: ResMut<ModelPlots>,
    mut console: ResMut<Console>,
) {
    eprintln!("loading run data");
    ops::try_deserialize(&mut *plots, &(CONFIG_PATH.to_owned() + "/model_plots.config").into());
    ops::try_deserialize(&mut *console, &(CONFIG_PATH.to_owned() + "/model_console.config").into());
}

/// write run data to disk
fn save_run_data(
    plots: Res<ModelPlots>,
    console: Res<Console>,
) {
    // load configurations from disk
    let root_path: std::path::PathBuf = CONFIG_PATH.into();

    eprintln!("serializing run_data");
    // save config files to disk
    ops::serialize(&*plots, &root_path.join("model_plots").with_extension("config"));
    ops::serialize(&*console, &root_path.join("model_console").with_extension("config"));
}

/// Enum of all the model variants
#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
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

/// Send Runs to UI
#[derive(Resource, Deref, DerefMut, Clone)]
pub struct RunSend(Sender<RunId>);

/// Receive Runs from UI
#[derive(Resource, Deref, DerefMut)]
pub struct RunRecv(Receiver<RunId>);

/// A struct which fully identifies the model
pub struct RunId(pub Models, pub RunInfo, pub Entity);

/// This struct represents an individual training run, it has the information to restart itself
#[derive(Serialize, Deserialize, Default, Clone, Component)]
pub struct RunInfo {
    pub config: Config,             // TODO: convert types to this more easily,
    pub model_class: String, // name for the class of models this falls under
    pub version: usize,      // id for this run
    pub comments: String,
    pub dataset: String,
    pub err_status: Option<String>, // True is returned successfully, false if Killed mid-run
    // pub checkpoints: Vec<(f32, std::path::PathBuf)>, // (step, path)
}

impl RunInfo {
    pub fn run_name(&self) -> String {
        format!("{}-v{}", self.model_class, self.version)
    }

    // pub fn add_checkpoint(&mut self, step: f32, path: std::path::PathBuf) {
    //     self.checkpoints.push((step, path));
    // }

    // pub fn get_checkpoint(&self, i: usize) -> Option<std::path::PathBuf> {
    //     self.checkpoints.get(i).and_then(|x| Some(x.1.clone()))
    // }

    pub fn show_basic(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            if self.comments.len() > 0 {
                ui.collapsing("comments", |ui| {
                    ui.label(&self.comments);
                });
            }

            // ui.collapsing("checkpoints", |ui| {
            //     egui::ScrollArea::vertical().id_source("click checkpoints").show(ui, |ui| {

            //         for (j, checkpoint) in self.checkpoints.iter() {
            //             // TODO: when checkpoint is clicked, show loss as well
            //             ui.horizontal(|ui| {
            //                 ui.label(format!("step {}", j));
            //                 ui.label(checkpoint.to_str().unwrap());
            //             });
            //         }
            //     });
            // });
            if self.err_status.is_some() {
                ui.label(format!("error status: {:?}", self.err_status));
            }
            ui.label(format!("dataset: {}", self.dataset));
            ui.label(format!("model class: {}", self.model_class));
            ui.collapsing("run configs", |ui| {
                super::immutable_show(&self.config, ui);
            });
        });

    }
}

/// Tracking performance, memory usage, etc.
#[derive(Resource, Default)]
pub struct RunStats {
    runs: HashMap<Entity, models::RunStats>
}

impl RunStats {
    pub fn has_stat(&self, id: Entity) -> bool {
        self.runs.contains_key(&id)
    }

    pub fn update(&mut self, id: Entity, stats: models::RunStats) {
        self.runs.insert(id, stats);
    }

    pub fn show_basic_stats(&self, id: Entity, ui: &mut egui::Ui) {
        if let Some(stat) = self.runs.get(&id) {
            if let Some(step_time) = stat.step_time {
                ui.label(format!("step time {:.5}s", step_time));
            }
        }
    }
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
pub struct Spawn(pub RunInfo, pub SpawnRun);


/// a line in (x, y), where x is guaranteed to be monotonic (strictly increasing)
#[derive(Serialize, Deserialize, Deref, DerefMut, Clone, Default, Debug)]
pub struct PlotLine(Vec<(f64, f64)>);

impl PlotLine {
    /// only adds a point to the plot if the x coordinate is strictly greater than the last x coordinate of self
    pub fn add(&mut self, p: (f64, f64)) {
        if self.len() == 0 || self.last().unwrap().0 < p.0 {
            self.push(p);
        }
    }

    /// extends self by other[i..] where other[i].0 is greater than the last x coordinate of self
    pub fn merge(&mut self, other: &PlotLine) {
        let i = if self.len() > 0 {
            let x = self.last().unwrap().0;
            let mut i = 0;
            for (j, y) in other.iter().enumerate() {
                if y.0 > x {
                    i = j;
                    break;
                }
            }
            i
        } else {
            0
        };
        self.extend_from_slice(&other[i..]);
    }

    /// applies a sliding average window to self, with window-1 0 padding to the left
    pub fn avg_smooth(&mut self, window: usize) {
        let div = window as f64;
        let mut sum = 0.0;
        for i in self.len().max(window) - window..self.len() {
            sum += self[i].1;
        }


        for i in (window..self.len()).rev() {
            let x = self[i].1;
            self[i].1 = sum / div;
            let w = self[i - window].1;
            sum += w - x;
        }

        for i in (0..self.len().min(window)).rev() {
            let x = self[i].1;
            self[i].1 = sum / (i + 1) as f64;
            sum -= x;
        }
    }
}

/// Uniquely identifies a line for a particular run
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Serialize, Deserialize, Default)]
pub struct PlotId {
    pub model: Models,
    pub run_name: String,
    pub title: String,
    pub x_title: String,
    pub y_title: String,
}

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub struct PlotIdRef<'a> {
    pub model: &'a Models,
    pub run_name: &'a str,
    pub title: &'a str,
}


#[derive(Serialize, Deserialize, Resource, Default)]
pub struct ModelPlots {
    lines: Vec<(PlotLine, PlotId)>,
    by_id: HashMap<PlotId, usize>,
}

impl ModelPlots {
    pub fn filter(&mut self, mut f: impl FnMut(&PlotId) -> bool) -> impl Iterator<Item = &mut (PlotLine, PlotId)> {
        self.lines.iter_mut().filter(move |x| f(&x.1))
    }

    pub fn get(&mut self, id: &PlotId) -> Option<&mut PlotLine> {
        self.by_id.get(id).and_then(|x| Some(&mut self.lines[*x].0))
    }

    pub fn contains(&self, id: &PlotId) -> bool {
        self.by_id.contains_key(id)
    }

    pub fn insert(&mut self, id: PlotId, line: PlotLine) {
        if !self.contains(&id) {
            self.by_id.insert(id.clone(), self.lines.len());
            self.lines.push((line, id));
        } else {
            let idx = *self.by_id.get(&id).unwrap();
            self.lines[idx] = (line, id);
        }
    }

    pub fn add_point(&mut self, id: &PlotId, point: (f64, f64)) {
        self.get(id).and_then(|x| Some(x.add(point)));
    }
}

#[derive(Default)]
struct ImageGridUi {
    images: Vec<egui::TextureHandle>,
    order: Vec<usize>,
}

impl ImageGridUi {
    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn flush(&mut self) {
        self.images.clear();
        self.order.clear();
    }

    pub fn take(&mut self, i: usize) -> egui::TextureHandle {
        assert!(i < self.len() && self.len() > 0);
        let idx = self.order.iter().find_position(|x| **x == self.len() - 1).unwrap().0;
        self.order.remove(idx);
        self.order.iter_mut().for_each(|x| if *x >= i { *x += 1; });
        self.images.remove(i)
    }

    pub fn insert(&mut self, i: usize, buf: egui::TextureHandle) {
        assert!(i < self.len());
        self.order.iter_mut().for_each(|x| if *x >= i { *x += 1; });
        self.order.push(self.images.len());
        self.images.insert(i, buf);
    }

    pub fn replace(&mut self, i: usize, buf: egui::TextureHandle) {
        assert!(i < self.len());
        self.images[i] = buf;
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        let temp = self.order[i];
        self.order[i] = self.order[j];
        self.order[j] = temp;
    }

    pub fn push(&mut self, buf: egui::TextureHandle) {
        self.order.push(self.len());
        self.images.push(buf);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, scale: f32) {
        let x = ui.max_rect().width();
        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut i = 0;
            while i < self.images.len() {
                ui.horizontal(|ui| {
                    let mut xi = x;
                    while xi > 0.0 {
                        let size = self.images[self.order[i]].size_vec2() * scale;
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                let mut order = self.order[i];
                                ui.add(egui::DragValue::new(&mut order));
                                order = order.min(self.len());
                                // if order is changed, shift everything between the new order and old order by 1
                                if order != self.order[i] {
                                    let old_order = self.order[i];
                                    let buf = self.take(old_order);
                                    self.insert(order, buf);
                                }
                                ui.image(&self.images[self.order[i]], size);
                            });
                        });
                        xi -= size.x;
                        i += 1;
                    }
                });
            }
        });
    }
}


pub fn wider_range(a: Range<f64>, b: Range<f64>) -> Range<f64> {
    a.start.min(b.start)..a.end.max(b.end)
}

pub fn compute_bounds<'a>(lines: impl Iterator<Item = (f64, f64)>) -> (Range<f64>, Range<f64>) {
    const INIT_BOUND: (Range<f64>, Range<f64>) = (0.0..0.0, 0.0..0.0);
    lines.fold(INIT_BOUND, |acc, val| {
        (wider_range(acc.0, val.0..val.0), wider_range(acc.1, val.1..val.1))
    })
}

fn render<'a>(
    title: &str,
    x_title: Option<&str>,
    y_title: Option<&str>,
    lines: impl Iterator<Item = (&'a str, &'a [(f64, f64)])> + Clone,
    res: (usize, usize)
) -> Result<Vec<u8>> {
    let mut buf = vec![255; res.0 * res.1 * 3]; // rgb format

    {
        let root = BitMapBackend::with_buffer(&mut buf, (res.0 as u32, res.1 as u32));

        let bounds = compute_bounds(
            lines.clone().map(|x| x.1.iter()).flatten().map(|x| *x));
        
        let area = root.into_drawing_area();
        let mut chart = ChartBuilder::on(&area)
            .caption(title, ("sans-serif", (5).percent_height()))
            .set_label_area_size(LabelAreaPosition::Left, (8).percent())
            .set_label_area_size(LabelAreaPosition::Bottom, (4).percent())
            .margin((1).percent())
            .build_cartesian_2d(
                bounds.0,
                bounds.1
            )?;
        
        let mut c = chart.configure_mesh();
        if let Some(x) = x_title { c.x_desc(x); }
        if let Some(y) = y_title { c.y_desc(y); }
        c.draw()?;

        for (idx, (name, line)) in lines.enumerate() {
            let color = Palette99::pick(idx).mix(0.9);
            chart
                .draw_series(LineSeries::new(
                    line.iter().map(|x| (x.0, x.1))
                , color.stroke_width(3)))?
                .label(name)
                .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;

        area.present()?;
    }

    Ok(buf)
}

fn to_texture(buf: &[u8], res: (usize, usize), ui: &mut egui::Ui) -> Result<egui::TextureHandle> {
    if buf.len() != res.0 * res.1 * 3 {
        return Err(Error::msg("incorrect length of buffer for the given resolution"));
    }

    let colorbuf: Vec<_> = buf.chunks_exact(3).map(|x| {
        egui::Color32::from_rgb(x[0], x[1], x[1])
    }).collect();

    let colorimage = egui::ColorImage {
        size: [res.0, res.1],
        pixels: colorbuf
    };

    let handle = ui.ctx().load_texture(
        "render chart to texture", colorimage, egui::TextureOptions::NEAREST);
    
    Ok(handle)
}

fn smooth_window(data: &[(f64, f64)], window_size: usize) -> Vec<(f64, f64)> {
    let mut vec = Vec::new();
    let div = window_size as f64;
    vec.reserve_exact(data.len());
    let mut sum = 0.0;
    for i in 0..(data.len().min(window_size - 1)) {
        sum += data[i].1;
        vec.push((data[i].0, sum / (i + 1) as f64));
    }
    
    for j in (data.len().min(window_size - 1))..data.len() {
        sum += data[j].1;
        vec.push((data[j].0, sum / div));
        sum -= data[j + 1 - window_size].1;
    }
    vec
}


#[derive(Serialize, Deserialize)]
pub struct PlotViewer {
    smooth: usize,
    res: (usize, usize),
    scale: f32,
    #[serde(skip)]
    charts: ImageGridUi, // An ImageGrid which has an internal buffer containing images, and a ordering to those images 
    #[serde(skip)]
    needs_render: HashMap<PlotId, bool>, // the lines that have been changed by the ui, and needs re-rendering
    #[serde(skip)]
    corresponding: HashMap<String, usize>, // title, im_buffer idx; the corresponding title to buffer in self.charts
    #[serde(skip)]
    cache: HashMap<PlotId, PlotLine>, // unique lines as represented by a hash of PlotId, dependent upon x_title, y_title, title, model and run_name
}

impl PlotViewer {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // adjust local rendering parameters
        ui.horizontal(|ui| {
            ui.label("rendered resolution: (x, y)");
            let mut needs_change = ui.add(egui::DragValue::new(&mut self.res.0)).changed();
            self.res.0 = self.res.0.max(64).min(2048);
            needs_change |= ui.add(egui::DragValue::new(&mut self.res.1)).changed();
            self.res.0 = self.res.0.max(64).min(2048);
            ui.label("smooth");
            needs_change |= ui.add(egui::DragValue::new(&mut self.smooth)).changed();
            self.smooth = self.smooth.max(1);
            ui.label("local scale");
            ui.add(egui::DragValue::new(&mut self.scale));
            if needs_change {
                self.mark_change_all();
            }
        });


        self.charts.ui(ui, self.scale);
    }

    pub fn mark_change_all(&mut self) {
        self.needs_render.iter_mut().for_each(|x| { *x.1 = true; })
    }


    pub fn render(&mut self, ui: &mut egui::Ui) -> Result<Vec<(&str, egui::TextureHandle)>> {
        let collected = self.needs_render.iter_mut()
            .filter_map(|x| { if *x.1 { *x.1 = false; Some(x.0)} else { None }  }) // filter for needs_change
            .map(|x| {  // apply line smoothing
                let mut line = self.cache.get(x).unwrap().clone();
                line.avg_smooth(self.smooth);
                (x, line)
            })
            .fold(HashMap::new(), |mut map, x| {  // collect like titles
                let prod = (&x.0.title, &x.0.x_title, &x.0.y_title);
                if !map.contains_key(&prod) { map.insert(prod, vec![x]); }
                else { map.get_mut(&prod).unwrap().push(x); }
                map
            });
        let new_charts: Result<Vec<_>> = collected.iter().map(|((title, x_title, y_title), line)| { // convert to expected input to render
                ((title, x_title, y_title), line.iter().map(|x| (&(*x.0.run_name), &x.1.0[..])))
            })
            .map(|((title, x_title, y_title), line)| {  // render each separately
                render(title, Some(x_title), Some(y_title), line, self.res).map(|x| (title, x))
            })
            .collect();  // move result outside
        let new_charts: Result<Vec<(&str, egui::TextureHandle)>> = new_charts?.iter().map(|(title, buf)| {
            let s: &str = &***title;
            let t: Result<(&str, _)> = to_texture(&buf, self.res, ui).map(|x| (s, x));
            t
        }).collect();        
        new_charts
    }

    pub fn update_image_buffers<'a>(&mut self, new_buffers: Vec<(&'a str, egui::TextureHandle)>) {
        for (title, handle) in new_buffers {
            if !self.corresponding.contains_key(title) {
                // add the new image to the last place
                self.charts.push(handle);
                self.corresponding.insert(title.to_string(), self.charts.len());
            } else {
                self.charts.replace(*self.corresponding.get(title).unwrap(), handle);
            }
        }
    }

    pub fn flush_image(&mut self) {
        self.charts.flush();
        self.corresponding.clear();
    }

    pub fn flush_cache(&mut self) {
        self.cache.clear();
        self.needs_render.clear();
    }

    pub fn update_cache<'a>(&mut self, lines: impl Iterator<Item = (&'a PlotId, &'a PlotLine)>) {
        for line in lines {
            if !self.cache.contains_key(&line.0) {
                self.needs_render.insert(line.0.clone(), true);
                self.cache.insert(line.0.clone(), line.1.clone());
            } else {
                *self.needs_render.get_mut(line.0).unwrap() = true;
                self.cache.get_mut(line.0).unwrap().merge(line.1);
            }
        }
    }
}



#[derive(Resource, Serialize, Deserialize)]
pub struct Console {
    pub console_msgs: VecDeque<String>,
    pub max_console_msgs: usize,
}

impl Console {
    pub fn new(n_logs: usize) -> Self {
        Console {
            console_msgs: VecDeque::new(),
            max_console_msgs: n_logs,
        }
    }

    pub fn log(&mut self, msg: String) {
        self.console_msgs.push_front(msg);
        if self.console_msgs.len() > self.max_console_msgs {
            self.console_msgs.pop_back();
        }
    }

    pub fn console_ui(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for text in &self.console_msgs {
                ui.label(text);
            }
        });
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


