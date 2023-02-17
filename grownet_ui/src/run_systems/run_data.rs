use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, Range};
use bevy::utils::HashSet;
use crossbeam::channel::{Sender, Receiver};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::egui;
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


/// ModelPlots contains the various plots of the model
#[derive(Resource, Default, Serialize, Deserialize)]
pub struct ModelPlots {
    graphs: HashMap<(String, String, String), PlotGraph>, // (title, xtitle, ytitle)
}

impl ModelPlots {
    /// Each plot has a title, and under each title, multiple lines are graphed, by run
    /// Inserts a new plot with title if there is none
    /// Appends onto existing run if name and title are pre-existing, else creates a new run
    pub fn add_plot(&mut self, point: PlotPoint, info: &RunInfo) {
        let prod = (point.title.to_string(), point.x_title.to_string(), point.y_title.to_string());
        if !self.graphs.contains_key(&prod) {
            self.graphs.insert(prod.clone(), PlotGraph { 
                title: point.title.into(), 
                x_title: point.x_title.into(), 
                y_title: point.y_title.into(), 
                plots: HashMap::new() 
            });
        }

        let graph = self.graphs.get_mut(&prod).unwrap();
        let name = info.run_name();
        if !graph.plots.contains_key(&name) {
            graph.plots.insert(name.clone(), Vec::new());
        }
        graph.plots.get_mut(&name).unwrap().push((point.x, point.y));
    }
}

pub type Line = Vec<(f64, f64)>;

/// Each PlotGraph signifies a graph which contains multiple lines, from multiple runs
#[derive(Deserialize, Serialize, Default)]
pub struct PlotGraph {
    title: String,
    x_title: String,
    y_title: String,
    plots: HashMap<String, Vec<(f64, f64)>>,
}

impl PlotGraph {
    pub fn pack<'a>(&'a self, mut f: impl FnMut(&str) -> bool) -> RenderPack<'a> {
        RenderPack { 
            title: &self.title, 
            x_title: &self.x_title, 
            y_title: &self.y_title, 
            lines: self.plots.iter()
                .filter(|x| f(&x.0))
                .map(|x| (&*(*x.0), &x.1[..]))
                .collect()
        }
    }
}


pub struct RenderPack<'a> {
    title: &'a str,
    x_title: &'a str,
    y_title: &'a str,
    lines: Vec<(&'a str, &'a [(f64, f64)])>,
}

pub struct OwnedRenderPack {
    title: String,
    x_title: String,
    y_title: String,
    lines: HashMap<String, Vec<(f64, f64)>>,
}

impl OwnedRenderPack {
    pub fn borrow(&self) -> RenderPack<'_> {
        RenderPack { 
            title: &self.title, 
            x_title: &self.x_title, 
            y_title: &self.y_title, 
            lines: self.lines.iter().map(|x| { (&**x.0, &x.1[..]) }).collect() 
        }
    }

    pub fn flush(&mut self) {
        self.lines.clear();
    }

    pub fn update<'a>(&mut self, lines: &[(&'a str, &'a [(f64, f64)])]) -> bool {
        let mut changed = false;
        for (n, l) in lines {
            if !self.lines.contains_key(*n) {
                self.lines.insert(n.to_string(), l.to_vec());
                changed = true;
            } else {
                let line = self.lines.get_mut(*n).unwrap();
                if line != l {
                    line.clear();
                    line.extend_from_slice(l);
                    changed = true;
                }
            }
        }
        changed
    }
}

impl<'a> From<RenderPack<'a>> for OwnedRenderPack {
    fn from(x: RenderPack<'a>) -> Self {
        let mut map = HashMap::new();
        x.lines.iter().for_each(|x| {
            map.insert(x.0.to_string(), x.1.to_vec());
        });
        OwnedRenderPack { 
            title: x.title.into(), 
            x_title: x.x_title.into(), 
            y_title: x.y_title.into(), 
            lines: map }
    }
}


pub fn wider_range(a: Range<f64>, b: Range<f64>) -> Range<f64> {
    a.start.min(b.start)..a.end.max(b.end)
}

pub fn compute_bounds<'a>(lines: impl Iterator<Item = &'a [(f64, f64)]>) -> (Range<f64>, Range<f64>) {
    const INIT_FOLD: (Range<f64>, Range<f64>) = (0.0..1.0, 0.0..1.0);
    lines.map(|x| {
        x.iter().fold(INIT_FOLD, |acc, val| {
            (wider_range(acc.0, val.0..val.0), wider_range(acc.1, val.1..val.1))
        })
    }).fold(INIT_FOLD, |acc, val| {
        (wider_range(acc.0, val.0), wider_range(acc.1, val.1))
    })
}

fn render<'a>(
    title: &str,
    x_title: &str,
    y_title: &str,
    lines: &[(&str, &[(f64, f64)])],
    res: (usize, usize)
) -> Result<Vec<u8>> {
    let mut buf = vec![255; res.0 * res.1 * 3]; // rgb format

    {
        let root = BitMapBackend::with_buffer(&mut buf, (res.0 as u32, res.1 as u32));
    
        let bounds = compute_bounds(lines.iter().map(|x| x.1));
        
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
        
        chart
            .configure_mesh()
            .x_desc(x_title)
            .y_desc(y_title)
            .draw()?;

        for (idx, (name, line)) in lines.iter().enumerate() {
            let color = Palette99::pick(idx).mix(0.9);
            chart
                .draw_series(LineSeries::new(
                    line.iter().map(|x| (x.0, x.1))
                , color.stroke_width(3)))?
                .label(*name)
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
    order: Vec<String>,
    #[serde(skip)]
    charts: HashMap<String, RenderedChart>,
    #[serde(skip)]
    needs_render: HashMap<String, bool>,
    #[serde(skip)]
    cache: HashMap<String, OwnedRenderPack>,
}

impl PlotViewer {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("rendered resolution: (x, y)");
            let mut needs_change = ui.add(egui::DragValue::new(&mut self.res.0)).changed();
            self.res.0 = self.res.0.max(64).min(2048);
            needs_change |= ui.add(egui::DragValue::new(&mut self.res.1)).changed();
            self.res.0 = self.res.0.max(64).min(2048);
            ui.label("smooth");
            ui.add(egui::DragValue::new(&mut self.smooth));
            self.smooth = self.smooth.max(1);
            ui.label("local scale");
            ui.add(egui::DragValue::new(&mut self.scale));
            if needs_change {
                self.mark_change_all();
            }
        });


        let size = ui.max_rect();
        let rep_x = ((size.width() / self.res.0 as f32) * self.scale) as usize;
        let rep_y = self.charts.len() / rep_x;
        egui::ScrollArea::vertical().show(ui, |ui| {
            let _ = self.render(ui).map_err(|x| {
                ui.label(format!("plot rendering error: {}", x));
            });

            for i in 0..rep_y {
                ui.horizontal(|ui| {
                    for j in 0..rep_x {     
                        let idx = i * rep_y + j;
                        let name = &self.order[idx];
                        self.charts.get_mut(name).unwrap().show(ui, self.scale);
                    }
                });
            }
            // remainder
            ui.horizontal(|ui| {
                for j in rep_x * rep_y..self.charts.len() {
                    self.charts.get_mut(&self.order[j]).unwrap().show(ui, self.scale);
                }
            });
        });
    }

    pub fn mark_change_all(&mut self) {
        self.needs_render.iter_mut().for_each(|x| { *x.1 = true; })
    }

    pub fn render(&mut self, ui: &mut egui::Ui) -> Result<()> {
        for (k, v) in self.needs_render.iter_mut() {
            if *v {
                self.charts.get_mut(k).unwrap().adjust(self.cache.get(k).unwrap().borrow(), ui, self.smooth, self.res)?;
                *v = false;
            }
        }
        Ok(())
    }

    pub fn update_cache<'a>(&mut self, packs: impl Iterator<Item = RenderPack<'a>>) {
        for pack in packs {
            if !self.charts.contains_key(pack.title) {
                self.needs_render.insert(pack.title.into(), true);
                self.cache.insert(pack.title.into(), pack.into());
            } else {
                *self.needs_render.get_mut(pack.title).unwrap() |= self.cache.get_mut(pack.title).unwrap().update(&pack.lines);
            }
        }
    }

    /// warning, does not flush titles from cache
    pub fn flush(&mut self) {
        self.cache.iter_mut().for_each(|x| x.1.flush());
        self.needs_render.clear();
    }
}

pub struct RenderedChart {
    texture: egui::TextureHandle,
    
}

impl RenderedChart {
    pub fn new<'a>(pack: RenderPack<'a>, res: (usize, usize), ui: &mut egui::Ui) -> Result<Self> {
        let render = render(pack.title, pack.x_title, pack.y_title, &pack.lines, res)?;
        let handle = to_texture(&render, res, ui)?;

        Ok(Self {
            texture: handle,
        })
    }

    pub fn adjust<'a>(&mut self, pack: RenderPack<'a>, ui: &mut egui::Ui, smooth: usize, res: (usize, usize)) -> Result<()> {
        let buf = if smooth == 1 {
            render(pack.title, pack.x_title, pack.y_title, &pack.lines, res)?
        } else {
            let new_lines: Vec<_> = pack.lines.iter().map(|x| smooth_window(x.1, smooth)).collect();
            let lines: Vec<_> = pack.lines.iter().zip(new_lines.iter()).map(|x| { (x.0.0, &x.1[..]) }).collect();
            render(pack.title, pack.x_title, pack.y_title, &lines, res)?
        };
        let handle = to_texture(&buf, res, ui)?;
        self.texture = handle;
        Ok(())
    }

    pub fn show(&mut self, ui: &mut egui::Ui, scale: f32) {
        let mut size = self.texture.size_vec2();
        size *= scale;
        ui.image(&self.texture, size);
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

#[test]
fn plotters() {
    let a = vec![1, 2, 3];
    let b = vec![1, 2, 4];

    println!("{}", a == &b[..]);
}