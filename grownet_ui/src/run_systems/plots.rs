use std::ops::{Range, Deref, DerefMut};
use std::collections::{HashMap, VecDeque, HashSet};

use anyhow::{Error, Result};
use itertools::Itertools;
use num::complex::ComplexFloat;
use plotters::prelude::*;
use bevy::prelude::*;
use bevy_egui::egui;
use serde::{Serialize, Deserialize};

use super::run_data::Models;
use model_lib::models::PlotPoint;


// Five step rendering pipeline
// 1. figure out the entire set of lines to render, as in, what lines should the current plot show
//      based off of models, run_name, dataset, etc.
// 2. Batch together lines so that each batch represents a single graph/plot, under which multiple lines could fall under
// 3. Figure out if the batch needs to be rendered after all
// 4. Recompute lines based off of metrics if necessary, and render the line to a raw buffer (Vec<u8>)
// 5. Update the plot cache


/// Use line stats to quickly compare if two lines are equal, and lightly cache resulting statistics
#[derive(Clone, Copy)]
struct LineStats {
    len: usize,
    last_x: Option<(f64, f64)>
}

impl LineStats {
    const EPS: f64 = 1e-6;
    /// since lines are monotonic in x, we can say that two lines are equal if their lengths
    /// are equal, and the end points are approximately equal
    fn dirty_eq(&self, other: &LineStats) -> bool {
        self.len == other.len && {
            if self.last_x.is_none() {
                true
            } else {
                let (x, y) = self.last_x.unwrap();
                let (x1, y1) = self.last_x.unwrap();
                (x - x1).abs() < Self::EPS && (y - y1).abs() < Self::EPS
            }
        }
    }
}

/// a line in (x, y), where x is guaranteed to be monotonic (strictly increasing)
#[derive(Serialize, Deserialize, Deref, DerefMut, Clone, Default, Debug)]
pub struct PlotLine(Vec<(f64, f64)>);

impl PlotLine {
    fn stats(&self) -> LineStats {
        LineStats { len: self.len(), last_x: self.last().and_then(|x| Some(*x)) }
    }

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
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Serialize, Deserialize, Default, Debug)]
pub struct PlotId {
    pub model: Models,
    pub run_name: String,
    pub title: String,
    pub x_title: String,
    pub y_title: String,
}

/// Uniquely identifies a plot
#[derive(PartialEq, Eq, Clone, Default)]
pub struct GraphId(String, String, String); // title, x-title, y-title

impl From<&PlotId> for GraphId {
    fn from(x: &PlotId) -> Self {
        GraphId((&x.title).into(), (&x.x_title).into(), (&x.y_title).into())
    }
}

#[derive(Resource, Serialize, Deserialize, Default)]
pub struct PlotViewerV1 {
    filter: BasicRenderFilter,
    v_cache: ViewCache,
    params: ComputeRender,
    p_cache: PlotCache,
    // some ui configuration parameters
    local_scale: f32
}

impl PlotViewerV1 {
    pub fn ui(&mut self, ui: &mut egui::Ui, lines: &ModelPlots) {
        // adjust local rendering parameters, filters, etc.
        ui.horizontal(|ui| {
            ui.label("rendered resolution: (x, y)");
            ui.add(egui::DragValue::new(&mut self.params.res.0));
            self.params.res.0 = self.params.res.0.max(64).min(2048);
            ui.add(egui::DragValue::new(&mut self.params.res.1));
            self.params.res.0 = self.params.res.0.max(64).min(2048);
            ui.label("smooth");
            ui.add(egui::DragValue::new(&mut self.params.smooth));
            self.params.smooth = self.params.smooth.max(1);
            ui.label("local scale");
            ui.add(egui::Slider::new(&mut self.local_scale, 0.0..=1.0));
        });

        egui::ComboBox::from_label("filter by model")
            .selected_text(format!("{}", self.filter.model))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.filter.model, Models::BASELINE, "baseline");
            });

        if let Err(e) = self.compute_whole(ui, lines) {
            ui.label(format!("rendering error {}", e));
        }

        // now show the images, after rendering, make it so that the ui portion is local, thus allowing greater
        // flexibility
        let width = ui.available_width();
        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut i = 0;
            while i < self.p_cache.images.len() {
                let size = self.p_cache.images[i].1.size_vec2();
                // if the image does not fit within the current width, scale it down
                if size.x > width {
                    let fitting_scale = width / size.x;
                    self.display_single_image(ui, i, fitting_scale);
                }
                // otherwise, fit as many images in the current width as possible
                ui.horizontal(|ui| {
                    let mut xi = width;
                    while xi > 0.0 && i < self.p_cache.images.len() {
                        self.display_single_image(ui, i, self.local_scale);             
                        xi -= size.x;
                        i += 1;
                    }
                });
            }
        });

    }

    fn display_single_image(&mut self, ui: &mut egui::Ui, i: usize, alt_scale: f32) {
        let size = self.p_cache.images[i].1.size_vec2() * alt_scale;
        ui.group(|ui| {
            ui.vertical(|ui| {
                let mut order = i;
                ui.add(egui::DragValue::new(&mut order));
                order = order.min(self.p_cache.images.len() - 1);
                // if order is changed, shift everything between the new order and old order by 1
                if order != i {
                    let item = self.p_cache.images.remove(i);
                    self.p_cache.images.insert(order, item);
                }
                ui.image(&self.p_cache.images[i].1, size);
            });
        });
    }

    fn compute_whole(&mut self, ui: &mut egui::Ui, lines: &ModelPlots) -> Result<()> {
        let image_bufs = self.compute(lines)?;
        // step 5
        let textures: Result<Vec<_>> = image_bufs.into_iter().map(|x| { x.to_texture(ui) }).collect();
        self.p_cache.update_cache(textures?.into_iter());
        Ok(())
    }

    // split this for testing purposes
    fn compute(&mut self, lines: &ModelPlots) -> Result<Vec<RenderedBatch>> {
        let every_line = lines.lines.iter();
        // step 1
        let pre_filter = self.filter.filter(every_line);
        // step 2
        let graphs = PlotBatch::batch_by_title(pre_filter);
        // step 3
        let need_render = graphs.into_iter().filter(|x| self.v_cache.needs_render(x));
        // step 4
        let image_bufs = self.params.render(need_render)?;
        Ok(image_bufs)
    }
}


use egui::plot;

fn get_or_insert<'a, K: std::hash::Hash + Eq + Clone, T>(map: &'a mut HashMap<K, T>, key: &K, default: impl Fn() -> T) -> &'a mut T {
    if !map.contains_key(key) {
        map.insert(key.clone(), default());
    }
    map.get_mut(key).unwrap()
}

fn contains<T>(vec: &Vec<T>, mut f: impl FnMut(&T) -> bool) -> Option<usize> {
    for (i, x) in vec.iter().enumerate() {
        if f(x) {
            return Some(i);
        }
    }
    None
}

fn get_run_color(run_name: &str) -> (u8, u8, u8) {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(run_name.as_bytes());
    let color_id = hasher.finish();
    let color = Palette99::pick(color_id as usize).mix(0.9);
    color.rgb()
}

struct SmoothIter<It> {
    window_size: usize,
    window: VecDeque<(f64, f64)>,
    sum: f64,
    iter: It
}

impl<It> SmoothIter<It> {
    fn new(iter: It, window: usize) -> Self {
        let mut sum_window = VecDeque::new();
        sum_window.reserve_exact(window);
        Self { window_size: window, window: sum_window, sum: 0.0, iter }
    }
}

impl<It: Iterator<Item = (f64, f64)>> Iterator for SmoothIter<It> {
    type Item = (f64, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(point) = self.iter.next() {
            if self.window.len() < self.window_size {
                self.window.push_back(point);
                self.sum += point.1;
            } else {
                let prev_point = self.window.pop_front().unwrap();
                self.sum -= prev_point.1;
                self.sum += point.1;
                self.window.push_back(point);
            }
            Some((point.0, self.sum / self.window.len() as f64))
        } else {
            None
        }
    }
}

#[derive(Resource, Serialize, Deserialize)]
pub struct PlotViewerV2 {
    display_model: Models,
    display_runs: HashMap<Models, Vec<((u8, u8, u8), String, bool)>>, // (line color, run_names, display)
    display_titles: HashMap<Models, Vec<(String, bool)>>, // (title_names, display)
    // some ui configuration parameters
    graphs_per_row: usize,
    smooth_window: usize,
}

impl PlotViewerV2 {
    pub fn ui(&mut self, ui: &mut egui::Ui, lines: &ModelPlots) {
        // adjust local rendering parameters, filters, etc.
        // ui to adjust which lines to show
        let cur_display_titles = get_or_insert(&mut self.display_titles, &self.display_model, || Vec::new());
        let cur_display_runs = get_or_insert(&mut self.display_runs, &self.display_model, || Vec::new());
        ui.vertical(|ui| {
            egui::ComboBox::from_id_source("filter by model")
                .selected_text(format!("{}", self.display_model))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.display_model, Models::BASELINE, "baseline");
                });
                                // pick which titles to show
            ui.collapsing("graphs", |ui| {
                for (title_name, display) in cur_display_titles.iter_mut() {
                    ui.checkbox(display, &*title_name);
                }
            });

            // pick which runs to show
            for (color, run_name, display) in cur_display_runs.iter_mut() {
                ui.horizontal(|ui| {
                    let color = if *display {
                        *color
                    } else {
                        (5, 10, 10)
                    };
                    if ui.add(egui::Button::new("  ").fill(egui::Color32::from_rgb(color.0, color.1, color.2))).clicked() {
                        *display = !*display;
                    }
                    ui.label(&*run_name);
                });
            }
            ui.label("graphs per row");
            ui.add(egui::Slider::new(&mut self.graphs_per_row, 1..=5));
            ui.label("smooth window");
            ui.add(egui::DragValue::new(&mut self.smooth_window));
            self.smooth_window = self.smooth_window.max(1);
        });

        // now actually show the lines
        let all_lines = lines.lines.iter().filter(|(id, _)| id.model == self.display_model);
        let mut to_plot = Vec::new();
        for (pid, line) in all_lines {
            let title_idx = contains(cur_display_titles, |(title, _)| pid.title.eq(title));
            let run_idx = contains(cur_display_runs, |(_, run_name, _)| pid.run_name.eq(run_name));
            if title_idx.is_none() {
                cur_display_titles.insert(0, (pid.title.clone(), true));
            }
            if run_idx.is_none() {
                let run_color = get_run_color(&pid.run_name);
                cur_display_runs.insert(0, (run_color, pid.run_name.clone(), true));
            }
            if title_idx.is_none() || run_idx.is_none() {
                to_plot.push((pid, line));
            } else if cur_display_titles[title_idx.unwrap()].1 && cur_display_runs[run_idx.unwrap()].2 {
                to_plot.push((pid, line));
            }
        }

        ui.ctx().request_repaint();
        let batch_by_title = PlotBatch::batch_by_title(to_plot.into_iter());
        let available_width = ui.available_width();
        self.graphs_per_row = self.graphs_per_row.min(batch_by_title.len()).max(1);
        let graph_width = available_width / self.graphs_per_row as f32;
        let num_cols = (batch_by_title.len() + self.graphs_per_row - 1) / self.graphs_per_row;
        // println!("{}", num_cols);
        ui.vertical(|ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for col in 0..num_cols {
                    ui.horizontal(|ui| {
                        // println!("{:?}", col*graphs_per_row..((col + 1) * graphs_per_row).min(batch_by_title.len()));
                        for i in col*self.graphs_per_row..((col + 1) * self.graphs_per_row).min(batch_by_title.len()) {
                            ui.vertical(|ui| {
                                let graph = &batch_by_title[i];
                                ui.label(&graph.gid.0);
                                let plot = plot::Plot::new(&graph.gid.0)
                                    .auto_bounds_x().auto_bounds_y()
                                    .allow_scroll(false)
                                    .allow_drag(false)
                                    .view_aspect(1.5)
                                    .width(graph_width);
                                plot.show(ui, |plot_ui| {
                                    for (pid, line) in &graph.plots {
                                        let color = get_run_color(&pid.run_name);
                                        let smoothed_line = SmoothIter::new(
                                            line.iter().map(|point| *point), self.smooth_window)
                                            .map(|point| [point.0, point.1]);
                                        let line = plot::Line::new(plot::PlotPoints::from_iter(smoothed_line))
                                            .color(egui::Color32::from_rgb(color.0, color.1, color.2))
                                            .style(plot::LineStyle::Solid);
                                        plot_ui.line(line);
                                    }
                                });
                                ui.shrink_width_to_current();
                                ui.separator();
                            });
                        }
                    });
                }
            });
        });
            
        
    }

}

impl Default for PlotViewerV2 {
    fn default() -> Self {
        Self { 
            display_model: Models::BASELINE, 
            display_runs: HashMap::new(),
            display_titles: HashMap::new(),
            graphs_per_row: 1,
            smooth_window: 1,
        }
    }
}

/// Step 1
#[derive(Resource, Serialize, Deserialize, Default)]
struct BasicRenderFilter {
    model: Models,
}

impl BasicRenderFilter {
    fn filter<'a>(&self, ids: impl Iterator<Item = (&'a PlotId, &'a PlotLine)>) -> impl Iterator<Item = (&'a PlotId, &'a PlotLine)> {
        let model = self.model;
        ids.filter(move |x| x.0.model == model)
    }
}

// Step 2
struct PlotBatch<'a> {
    gid: GraphId,
    plots: Vec<(&'a PlotId, &'a PlotLine)>
}

impl<'a> PlotBatch<'a> {
    fn batch_by_title(items: impl Iterator<Item = (&'a PlotId, &'a PlotLine)>) -> Vec<PlotBatch<'a>> {
        let batches = batch(items, |(id1, _), (id2, _)| {
            id1.title == id2.title && id1.x_title == id2.x_title && id1.y_title == id2.y_title
        });
        batches.into_iter().map(|x| {
            let first = x.get(0).unwrap().0;
            PlotBatch { gid: first.into(), plots: x }
        }).collect()
    }
}

/// Step 3
#[derive(Resource, Serialize, Deserialize, Default)]
struct ViewCache {
    #[serde(skip)]
    line_info: HashMap<PlotId, LineStats>,
}

impl ViewCache {
    fn needs_render<'a>(&self, plot_batch: &PlotBatch<'a>) -> bool {
        for (pid, line) in plot_batch.plots.iter() {
            if !self.line_info.contains_key(pid) || self.line_info.get(pid).unwrap().dirty_eq(&line.stats()) {
                return true;
            }
        }
        false
    }

    // update the line-infos so that the cache is up to date with the latest changes
    fn update<'a>(&mut self, to_render: impl Iterator<Item = (&'a PlotId, &'a PlotLine)>) {
        for (id, line) in to_render {
            if self.line_info.contains_key(id) {
                *self.line_info.get_mut(id).unwrap() = line.stats(); 
            } else {
                self.line_info.insert(id.clone(), line.stats());
            }
        }
    } 
}

/// Step 4
#[derive(Resource, Serialize, Deserialize)]
struct ComputeRender {
    smooth: usize,
    res: (usize, usize)
}

impl Default for ComputeRender {
    fn default() -> Self {
        ComputeRender { smooth: 1, res: (512, 348) }
    }
}

struct RenderedBatch {
    gid: GraphId,
    buf: Vec<u8>,
    res: (usize, usize)
}

impl RenderedBatch {
    fn to_texture(self, ui: &mut egui::Ui) -> Result<RenderedTexture> {
        let gid = self.gid;
        let texture = to_texture(&self.buf, self.res, ui)?;
        Ok(RenderedTexture { gid: gid, texture })
    }
}

impl ComputeRender {
    fn render<'a>(&self, items: impl Iterator<Item = PlotBatch<'a>>) -> Result<Vec<RenderedBatch>> {
        let should_recompute = self.smooth > 1;
        let mut rendered = Vec::new();
        for plot in items {
            let PlotBatch { gid, plots } = plot;
            if should_recompute {
                let new_lines: Vec<(&PlotId, PlotLine)> = plots.into_iter().map(|(id, plt)| {
                    let mut new_line = plt.clone();
                    new_line.avg_smooth(self.smooth);
                    (id, new_line)
                }).collect();
                let render_it = new_lines.iter().map(|(id, line)| {
                    let run_name: &str = &id.run_name;
                    (run_name, line.as_slice())
                });
                let buf = render(&gid.0, Some(&gid.1), Some(&gid.2), render_it, self.res)?;
                rendered.push(RenderedBatch { gid, buf, res: self.res });
            } else { // identical code because cannot new_lines in an inner block
                let render_it = plots.iter().map(|(id, line)| {
                    let run_name: &str = &id.run_name;
                    (run_name, line.as_slice())
                });
                let buf = render(&gid.0, Some(&gid.1), Some(&gid.2), render_it, self.res)?;
                rendered.push(RenderedBatch { gid, buf, res: self.res });
            }
        }

        Ok(rendered)
    }
}

struct RenderedTexture {
    gid: GraphId,
    texture: egui::TextureHandle
}


/// Step 5
#[derive(Resource, Serialize, Deserialize, Default)]
struct PlotCache {
    #[serde(skip)]
    images: Vec<(GraphId, egui::TextureHandle)>,
}

impl PlotCache {
    pub fn contains(&self, id: &GraphId) -> Option<usize> { 
        for (i, v) in self.images.iter().enumerate() {
            if v.0 == *id {
                return Some(i)
            }
        }
        None
    }

    fn update_cache(&mut self, items: impl Iterator<Item = RenderedTexture>) {
        for batch in items {
            if let Some(j) = self.contains(&batch.gid) {
                self.images[j].1 = batch.texture;
            } else {
                self.images.push((batch.gid.clone(), batch.texture));
            }
        }
    }
}


#[derive(Serialize, Deserialize, Resource, Default, Debug)]
pub struct ModelPlots {
    lines: HashMap<PlotId, PlotLine>,
}

impl ModelPlots {
    pub fn filter(&self, mut f: impl FnMut(&PlotId) -> bool) -> impl Iterator<Item = (&PlotId, &PlotLine)> {
        self.lines.iter().filter(move |x| f(&x.0))
    }

    pub fn filter_mut(&mut self, mut f: impl FnMut(&PlotId) -> bool) -> impl Iterator<Item = (&PlotId, &mut PlotLine)> {
        self.lines.iter_mut().filter(move |x| f(&x.0))
    }

    pub fn get(&self, id: &PlotId) -> Option<&PlotLine> {
        self.lines.get(id)
    }

    pub fn get_mut(&mut self, id: &PlotId) -> Option<&mut PlotLine> {
        self.lines.get_mut(id)
    }

    pub fn contains(&self, id: &PlotId) -> bool {
        self.lines.contains_key(id)
    }

    pub fn insert(&mut self, id: PlotId, line: PlotLine) {
        self.lines.insert(id, line);
    }

    pub fn add_point(&mut self, id: &PlotId, point: (f64, f64)) {
        if !self.lines.contains_key(id) { // if this plot id is not in self, since changed and lines have the same set of keys
            let mut new_line = PlotLine::default();
            new_line.add(point);
            self.insert(id.clone(), new_line);
        } else {
            self.get_mut(id).and_then(|x| Some(x.add(point)));
        }
    }
}


// batches together into one vec if they are equal under eq, each of the sub-vectors are non-empty
// requires that eq be reflexive, that is, if a != b and b == c => a != c
pub fn batch<T: Copy>(items: impl Iterator<Item = T>, eq: impl Fn(T, T) -> bool) -> Vec<Vec<T>> {
    let mut set: Vec<Vec<T>> = Vec::new();
    
    for i in items {
        let mut changed = false;
        for s in set.iter_mut() {
            if eq(i, s[0]) {
                s.push(i);
                changed = true;
                break;
            }
        }
        if !changed { // none of the collections equal i, create a new collection
            set.push(vec![i]);
        }
    }

    set
} 

fn get_line_color(pid: &PlotId) -> usize {
    todo!()
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

        for (_idx, (name, line)) in lines.enumerate() {
            // make it so that each run gets its own color
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            hasher.write(name.as_bytes());
            let color_id = hasher.finish();
            let color = Palette99::pick(color_id as usize).mix(0.9);
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

pub fn wider_range(a: Range<f64>, b: Range<f64>) -> Range<f64> {
    a.start.min(b.start)..a.end.max(b.end)
}

pub fn compute_bounds<'a>(lines: impl Iterator<Item = (f64, f64)>) -> (Range<f64>, Range<f64>) {
    const INIT_BOUND: (Range<f64>, Range<f64>) = (0.0..0.0, 0.0..0.0);
    lines.fold(INIT_BOUND, |acc, val| {
        (wider_range(acc.0, val.0..val.0), wider_range(acc.1, val.1..val.1))
    })
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

#[test]
fn test_render() {
    let mut plots = ModelPlots::default();
    let test_id = PlotId { model: Models::BASELINE, run_name: "baselinev1".into(), 
        title: "test loss".into(), x_title: "steps".into(), y_title: "loss".into()  };
    
    for i in 0..100 {
        plots.add_point(&test_id, (i as f64, (i as f64).sin()));
    }

    //println!("{:?}", plots);
    let mut render = PlotViewerV1::default();
    render.compute(&plots).expect("failed to render plots");
}