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

/// View plot by model
#[derive(Resource, Serialize, Deserialize, Default)]
pub struct PlotViewerV1 {
    viewer: BasicViewer,
    model: Models,
}

impl PlotViewerV1 {
    pub fn ui(&mut self, ui: &mut egui::Ui, lines: &ModelPlots) {
        // adjust local rendering parameters
        ui.horizontal(|ui| {
            ui.label("rendered resolution: (x, y)");
            ui.add(egui::DragValue::new(&mut self.viewer.res.0));
            self.viewer.res.0 = self.viewer.res.0.max(64).min(2048);
            ui.add(egui::DragValue::new(&mut self.viewer.res.1));
            self.viewer.res.0 = self.viewer.res.0.max(64).min(2048);
            ui.label("smooth");
            ui.add(egui::DragValue::new(&mut self.viewer.smooth));
            self.viewer.smooth = self.viewer.smooth.max(1);
            ui.label("local scale");
            ui.add(egui::Slider::new(&mut self.viewer.scale, 0.0..=1.0));
        });

        egui::ComboBox::from_label("filter by model")
            .selected_text(format!("{}", self.model))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.model, Models::BASELINE, "baseline");
            });

        if let Err(e) = self.render_update(lines, ui) {
            ui.label(format!("unable to render update {}", e));
        }
        
        let scale = self.viewer.scale;
        self.viewer.ui(ui, scale);
    }

    pub fn render_update(&mut self, lines: &ModelPlots, ui: &mut egui::Ui) -> Result<()> {
        let bufs = self.render(lines)?;
        for (gid, buf) in bufs {
            let texture = to_texture(&buf, self.viewer.res, ui)?;
            self.viewer.update(&gid, texture);
        }
        Ok(())
    }

    pub fn render(&mut self, lines: &ModelPlots) -> Result<Vec<(GraphId, Vec<u8>)>> {
        let changed = lines.needs_render();
        // not everything changed needs to be re-rendered by this plot viewer, so filter for things
        // that do need to be rendered
        // render only if the viewer doesn't contain the id, or the line has changed
        let to_render: Vec<_> = changed 
            .filter(|(changed, id)| id.model == self.model && self.viewer.should_render(id, *changed)).collect();
        
        // update line to prevent re-rendering next time
        to_render.iter().for_each(|(line, id)| { self.viewer.update_line_info(id, *line); });
        
        // for any lines which needs to be recomputed, store them in recomputed lines
        // after computation is done, join the two 
        let mut recomputed_lines = Vec::new();
        let mut recomputed_idx = Vec::new();
        let mut new_to_render = Vec::new();
        for (i, (_, id)) in to_render.iter().enumerate() {
            // if needs to recompute the line
            if self.viewer.need_recompute() {
                let pline = lines.get(id).unwrap().clone();
                recomputed_lines.push(self.viewer.compute_line(pline));
                recomputed_idx.push((*id, i));
            } else {
                new_to_render.push((*id, lines.get(id).unwrap()));
            }
        }
        // add the recomputed lines to the new_render
        for i in recomputed_idx.iter() {
            new_to_render.push((i.0, &recomputed_lines[i.1]));
        }

        // batch together plotids that have the same title, x_title, and y_title
        let batch_by_title = batch(new_to_render.iter(), |(id1, _), (id2, _)| {
            id1.title == id2.title && id1.x_title == id2.x_title && id1.y_title == id2.y_title
        });


        let res = self.viewer.res;
        // convert to GraphId, and texture
        let rendered_textures = batch_by_title.iter().map(|x| {
            let first = x.get(0).unwrap().0; // batch guarantees that each sub-vec is non-empty
            let title: &str = &first.title;
            let x_title: Option<&str> = Some(&first.x_title);
            let y_title: Option<&str> = Some(&first.y_title);

            // convert to expected format of the rendering function, just some coercing
            let lines = x.iter().map(|(pid, pline)| {
                let name: &str = &pid.run_name;
                (name, pline.as_slice())
            });

            let texture = render(
                title, x_title, y_title, lines, res
            );
            let gid: GraphId = first.into();
            (gid, texture)
        });
        let mut out = Vec::new();
        for (gid, buf) in rendered_textures {
            out.push((gid, buf?));
        }
        
        Ok(out)
    }
}

#[derive(Serialize, Deserialize)]
struct BasicViewer {
    #[serde(skip)]
    viewer: ImageGrid<GraphId>,
    #[serde(skip)]
    line_info: HashMap<PlotId, LineStats>,
    smooth: usize,
    res: (usize, usize),
    scale: f32,
}

impl Default for BasicViewer {
    fn default() -> Self {
        Self { smooth: 1, res: (1024, 768), scale: 1.0, viewer: ImageGrid::default(), line_info: HashMap::default() }
    }
}

impl BasicViewer {
    fn compute_line(&self, mut line: PlotLine) -> PlotLine {
        line.avg_smooth(self.smooth);
        line
    }
    fn need_recompute(&self) -> bool {
        self.smooth > 1
    }

    fn update_line_info(&mut self, id: &PlotId, line_stats: LineStats) {
        if self.line_info.contains_key(id) {
            *self.line_info.get_mut(id).unwrap() = line_stats;
        } else {
            self.line_info.insert(id.clone(), line_stats);
        }
    }
    fn should_render(&self, id: &PlotId, line_stats: LineStats) -> bool {
        self.viewer.contains(&id.into()).is_none() || !self.line_info.get(id).unwrap().dirty_eq(&line_stats)
    }

    fn update(&mut self, id: &GraphId, texture: egui::TextureHandle) {
        self.viewer.update(&id, texture);
    }

}

impl Deref for BasicViewer {
    type Target = ImageGrid<GraphId>;
    fn deref(&self) -> &Self::Target {
        &self.viewer
    }
}

impl DerefMut for BasicViewer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.viewer
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

    /// (is changed, the associated PlotId)
    fn needs_render(&self) -> impl Iterator<Item = (LineStats, &PlotId)> + Clone {
        let results = self.lines.iter().map(|x| (x.1.stats(), x.0));
        results
    }
}

#[derive(Default)]
pub struct ImageGrid<Id> {
    images: Vec<(Id, egui::TextureHandle)>,
}

impl<Id: Clone + Eq> ImageGrid<Id> {
    pub fn contains(&self, id: &Id) -> Option<usize> { 
        for (i, v) in self.images.iter().enumerate() {
            if v.0 == *id {
                return Some(i)
            }
        }
        None
    }

    pub fn clear(&mut self) { self.images.clear(); }

    /// updates the texture when a PlotId matches, else push the pair to the end, cloning PlotId
    pub fn update(&mut self, id: &Id, texture: egui::TextureHandle) {
        if let Some(j) = self.contains(id) {
            self.images[j].1 = texture;
        } else {
            self.images.push((id.clone(), texture));
        }
    }

    fn display_single_image(&mut self, ui: &mut egui::Ui, i: usize, alt_scale: f32) {
        let size = self.images[i].1.size_vec2() * alt_scale;
        ui.group(|ui| {
            ui.vertical(|ui| {
                let mut order = i;
                ui.add(egui::DragValue::new(&mut order));
                order = order.min(self.images.len() - 1);
                // if order is changed, shift everything between the new order and old order by 1
                if order != i {
                    let item = self.images.remove(i);
                    self.images.insert(order, item);
                }
                ui.image(&self.images[i].1, size);
            });
        });
    }

    /// Shows a grid of images with option to order them
    pub fn ui(&mut self, ui: &mut egui::Ui, scale: f32) {
        let width = ui.available_width();
        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut i = 0;
            while i < self.images.len() {
                let size = self.images[i].1.size_vec2();
                // if the image does not fit within the current width, scale it down
                if size.x > width {
                    let fitting_scale = width / size.x;
                    self.display_single_image(ui, i, fitting_scale);
                }
                // otherwise, fit as many images in the current width as possible
                ui.horizontal(|ui| {
                    let mut xi = width;
                    while xi > 0.0 && i < self.images.len() {
                        self.display_single_image(ui, i, scale);             
                        xi -= size.x;
                        i += 1;
                    }
                });
            }
        });
    }
}


// batches together into one vec if they are equal under eq, each of the sub-vectors are non-empty
// requires that eq be reflexive, that is, if a != b and b == c => a != c
pub fn batch<'a, T>(items: impl Iterator<Item = &'a T>, eq: impl Fn(&'a T, &'a T) -> bool) -> Vec<Vec<&'a T>> {
    let mut set: Vec<Vec<&T>> = Vec::new();
    
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
    render.render(&plots).expect("failed to render plots");
}