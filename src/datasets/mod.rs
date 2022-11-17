use std::fs;
use std::path;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use bevy::prelude::*;
use bevy::app::AppExit;
use bevy_egui::{egui, EguiContext};
use ndarray::prelude::*;
use serde::{Serialize, Deserialize};
use strum::{IntoEnumIterator, EnumIter};

pub mod mnist;
pub mod transforms;

pub trait Config {
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str);
}

/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset: Sync + Send {
    type DataPoint;
    fn next(&mut self) -> Self::DataPoint;
    fn shuffle(&mut self);
}

/// Each Dataset has two structs, that of the parameters it holds
/// and the data that it holds, this trait is implemented on the parameters for the dataset
/// this separation is made as the parameters are usually light and copyible, while
/// the data is not light, and require some non-negliable compute for the setup.
/// This trait adjusts the parameters, and builds the dataset on the parameters it holds.
pub trait DatasetUI: Sync + Send {
    fn ui(&mut self, ui: &mut egui::Ui);
    fn build(&self) -> DatasetTypes;
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str);
}

/// This is the unification of possible Datasets behaviors, constructed from DatasetUI, or
/// some other parameter-adjusting setup trait.
pub enum DatasetTypes {
    Classification(Box<dyn Dataset<DataPoint = ImClassifyDataPoint>>),
    // Detection(Box<dyn ImDetection>),
    // Generation(Box<dyn ImGeneration),
}

/// The unification of every possible Dataset supported in a single type.
#[derive(Clone, Copy, Debug, EnumIter, PartialEq, Eq, Hash)]
pub enum PossibleDatasets {
    MNIST
}

impl PossibleDatasets {
    pub fn name(&self) -> &str {
        match self {
            Self::MNIST => "mnist"
        }
    }
    fn user_config_path(&self) -> &str {
        match self {
            Self::MNIST => "assets/user_configs/datasets/mnist.ron"
        }
    }
    fn default_config_path(&self) -> &str {
        match self {
            Self::MNIST => "assets/configs/datasets/mnist.ron"
        }
    }
    pub fn get_param(&self) -> Box<dyn DatasetUI> {
        match self {
            Self::MNIST => Box::new(mnist::MnistParams::default())
        }
    }
}

/// The Data types that the datasets output and transforms input.
pub struct ImageDataPoint {
    pub image: Array4<f32>
}

/// The data point associated with the image detection task, this is the type which
/// gets fed into the model
pub struct ImClassifyDataPoint {
    pub image: ImageDataPoint,
    pub label: Vec<u32>
}

// pub struct ObjDetectionDataPoint;
// pub struct TextGenerationDataPoint;
// pub struct ImageModelingDataPoint;


///////////////////////////////////////////////////////////////////////////////////
/// Integrating and setting up the relevant setup with bevy.
pub struct DatasetPlugin;
impl Plugin for DatasetPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_startup_system(setup_datasets)
            .add_system(save_dataset_params);

    }
}

/// A hash map containing serialized versions of dataset parameters on app startup
/// so that upon app shutdown, if the serialized string of the possibly altered
/// dataset parameters are different from startup, then save the new params into a
/// a new file, for future persistence
struct DatasetConfigs(HashMap<PossibleDatasets, String>);

pub struct DatasetParams(HashMap<PossibleDatasets, Box<dyn DatasetUI>>);

/// Bevy startup system for setting up all dataset parameters
fn setup_datasets(mut commands: Commands) {
    let d_enum: Vec<_> = PossibleDatasets::iter().collect();
    let mut serial_params = HashMap::<PossibleDatasets, String>::new();
    let mut data_params = HashMap::<PossibleDatasets, Box<dyn DatasetUI>>::new();

    for s in d_enum {
        let serialized: String;
        if path::Path::new(s.user_config_path()).exists() {
            serialized = fs::read_to_string(s.user_config_path()).unwrap();
        } else {
            serialized = fs::read_to_string(s.default_config_path()).unwrap();
        }
        serial_params.insert(s, serialized.clone());
        let mut param = s.get_param();
        param.load_config(&serialized);
        data_params.insert(s, param);
    }
    
    let configs = DatasetConfigs(serial_params);
    let params = DatasetParams(data_params);

    commands.insert_resource(params);
    commands.insert_resource(configs);
}

/// Bevy shutdown system for saving changes to dataset parameters
/// TODO: Save every n seconds? 
fn save_dataset_params(
    mut exit: EventReader<AppExit>,
    config: Res<DatasetConfigs>,
    params: Res<DatasetParams>
) {
    let mut exited = false;
    for _ in exit.iter() {
        exited = true;
    }
    if exited {
        for (dataset, param) in params.0.iter() {
            if config.0.contains_key(dataset) && config.0[dataset] != param.config() {
                let dir = path::Path::new(dataset.user_config_path()).parent().unwrap();
                if !dir.exists() {
                    fs::create_dir_all(dir).unwrap();
                }
                fs::write(dataset.user_config_path(), param.config()).unwrap();
            }
        }  
    }
}

pub fn concat_im_size_eq(imgs: &[&Array3<f32>]) -> ImageDataPoint {
    let whc = imgs[0].dim();
    let b = imgs.len();
    let mut img = Array4::<f32>::zeros((b, whc.0, whc.1, whc.2));
    for i in 0..b {
        let mut smut = img.slice_mut(s![i, .., .., ..]);
        smut.zip_mut_with(imgs[i], |a, b| {
            *a = *b;
        });
    }
    ImageDataPoint { image: img }
}

#[test]
fn serialize_test() {
    let param: mnist::MnistParams = ron::from_str(&fs::read_to_string("assets/configs/datasets/mnist.ron").unwrap()).unwrap();
    println!("{:?}", param);
    let str_repr = ron::to_string(&param).unwrap();
    println!("{}", str_repr);
}

#[test]
fn write_test() {
    let test_file = "test_folder/test.ron";
    let msg = "hello";
    let dir = path::Path::new(test_file).parent().unwrap();
    if !dir.exists() {
        fs::create_dir_all(dir).unwrap();
    }
    fs::write(test_file, msg).unwrap();
}