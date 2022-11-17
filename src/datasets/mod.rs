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

/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset: Sync + Send {
    type DataPoint;
    fn next(&mut self) -> Self::DataPoint;
}

/// Each Dataset has two structs, that of the parameters it holds
/// and the data that it holds, this trait is implemented on the parameters for the dataset
/// this separation is made as the parameters are usually light and copyible, while
/// the data is not light, and require some non-negliable compute for the setup.
/// This trait adjusts the parameters, and builds the dataset on the parameters it holds.
pub trait DatasetUI {
    fn ui(&mut self, ui: &mut egui::Ui);
    fn build(&self) -> DatasetTypes;
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
}


/// The Data types that the datasets output and transforms input.
pub struct ImageDataPoint {
    pub image: Array4<f32>
}

/// The data point associated with the image detection task, this is the type which
/// gets fed into the model
pub struct ImClassifyDataPoint {
    pub image: Array4<f32>,
    pub label: Vec<u16>
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

/// Bevy startup system for setting up all dataset parameters
fn setup_datasets(mut commands: Commands) {
    let d_enum: Vec<_> = PossibleDatasets::iter().collect();
    let mut serial_params = HashMap::<PossibleDatasets, String>::new();

    for s in d_enum {
        if path::Path::new(s.user_config_path()).exists() {
            serial_params.insert(s, fs::read_to_string(s.user_config_path()).unwrap());
        } else {
            serial_params.insert(s, fs::read_to_string(s.default_config_path()).unwrap());
        }
    }
    
    let mnist: mnist::MnistParams = ron::from_str(&serial_params[&PossibleDatasets::MNIST]).unwrap();

    let all_params = DatasetParams { mnist };
    let configs = DatasetConfigs(serial_params);

    commands.insert_resource(all_params);
    commands.insert_resource(configs);
}

/// Bevy shutdown system for saving changes to dataset parameters
/// TODO: Save every n seconds? 
fn save_dataset_params(
    mut exit: EventReader<AppExit>,
    config: Res<DatasetConfigs>,
    params: Res<DatasetParams>
) {
    let save_if_neq = |d: PossibleDatasets, a| {
        if a != config.0[&d] {
            let dir = path::Path::new(d.user_config_path()).parent().unwrap();
            if !dir.exists() {
                fs::create_dir_all(dir).unwrap();
            }
            fs::write(d.user_config_path(), a).unwrap();
        }
    };
    let mut exited = false;
    for _ in exit.iter() {
        exited = true;
    }
    if exited {
        save_if_neq(PossibleDatasets::MNIST, ron::to_string(&params.mnist).unwrap());    
    }
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