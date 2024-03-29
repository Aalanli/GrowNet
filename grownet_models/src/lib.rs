#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use std::fmt::Display;
use std::{path::PathBuf, collections::HashMap};
use std::any::Any;

use anyhow::{Context, Result};
use ron;
use serde::{de::DeserializeOwned, Serialize};
use derive_more::{Deref, DerefMut};

pub use grownet_macros::Flatten;
pub mod flatten;
pub use flatten::{Flatten, World};

pub mod allocator;
pub mod ctx;
pub mod configs;
pub mod datasets;
pub mod models;
pub mod ops;
pub use configs::{Config, Options};

pub mod nn;

use crate as model_lib;

/// Convert from idents and literals to Options
/// convert whatever is in [...] by calling config!(...)
#[macro_export]
macro_rules! opt {
    (Path($i:literal)) => {
        {
            let path: std::path::PathBuf = $i.into();
            Options::from(path)
        }
    };
    ($i:literal) => {
        Options::from($i)
    };
    (Path($i:ident)) => {
        {
            let path: PathBuf = $i.into();
            crate::Options::from(path)
        }
    };
    ($i:ident) => {
        Options::from($i)
    };
    ([$($i:tt)*]) => {
        Options::CONFIG(config!($($i)*))
    };
}

/// Convert into a config, inputs are pairs separated by comma,
/// ex. config!(("a", 1), ("b", 1.0), ("c", [("d", "e")]))
/// where the first pair is the key, and the second is the value, converted to Options type
/// anything in square brackets will be recursively converted to Config
#[macro_export]
macro_rules! config {
    ($(($k:literal,$($i:tt)*)),*) => {
        {
            let mut config = Vec::<(String, Options)>::new();
            $(
                config.push((($k).into(), opt!($($i)*)));
            )*
            Config::new(config)
        }
    };
}
