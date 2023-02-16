#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use ron;
use serde::{de::DeserializeOwned, Serialize};

pub mod allocator;
pub mod configs;
pub mod datasets;
pub mod models;
pub mod ops;
pub use configs::{Config, Options};

pub trait Configure: Send + Sync {
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str) -> Result<()>;
}

impl<T: Serialize + DeserializeOwned + Send + Sync> Configure for T {
    fn config(&self) -> String {
        ron::to_string(self).unwrap()
    }
    fn load_config(&mut self, config: &str) -> Result<()> {
        *self = ron::from_str(config).context(format!("Failed to load context {}", config))?;
        Ok(())
    }
}

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
            Options::from(path)
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

pub use grownet_macros::Config as ConfigMacro;
#[test]
fn config_proc_macro_test() {
    #[derive(ConfigMacro)]
    struct Test {
        a: f32,
        b: usize,
        #[no_op]
        c: fn(usize) -> usize,
    }

    impl Default for Test {
        fn default() -> Self {
            Test {
                c: |x| x,
                a: 0.0,
                b: 1,
            }
        }
    }

    let t = Test::default();
    println!("{}", t.config());
}