#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use anyhow::{Result, Context};
use serde::{Serialize, de::DeserializeOwned};
use ron;

pub mod datasets;
pub mod models;
pub mod ops;
pub mod allocator;


pub trait Config: Send + Sync {
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str) -> Result<()>;
}

impl<T: Serialize + DeserializeOwned + Send + Sync> Config for T {
    fn config(&self) -> String {
        ron::to_string(self).unwrap()
    }
    fn load_config(&mut self, config: &str) -> Result<()> {
        *self = ron::from_str(config).context(format!("Failed to load context {}", config))?;
        Ok(())
    }
}

pub use grownet_macros::Config as ConfigMacro;
#[test]
fn config_proc_macro_test() {
    #[derive(ConfigMacro)]
    struct Test {
        a: f32,
        b: usize,
        #[no_op]
        c: fn(usize) -> usize
    }

    impl Default for Test {
        fn default() -> Self {
            Test { c: |x| x, a: 0.0, b: 1 }
        }
    }

    let t = Test::default();
    println!("{}", t.config());
}

