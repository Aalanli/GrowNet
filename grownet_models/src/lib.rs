#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use anyhow::Result;

pub mod datasets;
pub mod models;
pub mod ops;
pub mod allocator;


pub trait Config: Send + Sync {
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str) -> Result<()>;
}
