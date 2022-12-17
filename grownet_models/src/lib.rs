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

use grownet_macros::{Test, Config};
#[test]
fn config_proc_macro_test() {
    #[derive(Default, Config)]
    struct Test {
        a: datasets::transforms::Normalize,
        b: datasets::transforms::Normalize
    }

    let t = Test::default();
    println!("{}", t.config());
}

trait Test {
    fn call(&self);
}

struct A {}
impl Test for A {
    fn call(&self) {
        println!("A");
    }
}
struct B {}
impl Test for B {
    fn call(&self) {
        println!("B");
    }
}


#[test]
fn derive_test() {
    #[derive(Test)]
    struct C {
        a: A,
        b: B
    }
    
    let c = C { a: A {}, b: B {}};
    c.call();
}