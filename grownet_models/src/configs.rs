use std::fmt::{Display, format};
use std::ops::{Deref, DerefMut, Div, Shr, AddAssign};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Error, Result, Context};
use serde::{Serialize, Deserialize};


pub mod cast {
    pub struct Path;
    pub struct Int;
    pub struct Float;
    pub struct Str;
    pub struct Config;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Options {
    INT(isize),
    FLOAT(f64),
    STR(String),
    PATH(PathBuf),
    CONFIG(Config)
}

impl Options {
    pub fn is_same(&self, other: &Options) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    pub fn is_config(&self) -> bool {
        match self {
            Options::CONFIG(_) => true,
            _ => false,
        }
    }

    /// Not allowed the change the variant, only updates what's inside
    pub fn update(&mut self, val: &Options) -> Result<()> {
        if !self.is_same(val) {
            return Err(Error::msg(format!("Error updating, not the same variant \nself: {:?}, \nother: {:?}", self, val)));
        }

        match (self, val) {
            (Options::CONFIG(a), Options::CONFIG(b)) => {
                a.update(b).context("CONFIG")?;
            }
            (a, b) => { *a = b.clone(); }
        }
        Ok(())
    }

    fn display_(&self, padding: usize, name: Option<&str>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pad = " ".repeat(padding);
        match self {
            Options::INT(i)     => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}i: {}", i)                    
                }
            },
            Options::FLOAT(i)     => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}f: {}", i)                    
                }
            }
            Options::STR(i)    => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}s: {}", i)                    
                }
            }
            Options::PATH(i)  => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i.to_str().unwrap())
                } else {
                      
                    writeln!(f, "{pad}p: {}", i.to_str().unwrap())
                }
            }
            Options::CONFIG(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}config: {}", name)?; i.display_(padding + 2, f) 
                } else {
                    writeln!(f, "{pad}config: ")?; i.display_(padding + 2, f) 
                }
            }
        }
    }
}

impl Display for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_(0, None, f)
    }
}

/// From overloads for various types
macro_rules! from_overloads {
    ($otype:tt, $cast_to:tt, $opt:ident) => {
        impl From<$otype> for Options {
            fn from(i: $otype) -> Self {
                Options::$opt(i as $cast_to)
            }
        }
        
    };
}

from_overloads!(i32, isize, INT);
from_overloads!(i64, isize, INT);
from_overloads!(u32, isize, INT);
from_overloads!(u64, isize, INT);
from_overloads!(isize, isize, INT);
from_overloads!(usize, isize, INT);
from_overloads!(f32, f64, FLOAT);
from_overloads!(f64, f64, FLOAT);

impl From<String> for Options {
    fn from(i: String) -> Self {
        Options::STR(i)
    }
}

impl<'a> From<&'a str> for Options {
    fn from(i: &'a str) -> Self {
        Options::STR(i.into())
    }
}

impl From<PathBuf> for Options {
    fn from(i: PathBuf) -> Self {
        Options::PATH(i)
    }
}

impl From<Vec<(String, Options)>> for Options {
    fn from(i: Vec<(String, Options)>) -> Self {
        Options::CONFIG(Config::new(i))
    }
}

/// Convert from idents and literals to Options
/// convert whatever is in [...] by calling config!(...)
#[macro_export]
macro_rules! opt {
    (Path($i:literal)) => {
        {
            let path: PathBuf = $i.into();
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


#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Config {
    map: HashMap<String, Options>,
    order: Vec<String>
}

pub struct ConfigIter<'a> {
    config: &'a Config,
    idx: usize,
}

pub struct ConfigIterMut<'a> {
    config: &'a mut Config,
    idx: usize,
}

impl<'a> Iterator for ConfigIter<'a> {
    type Item = (&'a str, &'a Options);
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.config.index(self.idx);
        self.idx += 1;
        if let Some(idx) = idx {
            Some((&self.config.order[self.idx - 1], idx))
        } else {
            None
        }
    }
}

impl<'a> Iterator for ConfigIterMut<'a> {
    type Item = (&'a str, &'a mut Options);
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.config.index_mut(self.idx);
        self.idx += 1;
        if let Some(idx) = idx {
            let idx = idx as *mut Options;
            let str = &self.config.order[self.idx - 1] as *const String;
            unsafe {
                Some((&*str, &mut *idx))
            }
        } else {
            None
        }
    }
}

impl Config {
    pub fn new(configs: Vec<(String, Options)>) -> Self {
        let mut map = HashMap::new();
        let mut order = Vec::new();
        for (name, config) in configs {
            if !map.contains_key(&name) {
                map.insert(name.clone(), config);
                order.push(name);
            }
        }
        Config { map, order }
    }

    /// Are the keys the same between the two configs?
    /// Allows for different ordering of keys, but key values need to have
    /// the same value, and if the variant is Config, then recursively check
    pub fn is_same(&self, other: &Config) -> bool {
        for i in &self.order {
            if !other.order.contains(i) {
                return false;
            }
        }
        for i in &other.order {
            if !self.order.contains(i) {
                return false;
            }
        } // both contains the same keys
        
        for i in &self.order {
            let a = self.map.get(i).unwrap();
            let b = other.map.get(i).unwrap();
            if !a.is_same(b) { return false; }
            match (a, b) {
                (Options::CONFIG(a), Options::CONFIG(b)) => {
                    if !a.is_same(b) {
                        return false;
                    }
                }
                (_, _) => {}
            } 
        }
        
        true
    }

    /// Updates the current config with a new config if the new config contains
    /// all the keys of the current config, and each key is the 'same' as defined by is_same
    pub fn update(&mut self, val: &Config) -> Result<()> {
        for (k, v) in self.map.iter_mut() {
            let v1 = val.map.get(k).ok_or(Error::msg(format!("failed to retrieve key {}", k)))?;
            v.update(v1).context(format!("On key {}", k))?;
        }
        Ok(())
    }

    /// Updates config at that entry, errors if key does not exist
    /// or if the variants are not the same
    pub fn update_key(&mut self, key: &String, val: &Options) -> Result<()> {
        let entry = self.map.get_mut(key).ok_or(Error::msg(format!("failed to retrieve key {}", key)))?;
        entry.update(val).context(format!("Error on key {}", key))
    }

    /// Insert key inserts a new value into the config if there isn't one
    /// already, returning error. The order is appended last
    pub fn insert(&mut self, key: &String, val: &Options) -> Result<()> {
        if !self.map.contains_key(key) {
            self.map.insert(key.clone(), val.clone());
            Ok(())
        } else {
            Err(Error::msg(format!("contains key {}", key)))
        }
    }

    /// Adds the fields of the other Config to the current one,
    /// errors if there are any overlapping fields
    pub fn add(&mut self, other: &Config) -> Result<()> {
        for (k, i) in other.map.iter() {
            self.insert(k, i)?;
        }
        Ok(())
    }

    pub fn index(&self, i: usize) -> Option<&Options> {
        if i < self.order.len() {
            Some(self.map.get(&self.order[i]).unwrap())
        } else {
            None
        }
    }

    pub fn index_mut(&mut self, i: usize) -> Option<&mut Options> {
        if i < self.order.len() {
            Some(self.map.get_mut(&self.order[i]).unwrap())
        } else {
            None
        }
    }

    pub fn iter(&self) -> ConfigIter {
        ConfigIter { config: self, idx: 0 }
    }

    pub fn iter_mut(&mut self) -> ConfigIterMut {
        ConfigIterMut { config: self, idx: 0 }
    }

    fn display_(&self, padding: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (k, v) in self.iter() {
            v.display_(padding, Some(k), f)?;
        }
        Ok(())
    }
}

impl Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_(0, f)
    }
}

/// Convenient syntax to use to access config fields, panics if cast or key is unavailable
/// (a: Config) / "a" gets the Options at that key address
/// (a: Options) / casts::T trys to cast the option to a particular type
impl<'a> Div<&'static str> for &'a Config {
    type Output = &'a Options;
    fn div(self, rhs: &'static str) -> Self::Output {
        let h = self.map.get(rhs).expect(&format!("Div op: unable to get key {}", rhs));
        h
    }
}

impl<'a> Div<&'static str> for &'a mut Config {
    type Output = &'a mut Options;
    fn div(self, rhs: &'static str) -> Self::Output {
        let h = self.map.get_mut(rhs).expect(&format!("Div op: unable to get key {}", rhs));
        h
    }
}

macro_rules! div_overloads {
    ($cast_ty:ty, $out_ty:ty, $opt:ident) => {
        impl<'a> Div<$cast_ty> for &'a Options {
            type Output = &'a $out_ty;
            fn div(self, _rhs: $cast_ty) -> Self::Output {
                match self {
                    Options::$opt(i) => i,
                    _ => panic!("Div op: not an {}!", stringify!(opt))
                }
            }
        }

        impl<'a> Div<$cast_ty> for &'a mut Options {
            type Output = &'a mut $out_ty;
            fn div(self, _rhs: $cast_ty) -> Self::Output {
                match self {
                    Options::$opt(i) => i,
                    _ => panic!("Div op: not an {}!", stringify!(opt))
                }
            }
        }
    };
}

div_overloads!(cast::Int, isize, INT);
div_overloads!(cast::Float, f64, FLOAT);
div_overloads!(cast::Str, String, STR);
div_overloads!(cast::Path, PathBuf, PATH);
div_overloads!(cast::Config, Config, CONFIG);

#[test]
fn config_macro_test() {
    let k = "a".to_string();
    
    let mut _a = config!(
        ("a", 1), 
        ("b", 3.0), 
        ("c", [
            ("d", 1), 
            ("f", "3"), 
            ("g", Path("k"))
            ]),
        ("d", Path(k))
    );
    let _k = &mut _a / "c" / cast::Config / "d";
    println!("{}", _a);
}