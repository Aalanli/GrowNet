use std::collections::HashMap;
use std::fmt::{format, Display};
use std::ops::{AddAssign, Deref, DerefMut, Div, Index, IndexMut, Shr};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Options {
    INT(isize),
    FLOAT(f64),
    STR(String),
    BOOL(bool),
    PATH(PathBuf),
    CONFIG(Config),
}

/// Wrap basetype to Options type
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
from_overloads!(bool, bool, BOOL);

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

/// Unwrap Options type to base type, cloning each time
macro_rules! into_owned_overloads {
    ($to_type:ty, $opt:ident) => {
        impl From<&Options> for $to_type {
            fn from(i: &Options) -> Self {
                if let Options::$opt(i) = i {
                    i.clone() as $to_type
                } else {
                    panic!("Not variant {}", stringify!(opt));
                }
            }
        }
    };
}

into_owned_overloads!(i32, INT);
into_owned_overloads!(i64, INT);
into_owned_overloads!(u32, INT);
into_owned_overloads!(u64, INT);
into_owned_overloads!(isize, INT);
into_owned_overloads!(usize, INT);
into_owned_overloads!(f32, FLOAT);
into_owned_overloads!(f64, FLOAT);
into_owned_overloads!(bool, BOOL);
into_owned_overloads!(String, STR);
into_owned_overloads!(PathBuf, PATH);

/// Unwrap Options type to a reference of a base type,
macro_rules! into_ref_overloads {
    ($to_type:ty, $opt:ident, $($imut:tt)*) => {
        impl<'a> From<&'a $($imut)* Options> for &'a $($imut)* $to_type {
            fn from(i: &'a $($imut)* Options) -> Self {
                if let Options::$opt(i) = i {
                    i
                } else {
                    panic!("Not variant {}", stringify!(opt));
                }
            }
        }
    };
}

into_ref_overloads!(isize, INT, mut);
into_ref_overloads!(f64, FLOAT, mut);
into_ref_overloads!(bool, BOOL, mut);
into_ref_overloads!(str, STR, mut);
into_ref_overloads!(String, STR, mut);
into_ref_overloads!(PathBuf, PATH, mut);
into_ref_overloads!(isize, INT,);
into_ref_overloads!(f64, FLOAT,);
into_ref_overloads!(bool, BOOL,);
into_ref_overloads!(str, STR,);
into_ref_overloads!(String, STR,);
into_ref_overloads!(PathBuf, PATH,);

impl Options {
    pub fn is_int(&self) -> bool {
        if let Options::INT(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_float(&self) -> bool {
        if let Options::FLOAT(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_str(&self) -> bool {
        if let Options::STR(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_bool(&self) -> bool {
        if let Options::BOOL(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_path(&self) -> bool {
        if let Options::PATH(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_config(&self) -> bool {
        if let Options::CONFIG(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_same(&self, other: &Options) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// Not allowed the change the variant, only updates what's inside
    pub fn update(&mut self, val: &Options) -> Result<()> {
        if !self.is_same(val) {
            return Err(Error::msg(format!(
                "Error updating, not the same variant \nself: {:?}, \nother: {:?}",
                self, val
            )));
        }

        match (self, val) {
            (Options::CONFIG(a), Options::CONFIG(b)) => {
                a.update(b).context("CONFIG")?;
            }
            (a, b) => {
                *a = b.clone();
            }
        }
        Ok(())
    }

    fn display_(
        &self,
        padding: usize,
        name: Option<&str>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let pad = " ".repeat(padding);
        match self {
            Options::INT(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}i: {}", i)
                }
            }
            Options::FLOAT(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}f: {}", i)
                }
            }
            Options::STR(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}s: {}", i)
                }
            }
            Options::BOOL(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i)
                } else {
                    writeln!(f, "{pad}b: {}", i)
                }
            }
            Options::PATH(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}{}: {}", name, i.to_str().unwrap())
                } else {
                    writeln!(f, "{pad}p: {}", i.to_str().unwrap())
                }
            }
            Options::CONFIG(i) => {
                if let Some(name) = name {
                    writeln!(f, "{pad}config: {}", name)?;
                    i.display_(padding + 2, f)
                } else {
                    writeln!(f, "{pad}config: ")?;
                    i.display_(padding + 2, f)
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

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Config {
    map: HashMap<String, Options>,
    order: Vec<String>,
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
            unsafe { Some((&*str, &mut *idx)) }
        } else {
            None
        }
    }
}

impl Config {
    pub fn valid_key(key: &str) -> bool {
        !key.contains("/")
    }

    /// Constructs a Config with the given key, option pairs, ignores repeats
    /// and any names with "/" in them
    pub fn new(configs: Vec<(String, Options)>) -> Self {
        let mut map = HashMap::new();
        let mut order = Vec::new();
        for (name, config) in configs {
            if !map.contains_key(&name) && Self::valid_key(&name) {
                map.insert(name.clone(), config);
                order.push(name);
            }
        }
        Config { map, order }
    }

    /// Are the keys the same between the two configs?
    /// Allows for different ordering of keys, but key values need to have
    /// the same variant, and if the variant is Config, then recursively check
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
            if !a.is_same(b) {
                return false;
            }
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

    /// Is the current Config a subset of the other config? Where equality is defined as above.
    pub fn subset(&self, other: &Config) -> bool {
        for i in &self.order {
            if !other.order.contains(i) {
                return false;
            }
        }

        for i in &self.order {
            let a = self.map.get(i).unwrap();
            let b = other.map.get(i).unwrap();
            if !a.is_same(b) {
                return false;
            }
            match (a, b) {
                (Options::CONFIG(a), Options::CONFIG(b)) => {
                    if !a.subset(b) {
                        return false;
                    }
                }
                (_, _) => {}
            }
        }

        true
    }

    /// Updates the current config by other by replacing all values of self with other, if the variants are the same
    /// do this recursively for configs
    pub fn update(&mut self, other: &Config) -> Result<()> {
        for (k, v) in self.map.iter_mut() {
            let v1 = other
                .map
                .get(k)
                .ok_or(Error::msg(format!("failed to retrieve key {}", k)))?;
            v.update(v1).context(format!("On key {}", k))?;
        }
        Ok(())
    }

    /// Updates config at that entry, errors if key does not exist
    /// or if the variants are not the same
    pub fn update_key(&mut self, key: &String, val: &Options) -> Result<()> {
        let entry = self
            .map
            .get_mut(key)
            .ok_or(Error::msg(format!("failed to retrieve key {}", key)))?;
        entry.update(val).context(format!("Error on key {}", key))
    }

    /// Insert key inserts a new value into the config if there isn't one
    /// already, returning error. The order is appended last
    pub fn insert(&mut self, key: &str, val: &Options) -> Result<()> {
        if !Self::valid_key(key) {
            Err(Error::msg("invalid key, contains '/'"))
        } else if !self.map.contains_key(key) {
            self.order.push(key.to_string());
            self.map.insert(key.into(), val.clone());
            Ok(())
        } else {
            Err(Error::msg(format!("already contains key {}", key)))
        }
    }

    /// Adds the fields of the other Config to the current one,
    /// errors if there are any overlapping fields
    pub fn disjoint_union(&mut self, other: &Config) -> Result<()> {
        for (k, i) in other.map.iter() {
            self.insert(k, i)?;
        }
        Ok(())
    }

    /// Shorthand for insert(key, Options::CONFIG(other)), but by value instead of by reference
    pub fn add(&mut self, key: &str, other: Config) -> Result<()> {
        if !self.map.contains_key(key) {
            self.order.push(key.to_string());
            self.map.insert(key.into(), Options::CONFIG(other));
            Ok(())
        } else {
            Err(Error::msg(format!("contains key {}", key)))
        }
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

    /// gets the value from the path as directed by i
    /// the path can be recursive, ex: d/i, which assumes that there exists a Config variant
    /// at the key d, which also has a key i
    pub fn get(&self, k: &str) -> Option<&Options> {
        let mut p = k.split("/");
        let first_key = p.next();
        if first_key.is_none() {
            return None;
        }
        let first_key = first_key.unwrap();
        let mut k = if let Some(x) = self.map.get(first_key) {
            x
        } else {
            return None;
        };

        for s in p {
            if let Options::CONFIG(c) = k {
                if let Some(x) = c.map.get(s) {
                    k = x;
                    continue;
                }
            }
            return None;
        }
        Some(k)
    }

    pub fn get_mut(&mut self, k: &str) -> Option<&mut Options> {
        let mut p = k.split("/");
        let first_key = p.next();
        if first_key.is_none() {
            return None;
        }
        let first_key = first_key.unwrap();
        let mut k = if let Some(x) = self.map.get_mut(first_key) {
            x
        } else {
            return None;
        };

        for s in p {
            if let Options::CONFIG(c) = k {
                if let Some(x) = c.map.get_mut(s) {
                    k = x;
                    continue;
                }
            }
            return None;
        }
        Some(k)
    }

    /// same as get, but can panic
    pub fn uget(&self, k: &str) -> &Options {
        let mut p = k.split("/");
        let first_key = p.next();
        let first_key = first_key.expect("No key exists!");
        let mut k = self
            .map
            .get(first_key)
            .expect(&format!("Key {} does not exist in config", first_key));

        for s in p {
            if let Options::CONFIG(c) = k {
                k = c
                    .map
                    .get(s)
                    .expect(&format!("Key {} does not exist in config {}", first_key, c));
            }
        }
        k
    }

    pub fn uget_mut(&mut self, k: &str) -> &mut Options {
        let mut p = k.split("/");
        let first_key = p.next();
        let first_key = first_key.expect("No key exists!");
        let mut k = self
            .map
            .get_mut(first_key)
            .expect(&format!("Key {} does not exist in config", first_key));

        for s in p {
            if let Options::CONFIG(c) = k {
                k = c
                    .map
                    .get_mut(s)
                    .expect(&format!("Key {} does not exist in config", first_key));
            }
        }
        k
    }

    pub fn iter(&self) -> ConfigIter {
        ConfigIter {
            config: self,
            idx: 0,
        }
    }

    pub fn iter_mut(&mut self) -> ConfigIterMut {
        ConfigIterMut {
            config: self,
            idx: 0,
        }
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

impl Index<&str> for Config {
    type Output = Options;
    fn index(&self, index: &str) -> &Self::Output {
        self.uget(index)
    }
}

impl Index<&str> for Options {
    type Output = Options;
    fn index(&self, index: &str) -> &Self::Output {
        if let Options::CONFIG(c) = self {
            c.uget(index)
        } else {
            panic!("No Config variant at index {}", index)
        }
    }
}

impl IndexMut<&str> for Config {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.uget_mut(index)
    }
}

impl IndexMut<&str> for Options {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        if let Options::CONFIG(c) = self {
            c.uget_mut(index)
        } else {
            panic!("No Config variant at index {}", index)
        }
    }
}

#[test]
fn config_macro_test() {
    use crate::{config, opt};
    let k = "a".to_string();

    let mut _a = config!(
        ("a", 1),
        ("b", 3.0),
        ("c", [("d", 1), ("f", "3"), ("g", Path("k")), ("ok", true)]),
        ("d", Path(k))
    );

    let _b = config!(
        ("h", "some"),
        ("t", "times")
    );

    let val: i32 = (&_a["c/d"]).into();
    println!("{}", val);
    let _k: i32 = (&_a["c"]["d"]).into();
    println!("{}", _a);
}
