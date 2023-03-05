use std::{any::Any, error::Error};
use std::fmt::Display;
use std::collections::HashMap;

use derive_more::{Deref, DerefMut};
use grownet_macros::Flatten;

#[derive(Default)]
pub struct World<'a> {
    objects: Vec<&'a mut dyn Any>,
    field_path: Vec<String>,
}

pub trait Flatten {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>);
}

impl<'a> World<'a> {
    pub fn new() -> World<'a> {
        Default::default()
    }

    pub fn query_mut<'b, T: 'static>(&'b mut self) -> impl Iterator<Item = &'a mut T> + 'b {
        self.objects.iter_mut()
            .map(|x| {
                // compiler error otherwise, saying 'b: 'a is necessary, implying this function
                // borrows mutably for as long as 'a, eg. can only be called once. 
                // The only requirement is 'a: 'b, which is already satisfied
                let x = unsafe {
                    let x = x as *mut &mut dyn Any;
                    let x = x as *mut *mut dyn Any;
                    &mut **x
                };
                x.downcast_mut::<T>()
            })
            .filter(|x| x.is_some())        
            .map(|x| x.unwrap())
    }

    pub fn query_mut_with_path<'b, T: 'static>(&'b mut self) -> impl Iterator<Item = (&'a str, &'a mut T)> + 'b {
        self.objects.iter_mut().zip(self.field_path.iter())
            .map(|(x, path)| {
                // compiler error otherwise, saying 'b: 'a is necessary, implying this function
                // borrows mutably for as long as 'a, eg. can only be called once. 
                // The only requirement is 'a: 'b, which is already satisfied
                let x = unsafe {
                    let x = x as *mut &mut dyn Any;
                    let x = x as *mut *mut dyn Any;
                    &mut **x
                };
                let path = unsafe {
                    let path = path.as_str();
                    let path = path as *const str;
                    &*path
                };
                (x.downcast_mut::<T>(), path)
            })
            .filter(|x| x.0.is_some())        
            .map(|x| (x.1, x.0.unwrap()))
    }

    pub fn push<T: 'static>(&mut self, push: String, a: &'a mut T) {
        let a: &mut dyn Any = a;
        self.objects.push(a);
        self.field_path.push(push);
    }

    pub fn clear(mut self) -> Self {
        self.objects.clear();
        self.field_path.clear();
        self
    }
}

impl<'a, T: Flatten> From<&'a mut T> for World<'a> {
    fn from(value: &'a mut T) -> Self {
        let mut world = World::new();
        value.flatten("".to_string(), &mut world);
        world
    }
}

#[derive(Deref, DerefMut)]
pub struct FlattenVec<T: Flatten>(Vec<T>);

#[derive(Deref, DerefMut)]
pub struct FlattenHashMap<K: Display, V: Flatten>(HashMap<K, V>);

impl<T> Flatten for FlattenVec<T> 
where T: 'static + Flatten {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        for (n, i) in self.0.iter_mut().enumerate() {
            i.flatten(format!("{}-vec[{}]", path, n), world)
        }
    }
}

impl<K, V> Flatten for FlattenHashMap<K, V>
where K: Display, V: Flatten
{
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        for (k, v) in self.iter_mut() {
            v.flatten(format!("{}-hashmap: {}", path, k), world);
        }
    }
}


macro_rules! derive_flatten_concrete {
    ($type:ty) => {
        impl Flatten for $type {
            fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
                world.push(path, self);
            }
        }
    };
}

derive_flatten_concrete!(f32);
derive_flatten_concrete!(f64);
derive_flatten_concrete!(u32);
derive_flatten_concrete!(u64);
derive_flatten_concrete!(i32);
derive_flatten_concrete!(i64);
derive_flatten_concrete!(usize);
derive_flatten_concrete!(isize);
derive_flatten_concrete!(String);

impl<const N: usize, T: 'static> Flatten for [T; N] {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        world.push(path, self);
    }
}

impl<T: 'static> Flatten for Option<T> {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        world.push(path, self);
    }
}

impl<T: 'static, E: 'static> Flatten for Result<T, E> {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        world.push(path, self);
    }
}


impl<T: 'static> Flatten for Vec<T> {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        world.push(path, self);
    }
}

impl<K: 'static, V: 'static> Flatten for HashMap<K, V> {
    fn flatten<'a>(&'a mut self, path: String, world: &mut World<'a>) {
        world.push(path, self);
    }
}

#[test]
fn test_struct1() {
    #[derive(Flatten, Default)]
    struct Test {
        a: f32,
        b: f64,
        c: Vec<u32>,
    }

    let mut world = World::default();
    let mut test = Test::default();
    test.a += 1.0;
    test.flatten("".to_string(), &mut world);
    for i in world.query_mut::<f32>() {
        println!("{}", i);
    }
    
}