use std::marker::PhantomData;

use serde::{Serialize, de::DeserializeOwned, Deserialize};
use anyhow::{Result, Context};
use tch::Tensor;

use crate::ops;
use crate::Config;
use super::data;

pub trait Transform: Config + Sized + Clone {
    type In;
    type Out;
    fn transform(&self, data: Self::In) -> Self::Out;
    fn compose<T: Transform<In = Self::Out>>(self, t: T) -> Compose<Self, T> {
        Compose { t1: self, t2: t }
    }
}

#[derive(Clone)]
pub struct Concat<T1, T2, F, O> {
    pub t1: T1,
    pub t2: T2,
    pub f: F,
    _out: PhantomData<O>
}

impl<T1, T2, F, O> Concat<T1, T2, F, O>
where T1: Transform, T2: Transform, F: Fn(&T1, &T2, T1::In) -> O + Send + Sync + Clone,
    O: Send + Sync + Clone
{
    pub fn new(t1: T1, t2: T2, f: F) -> Self {
        Self { t1, t2, f, _out: PhantomData }
    }
}

impl<T1, T2, F, O> Config for Concat<T1, T2, F, O>
where T1: Transform, T2: Transform, F: Fn(&T1, &T2, T1::In) -> O + Send + Sync + Clone,
    O: Send + Sync + Clone
{
    fn config(&self) -> String {
        let pre = self.t1.config();
        let post = self.t2.config();
        ron::to_string(&(pre, post)).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let (pre, post): (String, String) = ron::from_str(config)?;
        self.t1.load_config(&pre)?;
        self.t2.load_config(&post)?;
        Ok(())
    }
}

impl<T1, T2, F, O> Transform for Concat<T1, T2, F, O>
where T1: Transform, T2: Transform, F: Fn(&T1, &T2, T1::In) -> O + Send + Sync + Clone,
    O: Send + Sync + Clone
{
    type In = T1::In;
    type Out = O;
    fn transform(&self, data: Self::In) -> Self::Out {
        let f = &self.f;
        let x = f(&self.t1, &self.t2, data);
        x
    }
}

#[derive(Clone)]
pub struct Compose<T1, T2> {
    pub t1: T1,
    pub t2: T2
}

impl<T1, T2> Config for Compose<T1, T2>
where T1: Config, T2: Config
{
    fn config(&self) -> String {
        let pre = self.t1.config();
        let post = self.t2.config();
        ron::to_string(&(pre, post)).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let (pre, post): (String, String) = ron::from_str(config)?;
        self.t1.load_config(&pre)?;
        self.t2.load_config(&post)?;
        Ok(())
    }
}

impl<T1, T2> Transform for Compose<T1, T2> 
where T1: Transform, T2: Transform<In = T1::Out>
{
    type In = T1::In;
    type Out = T2::Out;
    fn transform(&self, data: Self::In) -> Self::Out {
        let x = self.t1.transform(data);
        self.t2.transform(x)
    }
}

#[derive(Clone)]
pub struct FnTransform<F, In, Out> {
    pub f: F,
    _in: PhantomData<In>,
    _out: PhantomData<Out>
}

impl<F1, In1, Out1> FnTransform<F1, In1, Out1>
{
    pub fn new<In, Out>(f: impl Fn(In) -> Out) -> FnTransform<impl Fn(In) -> Out, In, Out> {
        FnTransform { f, _in: PhantomData, _out: PhantomData }
    } 
}

impl<In, Out, F: Fn(In) -> Out> From<F> for FnTransform<F, In, Out> {
    fn from(f: F) -> Self {
        FnTransform { f, _in: PhantomData, _out: PhantomData }
    }
}

impl<In, Out, F> Config for FnTransform<F, In, Out>
where In: Send + Sync, Out: Send + Sync, F: Send + Sync 
{
    fn config(&self) -> String {
        "".to_string()
    }
    fn load_config(&mut self, _config: &str) -> Result<()> {
        Ok(())
    }
}

impl<In, Out, F> Transform for FnTransform<F, In, Out>
where F: Send + Sync + Fn(In) -> Out + Clone, In: Send + Sync + Clone, Out: Send + Sync + Clone {
    type In = In;
    type Out = Out;
    fn transform(&self, data: Self::In) -> Self::Out {
        let f = &self.f;
        f(data)
    }
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Normalize {
    pub mu: f32,
    pub range: f32,
}

impl Default for Normalize {
    fn default() -> Self {
        Normalize { mu: 0.0, range: 2.0 }
    }
}

impl Transform for Normalize {
    type In = data::Image;
    type Out = data::Image;
    fn transform(&self, mut data: Self::In) -> Self::Out {
        let mut min = data.image[[0, 0, 0, 0]];
        let mut max = data.image[[0, 0, 0, 0]];
        data.image.for_each(|x| {
            min = min.min(*x);
            max = max.max(*x);
        });
        let width = self.range / (max - min);
        let center = (max + min) / 2.0;
        data.image.mapv_inplace(|x| {
            (x - center + self.mu) / width
        });
        data
    }
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BasicImAugumentation {
    pub flip: bool,
    pub crop: i64,
    pub cutout: i64,
}

impl BasicImAugumentation {
    pub fn transform(&self, x: &Tensor) -> Tensor {
        tch::vision::dataset::augmentation(x, self.flip, self.crop, self.cutout)
    } 
}

impl Default for BasicImAugumentation {
    fn default() -> Self {
        Self { flip: true, crop: 4, cutout: 8 }
    }
}

impl Transform for BasicImAugumentation {
    type In = data::Image;
    type Out = data::Image;
    fn transform(&self, data: Self::In) -> Self::Out {
        let ts = ops::convert_image_array(&data.image.view()).unwrap();
        let ts = self.transform(&ts);
        let im = ops::convert_image_tensor(&ts).unwrap();
        data::Image { image: im }
    }
}




#[test]
fn test_transform_fn() {
    let t1 = |x: u32| (x * 2) as usize;
    let t2 = |x: usize| x as f32 / 2.0;

    let t1: FnTransform<_, _, _> = t1.into();
    let t2: FnTransform<_, _, _> = t2.into();
    let t2 = t1.compose(t2);

    let h = t2.transform(1);
    assert!(h == 1.0);
}