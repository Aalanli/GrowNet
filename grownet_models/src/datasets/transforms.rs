use core::panic;

use ndarray::prelude::*;
use arrayfire as af;
use image::{self, ImageBuffer};

pub fn to_afarray(im: &Array4<f32>) -> af::Array<f32> {
    if im.is_standard_layout() {
        let dim = im.dim();
        let array = af::Array::new(im.as_slice().unwrap(), af::dim4!(dim.3 as u64, dim.2 as u64, dim.1 as u64, dim.0 as u64));
        array
    } else {
        // we want the array to be contiguous and in row_major order
        let im = im.as_standard_layout();
        let dim = im.dim();
        // since af_array is in col major order
        let array = af::Array::new(im.as_slice().unwrap(), af::dim4!(dim.3 as u64, dim.2 as u64, dim.1 as u64, dim.0 as u64));
        array
    }
}

pub fn from_afarray(im: af::Array<f32>) -> Array4<f32> {
    let dims = im.dims();
    let mut vec = vec![0.0f32; im.elements()];
    im.host(&mut vec);
    Array::from_shape_vec((dims[3] as usize, dims[2] as usize, dims[1] as usize, dims[0] as usize), vec).unwrap()
}

/// stack arrays together, making a new dimension in the first axis, requires that all
/// arrays are the same size
pub fn batch(imgs: &[&Array3<f32>]) -> Array4<f32> {
    let whc = imgs[0].dim();
    let b = imgs.len();
    let mut img = Array4::<f32>::zeros((b, whc.0, whc.1, whc.2));
    for i in 0..b {
        assert!(imgs[i].dim() == whc, "batch requires that all inputs be of equal size");
        let mut smut = img.slice_mut(s![i, .., .., ..]);
        smut.zip_mut_with(imgs[i], |a, b| {
            *a = *b;
        });
    }
    img
}

/// convert an array of shape [3, h, w] or [h, w, 3] to an RgbImage, panics if any other shape is given 
pub fn to_image(im: Array3<u8>) -> image::RgbImage {
    if im.dim().0 == 3 {
        let im = im.permuted_axes([1, 2, 0]);
        let im = im.as_standard_layout();
        let (h, w, _) = im.dim();
        let buf = im.as_slice().unwrap().to_vec();
        image::ImageBuffer::from_raw(w as u32, h as u32, buf).unwrap()
    } else if im.dim().2 == 3 {
        let (h, w, _) = im.dim();
        let buf = im.as_slice().unwrap().to_vec();
        image::ImageBuffer::from_raw(w as u32, h as u32, buf).unwrap()
    } else {
        panic!("not a rbg image, expected channel first or channel last with 3 channels");
    }
}

/// converts an image of width w, and height to a ndarray, if channel_last is true the resulting shape is 
/// [h, w, 3], otherwise, it is [3, h, w]
pub fn from_image(im: image::RgbImage, channel_last: bool) -> Array3<u8> {
    let (w, h) = (im.width(), im.height());
    let a = im.to_vec();
    let arr = Array::from_shape_vec((h as usize, w as usize, 3), a).unwrap();
    if channel_last {
        arr
    } else {
        arr.permuted_axes([2, 0, 1])
    }
}

#[test]
fn test_af_conversion() {
    let a = Array4::zeros((4, 3, 16, 16));
    let _af_array = to_afarray(&a);
}

#[test]
fn test_image_conversion() {
    let a = image::io::Reader::open("/home/allan/Programs/grownet/test_img.png").unwrap().decode().unwrap();
    let a = a.into_rgb8();
    let array_im = from_image(a.clone(), true);
    println!("{:?}", array_im.dim());
    let convert_back = to_image(array_im.clone());

    convert_back.save("/home/allan/Programs/grownet/converted.png").unwrap();
}