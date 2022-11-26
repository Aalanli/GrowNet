use std::f32::consts::PI;

use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::mesh::{self, PrimitiveTopology};
use bevy::render::render_resource::AsBindGroup;

const PI2: f32 = PI * 2.0;
pub fn build_circle(radius: f32, vertices: usize) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let n_vertices = vertices + 1;
    let n_triangles = vertices as u32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n_vertices);
    let mut indices: Vec<u32> = Vec::with_capacity((n_triangles as usize) * 3);

    for i in 0..vertices {
        let angle = (i as f32) / (vertices as f32) * PI2;
        let c = angle.cos();
        let s = angle.sin();

        let x = radius * c;
        let y = radius * s;

        let u = 0.5 * c + 0.5;
        let v = -0.5 * s + 0.5;

        positions.push([x, y, 0.]);
        normals.push([0., 0., 1.]);
        uvs.push([u, v]);
    }
    positions.push([0., 0., 0.]);
    normals.push([0., 0., 1.]);
    uvs.push([0.5, 0.5]);

    for i in 0..n_triangles {
        indices.push(i % n_triangles);
        indices.push((i + 1) % n_triangles);
        indices.push(n_triangles);
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    //mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(mesh::Indices::U32(indices)));

    mesh
}

fn create_ngon(
    radius: f32, 
    vertices: usize, 
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>) 
{
    for i in 0..vertices {
        let angle = (i as f32) / (vertices as f32) * PI2;
        let c = angle.cos();
        let s = angle.sin();

        let x = radius * c;
        let y = radius * s;

        let u = 0.5 * c + 0.5;
        let v = -0.5 * s + 0.5;

        positions.push([x, y, 0.]);
        normals.push([0., 0., 1.]);
    }

    positions.push([0., 0., 0.]);
    normals.push([0., 0., 1.]);

    let n_triangles = vertices as u32;
    for i in 0..n_triangles {
        indices.push(i % n_triangles);
        indices.push((i + 1) % n_triangles);
        indices.push(n_triangles);
    }
}

pub fn build_cylinder(radius: f32, length: f32, vertices: usize) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    create_ngon(radius, vertices, &mut positions, &mut normals, &mut indices);

    for i in 0..vertices {
        let angle = (i as f32) / (vertices as f32) * PI2;
        let c = angle.cos();
        let s = angle.sin();

        let x = radius * c;
        let y = radius * s;

        let u = 0.5 * c + 0.5;
        let v = -0.5 * s + 0.5;

        positions.push([x, y, 0.0]);
        normals.push([c, s, 0.0]);
    }

    for i in 0..vertices {
        let angle = (i as f32) / (vertices as f32) * PI2;
        let c = angle.cos();
        let s = angle.sin();

        let x = radius * c;
        let y = radius * s;

        let u = 0.5 * c + 0.5;
        let v = -0.5 * s + 0.5;

        positions.push([x, y, length]);
        normals.push([c, s, -1.]);
    }

    let faces = vertices as u32;
    let offset = 2 * faces + 1;
    for i in 0..faces {
        indices.push(offset + i);
        indices.push(offset + (i + 1) % faces);
        indices.push(offset + i + faces);

        indices.push(offset + i + faces + 1);
        indices.push(offset + (i + 1) % faces + faces);
        indices.push(offset + (i + 1) % faces);
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    //mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(mesh::Indices::U32(indices)));

    mesh

}
