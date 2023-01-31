use super::motion::{pan_orbit_camera, PanOrbitCamera};
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::render_resource::AsBindGroup;
use itertools::iproduct;

use bevy_egui::{egui, EguiContext};
const GLOBAL_SCALE: f32 = 10.0;

pub struct SimpleViewer;
impl Plugin for SimpleViewer {
    fn build(&self, app: &mut App) {
        app.add_startup_system(simple_viewer)
            .add_system(pan_orbit_camera);
    }
}
fn simple_viewer(mut commands: Commands) {
    commands.spawn_bundle(PointLightBundle {
        transform: Transform::from_xyz(0.0, GLOBAL_SCALE, 0.0),
        point_light: PointLight {
            intensity: 6000.,
            range: 400.,
            ..default()
        },
        ..default()
    });

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.7,
    });

    let translation = Vec3::new(-15.155, 17.358, 15.799);
    let radius = translation.length();

    commands
        .spawn_bundle(Camera3dBundle {
            //camera: Camera { viewport: Some(camera::Viewport {
            //    physical_position: UVec2 { x: 0, y: 0 },
            //     ..default()
            //}),
            //    ..default()},
            camera_3d: Camera3d {
                clear_color: ClearColorConfig::Custom(Color::rgb(0.9, 1.0, 1.0)),
                ..default()
            },
            transform: Transform::from_translation(translation).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert(PanOrbitCamera {
            radius,
            ..Default::default()
        });
}

pub struct VectorFieldPlugin {
    field: FieldDescriptor,
}

impl VectorFieldPlugin {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        let elems = x * y * z;
        let positions: Vec<Vec3> = iproduct!((0..x), (0..y), (0..z))
            .map(|(x, y, z)| Vec3::new(x as f32, y as f32, z as f32))
            .collect();
        let angles: Vec<Vec3> = (0..elems).into_iter().map(|_| Vec3::ZERO).collect();
        let lengths: Vec<f32> = (0..elems).into_iter().map(|_| 1.0).collect();
        VectorFieldPlugin {
            field: FieldDescriptor {
                bases: positions,
                point: angles,
                length: lengths,
                width_scale: 1.0,
            },
        }
    }
}

impl Default for VectorFieldPlugin {
    fn default() -> Self {
        let positions: Vec<Vec3> = iproduct!((0..8), (0..8), (0..8))
            .map(|(x, y, z)| Vec3::new(x as f32, y as f32, z as f32))
            .collect();
        let angles = positions
            .iter()
            .map(|ps| {
                let x = ps.x - 3.0;
                let y = ps.y - 3.0;
                let z = ps.z - 3.0;
                let x = -y + x * (-1.0 + x.powf(2.0) + y.powf(2.0)).powf(2.0);
                let y = x + y * (-1.0 + x.powf(2.0) + y.powf(2.0)).powf(2.0);
                let z = z + x * (y - z * z);
                Vec3::new(x, y, z)
            })
            .collect();
        let lengths = positions.iter().map(|x| x.length().sin()).collect();

        let desc = FieldDescriptor {
            bases: positions,
            point: angles,
            length: lengths,
            width_scale: 1.0,
        };
        VectorFieldPlugin { field: desc }
    }
}

impl Plugin for VectorFieldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.field.clone())
            .add_plugin(MaterialPlugin::<VectorMaterial>::default())
            .add_plugin(SimpleViewer)
            .add_startup_system(setup_field)
            .add_system(test_vec);
    }
}

fn setup_field(
    mut commands: Commands,
    mut vec_material: ResMut<Assets<VectorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    field: Res<FieldDescriptor>,
    asset_server: Res<AssetServer>,
) {
    let arrow = asset_server.load("arrow.stl");

    let min: Vec3 = field
        .bases
        .iter()
        .fold(field.bases[0], |acc, elem| acc.min(*elem));

    let max: Vec3 = field
        .bases
        .iter()
        .fold(field.bases[0], |acc, elem| acc.max(*elem));

    let scale = Vec3::from_array([10.0, 10.0, 10.0]) / (max - min);
    let shift = (max - min) / Vec3::from_array([2.0, 2.0, 2.0]);

    let field_id = commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane { size: GLOBAL_SCALE })),
            transform: Transform::from_xyz(0.0, -GLOBAL_SCALE / 2.0, 0.0),
            material: materials.add(StandardMaterial {
                base_color: Color::hex("ffd891").unwrap(),
                // vary key PBR parameters on a grid of spheres to show the effect
                metallic: 0.1,
                perceptual_roughness: 0.5,
                ..default()
            }),
            ..default()
        })
        .insert(Field)
        .id();

    for i in 0..field.bases.len() {
        let xyz = field.bases[i];
        let rot = field.point[i];
        let length = field.length[i];
        let width = 50.0 * field.width_scale + 500.0;
        let mut transform = Transform::from_scale(Vec3::new(100.0 * length + 500.0, width, width))
            .with_translation(
                (xyz - shift) * scale + Vec3::from_array([0.0, GLOBAL_SCALE / 2.0 + 0.3, 0.0]),
            );
        transform.rotate_x(rot.x);
        transform.rotate_y(rot.y);
        transform.rotate_z(rot.z);
        let vec = commands
            .spawn_bundle(MaterialMeshBundle {
                mesh: arrow.clone(),
                material: vec_material.add(VectorMaterial {
                    color: Color::rgba(0.1, 0.4, 0.9, 1.0),
                }),
                transform,
                ..Default::default()
            })
            .insert(bevy::pbr::NotShadowCaster {})
            .insert(Vector {})
            .id();
        commands.entity(field_id).add_child(vec);
    }
}

fn test_vec(
    mut egui_context: ResMut<EguiContext>,
    mut field_vecs: Query<&mut Transform, With<Vector>>,
    mut xyz: Local<(f32, f32, f32)>,
) {
    let past = *xyz;
    egui::Window::new("test vec").show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Slider::new(&mut xyz.0, -5.0..=5.0).text("x"));
        ui.add(egui::Slider::new(&mut xyz.1, -5.0..=5.0).text("y"));
        ui.add(egui::Slider::new(&mut xyz.2, -5.0..=5.0).text("z"));

        for mut v in field_vecs.iter_mut() {
            if *xyz != past {
                v.rotate_x(xyz.0 - past.0);
                v.rotate_y(xyz.1 - past.1);
                v.rotate_z(xyz.2 - past.2);
            }
        }
    });
}

#[derive(AsBindGroup, Debug, Clone, TypeUuid)]
#[uuid = "218c6a56-3cf7-44b1-bd27-37a1e4bbbe30"]
pub struct VectorMaterial {
    #[uniform(0)]
    pub color: Color,
}

impl Material for VectorMaterial {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        "shaders/test_frag.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

#[derive(Component)]
struct Field;

#[derive(Component)]
pub struct Vector;

#[derive(Clone)]
struct FieldDescriptor {
    bases: Vec<Vec3>,
    point: Vec<Vec3>,
    length: Vec<f32>,
    width_scale: f32,
}

fn test_animation(mut vecs: Query<&mut Transform, With<Vector>>) {
    for mut i in vecs.iter_mut() {
        let x = i.rotation.x;
        let angle = 0.1 * x;
        i.rotate_x(-angle);

        let y = i.rotation.y;
        let angle = 0.1 * y;
        i.rotate_y(-angle);

        let z = i.rotation.z;
        let angle = 0.1 * z;
        i.rotate_z(-angle);
    }
}
