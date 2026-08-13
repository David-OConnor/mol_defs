//! For calculating the accessible surface (AS), a proxy for accessibility of solvents
//! to a molecule. Used for drawing *surface*, *dots*, and related meshes. Related to the van der Waals
//! radius.
//!
//! Also used for other molecule-based meshes, like pockets for pharmacophores and docking.
//!
//! The surface itself is built with marching cubes and is always available: molecular volume and
//! topological surface area are derived from it, and those are inputs to property inference. Only
//! the parts which hand a mesh to a render engine — vertex colouring, and the `graphics::Mesh`
//! conversion — need the `render` feature.

use std::fmt::{Display, Formatter};

use bincode::{Decode, Encode};
#[cfg(feature = "render")]
use graphics::{Mesh, Vertex};
use lin_alg::f32::Vec3;
use mcubes::{MarchingCubes, MeshSide};

/// Vertex colours for a molecule-wrapping mesh. Computing them is Molchanica's job; the alias lives
/// here so `MeshColoring` and the meshes it applies to stay together.
pub type MeshColors = Vec<Option<(u8, u8, u8, u8)>>;

pub const SOLVENT_RAD: f32 = 1.4; // water probe

/// For  molecule-wrapping meshes. E.g. SAS around a protein, pockets etc.
#[derive(Clone, Copy, Debug, PartialEq, Default, Encode, Decode)]
pub enum MeshColoring {
    #[default]
    Solid, // todo: Wrap the color?
    Element,
    PartialCharge,
    /// aka greasiness
    Lipophilicity,
}

impl Display for MeshColoring {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Solid => "Solid",
            Self::Element => "Element",
            Self::PartialCharge => "Charge",
            Self::Lipophilicity => "Lipophilicity",
        };
        write!(f, "{v}")
    }
}

/// Create a mesh of the solvent-accessible surface. We do this using the ball-rolling method
/// based on Van-der-Waals radius, then use the Marching Cubes algorithm to generate an iso mesh with
/// iso value = 0.
///
/// Atoms is (posit, vdw radius).
///
/// Returns the marching-cubes mesh directly: callers that want to measure the surface (volume,
/// topological surface area) use it as-is, and `make_sas_mesh` converts it for the render engine.
pub fn make_sas_mesh_mc(atoms: &[(Vec3, f32)], radius: f32, precision: f32) -> mcubes::Mesh {
    if atoms.is_empty() {
        return mcubes::Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    // Bounding box and grid
    let mut bb_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut bb_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut r_max: f32 = 0.0;

    for (posit, vdw_radius) in atoms {
        let r = vdw_radius + radius;
        r_max = r_max.max(r);

        bb_min = Vec3::new(
            bb_min.x.min(posit.x),
            bb_min.y.min(posit.y),
            bb_min.z.min(posit.z),
        );

        bb_max = Vec3::new(
            bb_max.x.max(posit.x),
            bb_max.y.max(posit.y),
            bb_max.z.max(posit.z),
        );
    }
    bb_min -= Vec3::splat(r_max + precision);
    bb_max += Vec3::splat(r_max + precision);

    let dim_v = (bb_max - bb_min) / precision;
    let grid_dim = (
        dim_v.x.ceil() as usize + 1,
        dim_v.y.ceil() as usize + 1,
        dim_v.z.ceil() as usize + 1,
    );

    let n_voxels = grid_dim.0 * grid_dim.1 * grid_dim.2;

    // This can be any that is guaranteed to be well outside the SAS surface.
    // It prevents holes from appearing in the mesh due to not having a value outside to compare to.
    let far_val = (r_max + precision).powi(2) + 1.0;
    let mut field = vec![far_val; n_voxels];

    // Helper to flatten (x, y, z)
    let idx = |x: usize, y: usize, z: usize| -> usize { (z * grid_dim.1 + y) * grid_dim.0 + x };

    // Fill signed-squared-distance field
    for (center, vdw_radius) in atoms {
        let rad = *vdw_radius + radius;
        let rad2 = rad * rad;

        let lo = ((*center - Vec3::splat(rad)) - bb_min) / precision;
        let hi = ((*center + Vec3::splat(rad)) - bb_min) / precision;

        let (xi0, yi0, zi0) = (
            lo.x.floor().max(0.0) as usize,
            lo.y.floor().max(0.0) as usize,
            lo.z.floor().max(0.0) as usize,
        );
        let (xi1, yi1, zi1) = (
            hi.x.ceil().min((grid_dim.0 - 1) as f32) as usize,
            hi.y.ceil().min((grid_dim.1 - 1) as f32) as usize,
            hi.z.ceil().min((grid_dim.2 - 1) as f32) as usize,
        );

        for z in zi0..=zi1 {
            for y in yi0..=yi1 {
                for x in xi0..=xi1 {
                    let p = bb_min + Vec3::new(x as f32, y as f32, z as f32) * precision;
                    let d2 = (p - *center).magnitude_squared();
                    let v = d2 - rad2;
                    let f = &mut field[idx(x, y, z)];
                    if v < *f {
                        *f = v;
                    }
                }
            }
        }
    }

    // Convert to a mesh using Marchine Cubes.
    let sampling_interval = (
        grid_dim.0 as f32 - 1.0,
        grid_dim.1 as f32 - 1.0,
        grid_dim.2 as f32 - 1.0,
    );

    //  scale = precision because size / sampling_interval = precision
    let size = (
        sampling_interval.0 * precision,
        sampling_interval.1 * precision,
        sampling_interval.2 * precision,
    );

    // todo: The holes in our mesh seem related to the iso level chosen.
    let mc = MarchingCubes::new(grid_dim, size, sampling_interval, bb_min, field, 0.)
        .expect("marching cubes init");

    // Note: We're experiencing the opposite behavior than we expect here; we really want to draw outside.
    let mc_mesh = mc.generate(MeshSide::InsideOnly);

    mc_mesh
}

/// The solvent-accessible surface as a mesh the render engine can draw.
#[cfg(feature = "render")]
pub fn make_sas_mesh(atoms: &[(Vec3, f32)], radius: f32, precision: f32) -> Mesh {
    let mc_mesh = make_sas_mesh_mc(atoms, radius, precision);

    let vertices: Vec<Vertex> = mc_mesh
        .vertices
        .iter()
        // I'm not sure why we need to invert the normal here; same reason we use InsideOnly above.
        .map(|v| Vertex::new([v.posit.x, v.posit.y, v.posit.z], -v.normal))
        .collect();

    Mesh {
        vertices,
        indices: mc_mesh.indices,
        material: 0,
    }
}
