//! Data types for electron density, as measured by crystallography and Cryo-EM reflection data.
//!
//! Note: the algorithms which build these — FFT from structure factors, density-mesh generation,
//! and the symmetry expansion in `make_densities` — live in Molchanica. Only the types they produce
//! and consume are here, so molecules can carry density data without depending on an FFT or GPU stack.

use bio_files::DensityMap;
use lin_alg::f64::Vec3;

pub const DENSITY_CELL_MARGIN: f64 = 3.0;

// Density points must be within this distance in Å of a protein atom to be generated.
// This prevents displaying shapes from the neighbor
pub const DENSITY_MAX_DIST: f64 = 4.;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum MapStatus {
    /// Ordinary, or observed; the bulk of values.
    Observed,
    FreeSet,
    SystematicallyAbsent,
    OutsideHighResLimit,
    HigherThanResCutoff,
    LowerThanResCutoff,
    /// Ignored
    #[default]
    UnreliableMeasurement,
}

impl MapStatus {
    pub fn from_str(val: &str) -> Option<MapStatus> {
        match val.to_lowercase().as_ref() {
            "o" => Some(MapStatus::Observed),
            // "o" => Some(MapType::M2FOFC),
            // "d" => Some(MapType::DifferenceMap),
            "f" => Some(MapStatus::FreeSet),
            "-" => Some(MapStatus::SystematicallyAbsent),
            "<" => Some(MapStatus::OutsideHighResLimit),
            "h" => Some(MapStatus::HigherThanResCutoff),
            "l" => Some(MapStatus::LowerThanResCutoff),
            "x" => Some(MapStatus::UnreliableMeasurement),
            _ => {
                eprintln!("Fallthrough on map type: {val}");
                None
            }
        }
    }
}

#[allow(unused)]
/// Reflection data for a single Miller index set. Pieced together from 3 formats of CIF
/// file (Structure factors, map 2fo-fc, and map fo-fc), or an MTZ.
#[derive(Clone, Default, Debug)]
pub struct Reflection {
    /// Miller indices.
    pub h: i32,
    pub k: i32,
    pub l: i32,
    pub status: MapStatus,
    /// Amplitude. i.e. F_meas. From SF.
    pub amp: f64,
    /// Standard uncertainty (σ) of amplitude. i.e. F_meas_sigma_au. From SF.
    pub amp_uncertainty: f64,
    /// ie. FWT. From 2fo-fc.
    pub amp_weighted: Option<f64>,
    /// i.e. PHWT. In degrees. From 2fo-fc.
    pub phase_weighted: Option<f64>,
    /// i.e. FOM. From 2fo-fc.
    pub phase_figure_of_merit: Option<f64>,
    /// From fo-fc.
    pub delta_amp_weighted: Option<f64>,
    /// From fo-fc.
    pub delta_phase_weighted: Option<f64>,
    /// From fo-fc.
    pub delta_figure_of_merit: Option<f64>,
}

/// Miller-index-based reflection data.
#[derive(Clone, Debug, Default)]
pub struct ReflectionsData {
    /// X Y Z for a b c?
    pub space_group: String,
    pub cell_len_a: f32,
    pub cell_len_b: f32,
    pub cell_len_c: f32,
    pub cell_angle_alpha: f32,
    pub cell_angle_beta: f32,
    pub cell_angle_gamma: f32,
    pub points: Vec<Reflection>,
}

/// Electron density at a single point in space.
#[derive(Clone, Debug)]
pub struct DensityPt {
    /// In Å
    pub coords: Vec3,
    /// Normalized, using the unit cell volume, as reported in the reflection data.
    pub density: f64,
}

// todo: I'm not sure I like this. I think we should remove it.
/// One dense 3-D brick of map values. We use this struct to handle symmetry: ensuring full coverage
/// of all atoms.
#[derive(Clone, Debug)]
pub struct DensityRect {
    /// Cartesian coordinate of the center of voxel (0,0,0)
    pub origin_cart: Vec3,
    /// Size of one voxel along a, b, c in Å
    pub step: [f64; 3],
    /// (nx, ny, nz) – number of voxels stored
    pub dims: [usize; 3],
    /// See the header for the dimension breakdown. Usually:
    /// X is the fast (contiguous) dimension. Z is the slow (strided) dimension.
    /// See the Mapc/Mapr/Maps fields. If 1/2/3, it's as above.
    pub data: Vec<f32>,
}

impl DensityRect {
    /// Extract the smallest cube that covers all atoms plus `margin` Å.
    /// `margin = 0.0` means “touch each atom’s centre”.
    pub fn new(atom_posits: &[Vec3], map: &DensityMap, margin: f64) -> Self {
        let hdr = &map.hdr;
        let inner = &hdr.inner;
        let cell = &inner.cell;

        // Atom bounds in fractional coords, relative to map origin
        let mut min_r = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_r = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for p in atom_posits {
            // Cartesian to absolute fractional
            let mut f = cell.cartesian_to_fractional(*p);
            // Shift so that origin_frac becomes (0,0,0)
            f -= map.origin_frac;

            // keep unwrapped values (they can be < 0 or > 1)
            min_r = Vec3::new(min_r.x.min(f.x), min_r.y.min(f.y), min_r.z.min(f.z));
            max_r = Vec3::new(max_r.x.max(f.x), max_r.y.max(f.y), max_r.z.max(f.z));
        }

        // Extra margin in fractional units
        let margin_r = Vec3::new(margin / cell.a, margin / cell.b, margin / cell.c);
        min_r -= margin_r;
        max_r += margin_r;

        // Convert fractional to voxel indices
        let to_idx = |fr: f64, n: i32| -> isize { (fr * n as f64 - 0.5).floor() as isize };

        let lo_i = [
            to_idx(min_r.x, inner.mx),
            to_idx(min_r.y, inner.my),
            to_idx(min_r.z, inner.mz),
        ];
        let hi_i = [
            to_idx(max_r.x, inner.mx),
            to_idx(max_r.y, inner.my),
            to_idx(max_r.z, inner.mz),
        ];

        // inclusive → dims       (now guaranteed hi_i ≥ lo_i)
        let dims = [
            (hi_i[0] - lo_i[0] + 1) as usize,
            (hi_i[1] - lo_i[1] + 1) as usize,
            (hi_i[2] - lo_i[2] + 1) as usize,
        ];

        let lo_frac = Vec3::new(
            (lo_i[0] as f64 + 0.5) / inner.mx as f64,
            (lo_i[1] as f64 + 0.5) / inner.my as f64,
            (lo_i[2] as f64 + 0.5) / inner.mz as f64,
        ) + map.origin_frac; // back to absolute fractional

        let origin_cart = cell.fractional_to_cartesian(lo_frac);

        // Voxel step vectors in Å
        let step = [
            cell.a / inner.mx as f64,
            cell.b / inner.my as f64,
            cell.c / inner.mz as f64,
        ];

        let mut data = Vec::with_capacity(dims[0] * dims[1] * dims[2]);

        for kz in 0..dims[2] {
            for ky in 0..dims[1] {
                for kx in 0..dims[0] {
                    let idx_c = [
                        lo_i[0] + kx as isize,
                        lo_i[1] + ky as isize,
                        lo_i[2] + kz as isize,
                    ];

                    // Crystallographic → Cartesian center of this voxel
                    let frac = map.origin_frac
                        + Vec3::new(
                            (idx_c[0] as f64 + 0.5) / inner.mx as f64,
                            (idx_c[1] as f64 + 0.5) / inner.my as f64,
                            (idx_c[2] as f64 + 0.5) / inner.mz as f64,
                        );
                    let cart = cell.fractional_to_cartesian(frac);

                    let density = map.density_at_point_trilinear(cart);
                    let dens_sig = map.density_to_sig(density);
                    data.push(dens_sig);
                }
            }
        }

        Self {
            origin_cart,
            step,
            dims,
            data,
        }
    }
}
