use std::cmp::Ordering;

use crate::{
    binaural::BinauralWriter,
    error::AppError,
    hrir::Speaker,
    object::{ObjectTrim, ObjectZone},
};

/// Equal-power interpolation over the directional channels present in an HRIR.
pub(crate) struct SpatialPanner {
    buses: Vec<PanningBus>,
    output_bus_count: usize,
    lfe_bus: Option<usize>,
    stereo_trim_configuration: bool,
}

#[must_use]
pub(crate) fn direct_stereo_gains(speaker: Speaker) -> [f32; 2] {
    match speaker {
        Speaker::FrontCenter => [std::f32::consts::FRAC_1_SQRT_2; 2],
        Speaker::Lfe => [0.5; 2],
        Speaker::FrontLeft
        | Speaker::RearLeft
        | Speaker::SideLeft
        | Speaker::WideLeft
        | Speaker::TopFrontLeft
        | Speaker::TopSideLeft
        | Speaker::TopRearLeft => [1.0, 0.0],
        Speaker::FrontRight
        | Speaker::RearRight
        | Speaker::SideRight
        | Speaker::WideRight
        | Speaker::TopFrontRight
        | Speaker::TopSideRight
        | Speaker::TopRearRight => [0.0, 1.0],
        Speaker::RearCenter | Speaker::TopFrontCenter | Speaker::TopRearCenter => {
            [std::f32::consts::FRAC_1_SQRT_2; 2]
        }
    }
}

struct PanningBus {
    index: usize,
    speaker: Speaker,
    direction: [f32; 3],
    named: bool,
}

impl SpatialPanner {
    pub(crate) fn new(writer: &BinauralWriter) -> Result<Self, AppError> {
        let routes = writer.panning_routes().collect::<Vec<_>>();
        let has_named = |speaker| routes.iter().any(|route| route.speaker == Some(speaker));
        let global_num_mid = [
            Speaker::SideLeft,
            Speaker::SideRight,
            Speaker::RearLeft,
            Speaker::RearRight,
            Speaker::RearCenter,
            Speaker::WideLeft,
            Speaker::WideRight,
        ]
        .into_iter()
        .filter(|speaker| has_named(*speaker))
        .count();
        let global_num_top = [
            Speaker::TopFrontLeft,
            Speaker::TopFrontCenter,
            Speaker::TopFrontRight,
            Speaker::TopSideLeft,
            Speaker::TopSideRight,
            Speaker::TopRearLeft,
            Speaker::TopRearCenter,
            Speaker::TopRearRight,
        ]
        .into_iter()
        .filter(|speaker| has_named(*speaker))
        .count();
        let mut buses = routes
            .into_iter()
            .map(|route| {
                let named = route.speaker.is_some();
                PanningBus {
                    index: route.index,
                    speaker: route
                        .speaker
                        .unwrap_or_else(|| closest_reference_speaker(route.direction)),
                    direction: normalized(route.direction),
                    named,
                }
            })
            .collect::<Vec<_>>();
        buses.sort_by_key(|bus| bus.index);
        if buses.is_empty() {
            return Err(AppError::InvalidHrir(
                "HRIR has no directional channels for spatial rendering".into(),
            ));
        }
        Ok(Self {
            buses,
            output_bus_count: writer.bus_count(),
            lfe_bus: writer.bus(Speaker::Lfe),
            stereo_trim_configuration: global_num_mid == 0 && global_num_top == 0,
        })
    }

    #[must_use]
    pub(crate) const fn bus_count(&self) -> usize {
        self.output_bus_count
    }

    #[must_use]
    pub(crate) fn lfe_gains(&self, gain: f32) -> Vec<f32> {
        let mut gains = vec![0.0; self.output_bus_count];
        if let Some(bus) = self.lfe_bus {
            gains[bus] = gain;
        }
        gains
    }

    #[must_use]
    pub(crate) fn resultant_direction(&self, gains: &[f32], fallback: [f32; 3]) -> [f32; 3] {
        let mut result = [0.0; 3];
        for bus in &self.buses {
            // VBAP coefficients reconstruct the source vector linearly. An
            // energy centroid (squared coefficients) is appropriate for
            // incoherent loudspeakers, but it bends the direction when those
            // routes are collapsed back to one coherent continuous HRTF.
            let weight = gains.get(bus.index).copied().unwrap_or(0.0);
            for (axis, direction) in result.iter_mut().zip(bus.direction) {
                *axis += direction * weight;
            }
        }
        let length = result.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length > f32::EPSILON {
            result.map(|value| value / length)
        } else {
            normalized(fallback)
        }
    }

    #[must_use]
    pub(crate) fn untrimmed_point_gains(&self, direction: [f32; 3], gain: f32) -> Vec<f32> {
        let permitted = self.buses.iter().collect::<Vec<_>>();
        let mut gains = point_gains(direction, &permitted, self.output_bus_count);
        normalize_power(&mut gains);
        for output_gain in &mut gains {
            *output_gain *= gain;
        }
        gains
    }

    /// Pans one source position to the available virtual speakers.
    ///
    /// A non-zero `size` integrates point pans over the signalled rectangular
    /// extent. `gain` is applied after equal-power panning.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata contains NaN or infinity.
    #[allow(clippy::cast_precision_loss)] // The tiny HRIR bus count is exactly represented.
    #[allow(clippy::too_many_arguments)] // These are independent object-metadata dimensions.
    pub(crate) fn gains(
        &self,
        position: [f32; 3],
        size: [f32; 3],
        gain: f32,
        snap: bool,
        zone: ObjectZone,
        elevation: bool,
        divergence: f32,
        trim: ObjectTrim,
        speaker_anchored: bool,
    ) -> Result<Vec<f32>, AppError> {
        if position
            .iter()
            .chain(size.iter())
            .any(|value| !value.is_finite())
            || !gain.is_finite()
            || !divergence.is_finite()
        {
            return Err(AppError::Render(
                "spatial metadata contains a non-finite value".into(),
            ));
        }

        let has_rear = self.buses.iter().any(|bus| {
            bus.named
                && matches!(
                    bus.speaker,
                    Speaker::RearLeft | Speaker::RearRight | Speaker::RearCenter
                )
        });
        let mut permitted = self
            .buses
            .iter()
            .filter(|bus| panning_bus_permitted(bus, zone, elevation, has_rear))
            .collect::<Vec<_>>();
        if permitted.is_empty() {
            permitted = self
                .buses
                .iter()
                .filter(|bus| elevation || is_listener_plane(bus.direction))
                .collect();
        }
        if permitted.is_empty() {
            permitted = self.buses.iter().collect();
        }

        let mut position = position;
        let mut size = size;
        let mut divergence = divergence.clamp(0.0, 1.0);
        if !elevation {
            position[2] = 0.0;
            size[2] = 0.0;
        }
        if snap {
            position = snapped_position(position, zone, elevation);
            size = [0.0; 3];
            divergence = 0.0;
        }

        let pan = |at| {
            if size.iter().all(|value| value.abs() <= f32::EPSILON) {
                point_gains(at, &permitted, self.output_bus_count)
            } else {
                extent_gains(at, size, &permitted, self.output_bus_count)
            }
        };
        let mut gains = if divergence <= f32::EPSILON {
            pan(position)
        } else {
            let mut left = position;
            let mut right = position;
            // The normative positions are X +/- divergence / 2 in room X
            // coordinates [0, 1]. Renderer X is [-1, 1], so the affine
            // coordinate conversion doubles that offset.
            left[0] = (left[0] - divergence).max(-1.0);
            right[0] = (right[0] + divergence).min(1.0);
            let left = pan(left);
            let right = pan(right);
            left.into_iter()
                .zip(right)
                .map(|(left, right)| (0.5 * left.mul_add(left, right * right)).sqrt())
                .collect()
        };
        normalize_power(&mut gains);
        let num_mid = permitted
            .iter()
            .filter(|bus| bus.named && is_mid(bus.speaker))
            .count();
        let num_top = permitted
            .iter()
            .filter(|bus| bus.named && is_height(bus.speaker))
            .count();
        let trim_configuration = mid_trim_category(num_mid) + 3 * top_trim_category(num_top);
        let trim = trim.settings(trim_configuration);
        for bus in &self.buses {
            gains[bus.index] *= trim.balance_gain(bus.speaker, !is_listener_plane(bus.direction));
        }
        let object_gain = gain
            * trim.position_gain(
                position,
                speaker_anchored,
                self.stereo_trim_configuration,
                num_mid,
                num_top,
            );
        for output_gain in &mut gains {
            *output_gain *= object_gain;
        }
        Ok(gains)
    }
}

fn closest_reference_speaker(direction: [f32; 3]) -> Speaker {
    const REFERENCE: [Speaker; 18] = [
        Speaker::FrontLeft,
        Speaker::FrontRight,
        Speaker::FrontCenter,
        Speaker::RearLeft,
        Speaker::RearRight,
        Speaker::RearCenter,
        Speaker::SideLeft,
        Speaker::SideRight,
        Speaker::WideLeft,
        Speaker::WideRight,
        Speaker::TopFrontLeft,
        Speaker::TopFrontCenter,
        Speaker::TopFrontRight,
        Speaker::TopSideLeft,
        Speaker::TopSideRight,
        Speaker::TopRearLeft,
        Speaker::TopRearCenter,
        Speaker::TopRearRight,
    ];
    let direction = normalized(direction);
    REFERENCE
        .into_iter()
        .max_by(|left, right| {
            dot(direction, left.position())
                .partial_cmp(&dot(direction, right.position()))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(Speaker::FrontCenter)
}

#[derive(Clone, Copy)]
enum SnapZone {
    Front,
    Center,
    Side,
    Back,
    Wide,
    Elevated,
}

#[derive(Clone, Copy)]
struct SnapTarget {
    position: [f32; 3],
    zone: SnapZone,
}

// Normative 22.2 snap locations in metadata room coordinates.
const SNAP_TARGETS: [SnapTarget; 22] = [
    SnapTarget {
        position: [0.5, 0.0, 0.0],
        zone: SnapZone::Center,
    },
    SnapTarget {
        position: [0.5, 0.0, -1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.25, 0.75, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.5, 0.75, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.75, 0.75, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.25, 0.5, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.5, 0.5, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.75, 0.5, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.25, 0.25, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.5, 0.25, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.75, 0.25, 1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [0.0, 1.0, 0.0],
        zone: SnapZone::Back,
    },
    SnapTarget {
        position: [0.5, 1.0, 0.0],
        zone: SnapZone::Back,
    },
    SnapTarget {
        position: [1.0, 1.0, 0.0],
        zone: SnapZone::Back,
    },
    SnapTarget {
        position: [0.0, 0.5, 0.0],
        zone: SnapZone::Side,
    },
    SnapTarget {
        position: [1.0, 0.5, 0.0],
        zone: SnapZone::Side,
    },
    SnapTarget {
        position: [0.0, 0.2929, 0.0],
        zone: SnapZone::Wide,
    },
    SnapTarget {
        position: [1.0, 0.2929, 0.0],
        zone: SnapZone::Wide,
    },
    SnapTarget {
        position: [0.0, 0.0, 0.0],
        zone: SnapZone::Front,
    },
    SnapTarget {
        position: [1.0, 0.0, 0.0],
        zone: SnapZone::Front,
    },
    SnapTarget {
        position: [0.0, 0.0, -1.0],
        zone: SnapZone::Elevated,
    },
    SnapTarget {
        position: [1.0, 0.0, -1.0],
        zone: SnapZone::Elevated,
    },
];

fn snapped_position(position: [f32; 3], zone: ObjectZone, elevation: bool) -> [f32; 3] {
    let room = [
        position[0].mul_add(0.5, 0.5),
        (-position[1]).mul_add(0.5, 0.5),
        position[2],
    ];
    let target = SNAP_TARGETS
        .into_iter()
        .filter(|target| snap_target_permitted(target.zone, zone, elevation))
        .min_by(|left, right| {
            snap_distance(room, left.position).total_cmp(&snap_distance(room, right.position))
        })
        .expect("every zone retains at least one snap target");
    [
        target.position[0].mul_add(2.0, -1.0),
        (-target.position[1]).mul_add(2.0, 1.0),
        target.position[2],
    ]
}

fn snap_distance(position: [f32; 3], target: [f32; 3]) -> f32 {
    (position[0] - target[0]).powi(2) / 16.0
        + 4.0 * (position[1] - target[1]).powi(2)
        + 32.0 * (position[2] - target[2]).powi(2)
}

const fn snap_target_permitted(kind: SnapZone, zone: ObjectZone, elevation: bool) -> bool {
    match kind {
        SnapZone::Elevated => elevation,
        SnapZone::Front => matches!(
            zone,
            ObjectZone::All | ObjectZone::NoBack | ObjectZone::NoSide | ObjectZone::Screen
        ),
        SnapZone::Center => !matches!(zone, ObjectZone::Surround),
        SnapZone::Side => matches!(
            zone,
            ObjectZone::All | ObjectZone::NoBack | ObjectZone::Surround
        ),
        SnapZone::Back => matches!(
            zone,
            ObjectZone::All | ObjectZone::NoSide | ObjectZone::CentreAndBack | ObjectZone::Surround
        ),
        SnapZone::Wide => matches!(zone, ObjectZone::All | ObjectZone::NoBack),
    }
}

fn snapped_gains(position: [f32; 3], buses: &[&PanningBus], output_count: usize) -> Vec<f32> {
    let direction = source_direction(position);
    let closest = buses
        .iter()
        .min_by(|left, right| {
            dot(direction, left.direction)
                .partial_cmp(&dot(direction, right.direction))
                .unwrap_or(Ordering::Equal)
                .reverse()
                .then_with(|| left.index.cmp(&right.index))
        })
        .expect("permitted panning buses are non-empty");
    let mut gains = vec![0.0; output_count];
    gains[closest.index] = 1.0;
    gains
}

fn point_gains(position: [f32; 3], buses: &[&PanningBus], output_count: usize) -> Vec<f32> {
    let mut direction = source_direction(position);
    // OAMD can place objects below the listener. Profiles without measured
    // lower routes project those positions onto the middle layer instead of
    // leaking them into an upper route.
    if buses.iter().all(|bus| bus.direction[2] >= -1e-4) {
        direction[2] = direction[2].max(0.0);
    }
    direction = normalized(direction);

    if let Some(exact) = buses
        .iter()
        .find(|bus| dot(direction, bus.direction) >= 1.0 - 1e-7)
    {
        let mut gains = vec![0.0; output_count];
        gains[exact.index] = 1.0;
        return gains;
    }
    if direction[2] <= 1e-6 {
        if let Some(gains) = horizontal_vbap(direction, buses, output_count) {
            return gains;
        }
        let ground = buses
            .iter()
            .filter(|bus| is_listener_plane(bus.direction))
            .copied()
            .collect::<Vec<_>>();
        if !ground.is_empty() {
            // A partial listener-level layout has no valid VBAP pair outside
            // its angular span. Hold the nearest edge instead of leaking a
            // ground-plane object into an overhead channel.
            return snapped_gains(direction, &ground, output_count);
        }
    }
    if let Some(gains) = triplet_vbap(direction, buses, output_count) {
        return gains;
    }

    inverse_angle_fallback(direction, buses, output_count)
}

fn horizontal_vbap(
    direction: [f32; 3],
    buses: &[&PanningBus],
    output_count: usize,
) -> Option<Vec<f32>> {
    let mut ground = buses
        .iter()
        .filter(|bus| is_listener_plane(bus.direction))
        .copied()
        .collect::<Vec<_>>();
    if ground.len() < 2 {
        return None;
    }
    ground.sort_by(|left, right| {
        azimuth(left.direction)
            .partial_cmp(&azimuth(right.direction))
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.index.cmp(&right.index))
    });
    for index in 0..ground.len() {
        let left = ground[index];
        let right = ground[(index + 1) % ground.len()];
        let determinant =
            left.direction[0].mul_add(right.direction[1], -right.direction[0] * left.direction[1]);
        if determinant.abs() <= 1e-6 {
            continue;
        }
        let left_gain = direction[0]
            .mul_add(right.direction[1], -right.direction[0] * direction[1])
            / determinant;
        let right_gain = left.direction[0].mul_add(direction[1], -direction[0] * left.direction[1])
            / determinant;
        if left_gain >= -1e-5 && right_gain >= -1e-5 {
            let mut gains = vec![0.0; output_count];
            gains[left.index] = left_gain.max(0.0);
            gains[right.index] = right_gain.max(0.0);
            normalize_power(&mut gains);
            return Some(gains);
        }
    }
    None
}

fn triplet_vbap(
    direction: [f32; 3],
    buses: &[&PanningBus],
    output_count: usize,
) -> Option<Vec<f32>> {
    const EXHAUSTIVE_LIMIT: usize = 24;
    const LOCAL_CANDIDATES: usize = 12;

    let mut candidates = buses.to_vec();
    if candidates.len() > EXHAUSTIVE_LIMIT {
        candidates.sort_by(|left, right| {
            dot(direction, right.direction).total_cmp(&dot(direction, left.direction))
        });
        candidates.truncate(LOCAL_CANDIDATES);
    }
    let mut best = containing_triplet(direction, &candidates);
    if best.is_none() && candidates.len() != buses.len() {
        best = containing_triplet(direction, buses);
    }
    let best = best?;
    let (indices, coefficients) = best;
    let mut gains = vec![0.0; output_count];
    for (index, coefficient) in indices.into_iter().zip(coefficients) {
        gains[index] = coefficient.max(0.0);
    }
    normalize_power(&mut gains);
    Some(gains)
}

fn containing_triplet(
    direction: [f32; 3],
    buses: &[&PanningBus],
) -> Option<([usize; 3], [f32; 3])> {
    let mut best: Option<([usize; 3], [f32; 3], f32)> = None;
    for first in 0..buses.len() {
        for second in first + 1..buses.len() {
            for third in second + 1..buses.len() {
                let a = buses[first].direction;
                let b = buses[second].direction;
                let c = buses[third].direction;
                let determinant = dot(a, cross(b, c));
                if determinant.abs() <= 1e-6 {
                    continue;
                }
                let coefficients = [
                    dot(direction, cross(b, c)) / determinant,
                    dot(a, cross(direction, c)) / determinant,
                    dot(a, cross(b, direction)) / determinant,
                ];
                if coefficients.iter().any(|gain| *gain < -1e-5) {
                    continue;
                }
                // Maximising the vertex dot products chooses the most local
                // enclosing triplet without evaluating inverse trigonometry.
                let locality = dot(direction, a) + dot(direction, b) + dot(direction, c);
                if best
                    .as_ref()
                    .is_none_or(|(_, _, best_locality)| locality > *best_locality)
                {
                    best = Some((
                        [buses[first].index, buses[second].index, buses[third].index],
                        coefficients,
                        locality,
                    ));
                }
            }
        }
    }
    best.map(|(indices, coefficients, _)| (indices, coefficients))
}

fn inverse_angle_fallback(
    direction: [f32; 3],
    buses: &[&PanningBus],
    output_count: usize,
) -> Vec<f32> {
    let mut nearest = buses
        .iter()
        .map(|bus| {
            let dot = dot(direction, bus.direction).clamp(-1.0, 1.0);
            (bus.index, dot)
        })
        .collect::<Vec<_>>();
    nearest.sort_by(|left, right| right.1.total_cmp(&left.1));
    nearest.truncate(3.min(nearest.len()));

    let mut gains = vec![0.0; output_count];
    for (bus, dot) in nearest {
        // 2(1-cos(theta)) is the squared chord distance and closely tracks
        // theta² over the local angles used by the fallback.
        gains[bus] = 1.0 / (0.002_5 + 2.0 * (1.0 - dot));
    }
    normalize_power(&mut gains);
    gains
}

fn azimuth(direction: [f32; 3]) -> f32 {
    direction[0].atan2(direction[1])
}

#[allow(clippy::cast_precision_loss)] // The bounded quadrature count is exactly represented.
fn extent_gains(
    position: [f32; 3],
    size: [f32; 3],
    buses: &[&PanningBus],
    output_count: usize,
) -> Vec<f32> {
    const ABSCISSA: f32 = 0.774_596_7;
    const NODES: [(f32, f32); 3] = [
        (-ABSCISSA, 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (ABSCISSA, 5.0 / 9.0),
    ];
    let axes = size.map(|extent| {
        if extent.abs() <= f32::EPSILON {
            &NODES[1..2]
        } else {
            &NODES[..]
        }
    });
    let mut energy = vec![0.0; output_count];
    let mut total_weight = 0.0;
    for &(x, x_weight) in axes[0] {
        for &(y, y_weight) in axes[1] {
            for &(z, z_weight) in axes[2] {
                let weight = x_weight * y_weight * z_weight;
                let sample_position = [
                    position[0].mul_add(1.0, x * size[0]),
                    position[1].mul_add(1.0, y * size[1]),
                    position[2].mul_add(1.0, z * size[2]),
                ];
                let point = point_gains(sample_position, buses, output_count);
                for (sum, gain) in energy.iter_mut().zip(point) {
                    *sum += weight * gain * gain;
                }
                total_weight += weight;
            }
        }
    }
    for gain in &mut energy {
        *gain = (*gain / total_weight).sqrt();
    }
    normalize_power(&mut energy);
    energy
}

fn source_direction(position: [f32; 3]) -> [f32; 3] {
    if magnitude_squared(position) < 1e-8 {
        [0.0, 1.0, 0.0]
    } else {
        normalized(position)
    }
}

fn panning_bus_permitted(
    bus: &PanningBus,
    zone: ObjectZone,
    elevation: bool,
    has_rear: bool,
) -> bool {
    if is_listener_plane(bus.direction) {
        bus_permitted(bus.speaker, zone, elevation, has_rear)
    } else {
        elevation
    }
}

fn is_listener_plane(direction: [f32; 3]) -> bool {
    direction[2].abs() <= 1e-4
}

fn bus_permitted(speaker: Speaker, zone: ObjectZone, elevation: bool, has_rear: bool) -> bool {
    if is_height(speaker) {
        return elevation;
    }
    let screen = matches!(
        speaker,
        Speaker::FrontLeft | Speaker::FrontRight | Speaker::FrontCenter
    );
    let surround = matches!(
        speaker,
        Speaker::SideLeft
            | Speaker::SideRight
            | Speaker::RearLeft
            | Speaker::RearRight
            | Speaker::RearCenter
    );
    let distinct_back = matches!(
        speaker,
        Speaker::RearLeft | Speaker::RearRight | Speaker::RearCenter
    );
    let centre_and_back =
        distinct_back || (!has_rear && matches!(speaker, Speaker::SideLeft | Speaker::SideRight));
    // In the normative speaker-zone table, wides are always in the side
    // zone. Ls/Rs are excluded by the side mask only when distinct back
    // speakers exist; the back mask has no effect on a 5.x layout.
    let side = matches!(speaker, Speaker::WideLeft | Speaker::WideRight)
        || (has_rear && matches!(speaker, Speaker::SideLeft | Speaker::SideRight));
    match zone {
        ObjectZone::All => true,
        ObjectZone::NoBack => !distinct_back,
        ObjectZone::NoSide => !side,
        ObjectZone::CentreAndBack => speaker == Speaker::FrontCenter || centre_and_back,
        ObjectZone::Screen => screen,
        ObjectZone::Surround => surround,
    }
}

const fn is_height(speaker: Speaker) -> bool {
    matches!(
        speaker,
        Speaker::TopFrontLeft
            | Speaker::TopFrontCenter
            | Speaker::TopFrontRight
            | Speaker::TopSideLeft
            | Speaker::TopSideRight
            | Speaker::TopRearLeft
            | Speaker::TopRearCenter
            | Speaker::TopRearRight
    )
}

const fn is_mid(speaker: Speaker) -> bool {
    matches!(
        speaker,
        Speaker::SideLeft
            | Speaker::SideRight
            | Speaker::RearLeft
            | Speaker::RearRight
            | Speaker::RearCenter
            | Speaker::WideLeft
            | Speaker::WideRight
    )
}

const fn mid_trim_category(count: usize) -> usize {
    match count {
        0 => 0,
        1..=3 => 1,
        _ => 2,
    }
}

const fn top_trim_category(count: usize) -> usize {
    match count {
        0 => 0,
        1 | 2 => 1,
        _ => 2,
    }
}

fn normalized(value: [f32; 3]) -> [f32; 3] {
    let inverse = magnitude_squared(value).sqrt().recip();
    [value[0] * inverse, value[1] * inverse, value[2] * inverse]
}

const fn magnitude_squared(value: [f32; 3]) -> f32 {
    dot(value, value)
}

const fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

const fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize_power(gains: &mut [f32]) {
    let power = gains.iter().map(|gain| gain * gain).sum::<f32>().sqrt();
    if power > f32::EPSILON {
        for gain in gains {
            *gain /= power;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PanningBus, SpatialPanner, bus_permitted, containing_triplet, direct_stereo_gains, dot,
        is_height, is_mid, normalize_power, normalized, point_gains, snapped_position,
        triplet_vbap,
    };
    use crate::{
        binaural::BinauralWriter,
        hrir::{DirectionalHrir, HrirChannel, HrirSet, Speaker},
        object::{ObjectTrim, ObjectTrimMode, ObjectTrimSettings, ObjectZone},
    };

    #[test]
    fn panning_gains_are_equal_power() {
        let mut gains = [1.0, 2.0, 3.0];
        normalize_power(&mut gains);
        let power: f32 = gains.iter().map(|gain| gain * gain).sum();
        assert!((power - 1.0).abs() < 1e-6);
    }

    #[test]
    fn local_triplet_search_matches_exhaustive_search_on_a_dense_sphere() {
        let directions = fibonacci_sphere(66);
        let buses = directions
            .into_iter()
            .enumerate()
            .map(|(index, direction)| PanningBus {
                index,
                speaker: Speaker::FrontCenter,
                direction,
                named: false,
            })
            .collect::<Vec<_>>();
        let bus_refs = buses.iter().collect::<Vec<_>>();

        for direction in fibonacci_sphere(257) {
            let local = triplet_vbap(direction, &bus_refs, buses.len()).unwrap();
            let (indices, coefficients) = containing_triplet(direction, &bus_refs).unwrap();
            let mut exhaustive = vec![0.0; buses.len()];
            for (index, coefficient) in indices.into_iter().zip(coefficients) {
                exhaustive[index] = coefficient.max(0.0);
            }
            normalize_power(&mut exhaustive);
            let maximum_difference = local
                .iter()
                .zip(exhaustive)
                .map(|(local, exhaustive)| (local - exhaustive).abs())
                .fold(0.0, f32::max);
            assert!(
                maximum_difference < 1e-5,
                "local VBAP differed from exhaustive search by {maximum_difference}"
            );
        }
    }

    #[test]
    fn continuous_resultant_does_not_inherit_route_lattice_error() {
        let directions = fibonacci_sphere(66);
        let buses = directions
            .into_iter()
            .enumerate()
            .map(|(index, direction)| PanningBus {
                index,
                speaker: Speaker::FrontCenter,
                direction,
                named: false,
            })
            .collect::<Vec<_>>();
        let panner = SpatialPanner {
            output_bus_count: buses.len(),
            buses,
            lfe_bus: None,
            stereo_trim_configuration: false,
        };
        let bus_refs = panner.buses.iter().collect::<Vec<_>>();
        let mut maximum_error = 0.0_f32;
        for direction in fibonacci_sphere(4_096) {
            let gains = point_gains(direction, &bus_refs, panner.output_bus_count);
            let rendered = panner.resultant_direction(&gains, direction);
            let error = dot(direction, rendered)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            maximum_error = maximum_error.max(error);
        }
        assert!(
            maximum_error < 0.1,
            "continuous direction inherited {maximum_error:.3}° of route-lattice error"
        );
    }

    #[test]
    fn direct_stereo_pan_preserves_power() {
        let left = direct_stereo_gains(Speaker::FrontLeft);
        assert!((left[0] - 1.0).abs() < f32::EPSILON);
        assert!(left[1].abs() < f32::EPSILON);
        let center = direct_stereo_gains(Speaker::FrontCenter);
        assert!((center[0].mul_add(center[0], center[1] * center[1]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn exact_speaker_position_has_no_panning_leakage() {
        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::SideLeft,
            Speaker::SideRight,
        ]);
        let gains = panner
            .gains(
                Speaker::FrontLeft.position(),
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                true,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        assert_eq!(gains, [1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn trim_is_a_scalar_for_dynamic_objects_and_unity_for_beds() {
        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::SideLeft,
            Speaker::SideRight,
        ]);
        let trim = ObjectTrim::uniform(
            false,
            ObjectTrimSettings {
                mode: ObjectTrimMode::Custom,
                surround_db: -6.0,
                ..ObjectTrimSettings::default()
            },
        );
        let dynamic = panner
            .gains(
                Speaker::SideLeft.position(),
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                true,
                0.0,
                trim,
                false,
            )
            .unwrap();
        let bed = panner
            .gains(
                Speaker::SideLeft.position(),
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                true,
                0.0,
                trim,
                true,
            )
            .unwrap();

        assert!((dynamic[3] - 10_f32.powf(-6.0 / 20.0)).abs() < 1e-6);
        assert!((bed[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn zone_constraints_select_the_matching_trim_configuration() {
        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::SideLeft,
            Speaker::SideRight,
        ]);
        let mut configurations = [ObjectTrimSettings::default(); 9];
        configurations[0] = ObjectTrimSettings {
            mode: ObjectTrimMode::Custom,
            surround_db: -12.0,
            ..ObjectTrimSettings::default()
        };
        let gains = panner
            .gains(
                [0.0, -1.0, 0.0],
                [0.0; 3],
                1.0,
                false,
                ObjectZone::Screen,
                true,
                0.0,
                ObjectTrim::from_configurations(false, configurations),
                false,
            )
            .unwrap();
        let level = gains.iter().map(|gain| gain * gain).sum::<f32>().sqrt();

        assert!((level - 10_f32.powf(-12.0 / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn measured_directional_route_is_used_between_reference_speakers() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("directional.wav");
        let impulse = |speaker| HrirChannel {
            speaker,
            left: vec![1.0],
            right: vec![1.0],
        };
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![impulse(Speaker::FrontLeft), impulse(Speaker::FrontRight)],
            directional: vec![DirectionalHrir {
                direction: Speaker::FrontCenter.position(),
                left: vec![1.0],
                right: vec![1.0],
            }],
        };
        let writer = BinauralWriter::new_raw(
            &output,
            &hrir,
            None,
            0.0,
            [Speaker::FrontLeft, Speaker::FrontRight],
        )
        .unwrap();
        let panner = SpatialPanner::new(&writer).unwrap();
        let gains = panner
            .gains(
                Speaker::FrontCenter.position(),
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                true,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        let active = gains
            .iter()
            .enumerate()
            .filter(|(_, gain)| **gain > 0.999)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(active, [3]);
    }

    #[test]
    fn measured_lower_hemisphere_route_is_used_only_when_elevation_is_enabled() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("lower-directional.wav");
        let impulse = |speaker| HrirChannel {
            speaker,
            left: vec![1.0],
            right: vec![1.0],
        };
        let lower = normalized([0.0, 1.0, -1.0]);
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![impulse(Speaker::FrontLeft), impulse(Speaker::FrontRight)],
            directional: vec![DirectionalHrir {
                direction: lower,
                left: vec![1.0],
                right: vec![1.0],
            }],
        };
        let writer = BinauralWriter::new_raw(
            &output,
            &hrir,
            None,
            0.0,
            [Speaker::FrontLeft, Speaker::FrontRight],
        )
        .unwrap();
        let panner = SpatialPanner::new(&writer).unwrap();
        let elevated = panner
            .gains(
                lower,
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                true,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        let flattened = panner
            .gains(
                lower,
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                false,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();

        assert!((elevated[3] - 1.0).abs() < f32::EPSILON);
        assert!(flattened[3].abs() < f32::EPSILON);
        assert!((flattened[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((flattened[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn snap_uses_exactly_one_permitted_speaker() {
        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontRight,
            Speaker::RearLeft,
            Speaker::RearRight,
        ]);
        let gains = panner
            .gains(
                Speaker::RearLeft.position(),
                [1.0; 3],
                0.75,
                true,
                ObjectZone::NoBack,
                true,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        assert_eq!(gains.iter().filter(|gain| **gain != 0.0).count(), 1);
        assert!((gains[0] - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn snap_uses_the_weighted_room_grid_and_clears_divergence() {
        let snapped = snapped_position([0.9, 0.9, 0.9], ObjectZone::All, true);
        for (actual, expected) in snapped.into_iter().zip([0.5, 0.5, 1.0]) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }

        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
        ]);
        let gains = panner
            .gains(
                Speaker::FrontCenter.position(),
                [1.0; 3],
                1.0,
                true,
                ObjectZone::All,
                true,
                1.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        assert_eq!(gains, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn disabled_elevation_never_routes_to_a_height_bus() {
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontRight,
            Speaker::TopFrontLeft,
            Speaker::TopFrontRight,
        ];
        let panner = panner(&speakers);
        let gains = panner
            .gains(
                [0.0, 1.0, 1.0],
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                false,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        for (speaker, gain) in speakers.into_iter().zip(&gains) {
            if is_height(speaker) {
                assert!(gain.abs() < f32::EPSILON);
            }
        }
        assert!((gains[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((gains[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn width_spreads_sideways_without_spreading_upward() {
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::TopFrontCenter,
        ];
        let panner = panner(&speakers);
        let gains = panner
            .gains(
                Speaker::FrontCenter.position(),
                [1.0, 0.0, 0.0],
                1.0,
                false,
                ObjectZone::All,
                true,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        assert!(gains[0] > 0.0);
        assert!(gains[1] > 0.0);
        assert!(gains[2] > 0.0);
        assert!(gains[3].abs() < f32::EPSILON);
    }

    #[test]
    fn divergence_is_scaled_from_normative_room_coordinates() {
        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
        ]);
        let gains = panner
            .gains(
                Speaker::FrontCenter.position(),
                [0.0; 3],
                1.0,
                false,
                ObjectZone::All,
                true,
                1.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        assert!((gains[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(gains[1].abs() < f32::EPSILON);
        assert!((gains[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn horizontal_constraints_do_not_disable_the_height_zone() {
        let panner = panner(&[
            Speaker::FrontLeft,
            Speaker::FrontRight,
            Speaker::TopRearLeft,
            Speaker::TopRearRight,
        ]);
        let gains = panner
            .gains(
                Speaker::TopRearLeft.position(),
                [0.0; 3],
                1.0,
                false,
                ObjectZone::Screen,
                true,
                0.0,
                ObjectTrim::default(),
                false,
            )
            .unwrap();
        assert_eq!(gains, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn speaker_zone_membership_tracks_the_render_layout() {
        assert!(!bus_permitted(
            Speaker::WideLeft,
            ObjectZone::NoSide,
            true,
            false,
        ));
        assert!(bus_permitted(
            Speaker::SideLeft,
            ObjectZone::NoSide,
            true,
            false,
        ));
        assert!(bus_permitted(
            Speaker::SideLeft,
            ObjectZone::NoBack,
            true,
            false,
        ));

        assert!(!bus_permitted(
            Speaker::SideLeft,
            ObjectZone::NoSide,
            true,
            true,
        ));
        assert!(bus_permitted(
            Speaker::SideLeft,
            ObjectZone::NoBack,
            true,
            true,
        ));
        assert!(bus_permitted(
            Speaker::RearLeft,
            ObjectZone::Surround,
            true,
            true,
        ));
    }

    fn panner(speakers: &[Speaker]) -> SpatialPanner {
        SpatialPanner {
            buses: speakers
                .iter()
                .enumerate()
                .map(|(index, speaker)| PanningBus {
                    index,
                    speaker: *speaker,
                    direction: normalized(speaker.position()),
                    named: true,
                })
                .collect(),
            output_bus_count: speakers.len(),
            lfe_bus: None,
            stereo_trim_configuration: !speakers
                .iter()
                .any(|speaker| is_mid(*speaker) || is_height(*speaker)),
        }
    }

    #[allow(clippy::cast_precision_loss)] // Test grid sizes are tiny.
    fn fibonacci_sphere(count: usize) -> Vec<[f32; 3]> {
        let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        (0..count)
            .map(|index| {
                let z = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
                let radius = (1.0 - z * z).sqrt();
                let azimuth = golden_angle * index as f32;
                [azimuth.sin() * radius, azimuth.cos() * radius, z]
            })
            .collect()
    }
}
