use crate::hrir::Speaker;

const REFERENCE_SCREEN_ASPECT_RATIO: f32 = 1.78;
// The fixed virtual room places its front L/R references at x = +/-0.5.
const REFERENCE_SCREEN_HALF_WIDTH: f32 = 0.5;

/// Horizontal speaker-zone constraint carried by object metadata.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ObjectZone {
    #[default]
    All,
    NoBack,
    NoSide,
    CentreAndBack,
    Screen,
    Surround,
}

impl TryFrom<u8> for ObjectZone {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::All),
            1 => Ok(Self::NoBack),
            2 => Ok(Self::NoSide),
            3 => Ok(Self::CentreAndBack),
            4 => Ok(Self::Screen),
            5 => Ok(Self::Surround),
            reserved => Err(reserved),
        }
    }
}

/// Object-trim algorithm selected for the active virtual-speaker layout.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ObjectTrimMode {
    #[default]
    Disabled,
    Default,
    Custom,
}

/// Trim and balance parameters for one virtual-speaker configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObjectTrimSettings {
    pub mode: ObjectTrimMode,
    pub center_db: f32,
    pub surround_db: f32,
    pub height_db: f32,
    pub top_bottom_balance: f32,
    pub listener_balance: f32,
}

impl ObjectTrimSettings {
    /// Returns the scalar trim applied to a dynamic object after panning.
    ///
    /// Metadata positions use X/Y in [0, 1], while the renderer uses X/Y in
    /// [-1, 1] with Y pointing forwards. `num_mid` counts listener-plane
    /// surround speakers rather than the front L/C/R speakers.
    #[must_use]
    pub(crate) fn position_gain(
        self,
        position: [f32; 3],
        speaker_anchored: bool,
        stereo_configuration: bool,
        num_mid: usize,
        num_top: usize,
    ) -> f32 {
        if speaker_anchored {
            return 1.0;
        }

        let room_x = position[0].mul_add(0.5, 0.5);
        let room_y = (-position[1]).mul_add(0.5, 0.5);
        let room_z = position[2];
        let trim_db = match self.mode {
            ObjectTrimMode::Disabled => 0.0,
            ObjectTrimMode::Default => {
                #[allow(clippy::cast_precision_loss)] // Speaker counts are tiny.
                let layout_relief = 3.0 * ((num_mid as f32) / 4.0).min(1.0)
                    + 1.5 * ((num_top as f32) / 4.0).min(1.0);
                let maximum_trim = (-4.5 + layout_relief).min(0.0);
                let depth = (room_y / 0.6).clamp(0.0, 1.0);
                let height = ((room_z.abs() - 0.2) / 0.8).clamp(0.0, 1.0);
                maximum_trim * (depth + height).clamp(0.0, 1.0)
            }
            ObjectTrimMode::Custom => {
                let height_trim = self.height_db * room_z.abs();
                let surround_trim =
                    self.surround_db * if room_y >= 0.5 { 1.0 } else { 2.0 * room_y };
                let center_trim = if stereo_configuration {
                    let center_distance = (room_x - 0.5).abs() - 0.05;
                    let depth_distance = room_y - 0.05;
                    let height_distance = room_z.abs() - 0.05;
                    let transition = (center_distance.max(depth_distance).max(height_distance)
                        / 0.1)
                        .clamp(0.0, 1.0);
                    self.center_db * (1.0 - transition)
                } else {
                    0.0
                };
                center_trim + surround_trim + height_trim
            }
        };
        10_f32.powf(trim_db / 20.0)
    }

    /// Returns the separate equal-power front/back balance control.
    #[must_use]
    pub(crate) fn balance_gain(self, speaker: Speaker, top_or_bottom_plane: bool) -> f32 {
        if self.mode != ObjectTrimMode::Custom || speaker == Speaker::Lfe {
            return 1.0;
        }
        let balance = if top_or_bottom_plane {
            self.top_bottom_balance
        } else {
            self.listener_balance
        }
        .clamp(-1.0, 1.0);
        (1.0 - balance * speaker.position()[1]).max(0.0).sqrt()
    }
}

/// Layout-dependent level and front/back balance metadata for one object.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObjectTrim {
    pub warp_y: bool,
    pub disabled: bool,
    pub modes: [ObjectTrimMode; 9],
    pub center_db: [f32; 9],
    pub surround_db: [f32; 9],
    pub height_db: [f32; 9],
    pub top_bottom_balance: [f32; 9],
    pub listener_balance: [f32; 9],
}

impl ObjectTrim {
    #[must_use]
    pub(crate) fn default_algorithm() -> Self {
        Self {
            modes: [ObjectTrimMode::Default; 9],
            ..Self::default()
        }
    }

    #[must_use]
    pub(crate) fn uniform(warp_y: bool, settings: ObjectTrimSettings) -> Self {
        Self {
            warp_y,
            disabled: false,
            modes: [settings.mode; 9],
            center_db: [settings.center_db; 9],
            surround_db: [settings.surround_db; 9],
            height_db: [settings.height_db; 9],
            top_bottom_balance: [settings.top_bottom_balance; 9],
            listener_balance: [settings.listener_balance; 9],
        }
    }

    #[must_use]
    pub(crate) fn from_configurations(
        warp_y: bool,
        configurations: [ObjectTrimSettings; 9],
    ) -> Self {
        Self {
            warp_y,
            disabled: false,
            modes: configurations.map(|settings| settings.mode),
            center_db: configurations.map(|settings| settings.center_db),
            surround_db: configurations.map(|settings| settings.surround_db),
            height_db: configurations.map(|settings| settings.height_db),
            top_bottom_balance: configurations.map(|settings| settings.top_bottom_balance),
            listener_balance: configurations.map(|settings| settings.listener_balance),
        }
    }

    #[must_use]
    pub(crate) fn settings(self, configuration: usize) -> ObjectTrimSettings {
        if self.disabled {
            ObjectTrimSettings::default()
        } else {
            ObjectTrimSettings {
                mode: self.modes[configuration],
                center_db: self.center_db[configuration],
                surround_db: self.surround_db[configuration],
                height_db: self.height_db[configuration],
                top_bottom_balance: self.top_bottom_balance[configuration],
                listener_balance: self.listener_balance[configuration],
            }
        }
    }
}

/// Per-essence gain metadata for one ISF signal.
#[derive(Clone, Debug, PartialEq)]
pub struct IsfState {
    pub source_channel: usize,
    pub active: bool,
    pub gain: f32,
    pub trim: ObjectTrim,
}

/// Renderer-owned state for one decoded audio object.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // Independent flags are the OAMD wire semantics.
pub struct ObjectState {
    /// Zero-based channel in the decoder's object-essence output.
    pub source_channel: usize,
    pub active: bool,
    /// Fixed channel-bed assignment, or `None` for a dynamic object.
    pub bed_speaker: Option<Speaker>,
    /// Cartesian virtual-room position: left/right, rear/front, down/up.
    pub position: [f32; 3],
    /// Authored OAMD distance beyond the room boundary. `None` means that the
    /// mix did not specify a distance; infinity represents the far field.
    pub distance_factor: Option<f32>,
    pub gain: f32,
    /// Apparent width, depth, and height in room-coordinate units.
    pub size: [f32; 3],
    /// Constrain the object to the closest permitted virtual speaker.
    pub snap: bool,
    pub zone: ObjectZone,
    /// Whether the top/bottom speaker zone is permitted.
    pub elevation: bool,
    /// Horizontal energy spread supplied by extended object metadata.
    pub divergence: f32,
    pub trim: ObjectTrim,
}

/// One timestamped group of object-property updates.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialUpdate {
    pub sample_offset: usize,
    pub ramp_samples: usize,
    pub bed_speakers: Vec<Speaker>,
    pub isf: Vec<IsfState>,
    pub objects: Vec<ObjectState>,
}

/// Projects an in-room Cartesian position beyond the room boundary using the
/// OAMD distance factor.
#[must_use]
pub(crate) fn project_room_distance(position: [f32; 3], distance_factor: Option<f32>) -> [f32; 3] {
    let Some(factor) = distance_factor else {
        return position;
    };
    let boundary_axis = position.iter().copied().map(f32::abs).fold(0.0, f32::max);
    if boundary_axis <= f32::EPSILON {
        return position;
    }
    let finite_factor = if factor.is_infinite() {
        1_000_000.0
    } else {
        factor
    };
    let scale = finite_factor / boundary_axis;
    position.map(|coordinate| coordinate * scale)
}

/// Interpolates screen-anchored position and extent into the normalized room.
///
/// The reference screen is centred on the front wall, with width relative to
/// the virtual L/R speaker spacing and the normative 1.78 aspect ratio.
#[must_use]
pub(crate) fn interpolate_screen_geometry(
    room_position: [f32; 3],
    size: [f32; 3],
    screen_factor: f32,
    depth_factor: f32,
) -> ([f32; 3], [f32; 3]) {
    let screen_scale = [
        REFERENCE_SCREEN_HALF_WIDTH,
        1.0,
        REFERENCE_SCREEN_HALF_WIDTH / REFERENCE_SCREEN_ASPECT_RATIO,
    ];
    let screen_position: [f32; 3] =
        std::array::from_fn(|axis| room_position[axis] * screen_scale[axis]);
    let encoded_depth = ((1.0 - room_position[1]) * 0.5).clamp(0.0, 1.0);
    let interpolation = encoded_depth.powf(depth_factor) * screen_factor;
    let position = std::array::from_fn(|axis| {
        (room_position[axis] - screen_position[axis]).mul_add(interpolation, screen_position[axis])
    });
    let size = [
        size[0] * (screen_scale[0] + interpolation * (1.0 - screen_scale[0])),
        size[1],
        size[2] * (screen_scale[2] + interpolation * (1.0 - screen_scale[2])),
    ];
    (position, size)
}

#[cfg(test)]
mod tests {
    use super::{
        ObjectTrim, ObjectTrimMode, ObjectTrimSettings, interpolate_screen_geometry,
        project_room_distance,
    };

    #[test]
    #[allow(clippy::float_cmp)] // Inputs and expected results are exactly representable.
    fn distance_projects_from_the_room_boundary() {
        assert_eq!(
            project_room_distance([0.5, 0.25, 0.0], Some(2.0)),
            [2.0, 1.0, 0.0]
        );
        assert_eq!(
            project_room_distance([0.5, 0.25, 0.0], None),
            [0.5, 0.25, 0.0]
        );
    }

    #[test]
    fn screen_anchor_uses_the_reference_screen_at_the_front_wall() {
        let (position, size) = interpolate_screen_geometry([-1.0, 1.0, 1.0], [1.0; 3], 1.0, 2.0);
        assert!((position[0] + 0.5).abs() < f32::EPSILON);
        assert!((position[1] - 1.0).abs() < f32::EPSILON);
        assert!((position[2] - 0.5 / 1.78).abs() < f32::EPSILON);
        assert!((size[0] - 0.5).abs() < f32::EPSILON);
        assert!((size[1] - 1.0).abs() < f32::EPSILON);
        assert!((size[2] - 0.5 / 1.78).abs() < f32::EPSILON);
    }

    #[test]
    fn screen_anchor_interpolates_toward_room_coordinates_with_depth() {
        let (position, size) = interpolate_screen_geometry([-1.0, 0.0, 0.0], [1.0; 3], 1.0, 1.0);
        assert!((position[0] + 0.75).abs() < f32::EPSILON);
        assert!(position[1].abs() < f32::EPSILON);
        assert!(position[2].abs() < f32::EPSILON);
        assert!((size[0] - 0.75).abs() < f32::EPSILON);
        assert!((size[1] - 1.0).abs() < f32::EPSILON);
        assert!((size[2] - (0.5 + 0.25 / 1.78)).abs() < 1e-6);
    }

    #[test]
    fn default_trim_attenuates_rear_and_height_objects_for_sparse_layouts() {
        let trim = ObjectTrim::default_algorithm().settings(0);
        let front = trim.position_gain([0.0, 1.0, 0.0], false, true, 0, 0);
        let rear = trim.position_gain([0.0, -1.0, 0.0], false, true, 0, 0);
        let height = trim.position_gain([0.0, 1.0, 1.0], false, true, 0, 0);

        assert!((front - 1.0).abs() < f32::EPSILON);
        assert!((rear - 10_f32.powf(-4.5 / 20.0)).abs() < 1e-6);
        assert!((height - 10_f32.powf(-4.5 / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn custom_trim_is_one_scalar_and_bypasses_speaker_anchored_objects() {
        let trim = ObjectTrimSettings {
            mode: ObjectTrimMode::Custom,
            center_db: 3.0,
            surround_db: -6.0,
            height_db: -3.0,
            ..ObjectTrimSettings::default()
        };

        let front_center = trim.position_gain([0.0, 1.0, 0.0], false, true, 0, 0);
        let rear_height = trim.position_gain([0.0, -1.0, 1.0], false, false, 4, 4);
        let bed = trim.position_gain([0.0, -1.0, 1.0], true, false, 4, 4);

        assert!((front_center - 10_f32.powf(3.0 / 20.0)).abs() < 1e-6);
        assert!((rear_height - 10_f32.powf(-9.0 / 20.0)).abs() < 1e-6);
        assert!((bed - 1.0).abs() < f32::EPSILON);
    }
}
