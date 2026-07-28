use crate::hrir::Speaker;

/// Per-essence gain metadata for one ISF signal.
#[derive(Clone, Debug, PartialEq)]
pub struct IsfState {
    pub source_channel: usize,
    pub active: bool,
    pub gain: f32,
}

/// Renderer-owned state for one decoded audio object.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectState {
    /// Zero-based channel in the decoder's object-essence output.
    pub source_channel: usize,
    pub active: bool,
    /// True when this essence is a channel bed represented inside JOC.
    pub bed: bool,
    /// Cartesian virtual-room position: left/right, rear/front, down/up.
    pub position: [f32; 3],
    pub gain: f32,
    /// Scalar diffusion used by the binaural panner.
    pub size: f32,
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
