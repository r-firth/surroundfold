use std::fmt;

use serde::Serialize;

use crate::{error::AppError, media::ContainerManifest};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecodeCapability {
    TrueHdObjects,
    JocObjects,
    Channels,
    Unsupported,
}

impl fmt::Display for DecodeCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TrueHdObjects => "truehd-objects",
            Self::JocObjects => "joc-objects",
            Self::Channels => "channels",
            Self::Unsupported => "unsupported",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioTrack {
    pub audio_index: usize,
    pub stream_index: usize,
    pub codec: String,
    pub profile: Option<String>,
    pub language: Option<String>,
    pub channels: Option<u16>,
    pub sample_rate: Option<u32>,
    pub capability: DecodeCapability,
    pub selection_rank: u16,
}

impl AudioTrack {
    #[must_use]
    pub fn from_manifest(manifest: &ContainerManifest) -> Vec<Self> {
        manifest
            .streams
            .iter()
            .filter(|stream| stream.codec_type == "audio")
            .enumerate()
            .map(|(audio_index, stream)| {
                let (capability, selection_rank) =
                    codec_capability(&stream.codec_name, stream.profile.as_deref());
                Self {
                    audio_index,
                    stream_index: stream.index,
                    codec: stream.codec_name.clone(),
                    profile: stream.profile.clone(),
                    language: stream.tag("language").map(str::to_owned),
                    channels: stream.channels,
                    sample_rate: stream.sample_rate,
                    capability,
                    selection_rank,
                }
            })
            .collect()
    }
}

/// Selects an explicit audio track or the highest-ranked compatible candidate.
///
/// # Errors
///
/// Returns an error when the requested track does not exist, is unsupported, or
/// no automatic candidate satisfies the requested language.
pub fn select_track<'a>(
    tracks: &'a [AudioTrack],
    explicit_index: Option<usize>,
    language: Option<&str>,
) -> Result<&'a AudioTrack, AppError> {
    if let Some(index) = explicit_index {
        let track = tracks
            .iter()
            .find(|track| track.audio_index == index)
            .ok_or_else(|| {
                let available = tracks
                    .iter()
                    .map(|track| track.audio_index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                AppError::UnsupportedInput(format!(
                    "audio track {index} does not exist; available indices: {}",
                    if available.is_empty() {
                        "none"
                    } else {
                        &available
                    }
                ))
            })?;
        if track.capability == DecodeCapability::Unsupported {
            return Err(AppError::UnsupportedInput(format!(
                "audio track {index} uses unsupported codec {}",
                track.codec
            )));
        }
        return Ok(track);
    }

    let mut candidates = tracks
        .iter()
        .filter(|track| track.capability != DecodeCapability::Unsupported)
        .filter(|track| {
            language.is_none_or(|wanted| {
                track
                    .language
                    .as_deref()
                    .is_some_and(|actual| actual.eq_ignore_ascii_case(wanted))
            })
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(AppError::UnsupportedInput(match language {
            Some(language) => {
                format!("no renderable audio track has language '{language}'")
            }
            None => "the input has no renderable audio track".into(),
        }));
    }
    candidates.sort_by_key(|track| (track.selection_rank, track.audio_index));
    Ok(candidates[0])
}

fn codec_capability(codec: &str, profile: Option<&str>) -> (DecodeCapability, u16) {
    match codec {
        "truehd" => (DecodeCapability::TrueHdObjects, 30),
        "eac3"
            if profile.is_some_and(|value| {
                value
                    .split(|character: char| !character.is_ascii_alphanumeric())
                    .any(|word| word.eq_ignore_ascii_case("atmos"))
            }) =>
        {
            (DecodeCapability::JocObjects, 35)
        }
        "eac3" => (DecodeCapability::Channels, 40),
        value if value.starts_with("pcm_f") => (DecodeCapability::Channels, 60),
        value if value.starts_with("pcm_") => (DecodeCapability::Channels, 70),
        "dts" if profile.is_some_and(|value| value.contains("DTS-HD")) => {
            (DecodeCapability::Channels, 80)
        }
        "flac" => (DecodeCapability::Channels, 90),
        "opus" => (DecodeCapability::Channels, 100),
        "dts" => (DecodeCapability::Channels, 110),
        "ac3" => (DecodeCapability::Channels, 120),
        "aac" | "alac" | "vorbis" | "mp3" => (DecodeCapability::Channels, 130),
        _ => (DecodeCapability::Unsupported, u16::MAX),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::media::{ContainerManifest, FormatManifest, StreamManifest};

    use super::{AudioTrack, DecodeCapability, select_track};

    fn track(index: usize, rank: u16, language: Option<&str>) -> AudioTrack {
        AudioTrack {
            audio_index: index,
            stream_index: index + 3,
            codec: "fixture".into(),
            profile: None,
            language: language.map(str::to_owned),
            channels: Some(2),
            sample_rate: Some(48_000),
            capability: DecodeCapability::Channels,
            selection_rank: rank,
        }
    }

    #[test]
    fn automatic_selection_uses_rank_then_original_order() {
        let tracks = [track(0, 90, None), track(1, 30, None), track(2, 30, None)];
        assert_eq!(select_track(&tracks, None, None).unwrap().audio_index, 1);
    }

    #[test]
    fn language_filter_is_required_not_advisory() {
        let tracks = [track(0, 30, Some("eng")), track(1, 40, Some("jpn"))];
        assert_eq!(
            select_track(&tracks, None, Some("JPN"))
                .unwrap()
                .audio_index,
            1
        );
        assert!(select_track(&tracks, None, Some("fra")).is_err());
    }

    #[test]
    fn explicit_selection_bypasses_ranking() {
        let tracks = [track(0, 10, None), track(1, 100, None)];
        assert_eq!(select_track(&tracks, Some(1), None).unwrap().audio_index, 1);
    }

    #[test]
    fn atmos_profile_selects_joc_object_reconstruction() {
        let manifest = ContainerManifest {
            format: FormatManifest {
                format_name: "matroska".into(),
                start_time: None,
                duration: None,
                tags: BTreeMap::new(),
            },
            streams: vec![StreamManifest {
                index: 2,
                codec_type: "audio".into(),
                codec_name: "eac3".into(),
                profile: Some("Dolby Digital Plus + Dolby Atmos".into()),
                channels: Some(6),
                channel_layout: Some("5.1(side)".into()),
                sample_rate: Some(48_000),
                bits_per_raw_sample: Some(24),
                initial_padding: None,
                start_time: None,
                duration: None,
                disposition: BTreeMap::new(),
                tags: BTreeMap::new(),
            }],
            chapters: Vec::new(),
        };

        assert_eq!(
            AudioTrack::from_manifest(&manifest)[0].capability,
            DecodeCapability::JocObjects
        );
    }
}
