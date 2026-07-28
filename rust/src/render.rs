use std::{ffi::OsString, fs::File, io::BufReader, path::Path};

use crate::{
    binaural::BinauralWriter,
    dsp::DEFAULT_CONVOLUTION_BLOCK,
    error::AppError,
    hrir::{HrirSet, Speaker},
    media::StreamManifest,
    process::ProcessRunner,
    room::RoomCorrection,
    stream_io::read_up_to,
    upmix::{ChannelProcessingOptions, ChannelProcessor},
};

#[derive(Clone, Copy, Debug)]
pub struct RenderResult {
    pub sample_rate: u32,
    pub frames: u64,
    pub peak_before_limiting: f32,
}

#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)] // Independent render switches mirror the public CLI.
pub struct ChannelRenderOptions {
    pub gain_db: f64,
    pub surround_swap: bool,
    pub matrix: bool,
    pub upconvert: bool,
    pub effect: f32,
    pub smoothness: f32,
    pub mute_bed: bool,
    pub mute_ground: bool,
    pub speaker_virtualizer: bool,
}

/// Decodes one selected channel-based stream, performs binaural convolution,
/// and writes deterministic stereo 16-bit PCM.
///
/// # Errors
///
/// Returns an error for unsupported layouts, missing stream properties, ffmpeg
/// decode failures, malformed decoded PCM, cancellation, convolution failures,
/// and WAV write failures.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation // The validated f64 CLI gain is applied to an f32 DSP pipeline.
)]
pub fn render_channel_track(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    input: &Path,
    stream: &StreamManifest,
    hrir: &HrirSet,
    room_correction: Option<&RoomCorrection>,
    options: ChannelRenderOptions,
    decoded_path: &Path,
    output_wave: &Path,
) -> Result<RenderResult, AppError> {
    let channel_count = usize::from(stream.channels.ok_or_else(|| {
        AppError::UnsupportedInput("selected stream does not report a channel count".into())
    })?);
    if channel_count == 0 {
        return Err(AppError::UnsupportedInput(
            "selected stream reports zero channels".into(),
        ));
    }
    let speakers = source_layout(stream.channel_layout.as_deref(), channel_count)?;
    decode_to_raw(
        runner,
        ffmpeg,
        input,
        stream.index,
        hrir.sample_rate,
        decoded_path,
    )?;
    render_raw(
        runner,
        decoded_path,
        output_wave,
        channel_count,
        &speakers,
        hrir,
        room_correction,
        options,
    )
}

pub(crate) fn decode_to_raw(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    input: &Path,
    stream_index: usize,
    sample_rate: u32,
    output: &Path,
) -> Result<(), AppError> {
    let arguments = vec![
        OsString::from("-nostdin"),
        OsString::from("-hide_banner"),
        OsString::from("-loglevel"),
        OsString::from("error"),
        OsString::from("-i"),
        input.as_os_str().to_os_string(),
        OsString::from("-map"),
        OsString::from(format!("0:{stream_index}")),
        OsString::from("-vn"),
        OsString::from("-sn"),
        OsString::from("-dn"),
        OsString::from("-ar"),
        OsString::from(sample_rate.to_string()),
        OsString::from("-c:a"),
        OsString::from("pcm_f32le"),
        OsString::from("-f"),
        OsString::from("f32le"),
        OsString::from("-y"),
        output.as_os_str().to_os_string(),
    ];
    let result = runner.run(ffmpeg, &arguments)?;
    if !result.status.success() {
        return Err(AppError::Render(format!(
            "ffmpeg could not decode selected stream ({}): {}",
            result.status,
            String::from_utf8_lossy(&result.stderr).trim()
        )));
    }
    if !output.is_file() {
        return Err(AppError::Render(
            "ffmpeg reported success without creating decoded PCM".into(),
        ));
    }
    Ok(())
}

pub(crate) fn demux_copy(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    input: &Path,
    stream_index: usize,
    format: &str,
    description: &str,
    output: &Path,
) -> Result<(), AppError> {
    let arguments = vec![
        OsString::from("-nostdin"),
        OsString::from("-hide_banner"),
        OsString::from("-loglevel"),
        OsString::from("error"),
        OsString::from("-i"),
        input.as_os_str().to_os_string(),
        OsString::from("-map"),
        OsString::from(format!("0:{stream_index}")),
        OsString::from("-vn"),
        OsString::from("-sn"),
        OsString::from("-dn"),
        OsString::from("-c:a"),
        OsString::from("copy"),
        OsString::from("-f"),
        OsString::from(format),
        OsString::from("-y"),
        output.as_os_str().to_os_string(),
    ];
    let result = runner.run(ffmpeg, &arguments)?;
    if !result.status.success() {
        return Err(AppError::Render(format!(
            "ffmpeg could not demux the selected {description} stream ({}): {}",
            result.status,
            String::from_utf8_lossy(&result.stderr).trim()
        )));
    }
    if !output.is_file() {
        return Err(AppError::Render(format!(
            "ffmpeg reported success without creating the {description} elementary stream"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // These are the complete inputs to one linear streaming pass.
fn render_raw(
    runner: &ProcessRunner,
    decoded_path: &Path,
    output_wave: &Path,
    channels: usize,
    speakers: &[Speaker],
    hrir: &HrirSet,
    room_correction: Option<&RoomCorrection>,
    options: ChannelRenderOptions,
) -> Result<RenderResult, AppError> {
    let mut input = BufReader::new(File::open(decoded_path).map_err(|source| AppError::File {
        path: decoded_path.to_path_buf(),
        source,
    })?);
    let mut writer = BinauralWriter::new(
        output_wave,
        hrir,
        room_correction,
        options.gain_db,
        hrir.channels.iter().map(|channel| channel.speaker),
    )?;
    let mut processor = ChannelProcessor::new(
        speakers,
        hrir,
        &writer,
        ChannelProcessingOptions {
            matrix: options.matrix,
            upconvert: options.upconvert,
            effect: options.effect,
            smoothness: options.smoothness,
            surround_swap: options.surround_swap,
            mute_bed: options.mute_bed,
            mute_ground: options.mute_ground,
            speaker_virtualizer: options.speaker_virtualizer,
        },
    )?;
    let frame_bytes = channels
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| AppError::Render("decoded frame size overflowed".into()))?;
    let mut bytes = vec![0_u8; DEFAULT_CONVOLUTION_BLOCK * frame_bytes];
    let mut samples = vec![0.0_f32; DEFAULT_CONVOLUTION_BLOCK * channels];

    loop {
        runner.check_cancelled()?;
        let bytes_read = read_up_to(&mut input, &mut bytes)?;
        if bytes_read == 0 {
            break;
        }
        if bytes_read % frame_bytes != 0 {
            return Err(AppError::Render(format!(
                "decoded PCM ended with {} incomplete bytes",
                bytes_read % frame_bytes
            )));
        }
        for (frame, frame_bytes) in bytes[..bytes_read].chunks_exact(frame_bytes).enumerate() {
            for (channel, sample_bytes) in frame_bytes.chunks_exact(4).enumerate() {
                samples[frame * channels + channel] = f32::from_le_bytes(
                    sample_bytes
                        .try_into()
                        .expect("chunks_exact guarantees four bytes"),
                );
            }
        }
        processor.process_interleaved(&samples[..bytes_read / 4], &mut writer)?;
    }
    writer.finish()
}

pub(crate) fn source_layout(
    layout: Option<&str>,
    channels: usize,
) -> Result<Vec<Speaker>, AppError> {
    use Speaker::{
        FrontCenter as C, FrontLeft as L, FrontRight as R, Lfe, RearCenter as Rc, RearLeft as Rl,
        RearRight as Rr, SideLeft as Sl, SideRight as Sr,
    };
    let known: Option<&[Speaker]> = match layout.unwrap_or_default() {
        "mono" => Some(&[C]),
        "stereo" => Some(&[L, R]),
        "2.1" => Some(&[L, R, Lfe]),
        "3.0" => Some(&[L, R, C]),
        "3.0(back)" => Some(&[L, R, Rc]),
        "4.0" => Some(&[L, R, C, Rc]),
        "quad" => Some(&[L, R, Rl, Rr]),
        "quad(side)" => Some(&[L, R, Sl, Sr]),
        "5.0" => Some(&[L, R, C, Rl, Rr]),
        "5.0(side)" => Some(&[L, R, C, Sl, Sr]),
        "5.1" => Some(&[L, R, C, Lfe, Rl, Rr]),
        "5.1(side)" => Some(&[L, R, C, Lfe, Sl, Sr]),
        "6.1" => Some(&[L, R, C, Lfe, Rc, Sl, Sr]),
        "7.1" => Some(&[L, R, C, Lfe, Rl, Rr, Sl, Sr]),
        _ => None,
    };
    let speakers = known
        .or_else(|| default_source_layout(channels))
        .ok_or_else(|| {
            AppError::UnsupportedInput(format!(
                "unsupported {channels}-channel layout '{}'",
                layout.unwrap_or("unknown")
            ))
        })?;
    if speakers.len() != channels {
        return Err(AppError::UnsupportedInput(format!(
            "layout '{}' describes {} channels, but ffprobe reported {channels}",
            layout.unwrap_or("unknown"),
            speakers.len()
        )));
    }
    Ok(speakers.to_vec())
}

#[rustfmt::skip]
fn default_source_layout(channels: usize) -> Option<&'static [Speaker]> {
    use Speaker::{
        FrontCenter as C, FrontLeft as L, FrontRight as R, Lfe,
        RearLeft as Rl, RearRight as Rr, SideLeft as Sl, SideRight as Sr,
        TopFrontCenter as Tfc, TopFrontLeft as Tfl, TopFrontRight as Tfr,
        TopRearCenter as Trc, TopRearLeft as Trl, TopRearRight as Trr,
        WideLeft as Wl, WideRight as Wr,
    };
    const LAYOUTS: &[&[Speaker]] = &[
        &[],
        &[C],
        &[L, R],
        &[L, R, C],
        &[L, R, Sl, Sr],
        &[L, R, C, Sl, Sr],
        &[L, R, C, Lfe, Sl, Sr],
        &[L, R, C, Rl, Rr, Sl, Sr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr],
        &[L, R, C, Rl, Rr, Sl, Sr, Tfl, Tfr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Tfl, Tfr],
        &[L, R, C, Rl, Rr, Sl, Sr, Tfl, Tfr, Trl, Trr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Tfl, Tfr, Trl, Trr],
        &[L, R, C, Rl, Rr, Sl, Sr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
        &[L, R, C, Rl, Rr, Sl, Sr, Wl, Wr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Wl, Wr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
    ];
    LAYOUTS.get(channels).copied()
}

#[cfg(test)]
mod tests {
    use super::source_layout;
    use crate::hrir::Speaker;

    #[test]
    fn ffmpeg_side_layout_maps_to_side_speakers() {
        assert_eq!(
            source_layout(Some("5.1(side)"), 6).unwrap(),
            [
                Speaker::FrontLeft,
                Speaker::FrontRight,
                Speaker::FrontCenter,
                Speaker::Lfe,
                Speaker::SideLeft,
                Speaker::SideRight,
            ]
        );
    }

    #[test]
    fn mismatched_layout_is_rejected() {
        assert!(source_layout(Some("stereo"), 6).is_err());
    }
}
