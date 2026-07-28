use std::{
    collections::VecDeque,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use crate::{
    binaural::BinauralWriter,
    eac3::{Eac3Frame, FrameReader, MetadataPayload, StreamType, SyncFrameHeader},
    error::AppError,
    hrir::{HrirSet, Speaker},
    isf::IsfConfig,
    joc::{DownmixConfiguration, JocDecoder, JocFrame, QMF_SUBBANDS},
    media::StreamManifest,
    oamd::OamdDecoder,
    object_render::{ObjectRenderOptions, ObjectRenderer},
    process::ProcessRunner,
    qmf::{JocReconstructor, RECONSTRUCTION_DELAY},
    render::{RenderResult, decode_to_raw, demux_copy, source_layout},
    room::RoomCorrection,
};

const FLUSH_TIMESLOTS: usize = 24;

#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)] // Independent render switches mirror the public CLI.
pub struct Eac3RenderOptions {
    pub gain_db: f64,
    pub surround_swap: bool,
    pub mute_bed: bool,
    pub mute_ground: bool,
    pub speaker_virtualizer: bool,
}

/// Demuxes E-AC-3, lets `FFmpeg` decode only its channel core, reconstructs JOC
/// object essences in Rust, and binaurally renders OAMD positions.
///
/// # Errors
///
/// Returns an error for malformed or missing OAMD/JOC metadata, PCM/frame
/// desynchronization, unsupported downmix layouts, reconstruction failures, or
/// output failures.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keeps pipeline setup and execution together.
pub fn render_eac3_track(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    input: &Path,
    stream: &StreamManifest,
    hrir: &HrirSet,
    room_correction: Option<&RoomCorrection>,
    options: Eac3RenderOptions,
    elementary_path: &Path,
    decoded_path: &Path,
    output_wave: &Path,
) -> Result<RenderResult, AppError> {
    let source_channels = usize::from(stream.channels.ok_or_else(|| {
        AppError::UnsupportedInput("selected E-AC-3 stream has no channel count".into())
    })?);
    let source_speakers = source_layout(stream.channel_layout.as_deref(), source_channels)?;
    if stream.sample_rate != Some(hrir.sample_rate) {
        return Err(AppError::UnsupportedInput(format!(
            "DD+/Atmos sample rate {} does not match HRIR sample rate {}; object-domain resampling is not implemented yet",
            stream
                .sample_rate
                .map_or_else(|| "unknown".into(), |rate| rate.to_string()),
            hrir.sample_rate
        )));
    }

    demux_copy(
        runner,
        ffmpeg,
        input,
        stream.index,
        "eac3",
        "E-AC-3",
        elementary_path,
    )?;
    decode_to_raw(
        runner,
        ffmpeg,
        elementary_path,
        0,
        hrir.sample_rate,
        decoded_path,
    )?;

    let elementary = File::open(elementary_path).map_err(|source| AppError::File {
        path: elementary_path.to_path_buf(),
        source,
    })?;
    let decoded = File::open(decoded_path).map_err(|source| AppError::File {
        path: decoded_path.to_path_buf(),
        source,
    })?;
    let writer = BinauralWriter::new(
        output_wave,
        hrir,
        room_correction,
        options.gain_db,
        hrir.channels.iter().map(|channel| channel.speaker),
    )?;
    let renderer = ObjectRenderer::new(
        writer,
        hrir,
        ObjectRenderOptions {
            surround_swap: options.surround_swap,
            mute_bed: options.mute_bed,
            mute_ground: options.mute_ground,
            speaker_virtualizer: options.speaker_virtualizer,
        },
    )?;
    let mut pipeline = Eac3ObjectPipeline {
        runner,
        hrir,
        source_channels,
        source_speakers,
        pcm: BufReader::new(decoded),
        renderer,
        oamd: OamdDecoder::new(),
        joc: JocDecoder::new(),
        reconstructor: JocReconstructor::new(),
        last_joc: None,
        qmf_delay_remaining: RECONSTRUCTION_DELAY,
        lfe_queue: VecDeque::new(),
        lfe_present: None,
        object_count: None,
        isf: None,
        isf_initialized: false,
        source_samples: 0,
        emitted_samples: 0,
    };
    let mut frames = FrameReader::new(BufReader::new(elementary));
    let mut pending = None;
    while let Some(frame) = frames.next_frame()? {
        runner.check_cancelled()?;
        if frame.header.stream_type == StreamType::Independent && frame.header.substream_id == 0 {
            if let Some(program) = pending.replace(ProgramFrame::new(frame)) {
                pipeline.process(&program)?;
            }
        } else {
            if frame.header.stream_type != StreamType::Dependent {
                return Err(invalid(
                    "selected E-AC-3 stream contains a second independent program",
                ));
            }
            pending
                .as_mut()
                .ok_or_else(|| {
                    invalid("E-AC-3 elementary stream starts with a dependent substream")
                })?
                .add_dependent(frame)?;
        }
    }
    if let Some(program) = pending {
        pipeline.process(&program)?;
    }
    pipeline.finish()
}

struct ProgramFrame {
    header: SyncFrameHeader,
    sample_start: u64,
    payloads: Vec<MetadataPayload>,
}

impl ProgramFrame {
    fn new(frame: Eac3Frame) -> Self {
        Self {
            header: frame.header,
            sample_start: frame.sample_start,
            payloads: frame.payloads,
        }
    }

    fn add_dependent(&mut self, frame: Eac3Frame) -> Result<(), AppError> {
        if frame.sample_start != self.sample_start {
            return Err(invalid(
                "dependent E-AC-3 substream is not aligned with its primary frame",
            ));
        }
        if frame.header.sample_rate != self.header.sample_rate
            || frame.header.sample_count() != self.header.sample_count()
        {
            return Err(invalid(
                "dependent E-AC-3 substream has different timing from its primary frame",
            ));
        }
        self.payloads.extend(frame.payloads);
        Ok(())
    }
}

struct Eac3ObjectPipeline<'a> {
    runner: &'a ProcessRunner,
    hrir: &'a HrirSet,
    source_channels: usize,
    source_speakers: Vec<Speaker>,
    pcm: BufReader<File>,
    renderer: ObjectRenderer<'a>,
    oamd: OamdDecoder,
    joc: JocDecoder,
    reconstructor: JocReconstructor,
    last_joc: Option<JocFrame>,
    qmf_delay_remaining: usize,
    lfe_queue: VecDeque<f32>,
    lfe_present: Option<bool>,
    object_count: Option<usize>,
    isf: Option<IsfConfig>,
    isf_initialized: bool,
    source_samples: usize,
    emitted_samples: usize,
}

impl Eac3ObjectPipeline<'_> {
    fn process(&mut self, program: &ProgramFrame) -> Result<(), AppError> {
        self.runner.check_cancelled()?;
        if program.header.sample_rate != self.hrir.sample_rate {
            return Err(AppError::UnsupportedInput(format!(
                "E-AC-3 frame sample rate {} does not match HRIR sample rate {}",
                program.header.sample_rate, self.hrir.sample_rate
            )));
        }
        let oamd_payload = exactly_one_payload(&program.payloads, 11, "OAMD")?;
        let joc_payload = exactly_one_payload(&program.payloads, 14, "JOC")?;
        let oamd = self.oamd.decode(oamd_payload)?;
        let joc = self
            .joc
            .decode(joc_payload, program.header.sample_count())?;
        if oamd.joc_object_count != joc.object_count {
            return Err(invalid(format!(
                "OAMD describes {} JOC essences, JOC reconstructs {}",
                oamd.joc_object_count, joc.object_count
            )));
        }
        if self
            .object_count
            .replace(oamd.object_count)
            .is_some_and(|previous| previous != oamd.object_count)
        {
            return Err(AppError::UnsupportedInput(
                "OAMD object count changes within the stream".into(),
            ));
        }
        if self.isf_initialized && self.isf != oamd.isf {
            return Err(AppError::UnsupportedInput(
                "OAMD ISF configuration changes within the stream".into(),
            ));
        }
        self.isf = oamd.isf;
        self.isf_initialized = true;
        let has_lfe = !oamd.lfe_object_indices.is_empty();
        if self
            .lfe_present
            .replace(has_lfe)
            .is_some_and(|previous| previous != has_lfe)
        {
            return Err(AppError::UnsupportedInput(
                "OAMD LFE assignment changes within the stream".into(),
            ));
        }

        let sample_count = program.header.sample_count();
        let pcm = read_pcm(&mut self.pcm, sample_count, self.source_channels)?;
        self.source_samples = self
            .source_samples
            .checked_add(sample_count)
            .ok_or_else(|| invalid("E-AC-3 sample count overflowed"))?;
        let planar = deinterleave(&pcm, self.source_channels);
        let channel_indices = joc_channel_indices(joc.downmix, &self.source_speakers)?;
        if channel_indices.len() != joc.input_channels {
            return Err(invalid("JOC input-channel mapping has the wrong size"));
        }
        let downmix = channel_indices
            .iter()
            .map(|index| planar[*index].as_slice())
            .collect::<Vec<_>>();
        let reconstructed = self.reconstructor.reconstruct(&downmix, &joc)?;

        let lfe_index = self
            .source_speakers
            .iter()
            .position(|speaker| *speaker == Speaker::Lfe);
        if has_lfe != lfe_index.is_some() {
            return Err(invalid(
                "OAMD and the decoded E-AC-3 layout disagree about LFE presence",
            ));
        }
        if let Some(index) = lfe_index {
            self.lfe_queue.extend(planar[index].iter().copied());
        }
        self.renderer
            .schedule_at(program.sample_start, oamd.updates)?;
        self.emit_aligned(&reconstructed)?;
        self.last_joc = Some(joc);
        Ok(())
    }

    fn emit_aligned(&mut self, objects: &[Vec<f32>]) -> Result<(), AppError> {
        let available = objects.first().map_or(0, Vec::len);
        if objects.iter().any(|object| object.len() != available) {
            return Err(invalid(
                "QMF reconstruction returned unequal object lengths",
            ));
        }
        let drop = self.qmf_delay_remaining.min(available);
        self.qmf_delay_remaining -= drop;
        let remaining_source = self.source_samples.saturating_sub(self.emitted_samples);
        let emit = (available - drop).min(remaining_source);
        if emit == 0 {
            return Ok(());
        }

        let has_lfe = self.lfe_present.unwrap_or(false);
        let channels = objects.len() + usize::from(has_lfe);
        let mut interleaved = Vec::with_capacity(emit * channels);
        for sample in drop..drop + emit {
            for object in objects {
                interleaved.push(object[sample]);
            }
            if has_lfe {
                interleaved.push(
                    self.lfe_queue
                        .pop_front()
                        .ok_or_else(|| invalid("LFE bypass fell behind reconstructed objects"))?,
                );
            }
        }
        let mut speakers = vec![None; objects.len()];
        if has_lfe {
            speakers.push(Some(Speaker::Lfe));
        }
        self.renderer.push_samples(
            self.hrir.sample_rate,
            emit,
            channels,
            &interleaved,
            &speakers,
            self.isf,
        )?;
        self.emitted_samples += emit;
        Ok(())
    }

    fn finish(mut self) -> Result<RenderResult, AppError> {
        let last_joc = self
            .last_joc
            .take()
            .ok_or_else(|| invalid("E-AC-3 stream contains no complete Atmos frames"))?;
        while self.emitted_samples < self.source_samples {
            self.runner.check_cancelled()?;
            let held = last_joc.hold_last(FLUSH_TIMESLOTS);
            let silence = vec![vec![0.0_f32; FLUSH_TIMESLOTS * QMF_SUBBANDS]; held.input_channels];
            let inputs = silence.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let reconstructed = self.reconstructor.reconstruct(&inputs, &held)?;
            self.emit_aligned(&reconstructed)?;
        }
        if self.lfe_present.unwrap_or(false) && !self.lfe_queue.is_empty() {
            return Err(invalid(format!(
                "{} unconsumed LFE samples remain after QMF alignment",
                self.lfe_queue.len()
            )));
        }
        let mut extra = [0_u8; 1];
        if self.pcm.read(&mut extra)? != 0 {
            return Err(invalid(
                "FFmpeg decoded more PCM samples than the E-AC-3 frame timeline",
            ));
        }
        self.renderer.finish()
    }
}

fn exactly_one_payload<'a>(
    payloads: &'a [MetadataPayload],
    id: u32,
    name: &str,
) -> Result<&'a MetadataPayload, AppError> {
    let mut matches = payloads.iter().filter(|payload| payload.id == id);
    let payload = matches
        .next()
        .ok_or_else(|| invalid(format!("E-AC-3 Atmos frame has no {name} payload")))?;
    if matches.next().is_some() {
        return Err(invalid(format!(
            "E-AC-3 Atmos frame has multiple {name} payloads"
        )));
    }
    Ok(payload)
}

fn read_pcm(reader: &mut impl Read, frames: usize, channels: usize) -> Result<Vec<f32>, AppError> {
    let samples = frames
        .checked_mul(channels)
        .ok_or_else(|| invalid("decoded E-AC-3 sample count overflowed"))?;
    let mut bytes = vec![
        0_u8;
        samples
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid("decoded E-AC-3 byte count overflowed"))?
    ];
    reader.read_exact(&mut bytes).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            invalid("FFmpeg PCM ended before the E-AC-3 frame timeline")
        } else {
            error.into()
        }
    })?;
    let decoded = bytes
        .chunks_exact(4)
        .map(|sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]))
        .collect::<Vec<_>>();
    if decoded.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("FFmpeg decoded non-finite E-AC-3 PCM"));
    }
    Ok(decoded)
}

fn deinterleave(interleaved: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let mut planar = vec![Vec::with_capacity(interleaved.len() / channels); channels];
    for frame in interleaved.chunks_exact(channels) {
        for (channel, sample) in frame.iter().copied().enumerate() {
            planar[channel].push(sample);
        }
    }
    planar
}

fn joc_channel_indices(
    configuration: DownmixConfiguration,
    speakers: &[Speaker],
) -> Result<Vec<usize>, AppError> {
    use Speaker::{
        FrontCenter as C, FrontLeft as L, FrontRight as R, RearLeft as Rl, RearRight as Rr,
        SideLeft as Sl, SideRight as Sr, TopFrontLeft as Tfl, TopFrontRight as Tfr,
    };
    let surround_left = find_speaker(speakers, &[Sl, Rl])?;
    let surround_right = find_speaker(speakers, &[Sr, Rr])?;
    let mut result = vec![
        find_speaker(speakers, &[L])?,
        find_speaker(speakers, &[R])?,
        find_speaker(speakers, &[C])?,
        surround_left,
        surround_right,
    ];
    match configuration {
        DownmixConfiguration::FiveChannel | DownmixConfiguration::FiveChannelPhaseShifted => {}
        DownmixConfiguration::SevenChannel => {
            result.push(find_speaker(speakers, &[Rl])?);
            result.push(find_speaker(speakers, &[Rr])?);
        }
        DownmixConfiguration::FiveChannelHeight
        | DownmixConfiguration::FiveChannelHeightPhaseShifted => {
            result.push(find_speaker(speakers, &[Tfl])?);
            result.push(find_speaker(speakers, &[Tfr])?);
        }
    }
    let mut unique = result.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != result.len() {
        return Err(AppError::UnsupportedInput(
            "decoded E-AC-3 layout cannot provide distinct JOC input channels".into(),
        ));
    }
    Ok(result)
}

fn find_speaker(speakers: &[Speaker], candidates: &[Speaker]) -> Result<usize, AppError> {
    candidates
        .iter()
        .find_map(|candidate| speakers.iter().position(|speaker| speaker == candidate))
        .ok_or_else(|| {
            AppError::UnsupportedInput(format!(
                "decoded E-AC-3 layout is missing JOC channel {}",
                candidates
                    .iter()
                    .map(|speaker| format!("{speaker:?}"))
                    .collect::<Vec<_>>()
                    .join("/")
            ))
        })
}

fn invalid(message: impl Into<String>) -> AppError {
    AppError::Render(format!("invalid DD+/Atmos stream: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::{deinterleave, joc_channel_indices};
    use crate::{hrir::Speaker, joc::DownmixConfiguration};

    #[test]
    fn ffmpeg_side_layout_maps_to_joc_order_without_lfe() {
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontRight,
            Speaker::FrontCenter,
            Speaker::Lfe,
            Speaker::SideLeft,
            Speaker::SideRight,
        ];
        assert_eq!(
            joc_channel_indices(DownmixConfiguration::FiveChannel, &speakers).unwrap(),
            [0, 1, 2, 4, 5]
        );
    }

    #[test]
    fn deinterleave_preserves_frame_and_channel_order() {
        assert_eq!(
            deinterleave(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3),
            [vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
    }
}
