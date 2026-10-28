//! WGPU-accelerated windowed frontend.
//!
//!   cargo run --release --example play_gpu -- <game.cue> [bios.bin] [--fast]
//!       [--dynarec] [--res=WIDTHxHEIGHT]

use aruanpsx::{
    Emulator, PadInput, buttons as b,
    disc::Disc,
    gpu::{HwOp, HwPrim, hw_flags as HF},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use gilrs::{Axis, Button, EventType, Gilrs};
use std::cell::UnsafeCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

const DEFAULT_BIOS: &str = "F:/Games/Emulators/Playstation/PSX/Duckstation/Bios/SCPH1001.BIN";
const SRC_RATE: f64 = 44_100.0;
const AUDIO_RING_CAP: usize = 65_536;
const AUDIO_RING_MASK: usize = AUDIO_RING_CAP - 1;

/// Single-producer/single-consumer audio FIFO. The emulation thread is
/// the only producer and CPAL's callback is the only consumer; unlike a
/// mutex queue, neither side can block the real-time audio callback.
struct AudioRing {
    slots: Box<[UnsafeCell<(i16, i16)>]>,
    read: AtomicUsize,
    write: AtomicUsize,
    underruns: AtomicU64,
    overruns: AtomicU64,
}

// Slot ownership is transferred using the acquire/release indices below.
unsafe impl Sync for AudioRing {}

impl AudioRing {
    fn new() -> Self {
        let slots = (0..AUDIO_RING_CAP)
            .map(|_| UnsafeCell::new((0, 0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
            underruns: AtomicU64::new(0),
            overruns: AtomicU64::new(0),
        }
    }

    fn push(&self, sample: (i16, i16)) {
        let write = self.write.load(Ordering::Relaxed);
        let next = (write + 1) & AUDIO_RING_MASK;
        if next == self.read.load(Ordering::Acquire) {
            self.overruns.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // SAFETY: only the producer writes a slot, and it is not reused
        // until the consumer publishes a later read index.
        unsafe { *self.slots[write].get() = sample };
        self.write.store(next, Ordering::Release);
    }

    fn pop(&self) -> Option<(i16, i16)> {
        let read = self.read.load(Ordering::Relaxed);
        if read == self.write.load(Ordering::Acquire) {
            self.underruns.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        // SAFETY: the producer published this slot before advancing write.
        let sample = unsafe { *self.slots[read].get() };
        self.read
            .store((read + 1) & AUDIO_RING_MASK, Ordering::Release);
        Some(sample)
    }

    fn len(&self) -> usize {
        self.write
            .load(Ordering::Acquire)
            .wrapping_sub(self.read.load(Ordering::Acquire))
            & AUDIO_RING_MASK
    }
}

type AudioQueue = Arc<AudioRing>;

fn start_audio(queue: AudioQueue) -> Option<cpal::Stream> {
    let dev = cpal::default_host().default_output_device()?;
    let cfg = dev.default_output_config().ok()?;
    let rate = cfg.sample_rate().0;
    let channels = cfg.channels() as usize;
    let nominal_step = SRC_RATE / rate as f64;
    // Match FRAME_AUDIO_CUSHION: about 200 ms of NTSC audio, enough for
    // the worst one-time boot bursts (~135 ms) to pass without an underrun.
    let target_fill = 12 * 735;
    let (mut pos, mut cur, mut next) = (0.0f64, (0i16, 0i16), (0i16, 0i16));
    let mut primed = false;
    let mut step = nominal_step;
    let mut adjust_count = 0usize;
    let mut pull = move |q: &AudioRing| {
        let fill = q.len();
        if !primed {
            if fill < target_fill {
                return (0.0, 0.0);
            }
            primed = true;
        }
        if fill == 0 {
            q.underruns.fetch_add(1, Ordering::Relaxed);
            primed = false;
            return (0.0, 0.0);
        }
        if adjust_count == 0 {
            // Gentle depth servo: enough to absorb clock/scheduler skew,
            // but capped at 2% so depth changes from frame drops never
            // produce an audible tape-rewind warble. Deep deficits fall
            // through to an underrun (a clean pause) instead.
            let error = (fill as f64 - target_fill as f64) / target_fill as f64;
            step = nominal_step * (1.0 + (error * 0.05).clamp(-0.02, 0.02));
        }
        adjust_count = (adjust_count + 1) & 127;
        pos += step;
        while pos >= 1.0 {
            pos -= 1.0;
            cur = next;
            next = q.pop().unwrap_or((0, 0));
        }
        let t = pos as f32;
        let l = cur.0 as f32 + (next.0 as f32 - cur.0 as f32) * t;
        let r = cur.1 as f32 + (next.1 as f32 - cur.1 as f32) * t;
        (l / 32768.0, r / 32768.0)
    };
    let err = |e| eprintln!("audio error: {e}");
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_output_stream(
            &cfg.config(),
            move |out: &mut [f32], _| {
                for frame in out.chunks_mut(channels) {
                    let (l, r) = pull(&queue);
                    for (i, sample) in frame.iter_mut().enumerate() {
                        *sample = if i % 2 == 0 { l } else { r };
                    }
                }
            },
            err,
            None,
        ),
        cpal::SampleFormat::I16 => dev.build_output_stream(
            &cfg.config(),
            move |out: &mut [i16], _| {
                for frame in out.chunks_mut(channels) {
                    let (l, r) = pull(&queue);
                    for (i, sample) in frame.iter_mut().enumerate() {
                        *sample = ((if i % 2 == 0 { l } else { r }) * 32767.0) as i16;
                    }
                }
            },
            err,
            None,
        ),
        _ => return None,
    }
    .ok()?;
    stream.play().ok()?;
    Some(stream)
}

fn pad_state(
    keys: &HashSet<KeyCode>,
    gilrs: Option<&Gilrs>,
    raw_buttons: &HashSet<(usize, u32)>,
    start_code: Option<u32>,
    select_code: Option<u32>,
) -> u16 {
    let key_map = [
        (KeyCode::ArrowUp, b::UP),
        (KeyCode::ArrowDown, b::DOWN),
        (KeyCode::ArrowLeft, b::LEFT),
        (KeyCode::ArrowRight, b::RIGHT),
        (KeyCode::KeyZ, b::CROSS),
        (KeyCode::KeyX, b::CIRCLE),
        (KeyCode::KeyA, b::SQUARE),
        (KeyCode::KeyS, b::TRIANGLE),
        (KeyCode::KeyQ, b::L1),
        (KeyCode::KeyW, b::R1),
        (KeyCode::KeyE, b::L2),
        (KeyCode::KeyR, b::R2),
        (KeyCode::Enter, b::START),
        (KeyCode::Backspace, b::SELECT),
    ];
    let mut bits = key_map
        .iter()
        .filter(|(key, _)| keys.contains(key))
        .fold(0, |acc, (_, mask)| acc | mask);
    if let Some(g) = gilrs {
        for (id, pad) in g.gamepads() {
            let id = usize::from(id);
            if start_code.is_some_and(|c| raw_buttons.contains(&(id, c))) {
                bits |= b::START;
            }
            if select_code.is_some_and(|c| raw_buttons.contains(&(id, c))) {
                bits |= b::SELECT;
            }
            let map = [
                (Button::DPadUp, b::UP),
                (Button::DPadDown, b::DOWN),
                (Button::DPadLeft, b::LEFT),
                (Button::DPadRight, b::RIGHT),
                (Button::South, b::CROSS),
                (Button::East, b::CIRCLE),
                (Button::West, b::SQUARE),
                (Button::North, b::TRIANGLE),
                (Button::LeftTrigger, b::L1),
                (Button::RightTrigger, b::R1),
                (Button::LeftTrigger2, b::L2),
                (Button::RightTrigger2, b::R2),
                (Button::Start, b::START),
                (Button::Select, b::SELECT),
            ];
            for (button, mask) in map {
                if pad.is_pressed(button) {
                    bits |= mask;
                }
            }
            let (x, y) = (pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY));
            if x < -0.5 {
                bits |= b::LEFT;
            }
            if x > 0.5 {
                bits |= b::RIGHT;
            }
            if y > 0.5 {
                bits |= b::UP;
            }
            if y < -0.5 {
                bits |= b::DOWN;
            }
        }
    }
    bits
}

fn parse_window_size(value: &str) -> Result<PhysicalSize<u32>, String> {
    let mut parts = value.split(|c| matches!(c, 'x' | 'X' | '×'));
    let width = parts
        .next()
        .ok_or_else(|| "expected WIDTHxHEIGHT".to_string())?
        .parse::<u32>()
        .map_err(|_| "width must be a positive integer".to_string())?;
    let height = parts
        .next()
        .ok_or_else(|| "expected WIDTHxHEIGHT".to_string())?
        .parse::<u32>()
        .map_err(|_| "height must be a positive integer".to_string())?;
    if parts.next().is_some() || !(320..=8192).contains(&width) || !(240..=8192).contains(&height) {
        return Err("resolution must be between 320x240 and 8192x8192".into());
    }
    Ok(PhysicalSize::new(width, height))
}

fn parse_internal_scale(value: &str) -> Result<u8, String> {
    match value.to_ascii_lowercase().as_str() {
        "1x" | "native" => return Ok(1),
        "2x" => return Ok(2),
        "4x" => return Ok(4),
        "6x" => return Ok(6),
        "8x" => return Ok(8),
        _ => {}
    }
    let mut parts = value.split(|c| matches!(c, 'x' | 'X' | '×'));
    let width = parts
        .next()
        .ok_or_else(|| "expected 2x, 4x, 8x, or WIDTHxHEIGHT".to_string())?
        .parse::<u32>()
        .map_err(|_| "expected 2x, 4x, 8x, or WIDTHxHEIGHT".to_string())?;
    let height = parts
        .next()
        .ok_or_else(|| "expected 2x, 4x, 8x, or WIDTHxHEIGHT".to_string())?
        .parse::<u32>()
        .map_err(|_| "height must be an integer".to_string())?;
    if parts.next().is_some() || width < 512 || height < 240 {
        return Err("internal resolution must be at least 512x240".into());
    }
    let requested = (width as f64 / 512.0).min(height as f64 / 240.0);
    Ok([1u8, 2, 4, 6, 8]
        .into_iter()
        .min_by(|a, b| ((*a as f64 - requested).abs()).total_cmp(&(*b as f64 - requested).abs()))
        .unwrap_or(1))
}

fn parse_texture_filter(value: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "nearest" => Ok(false),
        "bilinear" => Ok(true),
        _ => Err("expected nearest or bilinear".into()),
    }
}

fn parse_crt_filter(value: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err("expected on or off".into()),
    }
}

/// How far behind the wall clock the frame grid may fall before we give
/// up on repaying the debt outright (long modal drags, debugger pauses).
const MAX_FRAME_DEBT: Duration = Duration::from_millis(400);
/// Startup/resync cushion: seeds enough owed frames to build ~200 ms of
/// audio ahead of consumption so one-time boot bursts can't starve the
/// output device.
const FRAME_AUDIO_CUSHION: Duration = Duration::from_millis(200);

/// Advance the frame deadline on a fixed grid so a late frame is repaid
/// by running the next frames back-to-back (the emulator headroom turns
/// the spike into buffered audio instead of an underrun). If the grid is
/// hopelessly behind, resync to a bounded catch-up burst.
fn advance_frame_deadline(next_frame: Instant, period: Duration, now: Instant) -> Instant {
    let next = next_frame + period;
    if next + MAX_FRAME_DEBT < now {
        now.checked_sub(FRAME_AUDIO_CUSHION).unwrap_or(now)
    } else {
        next
    }
}

/// Reseed the frame grid after a long stall (or at startup) so the next
/// frames run immediately and rebuild the audio cushion.
fn seeded_frame_deadline(now: Instant) -> Instant {
    now.checked_sub(FRAME_AUDIO_CUSHION).unwrap_or(now)
}

#[cfg(target_os = "windows")]
#[link(name = "winmm")]
unsafe extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
    fn timeEndPeriod(period: u32) -> u32;
    fn joyGetPosEx(id: u32, info: *mut JoyInfoEx) -> u32;
    fn joyGetNumDevs() -> u32;
}

/// Classic winmm joystick reading: sees buttons the WGI backend drops
/// (generic HID pads often report Select/Start on usages gilrs misses).
#[repr(C)]
#[derive(Default)]
struct JoyInfoEx {
    dw_size: u32,
    dw_flags: u32,
    dw_xpos: u32,
    dw_ypos: u32,
    dw_zpos: u32,
    dw_rpos: u32,
    dw_upos: u32,
    dw_vpos: u32,
    dw_buttons: u32,
    dw_button_number: u32,
    dw_pov: u32,
    dw_reserved1: u32,
    dw_reserved2: u32,
}

#[cfg(target_os = "windows")]
const JOY_RETURNALL: u32 = 0xFF;
#[cfg(target_os = "windows")]
const JOYERR_NOERROR: u32 = 0;
const JOY_POVCENTERED: u32 = 0xFFFF;

/// Joystick IDs present at startup (probed with a position read).
fn scan_winmm_pads() -> Vec<u32> {
    #[cfg(target_os = "windows")]
    {
        let max = unsafe { joyGetNumDevs() }.max(16);
        (0..max)
            .filter(|&id| read_winmm_pad(id).is_some())
            .collect()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}

fn read_winmm_pad(id: u32) -> Option<JoyInfoEx> {
    #[cfg(target_os = "windows")]
    {
        let mut info = JoyInfoEx {
            dw_size: std::mem::size_of::<JoyInfoEx>() as u32,
            dw_flags: JOY_RETURNALL,
            ..Default::default()
        };
        (unsafe { joyGetPosEx(id, &mut info) } == JOYERR_NOERROR).then_some(info)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = id;
        None
    }
}

/// Standard DirectInput numbering for PS-style pads (and Switch-Pro
/// clones like the GameSir ones): 8 = Select, 9 = Start, POV hat =
/// d-pad — safe to OR in for every pad since those indices are
/// unambiguous. `full` additionally maps the whole pad using the
/// dominant non-XInput layout (face buttons in PS position order);
/// used only when gilrs sees no pad at all, where an imperfect guess
/// beats a dead controller.
fn winmm_info_bits(info: &JoyInfoEx, full: bool) -> u16 {
    let mut bits = 0;
    let btn = info.dw_buttons;
    if btn & (1 << 9) != 0 {
        bits |= b::START;
    }
    if btn & (1 << 8) != 0 {
        bits |= b::SELECT;
    }
    if full {
        for (idx, mask) in [
            (0, b::SQUARE),
            (1, b::CROSS),
            (2, b::CIRCLE),
            (3, b::TRIANGLE),
            (4, b::L1),
            (5, b::R1),
            (6, b::L2),
            (7, b::R2),
        ] {
            if btn & (1 << idx) != 0 {
                bits |= mask;
            }
        }
    }
    let pov = info.dw_pov;
    if pov != JOY_POVCENTERED {
        if pov <= 4500 || pov >= 31500 {
            bits |= b::UP;
        }
        if (13500..=22500).contains(&pov) {
            bits |= b::DOWN;
        }
        if (4500..=13500).contains(&pov) {
            bits |= b::RIGHT;
        }
        if (22500..=31500).contains(&pov) {
            bits |= b::LEFT;
        }
    }
    bits
}

struct FrameTimerResolution {
    requested: bool,
}

impl FrameTimerResolution {
    fn request() -> Self {
        #[cfg(target_os = "windows")]
        let requested = unsafe { timeBeginPeriod(1) == 0 };
        #[cfg(not(target_os = "windows"))]
        let requested = false;
        Self { requested }
    }
}

impl Drop for FrameTimerResolution {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        if self.requested {
            unsafe {
                timeEndPeriod(1);
            }
        }
    }
}

struct GpuPresenter {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_linear: wgpu::BindGroup,
    bind_nearest: wgpu::BindGroup,
    crt_uniform: wgpu::Buffer,
    source_texture: wgpu::Texture,
    source_view: wgpu::TextureView,
    rgba: Vec<u8>,
    source_size: (u32, u32),
    adapter_name: String,
    hw: Option<HwRenderer>,
}

impl GpuPresenter {
    async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create WGPU surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("no hardware graphics adapter available");
        let adapter_name = format!(
            "{:?} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("AES Oxide display device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("request WGPU device");
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface unsupported by graphics adapter");
        // Frame pacing is handled by the event loop; avoid a second vsync
        // wait in presentation that can throttle an already-late frame.
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &config);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PSX display texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let crt_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PSX CRT presentation settings"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PSX display scaler"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                struct Out {
                    @builtin(position) pos: vec4<f32>,
                    @location(0) uv: vec2<f32>,
                };
                @vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
                    var p = array<vec2<f32>, 3>(
                        vec2<f32>(-1.0,  1.0),
                        vec2<f32>( 3.0,  1.0),
                        vec2<f32>(-1.0, -3.0)
                    );
                    var uv = array<vec2<f32>, 3>(
                        vec2<f32>(0.0, 0.0),
                        vec2<f32>(2.0, 0.0),
                        vec2<f32>(0.0, 2.0)
                    );
                    var out: Out;
                    out.pos = vec4<f32>(p[i], 0.0, 1.0);
                    out.uv = uv[i];
                    return out;
                }
                @group(0) @binding(0) var image: texture_2d<f32>;
                @group(0) @binding(1) var image_sampler: sampler;
                struct CrtSettings { enabled: f32, scale: f32, pad0: f32, pad1: f32 }
                @group(0) @binding(2) var<uniform> crt: CrtSettings;
                @fragment fn fs(in: Out) -> @location(0) vec4<f32> {
                    let sample = textureSample(image, image_sampler, in.uv);
                    if crt.enabled < 0.5 { return sample; }
                    let dims = vec2<f32>(textureDimensions(image));
                    let source = in.uv * dims / crt.scale;
                    let line_phase = fract(source.y);
                    let line = 0.90 + 0.10 * (1.0 - abs(2.0 * line_phase - 1.0));
                    let phase = u32(floor(source.x * 3.0)) % 3u;
                    let mask = vec3<f32>(
                        select(0.965, 1.0, phase == 0u),
                        select(0.965, 1.0, phase == 1u),
                        select(0.965, 1.0, phase == 2u),
                    );
                    return vec4<f32>(sample.rgb * mask * line, sample.a);
                }
                "#
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PSX display pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PSX display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = |filter| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("PSX output sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        };
        let tex_size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let source_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PSX framebuffer"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let source_view = source_texture.create_view(&Default::default());
        let linear_sampler = sampler(wgpu::FilterMode::Linear);
        let nearest_sampler = sampler(wgpu::FilterMode::Nearest);
        let make_bind = |s: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PSX framebuffer binding"),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(s),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: crt_uniform.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_linear = make_bind(&linear_sampler);
        let bind_nearest = make_bind(&nearest_sampler);
        eprintln!("WGPU display adapter: {adapter_name}");
        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_linear,
            bind_nearest,
            crt_uniform,
            source_texture,
            source_view,
            rgba: Vec::new(),
            source_size: (1, 1),
            adapter_name,
            hw: None,
        }
    }

    fn enable_hw(&mut self, scale: u32) {
        if self.hw.is_none() {
            self.hw = Some(HwRenderer::new(
                &self.device,
                &self.queue,
                scale,
                self.config.format,
            ));
            eprintln!(
                "Hardware renderer: {scale}x internal on {self_adapter}",
                self_adapter = self.adapter_name
            );
        }
    }

    /// Hardware-rendered frame: replay ops into the scaled target and
    /// present its display crop — the CPU path is bypassed entirely.
    fn render_hw(&mut self, window: &Window, crop: (u32, u32, u32, u32), smooth: bool, crt: bool) {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        if size.width != self.config.width || size.height != self.config.height {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
        let Some(hw) = self.hw.as_mut() else { return };
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PSX hw frame encoder"),
            });
        let aspect = 4.0f32 / 3.0;
        let (ww, wh) = (size.width as f32, size.height as f32);
        let (vw, vh) = if ww / wh > aspect {
            (wh * aspect, wh)
        } else {
            (ww, ww / aspect)
        };
        hw.present(
            &self.queue,
            &mut encoder,
            &view,
            ((ww - vw) * 0.5, (wh - vh) * 0.5, vw, vh),
            crop,
            smooth,
            crt,
        );
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }

    fn replay_hw(&mut self, ops: &[HwOp], vram: &[u16], texture_filter: bool) {
        if let Some(hw) = self.hw.as_mut() {
            hw.replay(&self.device, &self.queue, ops, vram, texture_filter);
        }
    }

    fn resize_source(&mut self, width: u32, height: u32) {
        if self.source_size == (width, height) {
            return;
        }
        self.source_size = (width, height);
        self.source_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PSX framebuffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.source_view = self.source_texture.create_view(&Default::default());
        // Rebuild sampler bindings so they point at the resized texture view.
        let layout = self.pipeline.get_bind_group_layout(0);
        let make_sampler = |filter| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("PSX output sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        };
        let linear = make_sampler(wgpu::FilterMode::Linear);
        let nearest = make_sampler(wgpu::FilterMode::Nearest);
        let make_bind = |sampler: &wgpu::Sampler| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PSX framebuffer binding"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.crt_uniform.as_entire_binding(),
                    },
                ],
            })
        };
        self.bind_linear = make_bind(&linear);
        self.bind_nearest = make_bind(&nearest);
    }

    fn render(
        &mut self,
        window: &Window,
        pixels: &[u32],
        width: usize,
        height: usize,
        smooth: bool,
        crt: bool,
        crt_scale: f32,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        self.resize_source(width as u32, height as u32);
        self.rgba.clear();
        self.rgba.reserve(pixels.len() * 4);
        for &p in pixels {
            self.rgba
                .extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8, 255]);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width as u32 * 4),
                rows_per_image: Some(height as u32),
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );
        self.queue.write_buffer(
            &self.crt_uniform,
            0,
            bytemuck::cast_slice(&[crt as u8 as f32, crt_scale, 0.0f32, 0.0f32]),
        );
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        if size.width != self.config.width || size.height != self.config.height {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PSX frame encoder"),
            });
        let attachment = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PSX frame render pass"),
                color_attachments: &attachment,
                ..Default::default()
            });
            let aspect = 4.0f32 / 3.0;
            let (ww, wh) = (size.width as f32, size.height as f32);
            let (vw, vh) = if ww / wh > aspect {
                (wh * aspect, wh)
            } else {
                (ww, ww / aspect)
            };
            pass.set_viewport((ww - vw) * 0.5, (wh - vh) * 0.5, vw, vh, 0.0, 1.0);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(
                0,
                if smooth {
                    &self.bind_linear
                } else {
                    &self.bind_nearest
                },
                &[],
            );
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

// ---- hardware PSX renderer ------------------------------------------
// Replays the core's recorded op stream onto the host GPU: native VRAM
// lives in an R16Uint texture for texel reads, polygons rasterize into
// an RGBA8 scaled target. This is the same split Beetle-PSX HW and
// DuckStation use — native domain for texture/CPU data, scaled domain
// for rendered output.

const RESOLVE_F: u32 = 1 << 13; // frontend-internal flag

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVert {
    pos: [f32; 2],
    uv: [f32; 2],
    col: [f32; 4],
    p0: u32, // tpage | clut<<16
    p1: u32, // texwin (20 bits)
    p2: u32, // prim color
    flags: u32,
    p3: u32, // signed uv min/max
    p4: u32, // signed vv min/max
}

struct HwDraw {
    pipe: usize,
    scissor: [u32; 4],
    first_index: u32,
    index_count: u32,
}

const PIPE_OPAQUE: usize = 0;
const PIPE_OPAQUE_MC: usize = 1;
const PIPE_ST0: usize = 2;
const PIPE_ST1: usize = 3;
const PIPE_ST2: usize = 4;
const PIPE_ST3: usize = 5;
const PIPE_ST1_MC: usize = 6;
const PIPE_ST2_MC: usize = 7;
const PIPES: usize = 8;

struct HwRenderer {
    scale: u32,
    output_srgb: bool,
    vram_tex: wgpu::Texture,
    #[allow(dead_code)]
    vram_view: wgpu::TextureView,
    #[allow(dead_code)]
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    #[allow(dead_code)]
    uniform: wgpu::Buffer,
    vram_bind: wgpu::BindGroup,
    pipes: Vec<wgpu::RenderPipeline>,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    verts: Vec<GpuVert>,
    indices: Vec<u32>,
    draws: Vec<HwDraw>,
    scratch: Vec<u16>,
    vram_uploaded: bool,
    scanout: wgpu::RenderPipeline,
    scanout_uniform: wgpu::Buffer,
    #[allow(dead_code)]
    scanout_layout: wgpu::BindGroupLayout,
    bind_linear: wgpu::BindGroup,
    bind_nearest: wgpu::BindGroup,
}

const HW_SHADER: &str = r#"
const F_TEX: u32 = 1u;
const F_RAW: u32 = 8u;
const F_MASK_SET: u32 = 16u;
const F_DITHER: u32 = 64u;
const F_ONLY_OPAQUE: u32 = 2048u;
const F_ONLY_SEMI: u32 = 4096u;
const F_RESOLVE: u32 = 8192u;

struct U { scale: f32, tw: f32, th: f32, tex_filter: f32 }
@group(0) @binding(0) var<uniform> uni: U;
@group(0) @binding(1) var vram_tex: texture_2d<u32>;

struct Vin {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) col: vec4<f32>,
    @location(3) p0: u32,
    @location(4) p1: u32,
    @location(5) p2: u32,
    @location(6) flags: u32,
    @location(7) p3: u32,
    @location(8) p4: u32,
};
struct Vout {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(linear) uv: vec2<f32>,
    @location(1) @interpolate(linear) col: vec4<f32>,
    @location(2) @interpolate(flat) p0: u32,
    @location(3) @interpolate(flat) p1: u32,
    @location(4) @interpolate(flat) p2: u32,
    @location(5) @interpolate(flat) flags: u32,
    @location(6) @interpolate(flat) p3: u32,
    @location(7) @interpolate(flat) p4: u32,
};

@vertex fn vs(v: Vin) -> Vout {
    var o: Vout;
    let p = v.pos * uni.scale;
    o.pos = vec4<f32>(p.x / uni.tw * 2.0 - 1.0, 1.0 - p.y / uni.th * 2.0, 0.0, 1.0);
    o.uv = v.uv;
    o.col = v.col;
    o.p0 = v.p0;
    o.p1 = v.p1;
    o.p2 = v.p2;
    o.flags = v.flags;
    o.p3 = v.p3;
    o.p4 = v.p4;
    return o;
}

const DT = array<array<i32, 4>, 4>(
    array<i32, 4>(-4, 0, -3, 1),
    array<i32, 4>(2, -2, 3, -1),
    array<i32, 4>(-3, 1, -4, 0),
    array<i32, 4>(3, -1, 2, -2),
);

fn vram_px(x: i32, y: i32) -> u32 {
    return textureLoad(vram_tex, vec2<i32>(x & 1023, y & 511), 0).r;
}

// Returns (15-bit color, semitransparency bit).
fn texel(u_in: i32, v_in: i32, tpage: u32, clut: u32, texwin: u32) -> vec2<u32> {
    let wx = i32(texwin & 31u);
    let wy = i32((texwin >> 5u) & 31u);
    let ox = i32((texwin >> 10u) & 31u);
    let oy = i32((texwin >> 15u) & 31u);
    let u = (u_in & ~(wx * 8)) | ((ox & wx) * 8);
    let v = (v_in & ~(wy * 8)) | ((oy & wy) * 8);
    let tx = i32((tpage & 15u) * 64u);
    let ty = select(0, 256, (tpage & 16u) != 0u);
    let ud = u & 255;
    let vd = v & 255;
    let depth = (tpage >> 7u) & 3u;
    var c: u32;
    if depth <= 1u {
        let shift = u32(select((ud & 1) * 8, (ud & 3) * 4, depth == 0u));
        let cx = i32(clut & 63u) * 16;
        let cy = i32((clut >> 6u) & 511u);
        let w = vram_px(tx + ud / select(2, 4, depth == 0u), ty + vd);
        let idx = (w >> shift) & select(0xFFu, 0xFu, depth == 0u);
        c = vram_px(cx + i32(idx), cy);
    } else {
        c = vram_px(tx + ud, ty + vd);
    }
    return vec2<u32>(c & 0x7FFFu, c >> 15u);
}

fn lo16(v: u32) -> i32 { return bitcast<i32>(v << 16u) >> 16; }
fn hi16(v: u32) -> i32 { return bitcast<i32>(v) >> 16; }

// Optional bilinear filtering is applied after CLUT decode, bounded to
// the primitive's own UV rectangle so adjacent VRAM textures never bleed.
// Transparent palette entries are omitted and remaining weights renormalized.
fn filtered_texel(
    uv: vec2<f32>,
    tpage: u32,
    clut: u32,
    texwin: u32,
    bounds: vec4<i32>,
) -> vec4<f32> {
    let base = vec2<i32>(floor(uv));
    if uni.tex_filter < 0.5 {
        let t = texel(base.x, base.y, tpage, clut, texwin);
        if t.x == 0u && t.y == 0u { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
        return vec4<f32>(
            f32(t.x & 31u),
            f32((t.x >> 5u) & 31u),
            f32((t.x >> 10u) & 31u),
            f32(t.y),
        );
    }
    let f = fract(uv);
    let coords = array<vec2<i32>, 4>(
        base,
        base + vec2<i32>(1, 0),
        base + vec2<i32>(0, 1),
        base + vec2<i32>(1, 1),
    );
    let weights = array<f32, 4>(
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y,
    );
    var rgb = vec3<f32>(0.0);
    var semi_weight = 0.0;
    var total = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        let p = clamp(coords[i], bounds.xy, bounds.zw);
        let t = texel(p.x, p.y, tpage, clut, texwin);
        if t.x != 0u || t.y != 0u {
            let weight = weights[i];
            rgb += vec3<f32>(
                f32(t.x & 31u),
                f32((t.x >> 5u) & 31u),
                f32((t.x >> 10u) & 31u),
            ) * weight;
            total += weight;
            semi_weight += weight * f32(t.y);
        }
    }
    if total <= 0.0 { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
    return vec4<f32>(rgb / total, select(0.0, 1.0, semi_weight >= total * 0.5));
}

@fragment fn fs(in: Vout) -> @location(0) vec4<f32> {
    let tpage = in.p0 & 0xFFFFu;
    let clut = in.p0 >> 16u;
    let texwin = in.p1 & 0xFFFFFu;
    let flags = in.flags;
    let isc = i32(uni.scale);
    if (flags & F_RESOLVE) != 0u {
        let c = vram_px(i32(in.pos.x) / isc, i32(in.pos.y) / isc);
        let c5 = vec3<i32>(i32(c & 31u), i32((c >> 5u) & 31u), i32((c >> 10u) & 31u));
        let e = (c5 << vec3<u32>(3u)) | (c5 >> vec3<u32>(2u));
        return vec4<f32>(vec3<f32>(e) / 255.0, f32(c >> 15u));
    }
    var rgb8: vec3<i32>;
    var semi = 0u;
    if (flags & F_TEX) != 0u {
        let bounds = vec4<f32>(
            f32(lo16(in.p3)),
            f32(lo16(in.p4)),
            f32(hi16(in.p3)),
            f32(hi16(in.p4)),
        );
        let uv = clamp(in.uv, bounds.xy, bounds.zw);
        let t = filtered_texel(uv, tpage, clut, texwin, vec4<i32>(bounds));
        if t.a < 0.0 { discard; }
        if (flags & F_ONLY_OPAQUE) != 0u && t.a > 0.5 { discard; }
        if (flags & F_ONLY_SEMI) != 0u && t.a <= 0.5 { discard; }
        semi = u32(t.a > 0.5);
        let t5 = vec3<i32>(t.rgb);
        if (flags & F_RAW) != 0u {
            rgb8 = (t5 << vec3<u32>(3u)) | (t5 >> vec3<u32>(2u));
        } else {
            let mc = vec3<i32>(i32(in.col.r), i32(in.col.g), i32(in.col.b));
            rgb8 = min(t5 * mc / 16, vec3<i32>(255));
        }
    } else {
        rgb8 = vec3<i32>(i32(in.col.r), i32(in.col.g), i32(in.col.b));
    }
    if (flags & F_DITHER) != 0u {
        let nx = (i32(in.pos.x) / isc) & 3;
        let ny = (i32(in.pos.y) / isc) & 3;
        rgb8 += vec3<i32>(DT[ny][nx]);
    }
    let q = clamp(rgb8, vec3<i32>(0), vec3<i32>(255)) >> vec3<u32>(3u);
    let o8 = (q << vec3<u32>(3u)) | (q >> vec3<u32>(2u));
    let mask = f32((flags & F_MASK_SET) != 0u || semi != 0u);
    return vec4<f32>(vec3<f32>(o8) / 255.0, mask);
}
"#;

const SCANOUT_SHADER: &str = r#"
struct SU { ox: f32, oy: f32, cw: f32, ch: f32, output_srgb: f32, scale: f32, crt: f32, pad0: f32 }
@group(0) @binding(0) var<uniform> su: SU;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0), vec2<f32>(-1.0, -3.0));
    var uv = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0));
    var o: Out;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.uv = uv[i];
    return o;
}
@fragment fn fs(in: Out) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(img));
    let tuv = (vec2<f32>(su.ox, su.oy) + in.uv * vec2<f32>(su.cw, su.ch)) / dims;
    let sample = textureSample(img, smp, tuv);
    var color = sample.rgb;
    if su.crt > 0.5 {
        let source = vec2<f32>(su.ox, su.oy) + in.uv * vec2<f32>(su.cw, su.ch);
        let native_x = source.x / su.scale;
        let native_y = source.y / su.scale;
        let line_phase = fract(native_y);
        let line = 0.90 + 0.10 * (1.0 - abs(2.0 * line_phase - 1.0));
        let phase = u32(floor(native_x * 3.0)) % 3u;
        let mask = vec3<f32>(
            select(0.965, 1.0, phase == 0u),
            select(0.965, 1.0, phase == 1u),
            select(0.965, 1.0, phase == 2u),
        );
        color *= mask * line;
    }
    if su.output_srgb > 0.5 {
        let lo = color / 12.92;
        let hi = pow((color + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
        return vec4<f32>(select(hi, lo, color <= vec3<f32>(0.04045)), sample.a);
    }
    return vec4<f32>(color, sample.a);
}
"#;

impl HwRenderer {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scale: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let (tw, th) = (1024 * scale, 512 * scale);
        let output_srgb = surface_format.is_srgb();
        let vram_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PSX native VRAM"),
            size: wgpu::Extent3d {
                width: 1024,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let vram_view = vram_tex.create_view(&Default::default());
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PSX scaled render target"),
            size: wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PSX hw uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &uniform,
            0,
            bytemuck::cast_slice(&[scale as f32, tw as f32, th as f32, 0.0]),
        );
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PSX hw layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let vram_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PSX hw bind"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&vram_view),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PSX hw shader"),
            source: wgpu::ShaderSource::Wgsl(HW_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PSX hw pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let vtx_attrs: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 32,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 36,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 40,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 44,
                shader_location: 6,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 48,
                shader_location: 7,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 52,
                shader_location: 8,
            },
        ];
        use wgpu::BlendFactor as a;
        let o = wgpu::BlendOperation::Add;
        let rev = wgpu::BlendOperation::ReverseSubtract;
        // alpha write-through where no mask logic applies
        let alpha_replace = wgpu::BlendComponent {
            src_factor: a::One,
            dst_factor: a::Zero,
            operation: o,
        };
        let keep_mask = wgpu::BlendComponent {
            src_factor: a::OneMinusDstAlpha,
            dst_factor: a::DstAlpha,
            operation: o,
        };
        let with_alpha = |color: wgpu::BlendComponent, alpha| wgpu::BlendState { color, alpha };
        let blends: [Option<wgpu::BlendState>; PIPES] = [
            Some(with_alpha(wgpu::BlendComponent::REPLACE, alpha_replace)),
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::OneMinusDstAlpha,
                    dst_factor: a::DstAlpha,
                    operation: o,
                },
                keep_mask,
            )),
            // ST0: dst*0.5 + src*0.5 via blend constant 0.5
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::Constant,
                    dst_factor: a::Constant,
                    operation: o,
                },
                alpha_replace,
            )),
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::One,
                    dst_factor: a::One,
                    operation: o,
                },
                alpha_replace,
            )),
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::One,
                    dst_factor: a::One,
                    operation: rev,
                },
                alpha_replace,
            )),
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::Constant,
                    dst_factor: a::One,
                    operation: o,
                },
                alpha_replace,
            )),
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::OneMinusDstAlpha,
                    dst_factor: a::One,
                    operation: o,
                },
                keep_mask,
            )),
            Some(with_alpha(
                wgpu::BlendComponent {
                    src_factor: a::OneMinusDstAlpha,
                    dst_factor: a::One,
                    operation: rev,
                },
                keep_mask,
            )),
        ];
        let pipes = blends
            .iter()
            .enumerate()
            .map(|(_, blend)| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("PSX hw pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<GpuVert>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: vtx_attrs,
                        })],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: *blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            })
            .collect();
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PSX hw verts"),
            size: 8 << 20,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PSX hw indices"),
            size: 4 << 20,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // scanout pipeline: samples a crop of the scaled target
        let scanout_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PSX scanout layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let scanout_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PSX scanout shader"),
            source: wgpu::ShaderSource::Wgsl(SCANOUT_SHADER.into()),
        });
        let scanout_layout_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PSX scanout pipeline layout"),
            bind_group_layouts: &[Some(&scanout_layout)],
            immediate_size: 0,
        });
        let scanout_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PSX scanout uniform"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = |filter| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        let mk_bind = |s: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PSX scanout bind"),
                layout: &scanout_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scanout_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&target_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(s),
                    },
                ],
            })
        };
        let bind_linear = mk_bind(&sampler(wgpu::FilterMode::Linear));
        let bind_nearest = mk_bind(&sampler(wgpu::FilterMode::Nearest));
        Self {
            scale,
            output_srgb,
            vram_tex,
            vram_view,
            target,
            target_view,
            uniform,
            vram_bind,
            pipes,
            vbuf,
            ibuf,
            verts: Vec::new(),
            indices: Vec::new(),
            draws: Vec::new(),
            scratch: Vec::new(),
            vram_uploaded: false,
            scanout: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("PSX scanout pipeline"),
                layout: Some(&scanout_layout_pl),
                vertex: wgpu::VertexState {
                    module: &scanout_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &scanout_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            }),
            scanout_uniform,
            scanout_layout,
            bind_linear,
            bind_nearest,
        }
    }

    /// Queue the native-VRAM upload of a rect from the CPU mirror.
    fn upload_rect(
        &mut self,
        queue: &wgpu::Queue,
        vram: &[u16],
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) {
        let (w, h) = (x1 - x0, y1 - y0);
        if w == 0 || h == 0 {
            return;
        }
        self.scratch.clear();
        self.scratch.reserve((w * h) as usize);
        for y in y0..y1 {
            let row = (y as usize) * 1024 + x0 as usize;
            self.scratch.extend_from_slice(&vram[row..row + w as usize]);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.vram_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.scratch),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 2),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Emit vertices + draw entries for one primitive.
    fn emit_prim(&mut self, p: &HwPrim) {
        let flags = p.flags as u32;
        let p0 = p.tpage as u32 | ((p.clut as u32) << 16);
        let p1 = p.texwin & 0xF_FFFF;
        let p2 = p.col & 0xFF_FFFF;
        let pack_bounds = |lo: i16, hi: i16| lo as u16 as u32 | ((hi as u16 as u32) << 16);
        let p3 = pack_bounds(p.uv_bounds[0], p.uv_bounds[2]);
        let p4 = pack_bounds(p.uv_bounds[1], p.uv_bounds[3]);
        let mk = |x: f32, y: f32, u: f32, v: f32, c: [f32; 4], flags: u32| GpuVert {
            pos: [x, y],
            uv: [u, v],
            col: c,
            p0,
            p1,
            p2,
            flags,
            p3,
            p4,
        };
        let col_of = |v: &aruanpsx::gpu::HwVert| [v.r as f32, v.g as f32, v.b as f32, 255.0];
        // expanded vertex list for the prim
        let mut out: Vec<GpuVert> = Vec::with_capacity(4);
        if p.flags & HF::LINE != 0 {
            let (a, b) = (&p.verts[0], &p.verts[1]);
            let (dx, dy) = ((b.x - a.x) as f32, (b.y - a.y) as f32);
            let len = (dx * dx + dy * dy).sqrt().max(1e-6);
            let (px, py) = (-dy / len * 0.5, dx / len * 0.5);
            let (ax, ay, bx, by) = (a.x as f32, a.y as f32, b.x as f32, b.y as f32);
            let (ca, cb) = (col_of(a), col_of(b));
            out.push(mk(ax - px, ay - py, 0.0, 0.0, ca, flags));
            out.push(mk(bx - px, by - py, 0.0, 0.0, cb, flags));
            out.push(mk(ax + px, ay + py, 0.0, 0.0, ca, flags));
            out.push(mk(bx + px, by + py, 0.0, 0.0, cb, flags));
        } else {
            for v in &p.verts[..p.n as usize] {
                let (mut u, mut vv) = (v.u as f32, v.v as f32);
                if p.flags & HF::RECT != 0 {
                    // Core vertices carry the rect UV boundaries; shift
                    // them to texel centers for pixel-center interpolation.
                    u = v.u as f32 - 0.5;
                    vv = v.v as f32 - 0.5;
                }
                out.push(mk(v.x as f32, v.y as f32, u, vv, col_of(v), flags));
            }
        }
        let idx: &[u32] = if p.flags & HF::LINE != 0 || p.n == 4 {
            &[0, 1, 2, 1, 2, 3]
        } else {
            &[0, 1, 2]
        };
        let sc = self.scale;
        let scissor = [
            (p.clip[0] as u32 * sc).min(1024 * sc),
            (p.clip[1] as u32 * sc).min(512 * sc),
            ((p.clip[2] - p.clip[0]).max(0) as u32 * sc).min(1024 * sc),
            ((p.clip[3] - p.clip[1]).max(0) as u32 * sc).min(512 * sc),
        ];
        let semi_mode = ((p.tpage >> 5) & 3) as usize;
        let mc = p.flags & HF::MASK_CHECK != 0;
        let st_pipe = |mode: usize, mc: bool| -> usize {
            match (mode, mc) {
                (1, true) => PIPE_ST1_MC,
                (2, true) => PIPE_ST2_MC,
                // avg/quarter + mask-check can't fold the constant into
                // the alpha trick; approximate by ignoring mask-check
                (m, _) => [PIPE_ST0, PIPE_ST1, PIPE_ST2, PIPE_ST3][m],
            }
        };
        let opaque_pipe = if mc { PIPE_OPAQUE_MC } else { PIPE_OPAQUE };
        // (pipeline, extra shader flags) per draw pass
        let passes: &[(usize, u32)] = if p.flags & HF::TEXTURED != 0 && p.flags & HF::SEMITRANS != 0
        {
            &[
                (opaque_pipe, HF::DRAW_OPAQUE_TEXELS as u32),
                (st_pipe(semi_mode, mc), HF::DRAW_SEMI_TEXELS as u32),
            ]
        } else if p.flags & HF::SEMITRANS != 0 {
            &[(st_pipe(semi_mode, mc), 0)]
        } else {
            &[(opaque_pipe, 0)]
        };
        for &(pipe, extra) in passes {
            let pass_base = self.verts.len() as u32;
            for v in &out {
                let mut v = *v;
                v.flags = flags | extra;
                self.verts.push(v);
            }
            let first_index = self.indices.len() as u32;
            for &i in idx {
                self.indices.push(pass_base + i);
            }
            let count = idx.len() as u32;
            if let Some(last) = self.draws.last_mut() {
                if last.pipe == pipe && last.scissor == scissor {
                    last.index_count += count;
                    continue;
                }
            }
            self.draws.push(HwDraw {
                pipe,
                scissor,
                first_index,
                index_count: count,
            });
        }
    }

    /// Emit a resolve quad: samples the native VRAM texture per native
    /// pixel and writes it into the scaled target.
    fn emit_resolve(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) {
        let base = self.verts.len() as u32;
        let flags = RESOLVE_F;
        let mk = |x: f32, y: f32| GpuVert {
            pos: [x, y],
            uv: [0.0; 2],
            col: [0.0; 4],
            p0: 0,
            p1: 0,
            p2: 0,
            flags,
            p3: 0,
            p4: 0,
        };
        let (fx0, fy0, fx1, fy1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
        self.verts
            .extend_from_slice(&[mk(fx0, fy0), mk(fx1, fy0), mk(fx0, fy1), mk(fx1, fy1)]);
        let first_index = self.indices.len() as u32;
        for &i in &[0u32, 1, 2, 1, 2, 3] {
            self.indices.push(base + i);
        }
        let sc = self.scale;
        self.draws.push(HwDraw {
            pipe: PIPE_OPAQUE,
            scissor: [x0 * sc, y0 * sc, (x1 - x0) * sc, (y1 - y0) * sc],
            first_index,
            index_count: 6,
        });
    }

    /// Replay one frame of ops: uploads first (texture data), then the
    /// draw list into the scaled target.
    fn replay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ops: &[HwOp],
        vram: &[u16],
        texture_filter: bool,
    ) {
        let (tw, th) = (1024 * self.scale, 512 * self.scale);
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[
                self.scale as f32,
                tw as f32,
                th as f32,
                texture_filter as u8 as f32,
            ]),
        );
        if !self.vram_uploaded {
            self.upload_rect(queue, vram, 0, 0, 1024, 512);
            self.vram_uploaded = true;
        }
        self.verts.clear();
        self.indices.clear();
        self.draws.clear();
        for op in ops {
            match *op {
                HwOp::Upload { x0, y0, x1, y1 } | HwOp::Resolve { x0, y0, x1, y1 } => {
                    self.upload_rect(
                        queue,
                        vram,
                        x0 as u32,
                        y0 as u32,
                        (x1 as u32).min(1024),
                        (y1 as u32).min(512),
                    );
                    if matches!(op, HwOp::Resolve { .. }) {
                        self.emit_resolve(
                            x0 as u32,
                            y0 as u32,
                            (x1 as u32).min(1024),
                            (y1 as u32).min(512),
                        );
                    }
                }
                HwOp::Prim(p) => self.emit_prim(&p),
            }
        }
        if self.verts.is_empty() {
            return;
        }
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&self.verts));
        queue.write_buffer(&self.ibuf, 0, bytemuck::cast_slice(&self.indices));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PSX hw encoder"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PSX hw pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_bind_group(0, &self.vram_bind, &[]);
            for d in &self.draws {
                pass.set_scissor_rect(d.scissor[0], d.scissor[1], d.scissor[2], d.scissor[3]);
                pass.set_blend_constant(match d.pipe {
                    PIPE_ST0 => wgpu::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.5,
                        a: 0.5,
                    },
                    PIPE_ST3 => wgpu::Color {
                        r: 0.25,
                        g: 0.25,
                        b: 0.25,
                        a: 0.25,
                    },
                    _ => wgpu::Color::WHITE,
                });
                pass.set_pipeline(&self.pipes[d.pipe]);
                pass.draw_indexed(d.first_index..d.first_index + d.index_count, 0, 0..1);
            }
        }
        queue.submit([enc.finish()]);
    }

    /// Draw the display crop of the scaled target to the surface view.
    fn present(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        viewport: (f32, f32, f32, f32),
        crop: (u32, u32, u32, u32),
        smooth: bool,
        crt: bool,
    ) {
        let (x0, y0, x1, y1) = crop;
        let sc = self.scale;
        queue.write_buffer(
            &self.scanout_uniform,
            0,
            bytemuck::cast_slice(&[
                (x0 * sc) as f32,
                (y0 * sc) as f32,
                ((x1 - x0) * sc) as f32,
                ((y1 - y0) * sc) as f32,
                self.output_srgb as u8 as f32,
                sc as f32,
                crt as u8 as f32,
                0.0,
            ]),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PSX hw scanout"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_viewport(viewport.0, viewport.1, viewport.2, viewport.3, 0.0, 1.0);
        pass.set_pipeline(&self.scanout);
        pass.set_bind_group(
            0,
            if smooth {
                &self.bind_linear
            } else {
                &self.bind_nearest
            },
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

/// One emulated frame delivered to the presentation thread. Hardware
/// frames carry the host-GPU op list plus a VRAM snapshot for upload
/// ops; CPU frames carry the rendered framebuffer directly.
enum FrameData {
    Hw {
        ops: Vec<HwOp>,
        vram: Arc<Vec<u16>>,
        crop: (u32, u32, u32, u32),
    },
    Cpu {
        pixels: Vec<u32>,
        w: usize,
        h: usize,
        crt_scale: f32,
    },
}

/// Commands the UI can send the emulation thread.
enum EmuCmd {
    ToggleDither,
}

/// State shared between the UI thread and the emulation thread.
struct EmuShared {
    /// Latest merged pad bits (pressed = 1; the emulator inverts).
    pad_bits: AtomicU32,
    turbo: AtomicBool,
    quit: AtomicBool,
    /// Total emulated frames / nanoseconds, for the title stats.
    frames: AtomicU64,
    emu_ns: AtomicU64,
    refresh_bits: AtomicU64,
    mailbox: Mutex<Option<FrameData>>,
}

/// Emulation thread: owns the Emulator, paces frames on a fixed grid,
/// feeds the lock-free audio ring, and publishes each finished frame to
/// the mailbox. Because audio production lives here, modal window
/// resizes, mouse clicks, and presentation stalls can never starve the
/// audio callback.
fn emu_main(
    mut emu: Emulator,
    queue: AudioQueue,
    audio_enabled: bool,
    proxy: EventLoopProxy<()>,
    cmd_rx: Receiver<EmuCmd>,
    shared: Arc<EmuShared>,
    card_path: String,
) {
    let mut next_frame = seeded_frame_deadline(Instant::now());
    let mut last_vram: Option<Arc<Vec<u16>>> = None;
    let mut fb = Vec::new();
    while !shared.quit.load(Ordering::Relaxed) {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                EmuCmd::ToggleDither => emu.bus.gpu.dithering = !emu.bus.gpu.dithering,
            }
        }
        let turbo = shared.turbo.load(Ordering::Relaxed);
        let now = Instant::now();
        if !turbo && now < next_frame {
            let rem = next_frame - now;
            if rem > Duration::from_millis(2) {
                // Sleep the bulk of the slot (timeBeginPeriod keeps this
                // near 1 ms granularity), then spin the last stretch.
                thread::sleep(rem - Duration::from_millis(2));
            } else {
                std::hint::spin_loop();
            }
            continue;
        }

        let started = Instant::now();
        emu.set_pad(PadInput(!(shared.pad_bits.load(Ordering::Relaxed) as u16)));
        emu.run_frame();
        let samples = emu.take_audio();
        if audio_enabled && !turbo {
            for s in samples.chunks_exact(2) {
                queue.push((s[0], s[1]));
            }
        }

        let hw_render = emu.hw_render();
        let is24 = emu.gpu_display_is_24bpp();
        let frame = if hw_render && !is24 {
            let ops = emu.take_gpu_hw_ops();
            // The VRAM snapshot only matters for Upload/Resolve ops;
            // reuse the last one when this frame has none.
            if ops
                .iter()
                .any(|op| matches!(op, HwOp::Upload { .. } | HwOp::Resolve { .. }))
                || last_vram.is_none()
            {
                last_vram = Some(Arc::new(emu.bus.gpu.vram.clone()));
            }
            FrameData::Hw {
                ops,
                vram: last_vram.clone().unwrap(),
                crop: emu.gpu_display_crop(),
            }
        } else {
            if hw_render {
                let _ = emu.take_gpu_hw_ops();
            }
            fb.clear();
            let (w, h) = emu.framebuffer(&mut fb);
            let crt_scale = if is24 {
                1.0
            } else {
                emu.bus.gpu.internal_scale() as f32
            };
            FrameData::Cpu {
                pixels: std::mem::take(&mut fb),
                w,
                h,
                crt_scale,
            }
        };

        {
            let mut slot = shared.mailbox.lock().unwrap();
            match (&mut *slot, frame) {
                (
                    Some(FrameData::Hw { ops, vram, crop }),
                    FrameData::Hw {
                        ops: new_ops,
                        vram: new_vram,
                        crop: new_crop,
                    },
                ) => {
                    ops.extend(new_ops);
                    *vram = new_vram;
                    *crop = new_crop;
                }
                (slot, frame) => *slot = Some(frame),
            }
        }
        let _ = proxy.send_event(());
        shared.frames.fetch_add(1, Ordering::Relaxed);
        shared
            .emu_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        shared
            .refresh_bits
            .store(emu.refresh_rate().to_bits(), Ordering::Relaxed);

        let period = Duration::from_secs_f64(1.0 / emu.refresh_rate());
        next_frame = if turbo {
            Instant::now()
        } else {
            advance_frame_deadline(next_frame, period, Instant::now())
        };
    }
    let _ = std::fs::write(&card_path, emu.bus.sio.card_image(0));
}

struct App {
    queue: AudioQueue,
    _audio: Option<cpal::Stream>,
    gilrs: Option<Gilrs>,
    /// Raw gilrs event codes currently held down, for pads whose
    /// Start/Select arrive unmapped (Button::Unknown or wrong names).
    pressed_pad_buttons: HashSet<(usize, u32)>,
    pad_start_code: Option<u32>,
    pad_select_code: Option<u32>,
    winmm_pads: Vec<u32>,
    keys: HashSet<KeyCode>,
    window: Option<Arc<Window>>,
    presenter: Option<GpuPresenter>,
    hw_render_enabled: bool,
    hw_scale: u32,
    smooth: bool,
    texture_filter: bool,
    crt_filter: bool,
    shared: Arc<EmuShared>,
    cmd_tx: Sender<EmuCmd>,
    emu_thread: Option<JoinHandle<()>>,
    /// Last CPU frame and hw crop, re-presented when no new frame exists
    /// (e.g. resize-triggered presents between emulated frames).
    last_cpu_frame: Option<(Vec<u32>, usize, usize, f32)>,
    last_hw_crop: Option<(u32, u32, u32, u32)>,
    redraw_pending: bool,
    present_due: bool,
    resize_deadline: Option<Instant>,
    last_resize_present: Instant,
    frame_start: Instant,
    last_emu_frames: u64,
    last_emu_ns: u64,
    present_count: u32,
    output_time: Duration,
    output_size: PhysicalSize<u32>,
    title: String,
}

impl App {
    /// Pump gamepad/winmm input and publish the merged pad bits for the
    /// emulation thread.
    fn poll_input(&mut self) {
        if let Some(g) = self.gilrs.as_mut() {
            while let Some(ev) = g.next_event() {
                match ev.event {
                    EventType::ButtonPressed(_, code)
                    | EventType::ButtonChanged(_, _, code)
                    | EventType::ButtonRepeated(_, code) => {
                        self.pressed_pad_buttons
                            .insert((usize::from(ev.id), code.into_u32()));
                    }
                    EventType::ButtonReleased(_, code) => {
                        self.pressed_pad_buttons
                            .remove(&(usize::from(ev.id), code.into_u32()));
                    }
                    EventType::Connected => {
                        let pad = g.gamepad(ev.id);
                        eprintln!(
                            "gamepad {} connected: {} mapping={:?}",
                            usize::from(ev.id),
                            pad.name(),
                            pad.mapping_source(),
                        );
                    }
                    EventType::Disconnected => {
                        eprintln!("gamepad {} disconnected", usize::from(ev.id));
                    }
                    _ => {}
                }
            }
        }
        let mut pad_bits = pad_state(
            &self.keys,
            self.gilrs.as_ref(),
            &self.pressed_pad_buttons,
            self.pad_start_code,
            self.pad_select_code,
        );
        // winmm sees pads gilrs can't (non-WGI HID devices). Always take
        // Select/Start/d-pad from it; take the full conventional mapping
        // only when gilrs found no gamepads at all.
        let gilrs_padless = self
            .gilrs
            .as_ref()
            .is_none_or(|g| g.gamepads().next().is_none());
        for &id in &self.winmm_pads {
            if let Some(info) = read_winmm_pad(id) {
                pad_bits |= winmm_info_bits(&info, gilrs_padless);
            }
        }
        self.shared
            .pad_bits
            .store(pad_bits as u32, Ordering::Relaxed);
    }

    fn request_present(&mut self) {
        if self.redraw_pending {
            return;
        }
        if let Some(window) = &self.window {
            window.request_redraw();
            self.redraw_pending = true;
            self.present_due = true;
        }
    }

    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        self.shared.quit.store(true, Ordering::Relaxed);
        if let Some(handle) = self.emu_thread.take() {
            let _ = handle.join();
        }
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title(&self.title)
                        .with_inner_size(self.output_size),
                )
                .expect("create window"),
        );
        let mut presenter = pollster::block_on(GpuPresenter::new(window.clone()));
        if self.hw_render_enabled {
            presenter.enable_hw(self.hw_scale);
        }
        self.presenter = Some(presenter);
        self.window = Some(window);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        // The emulation thread finished a frame: present it.
        self.request_present();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if id != window.id() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => self.shutdown(event_loop),
            WindowEvent::Resized(_) => {
                let now = Instant::now();
                // Let the drag settle before a final swapchain reconfigure.
                self.resize_deadline = Some(now + Duration::from_millis(80));
                // Modal size/move loops still deliver resize events even
                // while about_to_wait is starved; keep presenting the
                // newest frame so video keeps updating mid-drag.
                if now.duration_since(self.last_resize_present) >= Duration::from_millis(33) {
                    self.last_resize_present = now;
                    self.request_present();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys.insert(code);
                    } else {
                        self.keys.remove(&code);
                    }
                    if code == KeyCode::Escape && event.state == ElementState::Pressed {
                        self.shutdown(event_loop);
                    }
                    if code == KeyCode::KeyF
                        && event.state == ElementState::Pressed
                        && !event.repeat
                    {
                        self.smooth = !self.smooth;
                    }
                    if code == KeyCode::KeyT
                        && event.state == ElementState::Pressed
                        && !event.repeat
                    {
                        self.texture_filter = !self.texture_filter;
                        eprintln!(
                            "polygon texture filtering: {}",
                            if self.texture_filter {
                                "bilinear"
                            } else {
                                "nearest"
                            }
                        );
                    }
                    if code == KeyCode::KeyC
                        && event.state == ElementState::Pressed
                        && !event.repeat
                    {
                        self.crt_filter = !self.crt_filter;
                        // CRT mode pairs with bilinear texture sampling;
                        // non-CRT mode shows the raw nearest-pixel image.
                        self.texture_filter = self.crt_filter;
                        eprintln!(
                            "CRT filter: {}",
                            if self.crt_filter {
                                "on (bilinear)"
                            } else {
                                "off (nearest)"
                            }
                        );
                    }
                    if code == KeyCode::KeyD
                        && event.state == ElementState::Pressed
                        && !event.repeat
                    {
                        let _ = self.cmd_tx.send(EmuCmd::ToggleDither);
                    }
                    if code == KeyCode::Tab {
                        self.shared
                            .turbo
                            .store(event.state == ElementState::Pressed, Ordering::Relaxed);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.redraw_pending = false;
                let present_due = std::mem::take(&mut self.present_due);
                let frame = self.shared.mailbox.lock().unwrap().take();
                let mut presented = false;
                if present_due {
                    let output_started = Instant::now();
                    if let Some(presenter) = self.presenter.as_mut() {
                        match frame {
                            Some(FrameData::Hw { ops, vram, crop }) => {
                                presenter.replay_hw(&ops, &vram, self.texture_filter);
                                presenter.render_hw(window, crop, self.smooth, self.crt_filter);
                                self.last_hw_crop = Some(crop);
                                presented = true;
                            }
                            Some(FrameData::Cpu {
                                pixels,
                                w,
                                h,
                                crt_scale,
                            }) => {
                                presenter.render(
                                    window,
                                    &pixels,
                                    w,
                                    h,
                                    self.smooth,
                                    self.crt_filter,
                                    crt_scale,
                                );
                                self.last_cpu_frame = Some((pixels, w, h, crt_scale));
                                presented = true;
                            }
                            None => {
                                // Resize/refresh present with no new frame:
                                // redraw the most recent content.
                                if let Some(crop) = self.last_hw_crop {
                                    presenter.render_hw(window, crop, self.smooth, self.crt_filter);
                                } else if let Some((pixels, w, h, scale)) = &self.last_cpu_frame {
                                    presenter.render(
                                        window,
                                        pixels,
                                        *w,
                                        *h,
                                        self.smooth,
                                        self.crt_filter,
                                        *scale,
                                    );
                                }
                            }
                        }
                    }
                    self.output_time += output_started.elapsed();
                    if presented {
                        self.present_count += 1;
                    }
                }
                if self.frame_start.elapsed() >= Duration::from_secs(1) {
                    let elapsed = self.frame_start.elapsed().as_secs_f64();
                    let emu_frames = self.shared.frames.load(Ordering::Relaxed);
                    let emu_ns = self.shared.emu_ns.load(Ordering::Relaxed);
                    let frame_delta = (emu_frames - self.last_emu_frames).max(1) as f64;
                    let emu_fps = (emu_frames - self.last_emu_frames) as f64 / elapsed;
                    let present_fps = self.present_count as f64 / elapsed;
                    let emu_ms = (emu_ns - self.last_emu_ns) as f64 / 1e6 / frame_delta;
                    let output_ms =
                        self.output_time.as_secs_f64() * 1000.0 / self.present_count.max(1) as f64;
                    let audio_underruns = self.queue.underruns.swap(0, Ordering::Relaxed);
                    let audio_overruns = self.queue.overruns.swap(0, Ordering::Relaxed);
                    let target_hz =
                        f64::from_bits(self.shared.refresh_bits.load(Ordering::Relaxed));
                    let tex_filter = if self.texture_filter {
                        "bilinear tex"
                    } else {
                        "nearest tex"
                    };
                    let crt_filter = if self.crt_filter { "CRT" } else { "no CRT" };
                    let gpu = self
                        .presenter
                        .as_ref()
                        .map(|p| p.adapter_name.as_str())
                        .unwrap_or("GPU");
                    window.set_title(&format!(
                        "{} — emu {:.0} / present {:.0} fps • {:.1} Hz target • emu {:.1} ms + output {:.1} ms • audio U{} O{} • {} • {} • {}",
                        self.title,
                        emu_fps,
                        present_fps,
                        target_hz,
                        emu_ms,
                        output_ms,
                        audio_underruns,
                        audio_overruns,
                        tex_filter,
                        crt_filter,
                        gpu
                    ));
                    self.last_emu_frames = emu_frames;
                    self.last_emu_ns = emu_ns;
                    self.present_count = 0;
                    self.frame_start = Instant::now();
                    self.output_time = Duration::ZERO;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.poll_input();
        let now = Instant::now();
        if self.resize_deadline.is_some_and(|deadline| now >= deadline) {
            // Drag settled: one final present reconfigures the swapchain.
            self.resize_deadline = None;
            self.last_resize_present = now;
            self.request_present();
        }
        // Idle tick: keeps gamepad input fresh and fires the resize
        // settle even while no frames or events arrive. Emulation and
        // audio run on their own thread, independent of this loop.
        event_loop.set_control_flow(ControlFlow::WaitUntil(now + Duration::from_millis(4)));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fast = args.iter().any(|a| a == "--fast");
    let dynarec = args.iter().any(|a| a == "--dynarec");
    let sw_render = args.iter().any(|a| a == "--sw");
    let mute = args.iter().any(|a| a == "--mute");
    let mut internal_scale = 1;
    let mut texture_filter = None;
    let mut crt_filter = None;
    let mut pad_start_code = None;
    let mut pad_select_code = None;
    let mut window_size = PhysicalSize::new(960, 720);
    for arg in &args {
        if let Some(value) = arg.strip_prefix("--res=") {
            internal_scale = parse_internal_scale(value).unwrap_or_else(|error| {
                eprintln!("invalid --res value {value:?}: {error}");
                std::process::exit(2);
            });
        }
        if let Some(value) = arg.strip_prefix("--window-size=") {
            window_size = parse_window_size(value).unwrap_or_else(|error| {
                eprintln!("invalid --window-size value {value:?}: {error}");
                std::process::exit(2);
            });
        }
        if let Some(value) = arg.strip_prefix("--texture-filter=") {
            texture_filter = Some(parse_texture_filter(value).unwrap_or_else(|error| {
                eprintln!("invalid --texture-filter value {value:?}: {error}");
                std::process::exit(2);
            }));
        }
        if let Some(value) = arg.strip_prefix("--crt=") {
            crt_filter = Some(parse_crt_filter(value).unwrap_or_else(|error| {
                eprintln!("invalid --crt value {value:?}: {error}");
                std::process::exit(2);
            }));
        }
        if let Some(value) = arg.strip_prefix("--pad-start-code=") {
            pad_start_code = Some(value.parse::<u32>().unwrap_or_else(|_| {
                eprintln!("invalid --pad-start-code value {value:?}: expected a raw code number");
                std::process::exit(2);
            }));
        }
        if let Some(value) = arg.strip_prefix("--pad-select-code=") {
            pad_select_code = Some(value.parse::<u32>().unwrap_or_else(|_| {
                eprintln!("invalid --pad-select-code value {value:?}: expected a raw code number");
                std::process::exit(2);
            }));
        }
    }
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let Some(cue) = pos.first() else {
        eprintln!(
            "usage: play_gpu <game.cue> [bios.bin] [--fast] [--dynarec] [--mute] [--res=2x|4x|6x|8x|WIDTHxHEIGHT] [--texture-filter=nearest|bilinear] [--crt=on|off]"
        );
        eprintln!(
            "Use --window-size=WIDTHxHEIGHT to resize only the window; press C to toggle CRT."
        );
        std::process::exit(1);
    };
    let bios_path = pos.get(1).map(|s| s.as_str()).unwrap_or(DEFAULT_BIOS);
    let mut emulator = Emulator::new(std::fs::read(bios_path).expect("read BIOS"));
    emulator.set_dynarec_enabled(dynarec);
    emulator.set_hw_render(internal_scale > 1 && !sw_render);
    if !emulator.set_internal_resolution_scale(internal_scale) {
        eprintln!("internal resolution scale {internal_scale}x is unsupported");
        std::process::exit(2);
    }
    emulator.insert_disc(Disc::load(std::path::Path::new(cue.as_str())).expect("load disc"));
    if fast && !emulator.fast_boot_disc() {
        eprintln!("fast boot unavailable; using BIOS boot");
    }
    let card_path = format!("{cue}.mcd");
    if let Ok(image) = std::fs::read(&card_path) {
        emulator.bus.sio.set_card_image(0, &image);
    }
    let queue = Arc::new(AudioRing::new());
    let audio = if mute {
        None
    } else {
        start_audio(queue.clone())
    };
    let audio_enabled = audio.is_some();
    if mute {
        eprintln!("Audio muted");
    } else if audio.is_none() {
        eprintln!("No audio output device; running silently");
    }
    let stem = std::path::Path::new(cue.as_str())
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let hw_render_enabled = emulator.hw_render();
    let _frame_timer = FrameTimerResolution::request();
    let event_loop = EventLoop::new().expect("create event loop");
    let proxy = event_loop.create_proxy();
    let (cmd_tx, cmd_rx) = channel();
    let shared = Arc::new(EmuShared {
        pad_bits: AtomicU32::new(0),
        turbo: AtomicBool::new(false),
        quit: AtomicBool::new(false),
        frames: AtomicU64::new(0),
        emu_ns: AtomicU64::new(0),
        refresh_bits: AtomicU64::new(60.0f64.to_bits()),
        mailbox: Mutex::new(None),
    });
    // Emulation + audio production live on their own thread so window
    // resizes, modal drags, and presentation can never starve them.
    let emu_thread = thread::spawn({
        let shared = shared.clone();
        let queue = queue.clone();
        let card_path = card_path.clone();
        move || emu_main(emulator, queue, audio_enabled, proxy, cmd_rx, shared, card_path)
    });
    let mut app = App {
        queue,
        _audio: audio,
        gilrs: Gilrs::new()
            .map_err(|e| eprintln!("gamepad init failed: {e}"))
            .ok(),
        pressed_pad_buttons: HashSet::new(),
        pad_start_code,
        pad_select_code,
        winmm_pads: {
            let ids = scan_winmm_pads();
            if !ids.is_empty() {
                eprintln!("winmm joystick fallback: {} pad(s) {ids:?}", ids.len());
            }
            ids
        },
        keys: HashSet::new(),
        window: None,
        presenter: None,
        hw_render_enabled,
        hw_scale: internal_scale as u32,
        smooth: true,
        texture_filter: texture_filter.unwrap_or(true),
        crt_filter: crt_filter.unwrap_or(true),
        shared,
        cmd_tx,
        emu_thread: Some(emu_thread),
        last_cpu_frame: None,
        last_hw_crop: None,
        redraw_pending: false,
        present_due: false,
        resize_deadline: None,
        last_resize_present: Instant::now(),
        frame_start: Instant::now(),
        last_emu_frames: 0,
        last_emu_ns: 0,
        present_count: 0,
        output_time: Duration::ZERO,
        output_size: window_size,
        title: format!("AES Oxide — {stem} — {internal_scale}x internal"),
    };
    event_loop.run_app(&mut app).expect("run window event loop");
}

#[cfg(test)]
mod tests {
    use super::{
        AUDIO_RING_CAP, AudioRing, Duration, Instant, Ordering, advance_frame_deadline,
        parse_crt_filter, parse_internal_scale, parse_texture_filter, parse_window_size,
    };

    #[test]
    fn parses_ascii_and_unicode_resolution_separators() {
        assert_eq!(parse_window_size("3840x2160").unwrap().width, 3840);
        assert_eq!(parse_window_size("3840×2160").unwrap().height, 2160);
    }

    #[test]
    fn rejects_invalid_or_oversized_resolution() {
        assert!(parse_window_size("3840").is_err());
        assert!(parse_window_size("0x2160").is_err());
        assert!(parse_window_size("16384x2160").is_err());
    }

    #[test]
    fn internal_resolution_selects_polygon_scale_not_window_size() {
        assert_eq!(parse_internal_scale("2x").unwrap(), 2);
        assert_eq!(parse_internal_scale("4x").unwrap(), 4);
        assert_eq!(parse_internal_scale("6x").unwrap(), 6);
        assert_eq!(parse_internal_scale("3840x2160").unwrap(), 8);
        assert_eq!(parse_internal_scale("3840×2160").unwrap(), 8);
    }

    #[test]
    fn texture_filter_modes_are_explicit() {
        assert!(!parse_texture_filter("nearest").unwrap());
        assert!(parse_texture_filter("bilinear").unwrap());
        assert!(parse_texture_filter("fsr").is_err());
        assert!(parse_crt_filter("on").unwrap());
        assert!(!parse_crt_filter("off").unwrap());
        assert!(parse_crt_filter("crt").is_err());
    }

    #[test]
    fn frame_grid_repays_late_frames_instead_of_slipping() {
        let period = Duration::from_micros(16_680);
        let slot = Instant::now();
        // A 135 ms boot burst ends 8+ frames behind the grid: the next
        // deadline must stay in the past so catch-up frames run now.
        let now = slot + Duration::from_millis(135);
        let next = advance_frame_deadline(slot, period, now);
        assert!(next <= now);
        // Successive catch-up frames keep stepping the same grid until
        // it realigns with the wall clock.
        let mut cursor = next;
        for _ in 0..9 {
            cursor = advance_frame_deadline(cursor, period, now);
        }
        assert!(cursor > now);
    }

    #[test]
    fn frame_grid_resyncs_after_a_long_stall() {
        let period = Duration::from_micros(16_680);
        let slot = Instant::now();
        // A multi-second stall (modal window drag) must not produce a
        // huge catch-up burst: resync to a bounded seeded debt.
        let now = slot + Duration::from_secs(5);
        let next = advance_frame_deadline(slot, period, now);
        assert!(next <= now);
        assert!(now - next <= Duration::from_millis(210));
    }

    #[test]
    fn audio_ring_preserves_order_and_drops_on_full_without_blocking() {
        let q = AudioRing::new();
        for n in 0..AUDIO_RING_CAP {
            let v = n as i16;
            q.push((v, v.wrapping_neg()));
        }
        for n in 0..(AUDIO_RING_CAP - 1) {
            let v = n as i16;
            assert_eq!(q.pop(), Some((v, v.wrapping_neg())));
        }
        assert_eq!(q.pop(), None);
        assert_eq!(q.overruns.load(Ordering::Relaxed), 1);
        assert_eq!(q.underruns.load(Ordering::Relaxed), 1);
    }
}
