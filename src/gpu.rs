//! GPU: GP0 (draw commands) / GP1 (control), GPUSTAT, GPUREAD.
//!
//! Foundation stage: VRAM is real (1 MiB), GP0 commands are decoded
//! enough for memory transfers (CPU↔VRAM, VRAM fill/copy) — the
//! rasterizer is stubbed but the frame path can already read the
//! display area, so BIOS/menu screenshots become possible as soon as
//! draw commands land.
//!
//! GPUSTAT bits (subset):
//!   0-3 texture page    5 dither   6 draw-to-display   8 mask
//!   10 textured-en?     11 irq     13 interlaced       14 odd line
//!   19-20 vres          21-22 hres 23 vmode            26 ready-cmd
//!   27 ready-dma-word   28 ready-dma-block  29-30 dma mode
//!   31 display-disable flip… (we keep a live vblank flag elsewhere)

const VRAM_W: usize = 1024;
const VRAM_H: usize = 512;

// ---- hardware-render op stream --------------------------------------
// When a host-GPU backend is attached the core keeps doing the full
// native-resolution pass (VRAM stays CPU-visible and exact) and records
// each drawing command as an op the backend replays into a scaled
// render target — the same split DuckStation/Beetle use: native domain
// for texture/CPU reads, scaled domain for rasterized output.

#[derive(Clone, Copy, Debug, Default)]
pub struct HwVert {
    /// native VRAM coords, draw offset already applied
    pub x: i16,
    pub y: i16,
    /// texture coords (native VRAM texel space)
    pub u: i16,
    pub v: i16,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub pad: u8,
}

pub mod hw_flags {
    pub const TEXTURED: u16 = 1 << 0;
    pub const GOURAUD: u16 = 1 << 1;
    pub const SEMITRANS: u16 = 1 << 2;
    pub const RAW: u16 = 1 << 3;
    pub const MASK_SET: u16 = 1 << 4;
    pub const MASK_CHECK: u16 = 1 << 5;
    pub const DITHER: u16 = 1 << 6;
    /// verts[0..2] are line endpoints; backend draws a scale-wide quad
    pub const LINE: u16 = 1 << 7;
    /// axis-aligned rect: verts[0].u/v hold raw u0/v0; backend
    /// generates corner UVs (incl. E1.12/13 flips stored in flags)
    pub const RECT: u16 = 1 << 8;
    pub const XFLIP: u16 = 1 << 9;
    pub const YFLIP: u16 = 1 << 10;
    /// pass selectors for two-draw semitransparency on textured prims
    pub const DRAW_OPAQUE_TEXELS: u16 = 1 << 11;
    pub const DRAW_SEMI_TEXELS: u16 = 1 << 12;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HwPrim {
    pub verts: [HwVert; 4],
    /// 3 = triangle, 4 = quad (triangulate 0,1,2 + 1,2,3)
    pub n: u8,
    pub pad: u8,
    pub tpage: u16,
    pub clut: u16,
    /// E2 texture-window register value
    pub texwin: u32,
    /// primitive colour (command colour / flat colour)
    pub col: u32,
    pub flags: u16,
    pub pad2: u16,
    /// draw-area clip, native coords, inclusive: x0,y0,x1,y1
    pub clip: [i16; 4],
    /// Primitive texture-coordinate bounds, inclusive in native texels.
    pub uv_bounds: [i16; 4],
}

#[derive(Clone, Copy, Debug)]
pub enum HwOp {
    /// CPU-side VRAM write (upload, blit, fill): upload the rect to the
    /// backend's native VRAM texture AND resolve it into the scaled
    /// surface nearest-neighbour — it displays as native-res content.
    Resolve {
        x0: u16,
        y0: u16,
        x1: u16,
        y1: u16,
    },
    /// Native texture data changed under rasterization: upload the rect
    /// to the native VRAM texture only (framebuffer-as-texture feedback)
    /// — must NOT resolve into the scaled surface.
    Upload {
        x0: u16,
        y0: u16,
        x1: u16,
        y1: u16,
    },
    Prim(HwPrim),
}

/// Raster vertex: position (post draw-offset), color, texcoord.
#[derive(Clone, Copy, Default)]
struct Vert {
    x: i32,
    y: i32,
    r: i32,
    g: i32,
    b: i32,
    u: i32,
    v: i32,
}

/// Per-primitive shading context.
#[derive(Clone, Copy)]
struct Shade {
    textured: bool,
    gouraud: bool,
    semitrans: bool,
    /// Raw texture: texels are not modulated by the vertex colour.
    raw: bool,
    tpage: u32,
    clut: u32,
}

#[derive(Clone, Copy)]
struct Interp {
    value: i32,
    rem: i64,
    delta: i32,
    delta_rem: i64,
}

/// 4x4 ordered-dither offsets (GP0 E1 bit9).
const DITHER: [[i32; 4]; 4] = [
    [-4, 0, -3, 1],
    [2, -2, 3, -1],
    [-3, 1, -4, 0],
    [3, -1, 2, -2],
];

/// 8-bit channel → 5-bit with optional dithering.
#[inline]
fn ch5(v: i32, dither: bool, x: i32, y: i32) -> u16 {
    let v = if dither {
        v + DITHER[(y & 3) as usize][(x & 3) as usize]
    } else {
        v
    };
    (v.clamp(0, 255) >> 3) as u16
}

fn shade15(r: i32, g: i32, b: i32, dither: bool, x: i32, y: i32) -> u16 {
    ch5(r, dither, x, y) | ch5(g, dither, x, y) << 5 | ch5(b, dither, x, y) << 10
}

/// Texel x vertex colour (0x80 = 1.0) at 8-bit precision, dithered.
fn modulate_d(tex: u16, col: u32, dither: bool, x: i32, y: i32) -> u16 {
    let f = |t: u16, c: u32| ((t as u32 * 8) * c / 0x80) as i32;
    shade15(
        f(tex & 0x1F, col & 0xFF),
        f((tex >> 5) & 0x1F, (col >> 8) & 0xFF),
        f((tex >> 10) & 0x1F, (col >> 16) & 0xFF),
        dither,
        x,
        y,
    )
}

/// 24-bit BGR command color → 15-bit framebuffer pixel.
fn rgb15(c: u32) -> u16 {
    (((c >> 16) & 0xFF) as u16 >> 3) << 10
        | (((c >> 8) & 0xFF) as u16 >> 3) << 5
        | ((c & 0xFF) as u16 >> 3)
}

#[inline]
fn split_interpolation(numerator: i64, area: i64) -> (i32, i64) {
    (
        numerator.div_euclid(area) as i32,
        numerator.rem_euclid(area),
    )
}

#[inline]
fn interpolation(numerator: i64, delta: (i32, i64), area: i64) -> Interp {
    let (value, rem) = split_interpolation(numerator, area);
    Interp {
        value,
        rem,
        delta: delta.0,
        delta_rem: delta.1,
    }
}

#[inline]
fn edge_value(a: &Vert, b: &Vert, x: i32, y: i32) -> i64 {
    (b.x - a.x) as i64 * (y - a.y) as i64 - (b.y - a.y) as i64 * (x - a.x) as i64
}

fn edge_span(a: &Vert, b: &Vert, x0: i32, x1: i32, y: i32, bias: i64) -> Option<(i32, i32)> {
    let base = edge_value(a, b, x0, y) + bias;
    let dx = -(b.y - a.y) as i64;
    if dx > 0 {
        let offset = if base >= 0 { 0 } else { (-base + dx - 1) / dx };
        let left = x0 as i64 + offset;
        (left <= x1 as i64).then_some((left as i32, x1))
    } else if dx < 0 {
        if base < 0 {
            return None;
        }
        let right = x0 as i64 + base / -dx;
        (right >= x0 as i64).then_some((x0, right.min(x1 as i64) as i32))
    } else {
        (base >= 0).then_some((x0, x1))
    }
}

fn triangle_span(v: &[Vert; 3], x0: i32, x1: i32, y: i32, biases: [i64; 3]) -> Option<(i32, i32)> {
    let edges = [
        (&v[1], &v[2], biases[0]),
        (&v[2], &v[0], biases[1]),
        (&v[0], &v[1], biases[2]),
    ];
    let mut left = x0;
    let mut right = x1;
    for (a, b, bias) in edges {
        let (edge_left, edge_right) = edge_span(a, b, x0, x1, y, bias)?;
        left = left.max(edge_left);
        right = right.min(edge_right);
        if left > right {
            return None;
        }
    }
    Some((left, right))
}

/// Texture fetch from native VRAM. `texwin` is the E2 texture-window
/// register. Free function so enhanced-resolution scanline workers can
/// share read-only access without borrowing the whole GPU.
fn texel_at(
    vram: &[u16],
    texwin: u32,
    u: i32,
    v: i32,
    tpage: u32,
    clut: u32,
) -> Option<(u16, bool)> {
    // Texture window: (coord & !(mask*8)) | ((offset & mask)*8)
    let wx = (texwin & 0x1F) as i32;
    let wy = ((texwin >> 5) & 0x1F) as i32;
    let ox = ((texwin >> 10) & 0x1F) as i32;
    let oy = ((texwin >> 15) & 0x1F) as i32;
    let u = (u & !(wx * 8)) | ((ox & wx) * 8);
    let v = (v & !(wy * 8)) | ((oy & wy) * 8);
    let tx = ((tpage & 0xF) * 64) as usize;
    let ty = if tpage & 0x10 != 0 { 256usize } else { 0 };
    let (u, v) = ((u & 0xFF) as usize, (v & 0xFF) as usize);
    let depth = (tpage >> 7) & 3;
    let c = match depth {
        0 | 1 => {
            let (shift, cx, cy) = if depth == 0 {
                (
                    (u & 3) * 4,
                    ((clut & 0x3F) * 16) as usize,
                    ((clut >> 6) & 0x1FF) as usize,
                )
            } else {
                (
                    (u & 1) * 8,
                    ((clut & 0x3F) * 16) as usize,
                    ((clut >> 6) & 0x1FF) as usize,
                )
            };
            let w = vram[((ty + v) & (VRAM_H - 1)) * VRAM_W
                + ((tx + u / if depth == 0 { 4 } else { 2 }) & (VRAM_W - 1))];
            let idx = ((w >> shift) & if depth == 0 { 0xF } else { 0xFF }) as usize;
            vram[(cy * VRAM_W + cx + idx) & (VRAM_W * VRAM_H - 1)]
        }
        _ => vram[((ty + v) & (VRAM_H - 1)) * VRAM_W + ((tx + u) & (VRAM_W - 1))],
    };
    if c == 0 {
        None
    } else {
        Some((c & 0x7FFF, c & 0x8000 != 0))
    }
}

/// Semi-transparency blend of `src` over `dst`.
/// Modes (tpage bits5-6): 0=B/2+F/2 1=B+F 2=B-F 3=B+F/4.
fn blend_px(dst: u16, src: u16, tpage: u32) -> u16 {
    let mode = (tpage >> 5) & 3;
    let (sr, sg, sb) = (
        (src & 0x1F) as i32,
        ((src >> 5) & 0x1F) as i32,
        ((src >> 10) & 0x1F) as i32,
    );
    let (dr, dg, db) = (
        (dst & 0x1F) as i32,
        ((dst >> 5) & 0x1F) as i32,
        ((dst >> 10) & 0x1F) as i32,
    );
    let f = |d: i32, s: i32| -> u16 {
        let v = match mode {
            0 => d / 2 + s / 2,
            1 => d + s,
            2 => d - s,
            _ => d + s / 4,
        };
        v.clamp(0, 31) as u16
    };
    (f(dr, sr) | (f(dg, sg) << 5) | (f(db, sb) << 10)) | (src & 0x8000)
}

/// Read-only inputs shared by enhanced-resolution scanline workers.
struct HiCtx<'a> {
    vram: &'a [u16],
    texwin: u32,
    e1: u32,
    mask_set: bool,
    mask_check: bool,
    v: [Vert; 3],
    sh: Shade,
    dither: bool,
    color_delta: Option<[(i32, i64); 3]>,
    uv_delta: Option<[(i32, i64); 2]>,
    umin: i32,
    umax: i32,
    vmin: i32,
    vmax: i32,
    area: i64,
}

/// Rasterize one scanline span of a triangle into an enhanced-target
/// row slice. Pixel-identical to the sequential path: interpolators
/// start at the span edge and advance exactly as in tri().
#[allow(clippy::too_many_arguments)]
fn tri_row_hires(ctx: &HiCtx, row: &mut [u16], y: i32, span_x0: i32, span_x1: i32) {
    let v = &ctx.v;
    let sh = ctx.sh;
    let area = ctx.area;
    let weights = [
        edge_value(&v[1], &v[2], span_x0, y),
        edge_value(&v[2], &v[0], span_x0, y),
        edge_value(&v[0], &v[1], span_x0, y),
    ];
    let weighted = |a: i32, b: i32, c: i32| -> i64 {
        a as i64 * weights[0] + b as i64 * weights[1] + c as i64 * weights[2]
    };
    let mut colors = ctx.color_delta.map(|delta| {
        [
            interpolation(weighted(v[0].r, v[1].r, v[2].r), delta[0], area),
            interpolation(weighted(v[0].g, v[1].g, v[2].g), delta[1], area),
            interpolation(weighted(v[0].b, v[1].b, v[2].b), delta[2], area),
        ]
    });
    let mut texcoords = ctx.uv_delta.map(|delta| {
        [
            interpolation(weighted(v[0].u, v[1].u, v[2].u), delta[0], area),
            interpolation(weighted(v[0].v, v[1].v, v[2].v), delta[1], area),
        ]
    });
    for x in span_x0..=span_x1 {
        let i = x as usize;
        let (r, g, b) = if sh.gouraud {
            let c = colors.as_ref().unwrap();
            (c[0].value, c[1].value, c[2].value)
        } else {
            (v[0].r, v[0].g, v[0].b)
        };
        let out = if sh.textured {
            let uv = texcoords.as_ref().unwrap();
            let u = uv[0].value.clamp(ctx.umin, ctx.umax);
            let tv = uv[1].value.clamp(ctx.vmin, ctx.vmax);
            texel_at(ctx.vram, ctx.texwin, u, tv, sh.tpage, sh.clut).map(|(t, semi)| {
                let m = if sh.raw {
                    t
                } else {
                    let col = (r.clamp(0, 255) as u32)
                        | ((g.clamp(0, 255) as u32) << 8)
                        | ((b.clamp(0, 255) as u32) << 16);
                    modulate_d(t, col, ctx.dither, x, y)
                };
                if sh.semitrans && semi {
                    blend_px(row[i], m, sh.tpage)
                } else {
                    m | if semi { 0x8000 } else { 0 }
                }
            })
        } else {
            let c15 = shade15(r, g, b, ctx.dither, x, y);
            Some(if sh.semitrans {
                blend_px(row[i], c15, ctx.e1)
            } else {
                c15
            })
        };
        if let Some(mut c) = out {
            if ctx.mask_check && row[i] & 0x8000 != 0 {
            } else {
                if ctx.mask_set {
                    c |= 0x8000;
                }
                row[i] = c;
            }
        }
        if let Some(c) = colors.as_mut() {
            for interp in c {
                interp.advance(area);
            }
        }
        if let Some(uv) = texcoords.as_mut() {
            for interp in uv {
                interp.advance(area);
            }
        }
    }
}

impl Interp {
    #[inline]
    fn advance(&mut self, area: i64) {
        self.value += self.delta;
        self.rem += self.delta_rem;
        if self.rem >= area {
            self.value += 1;
            self.rem -= area;
        } else if self.rem < 0 {
            self.value -= 1;
            self.rem += area;
        }
    }
}

/// Multiply a texel by the primitive's modulation color (0x80 = 1.0).
fn modulate(tex: u16, col: u32) -> u16 {
    let f = |t: u16, c: u32| (((t as u32) * c) / 0x80).min(0x1F) as u16;
    f(tex & 0x1F, col & 0xFF)
        | (f((tex >> 5) & 0x1F, (col >> 8) & 0xFF) << 5)
        | (f((tex >> 10) & 0x1F, (col >> 16) & 0xFF) << 10)
}

pub struct Gpu {
    pub vram: Vec<u16>,
    hires_vram: Vec<u16>,
    render_scale: usize,
    render_hires: bool,
    suppress_hires_mirror: bool,
    /// Host-GPU backend attached: drawing commands are recorded as
    /// ops instead of being rasterized on the CPU.
    hw_render: bool,
    /// Recorded op stream, drained by the backend each frame.
    hw_ops: Vec<HwOp>,
    /// VRAM rects dirtied by rasterization since the last textured
    /// draw — resolved lazily when a primitive samples them.
    raster_dirty: Vec<(u16, u16, u16, u16)>,
    /// GP0 command fifo state
    gp0_cmd: u32,
    gp0_len: usize,
    gp0_pos: usize,
    gp0_params: [u32; 16],
    /// CPU→VRAM transfer: (x, y, w, h, words_left, words_total)
    vram_wr: Option<(u16, u16, u16, u16, u32, u32)>,
    /// VRAM→CPU transfer: (x, y, w, h, words_left, words_total)
    vram_rd: Option<(u16, u16, u16, u16, u32, u32)>,
    /// Polyline in progress: raw words seen so far (verts + colors)
    poly_words: Vec<u32>,
    /// Draw offset (GP0 E5): signed, added to every vertex.
    off_x: i32,
    off_y: i32,
    /// Display area (from GP1 0x05)
    pub disp_x: u16,
    pub disp_y: u16,
    /// Horizontal/vertical display ranges (GP1 0x06/0x07)
    pub h_range: (u16, u16),
    pub v_range: (u16, u16),
    /// Draw area top-left (GP0 E1/ E3-E4 offsets)
    pub draw_x: u16,
    pub draw_y: u16,
    /// GP0 E1/E2/E3/E4 attribute words (draw mode, tex window,
    /// clip corners).
    regs: [u32; 8],
    /// E2 texture window (mask/offset applied to texcoords).
    texwin: u32,
    /// E6.0 — set mask bit15 on drawn pixels; E6.1 — don't draw
    /// where vram bit15 already set.
    mask_set: bool,
    mask_check: bool,
    /// GP0(1Fh) latched GPU IRQ (GPUSTAT.24; also I_STAT bit1).
    pub irq1: bool,
    /// GP1(03h): display disabled.
    disp_disabled: bool,
    /// GP1(04h) dma direction 0-3.
    dma_dir: u8,
    /// GP1(08h) display mode word.
    disp_mode: u32,
    /// Interlace field parity (toggles per frame).
    pub field_odd: bool,
    /// GP1(10h) info request index.
    info_sel: u8,
    /// Inside vertical blanking (driven by the video timing model).
    pub vblank: bool,
    /// GPUSTAT bit31: odd scanline/field being output (timing model).
    pub odd_line: bool,
    /// Apply the console's 4x4 ordered dither (GP0 E1.9). Off gives a
    /// cleaner image; on matches hardware output.
    pub dithering: bool,
    /// Union of every display crop seen so far (native VRAM coords).
    /// Enhanced polygon passes only need to cover regions that can ever
    /// be shown; games double-buffer, so this must accumulate rather
    /// than track only the currently displayed buffer.
    seen_display: (u32, u32, u32, u32),
}

impl Gpu {
    pub fn new() -> Self {
        Self {
            vram: vec![0; VRAM_W * VRAM_H],
            hires_vram: Vec::new(),
            render_scale: 1,
            render_hires: false,
            suppress_hires_mirror: false,
            hw_render: false,
            hw_ops: Vec::new(),
            raster_dirty: Vec::new(),
            gp0_cmd: 0,
            gp0_len: 0,
            gp0_pos: 0,
            gp0_params: [0; 16],
            vram_wr: None,
            vram_rd: None,
            poly_words: Vec::new(),
            off_x: 0,
            off_y: 0,
            disp_x: 0,
            disp_y: 0,
            h_range: (0x200, 0xC00),
            v_range: (0x10, 0x100),
            draw_x: 0,
            draw_y: 0,
            regs: [0; 8],
            texwin: 0,
            mask_set: false,
            mask_check: false,
            irq1: false,
            disp_disabled: true,
            dma_dir: 0,
            disp_mode: 0,
            field_odd: false,
            info_sel: 0,
            vblank: false,
            odd_line: false,
            dithering: false,
            seen_display: (0, 0, 0, 0),
        }
    }

    /// Native-coordinate crop the display reads right now, clamped to
    /// VRAM. Matches display_rgb()'s source region.
    fn display_crop(&self) -> (u32, u32, u32, u32) {
        let (w, h480) = self.display_size();
        let lines = (self.v_range.1 as i32 - self.v_range.0 as i32).clamp(16, 288) as u32;
        let h = if h480 == 480 { lines * 2 } else { lines };
        let x0 = self.disp_x as u32;
        let y0 = self.disp_y as u32;
        (
            x0,
            y0,
            (x0 + w).min(VRAM_W as u32),
            (y0 + h).min(VRAM_H as u32),
        )
    }

    /// Fold the current display crop into the seen-display union so the
    /// enhanced pass covers every buffer that may be shown. Newly added
    /// area is backfilled from native VRAM so a just-revealed buffer
    /// shows correct content instead of stale pixels.
    fn note_display(&mut self) {
        let (x0, y0, x1, y1) = self.display_crop();
        if self.seen_display.2 <= self.seen_display.0 || self.seen_display.3 <= self.seen_display.1
        {
            self.seen_display = (x0, y0, x1, y1);
            self.mirror_hires_rect(x0, y0, x1, y1);
            return;
        }
        // Rectangles of the crop not already covered by the union:
        // left/right strips full height, then top/bottom between them.
        let (ox0, oy0, ox1, oy1) = self.seen_display;
        if x0 < ox0 {
            self.mirror_hires_rect(x0, y0, ox0.min(x1), y1);
        }
        if x1 > ox1 {
            self.mirror_hires_rect(ox1.max(x0), y0, x1, y1);
        }
        let ix0 = x0.max(ox0);
        let ix1 = x1.min(ox1);
        if y0 < oy0 {
            self.mirror_hires_rect(ix0, y0, ix1, oy0.min(y1));
        }
        if y1 > oy1 {
            self.mirror_hires_rect(ix0, oy1.max(y0), ix1, y1);
        }
        let s = &mut self.seen_display;
        s.0 = s.0.min(x0);
        s.1 = s.1.min(y0);
        s.2 = s.2.max(x1);
        s.3 = s.3.max(y1);
    }

    /// Block-mirror a native VRAM rectangle into the enhanced surface.
    fn mirror_hires_rect(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) {
        if self.render_scale <= 1 {
            return;
        }
        for y in y0..y1.min(VRAM_H as u32) {
            for x in x0..x1.min(VRAM_W as u32) {
                let (x, y) = (x as usize, y as usize);
                self.mirror_hires_pixel(x, y, self.vram[y * VRAM_W + x]);
            }
        }
    }

    /// Enable 2x/4x/8x polygon rasterization while retaining native
    /// VRAM for texture reads and CPU-visible transfers.
    pub fn set_internal_scale(&mut self, scale: u8) -> bool {
        let scale = match scale {
            1 | 2 | 4 | 8 => scale as usize,
            _ => return false,
        };
        if self.render_scale == scale {
            return true;
        }
        self.render_scale = scale;
        self.render_hires = false;
        self.suppress_hires_mirror = false;
        self.hires_vram.clear();
        if scale > 1 {
            let width = VRAM_W * scale;
            self.hires_vram.resize(VRAM_W * VRAM_H * scale * scale, 0);
            for y in 0..VRAM_H {
                for x in 0..VRAM_W {
                    self.mirror_hires_pixel(x, y, self.vram[y * VRAM_W + x]);
                }
            }
            debug_assert_eq!(self.hires_vram.len(), width * VRAM_H * scale);
        }
        true
    }

    pub fn internal_scale(&self) -> u8 {
        self.render_scale as u8
    }

    /// Attach a host-GPU backend. When enabled (with scale > 1), draw
    /// commands are recorded into an op stream and the CPU hires pass
    /// is skipped; native VRAM stays authoritative for CPU reads.
    pub fn set_hw_render(&mut self, on: bool) {
        self.hw_render = on;
        self.hw_ops.clear();
        self.raster_dirty.clear();
    }

    pub fn hw_render(&self) -> bool {
        self.hw_render
    }

    /// Drain recorded ops for the backend to replay this frame.
    pub fn take_hw_ops(&mut self) -> Vec<HwOp> {
        std::mem::take(&mut self.hw_ops)
    }

    /// Display crop in native VRAM coords for the backend's scanout.
    pub fn hw_display_crop(&self) -> (u32, u32, u32, u32) {
        self.display_crop()
    }

    fn mirror_hires_pixel(&mut self, x: usize, y: usize, color: u16) {
        let scale = self.render_scale;
        if scale <= 1 || self.suppress_hires_mirror || self.hw_render {
            return;
        }
        let width = VRAM_W * scale;
        for dy in 0..scale {
            let row = (y * scale + dy) * width + x * scale;
            self.hires_vram[row..row + scale].fill(color);
        }
    }

    fn write_native_pixel(&mut self, x: usize, y: usize, color: u16) {
        let index = (y * VRAM_W + x) & (VRAM_W * VRAM_H - 1);
        self.vram[index] = color;
        self.mirror_hires_pixel(index % VRAM_W, index / VRAM_W, color);
    }

    // ---- hardware-render op recording --------------------------------

    /// Mark a native-res rect as dirtied by rasterization; resolved to
    /// the backend's native texture lazily when a textured draw might
    /// sample it (framebuffer-as-texture feedback).
    fn dirty_raster_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        let clamp = |v: i32, hi: i32| v.clamp(0, hi - 1) as u16;
        let r = (
            clamp(x0, VRAM_W as i32),
            clamp(y0, VRAM_H as i32),
            clamp(x1, VRAM_W as i32),
            clamp(y1, VRAM_H as i32),
        );
        if r.2 < r.0 || r.3 < r.1 {
            return;
        }
        // merge when the list grows: keep one bounding box
        if self.raster_dirty.len() >= 8 {
            let mut u = self.raster_dirty[0];
            for &r in &self.raster_dirty[1..] {
                u = (u.0.min(r.0), u.1.min(r.1), u.2.max(r.2), u.3.max(r.3));
            }
            self.raster_dirty.clear();
            self.raster_dirty.push(u);
        }
        self.raster_dirty.push(r);
    }

    /// Emit Sync ops for rasterized regions a textured prim may sample.
    /// tpage/clut give the texture window the draw reads from.
    fn flush_dirty_for_tex(&mut self, tpage: u16, clut: u16) {
        if self.raster_dirty.is_empty() {
            return;
        }
        let tx = ((tpage & 0xF) * 64) as u32;
        let ty = if tpage & 0x10 != 0 { 256u32 } else { 0 };
        let page = (tx, ty, tx + 256, ty + 256);
        let cx = ((clut & 0x3F) * 16) as u32;
        let cy = ((clut >> 6) & 0x1FF) as u32;
        let clut_r = (cx, cy, cx + 256, cy + 1);
        let dirty = std::mem::take(&mut self.raster_dirty);
        for r in dirty {
            let hit = |b: (u32, u32, u32, u32)| {
                r.0 as u32 <= b.2 && r.2 as u32 >= b.0 && r.1 as u32 <= b.3 && r.3 as u32 >= b.1
            };
            if hit(page) || hit(clut_r) {
                self.hw_ops.push(HwOp::Upload {
                    x0: r.0,
                    y0: r.1,
                    x1: r.2 + 1,
                    y1: r.3 + 1,
                });
            } else {
                self.raster_dirty.push(r);
            }
        }
    }

    /// Record one primitive for the hardware backend and mark its
    /// native footprint dirty for later texture feedback.
    fn record_prim(&mut self, v: &[Vert], n: usize, sh: &Shade, extra: u16) {
        if sh.textured {
            self.flush_dirty_for_tex(sh.tpage as u16, sh.clut as u16);
        }
        let (cx0, cy0, cx1, cy1) = self.clip();
        let mut hv = [HwVert::default(); 4];
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        let (mut u0, mut v0, mut u1, mut v1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for (i, vert) in v.iter().enumerate().take(4) {
            hv[i] = HwVert {
                x: vert.x as i16,
                y: vert.y as i16,
                u: vert.u as i16,
                v: vert.v as i16,
                r: vert.r as u8,
                g: vert.g as u8,
                b: vert.b as u8,
                pad: 0,
            };
            x0 = x0.min(vert.x);
            x1 = x1.max(vert.x);
            y0 = y0.min(vert.y);
            y1 = y1.max(vert.y);
            u0 = u0.min(vert.u);
            v0 = v0.min(vert.v);
            u1 = u1.max(vert.u);
            v1 = v1.max(vert.v);
        }
        // Rect vertex UVs describe texel-boundary corners; the covered
        // texel centers stop one before the high boundary.
        if extra & hw_flags::RECT != 0 {
            u1 = u1.saturating_sub(1);
            v1 = v1.saturating_sub(1);
        }
        let mut flags = extra;
        flags |= hw_flags::TEXTURED * sh.textured as u16
            | hw_flags::GOURAUD * sh.gouraud as u16
            | hw_flags::SEMITRANS * sh.semitrans as u16
            | hw_flags::RAW * sh.raw as u16
            | hw_flags::MASK_SET * self.mask_set as u16
            | hw_flags::MASK_CHECK * self.mask_check as u16;
        if self.dithering
            && self.regs[1] & (1 << 9) != 0
            && (sh.gouraud || (sh.textured && !sh.raw))
        {
            flags |= hw_flags::DITHER;
        }
        self.hw_ops.push(HwOp::Prim(HwPrim {
            verts: hv,
            n: n as u8,
            pad: 0,
            tpage: sh.tpage as u16,
            clut: sh.clut as u16,
            texwin: self.texwin,
            col: self.gp0_cmd & 0xFF_FFFF,
            flags,
            pad2: 0,
            clip: [cx0 as i16, cy0 as i16, (cx1 + 1) as i16, (cy1 + 1) as i16],
            uv_bounds: [u0 as i16, v0 as i16, u1 as i16, v1 as i16],
        }));
        self.dirty_raster_rect(x0, y0, x1, y1);
    }

    /// GPUSTAT — composed live from the E1/E6/GP1 registers.
    ///   0-10 = E1 draw-mode bits   11 = E6.0 set-mask
    ///   12 = E6.1 check-mask       13 = interlace field
    ///   14 = reverseflag           15 = tpage Y bit9
    ///   16-22 display-mode fields  23 = display disabled
    ///   24 = IRQ1                  25 = DRQ
    ///   26 ready-cmd  27 read-data 28 fifo-empty  29-30 dma-dir
    ///   31 odd/even field (0 in vblank)
    pub fn read_stat(&mut self) -> u32 {
        let e1 = self.regs[1];
        let dm = self.disp_mode;
        let interlaced = dm & (1 << 5) != 0;
        let mut s = e1 & 0x7FF;
        s |= (self.mask_set as u32) << 11;
        s |= (self.mask_check as u32) << 12;
        s |= ((!interlaced || self.field_odd) as u32) << 13;
        s |= ((dm >> 7) & 1) << 14;
        s |= (e1 & (1 << 11)) << 4; // tpage Y bit9
        s |= ((dm >> 6) & 1) << 16; // hres2
        s |= (dm & 3) << 17; // hres1
        s |= ((dm >> 2) & 1) << 19; // vres
        s |= ((dm >> 3) & 1) << 20; // vmode NTSC/PAL
        s |= ((dm >> 4) & 1) << 21; // 24bpp display
        s |= ((dm >> 5) & 1) << 22; // interlace on
        s |= (self.disp_disabled as u32) << 23;
        s |= (self.irq1 as u32) << 24;
        // DRQ: meaning depends on dma direction
        let drq = match self.dma_dir {
            1 | 2 => (self.vram_rd.is_none()) as u32,
            3 => self.vram_rd.is_some() as u32,
            _ => 0,
        };
        s |= drq << 25;
        s |= 1 << 26; // ready for command word
        s |= (self.vram_rd.is_some() as u32) << 27;
        s |= 1 << 28; // write fifo empty
        s |= (self.dma_dir as u32) << 29;
        // odd/even line (or field when interlaced); 0 during vblank
        s |= ((!self.vblank && self.odd_line) as u32) << 31;
        s
    }

    /// GPUREAD — pulls the next word of an in-flight VRAM→CPU read;
    /// otherwise serves GP1(10h) info requests.
    pub fn read_data(&mut self) -> u32 {
        if let Some((x, y, w, _h, left, total)) = self.vram_rd {
            if left == 0 {
                self.vram_rd = None;
                return 0;
            }
            let done = total - left;
            let px = x as usize + (done % w as u32) as usize;
            let py = y as usize + (done / w as u32) as usize;
            let lo = self.vram[(py * VRAM_W + px) & (VRAM_W * VRAM_H - 1)] as u32;
            let px2 = x as usize + ((done + 1) % w as u32) as usize;
            let py2 = y as usize + ((done + 1) / w as u32) as usize;
            let hi = self.vram[(py2 * VRAM_W + px2) & (VRAM_W * VRAM_H - 1)] as u32;
            self.vram_rd = Some((x, y, w, _h, left.saturating_sub(2), total));
            return lo | (hi << 16);
        }
        // GP1(10h) info readback
        match self.info_sel {
            2 => self.texwin,
            3 => self.regs[3] & 0x7_FFFF,  // draw area top-left
            4 => self.regs[4] & 0x7_FFFF,  // draw area bottom-right
            5 => self.regs[5] & 0x3F_FFFF, // draw offset
            7 => 2,                        // GPU type (old, 160pin)
            _ => 0,
        }
    }

    /// GP0 — one 32-bit command word (or parameter / VRAM data).
    pub fn gp0(&mut self, v: u32) {
        // mid-transfer? feed the CPU→VRAM stream
        if let Some((x, y, w, _h, left, total)) = self.vram_wr {
            let done = total - left;
            for half in 0..2u32 {
                let idx = done + half;
                let px = x as usize + (idx % w as u32) as usize;
                let py = y as usize + (idx / w as u32) as usize;
                self.write_native_pixel(px, py, (v >> (half * 16)) as u16);
            }
            self.vram_wr = if left > 2 {
                Some((x, y, w, _h, left - 2, total))
            } else {
                if self.hw_render {
                    self.hw_ops.push(HwOp::Resolve {
                        x0: x,
                        y0: y,
                        x1: (x + w).min(VRAM_W as u16),
                        y1: (y + _h).min(VRAM_H as u16),
                    });
                }
                None
            };
            return;
        }

        // polyline: words are vertices (+ colors for gouraud) until
        // the 0x55555555 terminator arrives.
        if !self.poly_words.is_empty() {
            self.poly_words.push(v);
            if v & 0xF000_F000 == 0x5000_5000 {
                let words = std::mem::take(&mut self.poly_words);
                self.finish_polyline(&words);
            }
            return;
        }

        // parameter word for a pending multiword command
        if self.gp0_len > 0 {
            if self.gp0_pos < 15 {
                self.gp0_params[self.gp0_pos] = v;
                self.gp0_pos += 1;
            }
            self.gp0_len -= 1;
            if self.gp0_len == 0 {
                self.gp0_finish();
            }
            return;
        }

        // command word
        let cmd = v >> 24;
        self.gp0_cmd = v;
        self.gp0_pos = 0;
        // parameter-word count per command (command word excluded).
        // Textured/shaded primitives interleave color/vertex/uv words.
        let len: usize = match cmd {
            0x00 | 0x01 => 0, // NOP / clear cache
            0x1F => {
                // interrupt request → GPUSTAT.24 + I_STAT bit1
                self.irq1 = true;
                0
            }
            0x02 => 2,         // fill rect
            0x20..=0x23 => 3,  // mono tri
            0x24..=0x27 => 6,  // textured tri
            0x28..=0x2B => 4,  // mono quad
            0x2C..=0x2F => 8,  // textured quad
            0x30..=0x33 => 5,  // gouraud tri
            0x34..=0x37 => 8,  // gouraud tex tri
            0x38..=0x3B => 7,  // gouraud quad
            0x3C..=0x3F => 11, // gouraud tex quad
            0x40..=0x43 => 2,  // mono line
            0x48..=0x4F | 0x58..=0x5F => {
                // polyline: variable, terminated by 0x55555555
                self.poly_words.push(v);
                return;
            }
            0x50..=0x53 => 3, // gouraud line
            0x60..=0x63 => 2, // mono rect (var)
            0x64..=0x67 => 3, // tex sprite (var)
            0x68..=0x6B => 1, // mono 1x1
            0x6C..=0x6F => 2, // tex 1x1
            0x70..=0x73 => 1, // mono 8x8
            0x74..=0x77 => 2, // tex 8x8
            0x78..=0x7B => 1, // mono 16x16
            0x7C..=0x7F => 2, // tex 16x16
            0x80..=0x9F => 3, // vram→vram copy
            0xA0..=0xBF => 2, // cpu→vram rect
            0xC0..=0xDF => 2, // vram→cpu rect
            0xE1..=0xE6 => {
                self.regs[(cmd - 0xE0) as usize] = v;
                match cmd {
                    0xE2 => self.texwin = v,
                    0xE5 => {
                        self.off_x = (((v & 0x7FF) << 21) as i32) >> 21;
                        self.off_y = ((((v >> 11) & 0x7FF) << 21) as i32) >> 21;
                    }
                    0xE6 => {
                        self.mask_set = v & 1 != 0;
                        self.mask_check = v & 2 != 0;
                    }
                    _ => {}
                }
                0
            }
            _ => 0,
        };
        if len > 0 {
            self.gp0_len = len;
        }
    }

    /// All params for `gp0_cmd` arrived — perform it.
    fn gp0_finish(&mut self) {
        let cmd = self.gp0_cmd >> 24;
        match cmd {
            0x20..=0x3F => self.finish_poly(cmd),
            0x40..=0x5F => self.finish_line(cmd),
            0x60..=0x7F => self.finish_rect(cmd),
            0x02 => {
                // fill rect: p0=color, p1=yx, p2=wh
                let yx = self.gp0_params[0];
                let wh = self.gp0_params[1];
                let c = {
                    let bgr = self.gp0_cmd & 0xFF_FFFF;
                    let r = (bgr & 0xFF) as u16;
                    let g = ((bgr >> 8) & 0xFF) as u16;
                    let b = ((bgr >> 16) & 0xFF) as u16;
                    ((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3)
                };
                let x = (yx & 0x3F0) as u16;
                let y = ((yx >> 16) & 0x1FF) as u16;
                let w = (((wh & 0x3FF) + 0xF) & !0xF).max(0x10) as u16;
                let h = (wh >> 16 & 0x1FF).max(1) as u16;
                for yy in y..y + h {
                    for xx in x..x + w {
                        self.write_native_pixel(xx as usize, yy as usize, c);
                    }
                }
                if self.hw_render {
                    self.hw_ops.push(HwOp::Resolve {
                        x0: x,
                        y0: y,
                        x1: x + w,
                        y1: y + h,
                    });
                }
            }
            0xA0 | 0xC0 => {
                // rect: p0=yx, p1=wh → begin data phase
                let yx = self.gp0_params[0];
                let wh = self.gp0_params[1];
                let w = (wh & 0x3FF).max(1) as u16;
                let h = ((wh >> 16) & 0x1FF).max(1) as u16;
                let x = (yx & 0x3FF) as u16;
                let y = ((yx >> 16) & 0x1FF) as u16;
                let words = (w as u32 * h as u32 + 1) & !1;
                if self.gp0_cmd >> 24 == 0xA0 {
                    self.vram_wr = Some((x, y, w, h, words, words));
                } else {
                    self.vram_rd = Some((x, y, w, h, words, words));
                }
            }
            0x80 => {
                // vram copy: p0=src yx, p1=dst yx, p2=wh
                let s = self.gp0_params[0];
                let d = self.gp0_params[1];
                let wh = self.gp0_params[2];
                let sx = (s & 0x3FF) as usize;
                let sy = ((s >> 16) & 0x1FF) as usize;
                let dx = (d & 0x3FF) as usize;
                let dy = ((d >> 16) & 0x1FF) as usize;
                let w = (wh & 0x3FF).max(1) as usize;
                let h = ((wh >> 16) & 0x1FF).max(1) as usize;
                // copy via a bounce (regions may overlap)
                let mut tmp = Vec::with_capacity(w * h);
                for yy in 0..h {
                    for xx in 0..w {
                        tmp.push(self.vram[((sy + yy) * VRAM_W + sx + xx) & (VRAM_W * VRAM_H - 1)]);
                    }
                }
                for yy in 0..h {
                    for xx in 0..w {
                        self.write_native_pixel(dx + xx, dy + yy, tmp[yy * w + xx]);
                    }
                }
                if self.hw_render {
                    self.hw_ops.push(HwOp::Resolve {
                        x0: dx as u16,
                        y0: dy as u16,
                        x1: (dx + w).min(VRAM_W) as u16,
                        y1: (dy + h).min(VRAM_H) as u16,
                    });
                }
            }
            _ => {}
        }
    }

    // ---- rasterizer --------------------------------------------------

    /// Vertex after draw-offset, plus shading + texcoord.
    fn vtx(&self, w: u32, c: u32) -> Vert {
        let scale = if self.render_hires {
            self.render_scale as i32
        } else {
            1
        };
        Vert {
            x: ((((w << 21) as i32) >> 21) + self.off_x) * scale,
            y: ((((w >> 16) << 21) as i32 >> 21) + self.off_y) * scale,
            r: (c & 0xFF) as i32,
            g: ((c >> 8) & 0xFF) as i32,
            b: ((c >> 16) & 0xFF) as i32,
            u: 0,
            v: 0,
        }
    }

    /// Polygons 0x20-0x3F: tri/quad × flat/gouraud × untex/textured.
    fn finish_poly(&mut self, cmd: u32) {
        if self.hw_render || self.render_scale == 1 {
            // native pass only; hw mode additionally records the prim
            self.finish_poly_pass(cmd);
            return;
        }
        self.suppress_hires_mirror = true;
        self.render_hires = false;
        self.finish_poly_pass(cmd);
        self.suppress_hires_mirror = false;
        self.render_hires = true;
        self.finish_poly_pass(cmd);
        self.render_hires = false;
    }

    fn finish_poly_pass(&mut self, cmd: u32) {
        let p = self.gp0_params;
        let quad = cmd & 0x08 != 0;
        let gouraud = cmd & 0x10 != 0;
        let textured = cmd & 0x04 != 0;
        let st = cmd & 0x02 != 0;
        let raw = textured && cmd & 0x01 != 0;
        let col0 = self.gp0_cmd & 0xFF_FFFF;
        let mut v = [Vert::default(); 4];
        let mut tpage = self.regs[1]; // sprites use E1; polys get it from uv
        let mut clut = 0u32;
        let mut i = 0usize;
        let n = if quad { 4 } else { 3 };
        for k in 0..n {
            let c = if gouraud {
                if k == 0 {
                    col0
                } else {
                    let w = p[i];
                    i += 1;
                    w & 0xFF_FFFF
                }
            } else {
                col0
            };
            v[k] = self.vtx(p[i], c);
            i += 1;
            if textured {
                let uv = p[i];
                i += 1;
                v[k].u = (uv & 0xFF) as i32;
                v[k].v = ((uv >> 8) & 0xFF) as i32;
                if k == 0 {
                    clut = uv >> 16;
                }
                if k == 1 {
                    tpage = uv >> 16;
                }
            }
        }
        if textured {
            // a textured polygon's tpage also updates the draw mode
            self.regs[1] = (self.regs[1] & !0x9FF) | (tpage & 0x9FF);
        }
        let sh = Shade {
            textured,
            gouraud,
            semitrans: st,
            raw,
            tpage,
            clut,
        };
        if self.hw_render {
            self.record_prim(&v[..n], n, &sh, 0);
        }
        self.tri(&[v[0], v[1], v[2]], sh);
        if quad {
            self.tri(&[v[1], v[2], v[3]], sh);
        }
    }

    /// Lines 0x40-0x5F (single-segment; polylines go via finish_polyline).
    fn finish_line(&mut self, cmd: u32) {
        let p = &self.gp0_params;
        let gouraud = cmd & 0x10 != 0;
        let st = cmd & 0x02 != 0;
        let c0 = self.gp0_cmd & 0xFF_FFFF;
        let c1 = if gouraud { p[1] & 0xFF_FFFF } else { c0 };
        let (a, b) = if gouraud {
            (self.vtx(p[0], c0), self.vtx(p[2], c1))
        } else {
            (self.vtx(p[0], c0), self.vtx(p[1], c0))
        };
        if self.hw_render {
            let sh = Shade {
                textured: false,
                gouraud,
                semitrans: st,
                raw: false,
                tpage: self.regs[1],
                clut: 0,
            };
            self.record_prim(&[a, b], 2, &sh, hw_flags::LINE);
        }
        self.line(a, b, st);
    }

    /// Polyline strip — `words` ends with the 0x55555555 terminator.
    fn finish_polyline(&mut self, words: &[u32]) {
        let cmd = words[0] >> 24;
        let gouraud = cmd & 0x10 != 0;
        let st = cmd & 0x02 != 0;
        let mut col = words[0] & 0xFF_FFFF;
        let mut idx = 1usize;
        let mut prev: Option<Vert> = None;
        while idx < words.len() && words[idx] & 0xF000_F000 != 0x5000_5000 {
            if gouraud && prev.is_some() {
                col = words[idx] & 0xFF_FFFF;
                idx += 1;
                if idx >= words.len() || words[idx] & 0xF000_F000 == 0x5000_5000 {
                    break;
                }
            }
            let v = self.vtx(words[idx], col);
            idx += 1;
            if let Some(p) = prev {
                if self.hw_render {
                    let sh = Shade {
                        textured: false,
                        gouraud,
                        semitrans: st,
                        raw: false,
                        tpage: self.regs[1],
                        clut: 0,
                    };
                    self.record_prim(&[p, v], 2, &sh, hw_flags::LINE);
                }
                self.line(p, v, st);
            }
            prev = Some(v);
        }
    }

    /// Rectangles / sprites 0x60-0x7F.
    fn finish_rect(&mut self, cmd: u32) {
        let p = &self.gp0_params;
        let textured = cmd & 0x04 != 0;
        let st = cmd & 0x02 != 0;
        let col = self.gp0_cmd & 0xFF_FFFF;
        let xy = p[0];
        let x = (((xy << 21) as i32) >> 21) + self.off_x;
        let y = (((xy >> 16) << 21) as i32 >> 21) + self.off_y;
        let mut clut = 0u32;
        let (mut u0, mut v0) = (0i32, 0i32);
        let mut i = 1usize;
        if textured {
            let uv = p[i];
            i += 1;
            u0 = (uv & 0xFF) as i32;
            v0 = ((uv >> 8) & 0xFF) as i32;
            clut = uv >> 16;
        }
        let (w, h) = match cmd & 0x18 {
            0x08 => (1i32, 1i32),
            0x10 => (8, 8),
            0x18 => (16, 16),
            _ => {
                let wh = p[i];
                ((wh & 0x3FF) as i32, ((wh >> 16) & 0x1FF) as i32)
            }
        };
        let raw = cmd & 0x01 != 0;
        let sh = Shade {
            textured,
            gouraud: false,
            semitrans: st,
            raw,
            tpage: self.regs[1],
            clut,
        };
        // E1 bits12/13: textured-rect x/y flip
        let xflip = self.regs[1] & (1 << 12) != 0;
        let yflip = self.regs[1] & (1 << 13) != 0;
        if self.hw_render {
            // corners TL,TR,BL,BR; verts[0] carries raw u0/v0, the
            // backend derives per-corner UVs incl. flips (RECT flag)
            let (u0x, u1x) = if xflip { (u0 + w, u0) } else { (u0, u0 + w) };
            let (v0y, v1y) = if yflip { (v0 + h, v0) } else { (v0, v0 + h) };
            let mk = |x: i32, y: i32, u: i32, v: i32| Vert {
                x,
                y,
                u,
                v,
                r: (col & 0xFF) as i32,
                g: ((col >> 8) & 0xFF) as i32,
                b: ((col >> 16) & 0xFF) as i32,
            };
            let v = [
                mk(x, y, u0x, v0y),
                mk(x + w, y, u1x, v0y),
                mk(x, y + h, u0x, v1y),
                mk(x + w, y + h, u1x, v1y),
            ];
            self.record_prim(
                &v,
                4,
                &sh,
                hw_flags::RECT | hw_flags::XFLIP * xflip as u16 | hw_flags::YFLIP * yflip as u16,
            );
        }
        let (cx0, cy0, cx1, cy1) = self.clip();
        for dy in 0..h.max(0) {
            for dx in 0..w.max(0) {
                let (px, py) = (x + dx, y + dy);
                if px < cx0 || px > cx1 || py < cy0 || py > cy1 {
                    continue;
                }
                let (tu, tv) = (
                    if xflip { w - 1 - dx } else { dx },
                    if yflip { h - 1 - dy } else { dy },
                );
                let c = if textured {
                    match self.texel(u0 + tu, v0 + tv, sh.tpage, sh.clut) {
                        Some((t, semi)) => {
                            let m = if sh.raw { t } else { modulate(t, col) };
                            if st && semi {
                                Some(self.blend(px, py, m, sh.tpage))
                            } else {
                                Some(m | if semi { 0x8000 } else { 0 })
                            }
                        }
                        None => None,
                    }
                } else {
                    let c15 = rgb15(col);
                    if st {
                        Some(self.blend(px, py, c15, self.regs[1]))
                    } else {
                        Some(c15)
                    }
                };
                if let Some(c) = c {
                    self.put(px, py, c);
                }
            }
        }
    }

    /// Flat/gouraud, optionally textured triangle. Pixels strictly
    /// inside plus those on top/left edges are drawn; right/bottom
    /// edges belong to the neighbouring primitive.
    fn tri(&mut self, v: &[Vert; 3], sh: Shade) {
        let edge = edge_value;
        let mut v = *v;
        let mut area = edge(&v[0], &v[1], v[2].x, v[2].y);
        if area == 0 {
            return;
        }
        if area < 0 {
            v.swap(1, 2);
            area = -area;
        }
        let minx = v[0].x.min(v[1].x).min(v[2].x);
        let maxx = v[0].x.max(v[1].x).max(v[2].x);
        let miny = v[0].y.min(v[1].y).min(v[2].y);
        let maxy = v[0].y.max(v[1].y).max(v[2].y);
        let (vram_w, vram_h) = if self.render_hires {
            (
                VRAM_W as i32 * self.render_scale as i32,
                VRAM_H as i32 * self.render_scale as i32,
            )
        } else {
            (VRAM_W as i32, VRAM_H as i32)
        };
        if maxx - minx >= vram_w || maxy - miny >= vram_h {
            return;
        }
        let (cx0, cy0, cx1, cy1) = self.clip();
        let (x0, x1) = (minx.max(cx0), maxx.min(cx1));
        let (y0, y1) = (miny.max(cy0), maxy.min(cy1));
        if x0 > x1 || y0 > y1 {
            return;
        }
        // with positive area in this y-down space top edges run +x and
        // left edges run -y; those own their boundary pixels
        // (top-left rule), right/bottom edges don't.
        let bias = |a: &Vert, b: &Vert| -> i64 {
            if (a.y == b.y && b.x > a.x) || b.y < a.y {
                0
            } else {
                -1
            }
        };
        let (b0, b1, b2) = (bias(&v[1], &v[2]), bias(&v[2], &v[0]), bias(&v[0], &v[1]));
        let dither = self.dithering
            && self.regs[1] & (1 << 9) != 0
            && (sh.gouraud || (sh.textured && !sh.raw));
        // interpolation at the edges can overshoot the vertex range by
        // a texel and wrap into neighbouring texture data
        let (umin, umax) = (
            v[0].u.min(v[1].u).min(v[2].u),
            v[0].u.max(v[1].u).max(v[2].u),
        );
        let (vmin, vmax) = (
            v[0].v.min(v[1].v).min(v[2].v),
            v[0].v.max(v[1].v).max(v[2].v),
        );
        let edge_dx = [
            -(v[2].y - v[1].y) as i64,
            -(v[0].y - v[2].y) as i64,
            -(v[1].y - v[0].y) as i64,
        ];
        let weighted = |a: i32, b: i32, c: i32, w: [i64; 3]| -> i64 {
            a as i64 * w[0] + b as i64 * w[1] + c as i64 * w[2]
        };
        let weighted_dx = |a: i32, b: i32, c: i32| -> i64 {
            a as i64 * edge_dx[0] + b as i64 * edge_dx[1] + c as i64 * edge_dx[2]
        };
        let color_delta = sh.gouraud.then(|| {
            [
                split_interpolation(weighted_dx(v[0].r, v[1].r, v[2].r), area),
                split_interpolation(weighted_dx(v[0].g, v[1].g, v[2].g), area),
                split_interpolation(weighted_dx(v[0].b, v[1].b, v[2].b), area),
            ]
        });
        let uv_delta = sh.textured.then(|| {
            [
                split_interpolation(weighted_dx(v[0].u, v[1].u, v[2].u), area),
                split_interpolation(weighted_dx(v[0].v, v[1].v, v[2].v), area),
            ]
        });
        let biases = [b0, b1, b2];
        if self.render_hires {
            // Enhanced target: same rasterization against a scaled
            // surface. Scanlines are independent, so large polygons
            // are split into row bands across worker threads.
            let ctx = HiCtx {
                vram: &self.vram,
                texwin: self.texwin,
                e1: self.regs[1],
                mask_set: self.mask_set,
                mask_check: self.mask_check,
                v,
                sh,
                dither,
                color_delta,
                uv_delta,
                umin,
                umax,
                vmin,
                vmax,
                area,
            };
            let hires_w = VRAM_W * self.render_scale;
            let mut spans: Vec<(i32, i32, i32)> = Vec::with_capacity((y1 - y0 + 1) as usize);
            for y in y0..=y1 {
                if let Some((sx0, sx1)) = triangle_span(&v, x0, x1, y, biases) {
                    spans.push((y, sx0, sx1));
                }
            }
            let total_px: usize = spans.iter().map(|&(_, a, b)| (b - a + 1) as usize).sum();
            // Threading pays off only for meaningful fills; spawning
            // costs ~10-20us per band so small polygons stay sequential
            // and medium ones use fewer bands.
            let threads = if spans.len() >= 4 && total_px >= 24_576 {
                let max = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(8);
                ((total_px / 32_768) as usize).clamp(1, max)
            } else {
                1
            };
            if threads <= 1 {
                for &(y, sx0, sx1) in &spans {
                    let off = y as usize * hires_w;
                    tri_row_hires(&ctx, &mut self.hires_vram[off..off + hires_w], y, sx0, sx1);
                }
            } else {
                // partition spans into contiguous, pixel-balanced bands
                let target = total_px.div_ceil(threads);
                let mut bands: Vec<(usize, usize)> = Vec::with_capacity(threads);
                let (mut start, mut acc) = (0usize, 0usize);
                for (idx, &(_, a, b)) in spans.iter().enumerate() {
                    acc += (b - a + 1) as usize;
                    if acc >= target {
                        bands.push((start, idx + 1));
                        start = idx + 1;
                        acc = 0;
                    }
                }
                if start < spans.len() {
                    bands.push((start, spans.len()));
                }
                let ctx = &ctx;
                let spans = &spans;
                let mut rest = self.hires_vram.as_mut_slice();
                let mut cursor = 0;
                std::thread::scope(|s| {
                    for &(bs, be) in &bands {
                        let (row0, row1) = (spans[bs].0, spans[be - 1].0);
                        let skip = (row0 - cursor) as usize * hires_w;
                        let take = (row1 - row0 + 1) as usize * hires_w;
                        let r = std::mem::replace(&mut rest, &mut []);
                        let (head, tail) = r.split_at_mut(skip + take);
                        rest = tail;
                        cursor = row1 + 1;
                        let band_rows = &mut head[skip..];
                        s.spawn(move || {
                            for &(y, sx0, sx1) in &spans[bs..be] {
                                let off = (y - row0) as usize * hires_w;
                                tri_row_hires(ctx, &mut band_rows[off..off + hires_w], y, sx0, sx1);
                            }
                        });
                    }
                });
            }
            return;
        }
        for y in y0..=y1 {
            let Some((span_x0, span_x1)) = triangle_span(&v, x0, x1, y, biases) else {
                continue;
            };
            let weights = [
                edge(&v[1], &v[2], span_x0, y),
                edge(&v[2], &v[0], span_x0, y),
                edge(&v[0], &v[1], span_x0, y),
            ];
            let mut colors = color_delta.map(|delta| {
                [
                    interpolation(weighted(v[0].r, v[1].r, v[2].r, weights), delta[0], area),
                    interpolation(weighted(v[0].g, v[1].g, v[2].g, weights), delta[1], area),
                    interpolation(weighted(v[0].b, v[1].b, v[2].b, weights), delta[2], area),
                ]
            });
            let mut texcoords = uv_delta.map(|delta| {
                [
                    interpolation(weighted(v[0].u, v[1].u, v[2].u, weights), delta[0], area),
                    interpolation(weighted(v[0].v, v[1].v, v[2].v, weights), delta[1], area),
                ]
            });
            for x in span_x0..=span_x1 {
                let (r, g, b) = if sh.gouraud {
                    let c = colors.as_ref().unwrap();
                    (c[0].value, c[1].value, c[2].value)
                } else {
                    (v[0].r, v[0].g, v[0].b)
                };
                if sh.textured {
                    let uv = texcoords.as_ref().unwrap();
                    let u = uv[0].value.clamp(umin, umax);
                    let tv = uv[1].value.clamp(vmin, vmax);
                    if let Some((t, semi)) = self.texel(u, tv, sh.tpage, sh.clut) {
                        let m = if sh.raw {
                            t
                        } else {
                            let col = (r.clamp(0, 255) as u32)
                                | ((g.clamp(0, 255) as u32) << 8)
                                | ((b.clamp(0, 255) as u32) << 16);
                            modulate_d(t, col, dither, x, y)
                        };
                        let c = if sh.semitrans && semi {
                            self.blend(x, y, m, sh.tpage)
                        } else {
                            m | if semi { 0x8000 } else { 0 }
                        };
                        self.put(x, y, c);
                    }
                } else {
                    let c15 = shade15(r, g, b, dither, x, y);
                    let c = if sh.semitrans {
                        self.blend(x, y, c15, self.regs[1])
                    } else {
                        c15
                    };
                    self.put(x, y, c);
                }
                if let Some(c) = colors.as_mut() {
                    for interp in c {
                        interp.advance(area);
                    }
                }
                if let Some(uv) = texcoords.as_mut() {
                    for interp in uv {
                        interp.advance(area);
                    }
                }
            }
        }
    }

    /// 1-pixel Bresenham line with optional gouraud color ramp.
    fn line(&mut self, a: Vert, b: Vert, st: bool) {
        let (cx0, cy0, cx1, cy1) = self.clip();
        let (mut x, mut y) = (a.x, a.y);
        let dx = (b.x - a.x).abs();
        let dy = -(b.y - a.y).abs();
        let sx = if a.x < b.x { 1 } else { -1 };
        let sy = if a.y < b.y { 1 } else { -1 };
        let steps = dx.max(-dy).max(1);
        let mut err = dx + dy;
        let mut i = 0i32;
        loop {
            let t = |ca: i32, cb: i32| (ca + (cb - ca) * i / steps) as u16;
            let c15 = ((t(a.b, b.b) >> 3) << 10) | ((t(a.g, b.g) >> 3) << 5) | (t(a.r, b.r) >> 3);
            if x >= cx0 && x <= cx1 && y >= cy0 && y <= cy1 {
                let c = if st {
                    self.blend(x, y, c15, self.regs[1])
                } else {
                    c15
                };
                self.put(x, y, c);
            }
            if x == b.x && y == b.y {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
            i += 1;
        }
    }

    /// Texture lookup: returns (rgb15, semitrans-flag) or None when the
    /// texel is transparent black. Applies the E2 texture window to
    /// the coordinates first.
    fn texel(&self, u: i32, v: i32, tpage: u32, clut: u32) -> Option<(u16, bool)> {
        texel_at(&self.vram, self.texwin, u, v, tpage, clut)
    }

    /// Semi-transparency blend against the framebuffer pixel.
    /// Modes (tpage bits5-6): 0=B/2+F/2 1=B+F 2=B-F 3=B+F/4.
    fn blend(&self, x: i32, y: i32, src: u16, tpage: u32) -> u16 {
        let dst = if self.render_hires {
            let width = VRAM_W * self.render_scale;
            self.hires_vram[y as usize * width + x as usize]
        } else {
            self.vram[(y as usize * VRAM_W + x as usize) & (VRAM_W * VRAM_H - 1)]
        };
        blend_px(dst, src, tpage)
    }

    /// Clip rect from draw-area env (E3 top-left, E4 bottom-right).
    fn clip(&self) -> (i32, i32, i32, i32) {
        let e3 = self.regs[3];
        let e4 = self.regs[4];
        let scale = if self.render_hires {
            self.render_scale as u32
        } else {
            1
        };
        let mut clip = (
            ((e3 & 0x3FF) * scale) as i32,
            (((e3 >> 10) & 0x1FF) * scale) as i32,
            ((e4 & 0x3FF) * scale) as i32,
            (((e4 >> 10) & 0x1FF) * scale) as i32,
        );
        if self.render_hires && !self.display_24bpp() {
            // Only regions that can ever be displayed need enhanced
            // coverage; the union covers both double-buffer halves.
            let (sx0, sy0, sx1, sy1) = self.seen_display;
            if sx1 > sx0 && sy1 > sy0 {
                clip.0 = clip.0.max((sx0 * scale) as i32);
                clip.1 = clip.1.max((sy0 * scale) as i32);
                clip.2 = clip.2.min((sx1 * scale) as i32 - 1);
                clip.3 = clip.3.min((sy1 * scale) as i32 - 1);
            }
        }
        clip
    }

    fn put(&mut self, x: i32, y: i32, mut c: u16) {
        let (width, height) = if self.render_hires {
            (VRAM_W * self.render_scale, VRAM_H * self.render_scale)
        } else {
            (VRAM_W, VRAM_H)
        };
        if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 {
            return;
        }
        let i = y as usize * width + x as usize;
        // E6.1: don't draw where the mask bit is already set
        if self.mask_check {
            let old = if self.render_hires {
                self.hires_vram[i]
            } else {
                self.vram[i]
            };
            if old & 0x8000 != 0 {
                return;
            }
        }
        // E6.0: force mask bit on drawn pixels
        if self.mask_set {
            c |= 0x8000;
        }
        if self.render_hires {
            self.hires_vram[i] = c;
        } else {
            self.write_native_pixel(x as usize, y as usize, c);
        }
    }

    /// GP1 — control commands.
    pub fn gp1(&mut self, v: u32) {
        match v >> 24 {
            0x00 => {
                // reset: drop transfers, default registers, display off
                self.gp0_len = 0;
                self.poly_words.clear();
                self.vram_wr = None;
                self.vram_rd = None;
                self.regs = [0; 8];
                self.texwin = 0;
                self.off_x = 0;
                self.off_y = 0;
                self.mask_set = false;
                self.mask_check = false;
                self.irq1 = false;
                self.disp_disabled = true;
                self.dma_dir = 0;
                self.disp_mode = 0;
                self.disp_x = 0;
                self.disp_y = 0;
                self.h_range = (0x200, 0xC00);
                self.v_range = (0x10, 0x100);
                self.seen_display = (0, 0, 0, 0);
            }
            0x01 => {
                // clear command FIFO
                self.gp0_len = 0;
                self.poly_words.clear();
                self.vram_wr = None;
            }
            0x02 => self.irq1 = false,               // ack irq
            0x03 => self.disp_disabled = v & 1 != 0, // display enable
            0x04 => self.dma_dir = (v & 3) as u8,    // dma direction
            0x05 => {
                // display area start
                self.disp_x = (v & 0x3FF) as u16;
                self.disp_y = ((v >> 10) & 0x1FF) as u16;
                self.note_display();
            }
            0x06 => {
                // h display range
                self.h_range = ((v & 0xFFF) as u16, ((v >> 12) & 0xFFF) as u16);
            }
            0x07 => {
                // v display range
                self.v_range = ((v & 0x3FF) as u16, ((v >> 10) & 0x3FF) as u16);
                self.note_display();
            }
            0x08 => {
                // display mode
                self.disp_mode = v & 0xFF;
                self.note_display();
            }
            0x10..=0x1F => self.info_sel = (v & 7) as u8, // gpu info select
            _ => {}
        }
    }

    /// Native display resolution from GP1(08h): horizontal 256/320/
    /// 368/512/640, vertical 240/480.
    pub fn display_size(&self) -> (u32, u32) {
        let hres = if self.disp_mode & (1 << 6) != 0 {
            368
        } else {
            match self.disp_mode & 3 {
                0 => 256,
                1 => 320,
                2 => 512,
                _ => 640,
            }
        };
        let vres = if self.disp_mode & 0x24 == 0x24 {
            480
        } else {
            240
        };
        (hres, vres)
    }

    /// True when displaying 24bpp (GP1 08h bit4).
    pub fn display_24bpp(&self) -> bool {
        self.disp_mode & 0x10 != 0
    }

    /// True when the display is enabled (GP1 03h bit0 clear).
    pub fn display_enabled(&self) -> bool {
        !self.disp_disabled
    }

    /// PAL video mode (GP1(08h) bit3).
    pub fn is_pal(&self) -> bool {
        self.disp_mode & 0x08 != 0
    }

    /// Interlaced output (GP1(08h) bit5).
    pub fn interlaced(&self) -> bool {
        self.disp_mode & 0x20 != 0
    }

    /// GPU dot-clock divider for the current horizontal resolution.
    pub fn dot_divider(&self) -> u32 {
        if self.disp_mode & (1 << 6) != 0 {
            7
        } else {
            [10, 8, 5, 4][(self.disp_mode & 3) as usize]
        }
    }

    /// Render the visible display area into `out` as 0x00RRGGBB.
    /// Returns (width, height). A disabled display yields black.
    pub fn display_rgb(&self, out: &mut Vec<u32>) -> (usize, usize) {
        let (w, h480) = self.display_size();
        // 24bpp display reads use a packed byte stream that is not mirrored
        // into the enhanced surface; a host-GPU backend owns the scaled
        // surface entirely. Both cases output at native resolution.
        let scale = if self.display_24bpp() || self.hw_render {
            1
        } else {
            self.render_scale
        };
        let w = w as usize * scale;
        // visible line count from the vertical display range
        let lines = (self.v_range.1 as i32 - self.v_range.0 as i32).clamp(16, 288) as usize;
        let h = (if h480 == 480 { lines * 2 } else { lines }) * scale;
        out.clear();
        out.resize(w * h, 0);
        if self.disp_disabled {
            return (w, h);
        }
        let (dx, dy) = (self.disp_x as usize, self.disp_y as usize);
        if scale > 1 {
            let hires_w = VRAM_W * scale;
            let hires_h = VRAM_H * scale;
            let hires = &self.hires_vram;
            let expand = |band: &mut [u32], y0: usize| {
                for (ly, px_row) in band.chunks_exact_mut(w).enumerate() {
                    let row = ((dy * scale + y0 + ly) % hires_h) * hires_w;
                    for (x, px) in px_row.iter_mut().enumerate() {
                        let c = hires[row + ((dx * scale + x) % hires_w)] as u32;
                        let (r, g, b) = (c & 0x1F, (c >> 5) & 0x1F, (c >> 10) & 0x1F);
                        let e = |v: u32| (v << 3) | (v >> 2);
                        *px = e(r) << 16 | e(g) << 8 | e(b);
                    }
                }
            };
            // ~2M px at 4x — split into row bands across threads
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(8)
                .min(h / 8);
            if threads <= 1 {
                expand(out, 0);
            } else {
                let rows = h.div_ceil(threads);
                let mut rest = out.as_mut_slice();
                let mut y0 = 0usize;
                std::thread::scope(|s| {
                    while !rest.is_empty() {
                        let take = (rows * w).min(rest.len());
                        let r = std::mem::replace(&mut rest, &mut []);
                        let (head, tail) = r.split_at_mut(take);
                        rest = tail;
                        let band_y = y0;
                        y0 += head.len() / w;
                        s.spawn(move || expand(head, band_y));
                    }
                });
            }
            return (w, h);
        }
        for y in 0..h {
            let row = ((dy + y) & (VRAM_H - 1)) * VRAM_W;
            let o = &mut out[y * w..(y + 1) * w];
            if self.display_24bpp() {
                for (x, px) in o.iter_mut().enumerate() {
                    let b0 = dx * 2 + x * 3;
                    let byte = |i: usize| -> u32 {
                        let hw = self.vram[row + ((i / 2) & (VRAM_W - 1))];
                        ((hw >> ((i & 1) * 8)) & 0xFF) as u32
                    };
                    *px = byte(b0) << 16 | byte(b0 + 1) << 8 | byte(b0 + 2);
                }
            } else {
                for (x, px) in o.iter_mut().enumerate() {
                    let c = self.vram[row + ((dx + x) & (VRAM_W - 1))] as u32;
                    let (r, g, b) = (c & 0x1F, (c >> 5) & 0x1F, (c >> 10) & 0x1F);
                    let e = |v: u32| (v << 3) | (v >> 2);
                    *px = e(r) << 16 | e(g) << 8 | e(b);
                }
            }
        }
        (w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draw-area covering all VRAM, offset 0.
    fn gpu() -> Gpu {
        let mut g = Gpu::new();
        g.gp0(0xE300_0000);
        g.gp0(0xE400_0000 | (511 << 10) | 1023);
        g.gp0(0xE500_0000);
        g
    }

    fn count(g: &Gpu, c: u16) -> usize {
        g.vram.iter().filter(|&&p| p == c).count()
    }

    #[test]
    fn incremental_interpolation_matches_division() {
        for &area in &[1i64, 2, 7, 31, 1024, 524_000] {
            for &start in &[-area * 2 - 1, -area, -1, 0, 1, area - 1, area, area * 2 + 1] {
                for &delta in &[-area - 1, -area, -1, 0, 1, area, area + 1] {
                    let (dv, dr) = split_interpolation(delta, area);
                    let mut interp = interpolation(start, (dv, dr), area);
                    let mut numerator = start;
                    for _ in 0..12 {
                        assert_eq!(interp.value as i64, numerator.div_euclid(area));
                        interp.advance(area);
                        numerator += delta;
                    }
                }
            }
        }
    }

    #[test]
    fn triangle_spans_match_pixel_edge_rule() {
        let mut seed = 0xA341_316Cu32;
        for _ in 0..1000 {
            let mut v = [Vert::default(); 3];
            for p in &mut v {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                p.x = ((seed >> 16) % 41) as i32 - 5;
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                p.y = ((seed >> 16) % 41) as i32 - 5;
            }
            let area = edge_value(&v[0], &v[1], v[2].x, v[2].y);
            if area == 0 {
                continue;
            }
            if area < 0 {
                v.swap(1, 2);
            }
            let bias = |a: &Vert, b: &Vert| {
                if (a.y == b.y && b.x > a.x) || b.y < a.y {
                    0
                } else {
                    -1
                }
            };
            let biases = [bias(&v[1], &v[2]), bias(&v[2], &v[0]), bias(&v[0], &v[1])];
            for y in -5..=35 {
                let covered: Vec<i32> = (-5..=35)
                    .filter(|&x| {
                        edge_value(&v[1], &v[2], x, y) + biases[0] >= 0
                            && edge_value(&v[2], &v[0], x, y) + biases[1] >= 0
                            && edge_value(&v[0], &v[1], x, y) + biases[2] >= 0
                    })
                    .collect();
                let expected = covered.first().zip(covered.last()).map(|(&a, &b)| (a, b));
                assert_eq!(triangle_span(&v, -5, 35, y, biases), expected);
            }
        }
    }

    /// A quad split into two triangles must cover exactly w*h pixels:
    /// no seam on the shared diagonal, no right/bottom overdraw.
    #[test]
    fn quad_covers_exactly_without_seams() {
        for &(x0, y0, x1, y1) in &[(10, 10, 42, 26), (100, 50, 101, 51), (5, 7, 60, 9)] {
            let mut g = gpu();
            g.gp0(0x2800_00FF); // mono quad, red
            for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                g.gp0(((y as u32) << 16) | x as u32);
            }
            let red = 0x001F;
            assert_eq!(
                count(&g, red),
                ((x1 - x0) * (y1 - y0)) as usize,
                "quad {x0},{y0}-{x1},{y1}"
            );
            assert_eq!(g.vram[y0 * 1024 + x0], red);
            assert_eq!(g.vram[y0 * 1024 + x1], 0, "right edge must be excluded");
            assert_eq!(g.vram[y1 * 1024 + x0], 0, "bottom edge must be excluded");
        }
    }

    /// Two quads sharing an edge tile without gaps or overlap, in
    /// both winding orders.
    #[test]
    fn adjacent_quads_tile() {
        let mut g = gpu();
        // semi-transparent additive blend would reveal overlap; use
        // distinct colors and count instead
        for (i, (xa, xb)) in [(0u32, 17u32), (17, 40)].iter().enumerate() {
            g.gp0(0x2800_0000 | if i == 0 { 0xFF } else { 0xFF00 });
            let pts = if i == 0 {
                [(*xa, 3u32), (*xb, 3), (*xa, 30), (*xb, 30)]
            } else {
                [(*xb, 30), (*xa, 30), (*xb, 3), (*xa, 3)]
            };
            for (x, y) in pts {
                g.gp0((y << 16) | x);
            }
        }
        let filled = g.vram.iter().filter(|&&p| p != 0).count();
        assert_eq!(filled, 40 * 27);
    }

    #[test]
    fn enhanced_polygon_pass_keeps_native_vram_and_scales_display() {
        let mut native = gpu();
        let mut enhanced = gpu();
        assert!(enhanced.set_internal_scale(2));
        for g in [&mut native, &mut enhanced] {
            g.gp0(0x2000_00FF);
            for (x, y) in [(20u32, 20u32), (44, 20), (20, 44)] {
                g.gp0((y << 16) | x);
            }
        }

        assert_eq!(native.vram, enhanced.vram);
        let native_pixels = count(&native, 0x001F);
        let enhanced_pixels = enhanced
            .hires_vram
            .iter()
            .filter(|&&pixel| pixel == 0x001F)
            .count();
        assert!(native_pixels > 0);
        assert!(
            enhanced_pixels > native_pixels * 3,
            "native={native_pixels}, enhanced={enhanced_pixels}"
        );

        enhanced.disp_disabled = false;
        let mut output = Vec::new();
        let (width, height) = enhanced.display_rgb(&mut output);
        assert_eq!(width, native.display_size().0 as usize * 2);
        assert_eq!(height, 480);
        assert_eq!(output.len(), width * height);
    }

    /// Large enhanced polygons take the threaded row-band path. A flat
    /// quad makes dropped/duplicated rows or corrupt writes visible:
    /// every covered pixel must be exactly the flat colour.
    #[test]
    fn enhanced_parallel_rows_have_no_holes() {
        let mut g = gpu();
        assert!(g.set_internal_scale(4));
        g.gp0(0x2800_00FF); // flat quad, red — TL,TR,BL,BR order
        for (x, y) in [(40u32, 30u32), (400, 30), (40, 220), (400, 220)] {
            g.gp0((y << 16) | x);
        }
        let w = VRAM_W * 4;
        let filled = g.hires_vram.iter().filter(|&&p| p != 0).count();
        // scaled quad covers 1440x760 hires pixels minus edge ownership
        assert!(filled > 1_000_000, "filled={filled}");
        assert!(g.hires_vram.iter().all(|&p| p == 0 || p == 0x001F));
        // every row inside the quad must have coverage — no band gaps
        for y in 120..880usize {
            let row = &g.hires_vram[y * w..(y + 1) * w];
            assert!(row.iter().any(|&p| p == 0x001F), "row {y} has no coverage");
        }
    }

    #[test]
    fn hardware_rect_records_cyclic_quad_vertex_order() {
        let mut g = gpu();
        assert!(g.set_internal_scale(2));
        g.set_hw_render(true);
        g.gp0(0x6000_00FF); // variable-size flat rect
        g.gp0((20 << 16) | 10);
        g.gp0((7 << 16) | 13);
        let prim = g
            .take_hw_ops()
            .into_iter()
            .find_map(|op| match op {
                HwOp::Prim(p) => Some(p),
                _ => None,
            })
            .expect("recorded rectangle");
        assert_eq!(prim.n, 4);
        assert_eq!(
            prim.verts[..4]
                .iter()
                .map(|v| (v.x, v.y))
                .collect::<Vec<_>>(),
            [(10, 20), (23, 20), (10, 27), (23, 27)]
        );
    }

    #[test]
    fn hardware_textured_rect_records_inclusive_filter_bounds() {
        let mut g = gpu();
        assert!(g.set_internal_scale(2));
        g.set_hw_render(true);
        g.gp0(0x6400_00FF); // variable textured rectangle
        g.gp0((20 << 16) | 10);
        g.gp0((9 << 8) | 5); // UV origin
        g.gp0((7 << 16) | 13);
        let prim = g
            .take_hw_ops()
            .into_iter()
            .find_map(|op| match op {
                HwOp::Prim(p) => Some(p),
                _ => None,
            })
            .expect("recorded textured rectangle");
        assert_eq!(prim.uv_bounds, [5, 9, 17, 15]);
        assert_eq!(
            prim.verts[..4]
                .iter()
                .map(|v| (v.u, v.v))
                .collect::<Vec<_>>(),
            [(5, 9), (18, 9), (5, 16), (18, 16)]
        );
    }

    #[test]
    fn hardware_fill_records_native_resolve() {
        let mut g = gpu();
        assert!(g.set_internal_scale(2));
        g.set_hw_render(true);
        // GP0 fill at (16,12), aligned 16x2.
        g.gp0(0x0200_FFFF);
        g.gp0((12 << 16) | 16);
        g.gp0((2 << 16) | 16);
        assert!(g.take_hw_ops().iter().any(|op| matches!(
            op,
            HwOp::Resolve {
                x0: 16,
                y0: 12,
                x1: 32,
                y1: 14
            }
        )));
    }

    #[test]
    fn enhanced_scale_falls_back_for_24bpp_output() {
        let mut g = gpu();
        assert!(g.set_internal_scale(4));
        g.disp_mode |= 0x10;
        g.disp_disabled = false;
        let mut output = Vec::new();
        let (width, _) = g.display_rgb(&mut output);
        assert_eq!(width, g.display_size().0 as usize);
    }
}
