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

/// 4x4 ordered-dither offsets (GP0 E1 bit9).
const DITHER: [[i32; 4]; 4] = [[-4, 0, -3, 1], [2, -2, 3, -1], [-3, 1, -4, 0], [3, -1, 2, -2]];

/// 8-bit channel → 5-bit with optional dithering.
#[inline]
fn ch5(v: i32, dither: bool, x: i32, y: i32) -> u16 {
    let v = if dither { v + DITHER[(y & 3) as usize][(x & 3) as usize] } else { v };
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

/// Multiply a texel by the primitive's modulation color (0x80 = 1.0).
fn modulate(tex: u16, col: u32) -> u16 {
    let f = |t: u16, c: u32| (((t as u32) * c) / 0x80).min(0x1F) as u16;
    f(tex & 0x1F, col & 0xFF)
        | (f((tex >> 5) & 0x1F, (col >> 8) & 0xFF) << 5)
        | (f((tex >> 10) & 0x1F, (col >> 16) & 0xFF) << 10)
}

pub struct Gpu {
    pub vram: Vec<u16>,
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
}

impl Gpu {
    pub fn new() -> Self {
        Self {
            vram: vec![0; VRAM_W * VRAM_H],
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
        }
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
            3 => self.regs[3] & 0x7_FFFF, // draw area top-left
            4 => self.regs[4] & 0x7_FFFF, // draw area bottom-right
            5 => self.regs[5] & 0x3F_FFFF, // draw offset
            7 => 2,                      // GPU type (old, 160pin)
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
                self.vram[(py * VRAM_W + px) & (VRAM_W * VRAM_H - 1)] =
                    (v >> (half * 16)) as u16;
            }
            self.vram_wr = if left > 2 {
                Some((x, y, w, _h, left - 2, total))
            } else {
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
            0x00 | 0x01 => 0,                     // NOP / clear cache
            0x1F => {
                // interrupt request → GPUSTAT.24 + I_STAT bit1
                self.irq1 = true;
                0
            }
            0x02 => 2,                            // fill rect
            0x20..=0x23 => 3,                     // mono tri
            0x24..=0x27 => 6,                     // textured tri
            0x28..=0x2B => 4,                     // mono quad
            0x2C..=0x2F => 8,                     // textured quad
            0x30..=0x33 => 5,                     // gouraud tri
            0x34..=0x37 => 8,                     // gouraud tex tri
            0x38..=0x3B => 7,                     // gouraud quad
            0x3C..=0x3F => 11,                    // gouraud tex quad
            0x40..=0x43 => 2,                     // mono line
            0x48..=0x4F | 0x58..=0x5F => {
                // polyline: variable, terminated by 0x55555555
                self.poly_words.push(v);
                return;
            }
            0x50..=0x53 => 3,                     // gouraud line
            0x60..=0x63 => 2,                     // mono rect (var)
            0x64..=0x67 => 3,                     // tex sprite (var)
            0x68..=0x6B => 1,                     // mono 1x1
            0x6C..=0x6F => 2,                     // tex 1x1
            0x70..=0x73 => 1,                     // mono 8x8
            0x74..=0x77 => 2,                     // tex 8x8
            0x78..=0x7B => 1,                     // mono 16x16
            0x7C..=0x7F => 2,                     // tex 16x16
            0x80..=0x9F => 3,                     // vram→vram copy
            0xA0..=0xBF => 2,                     // cpu→vram rect
            0xC0..=0xDF => 2,                     // vram→cpu rect
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
                        self.vram[(yy as usize * VRAM_W + xx as usize)
                            & (VRAM_W * VRAM_H - 1)] = c;
                    }
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
                        tmp.push(self.vram
                            [((sy + yy) * VRAM_W + sx + xx)
                                & (VRAM_W * VRAM_H - 1)]);
                    }
                }
                for yy in 0..h {
                    for xx in 0..w {
                        self.vram
                            [((dy + yy) * VRAM_W + dx + xx)
                                & (VRAM_W * VRAM_H - 1)] = tmp[yy * w + xx];
                    }
                }
            }
            _ => {}
        }
    }

    // ---- rasterizer --------------------------------------------------

    /// Vertex after draw-offset, plus shading + texcoord.
    fn vtx(&self, w: u32, c: u32) -> Vert {
        Vert {
            x: (((w << 21) as i32) >> 21) + self.off_x,
            y: (((w >> 16) << 21) as i32 >> 21) + self.off_y,
            r: (c & 0xFF) as i32,
            g: ((c >> 8) & 0xFF) as i32,
            b: ((c >> 16) & 0xFF) as i32,
            u: 0,
            v: 0,
        }
    }

    /// Polygons 0x20-0x3F: tri/quad × flat/gouraud × untex/textured.
    fn finish_poly(&mut self, cmd: u32) {
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
                if k == 0 { col0 } else { let w = p[i]; i += 1; w & 0xFF_FFFF }
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
                if k == 0 { clut = uv >> 16; }
                if k == 1 { tpage = uv >> 16; }
            }
        }
        if textured {
            // a textured polygon's tpage also updates the draw mode
            self.regs[1] = (self.regs[1] & !0x9FF) | (tpage & 0x9FF);
        }
        let sh = Shade { textured, gouraud, semitrans: st, raw, tpage, clut };
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
                if idx >= words.len() || words[idx] & 0xF000_F000 == 0x5000_5000 { break; }
            }
            let v = self.vtx(words[idx], col);
            idx += 1;
            if let Some(p) = prev {
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
        let sh = Shade { textured, gouraud: false, semitrans: st, raw, tpage: self.regs[1], clut };
        // E1 bits12/13: textured-rect x/y flip
        let xflip = self.regs[1] & (1 << 12) != 0;
        let yflip = self.regs[1] & (1 << 13) != 0;
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
        let edge = |a: &Vert, b: &Vert, x: i32, y: i32| -> i64 {
            (b.x - a.x) as i64 * (y - a.y) as i64 - (b.y - a.y) as i64 * (x - a.x) as i64
        };
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
        if maxx - minx >= 1024 || maxy - miny >= 512 {
            return;
        }
        let (cx0, cy0, cx1, cy1) = self.clip();
        let (x0, x1) = (minx.max(cx0), (maxx - 1).min(cx1));
        let (y0, y1) = (miny.max(cy0), (maxy - 1).min(cy1));
        if x0 > x1 || y0 > y1 {
            return;
        }
        // with positive area in this y-down space, an edge a->b is a
        // "top" edge when horizontal and heading -x, "left" when
        // heading -y; those own their boundary pixels
        let bias = |a: &Vert, b: &Vert| -> i64 {
            if (a.y == b.y && b.x < a.x) || b.y < a.y { 0 } else { -1 }
        };
        let (b0, b1, b2) = (bias(&v[1], &v[2]), bias(&v[2], &v[0]), bias(&v[0], &v[1]));
        let dither = self.regs[1] & (1 << 9) != 0 && (sh.gouraud || (sh.textured && !sh.raw));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let w0 = edge(&v[1], &v[2], x, y);
                let w1 = edge(&v[2], &v[0], x, y);
                let w2 = edge(&v[0], &v[1], x, y);
                if w0 + b0 < 0 || w1 + b1 < 0 || w2 + b2 < 0 {
                    continue;
                }
                let lerp = |a: i32, b: i32, c: i32| -> i32 {
                    ((a as i64 * w0 + b as i64 * w1 + c as i64 * w2) / area) as i32
                };
                let (r, g, b) = if sh.gouraud {
                    (lerp(v[0].r, v[1].r, v[2].r), lerp(v[0].g, v[1].g, v[2].g), lerp(v[0].b, v[1].b, v[2].b))
                } else {
                    (v[0].r, v[0].g, v[0].b)
                };
                let c = if sh.textured {
                    let u = lerp(v[0].u, v[1].u, v[2].u);
                    let tv = lerp(v[0].v, v[1].v, v[2].v);
                    match self.texel(u, tv, sh.tpage, sh.clut) {
                        Some((t, semi)) => {
                            let m = if sh.raw {
                                t
                            } else {
                                let col = (r.clamp(0, 255) as u32)
                                    | ((g.clamp(0, 255) as u32) << 8)
                                    | ((b.clamp(0, 255) as u32) << 16);
                                modulate_d(t, col, dither, x, y)
                            };
                            if sh.semitrans && semi {
                                self.blend(x, y, m, sh.tpage)
                            } else {
                                m | if semi { 0x8000 } else { 0 }
                            }
                        }
                        None => continue,
                    }
                } else {
                    let c15 = shade15(r, g, b, dither, x, y);
                    if sh.semitrans { self.blend(x, y, c15, self.regs[1]) } else { c15 }
                };
                self.put(x, y, c);
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
            let c15 = ((t(a.b, b.b) >> 3) << 10)
                | ((t(a.g, b.g) >> 3) << 5)
                | (t(a.r, b.r) >> 3);
            if x >= cx0 && x <= cx1 && y >= cy0 && y <= cy1 {
                let c = if st { self.blend(x, y, c15, self.regs[1]) } else { c15 };
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
        // Texture window: (coord & !(mask*8)) | ((offset & mask)*8)
        let wx = (self.texwin & 0x1F) as i32;
        let wy = ((self.texwin >> 5) & 0x1F) as i32;
        let ox = ((self.texwin >> 10) & 0x1F) as i32;
        let oy = ((self.texwin >> 15) & 0x1F) as i32;
        let u = (u & !(wx * 8)) | ((ox & wx) * 8);
        let v = (v & !(wy * 8)) | ((oy & wy) * 8);
        let tx = ((tpage & 0xF) * 64) as usize;
        let ty = if tpage & 0x10 != 0 { 256usize } else { 0 };
        let (u, v) = ((u & 0xFF) as usize, (v & 0xFF) as usize);
        let depth = (tpage >> 7) & 3;
        let c = match depth {
            0 | 1 => {
                let (shift, cx, cy) = if depth == 0 {
                    ((u & 3) * 4, ((clut & 0x3F) * 16) as usize,
                     ((clut >> 6) & 0x1FF) as usize)
                } else {
                    ((u & 1) * 8, ((clut & 0x3F) * 16) as usize,
                     ((clut >> 6) & 0x1FF) as usize)
                };
                let w = self.vram[((ty + v) & (VRAM_H - 1)) * VRAM_W
                    + ((tx + u / if depth == 0 { 4 } else { 2 }) & (VRAM_W - 1))];
                let idx = ((w >> shift) & if depth == 0 { 0xF } else { 0xFF }) as usize;
                self.vram[(cy * VRAM_W + cx + idx) & (VRAM_W * VRAM_H - 1)]
            }
            _ => {
                self.vram[((ty + v) & (VRAM_H - 1)) * VRAM_W + ((tx + u) & (VRAM_W - 1))]
            }
        };
        if c == 0 {
            None
        } else {
            Some((c & 0x7FFF, c & 0x8000 != 0))
        }
    }

    /// Semi-transparency blend against the framebuffer pixel.
    /// Modes (tpage bits5-6): 0=B/2+F/2 1=B+F 2=B-F 3=B+F/4.
    fn blend(&self, x: i32, y: i32, src: u16, tpage: u32) -> u16 {
        let dst = self.vram[(y as usize * VRAM_W + x as usize)
            & (VRAM_W * VRAM_H - 1)];
        let mode = (tpage >> 5) & 3;
        let (sr, sg, sb) = ((src & 0x1F) as i32, ((src >> 5) & 0x1F) as i32, ((src >> 10) & 0x1F) as i32);
        let (dr, dg, db) = ((dst & 0x1F) as i32, ((dst >> 5) & 0x1F) as i32, ((dst >> 10) & 0x1F) as i32);
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

    /// Clip rect from draw-area env (E3 top-left, E4 bottom-right).
    fn clip(&self) -> (i32, i32, i32, i32) {
        let e3 = self.regs[3];
        let e4 = self.regs[4];
        (
            (e3 & 0x3FF) as i32,
            ((e3 >> 10) & 0x1FF) as i32,
            (e4 & 0x3FF) as i32,
            ((e4 >> 10) & 0x1FF) as i32,
        )
    }

    fn put(&mut self, x: i32, y: i32, mut c: u16) {
        if x < 0 || x >= VRAM_W as i32 || y < 0 || y >= VRAM_H as i32 {
            return;
        }
        let i = y as usize * VRAM_W + x as usize;
        // E6.1: don't draw where the mask bit is already set
        if self.mask_check && self.vram[i] & 0x8000 != 0 {
            return;
        }
        // E6.0: force mask bit on drawn pixels
        if self.mask_set {
            c |= 0x8000;
        }
        self.vram[i] = c;
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
            }
            0x01 => {
                // clear command FIFO
                self.gp0_len = 0;
                self.poly_words.clear();
                self.vram_wr = None;
            }
            0x02 => self.irq1 = false,          // ack irq
            0x03 => self.disp_disabled = v & 1 != 0, // display enable
            0x04 => self.dma_dir = (v & 3) as u8,   // dma direction
            0x05 => {                           // display area start
                self.disp_x = (v & 0x3FF) as u16;
                self.disp_y = ((v >> 10) & 0x1FF) as u16;
            }
            0x06 => {                           // h display range
                self.h_range = ((v & 0xFFF) as u16, ((v >> 12) & 0xFFF) as u16);
            }
            0x07 => {                           // v display range
                self.v_range = ((v & 0x3FF) as u16, ((v >> 10) & 0x3FF) as u16);
            }
            0x08 => self.disp_mode = v & 0xFF,  // display mode
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
        let vres = if self.disp_mode & 0x24 == 0x24 { 480 } else { 240 };
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
        let w = w as usize;
        // visible line count from the vertical display range
        let lines = (self.v_range.1 as i32 - self.v_range.0 as i32).clamp(16, 288) as usize;
        let h = if h480 == 480 { lines * 2 } else { lines };
        out.clear();
        out.resize(w * h, 0);
        if self.disp_disabled {
            return (w, h);
        }
        let (dx, dy) = (self.disp_x as usize, self.disp_y as usize);
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
