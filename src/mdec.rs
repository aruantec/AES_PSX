//! MDEC — macroblock decoder used for FMV.
//!
//!   0x1F801820 W: command/parameter FIFO   R: data-out FIFO
//!   0x1F801824 W: control                  R: status
//!
//! Commands (bits 29-31):
//!   1 decode: bits 27-28 depth (0=4bpp 1=8bpp 2=24bpp 3=15bpp),
//!     26 signed output, 25 set bit15, 0-15 parameter word count
//!   2 quant tables (bit0: +chroma table) — 16 or 32 words
//!   3 IDCT scale matrix — 32 words (64 signed halfwords)
//!
//! Input is a stream of RLE-coded 8x8 blocks (Cr, Cb, Y1..Y4 per
//! colour macroblock, a single Y block in monochrome modes). Each
//! block goes through dequantization, a separable 8x8 IDCT with the
//! uploaded matrix, then YCbCr→RGB into the output FIFO.

use std::collections::VecDeque;

/// Scan position k → matrix index (standard 8x8 zig-zag order).
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Clone, Copy, PartialEq)]
enum State {
    Idle,
    Decode,
    QuantTable,
    ScaleTable,
}

pub struct Mdec {
    state: State,
    /// Remaining parameter words for the current command.
    remaining: u32,
    cmd: u32,
    halfwords: VecDeque<u16>,
    table_buf: Vec<u32>,
    quant_y: [u8; 64],
    quant_c: [u8; 64],
    scale: [i16; 64],
    out: VecDeque<u32>,
    dma_in: bool,
    dma_out: bool,
    /// Block currently being assembled (for status bits 16-18).
    block_idx: u32,
}

impl Mdec {
    pub fn new() -> Self {
        Self {
            state: State::Idle,
            remaining: 0,
            cmd: 0,
            halfwords: VecDeque::new(),
            table_buf: Vec::new(),
            quant_y: [1; 64],
            quant_c: [1; 64],
            scale: [0; 64],
            out: VecDeque::new(),
            dma_in: false,
            dma_out: false,
            block_idx: 4,
        }
    }

    fn depth(&self) -> u32 {
        (self.cmd >> 27) & 3
    }

    pub fn read_stat(&self) -> u32 {
        let mut s = 0u32;
        if self.out.is_empty() {
            s |= 1 << 31;
        }
        if self.state != State::Idle {
            s |= 1 << 29;
        }
        if self.dma_in && self.state != State::Idle {
            s |= 1 << 28;
        }
        if self.dma_out && !self.out.is_empty() {
            s |= 1 << 27;
        }
        s |= ((self.cmd >> 25) & 0xF) << 23;
        s |= (self.block_idx & 7) << 16;
        s |= self.remaining.wrapping_sub(1) & 0xFFFF;
        s
    }

    pub fn write_ctrl(&mut self, v: u32) {
        if v & (1 << 31) != 0 {
            self.state = State::Idle;
            self.remaining = 0;
            self.cmd = 0;
            self.halfwords.clear();
            self.out.clear();
            self.block_idx = 4;
        }
        self.dma_in = v & (1 << 30) != 0;
        self.dma_out = v & (1 << 29) != 0;
    }

    pub fn read_data(&mut self) -> u32 {
        self.out.pop_front().unwrap_or(0)
    }

    pub fn out_len(&self) -> usize {
        self.out.len()
    }

    pub fn write_cmd(&mut self, v: u32) {
        if self.state != State::Idle && self.remaining > 0 {
            self.param(v);
            return;
        }
        self.cmd = v;
        match v >> 29 {
            1 => {
                self.state = State::Decode;
                self.remaining = v & 0xFFFF;
                self.halfwords.clear();
            }
            2 => {
                self.state = State::QuantTable;
                self.remaining = if v & 1 != 0 { 32 } else { 16 };
                self.table_buf.clear();
            }
            3 => {
                self.state = State::ScaleTable;
                self.remaining = 32;
                self.table_buf.clear();
            }
            _ => {
                self.state = State::Idle;
                self.remaining = 0;
            }
        }
        if self.remaining == 0 {
            self.state = State::Idle;
        }
    }

    fn param(&mut self, v: u32) {
        self.remaining -= 1;
        match self.state {
            State::Decode => {
                self.halfwords.push_back(v as u16);
                self.halfwords.push_back((v >> 16) as u16);
                self.decode_ready();
            }
            State::QuantTable | State::ScaleTable => self.table_buf.push(v),
            State::Idle => {}
        }
        if self.remaining == 0 {
            match self.state {
                State::QuantTable => {
                    let bytes: Vec<u8> = self
                        .table_buf
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect();
                    self.quant_y.copy_from_slice(&bytes[..64]);
                    if bytes.len() >= 128 {
                        self.quant_c.copy_from_slice(&bytes[64..128]);
                    }
                }
                State::ScaleTable => {
                    for (i, w) in self.table_buf.iter().enumerate() {
                        self.scale[i * 2] = *w as u16 as i16;
                        self.scale[i * 2 + 1] = (*w >> 16) as u16 as i16;
                    }
                }
                _ => {}
            }
            self.state = State::Idle;
        }
    }

    /// Try to pull one RLE block from the input queue. Returns None
    /// when the block isn't complete yet (input left untouched).
    fn take_block(&mut self, quant: &[u8; 64]) -> Option<[i32; 64]> {
        let mut i = 0usize;
        let hw = &self.halfwords;
        while i < hw.len() && hw[i] == 0xFE00 {
            i += 1;
        }
        if i >= hw.len() {
            return None;
        }
        // scan for the end of this block first
        let mut k = 0usize;
        let mut j = i + 1;
        loop {
            if j >= hw.len() {
                return None;
            }
            k += ((hw[j] >> 10) & 0x3F) as usize + 1;
            j += 1;
            if k > 63 {
                break;
            }
        }
        let sext = |n: u16| ((n as i32) << 22) >> 22;
        let mut blk = [0i32; 64];
        let first = hw[i];
        let q = ((first >> 10) & 0x3F) as i32;
        let mut val = sext(first & 0x3FF) * quant[0] as i32;
        let mut k = 0usize;
        let mut p = i + 1;
        loop {
            if q == 0 {
                val = sext(hw[p - 1] & 0x3FF) * 2;
            }
            let v = val.clamp(-0x400, 0x3FF);
            blk[if q > 0 { ZIGZAG[k] } else { k }] = v;
            let n = hw[p];
            p += 1;
            k += ((n >> 10) & 0x3F) as usize + 1;
            if k > 63 {
                break;
            }
            val = (sext(n & 0x3FF) * quant[k] as i32 * q + 4) / 8;
        }
        self.halfwords.drain(..p);
        self.idct(&mut blk);
        Some(blk)
    }

    fn idct(&self, blk: &mut [i32; 64]) {
        let mut tmp = [0i64; 64];
        // rows: tmp = blk · S  (S indexed [z*8 + x])
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0i64;
                for z in 0..8 {
                    sum += blk[y + z * 8] as i64 * self.scale[x + z * 8] as i64;
                }
                tmp[x + y * 8] = sum;
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0i64;
                for z in 0..8 {
                    sum += tmp[y + z * 8] * self.scale[x + z * 8] as i64;
                }
                // each pass scales by 2^-16 (matrix is 2x orthonormal
                // in 1.15 fixed point)
                blk[x + y * 8] = ((sum + (1 << 31)) >> 32) as i32;
            }
        }
    }

    fn decode_ready(&mut self) {
        loop {
            let ok = match self.depth() {
                0 | 1 => self.mono_block(),
                _ => self.color_block(),
            };
            if !ok {
                break;
            }
        }
    }

    fn mono_block(&mut self) -> bool {
        let q = self.quant_y;
        let Some(y) = self.take_block(&q) else {
            return false;
        };
        let signed = self.cmd & (1 << 26) != 0;
        let px: Vec<u8> = y
            .iter()
            .map(|&v| {
                let v = v.clamp(-128, 127);
                (if signed { v } else { v + 128 }) as u8
            })
            .collect();
        if self.depth() == 1 {
            for c in px.chunks_exact(4) {
                self.out
                    .push_back(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        } else {
            for c in px.chunks_exact(8) {
                let mut w = 0u32;
                for (i, b) in c.iter().enumerate() {
                    w |= ((*b >> 4) as u32) << (i * 4);
                }
                self.out.push_back(w);
            }
        }
        true
    }

    fn color_block(&mut self) -> bool {
        // need Cr, Cb, Y1..Y4 = 6 complete blocks before emitting;
        // decode them one at a time into a staging area.
        let snapshot: Vec<u16> = self.halfwords.iter().copied().collect();
        let (qy, qc) = (self.quant_y, self.quant_c);
        let mut blocks = Vec::with_capacity(6);
        for b in 0..6 {
            self.block_idx = [4, 5, 0, 1, 2, 3][b];
            match self.take_block(if b < 2 { &qc } else { &qy }) {
                Some(x) => blocks.push(x),
                None => {
                    self.halfwords = snapshot.into();
                    return false;
                }
            }
        }
        let (cr, cb) = (&blocks[0], &blocks[1]);
        let signed = self.cmd & (1 << 26) != 0;
        let bit15 = if self.cmd & (1 << 25) != 0 {
            0x8000u16
        } else {
            0
        };
        let mut rgb = [[0u8; 3]; 256];
        for (bi, yb) in blocks[2..].iter().enumerate() {
            let (bx, by) = ((bi & 1) * 8, (bi >> 1) * 8);
            for y in 0..8 {
                for x in 0..8 {
                    let ci = ((by + y) / 2) * 8 + (bx + x) / 2;
                    let (r_, b_) = (cr[ci] as f32, cb[ci] as f32);
                    let r = 1.402 * r_;
                    let g = -0.3437 * b_ - 0.7143 * r_;
                    let b = 1.772 * b_;
                    let l = yb[y * 8 + x] as f32;
                    let cv = |c: f32| {
                        let v = (l + c).round().clamp(-128.0, 127.0) as i32;
                        (if signed { v } else { v + 128 }) as u8
                    };
                    rgb[(by + y) * 16 + bx + x] = [cv(r), cv(g), cv(b)];
                }
            }
        }
        if self.depth() == 2 {
            let bytes: Vec<u8> = rgb.iter().flatten().copied().collect();
            for c in bytes.chunks_exact(4) {
                self.out
                    .push_back(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        } else {
            let px: Vec<u16> = rgb
                .iter()
                .map(|p| {
                    let c = |v: u8| {
                        ((if signed {
                            (v as i8 as i32 + 128) as u8
                        } else {
                            v
                        }) >> 3) as u16
                    };
                    c(p[0]) | c(p[1]) << 5 | c(p[2]) << 10 | bit15
                })
                .collect();
            for c in px.chunks_exact(2) {
                self.out.push_back(c[0] as u32 | (c[1] as u32) << 16);
            }
        }
        self.block_idx = 4;
        true
    }
}
