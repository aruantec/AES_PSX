//! SPU — 24 ADPCM voices, ADSR envelopes, stereo mix at 44.1 kHz.
//!
//! Register index = (addr - 0x1F801C00) / 2:
//!   voice v at v*8: +0 volL +1 volR +2 pitch +3 start +4 ADSR lo
//!                   +5 ADSR hi +6 envelope level +7 repeat
//!   0xC0/C1 main vol      0xC2/C3 reverb out vol
//!   0xC4/C5 KON           0xC6/C7 KOFF
//!   0xC8/C9 PMON          0xCA/CB NON (noise)
//!   0xCC/CD EON (reverb)  0xCE/CF ENDX (r)
//!   0xD1 reverb base      0xD2 IRQ address
//!   0xD3 transfer address 0xD4 transfer FIFO
//!   0xD5 SPUCNT           0xD6 transfer control
//!   0xD7 SPUSTAT          0xD8/D9 CD input volume
//!   0xDA/DB ext volume    0xDC/DD current main volume
//!   0xE0-0xFF reverb configuration
//!
//! Reverb is implemented with the documented work-area algorithm at
//! 22.05 kHz.

use crate::cdrom::CdAudio;

const RAM_SIZE: usize = 512 * 1024;
const RAM_MASK: u32 = RAM_SIZE as u32 - 1;
const CYCLES_PER_SAMPLE: u32 = 768;

const ADPCM_POS: [i32; 5] = [0, 60, 115, 98, 122];
const ADPCM_NEG: [i32; 5] = [0, 0, -52, -55, -60];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Attack,
    Decay,
    Sustain,
    Release,
    Off,
}

/// Generic envelope stepper shared by ADSR and volume sweeps.
#[derive(Clone, Copy, Default)]
struct Ramp {
    wait: u32,
}

impl Ramp {
    /// One 44.1 kHz tick. `rate` = shift<<2|step (7 bits).
    fn step(&mut self, level: i32, exp: bool, dec: bool, shift: u32, step: u32) -> i32 {
        if self.wait > 0 {
            self.wait -= 1;
            return level;
        }
        let mut cycles = 1u32 << shift.saturating_sub(11);
        let base = if dec { -8 + step as i32 } else { 7 - step as i32 };
        let mut delta = base << 11u32.saturating_sub(shift);
        if exp && !dec && level > 0x6000 {
            cycles *= 4;
        }
        if exp && dec {
            delta = delta * level / 0x8000;
        }
        self.wait = cycles - 1;
        (level + delta).clamp(0, 0x7FFF)
    }
}

#[derive(Clone, Copy)]
struct Voice {
    vol: [u16; 2],
    vol_level: [i32; 2],
    vol_ramp: [Ramp; 2],
    pitch: u16,
    start: u32,
    repeat: u32,
    adsr: u32,
    /// Address of the next block to fetch.
    addr: u32,
    samples: [i16; 28],
    /// Last 3 decoded samples of the previous block (interpolation).
    prev: [i16; 3],
    counter: u32,
    hist: [i32; 2],
    flags: u8,
    level: i32,
    phase: Phase,
    ramp: Ramp,
    out: i32,
    /// Repeat address set by software after key-on (wins over the
    /// block's loop-start flag).
    repeat_locked: bool,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            vol: [0; 2],
            vol_level: [0; 2],
            vol_ramp: [Ramp::default(); 2],
            pitch: 0,
            start: 0,
            repeat: 0,
            adsr: 0,
            addr: 0,
            samples: [0; 28],
            prev: [0; 3],
            counter: 0,
            hist: [0; 2],
            flags: 0,
            level: 0,
            phase: Phase::Off,
            ramp: Ramp::default(),
            out: 0,
            repeat_locked: false,
        }
    }
}

pub struct Spu {
    regs: [u16; 512],
    pub ram: Vec<u8>,
    voices: [Voice; 24],
    cnt: u16,
    stat_irq: bool,
    pub pending_irq: bool,
    irq_addr: u32,
    taddr: u32,
    endx: u32,
    kon_pending: u32,
    koff_pending: u32,
    main_vol: [i32; 2],
    cd_vol: [i32; 2],
    noise_level: i32,
    noise_timer: i32,
    capture_pos: u32,
    cyc: u32,
    // reverb
    rev_cur: u32,
    rev_odd: bool,
    rev_in: [i32; 2],
    rev_out: [i32; 2],
    out: Vec<i16>,
}

impl Spu {
    pub fn new() -> Self {
        Self {
            regs: [0; 512],
            ram: vec![0; RAM_SIZE],
            voices: [Voice::default(); 24],
            cnt: 0,
            stat_irq: false,
            pending_irq: false,
            irq_addr: 0,
            taddr: 0,
            endx: 0,
            kon_pending: 0,
            koff_pending: 0,
            main_vol: [0; 2],
            cd_vol: [0; 2],
            noise_level: 1,
            noise_timer: 0,
            capture_pos: 0,
            cyc: 0,
            rev_cur: 0,
            rev_odd: false,
            rev_in: [0; 2],
            rev_out: [0; 2],
            out: Vec::with_capacity(4096),
        }
    }

    /// Drain mixed output (s16 L/R interleaved, 44.1 kHz).
    pub fn take_audio(&mut self) -> Vec<i16> {
        std::mem::take(&mut self.out)
    }

    #[inline]
    fn touch(&mut self, addr: u32) {
        if self.cnt & (1 << 6) != 0 && (addr & RAM_MASK & !7) == self.irq_addr {
            if !self.stat_irq {
                self.stat_irq = true;
                self.pending_irq = true;
            }
        }
    }

    fn ram16(&self, a: u32) -> i16 {
        let a = (a & RAM_MASK & !1) as usize;
        i16::from_le_bytes([self.ram[a], self.ram[a + 1]])
    }

    fn put16(&mut self, a: u32, v: i16) {
        let a = (a & RAM_MASK & !1) as usize;
        self.ram[a..a + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn decode_block(&mut self, v: usize) {
        let a = self.voices[v].addr & RAM_MASK & !0xF;
        self.touch(a);
        self.touch(a + 8);
        let hdr = self.ram[a as usize];
        let flags = self.ram[a as usize + 1];
        let mut shift = (hdr & 0x0F) as u32;
        if shift > 12 {
            shift = 9;
        }
        let f = ((hdr >> 4) & 7).min(4) as usize;
        let vc = &mut self.voices[v];
        vc.prev = [vc.samples[25], vc.samples[26], vc.samples[27]];
        let [mut h1, mut h2] = vc.hist;
        for i in 0..28 {
            let b = self.ram[a as usize + 2 + i / 2];
            let nib = if i & 1 == 0 { b & 0xF } else { b >> 4 };
            let raw = (((nib as i8) << 4) as i32) << 8 >> shift;
            let s = (raw + ((h1 * ADPCM_POS[f] + h2 * ADPCM_NEG[f] + 32) >> 6)).clamp(-0x8000, 0x7FFF);
            vc.samples[i] = s as i16;
            h2 = h1;
            h1 = s;
        }
        vc.hist = [h1, h2];
        vc.flags = flags;
        if flags & 4 != 0 && !vc.repeat_locked {
            vc.repeat = a;
        }
    }

    fn key_on(&mut self, mask: u32) {
        for v in 0..24 {
            if mask & (1 << v) != 0 {
                let vc = &mut self.voices[v];
                vc.addr = vc.start;
                vc.counter = 0;
                vc.hist = [0; 2];
                vc.samples = [0; 28];
                vc.level = 0;
                vc.phase = Phase::Attack;
                vc.ramp = Ramp::default();
                vc.repeat_locked = false;
                self.endx &= !(1 << v);
                self.decode_block(v);
            }
        }
    }

    fn key_off(&mut self, mask: u32) {
        for v in 0..24 {
            if mask & (1 << v) != 0 && self.voices[v].phase != Phase::Off {
                self.voices[v].phase = Phase::Release;
                self.voices[v].ramp = Ramp::default();
            }
        }
    }

    fn envelope(vc: &mut Voice) {
        let lo = vc.adsr & 0xFFFF;
        let hi = vc.adsr >> 16;
        match vc.phase {
            Phase::Attack => {
                let exp = lo & 0x8000 != 0;
                vc.level = vc.ramp.step(vc.level, exp, false, (lo >> 10) & 0x1F, (lo >> 8) & 3);
                if vc.level >= 0x7FFF {
                    vc.phase = Phase::Decay;
                    vc.ramp = Ramp::default();
                }
            }
            Phase::Decay => {
                let sus = (((lo & 0xF) + 1) * 0x800) as i32;
                vc.level = vc.ramp.step(vc.level, true, true, (lo >> 4) & 0xF, 0);
                if vc.level <= sus {
                    vc.phase = Phase::Sustain;
                    vc.ramp = Ramp::default();
                }
            }
            Phase::Sustain => {
                let exp = hi & 0x8000 != 0;
                let dec = hi & 0x4000 != 0;
                vc.level = vc.ramp.step(vc.level, exp, dec, (hi >> 8) & 0x1F, (hi >> 6) & 3);
            }
            Phase::Release => {
                let exp = hi & 0x20 != 0;
                vc.level = vc.ramp.step(vc.level, exp, true, hi & 0x1F, 0);
                if vc.level == 0 {
                    vc.phase = Phase::Off;
                }
            }
            Phase::Off => vc.level = 0,
        }
    }

    fn volume(reg: u16, level: &mut i32, ramp: &mut Ramp) -> i32 {
        if reg & 0x8000 == 0 {
            *level = (((reg << 1) as i16) as i32).clamp(-0x8000, 0x7FFF);
            return *level;
        }
        // sweep mode: magnitude ramps, sign from phase bit
        let exp = reg & 0x4000 != 0;
        let dec = reg & 0x2000 != 0;
        let mag = ramp.step(level.abs(), exp, dec, ((reg >> 2) & 0x1F) as u32, (reg & 3) as u32);
        *level = if reg & 0x1000 != 0 { -mag } else { mag };
        *level
    }

    fn sample_voice(&mut self, v: usize, prev_out: i32) -> i32 {
        let noise = (self.regs[0xCA] as u32 | (self.regs[0xCB] as u32) << 16) & (1 << v) != 0;
        let pmon = v > 0 && (self.regs[0xC8] as u32 | (self.regs[0xC9] as u32) << 16) & (1 << v) != 0;
        let vc = &mut self.voices[v];
        if vc.phase == Phase::Off && vc.level == 0 {
            vc.out = 0;
            return 0;
        }
        let idx = (vc.counter >> 12) as usize;
        let frac = ((vc.counter >> 4) & 0xFF) as i32;
        let s = if noise {
            self.noise_level as i16 as i32
        } else {
            let at = |k: isize| -> i32 {
                let i = idx as isize + k;
                if i < 0 { vc.prev[(3 + i) as usize] as i32 } else { vc.samples[i as usize] as i32 }
            };
            // 4-point cubic (Catmull-Rom) across samples idx-3..idx
            let (p0, p1, p2, p3) = (at(-3), at(-2), at(-1), at(0));
            let t = frac;
            let a = -p0 + 3 * p1 - 3 * p2 + p3;
            let b = 2 * p0 - 5 * p1 + 4 * p2 - p3;
            let c = -p0 + p2;
            let r = ((((a * t >> 8) + b) * t >> 8) + c) * t >> 8;
            ((r >> 1) + p1).clamp(-0x8000, 0x7FFF)
        };
        Self::envelope(vc);
        let out = s * vc.level >> 15;
        vc.out = out;
        // advance pitch
        let mut step = vc.pitch as i32;
        if pmon {
            step = (step * (prev_out + 0x8000)) >> 15;
        }
        vc.counter += step.clamp(0, 0x3FFF) as u32;
        let mut need = false;
        if vc.counter >= 28 << 12 {
            vc.counter -= 28 << 12;
            need = true;
        }
        if need {
            let flags = vc.flags;
            if flags & 1 != 0 {
                self.endx |= 1 << v;
                let vc = &mut self.voices[v];
                vc.addr = vc.repeat;
                if flags & 2 == 0 && !noise {
                    vc.phase = Phase::Release;
                    vc.level = 0;
                }
            } else {
                self.voices[v].addr = self.voices[v].addr.wrapping_add(16);
            }
            self.decode_block(v);
        }
        out
    }

    fn noise_tick(&mut self) {
        let step = 4 + ((self.cnt >> 8) & 3) as i32;
        let shift = ((self.cnt >> 10) & 0xF) as u32;
        self.noise_timer -= step;
        let l = self.noise_level;
        let parity = ((l >> 15) ^ (l >> 12) ^ (l >> 11) ^ (l >> 10) ^ 1) & 1;
        if self.noise_timer < 0 {
            self.noise_level = ((l << 1) | parity) & 0xFFFF;
            self.noise_timer += 0x20000 >> shift;
            if self.noise_timer < 0 {
                self.noise_timer += 0x20000 >> shift;
            }
        }
    }

    fn tick_sample(&mut self, cd: &mut CdAudio) {
        if self.kon_pending != 0 {
            let m = std::mem::take(&mut self.kon_pending);
            self.key_on(m);
        }
        if self.koff_pending != 0 {
            let m = std::mem::take(&mut self.koff_pending);
            self.key_off(m);
        }
        self.noise_tick();
        let eon = self.regs[0xCC] as u32 | (self.regs[0xCD] as u32) << 16;
        let (mut l, mut r) = (0i32, 0i32);
        let (mut rl, mut rr) = (0i32, 0i32);
        let mut prev = 0;
        for v in 0..24 {
            let s = self.sample_voice(v, prev);
            prev = s;
            let vc = &mut self.voices[v];
            let vl = Self::volume(vc.vol[0], &mut vc.vol_level[0], &mut vc.vol_ramp[0]);
            let vr = Self::volume(vc.vol[1], &mut vc.vol_level[1], &mut vc.vol_ramp[1]);
            let (sl, sr) = (s * vl >> 15, s * vr >> 15);
            l += sl;
            r += sr;
            if eon & (1 << v) != 0 {
                rl += sl;
                rr += sr;
            }
        }
        // CD audio input
        let (cl, cr) = cd.pop();
        let cpos = self.capture_pos;
        self.put16(cpos * 2, cl as i16);
        self.put16(0x400 + cpos * 2, cr as i16);
        let v1 = self.voices[1].out as i16;
        let v3 = self.voices[3].out as i16;
        self.put16(0x800 + cpos * 2, v1);
        self.put16(0xC00 + cpos * 2, v3);
        for a in [cpos * 2, 0x400 + cpos * 2, 0x800 + cpos * 2, 0xC00 + cpos * 2] {
            self.touch(a);
        }
        self.capture_pos = (cpos + 1) & 0x1FF;
        if self.cnt & 1 != 0 {
            let (cl, cr) = (cl * self.cd_vol[0] >> 15, cr * self.cd_vol[1] >> 15);
            l += cl;
            r += cr;
            if self.cnt & 4 != 0 {
                rl += cl;
                rr += cr;
            }
        }
        if self.cnt & 0x80 != 0 {
            let (ol, or) = self.reverb(rl.clamp(-0x8000, 0x7FFF), rr.clamp(-0x8000, 0x7FFF));
            l += ol;
            r += or;
        }
        let mute = self.cnt & (1 << 14) == 0 || self.cnt & 0x8000 == 0;
        let ol = if mute { 0 } else { (l.clamp(-0x8000, 0x7FFF) * self.main_vol[0]) >> 15 };
        let or = if mute { 0 } else { (r.clamp(-0x8000, 0x7FFF) * self.main_vol[1]) >> 15 };
        if self.out.len() < 44_100 {
            self.out.push(ol.clamp(-0x8000, 0x7FFF) as i16);
            self.out.push(or.clamp(-0x8000, 0x7FFF) as i16);
        }
    }

    // ---- reverb (22.05 kHz, documented work-area algorithm) ----

    fn rv(&self, i: usize) -> i32 {
        self.regs[0xE0 + i] as i16 as i32
    }
    fn rva(&self, i: usize) -> i32 {
        self.regs[0xE0 + i] as i32 * 8
    }

    fn rev_addr(&self, a: i32) -> u32 {
        let base = self.regs[0xD1] as i32 * 8;
        let size = (RAM_SIZE as i32 - base).max(2);
        base as u32 + (self.rev_cur as i32 + a).rem_euclid(size) as u32
    }

    fn rget(&self, a: i32) -> i32 {
        self.ram16(self.rev_addr(a)) as i32
    }

    fn rput(&mut self, a: i32, v: i32) {
        let ad = self.rev_addr(a);
        self.put16(ad, v.clamp(-0x8000, 0x7FFF) as i16);
    }

    fn reverb(&mut self, inl: i32, inr: i32) -> (i32, i32) {
        self.rev_odd = !self.rev_odd;
        if self.rev_odd {
            self.rev_in = [inl, inr];
            return (self.rev_out[0], self.rev_out[1]);
        }
        let li = (self.rev_in[0] + inl) / 2;
        let ri = (self.rev_in[1] + inr) / 2;
        // register layout 0x1DC0.. (index from 0xE0)
        let (d_apf1, d_apf2) = (self.rva(0), self.rva(1));
        let (v_iir, v_comb1, v_comb2, v_comb3, v_comb4) = (self.rv(2), self.rv(3), self.rv(4), self.rv(5), self.rv(6));
        let v_wall = self.rv(7);
        let (v_apf1, v_apf2) = (self.rv(8), self.rv(9));
        let (m_lsame, m_rsame) = (self.rva(10), self.rva(11));
        let (m_lcomb1, m_rcomb1, m_lcomb2, m_rcomb2) = (self.rva(12), self.rva(13), self.rva(14), self.rva(15));
        let (d_lsame, d_rsame) = (self.rva(16), self.rva(17));
        let (m_ldiff, m_rdiff) = (self.rva(18), self.rva(19));
        let (m_lcomb3, m_rcomb3, m_lcomb4, m_rcomb4) = (self.rva(20), self.rva(21), self.rva(22), self.rva(23));
        let (d_ldiff, d_rdiff) = (self.rva(24), self.rva(25));
        let (m_lapf1, m_rapf1, m_lapf2, m_rapf2) = (self.rva(26), self.rva(27), self.rva(28), self.rva(29));
        let (v_lin, v_rin) = (self.rv(30), self.rv(31));
        let m = |a: i32, b: i32| (a * b) >> 15;

        let lin = m(li, v_lin);
        let rin = m(ri, v_rin);
        // same-side reflection
        let ls = m(lin + m(self.rget(d_lsame), v_wall) - self.rget(m_lsame- 2), v_iir)
            + self.rget(m_lsame- 2);
        let rs = m(rin + m(self.rget(d_rsame), v_wall) - self.rget(m_rsame- 2), v_iir)
            + self.rget(m_rsame- 2);
        self.rput(m_lsame, ls);
        self.rput(m_rsame, rs);
        // different-side reflection
        let ld = m(lin + m(self.rget(d_rdiff), v_wall) - self.rget(m_ldiff- 2), v_iir)
            + self.rget(m_ldiff- 2);
        let rd = m(rin + m(self.rget(d_ldiff), v_wall) - self.rget(m_rdiff- 2), v_iir)
            + self.rget(m_rdiff- 2);
        self.rput(m_ldiff, ld);
        self.rput(m_rdiff, rd);
        // early echo
        let mut lo = m(v_comb1, self.rget(m_lcomb1)) + m(v_comb2, self.rget(m_lcomb2))
            + m(v_comb3, self.rget(m_lcomb3)) + m(v_comb4, self.rget(m_lcomb4));
        let mut ro = m(v_comb1, self.rget(m_rcomb1)) + m(v_comb2, self.rget(m_rcomb2))
            + m(v_comb3, self.rget(m_rcomb3)) + m(v_comb4, self.rget(m_rcomb4));
        // all-pass 1
        lo -= m(v_apf1, self.rget(m_lapf1 - d_apf1));
        self.rput(m_lapf1, lo);
        lo = m(lo, v_apf1) + self.rget(m_lapf1 - d_apf1);
        ro -= m(v_apf1, self.rget(m_rapf1 - d_apf1));
        self.rput(m_rapf1, ro);
        ro = m(ro, v_apf1) + self.rget(m_rapf1 - d_apf1);
        // all-pass 2
        lo -= m(v_apf2, self.rget(m_lapf2 - d_apf2));
        self.rput(m_lapf2, lo);
        lo = m(lo, v_apf2) + self.rget(m_lapf2 - d_apf2);
        ro -= m(v_apf2, self.rget(m_rapf2 - d_apf2));
        self.rput(m_rapf2, ro);
        ro = m(ro, v_apf2) + self.rget(m_rapf2 - d_apf2);

        let out_l = m(lo.clamp(-0x8000, 0x7FFF), self.regs[0xC2] as i16 as i32);
        let out_r = m(ro.clamp(-0x8000, 0x7FFF), self.regs[0xC3] as i16 as i32);
        self.rev_out = [out_l, out_r];
        let base = self.regs[0xD1] as u32 * 8;
        let size = (RAM_SIZE as u32 - base).max(2);
        self.rev_cur = (self.rev_cur + 2) % size;
        (out_l, out_r)
    }

    pub fn tick(&mut self, cycles: u32, cd: &mut CdAudio) {
        self.cyc += cycles;
        while self.cyc >= CYCLES_PER_SAMPLE {
            self.cyc -= CYCLES_PER_SAMPLE;
            self.tick_sample(cd);
        }
    }

    pub fn read16(&mut self, off: u32) -> u16 {
        let i = ((off >> 1) & 0x1FF) as usize;
        match i {
            0..=0xBF if i % 8 == 6 => self.voices[i / 8].level as u16,
            0xCE => self.endx as u16,
            0xCF => (self.endx >> 16) as u16,
            0xD7 => {
                let mut s = self.cnt & 0x3F;
                if self.stat_irq {
                    s |= 1 << 6;
                }
                s |= (self.cnt & 0x20) << 2;
                s |= ((self.capture_pos >= 0x100) as u16) << 11;
                s
            }
            0xDC => self.main_vol[0] as u16,
            0xDD => self.main_vol[1] as u16,
            0x100..=0x12F => {
                let v = (i - 0x100) / 2;
                self.voices[v].vol_level[i & 1] as u16
            }
            _ => self.regs[i],
        }
    }

    pub fn write16(&mut self, off: u32, v: u16) {
        let i = ((off >> 1) & 0x1FF) as usize;
        self.regs[i] = v;
        if i < 0xC0 {
            let vc = &mut self.voices[i / 8];
            match i % 8 {
                0 => vc.vol[0] = v,
                1 => vc.vol[1] = v,
                2 => vc.pitch = v,
                3 => vc.start = (v as u32) * 8,
                4 => vc.adsr = (vc.adsr & 0xFFFF_0000) | v as u32,
                5 => vc.adsr = (vc.adsr & 0xFFFF) | (v as u32) << 16,
                6 => vc.level = v as i16 as i32,
                _ => {
                    vc.repeat = (v as u32) * 8;
                    vc.repeat_locked = true;
                }
            }
            return;
        }
        match i {
            0xC0 => self.main_vol[0] = Self::fixed_vol(v),
            0xC1 => self.main_vol[1] = Self::fixed_vol(v),
            0xC4 => self.kon_pending |= v as u32,
            0xC5 => self.kon_pending |= (v as u32) << 16,
            0xC6 => self.koff_pending |= v as u32,
            0xC7 => self.koff_pending |= (v as u32) << 16,
            0xCE | 0xCF => {}
            0xD1 => self.rev_cur = 0,
            0xD2 => self.irq_addr = (v as u32) * 8,
            0xD3 => {
                self.taddr = (v as u32) * 8;
            }
            0xD4 => {
                let a = self.taddr;
                self.touch(a);
                self.put16(a, v as i16);
                self.taddr = (a + 2) & RAM_MASK;
            }
            0xD5 => {
                self.cnt = v;
                if v & (1 << 6) == 0 {
                    self.stat_irq = false;
                }
            }
            0xD8 => self.cd_vol[0] = v as i16 as i32,
            0xD9 => self.cd_vol[1] = v as i16 as i32,
            _ => {}
        }
    }

    fn fixed_vol(v: u16) -> i32 {
        if v & 0x8000 != 0 {
            0x7FFF
        } else {
            ((v << 1) as i16) as i32
        }
    }

    pub fn dma_write(&mut self, w: u32) {
        let a = self.taddr;
        self.touch(a);
        self.put16(a, w as i16);
        self.put16(a + 2, (w >> 16) as i16);
        self.taddr = (a + 4) & RAM_MASK;
    }

    pub fn dma_read(&mut self) -> u32 {
        let a = self.taddr;
        self.touch(a);
        let v = self.ram16(a) as u16 as u32 | (self.ram16(a + 2) as u16 as u32) << 16;
        self.taddr = (a + 4) & RAM_MASK;
        v
    }
}
