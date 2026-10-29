//! GTE (cop2) — Geometry Transformation Engine.
//!
//! Full implementation from the psx-spx GTE chapter:
//!   - data regs 0-31 (vertices, FIFOs, accumulators) and control
//!     regs 0-31 (matrices, translations, projection, FLAG)
//!   - all commands: RTPS/RTPT, NCLIP, AVSZ3/4, MVMVA, SQR, OP,
//!     DPCS/DPCT/DCPL/INTPL, NCS/NCT/NCCS/NCCT/NCDS/NCDT/CC/CDP,
//!     GPF/GPL
//!   - 44-bit MAC1-3 accumulators with per-addition overflow flags,
//!     IR saturation via `lm`, SZ/SXY/OTZ/IR0/RGB-FIFO saturation,
//!     hardware UNR division with the documented inaccuracy
//!
//! Layout facts (hardware spec, not implementation choices):
//!   data: 0-5 VXYn/VZn, 6 RGBC, 7 OTZ, 8 IR0, 9-11 IR1-3,
//!         12-14 SXY0-2, 15 SXYP, 16-19 SZ0-3, 20-22 RGB0-2, 23 RES1,
//!         24 MAC0, 25-27 MAC1-3, 28 IRGB, 29 ORGB, 30 LZCS, 31 LZCR
//!   ctrl: 0-4 RT, 5-7 TR, 8-12 LLM, 13-15 BK, 16-20 LCM, 21-23 FC,
//!         24-25 OFX/OFY, 26 H, 27 DQA, 28 DQB, 29-30 ZSF3/4, 31 FLAG

const FLAG_MAC_POS: [u8; 3] = [30, 29, 28]; // MAC1..3 positive 44-bit
const FLAG_MAC_NEG: [u8; 3] = [27, 26, 25]; // MAC1..3 negative 44-bit
const MAC44_POS: i64 = (1 << 43) - 1;
const MAC44_NEG: i64 = -(1 << 43);

/// UNR division table — generated from the documented hardware
/// formula `unr_table[i] = max(0, (0x40000/(i+0x100)+1)/2 - 0x101)`.
const fn build_unr() -> [i32; 0x101] {
    let mut t = [0i32; 0x101];
    let mut i = 0;
    while i <= 0x100 {
        let v = ((0x40000 / (i as i64 + 0x100) + 1) / 2 - 0x101) as i32;
        t[i] = if v < 0 { 0 } else { v };
        i += 1;
    }
    t
}
static UNR_TABLE: [i32; 0x101] = build_unr();

pub struct Gte {
    /// Data registers 0-31 as raw 32-bit values.
    pub data: [u32; 32],
    /// Control registers 0-31 (FLAG at 31 stores bits 30-12; bit31 is
    /// computed on read).
    pub ctrl: [u32; 32],
    /// MAC0-3 shadows — MAC1-3 are 44-bit internally; data[24..27]
    /// hold only the low 32 bits. Public for snapshots.
    pub mac: [i64; 4],
    /// Working flag accumulator during a command.
    flag: u32,
}

impl Gte {
    pub fn new() -> Self {
        Self {
            data: [0; 32],
            ctrl: [0; 32],
            mac: [0; 4],
            flag: 0,
        }
    }

    // ---------------- register accessors ----------------

    #[inline]
    fn sx(&self, i: usize) -> i64 {
        (self.data[12 + i] & 0xFFFF) as u16 as i16 as i64
    }
    #[inline]
    fn sy(&self, i: usize) -> i64 {
        (self.data[12 + i] >> 16) as u16 as i16 as i64
    }
    #[inline]
    fn sz(&self, i: usize) -> i64 {
        (self.data[16 + i] & 0xFFFF) as i64
    }
    #[inline]
    fn ir(&self, i: usize) -> i64 {
        // i = 0..3 → IR0..IR3 at data[8..11]
        (self.data[8 + i] & 0xFFFF) as u16 as i16 as i64
    }
    #[inline]
    fn set_ir(&mut self, i: usize, v: i64) {
        self.data[8 + i] = (v as i32 as u16) as u32;
    }
    /// Vertex component: vi = 0..2 (V0..V2), comp = 0..2 (X,Y,Z).
    #[inline]
    fn v(&self, vi: usize, comp: usize) -> i64 {
        let r = vi * 2 + comp / 2;
        let sh = (comp & 1) * 16;
        ((self.data[r] >> sh) & 0xFFFF) as u16 as i16 as i64
    }
    /// Matrix element: base = 0 (RT), 8 (LLM), 16 (LCM); r,c = 0..2.
    fn mat(&self, base: usize, r: usize, c: usize) -> i64 {
        let i = base + (r * 3 + c) / 2;
        let sh = ((r * 3 + c) & 1) * 16;
        ((self.ctrl[i] >> sh) & 0xFFFF) as u16 as i16 as i64
    }
    #[inline]
    fn tr(&self, i: usize) -> i64 {
        self.ctrl[5 + i] as i32 as i64
    }
    #[inline]
    fn bk(&self, i: usize) -> i64 {
        self.ctrl[13 + i] as i32 as i64
    }
    #[inline]
    fn fc(&self, i: usize) -> i64 {
        self.ctrl[21 + i] as i32 as i64
    }
    #[inline]
    fn ofx(&self) -> i64 {
        self.ctrl[24] as i32 as i64
    }
    #[inline]
    fn ofy(&self) -> i64 {
        self.ctrl[25] as i32 as i64
    }
    #[inline]
    fn h(&self) -> i64 {
        (self.ctrl[26] & 0xFFFF) as i64
    }
    #[inline]
    fn dqa(&self) -> i64 {
        (self.ctrl[27] & 0xFFFF) as u16 as i16 as i64
    }
    #[inline]
    fn dqb(&self) -> i64 {
        self.ctrl[28] as i32 as i64
    }

    // ---------------- flag/saturation helpers ----------------

    /// 44-bit accumulator check for MAC1 (i=0), MAC2, MAC3.
    #[inline]
    fn a(&mut self, i: usize, v: i64) -> i64 {
        if v > MAC44_POS {
            self.flag |= 1 << FLAG_MAC_POS[i];
            MAC44_POS
        } else if v < MAC44_NEG {
            self.flag |= 1 << FLAG_MAC_NEG[i];
            MAC44_NEG
        } else {
            v
        }
    }

    /// MAC0 32-bit overflow check; returns the stored low-32 value.
    fn set_mac0(&mut self, v: i64) {
        if v > 0x7FFF_FFFF {
            self.flag |= 1 << 16;
        } else if v < -0x8000_0000 {
            self.flag |= 1 << 15;
        }
        self.mac[0] = v as i32 as i64;
        self.data[24] = v as i32 as u32;
    }

    /// IR saturation; i = 1..3 → flag bits 24/23/22.
    fn lim_ir(&mut self, i: usize, v: i64, lm: bool) -> i64 {
        let lo = if lm { 0 } else { -0x8000 };
        if v < lo {
            self.flag |= 1 << (25 - i);
            lo
        } else if v > 0x7FFF {
            self.flag |= 1 << (25 - i);
            0x7FFF
        } else {
            v
        }
    }

    /// Unsigned 16-bit saturation for SZ3/OTZ (flag bit 18).
    fn lim_u16(&mut self, v: i64) -> i64 {
        if !(0..=0xFFFF).contains(&v) {
            self.flag |= 1 << 18;
            v.clamp(0, 0xFFFF)
        } else {
            v
        }
    }

    /// Screen coordinate saturation −400h..+3FFh (flags 14/13).
    fn lim_sxy(&mut self, bit: u8, v: i64) -> i64 {
        if !(-0x400..=0x3FF).contains(&v) {
            self.flag |= 1 << bit;
            v.clamp(-0x400, 0x3FF)
        } else {
            v
        }
    }

    /// IR0 saturation 0..+1000h (flag 12).
    fn lim_ir0(&mut self, v: i64) -> i64 {
        if !(0..=0x1000).contains(&v) {
            self.flag |= 1 << 12;
            v.clamp(0, 0x1000)
        } else {
            v
        }
    }

    /// Color FIFO channel saturation to 0..FFh (flags 21/20/19).
    fn lim_col(&mut self, i: usize, v: i64) -> i64 {
        if !(0..=0xFF).contains(&v) {
            self.flag |= 1 << (21 - i);
            v.clamp(0, 0xFF)
        } else {
            v
        }
    }

    /// Store MAC1-3 (i=0..2 → data 25..27).
    #[inline]
    fn set_mac(&mut self, i: usize, v: i64) {
        self.mac[1 + i] = v;
        self.data[25 + i] = v as i32 as u32;
    }
    #[inline]
    fn mac(&self, i: usize) -> i64 {
        // i = 1..3
        self.mac[i]
    }

    fn push_sxy(&mut self, x: i64, y: i64) {
        self.data[12] = self.data[13];
        self.data[13] = self.data[14];
        self.data[14] = ((x as i32 as u16) as u32) | (((y as i32 as u16) as u32) << 16);
    }

    fn push_sz(&mut self, z: i64) {
        self.data[16] = self.data[17];
        self.data[17] = self.data[18];
        self.data[18] = self.data[19];
        self.data[19] = (z as i32 as u16) as u32;
    }

    /// Push MAC1-3/16 → RGB FIFO (keeps CODE from RGBC), then
    /// IR1-3 = MAC1-3 saturated by lm.
    fn push_color(&mut self, lm: bool) {
        let code = self.data[6] & 0xFF00_0000;
        let r = self.lim_col(0, self.mac(1) >> 4) as u32;
        let g = self.lim_col(1, self.mac(2) >> 4) as u32;
        let b = self.lim_col(2, self.mac(3) >> 4) as u32;
        self.data[20] = self.data[21];
        self.data[21] = self.data[22];
        self.data[22] = code | (b << 16) | (g << 8) | r;
        for i in 1..=3 {
            let v = self.lim_ir(i, self.mac(i), lm);
            self.set_ir(i, v);
        }
    }

    /// Hardware UNR division: n = min(0x1FFFF, ~H*0x20000/SZ3).
    fn divide(&mut self, h: u32, z3: u32) -> i64 {
        if (h as u64) < (z3 as u64) * 2 {
            let z = (z3 as u16).leading_zeros();
            let n = (h as u64) << z;
            let d = (z3 as u64) << z;
            let u = UNR_TABLE[((d - 0x7FC0) >> 7) as usize] as u64 + 0x101;
            let d = (0x2000080u64.wrapping_sub(d.wrapping_mul(u))) >> 8;
            let d = (0x80 + d.wrapping_mul(u)) >> 8;
            ((((n.wrapping_mul(d)) + 0x8000) >> 16).min(0x1FFFF)) as i64
        } else {
            self.flag |= (1 << 17) | (1 << 31);
            0x1FFFF
        }
    }

    /// Shared "(FC − MAC) * IR0 + MAC" stage used by the depth-cue
    /// commands. The (FC−MAC) intermediate lands in IR1-3 saturated
    /// as if lm=0.
    fn far_interp(&mut self, sf: u32) {
        let ir0 = self.ir(0);
        let sh = sf * 12;
        for i in 1..=3 {
            let mut t = self.a(i - 1, self.fc(i - 1) << 12);
            t = self.a(i - 1, t - self.mac(i));
            let iv = self.lim_ir(i, t >> sh, false);
            self.set_ir(i, iv);
            let m = self.a(i - 1, self.mac(i) + iv * ir0);
            self.set_mac(i - 1, m);
        }
    }

    /// Core 3x3 matrix multiply: MACn = (Tn*1000h + Mn1*V1 + Mn2*V2 +
    /// Mn3*V3) SAR (sf*12); IRn = MACn saturated (lm).
    /// `t` is the translation term in raw 32-bit units (None → 0).
    fn mul_mat_vec(&mut self, m: [i64; 9], v: [i64; 3], t: [i64; 3], sf: u32, lm: bool) {
        let sh = sf * 12;
        for i in 0..3 {
            let mut acc = self.a(i, t[i] << 12);
            acc = self.a(i, acc + m[i * 3] * v[0]);
            acc = self.a(i, acc + m[i * 3 + 1] * v[1]);
            acc = self.a(i, acc + m[i * 3 + 2] * v[2]);
            let mac = acc >> sh;
            self.set_mac(i, mac);
            let iv = self.lim_ir(i + 1, mac, lm);
            self.set_ir(i + 1, iv);
        }
    }

    /// Matrix rows by selector: 0=RT, 1=LLM, 2=LCM, 3=garbage matrix.
    fn matrix(&self, mx: u8) -> [i64; 9] {
        match mx {
            0 | 1 | 2 => {
                let base = (mx as usize) * 8;
                let mut m = [0i64; 9];
                for r in 0..3 {
                    for c in 0..3 {
                        m[r * 3 + c] = self.mat(base, r, c);
                    }
                }
                m
            }
            _ => {
                // Hardware bug: mx=3 reads outside the matrix region.
                // Result: -R*16, +R*16, IR0, RT13x3, RT22x3
                // (R = low byte of RGBC).
                let r = (self.data[6] & 0xFF) as i64;
                let rt13 = self.mat(0, 0, 2);
                let rt22 = self.mat(0, 1, 1);
                [
                    -r * 16,
                    r * 16,
                    self.ir(0),
                    rt13,
                    rt13,
                    rt13,
                    rt22,
                    rt22,
                    rt22,
                ]
            }
        }
    }

    /// Vector by selector: 0..2 = V0..V2, 3 = [IR1,IR2,IR3].
    fn vector(&self, v: u8) -> [i64; 3] {
        match v {
            0..=2 => [
                self.v(v as usize, 0),
                self.v(v as usize, 1),
                self.v(v as usize, 2),
            ],
            _ => [self.ir(1), self.ir(2), self.ir(3)],
        }
    }

    /// Translation by selector: 0=TR, 1=BK, 2=FC, 3=none.
    fn translation(&self, cv: u8) -> [i64; 3] {
        match cv {
            0 => [self.tr(0), self.tr(1), self.tr(2)],
            1 => [self.bk(0), self.bk(1), self.bk(2)],
            2 => [self.fc(0), self.fc(1), self.fc(2)],
            _ => [0, 0, 0],
        }
    }

    // ---------------- commands ----------------

    /// One RTPS pass on vertex vi. `depth` runs the DQA/DQB→IR0 tail
    /// (RTPT does it only for the last vertex).
    fn rtps(&mut self, vi: usize, sf: u32, depth: bool) {
        let sh = sf * 12;
        let v = [self.v(vi, 0), self.v(vi, 1), self.v(vi, 2)];
        let rt = self.matrix(0);
        let mut acc = [0i64; 3];
        for i in 0..3 {
            acc[i] = self.a(i, self.tr(i) << 12);
            acc[i] = self.a(i, acc[i] + rt[i * 3] * v[0]);
            acc[i] = self.a(i, acc[i] + rt[i * 3 + 1] * v[1]);
            acc[i] = self.a(i, acc[i] + rt[i * 3 + 2] * v[2]);
        }
        for i in 0..3 {
            let mac = acc[i] >> sh;
            self.set_mac(i, mac);
        }
        // IR1/IR2: saturated -8000h..+7FFFh (RTPS ignores lm)
        for i in 0..2 {
            let iv = self.lim_ir(i + 1, acc[i] >> sh, false);
            self.set_ir(i + 1, iv);
        }
        // IR3 quirk: saturated from MAC3, but FLAG.22 is raised only
        // when (MAC3 SAR 12) exceeds s16 — with sf=1 those coincide.
        if sf == 0 {
            self.set_ir(3, acc[2].clamp(-0x8000, 0x7FFF));
            if !(-0x8000..=0x7FFF).contains(&(acc[2] >> 12)) {
                self.flag |= 1 << 22;
            }
        } else {
            let iv = self.lim_ir(3, acc[2] >> 12, false);
            self.set_ir(3, iv);
        }
        // SZ3 = unshifted accumulator >> 12, saturated u16
        let sz3 = self.lim_u16(acc[2] >> 12);
        self.push_sz(sz3);
        // screen projection
        let n = self.divide(self.h() as u32, sz3 as u32);
        let x0 = n * self.ir(1) + self.ofx();
        self.set_mac0(x0);
        let sx = self.lim_sxy(14, x0 >> 16);
        let y0 = n * self.ir(2) + self.ofy();
        self.set_mac0(y0);
        let sy = self.lim_sxy(13, y0 >> 16);
        self.push_sxy(sx, sy);
        // depth cue
        if depth {
            let m0 = n * self.dqa() + self.dqb();
            self.set_mac0(m0);
            let ir0 = self.lim_ir0(m0 >> 12);
            self.set_ir(0, ir0);
        }
    }

    /// Normal-color core shared by NCS/NCCS/NCDS and the triple
    /// variants: IR = LLM*Vn, then IR = BK*1000h + LCM*IR.
    fn nc_light(&mut self, vi: usize, sf: u32, lm: bool) {
        let v = [self.v(vi, 0), self.v(vi, 1), self.v(vi, 2)];
        let llm = self.matrix(1);
        self.mul_mat_vec(llm, v, [0, 0, 0], sf, lm);
        let ir = [self.ir(1), self.ir(2), self.ir(3)];
        let lcm = self.matrix(2);
        let bk = [self.bk(0), self.bk(1), self.bk(2)];
        self.mul_mat_vec(lcm, ir, bk, sf, lm);
    }

    /// "MAC = R*IR<<4" stage of the NCCx/NCDx/DCPL/CC/CDP family.
    fn color_mul(&mut self, rgb: u32) {
        for i in 0..3 {
            let c = ((rgb >> (i * 8)) & 0xFF) as i64;
            let m = self.a(i, c * self.ir(i + 1) * 16);
            self.set_mac(i, m);
        }
    }

    /// Final "MAC SAR (sf*12); push color; IR = MAC" stage.
    fn sar_push(&mut self, sf: u32, lm: bool) {
        let sh = sf * 12;
        for i in 0..3 {
            let m = self.a(i, self.mac(i + 1)) >> sh;
            self.set_mac(i, m);
        }
        self.push_color(lm);
    }

    /// Execute a GTE command word; returns the hardware cycle count.
    pub fn command(&mut self, op: u32) -> u32 {
        self.flag = 0;
        let sf = (op >> 19) & 1;
        let lm = (op >> 10) & 1 != 0;
        let cycles = match op & 0x3F {
            0x01 => {
                self.rtps(0, sf, true);
                15
            }
            0x06 => {
                // NCLIP
                let m0 =
                    self.sx(0) * self.sy(1) + self.sx(1) * self.sy(2) + self.sx(2) * self.sy(0)
                        - self.sx(0) * self.sy(2)
                        - self.sx(1) * self.sy(0)
                        - self.sx(2) * self.sy(1);
                self.set_mac0(m0);
                8
            }
            0x0C => {
                // OP: IR3*D2-IR2*D3, IR1*D3-IR3*D1, IR2*D1-IR1*D2
                let d = [self.mat(0, 0, 0), self.mat(0, 1, 1), self.mat(0, 2, 2)];
                let ir = [self.ir(1), self.ir(2), self.ir(3)];
                let sh = sf * 12;
                let p = [
                    ir[2] * d[1] - ir[1] * d[2],
                    ir[0] * d[2] - ir[2] * d[0],
                    ir[1] * d[0] - ir[0] * d[1],
                ];
                for i in 0..3 {
                    let m = self.a(i, p[i]) >> sh;
                    self.set_mac(i, m);
                    let iv = self.lim_ir(i + 1, m, lm);
                    self.set_ir(i + 1, iv);
                }
                6
            }
            0x10 => {
                // DPCS: MAC = RGB<<16, depth cue
                let c = self.data[6];
                for i in 0..3 {
                    let m = self.a(i, ((c >> (i * 8)) & 0xFF) as i64 * 0x10000);
                    self.set_mac(i, m);
                }
                self.far_interp(sf);
                self.sar_push(sf, lm);
                8
            }
            0x11 => {
                // INTPL: MAC = IR<<12, depth cue
                for i in 0..3 {
                    let m = self.a(i, self.ir(i + 1) << 12);
                    self.set_mac(i, m);
                }
                self.far_interp(sf);
                self.sar_push(sf, lm);
                8
            }
            0x12 => {
                // MVMVA(sf,mx,v,cv,lm)
                let mx = ((op >> 17) & 3) as u8;
                let vs = ((op >> 15) & 3) as u8;
                let cv = ((op >> 13) & 3) as u8;
                let m = self.matrix(mx);
                let v = self.vector(vs);
                if cv == 2 {
                    // Bugged FC translation: the adder chain splits —
                    // part1 (Tx*1000h + M11*V1) only drives the flags
                    // (saturated as lm=0), part2 is the real result.
                    let t = self.translation(2);
                    let sh = sf * 12;
                    for i in 0..3 {
                        let mut acc = self.a(i, t[i] << 12);
                        acc = self.a(i, acc + m[i * 3] * v[0]);
                        let mac = acc >> sh;
                        self.set_mac(i, mac);
                        // sat check as if lm=0, result discarded
                        self.lim_ir(i + 1, mac, false);
                        let mut acc2 = self.a(i, m[i * 3 + 1] * v[1]);
                        acc2 = self.a(i, acc2 + m[i * 3 + 2] * v[2]);
                        let mac2 = acc2 >> sh;
                        self.set_mac(i, mac2);
                        let iv = self.lim_ir(i + 1, mac2, lm);
                        self.set_ir(i + 1, iv);
                    }
                } else {
                    let t = self.translation(cv);
                    self.mul_mat_vec(m, v, t, sf, lm);
                }
                8
            }
            0x13 => {
                // NCDS: normal color + depth cue, single
                self.nc_light(0, sf, lm);
                self.color_mul(self.data[6]);
                self.far_interp(sf);
                self.sar_push(sf, lm);
                19
            }
            0x14 => {
                // CDP: color depth cue — BK+LCM*IR, R*IR<<4, far interp
                let ir = [self.ir(1), self.ir(2), self.ir(3)];
                let lcm = self.matrix(2);
                let bk = [self.bk(0), self.bk(1), self.bk(2)];
                self.mul_mat_vec(lcm, ir, bk, sf, lm);
                self.color_mul(self.data[6]);
                self.far_interp(sf);
                self.sar_push(sf, lm);
                13
            }
            0x16 => {
                // NCDT: NCDS for V0,V1,V2
                for vi in 0..3 {
                    self.nc_light(vi, sf, lm);
                    self.color_mul(self.data[6]);
                    self.far_interp(sf);
                    self.sar_push(sf, lm);
                }
                44
            }
            0x1B => {
                // NCCS
                self.nc_light(0, sf, lm);
                self.color_mul(self.data[6]);
                self.sar_push(sf, lm);
                17
            }
            0x1C => {
                // CC: BK+LCM*IR → R*IR<<4 → SAR → push
                let ir = [self.ir(1), self.ir(2), self.ir(3)];
                let lcm = self.matrix(2);
                let bk = [self.bk(0), self.bk(1), self.bk(2)];
                self.mul_mat_vec(lcm, ir, bk, sf, lm);
                self.color_mul(self.data[6]);
                self.sar_push(sf, lm);
                11
            }
            0x1E => {
                // NCS
                self.nc_light(0, sf, lm);
                self.sar_push(sf, lm);
                14
            }
            0x20 => {
                // NCT
                for vi in 0..3 {
                    self.nc_light(vi, sf, lm);
                    self.sar_push(sf, lm);
                }
                30
            }
            0x28 => {
                // SQR: MACn = IRn² SHR (sf*12)
                let sh = sf * 12;
                for i in 0..3 {
                    let m = self.a(i, self.ir(i + 1) * self.ir(i + 1)) >> sh;
                    self.set_mac(i, m);
                    let iv = self.lim_ir(i + 1, m, lm);
                    self.set_ir(i + 1, iv);
                }
                5
            }
            0x29 => {
                // DCPL: MAC = R*IR<<4 + far interp
                self.color_mul(self.data[6]);
                self.far_interp(sf);
                self.sar_push(sf, lm);
                8
            }
            0x2A => {
                // DPCT: DPCS ×3, colors consumed from RGB0 (fifo bottom)
                for _ in 0..3 {
                    let c = self.data[20];
                    for i in 0..3 {
                        let m = self.a(i, ((c >> (i * 8)) & 0xFF) as i64 * 0x10000);
                        self.set_mac(i, m);
                    }
                    self.far_interp(sf);
                    self.sar_push(sf, lm);
                }
                17
            }
            0x2D => {
                // AVSZ3
                let m0 = self.mat_zsf(0) * (self.sz(1) + self.sz(2) + self.sz(3));
                self.set_mac0(m0);
                let o = self.lim_u16(m0 >> 12);
                self.data[7] = o as u32;
                5
            }
            0x2E => {
                // AVSZ4
                let m0 = self.mat_zsf(1) * (self.sz(0) + self.sz(1) + self.sz(2) + self.sz(3));
                self.set_mac0(m0);
                let o = self.lim_u16(m0 >> 12);
                self.data[7] = o as u32;
                6
            }
            0x30 => {
                // RTPT: V0,V1,V2 — depth cue only on the last
                self.rtps(0, sf, false);
                self.rtps(1, sf, false);
                self.rtps(2, sf, true);
                23
            }
            0x3D => {
                // GPF: MAC = IR*IR0 → SAR → push
                let sh = sf * 12;
                let ir0 = self.ir(0);
                for i in 0..3 {
                    let m = self.a(i, self.ir(i + 1) * ir0) >> sh;
                    self.set_mac(i, m);
                }
                self.push_color(lm);
                5
            }
            0x3E => {
                // GPL: MAC<<sf*12 + IR*IR0 → SAR → push
                let sh = sf * 12;
                let ir0 = self.ir(0);
                for i in 0..3 {
                    let base = self.a(i, self.mac(i + 1) << sh);
                    let m = self.a(i, base + self.ir(i + 1) * ir0) >> sh;
                    self.set_mac(i, m);
                }
                self.push_color(lm);
                5
            }
            0x3F => {
                // NCCT
                for vi in 0..3 {
                    self.nc_light(vi, sf, lm);
                    self.color_mul(self.data[6]);
                    self.sar_push(sf, lm);
                }
                39
            }
            _ => 0,
        };
        // FLAG: bit31 = OR of (30..23) and (18..13)
        let err = (self.flag & 0x7F80_0000) != 0 || (self.flag & 0x0007_E000) != 0;
        self.ctrl[31] = self.flag | ((err as u32) << 31);
        cycles
    }

    #[inline]
    fn mat_zsf(&self, i: usize) -> i64 {
        (self.ctrl[29 + i] & 0xFFFF) as u16 as i16 as i64
    }

    // ---------------- register file interface ----------------

    #[inline]
    pub fn read_data(&self, r: u8) -> u32 {
        match r & 31 {
            // sign-extended s16 regs
            1 | 3 | 5 => (self.data[(r & 31) as usize] & 0xFFFF) as u16 as i16 as i32 as u32,
            8..=11 => (self.data[(r & 31) as usize] & 0xFFFF) as u16 as i16 as i32 as u32,
            // SXYP mirrors SXY2
            15 => self.data[14],
            // SZn read zero-extended
            16..=19 => self.data[(r & 31) as usize] & 0xFFFF,
            // MAC0-3: low 32 bits of the shadow accumulators
            24..=27 => self.mac[(r & 31) as usize - 24] as i32 as u32,
            // IRGB/ORGB: both read IR1-3 collapsed to 5:5:5
            28 | 29 => {
                let f = |v: i64| -> u32 { (v / 0x80).clamp(0, 0x1F) as u32 };
                f(self.ir(1)) | (f(self.ir(2)) << 5) | (f(self.ir(3)) << 10)
            }
            30 => self.data[30],
            31 => self.data[31],
            _ => self.data[(r & 31) as usize],
        }
    }

    #[inline]
    pub fn write_data(&mut self, r: u8, v: u32) {
        match r & 31 {
            // SXYP: write pushes the SXY FIFO (SXY0←SXY1←SXY2←v)
            15 => {
                self.data[12] = self.data[13];
                self.data[13] = self.data[14];
                self.data[14] = v;
            }
            // MAC0-3 shadow registers
            24..=27 => {
                self.mac[(r & 31) as usize - 24] = v as i32 as i64;
                self.data[(r & 31) as usize] = v;
            }
            // IRGB: expand 5:5:5 → IR1-3 (*0x80 each)
            28 => {
                self.set_ir(1, ((v & 0x1F) * 0x80) as i64);
                self.set_ir(2, (((v >> 5) & 0x1F) * 0x80) as i64);
                self.set_ir(3, (((v >> 10) & 0x1F) * 0x80) as i64);
            }
            // ORGB is read-only
            29 => {}
            // LZCS → compute LZCR (leading zeros if >=0 else ones)
            30 => {
                self.data[30] = v;
                let n = if (v as i32) >= 0 {
                    v.leading_zeros()
                } else {
                    (!v).leading_zeros()
                };
                self.data[31] = n;
            }
            // LZCR is read-only
            31 => {}
            _ => {
                self.data[(r & 31) as usize] = v;
            }
        }
    }

    #[inline]
    pub fn read_ctrl(&self, r: u8) -> u32 {
        match r & 31 {
            // last matrix elements + s16 control regs sign-extend
            4 | 12 | 20 | 27 | 29 | 30 => {
                (self.ctrl[(r & 31) as usize] & 0xFFFF) as u16 as i16 as i32 as u32
            }
            // H: hardware bug — the unsigned 16-bit value is
            // sign-extended on register reads (RTPS math unaffected)
            26 => (self.ctrl[26] & 0xFFFF) as u16 as i16 as i32 as u32,
            // FLAG: bit31 computed from bits 30..23 and 18..13
            31 => {
                let f = self.ctrl[31] & 0x7FFF_F000;
                let err = (f & 0x7F80_0000) != 0 || (f & 0x0007_E000) != 0;
                f | ((err as u32) << 31)
            }
            _ => self.ctrl[(r & 31) as usize],
        }
    }

    #[inline]
    pub fn write_ctrl(&mut self, r: u8, v: u32) {
        if r & 31 == 31 {
            // FLAG: only bits 30-12 are writable
            self.ctrl[31] = v & 0x7FFF_F000;
        } else {
            self.ctrl[(r & 31) as usize] = v;
        }
    }
}
