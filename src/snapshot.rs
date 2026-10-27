//! Save/warm-boot state.
//!
//! Layout (little-endian, self-delim):
//!   magic "ARPSX" u32 | version u32 | cpu regs | hi/lo/pc/next_pc |
//!   cop0 | gte | ram | scratch | exp1? (no) | device blobs…
//!
//! For the warm-boot cache the key is
//!   SHA256(disc_id || bios_sha || CORE_VERSION)
//! so any stale snapshot is rejected automatically.

use sha2::Digest;

pub const MAGIC: u32 = 0x4152_5053; // "ARPS"
pub const VERSION: u32 = 1;

/// Build a cache key for a warm-boot snapshot.
pub fn cache_key(disc_id: &[u8; 32], bios: &[u8], core_version: &str) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(disc_id);
    h.update(sha2::Sha256::digest(bios));
    h.update(core_version.as_bytes());
    h.finalize().into()
}

#[derive(Clone)]
pub struct State {
    pub r: [u32; 32],
    pub hi: u32,
    pub lo: u32,
    pub pc: u32,
    pub next_pc: u32,
    pub cop0: [u32; 16],
    pub gte_data: [u32; 32],
    pub gte_ctrl: [u32; 32],
    pub gte_mac: [i64; 4],
    pub ram: Vec<u8>,
    pub scratch: [u8; 1024],
    /// Device register dumps — packed opaquely for now.
    pub irq_stat: u16,
    pub irq_mask: u16,
}

impl State {
    pub fn capture(emu: &crate::Emulator) -> Self {
        // cpu fields are private — go through accessors
        let c = &emu.cpu;
        Self {
            r: c.regs_copy(),
            hi: c.hi(),
            lo: c.lo(),
            pc: c.pc_raw(),
            next_pc: c.next_pc_raw(),
            cop0: c.cop0_copy(),
            gte_data: c.gte.data,
            gte_ctrl: c.gte.ctrl,
            gte_mac: c.gte.mac,
            ram: emu.bus.ram.clone(),
            scratch: emu.bus.scratch,
            irq_stat: emu.bus.irq.stat() as u16,
            irq_mask: emu.bus.irq.mask() as u16,
        }
    }

    pub fn apply(emu: &mut crate::Emulator, st: &Self) {
        emu.cpu.load_regs(st.r, st.hi, st.lo, st.pc, st.next_pc, st.cop0);
        emu.cpu.gte.data = st.gte_data;
        emu.cpu.gte.ctrl = st.gte_ctrl;
        emu.cpu.gte.mac = st.gte_mac;
        emu.bus.ram.copy_from_slice(&st.ram);
        emu.bus.scratch = st.scratch;
        emu.bus.irq.set_mask(st.irq_mask as u32);
        emu.bus.irq.stat = st.irq_stat;
    }

    /// Very small hand-rolled binary codec — fast + deterministic.
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 + 32 * 4 + 16 * 4 + 64 * 4 + self.ram.len());
        v.extend_from_slice(&MAGIC.to_le_bytes());
        v.extend_from_slice(&VERSION.to_le_bytes());
        for x in &self.r {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&self.hi.to_le_bytes());
        v.extend_from_slice(&self.lo.to_le_bytes());
        v.extend_from_slice(&self.pc.to_le_bytes());
        v.extend_from_slice(&self.next_pc.to_le_bytes());
        for x in &self.cop0 {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.gte_data {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.gte_ctrl {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.gte_mac {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&(self.ram.len() as u64).to_le_bytes());
        v.extend_from_slice(&self.ram);
        v.extend_from_slice(&self.scratch);
        v.extend_from_slice(&self.irq_stat.to_le_bytes());
        v.extend_from_slice(&self.irq_mask.to_le_bytes());
        v
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            if p + n <= buf.len() {
                let s = &buf[p..p + n];
                p += n;
                Some(s)
            } else {
                None
            }
        };
        if u32::from_le_bytes(take(4)?.try_into().ok()?) != MAGIC {
            return None;
        }
        if u32::from_le_bytes(take(4)?.try_into().ok()?) != VERSION {
            return None;
        }
        let mut r = [0u32; 32];
        for x in &mut r {
            *x = u32::from_le_bytes(take(4)?.try_into().ok()?);
        }
        let hi = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let lo = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let pc = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let next_pc = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let mut cop0 = [0u32; 16];
        for x in &mut cop0 {
            *x = u32::from_le_bytes(take(4)?.try_into().ok()?);
        }
        let mut gte_data = [0u32; 32];
        for x in &mut gte_data {
            *x = u32::from_le_bytes(take(4)?.try_into().ok()?);
        }
        let mut gte_ctrl = [0u32; 32];
        for x in &mut gte_ctrl {
            *x = u32::from_le_bytes(take(4)?.try_into().ok()?);
        }
        let mut gte_mac = [0i64; 4];
        for x in &mut gte_mac {
            *x = i64::from_le_bytes(take(8)?.try_into().ok()?);
        }
        let ram_len = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
        let ram = take(ram_len)?.to_vec();
        let mut scratch = [0u8; 1024];
        scratch.copy_from_slice(take(1024)?);
        let irq_stat = u16::from_le_bytes(take(2)?.try_into().ok()?);
        let irq_mask = u16::from_le_bytes(take(2)?.try_into().ok()?);
        Some(Self {
            r, hi, lo, pc, next_pc, cop0, gte_data, gte_ctrl, gte_mac,
            ram, scratch, irq_stat, irq_mask,
        })
    }
}
