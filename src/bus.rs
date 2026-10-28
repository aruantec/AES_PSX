//! Physical memory map and IO dispatch.
//!
//! The R3000A has no MMU; virtual→physical is a fixed segment
//! translation:
//!   KUSEG 0x0000_0000-0x7FFF_FFFF → identity
//!   KSEG0 0x8000_0000-0x9FFF_FFFF → cached, & 0x1FFF_FFFF
//!   KSEG1 0xA000_0000-0xBFFF_FFFF → uncached, & 0x1FFF_FFFF
//!   KSEG2 0xC000_0000+            → cache control at 0xFFFE_0130
//!
//! Physical regions:
//!   0000_0000-007F_FFFF  2 MiB main RAM, mirrored
//!   1F00_0000-1F7F_FFFF  expansion region 1 (open bus)
//!   1F80_0000-1F80_03FF  1 KiB scratchpad
//!   1F80_1000-1F80_2FFF  IO ports
//!   1FC0_0000-1FC7_FFFF  512 KiB BIOS ROM
//!
//! IO registers are dispatched by access width so that halfword and
//! byte accesses never perform hidden read-modify-write cycles on
//! side-effecting registers (FIFOs, ack-on-read flags).

use crate::cdrom::CdRom;
use crate::dma::Dma;
use crate::gpu::Gpu;
use crate::irq::Irq;
use crate::mdec::Mdec;
use crate::sio::Sio;
use crate::spu::Spu;
use crate::timers::Timers;

pub const RAM_SIZE: usize = 2 * 1024 * 1024;
pub const SCRATCH_SIZE: usize = 1024;
pub const BIOS_SIZE: usize = 512 * 1024;

#[derive(Clone, Copy, PartialEq)]
enum W {
    B,
    H,
    Wd,
}

pub struct Bus {
    pub ram: Vec<u8>,
    pub scratch: [u8; SCRATCH_SIZE],
    pub bios: Vec<u8>,
    pub irq: Irq,
    pub dma: Dma,
    pub timers: Timers,
    pub gpu: Gpu,
    pub spu: Spu,
    pub cdrom: CdRom,
    pub mdec: Mdec,
    pub sio: Sio,
    /// sysclock mirror, set by the core each step.
    pub now: u64,
    mem_ctrl: [u32; 9],
    ram_size: u32,
    pub cache_control: u32,
    pub post_led: u8,
    /// Captured debug output (SIO1, DUART, BIOS putchar hook).
    pub tty: String,
    /// Cop0 SR.Isc: data stores hit the (isolated) I-cache only.
    pub cache_isolated: bool,
    /// Access-time penalty accumulated by the current instruction.
    pub stall: u32,
    /// I-cache tags (timing model): 256 lines x 16 bytes.
    icache_tag: Vec<u32>,
}

const RAM_READ: u32 = 4;
const ROM_READ: u32 = 20;
const IO_READ: u32 = 2;
const ICACHE_FILL_BASE: u32 = 3;

impl Bus {
    pub fn new(bios: Vec<u8>) -> Self {
        assert_eq!(bios.len(), BIOS_SIZE, "BIOS dump must be 512 KiB");
        let mut cdrom = CdRom::new();
        cdrom.region = bios_region(&bios);
        Self {
            ram: vec![0; RAM_SIZE],
            scratch: [0; SCRATCH_SIZE],
            bios,
            irq: Irq::new(),
            dma: Dma::new(),
            timers: Timers::new(),
            gpu: Gpu::new(),
            spu: Spu::new(),
            cdrom,
            mdec: Mdec::new(),
            sio: Sio::new(),
            now: 0,
            mem_ctrl: [0; 9],
            ram_size: 0x0000_0B88,
            cache_control: 0,
            post_led: 0,
            tty: String::new(),
            cache_isolated: false,
            stall: 0,
            icache_tag: vec![0; 256],
        }
    }

    #[inline]
    pub fn phys(vaddr: u32) -> u32 {
        if vaddr >> 29 >= 6 {
            vaddr
        } else {
            vaddr & 0x1FFF_FFFF
        }
    }

    #[inline]
    fn read_cost(p: u32) -> u32 {
        match p {
            _ if p < 0x0080_0000 => RAM_READ,
            0x1F80_0000..=0x1F80_03FF => 0,
            0x1FC0_0000..=0x1FC7_FFFF => ROM_READ,
            _ => IO_READ,
        }
    }

    #[inline]
    fn fetch_cost(&mut self, vaddr: u32, p: u32) {
        let cached = vaddr < 0xA000_0000 && self.cache_control & (1 << 11) != 0;
        if !cached || p >= 0x1F00_0000 {
            self.stall += Self::read_cost(p);
            return;
        }
        let line = p & !0xF;
        let idx = ((p >> 4) & 0xFF) as usize;
        if self.icache_tag[idx] != line | 1 {
            self.icache_tag[idx] = line | 1;
            self.stall += ICACHE_FILL_BASE + 4 - ((p >> 2) & 3);
        }
    }

    /// Instruction fetch: ignores cache isolation (Isc gates data
    /// accesses only).
    #[inline]
    pub fn fetch32(&mut self, vaddr: u32) -> u32 {
        let p = Self::phys(vaddr);
        self.fetch_cost(vaddr, p);
        match p {
            _ if p < 0x0080_0000 => {
                let i = (p & 0x1F_FFFC) as usize;
                u32::from_le_bytes(self.ram[i..i + 4].try_into().unwrap())
            }
            0x1FC0_0000..=0x1FC7_FFFF => {
                let i = (p & 0x7_FFFC) as usize;
                u32::from_le_bytes(self.bios[i..i + 4].try_into().unwrap())
            }
            _ => self.read(p, W::Wd),
        }
    }

    #[inline]
    pub fn load8(&mut self, vaddr: u32) -> u8 {
        self.load(vaddr, W::B) as u8
    }
    #[inline]
    pub fn load16(&mut self, vaddr: u32) -> u16 {
        self.load(vaddr, W::H) as u16
    }
    #[inline]
    pub fn load32(&mut self, vaddr: u32) -> u32 {
        self.load(vaddr, W::Wd)
    }
    #[inline]
    pub fn store8(&mut self, vaddr: u32, v: u8) {
        self.store(vaddr, v as u32, W::B)
    }
    #[inline]
    pub fn store16(&mut self, vaddr: u32, v: u16) {
        self.store(vaddr, v as u32, W::H)
    }
    #[inline]
    pub fn store32(&mut self, vaddr: u32, v: u32) {
        self.store(vaddr, v, W::Wd)
    }

    #[inline]
    fn load(&mut self, vaddr: u32, w: W) -> u32 {
        if self.cache_isolated {
            return 0;
        }
        let p = Self::phys(vaddr);
        self.stall += Self::read_cost(p);
        self.read(p, w)
    }

    #[inline]
    fn store(&mut self, vaddr: u32, v: u32, w: W) {
        if self.cache_isolated {
            // stores land in the isolated I-cache: the BIOS uses this
            // to invalidate lines; RAM is untouched
            self.icache_tag[((vaddr >> 4) & 0xFF) as usize] = 0;
            return;
        }
        let p = Self::phys(vaddr);
        self.write(p, v, w);
    }

    #[inline]
    fn mem_read(buf: &[u8], i: usize, w: W) -> u32 {
        match w {
            W::B => buf[i] as u32,
            W::H => u16::from_le_bytes([buf[i], buf[i + 1]]) as u32,
            W::Wd => u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()),
        }
    }

    #[inline]
    fn mem_write(buf: &mut [u8], i: usize, v: u32, w: W) {
        match w {
            W::B => buf[i] = v as u8,
            W::H => buf[i..i + 2].copy_from_slice(&(v as u16).to_le_bytes()),
            W::Wd => buf[i..i + 4].copy_from_slice(&v.to_le_bytes()),
        }
    }

    fn read(&mut self, p: u32, w: W) -> u32 {
        match p {
            _ if p < 0x0080_0000 => Self::mem_read(&self.ram, (p & 0x1F_FFFF) as usize, w),
            0x1F80_0000..=0x1F80_03FF => Self::mem_read(&self.scratch, (p & 0x3FF) as usize, w),
            0x1FC0_0000..=0x1FC7_FFFF => Self::mem_read(&self.bios, (p & 0x7_FFFF) as usize, w),
            0x1F80_1000..=0x1F80_2FFF => self.io_read(p, w),
            0xFFFE_0130 => self.cache_control,
            _ => match w {
                W::B => 0xFF,
                W::H => 0xFFFF,
                W::Wd => 0xFFFF_FFFF,
            },
        }
    }

    fn write(&mut self, p: u32, v: u32, w: W) {
        match p {
            _ if p < 0x0080_0000 => Self::mem_write(&mut self.ram, (p & 0x1F_FFFF) as usize, v, w),
            0x1F80_0000..=0x1F80_03FF => {
                Self::mem_write(&mut self.scratch, (p & 0x3FF) as usize, v, w)
            }
            0x1F80_1000..=0x1F80_2FFF => self.io_write(p, v, w),
            0xFFFE_0130 => self.cache_control = v,
            _ => {}
        }
    }

    /// Value of the 32-bit-aligned register containing `p` for
    /// side-effect-free register blocks.
    fn io_read(&mut self, p: u32, w: W) -> u32 {
        let sh = (p & 3) * 8;
        let narrow = |v: u32| -> u32 {
            match w {
                W::B => (v >> sh) & 0xFF,
                W::H => (v >> sh) & 0xFFFF,
                W::Wd => v,
            }
        };
        match p {
            0x1F80_1000..=0x1F80_1023 => narrow(self.mem_ctrl[((p - 0x1F80_1000) >> 2) as usize]),
            0x1F80_1060..=0x1F80_1063 => narrow(self.ram_size),
            0x1F80_1040..=0x1F80_105F => {
                let o = p & 0x1F;
                if w == W::Wd && o == 0x08 || o == 0x18 && w == W::Wd {
                    self.sio.read(p) | self.sio.read(p + 2) << 16
                } else {
                    let v = self.sio.read(p & !1);
                    if w == W::B && p & 1 != 0 {
                        (v >> 8) & 0xFF
                    } else {
                        v
                    }
                }
            }
            0x1F80_1070..=0x1F80_1073 => narrow(self.irq.stat()),
            0x1F80_1074..=0x1F80_1077 => narrow(self.irq.mask()),
            0x1F80_1080..=0x1F80_10FF => narrow(self.dma.read32(p & !3)),
            0x1F80_1100..=0x1F80_112F => narrow(self.timers.read32(p & !3)),
            0x1F80_1800..=0x1F80_1803 => {
                let n = match w {
                    W::B => 1,
                    W::H => 2,
                    W::Wd => 4,
                };
                let mut v = 0;
                for i in 0..n {
                    v |= (self.cdrom.read(p + i) as u32) << (i * 8);
                }
                v
            }
            0x1F80_1810..=0x1F80_1813 => narrow(self.gpu.read_data()),
            0x1F80_1814..=0x1F80_1817 => narrow(self.gpu.read_stat()),
            0x1F80_1820..=0x1F80_1823 => narrow(self.mdec.read_data()),
            0x1F80_1824..=0x1F80_1827 => narrow(self.mdec.read_stat()),
            0x1F80_1C00..=0x1F80_1FFF => {
                let off = p & 0x3FE;
                match w {
                    W::Wd => self.spu.read16(off) as u32 | (self.spu.read16(off + 2) as u32) << 16,
                    W::H => self.spu.read16(off) as u32,
                    W::B => (self.spu.read16(off) as u32 >> ((p & 1) * 8)) & 0xFF,
                }
            }
            0x1F80_2021 => 0x04, // DUART status: TX ready
            _ => match w {
                W::B => 0xFF,
                W::H => 0xFFFF,
                W::Wd => 0xFFFF_FFFF,
            },
        }
    }

    fn io_write(&mut self, p: u32, v: u32, w: W) {
        // merge a narrow write into a side-effect-free 32-bit register
        let merge = |old: u32| -> u32 {
            let sh = (p & 3) * 8;
            match w {
                W::B => (old & !(0xFF << sh)) | ((v & 0xFF) << sh),
                W::H => (old & !(0xFFFF << sh)) | ((v & 0xFFFF) << sh),
                W::Wd => v,
            }
        };
        match p {
            0x1F80_1000..=0x1F80_1023 => {
                let i = ((p - 0x1F80_1000) >> 2) as usize;
                self.mem_ctrl[i] = merge(self.mem_ctrl[i]);
            }
            0x1F80_1060..=0x1F80_1063 => self.ram_size = merge(self.ram_size),
            0x1F80_1040..=0x1F80_105F => {
                let o = p & 0x1F;
                if w == W::Wd && (o == 0x08 || o == 0x18) {
                    self.sio.write(p, v & 0xFFFF, self.now, &mut self.tty);
                    self.sio.write(p + 2, v >> 16, self.now, &mut self.tty);
                } else {
                    self.sio.write(p, v, self.now, &mut self.tty);
                }
            }
            0x1F80_1070..=0x1F80_1073 => {
                let m = merge(0xFFFF_FFFF);
                self.irq.ack(m);
            }
            0x1F80_1074..=0x1F80_1077 => {
                let m = merge(self.irq.mask());
                self.irq.set_mask(m);
            }
            0x1F80_1080..=0x1F80_10FF => {
                let a = p & !3;
                let mut old = self.dma.read32(a);
                if a == 0x1F80_10F4 {
                    old &= !0xFF00_0000; // never re-ack flags via merge
                }
                self.dma.write32(a, merge(old));
            }
            0x1F80_1100..=0x1F80_112F => {
                if p & 3 == 0 {
                    self.timers.write32(p, v & 0xFFFF);
                }
            }
            0x1F80_1800..=0x1F80_1803 => self.cdrom.write(p, v as u8),
            0x1F80_1810 => self.gpu.gp0(v),
            0x1F80_1814 => self.gpu.gp1(v),
            0x1F80_1820 => self.mdec.write_cmd(v),
            0x1F80_1824 => self.mdec.write_ctrl(v),
            0x1F80_1C00..=0x1F80_1FFF => {
                let off = p & 0x3FE;
                match w {
                    W::Wd => {
                        self.spu.write16(off, v as u16);
                        self.spu.write16(off + 2, (v >> 16) as u16);
                    }
                    _ => self.spu.write16(off, v as u16),
                }
            }
            0x1F80_2023 | 0x1F80_2080 => self.tty.push(v as u8 as char),
            0x1F80_2041 => self.post_led = v as u8,
            _ => {}
        }
    }
}

/// Console region from the BIOS version string ("... 12/04/95 A").
pub fn bios_region(bios: &[u8]) -> u8 {
    let needle = b"System ROM Version";
    if let Some(pos) = bios.windows(needle.len()).position(|w| w == needle) {
        let tail = &bios[pos..(pos + 48).min(bios.len())];
        let end = tail.iter().position(|&c| c == 0).unwrap_or(tail.len());
        if let Some(&c) = tail[..end].iter().rev().find(|c| c.is_ascii_alphabetic()) {
            return match c {
                b'E' => b'E',
                b'J' => b'I',
                _ => b'A',
            };
        }
    }
    b'A'
}
