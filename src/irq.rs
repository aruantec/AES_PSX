//! Interrupt controller: I_STAT (0x1F801070) + I_MASK (0x1F801074).
//!
//! Devices signal edges: `raise` latches the bit in I_STAT, software
//! acknowledges by writing 0 to it. The CPU sees the OR of
//! (I_STAT & I_MASK) on Cause.IP2.
//!
//! Line map: 0=vblank 1=GPU 2=CDROM 3=DMA 4=TMR0 5=TMR1 6=TMR2
//! 7=pad/memcard 8=SIO 9=SPU 10=PIO

pub const VBLANK: u16 = 1 << 0;
pub const GPU: u16 = 1 << 1;
pub const CDROM: u16 = 1 << 2;
pub const DMA: u16 = 1 << 3;
pub const TMR0: u16 = 1 << 4;
pub const TMR1: u16 = 1 << 5;
pub const TMR2: u16 = 1 << 6;
pub const PAD: u16 = 1 << 7;
pub const SIO: u16 = 1 << 8;
pub const SPU: u16 = 1 << 9;
pub const PIO: u16 = 1 << 10;

#[derive(Default)]
pub struct Irq {
    pub stat: u16,
    pub mask: u16,
}

impl Irq {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn raise(&mut self, line: u16) {
        self.stat |= line;
    }

    /// Writing 0-bits acknowledges; 1-bits are preserved.
    #[inline]
    pub fn ack(&mut self, v: u32) {
        self.stat &= v as u16;
    }

    #[inline]
    pub fn stat(&self) -> u32 {
        self.stat as u32
    }

    #[inline]
    pub fn mask(&self) -> u32 {
        self.mask as u32
    }

    #[inline]
    pub fn set_mask(&mut self, v: u32) {
        self.mask = v as u16 & 0x7FF;
    }

    #[inline]
    pub fn pending(&self) -> bool {
        self.stat & self.mask != 0
    }
}
