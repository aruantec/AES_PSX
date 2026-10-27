//! DMA controller: 7 channels at 0x1F801080..0x1F8010EF plus
//! DPCR (0xF0) / DICR (0xF4).
//!
//! Channels: 0=MDECin 1=MDECout 2=GPU 3=CDROM 4=SPU 5=PIO 6=OTC
//!
//! This module owns the register state; the copy loop itself lives
//! in the core (it needs the whole bus). Data moves instantly, but the
//! channel stays busy and its completion IRQ is deferred by roughly
//! the transfer's bus time so software sees a plausible ordering.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Burst,
    Slice,
    LinkedList,
}

#[derive(Clone, Copy, Default)]
pub struct Channel {
    pub madr: u32,
    pub bcr: u32,
    pub chcr: u32,
}

impl Channel {
    pub fn mode(&self) -> Mode {
        match (self.chcr >> 9) & 3 {
            0 => Mode::Burst,
            1 => Mode::Slice,
            _ => Mode::LinkedList,
        }
    }

    /// CHCR bit0: 0 = device→RAM, 1 = RAM→device.
    pub fn to_ram(&self) -> bool {
        self.chcr & 1 == 0
    }

    pub fn step(&self) -> u32 {
        if self.chcr & 2 != 0 { (-4i32) as u32 } else { 4 }
    }

    fn startable(&self) -> bool {
        self.chcr & (1 << 24) != 0
            && (self.mode() != Mode::Burst || self.chcr & (1 << 28) != 0)
    }
}

pub struct Dma {
    pub ch: [Channel; 7],
    pub dpcr: u32,
    pub dicr: u32,
    /// Channels ready to run (start condition met).
    pub pending: u8,
    /// Channels whose data moved and that complete at the given
    /// cycle.
    pub finishing: [Option<u64>; 7],
    irq_level: bool,
    /// Set on a DICR bit31 rising edge (consumed by the core).
    pub irq_edge: bool,
}

impl Dma {
    pub fn new() -> Self {
        Self {
            ch: [Channel::default(); 7],
            dpcr: 0x0765_4321,
            dicr: 0,
            pending: 0,
            finishing: [None; 7],
            irq_level: false,
            irq_edge: false,
        }
    }

    pub fn read32(&self, p: u32) -> u32 {
        match p & 0x7F {
            0x70 => self.dpcr,
            0x74 => self.dicr | ((self.irq_out() as u32) << 31),
            o => {
                let c = &self.ch[(o >> 4) as usize];
                match o & 0xF {
                    0 => c.madr,
                    4 => c.bcr,
                    8 => c.chcr,
                    _ => 0,
                }
            }
        }
    }

    pub fn write32(&mut self, p: u32, v: u32) {
        match p & 0x7F {
            0x70 => {
                self.dpcr = v;
                self.check_all();
            }
            0x74 => {
                let flags = (self.dicr & 0x7F00_0000) & !(v & 0x7F00_0000);
                self.dicr = (v & 0x00FF_803F) | flags;
                self.update_irq();
            }
            o => {
                let i = (o >> 4) as usize;
                match o & 0xF {
                    0 => self.ch[i].madr = v & 0xFF_FFFF,
                    4 => self.ch[i].bcr = v,
                    8 => {
                        let v = if i == 6 { (v & 0x5100_0000) | 2 } else { v & 0x7177_0703 };
                        self.ch[i].chcr = v;
                        self.check(i);
                    }
                    _ => {}
                }
            }
        }
    }

    fn check(&mut self, i: usize) {
        if self.finishing[i].is_none()
            && self.ch[i].startable()
            && (self.dpcr >> (i * 4)) & 8 != 0
        {
            self.pending |= 1 << i;
        }
    }

    fn check_all(&mut self) {
        for i in 0..7 {
            self.check(i);
        }
    }

    /// Mark channel `i` as transferred; it completes at `at`.
    pub fn started(&mut self, i: usize, at: u64) {
        self.pending &= !(1 << i);
        self.ch[i].chcr &= !(1 << 28);
        self.finishing[i] = Some(at);
    }

    /// Retire channels whose completion time passed.
    pub fn tick(&mut self, now: u64) {
        for i in 0..7 {
            if let Some(at) = self.finishing[i] {
                if now >= at {
                    self.finishing[i] = None;
                    self.ch[i].chcr &= !(1 << 24);
                    if self.dicr & (1 << (16 + i)) != 0 {
                        self.dicr |= 1 << (24 + i);
                    }
                    self.update_irq();
                }
            }
        }
    }

    pub fn busy(&self) -> bool {
        self.finishing.iter().any(|f| f.is_some())
    }

    fn update_irq(&mut self) {
        let lvl = self.irq_out();
        if lvl && !self.irq_level {
            self.irq_edge = true;
        }
        self.irq_level = lvl;
    }

    pub fn irq_out(&self) -> bool {
        let flagged = (self.dicr >> 24) & (self.dicr >> 16) & 0x7F;
        self.dicr & (1 << 15) != 0 || (self.dicr & (1 << 23) != 0 && flagged != 0)
    }

    /// Words to move for a burst/slice transfer.
    pub fn word_count(&self, i: usize) -> u32 {
        let b = self.ch[i].bcr;
        match self.ch[i].mode() {
            Mode::Burst => match b & 0xFFFF {
                0 => 0x1_0000,
                n => n,
            },
            Mode::Slice => {
                let size = match b & 0xFFFF {
                    0 => 0x1_0000,
                    n => n,
                };
                size * ((b >> 16) & 0xFFFF).max(1)
            }
            Mode::LinkedList => 0,
        }
    }
}
