//! Root counters: TMR0/1/2 at 0x1F801100 + 0x10 each.
//!   +0 current value   +4 mode   +8 target
//!
//! Mode register:
//!   0     sync enable           1-2  sync mode
//!   3     reset after target    4    IRQ at target
//!   5     IRQ at 0xFFFF         6    repeat (else one-shot)
//!   7     toggle (else pulse)   8-9  clock source
//!   10    IRQ request, active low (r)
//!   11    target reached (r, cleared on read)
//!   12    0xFFFF reached (r, cleared on read)
//!
//! Clock sources: T0 sysclk / dotclock, T1 sysclk / hblank,
//! T2 sysclk / sysclk÷8. Dotclock and hblank pulses are delivered by
//! the video timing model in the core.

use crate::irq;

pub const SYSCLK: u64 = 33_868_800;

#[derive(Clone, Default)]
struct Counter {
    count: u16,
    mode: u32,
    target: u16,
    /// One-shot mode: an IRQ was already delivered since the last
    /// mode write.
    fired: bool,
    /// Sync mode 3 (T0/T1): the first blank has been seen, run free.
    released: bool,
    /// Gated by the sync mode right now.
    paused: bool,
    /// Sysclk÷8 remainder for T2.
    div8: u32,
}

impl Counter {
    fn reset_mode(&mut self, v: u32) {
        self.mode = (v & 0x3FF) | (1 << 10);
        self.count = 0;
        self.fired = false;
        self.released = false;
        self.div8 = 0;
    }

    /// Target/overflow event; returns true when the IRQ line edges.
    fn event(&mut self, flag: u32, irq_bit: u32) -> bool {
        self.mode |= flag;
        if self.mode & irq_bit == 0 {
            return false;
        }
        let repeat = self.mode & (1 << 6) != 0;
        if !repeat && self.fired {
            return false;
        }
        self.fired = true;
        if self.mode & (1 << 7) != 0 {
            // toggle mode: the request bit flips, IRQ on 1→0
            self.mode ^= 1 << 10;
            self.mode & (1 << 10) == 0
        } else {
            // pulse mode: bit10 drops briefly, reads back as 1
            true
        }
    }

    /// Advance by `n` counter clocks; returns true on an IRQ edge.
    fn advance(&mut self, mut n: u32) -> bool {
        if self.paused || n == 0 {
            return false;
        }
        let mut irq = false;
        let reset_at_target = self.mode & (1 << 3) != 0;
        while n > 0 {
            let c = self.count as u32;
            if c == 0xFFFF {
                self.count = 0;
                n -= 1;
                continue;
            }
            let tgt = self.target as u32;
            let stop = if tgt > c { tgt } else { 0xFFFF };
            let dist = stop - c;
            if n < dist {
                self.count = (c + n) as u16;
                break;
            }
            n -= dist;
            self.count = stop as u16;
            if stop == tgt {
                irq |= self.event(1 << 11, 1 << 4);
                if reset_at_target {
                    self.count = 0;
                    continue;
                }
            }
            if self.count == 0xFFFF {
                irq |= self.event(1 << 12, 1 << 5);
            }
        }
        irq
    }
}

pub struct Timers {
    t: [Counter; 3],
    /// Pending irq bits to raise (consumed by the core).
    pub pending_irq: u16,
    in_hblank: bool,
    in_vblank: bool,
}

impl Timers {
    pub fn new() -> Self {
        let mut t: [Counter; 3] = Default::default();
        for c in &mut t {
            c.reset_mode(0);
        }
        Self {
            t,
            pending_irq: 0,
            in_hblank: false,
            in_vblank: false,
        }
    }

    fn line(i: usize) -> u16 {
        [irq::TMR0, irq::TMR1, irq::TMR2][i]
    }

    fn bump(&mut self, i: usize, n: u32) {
        if self.t[i].advance(n) {
            self.pending_irq |= Self::line(i);
        }
    }

    fn src(&self, i: usize) -> u32 {
        (self.t[i].mode >> 8) & 3
    }

    /// Advance sysclk-driven counters.
    pub fn tick(&mut self, cycles: u32) {
        if self.src(0) & 1 == 0 {
            self.bump(0, cycles);
        }
        if self.src(1) & 1 == 0 {
            self.bump(1, cycles);
        }
        if self.src(2) & 2 == 0 {
            self.bump(2, cycles);
        } else {
            let d = self.t[2].div8 + cycles;
            self.t[2].div8 = d & 7;
            self.bump(2, d >> 3);
        }
    }

    /// Dotclock pulses (T0 source 1/3).
    pub fn dots(&mut self, n: u32) {
        if self.src(0) & 1 != 0 {
            self.bump(0, n);
        }
    }

    /// Recompute the sync gates for T0/T1 after a blank change.
    fn regate(&mut self, i: usize, blank: bool, edge_start: bool) {
        let c = &mut self.t[i];
        if c.mode & 1 == 0 {
            c.paused = false;
            return;
        }
        match (c.mode >> 1) & 3 {
            0 => c.paused = blank,
            1 => {
                if edge_start {
                    c.count = 0;
                }
                c.paused = false;
            }
            2 => {
                if edge_start {
                    c.count = 0;
                }
                c.paused = !blank;
            }
            _ => {
                if edge_start {
                    c.released = true;
                }
                c.paused = !c.released;
            }
        }
    }

    pub fn set_hblank(&mut self, on: bool) {
        let start = on && !self.in_hblank;
        self.in_hblank = on;
        self.regate(0, on, start);
        if start && self.src(1) & 1 != 0 {
            self.bump(1, 1);
        }
    }

    pub fn set_vblank(&mut self, on: bool) {
        let start = on && !self.in_vblank;
        self.in_vblank = on;
        self.regate(1, on, start);
    }

    pub fn read32(&mut self, p: u32) -> u32 {
        let i = (((p >> 4) & 3) as usize).min(2);
        let c = &mut self.t[i];
        match p & 0xF {
            0 => c.count as u32,
            4 => {
                let m = c.mode;
                c.mode &= !((1 << 11) | (1 << 12));
                m
            }
            8 => c.target as u32,
            _ => 0,
        }
    }

    pub fn write32(&mut self, p: u32, v: u32) {
        let i = (((p >> 4) & 3) as usize).min(2);
        match p & 0xF {
            0 => self.t[i].count = v as u16,
            4 => {
                self.t[i].reset_mode(v);
                match i {
                    0 => self.regate(0, self.in_hblank, false),
                    1 => self.regate(1, self.in_vblank, false),
                    _ => {
                        let sync = v & 1 != 0 && matches!((v >> 1) & 3, 0 | 3);
                        self.t[2].paused = sync;
                    }
                }
            }
            8 => self.t[i].target = v as u16,
            _ => {}
        }
    }
}
