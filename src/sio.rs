//! Serial ports.
//!   0x1F801040-0x1F80104F  SIO0 — controllers + memory cards
//!   0x1F801050-0x1F80105F  SIO1 — serial link (BIOS debug output)
//!
//! SIO0 registers:
//!   +0 DATA (8-bit TX/RX)   +4 STAT   +8 MODE (16)
//!   +A CTRL (16)            +E BAUD (16)
//!
//! A byte exchange takes BAUD*8 sysclks. When the selected device
//! accepts the byte it pulls /ACK low shortly after; with CTRL.12
//! set that edge raises STAT.9 and I_STAT bit7. The last byte of a
//! device's reply is not acknowledged — that's how drivers detect the
//! end of a packet.

/// Delay between transfer end and the device's /ACK pulse.
const ACK_DELAY: u64 = 120;
/// /ACK low-pulse width.
const ACK_WIDTH: u64 = 100;

/// Digital/analog pad. Buttons are active-low.
#[derive(Default)]
struct Pad {
    pos: u8,
    buttons: u16,
}

impl Pad {
    /// Returns (reply, ack).
    fn exchange(&mut self, tx: u8) -> (u8, bool) {
        let p = self.pos;
        self.pos += 1;
        match p {
            0 => (0xFF, true), // address byte (0x01) already matched
            1 => {
                if tx == 0x42 {
                    (0x41, true)
                } else {
                    (0xFF, false)
                }
            }
            2 => (0x5A, true),
            3 => (self.buttons as u8, true),
            4 => ((self.buttons >> 8) as u8, false),
            _ => (0xFF, false),
        }
    }
}

const CARD_SECTORS: usize = 0x400;

/// 128 KiB memory card: 1024 sectors of 128 bytes.
struct Card {
    cmd: u8,
    pos: usize,
    addr: usize,
    prev: u8,
    chk: u8,
    data: Vec<u8>,
    /// FLAG bit3: card freshly inserted (cleared by a write).
    flag: u8,
    wbuf: [u8; 128],
    pub dirty: bool,
}

impl Card {
    fn new() -> Self {
        let mut c = Self {
            cmd: 0,
            pos: 0,
            addr: 0,
            prev: 0,
            chk: 0,
            data: vec![0; CARD_SECTORS * 128],
            flag: 0x08,
            wbuf: [0; 128],
            dirty: false,
        };
        c.format();
        c
    }

    /// Write an empty, formatted filesystem (header + 15 free dir
    /// frames) so games see a usable card instead of "unformatted".
    fn format(&mut self) {
        self.data.fill(0);
        let frame = |d: &mut [u8], sec: usize, f: &dyn Fn(&mut [u8])| {
            let s = &mut d[sec * 128..sec * 128 + 128];
            f(s);
            let x = s[..127].iter().fold(0u8, |a, b| a ^ b);
            s[127] = x;
        };
        frame(&mut self.data, 0, &|s| {
            s[0] = b'M';
            s[1] = b'C';
        });
        for i in 1..16 {
            frame(&mut self.data, i, &|s| {
                s[0] = 0xA0;
                s[8] = 0xFF;
                s[9] = 0xFF;
            });
        }
        for i in 16..36 {
            frame(&mut self.data, i, &|s| {
                s[0..4].fill(0xFF);
                s[8] = 0xFF;
                s[9] = 0xFF;
            });
        }
        frame(&mut self.data, 63, &|s| {
            s[0] = b'M';
            s[1] = b'C';
        });
    }

    fn select(&mut self) {
        self.cmd = 0;
        self.pos = 0;
    }

    fn exchange(&mut self, tx: u8) -> (u8, bool) {
        if self.pos == 0 {
            // address byte 0x81
            self.pos = 1;
            return (0xFF, true);
        }
        if self.pos == 1 {
            self.pos = 2;
            return match tx {
                0x52 | 0x57 | 0x53 => {
                    self.cmd = tx;
                    (self.flag, true)
                }
                _ => (0xFF, false),
            };
        }
        let i = self.pos - 2;
        self.pos += 1;
        let r = match self.cmd {
            0x52 => self.read_step(i, tx),
            0x57 => self.write_step(i, tx),
            _ => self.id_step(i),
        };
        self.prev = tx;
        r
    }

    fn read_step(&mut self, i: usize, tx: u8) -> (u8, bool) {
        let ok = self.addr < CARD_SECTORS;
        match i {
            0 => (0x5A, true),
            1 => (0x5D, true),
            2 => {
                self.addr = (tx as usize) << 8;
                (0x00, true)
            }
            3 => {
                self.addr |= tx as usize;
                (self.prev, true)
            }
            4 => (0x5C, true),
            5 => (0x5D, true),
            6 => {
                if !ok {
                    return (0xFF, false);
                }
                self.chk = (self.addr >> 8) as u8 ^ self.addr as u8;
                ((self.addr >> 8) as u8, true)
            }
            7 => (self.addr as u8, true),
            8..=135 => {
                let b = self.data[self.addr * 128 + i - 8];
                self.chk ^= b;
                (b, true)
            }
            136 => (self.chk, true),
            137 => (0x47, false),
            _ => (0xFF, false),
        }
    }

    fn write_step(&mut self, i: usize, tx: u8) -> (u8, bool) {
        match i {
            0 => (0x5A, true),
            1 => (0x5D, true),
            2 => {
                self.addr = (tx as usize) << 8;
                (0x00, true)
            }
            3 => {
                self.addr |= tx as usize;
                self.chk = (self.addr >> 8) as u8 ^ self.addr as u8;
                (self.prev, true)
            }
            4..=131 => {
                self.wbuf[i - 4] = tx;
                self.chk ^= tx;
                (self.prev, true)
            }
            132 => {
                let good = tx == self.chk;
                self.chk = if self.addr >= CARD_SECTORS {
                    0xFF
                } else if good {
                    let a = self.addr * 128;
                    self.data[a..a + 128].copy_from_slice(&self.wbuf);
                    self.flag &= !0x08;
                    self.dirty = true;
                    0x47
                } else {
                    0x4E
                };
                (self.prev, true)
            }
            133 => (0x5C, true),
            134 => (0x5D, true),
            135 => (self.chk, false),
            _ => (0xFF, false),
        }
    }

    fn id_step(&mut self, i: usize) -> (u8, bool) {
        const ID: [u8; 8] = [0x5A, 0x5D, 0x5C, 0x5D, 0x04, 0x00, 0x00, 0x80];
        match ID.get(i) {
            Some(&b) => (b, i + 1 < ID.len()),
            None => (0xFF, false),
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Dev {
    None,
    Pad,
    Card,
    /// Selected device ignored the address byte: Hi-Z, no ACK.
    Dead,
}

pub struct Sio {
    ctrl: u16,
    mode: u16,
    baud: u16,
    rx: Option<u8>,
    /// In-flight TX byte and its completion time.
    tx: Option<(u8, u64)>,
    /// Pending /ACK pulse start.
    ack_at: Option<u64>,
    ack_until: u64,
    ack_level: bool,
    irq_flag: bool,
    pub pending_irq: bool,
    pads: [Pad; 2],
    cards: [Card; 2],
    /// Pad slots with a controller plugged in.
    pub pad_connected: [bool; 2],
    pub card_inserted: [bool; 2],
    dev: Dev,
    sio1_ctrl: u16,
    sio1_mode: u16,
    sio1_baud: u16,
    /// Earliest cycle any timed event fires (fast-path for tick).
    next_event: u64,
}

impl Sio {
    pub fn new() -> Self {
        Self {
            ctrl: 0,
            mode: 0,
            baud: 0,
            rx: None,
            tx: None,
            ack_at: None,
            ack_until: 0,
            ack_level: false,
            irq_flag: false,
            pending_irq: false,
            pads: [Pad { pos: 0, buttons: 0xFFFF }, Pad { pos: 0, buttons: 0xFFFF }],
            cards: [Card::new(), Card::new()],
            pad_connected: [true, false],
            card_inserted: [true, false],
            dev: Dev::None,
            sio1_ctrl: 0,
            sio1_mode: 0,
            sio1_baud: 0,
            next_event: u64::MAX,
        }
    }

    pub fn set_buttons(&mut self, bits: u16) {
        self.pads[0].buttons = bits;
    }

    pub fn set_buttons_port(&mut self, port: usize, bits: u16) {
        self.pads[port & 1].buttons = bits;
    }

    pub fn card_image(&self, slot: usize) -> &[u8] {
        &self.cards[slot & 1].data
    }

    pub fn set_card_image(&mut self, slot: usize, img: &[u8]) {
        let c = &mut self.cards[slot & 1];
        let n = img.len().min(c.data.len());
        c.data[..n].copy_from_slice(&img[..n]);
        c.flag = 0x08;
    }

    /// True once since the last call if a card was written.
    pub fn take_card_dirty(&mut self, slot: usize) -> bool {
        std::mem::take(&mut self.cards[slot & 1].dirty)
    }

    fn slot(&self) -> usize {
        ((self.ctrl >> 13) & 1) as usize
    }

    fn reschedule(&mut self) {
        let mut n = u64::MAX;
        if let Some((_, t)) = self.tx {
            n = n.min(t);
        }
        if let Some(t) = self.ack_at {
            n = n.min(t);
        }
        if self.ack_level {
            n = n.min(self.ack_until);
        }
        self.next_event = n;
    }

    pub fn tick(&mut self, now: u64) {
        if now < self.next_event {
            return;
        }
        if let Some((b, t)) = self.tx {
            if now >= t {
                self.tx = None;
                let (reply, ack) = self.exchange(b);
                self.rx = Some(reply);
                if ack {
                    self.ack_at = Some(t + ACK_DELAY);
                }
            }
        }
        if let Some(t) = self.ack_at {
            if now >= t {
                self.ack_at = None;
                self.ack_level = true;
                self.ack_until = t + ACK_WIDTH;
                if self.ctrl & (1 << 12) != 0 && !self.irq_flag {
                    self.irq_flag = true;
                    self.pending_irq = true;
                }
            }
        }
        if self.ack_level && now >= self.ack_until {
            self.ack_level = false;
        }
        self.reschedule();
    }

    fn exchange(&mut self, tx: u8) -> (u8, bool) {
        if self.ctrl & 2 == 0 {
            return (0xFF, false);
        }
        let s = self.slot();
        if self.dev == Dev::None {
            self.dev = match tx {
                0x01 if self.pad_connected[s] => {
                    self.pads[s].pos = 0;
                    Dev::Pad
                }
                0x81 if self.card_inserted[s] => {
                    self.cards[s].select();
                    Dev::Card
                }
                _ => Dev::Dead,
            };
        }
        let r = match self.dev {
            Dev::Pad => self.pads[s].exchange(tx),
            Dev::Card => self.cards[s].exchange(tx),
            _ => (0xFF, false),
        };
        if !r.1 {
            self.dev = Dev::Dead;
        }
        r
    }

    fn stat(&self) -> u32 {
        let mut s = 1u32; // TX ready (started)
        if self.tx.is_none() {
            s |= 1 << 2; // TX finished
        }
        if self.rx.is_some() {
            s |= 1 << 1;
        }
        if self.ack_level {
            s |= 1 << 7;
        }
        if self.irq_flag {
            s |= 1 << 9;
        }
        s
    }

    fn write_ctrl(&mut self, v: u16) {
        if v & (1 << 6) != 0 {
            // reset: drop everything
            self.ctrl = 0;
            self.mode = 0;
            self.baud = 0;
            self.rx = None;
            self.tx = None;
            self.ack_at = None;
            self.ack_level = false;
            self.irq_flag = false;
            self.dev = Dev::None;
            self.reschedule();
            return;
        }
        if v & (1 << 4) != 0 {
            self.irq_flag = false;
        }
        let deselect = v & 2 == 0;
        let slot_change = (v ^ self.ctrl) & (1 << 13) != 0;
        self.ctrl = v & !((1 << 4) | (1 << 6));
        if deselect || slot_change {
            self.dev = Dev::None;
        }
    }

    pub fn read(&mut self, p: u32) -> u32 {
        match p & 0x1F {
            0x00..=0x03 => self.rx.take().map(|b| b as u32).unwrap_or(0xFF) | 0xFFFF_FF00,
            0x04..=0x07 => self.stat(),
            0x08 | 0x09 => self.mode as u32,
            0x0A | 0x0B => self.ctrl as u32,
            0x0E | 0x0F => self.baud as u32,
            0x14..=0x17 => 0x0000_0005, // SIO1 stat: TX ready/idle
            0x18 | 0x19 => self.sio1_mode as u32,
            0x1A | 0x1B => self.sio1_ctrl as u32,
            0x1E | 0x1F => self.sio1_baud as u32,
            _ => 0,
        }
    }

    pub fn write(&mut self, p: u32, v: u32, now: u64, tty: &mut String) {
        match p & 0x1F {
            0x00 => {
                let t = (self.baud.max(1) as u64) * 8;
                self.tx = Some((v as u8, now + t));
                self.reschedule();
            }
            0x08 => self.mode = v as u16,
            0x0A => self.write_ctrl(v as u16),
            0x0E => self.baud = v as u16,
            0x10 => tty.push(v as u8 as char),
            0x18 => self.sio1_mode = v as u16,
            0x1A => self.sio1_ctrl = v as u16,
            0x1E => self.sio1_baud = v as u16,
            _ => {}
        }
    }
}
