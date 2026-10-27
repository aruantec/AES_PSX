//! AruanPSX — an independent PlayStation emulator core.
//!
//! Design:
//! - Frontend-agnostic: no window/audio/input deps. The host calls
//!   [`Emulator::run_frame`], pulls the display with
//!   [`Emulator::framebuffer`] and audio with [`Emulator::take_audio`],
//!   and feeds pad state back with [`Emulator::set_pad`].
//! - Devices advance with the CPU's cycle count; the video model
//!   produces scanline/blanking edges for timers, GPUSTAT and VBlank.
//! - Save-state hooks for the warm-boot snapshot cache (`snapshot.rs`).
//!
//! Clean-room: hardware semantics from public hardware documentation
//! only; no ported code.

pub mod bus;
pub mod cdrom;
pub mod cpu;
pub mod disc;
pub mod dma;
pub mod gpu;
pub mod gte;
pub mod irq;
pub mod iso;
pub mod mdec;
pub mod sio;
pub mod snapshot;
pub mod spu;
pub mod timers;

use bus::Bus;
use cpu::Cpu;

/// Legacy fixed output size used by older frontends.
pub const FB_W: usize = 640;
pub const FB_H: usize = 480;

/// Buttons bit map (active-low, pad protocol order):
/// 0 SELECT 3 START 4 UP 5 RIGHT 6 DOWN 7 LEFT
/// 8 L2 9 R2 10 L1 11 R1 12 △ 13 ○ 14 ✕ 15 □
#[derive(Default, Clone, Copy)]
pub struct PadInput(pub u16);

pub mod buttons {
    pub const SELECT: u16 = 1 << 0;
    pub const START: u16 = 1 << 3;
    pub const UP: u16 = 1 << 4;
    pub const RIGHT: u16 = 1 << 5;
    pub const DOWN: u16 = 1 << 6;
    pub const LEFT: u16 = 1 << 7;
    pub const L2: u16 = 1 << 8;
    pub const R2: u16 = 1 << 9;
    pub const L1: u16 = 1 << 10;
    pub const R1: u16 = 1 << 11;
    pub const TRIANGLE: u16 = 1 << 12;
    pub const CIRCLE: u16 = 1 << 13;
    pub const CROSS: u16 = 1 << 14;
    pub const SQUARE: u16 = 1 << 15;
}

/// Scanline / blanking model. Positions are tracked in GPU clock
/// cycles derived from the CPU clock by the exact clock ratio.
struct Video {
    frac: u64,
    line_pos: u32,
    line: u32,
    hblank: bool,
    vblank: bool,
    dot_acc: u32,
    field: bool,
}

impl Video {
    fn new() -> Self {
        Self { frac: 0, line_pos: 0, line: 0, hblank: false, vblank: false, dot_acc: 0, field: false }
    }
}

/// PS-X EXE scheduled to replace the BIOS shell (fast boot).
struct PendingExe {
    data: Vec<u8>,
}

pub struct Emulator {
    pub cpu: Cpu,
    pub bus: Bus,
    cycles: u64,
    video: Video,
    frame_done: bool,
    frames: u64,
    cd_line: bool,
    gpu_irq_line: bool,
    fast_exe: Option<PendingExe>,
}

impl Emulator {
    /// `bios` must be a raw 512 KiB dump.
    pub fn new(bios: Vec<u8>) -> Self {
        Self {
            cpu: Cpu::new(),
            bus: Bus::new(bios),
            cycles: 0,
            video: Video::new(),
            frame_done: false,
            frames: 0,
            cd_line: false,
            gpu_irq_line: false,
            fast_exe: None,
        }
    }

    pub fn insert_disc(&mut self, d: disc::Disc) {
        self.bus.cdrom.insert(d);
    }

    /// Boot `exe` instead of the BIOS shell: the BIOS still runs its
    /// full kernel initialisation; when it hands control to the shell
    /// the executable is loaded and started in its place.
    pub fn set_fast_boot_exe(&mut self, exe: Vec<u8>) {
        self.fast_exe = Some(PendingExe { data: exe });
    }

    /// Fast-boot the disc's boot executable (skips the logo/shell).
    pub fn fast_boot_disc(&mut self) -> bool {
        let exe = {
            let Some(d) = self.bus.cdrom.disc_mut() else { return false };
            let Some(cnf) = iso::read_file(d, "SYSTEM.CNF") else { return false };
            let Some(path) = iso::boot_path(&cnf) else { return false };
            let Some(exe) = iso::read_file(d, &path) else { return false };
            exe
        };
        self.set_fast_boot_exe(exe);
        true
    }

    fn start_exe(&mut self, exe: &[u8]) -> bool {
        let Some(info) = iso::parse_exe(exe) else { return false };
        let data = &exe[info.data_off.min(exe.len())..];
        let n = (info.load_size as usize).min(data.len());
        let base = info.load_addr & 0x1F_FFFF;
        for (i, b) in data[..n].iter().enumerate() {
            self.bus.ram[(base as usize + i) & (bus::RAM_SIZE - 1)] = *b;
        }
        // memfill (bss) region
        let fill_addr = u32::from_le_bytes(exe[0x28..0x2C].try_into().unwrap());
        let fill_size = u32::from_le_bytes(exe[0x2C..0x30].try_into().unwrap());
        for i in 0..fill_size {
            self.bus.ram[((fill_addr + i) & 0x1F_FFFF) as usize] = 0;
        }
        self.cpu.set_reg(28, info.gp);
        if info.stack_addr != 0 {
            let sp = info.stack_addr.wrapping_add(info.stack_size);
            self.cpu.set_reg(29, sp);
            self.cpu.set_reg(30, sp);
        }
        self.cpu.set_pc(info.pc);
        true
    }

    pub fn set_pad(&mut self, input: PadInput) {
        self.bus.sio.set_buttons(input.0);
    }

    pub fn set_pad_port(&mut self, port: usize, input: PadInput) {
        self.bus.sio.set_buttons_port(port, input.0);
    }

    /// Drain the SPU's mixed stereo output (s16 L/R interleaved at
    /// 44.1 kHz).
    pub fn take_audio(&mut self) -> Vec<i16> {
        self.bus.spu.take_audio()
    }

    /// Bytes received on the debug TTY (SIO1/DUART + BIOS putchar).
    pub fn tty(&self) -> &str {
        &self.bus.tty
    }

    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    pub fn frame_count(&self) -> u64 {
        self.frames
    }

    /// Video refresh rate of the current display mode.
    pub fn refresh_rate(&self) -> f64 {
        if self.bus.gpu.is_pal() {
            53_222_400.0 / (3406.0 * 314.0)
        } else {
            53_693_175.0 / (3413.0 * 263.0)
        }
    }

    /// Render the current display into `out` (0x00RRGGBB); returns
    /// (width, height).
    pub fn framebuffer(&self, out: &mut Vec<u32>) -> (usize, usize) {
        self.bus.gpu.display_rgb(out)
    }

    /// Run until the next vblank.
    pub fn run_frame(&mut self) -> bool {
        self.frame_done = false;
        while !self.frame_done {
            self.step();
        }
        true
    }

    /// N-instruction helper for tests.
    pub fn run_cycles(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// One CPU instruction plus device time.
    pub fn step(&mut self) {
        self.bus.now = self.cycles;
        if self.fast_exe.is_some() && self.cpu.fetch_pc() == 0x8003_0000 {
            let exe = self.fast_exe.take().unwrap();
            self.start_exe(&exe.data);
        }
        let n = self.cpu.step(&mut self.bus);
        if !self.cpu.tty.is_empty() {
            self.bus.tty.push_str(&self.cpu.tty);
            self.cpu.tty.clear();
        }
        self.cycles += n as u64;
        self.advance(n);
    }

    fn advance(&mut self, n: u32) {
        let now = self.cycles;
        self.bus.now = now;
        self.bus.timers.tick(n);
        self.video_tick(n);

        let b = &mut self.bus;
        b.cdrom.tick(now);
        if b.cdrom.irq_line && !self.cd_line {
            b.irq.raise(irq::CDROM);
        }
        self.cd_line = b.cdrom.irq_line;

        b.spu.tick(n, &mut b.cdrom.audio);
        if b.spu.pending_irq {
            b.spu.pending_irq = false;
            b.irq.raise(irq::SPU);
        }

        b.sio.tick(now);
        if b.sio.pending_irq {
            b.sio.pending_irq = false;
            b.irq.raise(irq::PAD);
        }

        if b.gpu.irq1 && !self.gpu_irq_line {
            b.irq.raise(irq::GPU);
        }
        self.gpu_irq_line = b.gpu.irq1;

        if b.dma.pending != 0 {
            self.run_dma();
        }
        let b = &mut self.bus;
        if b.dma.busy() {
            b.dma.tick(now);
        }
        if b.dma.irq_edge {
            b.dma.irq_edge = false;
            b.irq.raise(irq::DMA);
        }

        if b.timers.pending_irq != 0 {
            b.irq.raise(b.timers.pending_irq);
            b.timers.pending_irq = 0;
        }
    }

    fn video_tick(&mut self, n: u32) {
        let pal = self.bus.gpu.is_pal();
        let v = &mut self.video;
        let (g, line_len, lines) = if pal {
            let scaled = v.frac + n as u64 * 11;
            let clocks = (scaled / 7) as u32;
            v.frac = scaled % 7;
            (clocks, 3406, 314)
        } else {
            let scaled = v.frac + n as u64 * 715_909;
            let clocks = (scaled / 451_584) as u32;
            v.frac = scaled % 451_584;
            (clocks, 3413, 263)
        };

        // dot clock for timer 0
        let div = self.bus.gpu.dot_divider();
        v.dot_acc += g;
        let dots = match div {
            4 => {
                let dots = v.dot_acc >> 2;
                v.dot_acc &= 3;
                dots
            }
            5 => {
                let dots = v.dot_acc / 5;
                v.dot_acc %= 5;
                dots
            }
            7 => {
                let dots = v.dot_acc / 7;
                v.dot_acc %= 7;
                dots
            }
            8 => {
                let dots = v.dot_acc >> 3;
                v.dot_acc &= 7;
                dots
            }
            10 => {
                let dots = v.dot_acc / 10;
                v.dot_acc %= 10;
                dots
            }
            _ => unreachable!("unsupported GPU dot divider {div}"),
        };
        if dots > 0 {
            self.bus.timers.dots(dots);
        }

        v.line_pos += g;
        // hblank begins after the visible portion of the line
        const HBLANK_START: u32 = 2560;
        if !v.hblank && v.line_pos >= HBLANK_START {
            v.hblank = true;
            self.bus.timers.set_hblank(true);
        }
        while v.line_pos >= line_len {
            v.line_pos -= line_len;
            v.hblank = false;
            self.bus.timers.set_hblank(false);
            v.line += 1;
            if v.line >= lines {
                v.line = 0;
                v.field = !v.field;
            }
            if v.line_pos >= HBLANK_START {
                v.hblank = true;
                self.bus.timers.set_hblank(true);
            }
            let gpu = &mut self.bus.gpu;
            let (mut top, mut bot) = (gpu.v_range.0 as u32, gpu.v_range.1 as u32);
            if bot <= top || bot > lines {
                (top, bot) = if pal { (0x23, 0x23 + 288) } else { (0x10, 0x100) };
            }
            let in_vb = v.line < top || v.line >= bot;
            gpu.field_odd = gpu.interlaced() && v.field;
            gpu.odd_line = if gpu.interlaced() { v.field } else { v.line & 1 != 0 };
            if in_vb != v.vblank {
                v.vblank = in_vb;
                gpu.vblank = in_vb;
                self.bus.timers.set_vblank(in_vb);
                if in_vb {
                    self.bus.irq.raise(irq::VBLANK);
                    self.frame_done = true;
                    self.frames += 1;
                }
            }
        }
    }

    fn run_dma(&mut self) {
        for ch in 0..7 {
            if self.bus.dma.pending & (1 << ch) == 0 {
                continue;
            }
            if let Some(cost) = self.dma_channel(ch) {
                let at = self.cycles + cost as u64;
                self.bus.dma.started(ch, at);
            }
        }
    }

    /// Move one channel's data. Returns the bus time it took, or None
    /// if the device isn't ready yet (channel stays pending).
    fn dma_channel(&mut self, ch: usize) -> Option<u32> {
        let c = self.bus.dma.ch[ch];
        let b = &mut self.bus;
        let step = c.step();
        if c.mode() == dma::Mode::LinkedList {
            if ch != 2 {
                return Some(1);
            }
            let mut a = c.madr & 0x1F_FFFC;
            let mut cost = 0u32;
            for _ in 0..0x10_0000 {
                let hdr = u32::from_le_bytes(b.ram[a as usize..a as usize + 4].try_into().unwrap());
                let count = hdr >> 24;
                for i in 0..count {
                    let p = ((a + 4 + i * 4) & 0x1F_FFFC) as usize;
                    let w = u32::from_le_bytes(b.ram[p..p + 4].try_into().unwrap());
                    b.gpu.gp0(w);
                }
                cost += count + 1;
                if hdr & 0x80_0000 != 0 {
                    break;
                }
                a = hdr & 0x1F_FFFC;
            }
            b.dma.ch[ch].madr = 0xFF_FFFF;
            return Some(cost);
        }
        let words = b.dma.word_count(ch);
        if ch == 1 && b.mdec.out_len() < words as usize {
            return None;
        }
        let mut a = c.madr & 0x1F_FFFC;
        let rd = |b: &Bus, a: u32| u32::from_le_bytes(b.ram[a as usize..a as usize + 4].try_into().unwrap());
        for i in 0..words {
            if ch == 6 {
                let v = if i + 1 == words { 0xFF_FFFF } else { a.wrapping_sub(4) & 0x1F_FFFF };
                b.ram[a as usize..a as usize + 4].copy_from_slice(&v.to_le_bytes());
            } else if c.to_ram() {
                let v = match ch {
                    1 => b.mdec.read_data(),
                    2 => b.gpu.read_data(),
                    3 => b.cdrom.dma_read(),
                    4 => b.spu.dma_read(),
                    _ => 0,
                };
                b.ram[a as usize..a as usize + 4].copy_from_slice(&v.to_le_bytes());
            } else {
                let v = rd(b, a);
                match ch {
                    0 => b.mdec.write_cmd(v),
                    2 => b.gpu.gp0(v),
                    4 => b.spu.dma_write(v),
                    _ => {}
                }
            }
            a = a.wrapping_add(step) & 0x1F_FFFC;
        }
        b.dma.ch[ch].madr = a;
        if c.mode() == dma::Mode::Slice {
            b.dma.ch[ch].bcr &= 0xFFFF;
        }
        Some(words)
    }

    // ---- snapshot hooks (see snapshot.rs) ----
    pub fn state(&self) -> snapshot::State {
        snapshot::State::capture(self)
    }
    pub fn restore(&mut self, st: &snapshot::State) {
        snapshot::State::apply(self, st);
    }
}
