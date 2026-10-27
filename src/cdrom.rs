//! CD-ROM controller: 4 byte-wide registers multiplexed by an index.
//!
//!   0x1F801800  index (w) / status (r)
//!   0x1F801801  command, sound-map regs (w) / response FIFO (r)
//!   0x1F801802  parameter FIFO, int-enable, volumes (w) / data FIFO (r)
//!   0x1F801803  request, int-ack, volumes (w) / int enable/flags (r)
//!
//! Interrupts are delivered one at a time: a pending reply waits
//! until software acknowledges the current one. Data sectors land in
//! a sector buffer; writing request.BFRD copies the buffer into the
//! data FIFO which DMA3 (or CPU reads of 0x1802) drain.
//!
//! XA-ADPCM sectors are decoded here and, together with CD-DA
//! playback, feed `CdAudio` — a 44.1 kHz stereo queue the SPU mixes.

use std::collections::VecDeque;

use crate::disc::{Disc, SECTOR_RAW};

const ST_ERROR: u8 = 0x01;
const ST_MOTOR: u8 = 0x02;
const ST_READ: u8 = 0x20;
const ST_SEEK: u8 = 0x40;
const ST_PLAY: u8 = 0x80;

/// Sysclks per sector at 1x (75 sectors/s).
const SECTOR_1X: u64 = 33_868_800 / 75;
/// Command → first response (INT3) latency.
const ACK_DELAY: u64 = 25_000;
/// INT3 → second response latency for quick commands.
const SECOND_DELAY: u64 = 40_000;

#[derive(Clone)]
struct Reply {
    at: u64,
    int: u8,
    bytes: Vec<u8>,
    /// For INT1 data replies: the sector to load into the buffer.
    sector: Option<Box<[u8; SECTOR_RAW]>>,
    /// Background reply (second response / sector) — cancelled by
    /// commands that abort the drive's current activity.
    asynch: bool,
}

/// 44.1 kHz stereo CD audio stream (XA-ADPCM + CD-DA) for the SPU.
pub struct CdAudio {
    pub fifo: VecDeque<(i16, i16)>,
    /// Applied volume matrix: LL, LR, RR, RL (0x80 = 100%).
    vol: [u8; 4],
    muted: bool,
    /// XA decoder history per channel + resampler state.
    xa_hist: [[i32; 2]; 2],
    rs_phase: u32,
    rs_prev: (i32, i32),
}

impl CdAudio {
    fn new() -> Self {
        Self {
            fifo: VecDeque::with_capacity(16384),
            vol: [0x80, 0, 0x80, 0],
            muted: false,
            xa_hist: [[0; 2]; 2],
            rs_phase: 0,
            rs_prev: (0, 0),
        }
    }

    /// One output frame, volume-mixed (silence when starved).
    pub fn pop(&mut self) -> (i32, i32) {
        let Some((l, r)) = self.fifo.pop_front() else {
            return (0, 0);
        };
        if self.muted {
            return (0, 0);
        }
        let (l, r) = (l as i32, r as i32);
        let [ll, lr, rr, rl] = self.vol.map(|v| v as i32);
        let ol = (l * ll + r * rl) >> 7;
        let or = (l * lr + r * rr) >> 7;
        (ol.clamp(-0x8000, 0x7FFF), or.clamp(-0x8000, 0x7FFF))
    }

    fn push(&mut self, l: i32, r: i32) {
        if self.fifo.len() < 44_100 / 2 {
            self.fifo.push_back((l.clamp(-0x8000, 0x7FFF) as i16, r.clamp(-0x8000, 0x7FFF) as i16));
        }
    }

    /// Resample a 37.8 kHz stereo frame stream to 44.1 kHz (ratio
    /// 7:6) by linear interpolation between consecutive input frames.
    fn push_37800(&mut self, frames: &[(i32, i32)]) {
        for &(l, r) in frames {
            // emit outputs whose position falls within (prev, cur]
            while self.rs_phase < 7 {
                let t = self.rs_phase as i32;
                let (pl, pr) = self.rs_prev;
                self.push(pl + (l - pl) * t / 7, pr + (r - pr) * t / 7);
                self.rs_phase += 6;
            }
            self.rs_phase -= 7;
            self.rs_prev = (l, r);
        }
    }

    fn reset_xa(&mut self) {
        self.xa_hist = [[0; 2]; 2];
        self.rs_phase = 0;
        self.rs_prev = (0, 0);
    }
}

const XA_POS: [i32; 4] = [0, 60, 115, 98];
const XA_NEG: [i32; 4] = [0, 0, -52, -55];

pub struct CdRom {
    index: u8,
    params: VecDeque<u8>,
    resp: VecDeque<u8>,
    queue: VecDeque<Reply>,
    int_flag: u8,
    int_enable: u8,
    request: u8,
    /// Pending volume matrix (applied by 0x1803.idx3 bit5).
    vol_pending: [u8; 4],
    mode: u8,
    stat: u8,
    /// Setloc target and whether it has been consumed by a seek.
    loc: u32,
    loc_pending: bool,
    /// Head position (next sector to read/play).
    head: u32,
    reading: bool,
    playing: bool,
    next_sector_at: u64,
    filter_file: u8,
    filter_chan: u8,
    /// Last sector delivered (sector buffer) and its header bytes.
    sector: Box<[u8; SECTOR_RAW]>,
    last_hdr: [u8; 8],
    data: VecDeque<u8>,
    disc: Option<Disc>,
    /// Console region letter reported by GetID ('A', 'E', 'I').
    pub region: u8,
    /// Track end for CD-DA autopause.
    play_end: u32,
    report_ctr: u32,
    pub audio: CdAudio,
    /// 0x1803.idx3 bit0: XA-ADPCM muted.
    adp_mute: bool,
    /// Level of the controller's IRQ output.
    pub irq_line: bool,
    pub now: u64,
}

#[inline]
fn bcd(v: u32) -> u8 {
    (((v / 10) % 10) << 4 | (v % 10)) as u8
}
#[inline]
fn unbcd(b: u8) -> u32 {
    ((b >> 4) * 10 + (b & 0xF)) as u32
}
fn msf(lba: u32) -> [u8; 3] {
    [bcd(lba / 4500), bcd((lba / 75) % 60), bcd(lba % 75)]
}

impl CdRom {
    pub fn new() -> Self {
        Self {
            index: 0,
            params: VecDeque::new(),
            resp: VecDeque::new(),
            queue: VecDeque::new(),
            int_flag: 0,
            int_enable: 0,
            request: 0,
            vol_pending: [0x80, 0, 0x80, 0],
            mode: 0,
            stat: ST_MOTOR,
            loc: 0,
            loc_pending: false,
            head: 0,
            reading: false,
            playing: false,
            next_sector_at: 0,
            filter_file: 0,
            filter_chan: 0,
            sector: Box::new([0; SECTOR_RAW]),
            last_hdr: [0; 8],
            data: VecDeque::with_capacity(2352),
            disc: None,
            region: b'A',
            play_end: u32::MAX,
            report_ctr: 0,
            audio: CdAudio::new(),
            adp_mute: false,
            irq_line: false,
            now: 0,
        }
    }

    pub fn insert(&mut self, d: Disc) {
        self.disc = Some(d);
        self.stat = ST_MOTOR;
    }

    pub fn disc_mut(&mut self) -> Option<&mut Disc> {
        self.disc.as_mut()
    }

    pub fn has_disc(&self) -> bool {
        self.disc.is_some()
    }

    fn sector_period(&self) -> u64 {
        if self.mode & 0x80 != 0 { SECTOR_1X / 2 } else { SECTOR_1X }
    }

    /// Absolute disc LBA (relative to 00:02:00) → disc-image LBA.
    fn track_of(&self, lba: u32) -> (u32, u32) {
        let mut hit = (1u32, 0u32);
        if let Some(d) = &self.disc {
            for (i, t) in d.tracks.iter().enumerate() {
                if t.index_lba <= lba {
                    hit = (i as u32 + 1, t.index_lba);
                }
            }
        }
        hit
    }

    fn track_count(&self) -> u32 {
        self.disc.as_ref().map(|d| d.tracks.len() as u32).unwrap_or(0)
    }

    fn disc_end(&self) -> u32 {
        self.disc.as_ref().map(|d| d.total_sectors()).unwrap_or(0)
    }

    fn push(&mut self, at: u64, int: u8, bytes: Vec<u8>, asynch: bool) {
        self.queue.push_back(Reply { at, int, bytes, sector: None, asynch });
    }

    fn abort_async(&mut self) {
        self.queue.retain(|r| !r.asynch);
    }

    fn seek_time(&self, target: u32) -> u64 {
        let d = (target as i64 - self.head as i64).unsigned_abs();
        if d < 16 {
            self.sector_period()
        } else {
            (60_000 + d * 12).min(1_400_000)
        }
    }

    /// Advance the controller to `now`.
    pub fn tick(&mut self, now: u64) {
        self.now = now;
        if (self.reading || self.playing) && now >= self.next_sector_at {
            let at = self.next_sector_at;
            self.next_sector_at = at + self.sector_period();
            if self.reading {
                self.read_sector(at);
            } else {
                self.play_sector(at);
            }
        }
        if self.int_flag == 0 {
            if let Some(front) = self.queue.front() {
                if front.at <= now {
                    let r = self.queue.pop_front().unwrap();
                    self.resp.clear();
                    self.resp.extend(r.bytes.iter().copied());
                    if let Some(s) = r.sector {
                        self.sector = s;
                    }
                    self.int_flag = r.int;
                }
            }
        }
        self.irq_line = self.int_flag & self.int_enable & 0x1F != 0;
    }

    fn read_sector(&mut self, at: u64) {
        self.stat = (self.stat & !(ST_SEEK | ST_PLAY)) | ST_READ | ST_MOTOR;
        let lba = self.head;
        self.head += 1;
        let Some(sec) = self.disc.as_mut().and_then(|d| d.read_sector_raw(lba)) else {
            self.reading = false;
            self.stat = (self.stat & !ST_READ) | ST_ERROR;
            self.push(at, 5, vec![self.stat, 0x04], true);
            self.stat &= !ST_ERROR;
            return;
        };
        let submode = sec[18];
        let (file, chan) = (sec[16], sec[17]);
        let xa_enabled = self.mode & 0x40 != 0;
        let filter = self.mode & 0x08 != 0;
        let is_audio = submode & 0x04 != 0 && sec[15] == 2;
        if xa_enabled && is_audio {
            if !self.adp_mute && (!filter || (file == self.filter_file && chan == self.filter_chan)) {
                self.decode_xa(&sec);
            }
            return;
        }
        self.last_hdr.copy_from_slice(&sec[12..20]);
        // newer sector replaces an undelivered older one
        if let Some(pos) = self.queue.iter().position(|r| r.int == 1 && r.sector.is_some()) {
            self.queue.remove(pos);
        }
        self.queue.push_back(Reply {
            at,
            int: 1,
            bytes: vec![self.stat],
            sector: Some(Box::new(sec)),
            asynch: true,
        });
    }

    fn play_sector(&mut self, at: u64) {
        let lba = self.head;
        if lba >= self.play_end && self.mode & 0x02 != 0 {
            // autopause at end of track
            self.playing = false;
            self.stat &= !ST_PLAY;
            self.push(at, 4, vec![self.stat], true);
            return;
        }
        self.head += 1;
        let Some(sec) = self.disc.as_mut().and_then(|d| d.read_sector_raw(lba)) else {
            self.playing = false;
            self.stat &= !ST_PLAY;
            self.push(at, 4, vec![self.stat], true);
            return;
        };
        let is_audio = self.disc.as_ref().map(|d| d.is_audio(lba)).unwrap_or(false);
        if is_audio {
            for i in 0..SECTOR_RAW / 4 {
                let l = i16::from_le_bytes([sec[i * 4], sec[i * 4 + 1]]) as i32;
                let r = i16::from_le_bytes([sec[i * 4 + 2], sec[i * 4 + 3]]) as i32;
                self.audio.push(l, r);
            }
        }
        if self.mode & 0x04 != 0 {
            // report: every 10th sector (alternating abs/relative)
            self.report_ctr += 1;
            if self.report_ctr % 10 == 0 {
                let (tn, tstart) = self.track_of(lba);
                let pos = lba + 150;
                let rel = lba.saturating_sub(tstart);
                let (m, s, f) = if (self.report_ctr / 10) % 2 == 0 {
                    let [m, s, f] = msf(pos);
                    (m, s, f)
                } else {
                    let [m, s, f] = msf(rel);
                    (m, s | 0x80, f)
                };
                self.push(at, 1, vec![self.stat, bcd(tn), 1, m, s, f, 0, 0], true);
            }
        }
    }

    /// Decode one XA-ADPCM sector into the CD audio stream.
    fn decode_xa(&mut self, sec: &[u8; SECTOR_RAW]) {
        let ci = sec[19];
        let stereo = ci & 0x01 != 0;
        let half_rate = ci & 0x04 != 0;
        let eight_bit = ci & 0x10 != 0;
        let data = &sec[24..24 + 18 * 128];
        let mut left: Vec<i32> = Vec::with_capacity(4032);
        let mut right: Vec<i32> = Vec::with_capacity(2016);
        let blocks = if eight_bit { 4 } else { 8 };
        for portion in data.chunks_exact(128) {
            for blk in 0..blocks {
                let hdr = portion[4 + blk];
                let mut shift = (hdr & 0x0F) as u32;
                if shift > 12 {
                    shift = 9;
                }
                let filt = ((hdr >> 4) & 3) as usize;
                let ch = if stereo { blk & 1 } else { 0 };
                let [mut old, mut older] = self.audio.xa_hist[ch];
                for j in 0..28 {
                    let raw = if eight_bit {
                        (portion[16 + j * 4 + blk] as i8 as i32) << 8
                    } else {
                        let b = portion[16 + j * 4 + blk / 2];
                        let nib = if blk & 1 == 0 { b & 0x0F } else { b >> 4 };
                        ((nib as i8) << 4) as i32 >> 4 << 12
                    };
                    let t = raw >> shift;
                    let s = (t + ((old * XA_POS[filt] + older * XA_NEG[filt] + 32) >> 6))
                        .clamp(-0x8000, 0x7FFF);
                    older = old;
                    old = s;
                    if stereo && ch == 1 {
                        right.push(s);
                    } else {
                        left.push(s);
                    }
                }
                self.audio.xa_hist[ch] = [old, older];
            }
        }
        let frames: Vec<(i32, i32)> = if stereo {
            left.iter().zip(right.iter()).map(|(&l, &r)| (l, r)).collect()
        } else {
            left.iter().map(|&s| (s, s)).collect()
        };
        if half_rate {
            let doubled: Vec<(i32, i32)> = frames.iter().flat_map(|&f| [f, f]).collect();
            self.audio.push_37800(&doubled);
        } else {
            self.audio.push_37800(&frames);
        }
    }

    pub fn dma_read(&mut self) -> u32 {
        let mut b = [0u8; 4];
        for x in &mut b {
            *x = self.data.pop_front().unwrap_or(0);
        }
        u32::from_le_bytes(b)
    }

    fn start_read(&mut self, now: u64) {
        let target = if self.loc_pending { self.loc } else { self.head };
        let seek = if self.loc_pending { self.seek_time(target) } else { 0 };
        self.loc_pending = false;
        self.head = target;
        self.reading = true;
        self.playing = false;
        self.stat = (self.stat | ST_MOTOR) & !ST_PLAY;
        if seek > 0 {
            self.stat |= ST_SEEK;
        }
        self.next_sector_at = now + seek + self.sector_period();
    }

    fn dispatch(&mut self, cmd: u8, now: u64) {
        let params: Vec<u8> = self.params.drain(..).collect();
        let p = |i: usize| params.get(i).copied().unwrap_or(0);
        let t = now + ACK_DELAY;
        let needs_disc = matches!(cmd, 0x02..=0x09 | 0x0B..=0x0D | 0x10..=0x16 | 0x1A | 0x1B | 0x1D | 0x1E);
        if needs_disc && self.disc.is_none() {
            self.push(t, 5, vec![0x11, 0x80], false);
            return;
        }
        let stat = self.stat;
        match cmd {
            0x01 => self.push(t, 3, vec![stat], false),
            0x02 => {
                self.loc = (unbcd(p(0)) * 4500 + unbcd(p(1)) * 75 + unbcd(p(2))).saturating_sub(150);
                self.loc_pending = true;
                self.push(t, 3, vec![stat], false);
            }
            0x03 => {
                // Play [track]
                self.abort_async();
                let tr = unbcd(p(0));
                if tr >= 1 && tr <= self.track_count() {
                    let start = self.disc.as_ref().unwrap().tracks[tr as usize - 1].index_lba;
                    self.head = start;
                    self.loc_pending = false;
                } else if self.loc_pending {
                    self.head = self.loc;
                    self.loc_pending = false;
                }
                let (tn, _) = self.track_of(self.head);
                self.play_end = self
                    .disc
                    .as_ref()
                    .and_then(|d| d.tracks.get(tn as usize).map(|t| t.index_lba))
                    .unwrap_or(u32::MAX);
                self.reading = false;
                self.playing = true;
                self.report_ctr = 0;
                self.stat = (self.stat | ST_PLAY | ST_MOTOR) & !(ST_READ | ST_SEEK);
                self.next_sector_at = now + self.sector_period();
                self.push(t, 3, vec![self.stat], false);
            }
            0x04 | 0x05 => self.push(t, 3, vec![stat], false),
            0x06 | 0x1B => {
                self.abort_async();
                self.audio.reset_xa();
                self.push(t, 3, vec![stat], false);
                self.start_read(now);
            }
            0x07 => {
                self.push(t, 3, vec![stat], false);
                self.stat |= ST_MOTOR;
                self.push(t + SECOND_DELAY, 2, vec![self.stat], true);
            }
            0x08 => {
                self.abort_async();
                self.reading = false;
                self.playing = false;
                self.stat &= !(ST_READ | ST_PLAY | ST_SEEK);
                self.push(t, 3, vec![self.stat], false);
                self.stat &= !ST_MOTOR;
                let d = if stat & ST_MOTOR != 0 { 0x0D38ACA } else { 0x0001D7B };
                self.push(t + d, 2, vec![self.stat], true);
            }
            0x09 => {
                self.abort_async();
                let busy = self.reading || self.playing;
                self.push(t, 3, vec![stat], false);
                self.reading = false;
                self.playing = false;
                self.stat &= !(ST_READ | ST_PLAY | ST_SEEK);
                let d = if !busy {
                    0x1DF2
                } else if self.mode & 0x80 != 0 {
                    0x10BD93
                } else {
                    0x21181C
                };
                self.push(t + d, 2, vec![self.stat], true);
            }
            0x0A => {
                self.abort_async();
                self.reading = false;
                self.playing = false;
                self.mode = 0x20;
                self.push(t, 3, vec![stat], false);
                self.stat = ST_MOTOR;
                self.push(t + SECOND_DELAY * 2, 2, vec![self.stat], true);
            }
            0x0B => {
                self.audio.muted = true;
                self.push(t, 3, vec![stat], false);
            }
            0x0C => {
                self.audio.muted = false;
                self.push(t, 3, vec![stat], false);
            }
            0x0D => {
                self.filter_file = p(0);
                self.filter_chan = p(1);
                self.push(t, 3, vec![stat], false);
            }
            0x0E => {
                self.mode = p(0);
                self.push(t, 3, vec![stat], false);
            }
            0x0F => self.push(t, 3, vec![stat, self.mode, 0, self.filter_file, self.filter_chan], false),
            0x10 => self.push(t, 3, self.last_hdr.to_vec(), false),
            0x11 => {
                let lba = self.head;
                let (tn, tstart) = self.track_of(lba);
                let rel = msf(lba.saturating_sub(tstart));
                let abs = msf(lba + 150);
                self.push(t, 3, vec![bcd(tn), 1, rel[0], rel[1], rel[2], abs[0], abs[1], abs[2]], false);
            }
            0x12 => {
                self.push(t, 3, vec![stat], false);
                self.push(t + SECOND_DELAY, 2, vec![self.stat], true);
            }
            0x13 => self.push(t, 3, vec![stat, 1, bcd(self.track_count())], false),
            0x14 => {
                let tr = unbcd(p(0));
                if tr > self.track_count() {
                    self.push(t, 5, vec![stat | ST_ERROR, 0x10], false);
                } else {
                    let lba = if tr == 0 {
                        self.disc_end()
                    } else {
                        self.disc.as_ref().unwrap().tracks[tr as usize - 1].index_lba
                    } + 150;
                    let m = msf(lba);
                    self.push(t, 3, vec![stat, m[0], m[1]], false);
                }
            }
            0x15 | 0x16 => {
                self.abort_async();
                self.reading = false;
                self.playing = false;
                let target = self.loc;
                let d = self.seek_time(target);
                self.loc_pending = false;
                self.head = target;
                self.stat = (self.stat | ST_MOTOR | ST_SEEK) & !(ST_READ | ST_PLAY);
                self.push(t, 3, vec![self.stat], false);
                self.stat &= !ST_SEEK;
                self.push(t + d, 2, vec![self.stat], true);
            }
            0x19 => match p(0) {
                0x20 => self.push(t, 3, vec![0x94, 0x09, 0x19, 0xC0], false),
                _ => self.push(t, 3, vec![stat], false),
            },
            0x1A => {
                self.push(t, 3, vec![stat], false);
                let audio_only = self.disc.as_ref().map(|d| d.is_audio(0)).unwrap_or(false);
                if audio_only {
                    self.push(t + SECOND_DELAY, 5, vec![stat | 0x08, 0x90, 0, 0, 0, 0, 0, 0], true);
                } else {
                    let r = self.region;
                    self.push(t + SECOND_DELAY, 2, vec![stat, 0x00, 0x20, 0x00, b'S', b'C', b'E', r], true);
                }
            }
            0x1C => {
                self.abort_async();
                self.reading = false;
                self.playing = false;
                self.push(t, 3, vec![stat], false);
            }
            0x1D => {
                self.push(t, 3, vec![stat], false);
                self.push(t + SECOND_DELAY, 2, vec![0; 10], true);
            }
            0x1E => {
                self.push(t, 3, vec![stat], false);
                self.push(t + SECOND_DELAY * 8, 2, vec![self.stat], true);
            }
            _ => self.push(t, 5, vec![stat | ST_ERROR, 0x40], false),
        }
    }

    // ---------------- register interface ----------------

    pub fn read(&mut self, p: u32) -> u8 {
        match p & 3 {
            0 => {
                let mut s = self.index & 3;
                if self.params.is_empty() {
                    s |= 1 << 3;
                }
                if self.params.len() < 16 {
                    s |= 1 << 4;
                }
                if !self.resp.is_empty() {
                    s |= 1 << 5;
                }
                if !self.data.is_empty() {
                    s |= 1 << 6;
                }
                if self.queue.iter().any(|r| !r.asynch) {
                    s |= 1 << 7;
                }
                s
            }
            1 => self.resp.pop_front().unwrap_or(0),
            2 => self.data.pop_front().unwrap_or(0),
            _ => match self.index & 1 {
                0 => self.int_enable | 0xE0,
                _ => self.int_flag | 0xE0,
            },
        }
    }

    pub fn write(&mut self, p: u32, v: u8) {
        match (p & 3, self.index) {
            (0, _) => self.index = v & 3,
            (1, 0) => self.dispatch(v, self.now),
            (1, 3) => self.vol_pending[2] = v,
            (1, _) => {}
            (2, 0) => {
                if self.params.len() < 16 {
                    self.params.push_back(v)
                }
            }
            (2, 1) => self.int_enable = v & 0x1F,
            (2, 2) => self.vol_pending[0] = v,
            (2, 3) => self.vol_pending[3] = v,
            (3, 0) => {
                if v & 0x80 != 0 {
                    if self.data.is_empty() {
                        let (off, len) = if self.mode & 0x20 != 0 { (12, 2340) } else { (24, 2048) };
                        self.data.extend(self.sector[off..off + len].iter().copied());
                    }
                } else {
                    self.data.clear();
                }
                self.request = v;
            }
            (3, 1) => {
                self.int_flag &= !(v & 0x1F);
                if v & 0x40 != 0 {
                    self.params.clear();
                }
            }
            (3, 2) => self.vol_pending[1] = v,
            (3, 3) => {
                self.adp_mute = v & 1 != 0;
                if v & 0x20 != 0 {
                    self.audio.vol = self.vol_pending;
                }
            }
            _ => {}
        }
        self.irq_line = self.int_flag & self.int_enable & 0x1F != 0;
    }
}
