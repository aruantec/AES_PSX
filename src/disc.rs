//! CD image layer — cue+bin, raw 2352 or 2048-byte data tracks.
//!
//! Keeps file handles open and reads sectors lazily; hashing covers
//! the TOC plus a sampling of data sectors, which is cheap enough to
//! run on every load (part of the instant-boot cache key).
//!
//! Multi-file cues (one .bin per track) give INDEX times relative to
//! their own file. We resolve them into absolute disc LBAs by
//! accumulating each file's sector count: file N's sector 0 sits at
//! disc LBA = total sectors of files 0..N-1.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const SECTOR_RAW: usize = 2352;
pub const SECTOR_DATA: usize = 2048;

/// mm:ss:ff (75 fps) → sector index.
#[inline]
pub fn msf_to_lba(m: u32, s: u32, f: u32) -> u32 {
    m * 60 * 75 + s * 75 + f
}

#[derive(Clone, Debug)]
pub struct Track {
    /// Absolute disc LBA of the INDEX 01 position.
    pub index_lba: u32,
    pub bin_path: PathBuf,
    /// Absolute disc LBA of this file's sector 0. Sector `lba` lives
    /// at file offset `(lba - file_lba_start) * sector_size`.
    pub file_lba_start: u32,
    /// 2352 = raw, 2048 = .iso-style data track.
    pub sector_size: usize,
    pub audio: bool,
}

pub struct Disc {
    pub tracks: Vec<Track>,
    /// One handle per distinct bin file.
    files: Vec<File>,
    /// `file_of[i]` indexes `files` for `tracks[i]`.
    file_of: Vec<usize>,
}

impl Disc {
    /// Load a `.cue` (or bare `.bin` treated as one MODE2 track).
    pub fn load(cue_path: &Path) -> std::io::Result<Self> {
        if cue_path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("bin"))
            .unwrap_or(false)
        {
            let f = File::open(cue_path)?;
            return Ok(Self {
                tracks: vec![Track {
                    index_lba: 0,
                    bin_path: cue_path.to_path_buf(),
                    file_lba_start: 0,
                    sector_size: SECTOR_RAW,
                    audio: false,
                }],
                files: vec![f],
                file_of: vec![0],
            });
        }

        // Pass 1: collect raw records — INDEX values are file-relative.
        struct Raw {
            file: PathBuf,
            idx1_rel: u32,
            fmt: usize,
            audio: bool,
        }
        let text = std::fs::read_to_string(cue_path)?;
        let dir = cue_path.parent().unwrap_or(Path::new("."));
        let mut raw: Vec<Raw> = Vec::new();
        let mut cur_file: Option<PathBuf> = None;
        // in-progress track: (audio, fmt, index01-rel); u32::MAX = no index yet
        let mut pending: Option<(bool, usize, u32)> = None;

        macro_rules! finish {
            () => {
                if let (Some(bp), Some((au, fmt, idx))) =
                    (cur_file.clone(), pending.take())
                {
                    if idx != u32::MAX {
                        raw.push(Raw {
                            file: bp,
                            idx1_rel: idx,
                            fmt,
                            audio: au,
                        });
                    }
                }
            };
        }

        for line in text.lines() {
            let l = line.trim();
            let up = l.to_ascii_uppercase();
            if up.starts_with("FILE ") {
                finish!(); // a new FILE closes any track still open
                if let (Some(a), Some(b)) =
                    (l.find('"').map(|i| i + 1), l.rfind('"'))
                {
                    cur_file = Some(dir.join(&l[a..b]));
                }
            } else if up.starts_with("TRACK ") {
                finish!();
                let audio = up.contains("AUDIO");
                let fmt = if up.contains("MODE1/2048") {
                    2048
                } else {
                    SECTOR_RAW
                };
                pending = Some((audio, fmt, u32::MAX));
            } else if up.starts_with("INDEX 01") {
                if let (Some(msf), Some((_, _, idx))) =
                    (l.rsplit(' ').next(), pending.as_mut())
                {
                    if let Some(v) = parse_msf(msf) {
                        *idx = v;
                    }
                }
            }
        }
        finish!();

        if raw.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cue parsed zero tracks",
            ));
        }

        // Pass 2: open each distinct bin, get its sector count, then
        // assign every track an absolute LBA. File i's sector 0 sits
        // at disc LBA = sum of previous files' sector counts.
        let mut files: Vec<PathBuf> = Vec::new();
        let mut file_of = Vec::with_capacity(raw.len());
        for t in &raw {
            let i = match files.iter().position(|p| *p == t.file) {
                Some(i) => i,
                None => {
                    files.push(t.file.clone());
                    files.len() - 1
                }
            };
            file_of.push(i);
        }
        let mut fvec = Vec::with_capacity(files.len());
        let mut cum = Vec::with_capacity(files.len());
        let mut acc = 0u32;
        for p in &files {
            let f = File::open(p)?;
            cum.push(acc);
            acc += (f.metadata()?.len() / SECTOR_RAW as u64) as u32;
            fvec.push(f);
        }
        let mut tracks = Vec::with_capacity(raw.len());
        for (i, t) in raw.iter().enumerate() {
            tracks.push(Track {
                index_lba: cum[file_of[i]] + t.idx1_rel,
                bin_path: t.file.clone(),
                file_lba_start: cum[file_of[i]],
                sector_size: t.fmt,
                audio: t.audio,
            });
        }
        Ok(Self {
            tracks,
            files: fvec,
            file_of,
        })
    }

    /// Total sectors on the disc (end of the last file).
    pub fn total_sectors(&self) -> u32 {
        match (self.tracks.last(), self.file_of.last()) {
            (Some(t), Some(&fi)) => {
                let len = self.files[fi].metadata().map(|m| m.len()).unwrap_or(0);
                t.file_lba_start + (len / t.sector_size as u64) as u32
            }
            _ => 0,
        }
    }

    /// Whether `lba` lies inside an audio track.
    pub fn is_audio(&self, lba: u32) -> bool {
        self.tracks
            .iter()
            .rev()
            .find(|t| lba >= t.file_lba_start.min(t.index_lba))
            .map(|t| t.audio)
            .unwrap_or(false)
    }

    /// Read a raw 2352-byte sector by absolute disc LBA. The track is
    /// chosen by file coverage (`file_lba_start`), not INDEX 01, so
    /// pregap sectors read correctly.
    pub fn read_sector_raw(&mut self, lba: u32) -> Option<[u8; SECTOR_RAW]> {
        let (track, fi) = self
            .tracks
            .iter()
            .zip(self.file_of.iter())
            .rev()
            .find(|(t, _)| lba >= t.file_lba_start)
            .map(|(t, i)| (t, *i))?;
        let off = (lba - track.file_lba_start) as u64 * track.sector_size as u64;
        let mut buf = [0u8; SECTOR_RAW];
        let cooked = track.sector_size == SECTOR_DATA;
        let f = &mut self.files[fi];
        f.seek(SeekFrom::Start(off)).ok()?;
        if cooked {
            // 2048-byte image: synthesize a MODE2/form1 frame around
            // the user data so the controller sees a normal sector.
            f.read_exact(&mut buf[24..24 + SECTOR_DATA]).ok()?;
            buf[1..11].fill(0xFF);
            let abs = lba + 150;
            let b = |v: u32| (((v / 10) << 4) | (v % 10)) as u8;
            buf[12] = b(abs / 4500);
            buf[13] = b((abs / 75) % 60);
            buf[14] = b(abs % 75);
            buf[15] = 2;
            buf[18] = 0x08;
            buf[22] = 0x08;
            return Some(buf);
        }
        f.read_exact(&mut buf).ok()?;
        Some(buf)
    }

    /// Read the 2048-byte data payload of a MODE2 sector.
    /// MODE2 form1 layout: 12 sync + 4 header + 8 subheader + 2048.
    pub fn read_sector_data(&mut self, lba: u32) -> Option<[u8; SECTOR_DATA]> {
        if self.tracks.first().map(|t| t.sector_size) == Some(2048) {
            let mut buf = [0u8; SECTOR_DATA];
            let off = lba as u64 * SECTOR_DATA as u64;
            let f = &mut self.files[self.file_of[0]];
            f.seek(SeekFrom::Start(off)).ok()?;
            f.read_exact(&mut buf).ok()?;
            return Some(buf);
        }
        let raw = self.read_sector_raw(lba)?;
        let mut out = [0u8; SECTOR_DATA];
        out.copy_from_slice(&raw[24..24 + SECTOR_DATA]);
        Some(out)
    }

    /// Disc identity for snapshot keys: SHA-256 over the TOC and a
    /// sampled set of sectors (start/middle/end + a few stragglers).
    /// Cheap — a handful of 2352-byte reads.
    pub fn identity_hash(&mut self) -> std::io::Result<[u8; 32]> {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        for t in &self.tracks {
            h.update(t.index_lba.to_le_bytes());
            h.update(t.sector_size.to_le_bytes());
            h.update([t.audio as u8]);
            h.update(
                t.bin_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .as_bytes(),
            );
        }
        // total sectors from last track's file length
        if let (Some(t), Some(fi)) = (self.tracks.last(), self.file_of.last()) {
            let meta_len = self.files[*fi].metadata()?.len() as u64;
            let last_sectors = (meta_len / t.sector_size as u64) as u32;
            h.update(last_sectors.to_le_bytes());
            let end = t.file_lba_start + last_sectors;
            for off in [0u32, end / 4, end / 2, end * 3 / 4, end - 1] {
                if let Some(sec) = self.read_sector_raw(off.min(end - 1)) {
                    h.update(sec);
                }
            }
        }
        Ok(h.finalize().into())
    }
}

/// Parse "MM:SS:FF" or "M:SS:FF" → LBA. Returns None on junk.
fn parse_msf(s: &str) -> Option<u32> {
    let mut it = s.split(':');
    let m: u32 = it.next()?.parse().ok()?;
    let sec: u32 = it.next()?.parse().ok()?;
    let f: u32 = it.next()?.parse().ok()?;
    Some(msf_to_lba(m, sec, f))
}
