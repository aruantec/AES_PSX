//! Minimal ISO9660 reader — just enough to walk the root directory
//! and pull a file (SYSTEM.CNF, the boot EXE) from track 1.
//!
//! Only primary volume descriptors and plain (non-extended) directory
//! records are handled; that covers every retail PSX disc.

use crate::disc::Disc;

/// A found directory entry.
pub struct Entry {
    pub name: String,
    pub lba: u32,
    pub size: u32,
    pub dir: bool,
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Parse one directory record; returns None on a zero-length record
/// (end of directory) — the caller stops there.
fn dir_entry(sec: &[u8], off: usize) -> Option<(Entry, usize)> {
    if off >= sec.len() {
        return None;
    }
    let len = sec[off] as usize;
    if len == 0 {
        return None;
    }
    let lba = le32(&sec[off + 2..off + 6]);
    let size = le32(&sec[off + 10..off + 14]);
    let flags = sec[off + 25];
    let name_len = sec[off + 32] as usize;
    let name = String::from_utf8_lossy(&sec[off + 33..off + 33 + name_len]).into_owned();
    Some((
        Entry {
            name,
            lba,
            size,
            dir: flags & 2 != 0,
        },
        len.max(1),
    ))
}

/// List the root directory of track 1.
pub fn root_dir(disc: &mut Disc) -> Option<Vec<Entry>> {
    // Primary Volume Descriptor at LBA 16; root dir record at +156.
    let pvd = disc.read_sector_data(16)?;
    if &pvd[1..6] != b"CD001" {
        return None;
    }
    let (root, _) = dir_entry(&pvd, 156)?;
    let mut out = Vec::new();
    let mut lba = root.lba;
    let mut remaining = root.size;
    while remaining > 0 {
        let sec = disc.read_sector_data(lba)?;
        let mut off = 0usize;
        while off < 2048 {
            match dir_entry(&sec, off) {
                Some((e, len)) => {
                    if e.name != "\0" && e.name != "\x01" {
                        out.push(e);
                    }
                    off += len;
                }
                None => break,
            }
        }
        lba += 1;
        remaining = remaining.saturating_sub(2048);
    }
    Some(out)
}

/// Read an entire file given its root-dir name (with or without
/// ";1" version suffix).
pub fn read_file(disc: &mut Disc, name: &str) -> Option<Vec<u8>> {
    let want = name.trim_end_matches(";1").to_ascii_uppercase();
    let ents = root_dir(disc)?;
    for e in ents {
        let en = e.name.trim_end_matches(";1").to_ascii_uppercase();
        if en == want && !e.dir {
            let mut buf = Vec::with_capacity(e.size as usize);
            let mut lba = e.lba;
            let mut left = e.size;
            while left > 0 {
                let sec = disc.read_sector_data(lba)?;
                let take = left.min(2048) as usize;
                buf.extend_from_slice(&sec[..take]);
                left -= take as u32;
                lba += 1;
            }
            return Some(buf);
        }
    }
    None
}

/// Parsed PS-X EXE header.
pub struct ExeInfo {
    pub pc: u32,
    pub gp: u32,
    pub load_addr: u32,
    pub load_size: u32,
    pub stack_addr: u32,
    pub stack_size: u32,
    /// Offset of the code image inside the file (0x800).
    pub data_off: usize,
}

/// Parse a PS-X EXE (2048-byte header, "PS-X EXE" magic).
pub fn parse_exe(file: &[u8]) -> Option<ExeInfo> {
    if file.len() < 0x800 || &file[0..8] != b"PS-X EXE" {
        return None;
    }
    let g = |o: usize| le32(&file[o..o + 4]);
    Some(ExeInfo {
        pc: g(0x10),
        gp: g(0x14),
        load_addr: g(0x18),
        load_size: g(0x1C),
        stack_addr: g(0x30),
        stack_size: g(0x34),
        data_off: 0x800,
    })
}

/// Parse the BOOT= line of SYSTEM.CNF → e.g. "cdrom:\PSX.EXE;1".
/// Returns the filename component.
pub fn boot_path(cnf: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(cnf);
    for line in text.lines() {
        let l = line.trim();
        if l.to_ascii_uppercase().starts_with("BOOT") {
            if let Some(eq) = l.find('=') {
                let v = l[eq + 1..].trim();
                // strip "cdrom:\" or "cdrom:" prefix
                let v = v.strip_prefix("cdrom:").unwrap_or(v);
                let v = v.trim_start_matches('\\').trim_start_matches('/');
                return Some(v.to_string());
            }
        }
    }
    None
}
