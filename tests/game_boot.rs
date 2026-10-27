//! End-to-end: real BIOS boots a real disc into the game.
//!
//! Skipped (with a message) when the local BIOS/game dumps are absent.
//! Override paths with PSX_BIOS / PSX_GAME env vars.

use aruanpsx::{disc::Disc, Emulator};

const BIOS: &[&str] = &[
    "F:/Games/Emulators/Playstation/PSX/Duckstation/Bios/SCPH1001.BIN",
    "/media/aruan/Data/Games/Emulators/Playstation/PSX/Duckstation/Bios/SCPH1001.BIN",
];
const GAME: &[&str] = &[
    "F:/Gaming/Games/Consoles/Playstation/PSX/Crash Bandicoot - Warped (USA)/Crash Bandicoot - Warped (USA).cue",
    "/media/aruan/Data/Gaming/Games/Consoles/Playstation/PSX/Crash Bandicoot - Warped (USA)/Crash Bandicoot - Warped (USA).cue",
];

fn find(env: &str, list: &[&str]) -> Option<String> {
    std::env::var(env)
        .ok()
        .into_iter()
        .chain(list.iter().map(|s| s.to_string()))
        .find(|p| std::path::Path::new(p).exists())
}

#[test]
fn bios_boots_disc_into_game() {
    let (Some(bios), Some(game)) = (find("PSX_BIOS", BIOS), find("PSX_GAME", GAME)) else {
        eprintln!("skip: BIOS or game image not found");
        return;
    };
    let mut emu = Emulator::new(std::fs::read(&bios).unwrap());
    emu.insert_disc(Disc::load(std::path::Path::new(&game)).unwrap());

    let mut fb = Vec::new();
    let mut launched_at = None;
    let mut rendered = false;
    let mut audio_peak = 0i32;
    for frame in 0..3000u32 {
        emu.run_frame();
        audio_peak = audio_peak.max(emu.take_audio().iter().map(|s| (*s as i32).abs()).max().unwrap_or(0));
        if launched_at.is_none() && emu.tty().contains("Execute !") {
            launched_at = Some(frame);
        }
        if let Some(at) = launched_at {
            // the game must draw something of its own after taking over
            if frame > at + 120 && frame % 30 == 0 {
                let (w, h) = emu.framebuffer(&mut fb);
                let lit = fb.iter().filter(|&&p| p & 0xF0F0F0 != 0).count();
                if lit > w * h / 20 {
                    rendered = true;
                    break;
                }
            }
        }
    }
    println!("{}", emu.tty());
    let at = launched_at.expect("BIOS never launched the disc's boot executable");
    assert!(emu.tty().contains("BOOT ="), "SYSTEM.CNF not parsed by the BIOS");
    assert!(rendered, "game launched at frame {at} but never rendered a visible frame");
    assert!(audio_peak > 1000, "no audio output during boot (peak {audio_peak})");
}
