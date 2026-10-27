//! Smoke tests — hand-assembled MIPS code in a synthetic "bios"
//! image plus a real-BIOS test gated on the file being present.

use aruanpsx::Emulator;

/// Build a fake 512K BIOS blob: entry 0x0000 = the given code; rest
/// zeros. `code` is big-endian instruction words.
fn fake_bios(code: &[u32]) -> Vec<u8> {
    let mut b = vec![0u8; 512 * 1024];
    for (i, w) in code.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    b
}

#[test]
fn executes_reset_vector() {
    // lui $t0,0x1234 ; addiu $t0,$t0,0x5678 ; sw $t0,0x100($0) ; loop
    let bios = fake_bios(&[
        0x3C08_1234, // lui t0,0x1234
        0x2508_5678, // addiu t0,0x5678
        0xAC08_0100, // sw t0,0x100(zero)
        0x0800_0003, // j self (0xBFC0000C → idx 0x0C/4 = 3)
        0x0000_0000,
    ]);
    let mut emu = Emulator::new(bios);
    for _ in 0..16 {
        emu.step();
    }
    // 0x12345678 should be sitting at RAM[0x100]
    let v = emu.bus.load32(0x8000_0100);
    assert_eq!(v, 0x1234_5678);
}

#[test]
fn load_delay_slot_semantics() {
    // lw $t0,0($t1) with t1 → ram; addiu $t1 should see OLD t0? no:
    // the delay means a following instruction reads the OLD value.
    //   lui t1,0x8000 ; lw t0,0(t1) ; move t2,t0(addu) — t2 gets OLD t0
    let mut bios = vec![0u8; 512 * 1024];
    // data at RAM[0] = 0xCAFEBABE is set post-init instead
    let code = [
        0x3C09_8000u32, // lui t1,0x8000
        0x8D28_0000,    // lw t0,0(t1)
        0x0128_5021,    // addu t2,t0,t1   (in delay slot — sees OLD t0)
        0xAC0A_0104,    // sw t2,0x104(zero)
        0x0800_0004,    // j self
    ];
    for (i, w) in code.iter().enumerate() {
        bios[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let mut emu = Emulator::new(bios);
    emu.bus.store32(0x8000_0000, 0xCAFE_BABE);
    for _ in 0..16 {
        emu.step();
    }
    // t2 = old_t0(0) + t1(0x80000000) = 0x80000000, stored at 0x104
    assert_eq!(emu.bus.load32(0x8000_0104), 0x8000_0000);
}

#[test]
fn real_bios_boots_past_reset() {
    let candidates = [
        "F:/Games/Emulators/Playstation/PSX/Duckstation/Bios/SCPH1001.BIN",
        "/media/aruan/Data/Games/Emulators/Playstation/PSX/Duckstation/Bios/SCPH1001.BIN",
    ];
    let (_path, bios) = match candidates
        .iter()
        .find_map(|p| std::fs::read(p).ok().map(|b| (*p, b)))
    {
        Some(x) => x,
        None => {
            eprintln!("skip: no bios at {:?}", candidates);
            return;
        }
    };
    let mut emu = Emulator::new(bios);
    let mut mapped_exec = false;
    // ~2 seconds of emulated time
    for _ in 0..30_000_000 {
        let pc = emu.cpu.pc();
        let phys = pc & 0x1FFF_FFFF;
        // executing from BIOS-installed kernel code in RAM
        if pc >= 0x8000_0000 && pc < 0x8000_8000 {
            mapped_exec = true;
        }
        // runaway guard: pc must stay in mapped space
        assert!(
            phys < 0x0080_0000
                || (0x1F80_0000..=0x1F80_03FF).contains(&phys)
                || (0x1FC0_0000..=0x1FC7_FFFF).contains(&phys),
            "cpu ran away at pc={pc:08X}"
        );
        emu.step();
    }
    let tty = emu.tty().to_string();
    println!("bios tty: {tty:?}");
    println!("pc: {:08X} istat={:04X} imask={:04X}",
        emu.cpu.pc(), emu.bus.irq.stat, emu.bus.irq.mask);
    // BIOS kernel code lives at 0x8000xxxx — reaching it means the
    // shell (or a kernel handler) is alive, not stuck in reset init.
    assert!(mapped_exec, "bios never reached its RAM kernel");
    assert_ne!(emu.cpu.pc(), 0xBFC0_0000);
}
