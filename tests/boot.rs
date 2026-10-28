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
fn writes_to_zero_register_are_ignored() {
    let bios = fake_bios(&[0x2400_0001]); // addiu $zero,$zero,1
    let mut emu = Emulator::new(bios);
    emu.step();
    assert_eq!(emu.cpu.rg(0), 0);
}

#[test]
fn dynarec_alu_matches_interpreter() {
    let r = |rs: u32, rt: u32, rd: u32, sa: u32, funct: u32| {
        (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    };
    let i = |op: u32, rs: u32, rt: u32, imm: u16| {
        (op << 26) | (rs << 21) | (rt << 16) | imm as u32
    };
    let words = [
        i(0x0F, 0, 8, 0x8123),              // lui t0,0x8123
        i(0x09, 8, 8, (-37i16) as u16),    // addiu t0,t0,-37
        i(0x0A, 8, 9, 0),                  // slti t1,t0,0
        i(0x0B, 8, 10, 0x8000),            // sltiu t2,t0,0x8000
        i(0x0C, 8, 11, 0x00FF),            // andi t3,t0,0x00ff
        i(0x0D, 11, 12, 0x8000),           // ori t4,t3,0x8000
        i(0x0E, 12, 13, 0x1234),           // xori t5,t4,0x1234
        r(0, 13, 14, 5, 0x00),             // sll t6,t5,5
        r(0, 14, 15, 3, 0x02),             // srl t7,t6,3
        r(0, 14, 24, 2, 0x03),             // sra t8,t6,2
        r(14, 15, 25, 0, 0x21),            // addu t9,t6,t7
        r(25, 24, 16, 0, 0x23),            // subu s0,t9,t8
        r(8, 12, 17, 0, 0x24),             // and s1,t0,t4
        r(8, 12, 18, 0, 0x25),             // or s2,t0,t4
        r(8, 12, 19, 0, 0x26),             // xor s3,t0,t4
        r(8, 12, 20, 0, 0x27),             // nor s4,t0,t4
        r(8, 14, 21, 0, 0x2A),             // slt s5,t0,t6
        r(8, 14, 22, 0, 0x2B),             // sltu s6,t0,t6
        r(13, 11, 23, 0, 0x04),            // sllv s7,t5,t3
        r(13, 11, 24, 0, 0x06),            // srlv t8,t5,t3
        r(13, 11, 25, 0, 0x07),            // srav t9,t5,t3
        0x0BF0_0015,                       // j self
        0,
    ];
    let bios = fake_bios(&words);
    let mut reference = Emulator::new(bios.clone());
    reference.set_dynarec_enabled(false);
    let mut compiled = Emulator::new(bios);
    compiled.set_dynarec_enabled(true);

    for _ in 0..80 {
        reference.step();
        compiled.step();
        assert_eq!(compiled.cpu.regs_copy(), reference.cpu.regs_copy());
        assert_eq!(compiled.cpu.fetch_pc(), reference.cpu.fetch_pc());
        assert_eq!(compiled.cycles(), reference.cycles());
    }
    assert!(compiled.dynarec_compiled_op_count() >= 20);
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
