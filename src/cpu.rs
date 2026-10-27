//! R3000A CPU interpreter — little-endian MIPS-I subset.
//!
//! The PSX CPU is an LSI R3000A: MIPS-I instruction set minus a few
//! unimplemented opcodes (no LL/SC, no TLB), plus cop0 (exception
//! control) and cop2 = the GTE geometry engine. Notable quirks:
//!
//! - Load-delay slot: a load's result is NOT visible to the next
//!   instruction (the delay slot sees the stale value). Modeled with
//!   a one-deep pending register write.
//! - Branch-delay slot: the instruction after any jump/branch always
//!   executes.
//! - No byte-order issues — the console is pure little-endian.
//!
//! Execution model: `step()` runs one instruction; cycle accounting
//! is coarse for now (1 cycle/instr + device event counts) — timing
//! refinement is handled by the scheduler layer.

use crate::bus::Bus;
use crate::gte::Gte;

// ---- exception codes (Cop0 CAUSE.ExcCode) ----
#[allow(dead_code)]
const EXC_INT: u32 = 0;   // Interrupt
const EXC_ADEL: u32 = 4;  // address error on load
const EXC_ADES: u32 = 5;  // address error on store
#[allow(dead_code)]
const EXC_IBE: u32 = 6;   // bus error on instruction fetch
#[allow(dead_code)]
const EXC_DBE: u32 = 7;   // bus error on data access
const EXC_SYS: u32 = 8;   // syscall
const EXC_BP: u32 = 9;    // breakpoint
const EXC_RI: u32 = 10;   // reserved/undefined instruction
#[allow(dead_code)]
const EXC_CPU: u32 = 11;  // coprocessor unusable
const EXC_OV: u32 = 12;   // arithmetic overflow

const COP0_SR: u8 = 12;
const COP0_CAUSE: u8 = 13;
const COP0_EPC: u8 = 14;
const COP0_BADVADDR: u8 = 8;

pub struct Cpu {
    /// r[0] is always zero.
    r: [u32; 32],
    hi: u32,
    lo: u32,
    /// Address of the next instruction to fetch.
    pc: u32,
    /// Address of the instruction after that — the two-stage pc
    /// pipeline makes delay slots fall out naturally.
    next_pc: u32,
    /// Current instruction's address (for EPC on exceptions).
    cur_pc: u32,
    /// True while executing a branch/jump delay slot.
    in_delay_slot: bool,
    /// Set by the just-executed instruction if it was a branch.
    took_branch_slot: bool,
    /// PC of the branch that owns the delay slot.
    branch_owner: u32,
    /// Load issued by the PREVIOUS instruction — commits after the
    /// current instruction executes.
    load_delay: (u8, u32),
    /// Load issued by the CURRENT instruction.
    load_new: (u8, u32),
    /// Cop0 registers (12=SR, 13=CAUSE, 14=EPC, plus breakpoints).
    pub cop0: [u32; 16],
    pub gte: Gte,
    /// Serial/tty capture: intercepts A(3C)=putchar, B(3D)=putchar.
    pub tty: String,
    /// Optional per-call hook log: (vector<<8)|index.
    pub call_log: Option<Vec<u32>>,
}

impl Cpu {
    pub fn new() -> Self {
        let mut c = Self {
            r: [0; 32],
            hi: 0,
            lo: 0,
            pc: 0xBFC0_0000,
            next_pc: 0xBFC0_0004,
            cur_pc: 0,
            in_delay_slot: false,
            took_branch_slot: false,
            branch_owner: 0,
            load_delay: (0, 0),
            load_new: (0, 0),
            cop0: [0; 16],
            gte: Gte::new(),
            tty: String::new(),
            call_log: None,
        };
        // reset state: BEV=1 so exceptions vector to ROM 0xBFC00180
        c.cop0[COP0_SR as usize] = 1 << 22;
        c.cop0[15] = 0x0000_0002; // PRId
        c
    }

    /// Address of the next instruction to execute.
    pub fn fetch_pc(&self) -> u32 {
        self.pc
    }

    pub fn pc(&self) -> u32 {
        self.cur_pc
    }

    // ---- snapshot accessors ----
    pub fn regs_copy(&self) -> [u32; 32] { self.r }
    pub fn hi(&self) -> u32 { self.hi }
    pub fn lo(&self) -> u32 { self.lo }
    pub fn pc_raw(&self) -> u32 { self.pc }
    pub fn next_pc_raw(&self) -> u32 { self.next_pc }
    pub fn cop0_copy(&self) -> [u32; 16] { self.cop0 }
    pub fn load_regs(
        &mut self,
        r: [u32; 32],
        hi: u32,
        lo: u32,
        pc: u32,
        next_pc: u32,
        cop0: [u32; 16],
    ) {
        self.r = r;
        self.r[0] = 0;
        self.hi = hi;
        self.lo = lo;
        self.pc = pc;
        self.next_pc = next_pc;
        self.cop0 = cop0;
        self.load_new = (0, 0);
        self.load_delay = (0, 0);
        self.in_delay_slot = false;
        self.took_branch_slot = false;
    }

    pub fn rg(&self, i: u8) -> u32 {
        self.r[i as usize]
    }

    pub fn set_reg(&mut self, i: u8, v: u32) {
        if i != 0 {
            self.r[i as usize] = v;
        }
    }

    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
        self.next_pc = pc.wrapping_add(4);
        self.load_delay = (0, 0);
        self.load_new = (0, 0);
        self.in_delay_slot = false;
        self.took_branch_slot = false;
    }

    #[inline]
    fn set_r(&mut self, i: u8, v: u32) {
        if i != 0 {
            // an instruction in a load's delay slot that writes the
            // load's target register wins: the pending load is dropped
            if self.load_delay.0 == i {
                self.load_delay = (0, 0);
            }
            self.r[i as usize] = v;
        }
    }

    /// One instruction. Returns elapsed sysclk cycles.
    pub fn step(&mut self, bus: &mut Bus) -> u32 {
        // hardware interrupt line → CAUSE.IP2 (bit 10)
        if bus.irq.pending() {
            self.cop0[COP0_CAUSE as usize] |= 1 << 10;
        } else {
            self.cop0[COP0_CAUSE as usize] &= !(1 << 10);
        }
        self.cur_pc = self.pc;
        self.in_delay_slot = self.took_branch_slot;
        self.took_branch_slot = false;
        if self.in_delay_slot {
            self.branch_owner = self.pc.wrapping_sub(4);
        }
        bus.stall = 0;
        let sr = self.cop0[COP0_SR as usize];
        let mut extra = 0;
        if sr & 1 != 0 && (self.cop0[COP0_CAUSE as usize] & sr & 0xFF00) != 0 {
            // A GTE command sitting at the interrupted address has
            // already been issued to the coprocessor when the IRQ is
            // recognised; the kernel's handler knows this and resumes
            // at EPC+4, so the command must execute now.
            let op = bus.fetch32(self.pc);
            if op >> 25 == 0x25 {
                extra += self.gte.command(op);
            }
            self.exception(EXC_INT);
            self.cur_pc = self.pc;
        }
        // kernel-vector entries: when execution lands on the A0/B0/C0
        // stubs the delay slot has already run — $t1 holds the
        // function index here, not at the jump site.
        if matches!(self.pc & 0x1FFF_FFFF, 0xA0 | 0xB0 | 0xC0) {
            self.bios_call_log(self.pc & 0x1FFF_FFFF, bus);
        }
        if self.pc & 3 != 0 {
            self.cop0[COP0_BADVADDR as usize] = self.pc;
            self.exception(EXC_ADEL);
            self.cur_pc = self.pc;
        }
        let op = bus.fetch32(self.pc);
        self.pc = self.next_pc;
        self.next_pc = self.pc.wrapping_add(4);

        extra += self.exec(op, bus);
        // Commit the pending load from the previous instruction. A new
        // load to the same register cancels the older write-back.
        let (lr, lv) = self.load_delay;
        if lr != 0 && lr != self.load_new.0 {
            self.r[lr as usize] = lv;
        }
        self.load_delay = self.load_new;
        self.load_new = (0, 0);
        self.r[0] = 0;
        1 + extra + bus.stall
    }

    fn exception(&mut self, code: u32) {
        // the instruction that issued the pending load retired before
        // the exception — complete it so the handler sees the value
        let (lr, lv) = self.load_delay;
        if lr != 0 {
            self.r[lr as usize] = lv;
        }
        self.load_delay = (0, 0);
        self.load_new = (0, 0);
        // push the KU/IE mode stack
        let sr = self.cop0[COP0_SR as usize];
        self.cop0[COP0_SR as usize] = (sr & !0x3F) | ((sr << 2) & 0x3F);
        let mut cause = self.cop0[COP0_CAUSE as usize];
        cause = (cause & !0x7C) | (code << 2);
        if self.in_delay_slot {
            cause |= 1 << 31;
            self.cop0[COP0_EPC as usize] = self.branch_owner;
        } else {
            cause &= !(1 << 31);
            self.cop0[COP0_EPC as usize] = self.cur_pc;
        }
        self.cop0[COP0_CAUSE as usize] = cause;
        let vector = if sr & (1 << 22) != 0 { 0xBFC0_0180 } else { 0x8000_0080 };
        self.pc = vector;
        self.next_pc = vector + 4;
        self.in_delay_slot = false;
        self.took_branch_slot = false;
    }

    #[inline]
    fn branch(&mut self, target: u32) {
        self.next_pc = target;
        self.took_branch_slot = true;
    }

    #[inline]
    fn link(&mut self, reg: u8) {
        let ra = self.cur_pc.wrapping_add(8);
        self.set_r(reg, ra);
    }

    // ---------- BIOS call interception (tty capture) ----------
    // BIOS functions are invoked by jalr/jr to 0xA0/0xB0/0xC0 with
    // the function index in $t1 (r9). We log the useful ones but let
    // the real code still run (the trampolines live in low RAM).
    fn bios_call_log(&mut self, target: u32, bus: &mut Bus) {
        // kernel stubs live at physical 0xA0/0xB0/0xC0 — the jump
        // target may carry KSEG0/1 bits (0x800000B0 etc.)
        if !matches!(target & 0x1FFF_FFFF, 0xA0 | 0xB0 | 0xC0) {
            return;
        }
        let target = target & 0x1FFF_FFFF;
        let f = self.rg(9); // $t1
        if let Some(log) = self.call_log.as_mut() {
            log.push((target << 8) | f);
        }
        match (target, f) {
            (0xA0, 0x3C) => {
                // putchar(char)
                let c = self.rg(4) as u8;
                if c != 0 {
                    self.tty.push(c as char);
                }
            }
            (0xA0, 0x3E) | (0xB0, 0x3D) => {
                // puts(char*) — but some BIOS print paths push single
                // characters through it; capture both forms.
                let a = self.rg(4);
                if a < 0x1_0000 {
                    if a != 0 {
                        self.tty.push(a as u8 as char);
                    }
                } else {
                    let mut a = a;
                    loop {
                        let c = bus.load8(a);
                        if c == 0 {
                            break;
                        }
                        self.tty.push(c as char);
                        a = a.wrapping_add(1);
                    }
                }
            }
            _ => {}
        }
    }

    // ----------------------------------------------------------

    /// Execute one fetched instruction; returns extra device cycles
    /// (GTE command timing) on top of the base 1-cycle cost.
    fn exec(&mut self, op: u32, bus: &mut Bus) -> u32 {
        let rs = ((op >> 21) & 0x1F) as u8;
        let rt = ((op >> 16) & 0x1F) as u8;
        let rd = ((op >> 11) & 0x1F) as u8;
        let sa = (op >> 6) & 0x1F;
        let imm = (op & 0xFFFF) as i16 as i32;
        let uimm = (op & 0xFFFF) as u32;
        let target = (self.cur_pc & 0xF000_0000) | ((op & 0x03FF_FFFF) << 2);

        match op >> 26 {
            0x00 => match op & 0x3F {
                0x00 => self.set_r(rd, self.rg(rt) << sa),            // SLL
                0x02 => self.set_r(rd, self.rg(rt) >> sa),            // SRL
                0x03 => self.set_r(rd, ((self.rg(rt) as i32) >> sa) as u32), // SRA
                0x04 => self.set_r(rd, self.rg(rt) << (self.rg(rs) & 31)),  // SLLV
                0x06 => self.set_r(rd, self.rg(rt) >> (self.rg(rs) & 31)),  // SRLV
                0x07 => self.set_r(rd, ((self.rg(rt) as i32) >> (self.rg(rs) & 31)) as u32), // SRAV
                0x08 => {
                    // JR
                    let t = self.rg(rs);
                    self.branch(t);
                }
                0x09 => {
                    // JALR rd, rs
                    let t = self.rg(rs);
                    self.link(rd);
                    self.branch(t);
                }
                0x0C => self.exception(EXC_SYS),                      // SYSCALL
                0x0D => self.exception(EXC_BP),                       // BREAK
                0x10 => { let v = self.hi; self.set_r(rd, v); }       // MFHI
                0x11 => self.hi = self.rg(rs),                        // MTHI
                0x12 => { let v = self.lo; self.set_r(rd, v); }       // MFLO
                0x13 => self.lo = self.rg(rs),                        // MTLO
                0x18 => {                                             // MULT
                    let p = (self.rg(rs) as i32 as i64) * (self.rg(rt) as i32 as i64);
                    self.lo = p as u32;
                    self.hi = (p >> 32) as u32;
                }
                0x19 => {                                             // MULTU
                    let p = (self.rg(rs) as u64) * (self.rg(rt) as u64);
                    self.lo = p as u32;
                    self.hi = (p >> 32) as u32;
                }
                0x1A => {                                             // DIV
                    let a = self.rg(rs) as i32;
                    let b = self.rg(rt) as i32;
                    if b == 0 {
                        self.lo = if a >= 0 { 0xFFFF_FFFF } else { 1 };
                        self.hi = a as u32;
                    } else if a == i32::MIN && b == -1 {
                        self.lo = i32::MIN as u32;
                        self.hi = 0;
                    } else {
                        self.lo = (a / b) as u32;
                        self.hi = (a % b) as u32;
                    }
                }
                0x1B => {                                             // DIVU
                    let a = self.rg(rs);
                    let b = self.rg(rt);
                    if b == 0 {
                        self.lo = 0xFFFF_FFFF;
                        self.hi = a;
                    } else {
                        self.lo = a / b;
                        self.hi = a % b;
                    }
                }
                0x20 => {                                             // ADD
                    let a = self.rg(rs) as i32;
                    let b = self.rg(rt) as i32;
                    match a.checked_add(b) {
                        Some(v) => self.set_r(rd, v as u32),
                        None => self.exception(EXC_OV),
                    }
                }
                0x21 => self.set_r(rd, self.rg(rs).wrapping_add(self.rg(rt))), // ADDU
                0x22 => {                                             // SUB
                    let a = self.rg(rs) as i32;
                    let b = self.rg(rt) as i32;
                    match a.checked_sub(b) {
                        Some(v) => self.set_r(rd, v as u32),
                        None => self.exception(EXC_OV),
                    }
                }
                0x23 => self.set_r(rd, self.rg(rs).wrapping_sub(self.rg(rt))), // SUBU
                0x24 => self.set_r(rd, self.rg(rs) & self.rg(rt)),    // AND
                0x25 => self.set_r(rd, self.rg(rs) | self.rg(rt)),    // OR
                0x26 => self.set_r(rd, self.rg(rs) ^ self.rg(rt)),    // XOR
                0x27 => self.set_r(rd, !(self.rg(rs) | self.rg(rt))), // NOR
                0x2A => self.set_r(rd, ((self.rg(rs) as i32) < self.rg(rt) as i32) as u32), // SLT
                0x2B => self.set_r(rd, (self.rg(rs) < self.rg(rt)) as u32),                 // SLTU
                _ => self.exception(EXC_RI),
            },
            0x01 => {
                // REGIMM: on the R3000 the rt field decodes loosely —
                // bit0 picks the condition (0=BLTZ, 1=BGEZ) and the
                // link happens only for rt = 0x10/0x11. Every other rt
                // value still branches instead of raising RI.
                let v = self.rg(rs) as i32;
                let t = self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2);
                if rt == 0x10 || rt == 0x11 {
                    self.link(31);
                }
                if rt & 1 != 0 {
                    if v >= 0 { self.branch(t) }
                } else if v < 0 {
                    self.branch(t)
                }
            }
            0x02 => { self.branch(target); }                          // J
            0x03 => { self.link(31); self.branch(target); }           // JAL
            0x04 => {                                                 // BEQ
                if self.rg(rs) == self.rg(rt) {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                }
            }
            0x05 => {                                                 // BNE
                if self.rg(rs) != self.rg(rt) {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                }
            }
            0x06 => {                                                 // BLEZ
                if (self.rg(rs) as i32) <= 0 {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                }
            }
            0x07 => {                                                 // BGTZ
                if (self.rg(rs) as i32) > 0 {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                }
            }
            // Branch-likely (R3000): a NOT-taken branch annuls its
            // delay slot — the next fetch must skip it entirely, so
            // bump pc (not just next_pc) past it.
            0x14 => {                                                 // BEQL
                if self.rg(rs) == self.rg(rt) {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                } else {
                    self.pc = self.cur_pc.wrapping_add(8);
                    self.next_pc = self.cur_pc.wrapping_add(12);
                }
            }
            0x15 => {                                                 // BNEL
                if self.rg(rs) != self.rg(rt) {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                } else {
                    self.pc = self.cur_pc.wrapping_add(8);
                    self.next_pc = self.cur_pc.wrapping_add(12);
                }
            }
            0x16 => {                                                 // BLEZL
                if (self.rg(rs) as i32) <= 0 {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                } else {
                    self.pc = self.cur_pc.wrapping_add(8);
                    self.next_pc = self.cur_pc.wrapping_add(12);
                }
            }
            0x17 => {                                                 // BGTZL
                if (self.rg(rs) as i32) > 0 {
                    self.branch(self.cur_pc.wrapping_add(4).wrapping_add((imm as u32) << 2));
                } else {
                    self.pc = self.cur_pc.wrapping_add(8);
                    self.next_pc = self.cur_pc.wrapping_add(12);
                }
            }
            0x08 => {                                                 // ADDI
                let a = self.rg(rs) as i32;
                match a.checked_add(imm) {
                    Some(v) => self.set_r(rt, v as u32),
                    None => self.exception(EXC_OV),
                }
            }
            0x09 => self.set_r(rt, self.rg(rs).wrapping_add(imm as u32)), // ADDIU
            0x0A => self.set_r(rt, ((self.rg(rs) as i32) < imm) as u32),  // SLTI
            0x0B => self.set_r(rt, (self.rg(rs) < (imm as u32)) as u32),  // SLTIU
            0x0C => self.set_r(rt, self.rg(rs) & uimm),               // ANDI
            0x0D => self.set_r(rt, self.rg(rs) | uimm),               // ORI
            0x0E => self.set_r(rt, self.rg(rs) ^ uimm),               // XORI
            0x0F => self.set_r(rt, uimm << 16),                       // LUI
            0x10 => self.cop0(op, bus),                               // COP0
            0x12 => return self.cop2(op),                             // COP2 (GTE)
            0x20 => {                                                 // LB
                let a = self.rg(rs).wrapping_add(imm as u32);
                let v = bus.load8(a) as i8 as i32 as u32;
                self.load_new = (rt, v);
            }
            0x21 => {                                                 // LH
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 1 != 0 { self.cop0[8] = a; { self.exception(EXC_ADEL); return 0; } }
                let v = bus.load16(a) as i16 as i32 as u32;
                self.load_new = (rt, v);
            }
            0x22 => {                                                 // LWL
                let a = self.rg(rs).wrapping_add(imm as u32);
                let aligned = bus.load32(a & !3);
                let cur = if rt == self.load_delay.0 { self.load_delay.1 } else { self.rg(rt) };
                let v = match a & 3 {
                    0 => (cur & 0x00FF_FFFF) | (aligned << 24),
                    1 => (cur & 0x0000_FFFF) | (aligned << 16),
                    2 => (cur & 0x0000_00FF) | (aligned << 8),
                    _ => aligned,
                };
                self.load_new = (rt, v);
            }
            0x23 => {                                                 // LW
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 3 != 0 { self.cop0[8] = a; { self.exception(EXC_ADEL); return 0; } }
                let v = bus.load32(a);
                self.load_new = (rt, v);
            }
            0x24 => {                                                 // LBU
                let a = self.rg(rs).wrapping_add(imm as u32);
                let v = bus.load8(a) as u32;
                self.load_new = (rt, v);
            }
            0x25 => {                                                 // LHU
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 1 != 0 { self.cop0[8] = a; { self.exception(EXC_ADEL); return 0; } }
                let v = bus.load16(a) as u32;
                self.load_new = (rt, v);
            }
            0x26 => {                                                 // LWR
                let a = self.rg(rs).wrapping_add(imm as u32);
                let aligned = bus.load32(a & !3);
                let cur = if rt == self.load_delay.0 { self.load_delay.1 } else { self.rg(rt) };
                let v = match a & 3 {
                    0 => aligned,
                    1 => (cur & 0xFF00_0000) | (aligned >> 8),
                    2 => (cur & 0xFFFF_0000) | (aligned >> 16),
                    _ => (cur & 0xFFFF_FF00) | (aligned >> 24),
                };
                self.load_new = (rt, v);
            }
            0x28 => {                                                 // SB
                let a = self.rg(rs).wrapping_add(imm as u32);
                bus.store8(a, self.rg(rt) as u8);
            }
            0x29 => {                                                 // SH
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 1 != 0 { self.cop0[8] = a; { self.exception(EXC_ADES); return 0; } }
                bus.store16(a, self.rg(rt) as u16);
            }
            0x2A => {                                                 // SWL
                let a = self.rg(rs).wrapping_add(imm as u32);
                let w = bus.load32(a & !3);
                let v = match a & 3 {
                    0 => (w & 0xFFFF_FF00) | (self.rg(rt) >> 24),
                    1 => (w & 0xFFFF_0000) | (self.rg(rt) >> 16),
                    2 => (w & 0xFF00_0000) | (self.rg(rt) >> 8),
                    _ => self.rg(rt),
                };
                bus.store32(a & !3, v);
            }
            0x2B => {                                                 // SW
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 3 != 0 { self.cop0[8] = a; { self.exception(EXC_ADES); return 0; } }
                bus.store32(a, self.rg(rt));
            }
            0x2E => {                                                 // SWR
                let a = self.rg(rs).wrapping_add(imm as u32);
                let w = bus.load32(a & !3);
                let v = match a & 3 {
                    0 => self.rg(rt),
                    1 => (w & 0x0000_00FF) | (self.rg(rt) << 8),
                    2 => (w & 0x0000_FFFF) | (self.rg(rt) << 16),
                    _ => (w & 0x00FF_FFFF) | (self.rg(rt) << 24),
                };
                bus.store32(a & !3, v);
            }
            0x32 => {                                                 // LWC2
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 3 != 0 { self.cop0[8] = a; { self.exception(EXC_ADEL); return 0; } }
                self.gte.write_data(rt, bus.load32(a));
            }
            0x3A => {                                                 // SWC2
                let a = self.rg(rs).wrapping_add(imm as u32);
                if a & 3 != 0 { self.cop0[8] = a; { self.exception(EXC_ADES); return 0; } }
                bus.store32(a, self.gte.read_data(rt));
            }
            _ => self.exception(EXC_RI),
        }
        // r0 stays zero no matter what
        self.r[0] = 0;
        0
    }

    fn cop0(&mut self, op: u32, bus: &mut Bus) {
        let rs = ((op >> 21) & 0x1F) as u8;
        let rd = ((op >> 11) & 0x1F) as u8;
        let rt = ((op >> 16) & 0x1F) as u8;
        match rs {
            0x00 => {
                // MFC0 (load-delayed like a memory load)
                let v = self.cop0[(rd & 15) as usize];
                self.load_new = (rt, v);
            }
            0x04 => {
                // MTC0
                let v = self.rg(rt);
                match rd {
                    COP0_SR => {
                        self.cop0[COP0_SR as usize] = v;
                        bus.cache_isolated = v & (1 << 16) != 0;
                    }
                    // only the software interrupt bits are writable
                    COP0_CAUSE => {
                        let c = &mut self.cop0[COP0_CAUSE as usize];
                        *c = (*c & !0x300) | (v & 0x300);
                    }
                    3 | 5 | 6 | 7 | 9 | 11 => self.cop0[rd as usize] = v,
                    _ => {}
                }
            }
            0x10 => match op & 0x3F {
                0x10 => {
                    // RFE: pop the KU/IE stack (old pair is kept)
                    let sr = self.cop0[COP0_SR as usize];
                    self.cop0[COP0_SR as usize] = (sr & !0xF) | ((sr >> 2) & 0xF);
                }
                _ => self.exception(EXC_RI),
            },
            _ => self.exception(EXC_RI),
        }
    }

    fn cop2(&mut self, op: u32) -> u32 {
        let rs = (op >> 21) & 0x1F;
        let rt = ((op >> 16) & 0x1F) as u8;
        let rd = ((op >> 11) & 0x1F) as u8;
        match rs {
            0x00 => {                                   // MFC2
                let v = self.gte.read_data(rd);
                self.load_new = (rt, v);
            }
            0x02 => {                                   // CFC2
                let v = self.gte.read_ctrl(rd);
                self.load_new = (rt, v);
            }
            0x04 => self.gte.write_data(rd, self.rg(rt)), // MTC2
            0x06 => self.gte.write_ctrl(rd, self.rg(rt)), // CTC2
            _ => {
                if rs & 0x10 != 0 {
                    return self.gte.command(op);        // GTE command
                } else {
                    self.exception(EXC_RI);
                }
            }
        }
        0
    }
}
