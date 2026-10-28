//! Native instruction translators for the R3000A interpreter.
//!
//! The first tier compiles register-only ALU instruction classes. Memory,
//! control-flow, coprocessor, and exception instructions remain interpreted
//! so they retain the interpreter's precise bus and delay-slot behavior.

#[cfg(feature = "dynarec")]
use cranelift_codegen::ir::{
    Function, InstBuilder, MemFlagsData, UserFuncName, condcodes::IntCC, types,
};
#[cfg(feature = "dynarec")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "dynarec")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "dynarec")]
use cranelift_module::{Module, default_libcall_names};

#[cfg(feature = "dynarec")]
type NativeOp = unsafe extern "C" fn(*mut crate::cpu::Cpu, u32);

#[cfg(feature = "dynarec")]
#[derive(Clone, Copy)]
enum Entry {
    Empty,
    Unsupported,
    Compiled(NativeOp),
}

#[cfg(feature = "dynarec")]
#[derive(Clone, Copy)]
enum Op {
    Sll,
    Srl,
    Sra,
    Sllv,
    Srlv,
    Srav,
    Addu,
    Subu,
    And,
    Or,
    Xor,
    Nor,
    Slt,
    Sltu,
    Addiu,
    Slti,
    Sltiu,
    Andi,
    Ori,
    Xori,
    Lui,
}

#[cfg(feature = "dynarec")]
impl Op {
    fn decode(word: u32) -> Option<Self> {
        let primary = word >> 26;
        let function = word & 0x3F;
        Some(match (primary, function) {
            (0x00, 0x00) => Self::Sll,
            (0x00, 0x02) => Self::Srl,
            (0x00, 0x03) => Self::Sra,
            (0x00, 0x04) => Self::Sllv,
            (0x00, 0x06) => Self::Srlv,
            (0x00, 0x07) => Self::Srav,
            (0x00, 0x21) => Self::Addu,
            (0x00, 0x23) => Self::Subu,
            (0x00, 0x24) => Self::And,
            (0x00, 0x25) => Self::Or,
            (0x00, 0x26) => Self::Xor,
            (0x00, 0x27) => Self::Nor,
            (0x00, 0x2A) => Self::Slt,
            (0x00, 0x2B) => Self::Sltu,
            (0x09, _) => Self::Addiu,
            (0x0A, _) => Self::Slti,
            (0x0B, _) => Self::Sltiu,
            (0x0C, _) => Self::Andi,
            (0x0D, _) => Self::Ori,
            (0x0E, _) => Self::Xori,
            (0x0F, _) => Self::Lui,
            _ => return None,
        })
    }

    fn writes_rd(self) -> bool {
        matches!(
            self,
            Self::Sll
                | Self::Srl
                | Self::Sra
                | Self::Sllv
                | Self::Srlv
                | Self::Srav
                | Self::Addu
                | Self::Subu
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::Nor
                | Self::Slt
                | Self::Sltu
        )
    }
}

pub struct Dynarec {
    enabled: bool,
    #[cfg(feature = "dynarec")]
    module: Option<JITModule>,
    #[cfg(feature = "dynarec")]
    entries: Box<[Entry]>,
    #[cfg(feature = "dynarec")]
    next_id: u32,
    #[cfg(feature = "dynarec")]
    compile_failures: u32,
}

impl Dynarec {
    pub fn new() -> Self {
        #[cfg(feature = "dynarec")]
        {
            let module = JITBuilder::new(default_libcall_names())
                .map(JITModule::new)
                .map_err(|error| format!("create Cranelift host backend: {error}"));
            if let Err(error) = &module {
                eprintln!("dynarec unavailable; using interpreter: {error}");
            }
            let module = module.ok();
            Self {
                enabled: false,
                module,
                entries: vec![Entry::Empty; 64 * 64].into_boxed_slice(),
                next_id: 0,
                compile_failures: 0,
            }
        }
        #[cfg(not(feature = "dynarec"))]
        {
            Self { enabled: false }
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    #[cfg(feature = "dynarec")]
    pub fn compiled_op_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| matches!(entry, Entry::Compiled(_)))
            .count()
    }

    #[cfg(not(feature = "dynarec"))]
    pub fn compiled_op_count(&self) -> usize {
        0
    }

    #[cfg(feature = "dynarec")]
    pub fn execute(&mut self, word: u32, cpu: &mut crate::cpu::Cpu) -> bool {
        if !self.enabled {
            return false;
        }
        let primary = (word >> 26) as usize;
        let subcode = if primary == 0 {
            (word & 0x3F) as usize
        } else {
            0
        };
        let slot = primary * 64 + subcode;
        if matches!(self.entries[slot], Entry::Empty) {
            let Some(op) = Op::decode(word) else {
                self.entries[slot] = Entry::Unsupported;
                return false;
            };
            self.entries[slot] = match self.compile(op) {
                Ok(code) => Entry::Compiled(code),
                Err(error) => {
                    self.compile_failures += 1;
                    if self.compile_failures == 1 {
                        eprintln!("dynarec compile failed; using interpreter: {error}");
                    }
                    Entry::Unsupported
                }
            };
        }
        let Entry::Compiled(code) = self.entries[slot] else {
            return false;
        };
        // The function pointer remains valid while `module` owns its code pages.
        unsafe { code(cpu, word) };
        true
    }

    #[cfg(not(feature = "dynarec"))]
    pub fn execute(&mut self, _word: u32, _cpu: &mut crate::cpu::Cpu) -> bool {
        false
    }

    #[cfg(feature = "dynarec")]
    fn compile(&mut self, op: Op) -> Result<NativeOp, String> {
        let module = self
            .module
            .as_mut()
            .ok_or_else(|| "Cranelift module is unavailable".to_string())?;
        let mut signature = module.make_signature();
        signature.params.extend([
            cranelift_codegen::ir::AbiParam::new(types::I64),
            cranelift_codegen::ir::AbiParam::new(types::I32),
        ]);
        let func_id = module
            .declare_anonymous_function(&signature)
            .map_err(|error| error.to_string())?;
        let mut context = module.make_context();
        context.func =
            Function::with_name_signature(UserFuncName::user(0, self.next_id), signature);
        self.next_id += 1;
        let mut function_context = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
            let block = builder.create_block();
            builder.switch_to_block(block);
            builder.append_block_params_for_function_params(block);
            let params = builder.block_params(block);
            let cpu_ptr = params[0];
            let word = params[1];
            let mem_flags = MemFlagsData::trusted();
            let result = Self::emit(op, &mut builder, cpu_ptr, word, mem_flags);
            let dest_bits = builder
                .ins()
                .ushr_imm_u(word, if op.writes_rd() { 11 } else { 16 });
            let dest = builder.ins().band_imm_u(dest_bits, 31);
            let zero = builder.ins().iconst(types::I32, 0);
            let should_write = builder.ins().icmp(IntCC::NotEqual, dest, zero);
            let write_block = builder.create_block();
            let done_block = builder.create_block();
            builder
                .ins()
                .brif(should_write, write_block, &[], done_block, &[]);

            builder.switch_to_block(write_block);
            let reg_base = builder
                .ins()
                .iadd_imm_s(cpu_ptr, std::mem::offset_of!(crate::cpu::Cpu, r) as i64);
            let dest_offset32 = builder.ins().ishl_imm_u(dest, 2);
            let dest_offset = builder.ins().uextend(types::I64, dest_offset32);
            let dest_ptr = builder.ins().iadd(reg_base, dest_offset);
            builder.ins().store(mem_flags, result, dest_ptr, 0);

            let pending_ptr = builder.ins().iadd_imm_s(
                cpu_ptr,
                (std::mem::offset_of!(crate::cpu::Cpu, load_delay)
                    + std::mem::offset_of!(crate::cpu::LoadDelay, reg)) as i64,
            );
            let pending = builder.ins().load(types::I8, mem_flags, pending_ptr, 0);
            let pending32 = builder.ins().uextend(types::I32, pending);
            let same_dest = builder.ins().icmp(IntCC::Equal, pending32, dest);
            let no_pending = builder.ins().iconst(types::I8, 0);
            let next_pending = builder.ins().select(same_dest, no_pending, pending);
            builder.ins().store(mem_flags, next_pending, pending_ptr, 0);
            builder.ins().jump(done_block, &[]);

            builder.switch_to_block(done_block);
            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize(module.target_config());
        }
        module
            .define_function(func_id, &mut context)
            .map_err(|error| format!("{error}\n{}", context.func.display()))?;
        module.clear_context(&mut context);
        module
            .finalize_definitions()
            .map_err(|error| error.to_string())?;
        let code = module.get_finalized_function(func_id);
        // Cranelift emits this exact two-argument C ABI signature above.
        Ok(unsafe { std::mem::transmute::<*const u8, NativeOp>(code) })
    }

    #[cfg(feature = "dynarec")]
    fn emit(
        op: Op,
        builder: &mut FunctionBuilder<'_>,
        cpu_ptr: cranelift_codegen::ir::Value,
        word: cranelift_codegen::ir::Value,
        mem_flags: MemFlagsData,
    ) -> cranelift_codegen::ir::Value {
        let reg_base = builder
            .ins()
            .iadd_imm_s(cpu_ptr, std::mem::offset_of!(crate::cpu::Cpu, r) as i64);
        let rs_bits = builder.ins().ushr_imm_u(word, 21);
        let rs = builder.ins().band_imm_u(rs_bits, 31);
        let rt_bits = builder.ins().ushr_imm_u(word, 16);
        let rt = builder.ins().band_imm_u(rt_bits, 31);
        let rs_offset32 = builder.ins().ishl_imm_u(rs, 2);
        let rt_offset32 = builder.ins().ishl_imm_u(rt, 2);
        let rs_offset = builder.ins().uextend(types::I64, rs_offset32);
        let rt_offset = builder.ins().uextend(types::I64, rt_offset32);
        let rs_ptr = builder.ins().iadd(reg_base, rs_offset);
        let rt_ptr = builder.ins().iadd(reg_base, rt_offset);
        let a = builder.ins().load(types::I32, mem_flags, rs_ptr, 0);
        let b = builder.ins().load(types::I32, mem_flags, rt_ptr, 0);
        let imm_u = builder.ins().band_imm_u(word, 0xFFFF);
        let imm_shifted = builder.ins().ishl_imm_u(imm_u, 16);
        let imm_s = builder.ins().sshr_imm_s(imm_shifted, 16);
        let raw_sa = builder.ins().ushr_imm_u(word, 6);
        let sa = builder.ins().band_imm_u(raw_sa, 31);
        let shift = builder.ins().band_imm_u(a, 31);
        let truth = |builder: &mut FunctionBuilder<'_>, cond| {
            let one = builder.ins().iconst(types::I32, 1);
            let zero = builder.ins().iconst(types::I32, 0);
            builder.ins().select(cond, one, zero)
        };

        match op {
            Op::Sll => builder.ins().ishl(b, sa),
            Op::Srl => builder.ins().ushr(b, sa),
            Op::Sra => builder.ins().sshr(b, sa),
            Op::Sllv => builder.ins().ishl(b, shift),
            Op::Srlv => builder.ins().ushr(b, shift),
            Op::Srav => builder.ins().sshr(b, shift),
            Op::Addu | Op::Addiu => {
                let addend = if matches!(op, Op::Addu) { b } else { imm_s };
                builder.ins().iadd(a, addend)
            }
            Op::Subu => builder.ins().isub(a, b),
            Op::And => builder.ins().band(a, b),
            Op::Or => builder.ins().bor(a, b),
            Op::Xor => builder.ins().bxor(a, b),
            Op::Nor => {
                let or = builder.ins().bor(a, b);
                builder.ins().bnot(or)
            }
            Op::Slt => {
                let cond = builder.ins().icmp(IntCC::SignedLessThan, a, b);
                truth(builder, cond)
            }
            Op::Sltu => {
                let cond = builder.ins().icmp(IntCC::UnsignedLessThan, a, b);
                truth(builder, cond)
            }
            Op::Slti => {
                let cond = builder.ins().icmp(IntCC::SignedLessThan, a, imm_s);
                truth(builder, cond)
            }
            Op::Sltiu => {
                let cond = builder.ins().icmp(IntCC::UnsignedLessThan, a, imm_s);
                truth(builder, cond)
            }
            Op::Andi => builder.ins().band(a, imm_u),
            Op::Ori => builder.ins().bor(a, imm_u),
            Op::Xori => builder.ins().bxor(a, imm_u),
            Op::Lui => builder.ins().ishl_imm_u(imm_u, 16),
        }
    }
}

impl Default for Dynarec {
    fn default() -> Self {
        Self::new()
    }
}
