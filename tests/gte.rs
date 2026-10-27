//! GTE command semantics — hand-computed vectors from the hardware
//! formulas (identity matrices, known division results, saturation).

use aruanpsx::gte::Gte;

/// Identity rotation matrix (1.0 = 0x1000), zero translation.
fn identity() -> Gte {
    let mut g = Gte::new();
    // RT: lo16/hi16 packed pairs; RT33 alone in ctrl[4]
    g.ctrl[0] = 0x0000_1000; // RT11=1.0, RT12=0
    g.ctrl[1] = 0; // RT13, RT21
    g.ctrl[2] = 0x0000_1000; // RT22, RT23
    g.ctrl[3] = 0; // RT31, RT32
    g.ctrl[4] = 0x1000; // RT33
    g
}

const SF: u32 = 1 << 19;

#[test]
fn rtps_identity() {
    let mut g = identity();
    g.data[0] = (0x200 << 16) | 0x100; // VX0=0x100, VY0=0x200
    g.data[1] = 0x1000; // VZ0
    g.ctrl[26] = 0x100; // H
    g.command(0x01 | SF); // RTPS sf=1
    assert_eq!(g.data[9] as i16, 0x100); // IR1
    assert_eq!(g.data[10] as i16, 0x200); // IR2
    assert_eq!(g.data[11] as i16, 0x1000); // IR3
    assert_eq!(g.data[19] as u16, 0x1000); // SZ3
    // n = (H*0x20000/SZ3 +1)/2 = 0x1000; SX2 = n*IR1+OFX >>16 = 0x10
    assert_eq!(g.data[14] & 0xFFFF, 0x10);
    assert_eq!(g.data[14] >> 16, 0x20);
}

#[test]
fn rtps_divide_overflow() {
    let mut g = identity();
    g.data[1] = 0x10; // VZ0 tiny → SZ3 small → division overflows
    g.ctrl[26] = 0x1000; // H >> SZ3*2
    g.command(0x01 | SF);
    assert_ne!(g.ctrl[31] & (1 << 17), 0, "divide overflow flag");
    assert_ne!(g.ctrl[31] & (1 << 31), 0, "master error flag");
}

#[test]
fn rtpt_pushes_three_verts() {
    let mut g = identity();
    for v in 0..3 {
        g.data[v * 2] = (0x300 << 16) | 0x80; // VXn=0x80, VYn=0x300
        g.data[v * 2 + 1] = 0x2000; // VZn
    }
    g.ctrl[26] = 0x400;
    g.command(0x30 | SF); // RTPT
    // 3 pushes: SZ1,SZ2,SZ3 = 0x2000; SZ0 still 0 (needs 4 pushes)
    assert_eq!(g.data[16] as u16, 0);
    assert_eq!(g.data[17] as u16, 0x2000);
    assert_eq!(g.data[18] as u16, 0x2000);
    assert_eq!(g.data[19] as u16, 0x2000);
}

#[test]
fn nclip_area() {
    let mut g = Gte::new();
    // SXY0=(0,0) SXY1=(0x100,0) SXY2=(0,0x100): MAC0 = 0x100*0x100
    g.data[12] = 0;
    g.data[13] = 0x100;
    g.data[14] = 0x0100_0000;
    g.command(0x06); // NCLIP
    assert_eq!(g.data[24] as i32, 0x1_0000);
}

#[test]
fn avsz3_average() {
    let mut g = Gte::new();
    g.data[17] = 0x100;
    g.data[18] = 0x200;
    g.data[19] = 0x300;
    g.ctrl[29] = 0x555; // ~1/3 scale
    g.command(0x2D); // AVSZ3
    let expect = (0x555i64 * (0x100 + 0x200 + 0x300)) >> 12;
    assert_eq!(g.data[7], expect as u32);
}

#[test]
fn sqr_vector() {
    let mut g = Gte::new();
    g.data[9] = 0x1000;
    g.data[10] = 0xF000u32; // -0x1000
    g.data[11] = 0x0800;
    g.command(0x28 | SF); // SQR sf=1
    assert_eq!(g.data[25] as i32, 0x1000); // MAC1 = 0x1000² >>12
    assert_eq!(g.data[26] as i32, 0x1000);
    assert_eq!(g.data[27] as i32, 0x0400); // 0x800²>>12
}

#[test]
fn op_outer_product() {
    let mut g = identity();
    g.data[9] = 0x100;
    g.data[10] = 0x200;
    g.data[11] = 0x300;
    g.command(0x0C | SF); // OP with D=diag(RT)=0x1000
    // MAC1 = (IR3*D2 - IR2*D3)>>12 = (0x300-0x200)*0x1000>>12 = 0x100
    assert_eq!(g.data[9] as i16, 0x100);
}

#[test]
fn mvmva_identity() {
    let mut g = identity();
    g.data[0] = (0x111 << 16) | 0x222;
    g.data[1] = 0x333;
    // MVMVA sf=1 mx=0(RT) v=0(V0) cv=0(TR) lm=0
    g.command(0x12 | SF);
    assert_eq!(g.data[9] as i16, 0x222);
    assert_eq!(g.data[10] as i16, 0x111);
    assert_eq!(g.data[11] as i16, 0x333);
}

#[test]
fn ir_saturation_lm() {
    let mut g = Gte::new();
    // force IR1 negative via MVMVA with negative TRX
    g.ctrl[5] = 0xFFFF_F000; // TRX = -0x1000 → *0x1000 >>12 = -0x1000
    // lm=0: IR1 = -0x1000 ok; lm=1: saturated to 0 with flag24
    g.command(0x12 | SF); // lm=0
    assert_eq!(g.data[9] as i16, -0x1000);
    g.command(0x12 | SF | (1 << 10)); // lm=1
    assert_eq!(g.data[9] as i16, 0);
    assert_ne!(g.ctrl[31] & (1 << 24), 0);
}

#[test]
fn irgb_orrgb() {
    let mut g = Gte::new();
    g.write_data(28, 0x7FFF); // IRGB: all channels 0x1F
    assert_eq!(g.data[9] as i16, 0x1F * 0x80);
    assert_eq!(g.data[10] as i16, 0x1F * 0x80);
    assert_eq!(g.data[11] as i16, 0x1F * 0x80);
    assert_eq!(g.read_data(29), 0x7FFF); // ORGB collapses back
}

#[test]
fn lzcs_lzcr() {
    let mut g = Gte::new();
    g.write_data(30, 0x00F0_0000);
    assert_eq!(g.read_data(31), 8);
    g.write_data(30, 0xFF00_0000u32);
    assert_eq!(g.read_data(31), 8); // leading ones
    g.write_data(30, 0);
    assert_eq!(g.read_data(31), 32);
}

#[test]
fn sxyp_push_only() {
    let mut g = Gte::new();
    g.data[12] = 0x11;
    g.data[13] = 0x22;
    g.data[14] = 0x33;
    // writing SXY1 directly must NOT push the fifo
    g.write_data(13, 0xAA);
    assert_eq!(g.data[12], 0x11);
    assert_eq!(g.data[13], 0xAA);
    assert_eq!(g.data[14], 0x33);
    // SXYP pushes
    g.write_data(15, 0x44);
    assert_eq!(g.data[12], 0xAA);
    assert_eq!(g.data[13], 0x33);
    assert_eq!(g.data[14], 0x44);
}

#[test]
fn color_fifo_push() {
    let mut g = Gte::new();
    // GPF: MAC = IR*IR0 → SAR → push RGB2 = MAC/16
    g.data[6] = 0x5500_0000; // CODE=0x55
    g.data[8] = 0x1000; // IR0=1.0
    g.data[9] = 0x1000;
    g.data[10] = 0x2000;
    g.data[11] = 0x3000;
    g.command(0x3D | SF); // GPF sf=1
    // MACn = IRn*0x1000 >>12 = IRn → RGB2 = IRn/16 (0x100,0x200,0x300
    // all exceed 0xFF → saturated + color fifo flags 21/20/19)
    assert_eq!(g.data[22] & 0xFF_FFFF, 0xFF_FFFF);
    assert_eq!(g.data[22] >> 24, 0x55);
    assert_eq!(g.ctrl[31] & (7 << 19), 7 << 19);
}
