# AruanPSX

AES Oxide - PlayStation (PSX) emulator core in Rust

A standalone library crate — the host drives `Emulator::run_frame`,
pulls the display framebuffer and 44.1 kHz stereo audio, and feeds
controller state back in.

Implemented from public hardware documentation (PSX architecture
notes).

## Current state

- R3000A interpreter (load/branch delay slots, cop0 exceptions)
- GTE with the full command set and register-file quirks
- Scanline-level video timing, software GPU rasterizer
- CD-ROM with real BIOS boot, XA-ADPCM + CD-DA audio
- SPU: ADPCM voices, ADSR, noise, reverb, CD mixing, IRQ
- MDEC decoder (FMV), DMA, timers, SIO0 pads + memory cards
- Real BIOS boots retail games end to end (verified with
  Crash Bandicoot: Warped)

## Building

```sh
cargo test --release            # unit + boot tests (needs a BIOS dump)
cargo run --release --example play -- <game.cue> [bios.bin] [--fast]
cargo run --release --example headless -- <bios> [cue] --frames 600
```

`play` is a windowed frontend (minifb + cpal + gilrs); `headless`
runs without a window and writes PNG screenshots. Boot tests skip
themselves when no BIOS/game dumps are present — point them at
images with `PSX_BIOS` / `PSX_GAME`.

## License

MIT
