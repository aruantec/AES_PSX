# AruanPSX

AES Oxide - PlayStation (PSX) emulator core in Rust

A standalone library crate — the host drives `Emulator::run_frame`,
pulls the display framebuffer and 44.1 kHz stereo audio, and feeds
controller state back in.

Implemented from public hardware documentation (PSX architecture
notes).

## Features

- R3000A interpreter with load/branch delay slots and COP0 exceptions
- GTE command set and register-file behavior
- Scanline-level video timing and a native-resolution software GPU
- Optional WGPU hardware polygon renderer for enhanced internal
  resolutions; draw commands rasterize on the host GPU and native VRAM
  is resolved lazily when CPU readback or framebuffer-texture feedback
  requires it. `Emulator::framebuffer()` synchronizes before reading;
  code reading `bus.gpu.vram` directly should call
  `bus.gpu.sync_native_vram()` first.
- CD-ROM, XA-ADPCM/CD-DA audio, SPU voices/ADSR/noise/reverb, MDEC,
  DMA, timers, controllers, and memory cards
- CPAL audio output uses a lock-free single-producer/single-consumer
  queue with adaptive resampling to the host device rate
- Real BIOS boot of retail discs, verified with Crash Bandicoot: Warped

## GPU Player

Use `play_gpu` for hardware rendering, audio, and gamepad support:

```sh
cargo run --release --example play_gpu -- <game.cue> [bios.bin] --res=4x
```

`--res` controls internal polygon rendering, not the host window.
Supported scales are `1x`, `2x`, `4x`, `6x`, and `8x`; a `WIDTHxHEIGHT`
value such as `3840x2160` selects the nearest supported scale. For a
typical 512x240 game mode, `3840x2160` selects `8x`.
`--window-size=WIDTHxHEIGHT` changes only the window. `--sw` forces
software rendering, `--fast` skips the BIOS shell, and `--dynarec`
enables the experimental CPU translator. `--texture-filter=nearest`
or `--texture-filter=bilinear` selects polygon texture filtering.

Controls: arrows = D-pad; Z/X/A/S = Cross/Circle/Square/Triangle;
Q/W = L1/R1; E/R = L2/R2; Enter = Start; Backspace = Select;
Tab = fast-forward; Esc = quit. `F` toggles display scaling
(smooth/sharp), `T` toggles polygon texture filtering, and `D`
toggles PS1 dithering. Gamepads are also supported.

Nearest texture sampling preserves the PS1 look. Bilinear filtering
filters decoded texture colors while clamping samples to each
primitive's UV bounds; it smooths low-resolution textures but can
soften pixel-art details. At `4x`, `6x`, and `8x`, bilinear is selected by
default; use `--texture-filter=nearest` or press `T` to change it.
`F` affects only final display scaling.

The hardware renderer rasterizes polygons on the host GPU. FSR 3/4 is
not used: temporal upscalers need motion vectors and frame history
that this PS1 renderer does not provide. Higher internal resolution
improves polygon edges; it does not add geometry or detail to the
original models and textures.

## Building

```sh
cargo test --release --offline
cargo run --release --example headless -- <bios> [cue] --frames 600
```

The original `play` frontend uses minifb; `play_gpu` uses WGPU for
hardware rendering and presentation. `headless` runs without a window
and writes PNG screenshots. Boot tests skip when dumps are unavailable;
set `PSX_BIOS` and `PSX_GAME` to run the real BIOS/disc test.

## License

MIT
