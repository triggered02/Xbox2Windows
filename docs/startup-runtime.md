# Startup runtime stand-ins (experimental)

This page describes the experimental desktop stand-ins under `src/startup_stubs/` and the tools
under `tools/startup/`. They let a console executable start on desktop Windows under a debugger,
so that its startup path can be studied one missing piece at a time.

This is not a port and not an emulator. No executable is playable. Every stand-in is opt-in, and
most answers are shortcuts that are logged as such. The open problems are tracked in the issues
whose titles start with `startup-runtime:`.

## Output window and fallback diagnostics

The desktop swap chain has its own top-level window, class `X2WOutputWindow`.
When a present cannot show its surface, the GDI fallback selects a visible,
unowned guest window in the same process. If only the presenter is visible,
the fallback leaves it alone. To exercise both z-orders with source-built
windows, build `startup_overlay_window` and run
`ctest --preset release --no-tests=error -R '^startup_overlay_window$' -V`.

## Null Texture2D descriptors

When the adapter lacks typed UAV support for B5G5R5A1_UNORM or render-target support for
B4G4R4A4_UNORM, the wrapper uses an R8G8B8A8_UNORM null descriptor if that format supports the
requested view. This applies only to Texture2D views with no resource or counter. Run
`ctest --test-dir out/bin --no-tests=error -R "^startup_null_views_" -V` against your build
directory to exercise both slots on hardware and WARP.

## What is measured

With the switches below, an x64 console executable on an AMD RDNA4 GPU (Windows 11, release
build of this branch):

- presents frames at about 29 fps through the D3D12 stand-in, and its translated UI draws reach the
  output window;
- reads the virtual gamepad and answers the first-run user prompt for one local user;
- streams data through the DirectStorage stand-in (about 20 000 requests in the first two minutes,
  zlib-inflated on the CPU);
- plays its intro video through the GPU-view mirror (colours are still wrong, see the issues);
- then stops in its texture-streaming path: either at a soft assert (an `int3` the title steps
  over when no debugger is attached), or, past that, on a black screen.

## Build

The stand-ins are part of the normal build. From a developer prompt:

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release --no-tests=error -R '^startup_stubs$' -V
```

The build produces `xmem.dll`, `d3d12_xs.dll`, `AcpHal.dll`, `XCRDAPI.dll`, `chad.dll`, `xapu.dll`,
`xg_xs.dll` and `GameInput.dll`. Without the switches, every export that has no implementation
traps: it logs `x2w-trap <dll>!<export>` with its argument registers and ends the process with a
code that names the export (see `src/startup_stubs/trap.h`).

## Run

Prepare a folder with the executable and its package DLLs yourself. Never commit or upload it.
Then:

```powershell
.\tools\startup\run.ps1 -Exe <folder>\<title>.exe -DataDir <package root> -LogDir out\startup\run1 `
    -Seconds 180 -Render -FakeUser -GpuMirror -ContinueAsserts 1000 -InputScript "4@70"
```

`run.ps1` copies the stand-ins next to the executable and starts `tools/startup/debug_launch.py`.
The debugger runs the process in a kill-on-close job with a memory cap, **offline**: file opens of
socket, NetBIOS and DNS devices are refused (`--offline 0` turns this off; do not). It writes:

| File | Content |
|---|---|
| `<log>.out` | why the run stopped: exception, registers, stack, likely return addresses, raw stack |
| `<log>/events.txt` | module loads and every exception with its time |
| `<log>/stderr.txt` | the stand-ins' logs (`x2w-d3d12`, `x2w-dstorage`, `x2w-xmem`, `x2w-runtime`, ...) |
| `<log>/files.txt` | every file the process opened |

`tools/startup/live_view.py` follows a run in a second console. `X2W_TRAP_DUMP="exe+RVA:len,reg+off:len"`
adds memory dumps to the stop report.

## Switches

| Variable | Effect |
|---|---|
| `X2W_XMEM_PERMISSIVE=1` | accepts the reservation, mapping and allocation shapes a startup run uses |
| `X2W_XMEM_GPU_MIRROR=1` | pairs a no-access graphics reservation with the read-write reservation of the same size made just before it, and maps every physical page into both (same section) |
| `X2W_D3D12_PERMISSIVE=1` | wraps the device: recording command lists, console present, view tracking |
| `X2W_D3D12_RENDER=1` | draws through translated pipelines from `X2W_SHADER_CACHE` |
| `X2W_SHADER_CACHE=<dir>` | translation cache: the runtime writes `req_<key>.{a,b,root}` for untranslated pipelines; `tools/startup/gfx2hlsl.py --pending <dir>` translates them |
| `X2W_DRAW_NULL_IDENTITY=1` | null descriptors read as identity values instead of zero |
| `X2W_D3D12_DUMP_SURFACE=<bmp>` | writes the last presented surface |
| `X2W_DRAW_DUMP_FRAME=<n>`, `X2W_DRAW_DUMP_DIR` | Full-resolution RGB snapshots of single-sample RGBA8/BGRA8 UNORM or sRGB targets after translated draws. A later present writes completed queue-fenced copies. |
| `X2W_AUDIO_PERMISSIVE=1` | audio HAL / APU stand-ins accept their calls (no output) |
| `X2W_RUNTIME_PROXY=1` | proxies the gaming runtime and logs every call |
| `X2W_FAKE_USER=1` | answers the user, gamertag and save APIs for one local user |
| `X2W_STORAGE_PATH`, `X2W_STORAGE_MB`, `X2W_TITLE_ID` | persistent local storage and the manifest values the runtime would return |
| `X2W_INPUT_SCRIPT=mask@sec,...` | timed virtual gamepad presses (0x4 = A); keyboard and XInput pad 0 also work |
| `X2W_SAMPLER_HEAP=1` | `gfx2hlsl.py` translates a sampler-heap load with a run-time offset into `SamplerState qh[2048] : register(s0, space1)` and writes `smpheap 2048 1 fallback 2047`. The stand-in uses entries 0 to 2046 for the bound heap and reserves entry 2047 for unaligned or out-of-range offsets. It also accepts the original `smpheap 2048 1` manifest with all 2,048 entries available. [#1143](https://github.com/R0Y00/Xbox2Windows/issues/1143) |
| `X2W_XG_VERBOSE=1` | logs every texture layout query with its mip sizes |
| `X2W_XG_BC7_4K_S_LAYOUT=1` | enables the BC7 tiled layout and diagnostic copy when `X2W_D3D12_PERMISSIVE=1` |
| `X2W_TRAP_CONTINUE=<exports>` | lets listed trap exports return 0 instead of ending the process |

## Findings

These are measured on one executable; treat them as hypotheses for others.

- **User API, second interface.** The user provider hands out a second interface for the
  gamertag (method 3: handle, component, size, buffer, used). It must be answered separately from
  the main user object, whose method 3 is the handle duplicate. Without a gamertag, the first-run
  prompt offers no player and never advances.
- **GPU-view reservations.** The title reserves the same large range twice in a row: read-write,
  then no-access with the graphics flag. It maps physical pages only into the first, yet places GPU
  resources (for example the video frame planes) at the same offset in the second. On the console
  the second is a GPU view of the first. Mapping each page's section into both views makes those
  resources readable; without it, a YUV video shows a solid green screen (all-zero planes).
- **DirectStorage ordering.** Requests must run after `Submit`, in order with status and fence
  entries. Titles fill destination page-array entries after `EnqueueRequest`; an entry that is 0
  at enqueue is read again when the request runs.
- **Sub-dword addressing (SDWA).** The translator lowers the byte and word selects of the 32-bit
  integer VOP1/VOP2 SDWA forms: a source selection is zero-extended, or sign-extended under `sext()`,
  and the low bits of the result are placed in the destination field, with `UNUSED_PAD` zeroing the
  others and `UNUSED_PRESERVE` keeping them. The selection helpers match `tools/pixel_aot.py`, which
  already lowers the same forms. Float SDWA forms and a `dst_unused:UNUSED_SEXT` destination still
  stop translation rather than translating to something inexact. [#1141](https://github.com/R0Y00/Xbox2Windows/issues/1141)
- **Soft asserts.** Retail code keeps `int3` instructions followed by a continuation. Without a
  debugger the title's handler steps over them; under the startup debugger use
  `-ContinueAsserts N`. One of them fires about ten times a second in the texture-streaming path
  (a chunked read whose length ends inside a compressed chunk); its cause is open.
- **Struct-returning methods.** When both register arguments of a wrapped method look like objects,
  the one on the calling thread's stack is the return buffer.
- **Inline samplers.** Pixel shaders build sampler descriptors from constants (`s_movk_i32`,
  `s_bfm_b32`, 64-bit moves); the translator folds those into static samplers.
- **Stream-cache identity.** A cached zlib stream is served only to the exact queued
  continuation entries in the same DirectStorage queue. The lookup checks the enqueued request
  id, the source, the compressed size and the destination chunk size. A later independent
  request at a matching offset still decodes its own bytes or fails without touching its
  destination.

## Shortcuts that are known to be wrong

- BCPack requests are completed with zero-filled destinations (blank textures).
- The XG stand-in normally uses desktop copyable footprints. Its opt-in BC7 diagnostic
  reports standard GFX10 4 KiB tiled geometry for a single-slice, single-mip, non-MSAA
  `BC7_UNORM` 2D texture with an unknown layout, default alignment, and no resource flags.
  The diagnostic selects the profile explicitly. Native XG selector values still need
  independent evidence. `CopyIntoSubresource` writes the visible BC7 blocks from pitched source rows
  and preserves tile padding. With `X2W_D3D12_RENDER=1`, it returns `E_NOTIMPL` pending a detiler for
  the upload path. The reported pitch describes the padded block-row extent of the surface.
  The synthetic copy check runs with `ctest --preset dev --no-tests=error -R '^startup_xg_bc7_layout$' -V`.
- The default translator uses a static default sampler for a dynamically indexed sampler-heap load.
  With the sampler-heap opt-in, translated draws snapshot the bound heap's 16-byte sampler entries and
  bind a private desktop table beside the static samplers. Missing, out-of-range, or unreadable entries
  use a transparent-black border sampler and have separate counters. The runtime keeps recorded tables
  alive through queue completion. Constant sampler-heap offsets still report an unsupported operation.
  Wrapped sampler heaps keep the 16-byte title stride separate from the desktop descriptor increment
  for CPU handles and descriptor copies.
  The source-built readback uses `ctest --preset dev --no-tests=error -R '^startup_sampler_heap_' -V`.
- Slot 86 graphics streams can submit VS/PS requests to the offline cache. The fixture sends
  complete shader containers and a wrapped root through WARP, runs `gfx2hlsl.py --pending`, and
  reloads the compiled stages. A second fixture draws a clipped triangle through the translated
  pipeline, submits the wrapped list, waits for its fence and checks all 32 RGBA pixels on WARP and
  hardware. Swapping the red/green shader changes the 16 covered pixels. Reproduce with
  `ctest --preset release --no-tests=error -R "^startup_stream_(request|draw)_"`.
  Other stage combinations and full stream rasterizer/depth state remain open.
- Translating the captured console colour-blend register state blocks remains open (#1216).
  A graphics stream can supply a desktop `D3D12_BLEND_DESC` directly. `src/startup_stubs/cb_blend.h`
  decodes `CB_BLENDn_CONTROL` and a `CB_TARGET_MASK` nibble into `D3D12_RENDER_TARGET_BLEND_DESC`. It
  refuses forms D3D12 can't express (`BOTH_SRC_ALPHA`, constant alpha without alpha blend factor
  support, MIN/MAX with factors other than ONE). Its draws match a CPU model of the register fields
  (`ctest --preset release --no-tests=error -R "^startup_cb_blend"`). Where those registers sit in the
  state block is not measured yet, so the draw path doesn't call it.
- Audio produces no output.

## Uniform first-lane reads

The startup shader translator accepts `v_readfirstlane_b32` in straight-line code with the
entry mask when its active VGPR holds an integer conversion of a fixed-offset scalar buffer
value. Run `python -B tests/startup_uniform_lane_check.py` to check the producer and the
varying-source, register-clobber, mask and branch controls.

## NV12 plane upload

For a single-mip, single-sample, even-sized 2D NV12 image the desktop stand-in
reports two linear footprints. XG copies luma and interleaved chroma separately,
and the draw uploader addresses both D3D12 subresources. NV12 arrays remain
unsupported. The source-built draw compares a synthetic image and its U/V-swapped
control against the same integer colour conversion on the CPU. This verifies
the stand-in path; the texture format used in the reported video remains to be confirmed.

```powershell
cmake --build --preset release --target startup_nv12_draw d3d12_xs_shim
ctest --preset release --no-tests=error -R '^startup_nv12_draw_' -V
```
