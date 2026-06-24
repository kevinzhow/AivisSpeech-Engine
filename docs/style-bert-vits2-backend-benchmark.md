# Style-Bert-VITS2 Backend Benchmark

This benchmark compares the current AivisSpeech Style-Bert-VITS2 path across
ONNX Runtime CPU, ONNX Runtime CUDA, TTS.cpp ggml/Vulkan, and TTS.cpp
ggml/Metal. It also records native binding runs on Windows Intel Arc B580 and
macOS Apple M1 Pro. RTF is `elapsed_seconds / output_duration_seconds`; lower
is better.

## Scope

- Date: 2026-06-24 to 2026-06-25, Asia/Tokyo.
- Benchmark profile: `warm_steady_state`, with `--warmup_runs 1 --runs 3`.
- Style: `888753760` from the local `まお` model.
- ONNX model: AIVMX/ONNX baseline.
- ggml model: AIVM/Safetensors metadata plus preconverted synthesis GGUF and
  GGUF JP-BERT.
- ggml path: native TTS.cpp binding, `tts-cpp-jp-bert`, `synthesize-front`.
- SDP: `tempoDynamicsScale=0.0`.

| label | text | chars |
| --- | --- | ---: |
| short | `テストです。` | 6 |
| medium | `今日はいい天気ですね。` | 11 |
| long | `これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。` | 41 |

## Device Parameters

| component | value |
| --- | --- |
| OS | Ubuntu 26.04 LTS, kernel `7.0.0-22-generic` |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores / 16 threads |
| ONNX Runtime | `onnxruntime-gpu 1.26.0`; available providers: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` |
| ONNX CPU provider | Active provider: `CPUExecutionProvider` |
| ONNX CUDA provider | Active providers after model load: `CUDAExecutionProvider`, `CPUExecutionProvider` |
| ggml Vulkan iGPU | AMD Radeon 780M Graphics (RADV PHOENIX), integrated GPU, vendor `0x1002`, device `0x1900`, Vulkan API `1.4.335`, Mesa `26.0.3-1ubuntu1`, UMA `1`, fp16 `0`, bf16 `0`, warp size `64`, shared memory `65536`, int dot `1` |
| ggml Vulkan NVIDIA | NVIDIA GeForce RTX 3060, discrete GPU, vendor `0x10de`, device `0x2504`, Vulkan API `1.4.329`, driver `595.71.05`, VRAM `12288 MiB`, PCI bus `00000000:01:00.0`, power limit `170 W`, UMA `0`, fp16 `0`, bf16 `1`, warp size `32`, shared memory `49152`, int dot `1` |
| macOS Metal host | macOS 27.0 build `26A5368g`, Apple M1 Pro, 10 CPU cores, 32 GiB RAM |
| ggml Metal | Apple M1 Pro, TTS.cpp `8e26ac0`, ggml `b6ad57d8`, `BUILD_SHARED_LIBS=ON`, `GGML_METAL=ON`, `GGML_METAL_NO_RESIDENCY=1`, Style-Bert AOT simdgroup-half Metal ConvTranspose1D enabled by default |

The benchmark pinned the TTS.cpp Vulkan device with `GGML_VK_VISIBLE_DEVICES`.
The captured TTS.cpp device evidence was:

```text
ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX) (radv) | uma: 1 | fp16: 0 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 (NVIDIA) | uma: 0 | fp16: 0 | bf16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: none
```

## Backend Matrix

| backend | engine path | model/input | device |
| --- | --- | --- | --- |
| ONNX CPU | AivisSpeech `StyleBertVITS2TTSEngine`, `tts_backend=onnx`, `use_gpu=False` | `.aivmx` / ONNX | Ryzen 7 8845HS CPU |
| ONNX CUDA | AivisSpeech `StyleBertVITS2TTSEngine`, `tts_backend=onnx`, `use_gpu=True` | `.aivmx` / ONNX | RTX 3060 through `CUDAExecutionProvider` |
| ggml Vulkan iGPU | AivisSpeech ggml backend through native TTS.cpp C API | `.aivm` / Safetensors + synthesis GGUF + JP-BERT GGUF | AMD Radeon 780M, `GGML_VK_VISIBLE_DEVICES=0` |
| ggml Vulkan NVIDIA | AivisSpeech ggml backend through native TTS.cpp C API | `.aivm` / Safetensors + synthesis GGUF + JP-BERT GGUF | RTX 3060, `GGML_VK_VISIBLE_DEVICES=1` |
| ggml Metal native | AivisSpeech ggml backend through native TTS.cpp C API | `.aivm` / Safetensors metadata + `kevinzhow/style-bert-vits2-gguf` synthesis GGUF + JP-BERT GGUF | Apple M1 Pro Metal |

The ONNX CUDA run was accepted only after checking the loaded ONNX session's
actual providers. This matters because ONNX Runtime can expose
`CUDAExecutionProvider` but still fall back to CPU if compatible CUDA/cuDNN
runtime libraries are missing.

## Linux Vulkan Results

| text length | ONNX CPU RTF | ONNX CUDA RTF | ggml Vulkan AMD 780M RTF | ggml Vulkan RTX 3060 RTF |
| --- | ---: | ---: | ---: | ---: |
| short | `0.318` | `0.278` | `0.251` | `0.146` |
| medium | `0.287` | `0.177` | `0.197` | `0.102` |
| long | `0.208` | `0.067` | `0.136` | `0.063` |
| overall mean | `0.271` | `0.174` | `0.195` | `0.104` |

Interpretation:

- The current native TTS.cpp JP-BERT ggml/Vulkan path is below `0.2` overall on
  the AMD 780M iGPU, and below `0.2` on medium and long text.
- Short text is still the hardest case for iGPU because fixed frontend and call
  overhead is amortized over only about one second of audio.
- ONNX CUDA is very strong on long text. The RTX 3060 ggml/Vulkan native path is
  still the best overall result in this run and is effectively tied with ONNX
  CUDA on the long sentence.

This 2026-06-25 rerun pulled AivisSpeech-Engine to `1e079d9`, pulled TTS.cpp to
`f389a96`, updated the TTS.cpp `ggml` submodule to `a9b84478`, and rebuilt both
the native shared library and Vulkan `tts-server`. The table uses
`/tmp/aivis-style-bert-vits2-benchmark-20260625/amd780m.json` for
the ONNX CPU, ONNX CUDA, and AMD 780M ggml columns, and
`/tmp/aivis-style-bert-vits2-benchmark-20260625/rtx3060.json` for the RTX 3060
ggml column. The RTX 3060 run also repeated the ONNX baselines; those values are
stored in the JSON artifact and were close to the AMD-run baseline values.

## Linux Vulkan Audio Preview

Each preview is the `run00` AAC artifact for that backend and text. The full
benchmark generated three measured AAC runs per backend/text; `--audio_output_dir`
records each generated path in `records[].audio_path`. AAC encoding runs after
the synthesis timer, so it is not included in RTF.

| text length | ONNX CPU | ONNX CUDA | ggml Vulkan AMD 780M | ggml Vulkan RTX 3060 |
| --- | --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_short.m4a) |
| medium | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_medium.m4a) |
| long | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_long.m4a) |

## Windows Intel Arc B580 Native Binding Result

This run was captured on 2026-06-25, Asia/Tokyo, on Windows 11 with an
Intel(R) Arc(TM) B580 Graphics device using driver `32.0.101.8826`.

The local TTS.cpp checkout was updated to `ea100b4` (`Export Style-Bert-VITS2
native C API`) and built as a shared Vulkan library. The build required one
local CMake fix for shared Windows linking: `src/CMakeLists.txt` adds
`../ggml-patches/llama-mmap.cpp` to the `tts` target so `tts.dll` contains the
model loader's mmap implementation.

Build artifact:

```text
C:\Users\kevin\TTS.cpp\build-vulkan-native\bin\tts.dll
```

Benchmark artifact:

```text
C:\Users\kevin\run-logs\aivis-style-bert-vits2-b580-native-warm.json
```

The run used `--warmup_runs 1 --runs 3`, `tts-cpp-jp-bert`,
`synthesize-front`, `accurate` Vulkan precision, and
`--ggml_native_library_path C:\Users\kevin\TTS.cpp\build-vulkan-native\bin\tts.dll`.
The benchmark transport was `native-binding`; no sidecar process or HTTP
transport was used.

The captured Vulkan device evidence was:

```text
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Arc(TM) B580 Graphics (Intel Corporation) | uma: 0 | fp16: 0 | bf16: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: none
```

| text length | chars | ONNX CPU RTF | ggml Vulkan Intel Arc B580 native RTF | ggml/ONNX CPU RTF ratio |
| --- | ---: | ---: | ---: | ---: |
| short | 6 | `0.454` | `0.128` | `0.281` |
| medium | 10 | `0.375` | `0.105` | `0.281` |
| long | 41 | `0.270` | `0.055` | `0.203` |
| overall mean | - | `0.366` | `0.096` | `0.262` |

Native timing breakdown:

| text length | JP-BERT seconds | synthesis seconds | total frontend seconds | numeric payload |
| --- | ---: | ---: | ---: | ---: |
| short | `0.041` | `0.080` | `0.042` | `108.3 KiB` |
| medium | `0.059` | `0.102` | `0.060` | `156.5 KiB` |
| long | `0.077` | `0.331` | `0.080` | `782.3 KiB` |

Interpretation:

- The Windows Intel Arc B580 native binding path is active and faster than ONNX
  CPU in this run, with overall ggml/Vulkan RTF at `26.2%` of ONNX CPU.
- The native C API path excludes sidecar HTTP, base64 payload, and WAV decode
  overhead from the ggml timing path.
- Short text remains more overhead-sensitive than long text. The long text
  amortizes frontend and native call overhead best, reaching `0.055` RTF.

## macOS Metal Native Result

This run used the TTS.cpp native C API from `libtts.dylib`, not the managed
`tts-server` sidecar. It used `kevinzhow/style-bert-vits2-gguf` for the
preconverted `mao-full-sdp.gguf` and JP-BERT GGUF artifacts. The TTS.cpp build
includes the Style-Bert decoder-specific Metal AOT simdgroup-half
ConvTranspose1D path from TTS.cpp `8e26ac0` / ggml `b6ad57d8`. It decomposes
output by stride phase, maps each phase to a matrix multiply tile, uses half
input/weight tiles with fp32 accumulation, writes cropped output directly, and
fuses the bias add. The previous f32 phase-tiled path remains available with
`STYLE_BERT_VITS2_METAL_CONV_TRANSPOSE_1D_KERNEL=phase_32x32_k128`.

| text length | ONNX CPU RTF | ggml Metal native RTF | Metal/ONNX CPU ratio |
| --- | ---: | ---: | ---: |
| short | `0.297` | `0.215` | `0.724` |
| medium | `0.275` | `0.153` | `0.556` |
| long | `0.244` | `0.127` | `0.523` |
| overall mean | `0.272` | `0.165` | `0.607` |

Native timing breakdown for `ggml-metal-jp-bert-native`:

| text length | frontend seconds | native synthesis seconds | native JP-BERT seconds |
| --- | ---: | ---: | ---: |
| short | `0.056` | `0.150` | `0.055` |
| medium | `0.053` | `0.208` | `0.053` |
| long | `0.075` | `0.888` | `0.073` |

The result JSON is
`/tmp/aivis-style-bert-vits2-benchmark-metal-native-aot-simdgroup.json`; the log
is `/tmp/aivis-style-bert-vits2-benchmark-metal-native-aot-simdgroup.log`. The
log showed:

```text
ggml_metal_device_init: GPU name:   MTL0 (Apple M1 Pro)
ggml_metal_device_init: use residency sets    = false
ggml_metal_init: found device: Apple M1 Pro
Using TTS.cpp ggml/metal native binding at /Users/kevinzhow/Github/TTS.cpp/build-metal-shared/src/libtts.dylib.
kernel_style_bert_vits2_conv_transpose_1d_phase_simdgroup_half_aot_k16_s8_ic512_crop4_f32
kernel_style_bert_vits2_conv_transpose_1d_phase_simdgroup_half_aot_k16_s8_ic256_crop4_f32
kernel_style_bert_vits2_conv_transpose_1d_phase_simdgroup_half_aot_k8_s2_ic128_crop3_f32
kernel_style_bert_vits2_conv_transpose_1d_phase_simdgroup_half_aot_k2_s2_ic64_crop0_f32
kernel_style_bert_vits2_conv_transpose_1d_phase_simdgroup_half_aot_k2_s2_ic32_crop0_f32
```

The AOT simdgroup-half ConvTranspose1D path is enabled by default when the native
binding sets `TTS_BACKEND=metal` on a Metal device with simdgroup matrix support.
The f32 phase-tiled and scalar fused paths are retained as explicit rollback
modes. A one-text native binding probe with `--warmup_runs 1 --runs 3` on the
same long input compared the f32 phase-tiled path with the new default:

| Metal ConvTranspose1D path | RTF | native synthesis seconds | decoder seconds | decoder nodes |
| --- | ---: | ---: | ---: | ---: |
| `STYLE_BERT_VITS2_METAL_CONV_TRANSPOSE_1D_KERNEL=phase_32x32_k128` | `0.141` | `1.454` | `0.993` | `428` |
| default AOT simdgroup-half | `0.135` | `1.384` | `0.939` | `428` |

For that probe, the default AOT simdgroup-half path reduced native synthesis
time by `4.8%`, RTF by `4.4%`, and decoder time by `5.4%` compared with the f32
phase-tiled path. The final decoder fixture stayed within tolerance:
`max_abs=0.000594173`, `rms=7.4558e-05`. The f32 phase-tiled rollback fixture
reported `max_abs=0.000498002`, `rms=7.29653e-05`.

## Reproduction

The ONNX CUDA baseline on this host required CUDA 12/cuDNN 9 compatible runtime
libraries. The system CUDA installation was not sufficient for
`onnxruntime-gpu 1.26.0`, so the benchmark prepended the CUDA 12 NVIDIA wheel
libraries from the local qwen3-tts.cpp virtualenv:

```bash
CUDA12_NVIDIA_LIBS="$(
  find /home/kevinzhow/github/qwen3-tts.cpp/.venv/lib/python3.13/site-packages/nvidia \
    -maxdepth 3 \
    -type d \
    -name lib \
    | paste -sd: -
)"
```

### Model Artifacts

The benchmark uses the public `まお` model:

| field | value |
| --- | --- |
| AivisHub UUID | `a59cb814-0083-4369-8542-f51a29e72af7` |
| version | `1.2.0` |
| AIVM/Safetensors size | `259776543` bytes |
| AIVMX/ONNX size | `258037076` bytes |

AivisHub exposes both source formats for the same model UUID. `AIVMX` is the
ONNX baseline input. `AIVM` is the Safetensors metadata/source package used by
the benchmark and by GGUF rebuilds:

```bash
MAO_MODEL_UUID=a59cb814-0083-4369-8542-f51a29e72af7

curl -L \
  -o /path/to/mao.aivm \
  "https://api.aivis-project.com/v1/aivm-models/${MAO_MODEL_UUID}/download?model_type=AIVM"

curl -L \
  -o /path/to/mao.aivmx \
  "https://api.aivis-project.com/v1/aivm-models/${MAO_MODEL_UUID}/download?model_type=AIVMX"
```

The synthesis and JP-BERT GGUF artifacts should be downloaded from the
preconverted Hugging Face bundle:

```bash
hf download kevinzhow/style-bert-vits2-gguf \
  voices/mao-full-sdp.gguf \
  frontend/style-bert-vits2-jp-bert.gguf \
  --local-dir /path/to/TTS.cpp/tmp/style-bert-vits2-gguf
```

| artifact | size | SHA-256 |
| --- | ---: | --- |
| `voices/mao-full-sdp.gguf` | `251099936` bytes | `51dd69888d62f16a54d48732cbfe789f326bc4192bd8f6b2876f8ed0b6807f71` |
| `frontend/style-bert-vits2-jp-bert.gguf` | `1314386784` bytes | `e10f4de90fb9f1aadbf2e5f79453406ff60a4fe77b6a4d314b1ee226118ecebf` |

If the GGUF needs to be rebuilt from source, use TTS.cpp's
`py-gguf/convert_style_bert_vits2_to_gguf` or the Engine's `AivmGgufCache`.
For normal benchmark reproduction, use the preconverted HF artifacts above.

AMD 780M iGPU run:

```bash
LD_LIBRARY_PATH="${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GGML_VK_VISIBLE_DEVICES=0 \
uv run python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path /home/kevinzhow/github/kokoro-tts/tmp/style-bert-vits2-assets/_downloads/mao.aivm \
  --aivmx_path /home/kevinzhow/github/kokoro-tts/tmp/aivisspeech-engine-data/Models/a59cb814-0083-4369-8542-f51a29e72af7.aivmx \
  --gguf_path /home/kevinzhow/github/TTS.cpp/tmp/style-bert-vits2-voices/mao-full-sdp.gguf \
  --jp_bert_gguf_path /home/kevinzhow/github/TTS.cpp/tmp/style-bert-vits2-jp-bert.gguf \
  --tts_server_path /home/kevinzhow/github/TTS.cpp/build-vulkan-latest-main/bin/tts-server \
  --ggml_native_library_path /home/kevinzhow/github/TTS.cpp/build-native-binding-shared/src/libtts.so \
  --onnx_baseline cpu \
  --onnx_baseline cuda \
  --ggml_backend vulkan \
  --ggml_frontend tts-cpp-jp-bert \
  --ggml_synthesis_endpoint synthesize-front \
  --ggml_vulkan_precision accurate \
  --style_id 888753760 \
  --text 'テストです。' \
  --text '今日はいい天気ですね。' \
  --text 'これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。' \
  --warmup_runs 1 \
  --runs 3 \
  --audio_output_dir /tmp/aivis-style-bert-vits2-benchmark-20260625/amd780m-audio \
  --output_json /tmp/aivis-style-bert-vits2-benchmark-20260625/amd780m.json \
  > /tmp/aivis-style-bert-vits2-benchmark-20260625/amd780m.log 2>&1
```

RTX 3060 Vulkan run:

```bash
LD_LIBRARY_PATH="${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GGML_VK_VISIBLE_DEVICES=1 \
uv run python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path /home/kevinzhow/github/kokoro-tts/tmp/style-bert-vits2-assets/_downloads/mao.aivm \
  --aivmx_path /home/kevinzhow/github/kokoro-tts/tmp/aivisspeech-engine-data/Models/a59cb814-0083-4369-8542-f51a29e72af7.aivmx \
  --gguf_path /home/kevinzhow/github/TTS.cpp/tmp/style-bert-vits2-voices/mao-full-sdp.gguf \
  --jp_bert_gguf_path /home/kevinzhow/github/TTS.cpp/tmp/style-bert-vits2-jp-bert.gguf \
  --tts_server_path /home/kevinzhow/github/TTS.cpp/build-vulkan-latest-main/bin/tts-server \
  --ggml_native_library_path /home/kevinzhow/github/TTS.cpp/build-native-binding-shared/src/libtts.so \
  --onnx_baseline cpu \
  --onnx_baseline cuda \
  --ggml_backend vulkan \
  --ggml_frontend tts-cpp-jp-bert \
  --ggml_synthesis_endpoint synthesize-front \
  --ggml_vulkan_precision accurate \
  --style_id 888753760 \
  --text 'テストです。' \
  --text '今日はいい天気ですね。' \
  --text 'これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。' \
  --warmup_runs 1 \
  --runs 3 \
  --audio_output_dir /tmp/aivis-style-bert-vits2-benchmark-20260625/rtx3060-audio \
  --output_json /tmp/aivis-style-bert-vits2-benchmark-20260625/rtx3060.json \
  > /tmp/aivis-style-bert-vits2-benchmark-20260625/rtx3060.log 2>&1
```

macOS Metal local run:

```bash
git -C /Users/kevinzhow/Github/TTS.cpp pull --ff-only origin main
git -C /Users/kevinzhow/Github/TTS.cpp submodule update --init --recursive

cmake -S /Users/kevinzhow/Github/TTS.cpp \
  -B /Users/kevinzhow/Github/TTS.cpp/build-metal-shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DBUILD_SHARED_LIBS=ON

cmake --build /Users/kevinzhow/Github/TTS.cpp/build-metal-shared \
  --target tts \
  -j "$(sysctl -n hw.ncpu)"

DYLD_LIBRARY_PATH="/Users/kevinzhow/Github/TTS.cpp/build-metal-shared/src:/Users/kevinzhow/Github/TTS.cpp/build-metal-shared/ggml/src:/Users/kevinzhow/Github/TTS.cpp/build-metal-shared/ggml/src/ggml-blas:/Users/kevinzhow/Github/TTS.cpp/build-metal-shared/ggml/src/ggml-metal:/Users/kevinzhow/Github/TTS.cpp/build-metal-shared/ggml/src/ggml-cpu" \
GGML_METAL_NO_RESIDENCY=1 \
STYLE_BERT_VITS2_DEBUG_TIMINGS=1 \
uv run --group dev python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path /Users/kevinzhow/.Trash/まお.aivm \
  --aivmx_path "/Users/kevinzhow/Library/Application Support/AivisSpeech-Engine/Models/a59cb814-0083-4369-8542-f51a29e72af7.aivmx" \
  --gguf_path /Users/kevinzhow/Github/TTS.cpp/tmp/style-bert-vits2-gguf/voices/mao-full-sdp.gguf \
  --jp_bert_gguf_path /Users/kevinzhow/Github/TTS.cpp/tmp/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert.gguf \
  --tts_server_path /Users/kevinzhow/Github/TTS.cpp/build-metal/bin/tts-server \
  --ggml_native_library_path /Users/kevinzhow/Github/TTS.cpp/build-metal-shared/src/libtts.dylib \
  --onnx_baseline cpu \
  --ggml_backend metal \
  --ggml_frontend tts-cpp-jp-bert \
  --ggml_synthesis_endpoint synthesize-front \
  --style_id 888753760 \
  --text 'テストです。' \
  --text '今日はいい天気ですね。' \
  --text 'これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。' \
  --warmup_runs 1 \
  --runs 3 \
  --output_json /tmp/aivis-style-bert-vits2-benchmark-metal-native-aot-simdgroup.json \
  > /tmp/aivis-style-bert-vits2-benchmark-metal-native-aot-simdgroup.log 2>&1
```

The Metal native run is labeled as `ggml-metal-jp-bert-native` in the JSON
report. `GGML_METAL_NO_RESIDENCY=1` avoids a ggml Metal process-exit assert
observed on the macOS 27.0 / Apple M1 Pro local test host.
