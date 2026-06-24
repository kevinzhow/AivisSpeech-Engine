# Style-Bert-VITS2 Backend Benchmark

This benchmark compares the current AivisSpeech Style-Bert-VITS2 path across
ONNX Runtime CPU, ONNX Runtime CUDA, and TTS.cpp ggml/Vulkan on both the local
AMD iGPU and NVIDIA dGPU. It also records a Windows Intel Arc B580 sidecar run.
RTF is `elapsed_seconds / output_duration_seconds`; lower is better.

## Scope

- Date: 2026-06-24, Asia/Tokyo.
- Benchmark profile: `warm_steady_state`, with `--warmup_runs 1 --runs 3`.
- Style: `888753760` from the local `まお` model.
- ONNX model: AIVMX/ONNX baseline.
- ggml model: AIVM/Safetensors source plus GGUF synthesis and GGUF JP-BERT.
- ggml path: native TTS.cpp binding, `tts-cpp-jp-bert`, `synthesize-front`,
  `accurate` Vulkan precision.
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

The ONNX CUDA run was accepted only after checking the loaded ONNX session's
actual providers. This matters because ONNX Runtime can expose
`CUDAExecutionProvider` but still fall back to CPU if compatible CUDA/cuDNN
runtime libraries are missing.

## Results

| text length | ONNX CPU RTF | ONNX CUDA RTF | ggml Vulkan AMD 780M RTF | ggml Vulkan RTX 3060 RTF |
| --- | ---: | ---: | ---: | ---: |
| short | `0.371` | `0.271` | `0.228` | `0.147` |
| medium | `0.281` | `0.178` | `0.180` | `0.102` |
| long | `0.203` | `0.067` | `0.128` | `0.064` |
| overall mean | `0.285` | `0.172` | `0.179` | `0.104` |

Interpretation:

- The current native TTS.cpp JP-BERT ggml/Vulkan path is below `0.2` overall on
  the AMD 780M iGPU, and below `0.2` on medium and long text.
- Short text is still the hardest case for iGPU because fixed frontend and call
  overhead is amortized over only about one second of audio.
- ONNX CUDA is very strong on long text. The RTX 3060 ggml/Vulkan native path is
  still the best overall result in this run and is effectively tied with ONNX
  CUDA on the long sentence.

The table uses `/tmp/aivis-style-bert-vits2-benchmark-amd780m-final.json` for
the ONNX CPU, ONNX CUDA, and AMD 780M ggml columns, and
`/tmp/aivis-style-bert-vits2-benchmark-rtx3060-final.json` for the RTX 3060
ggml column. The RTX 3060 run also repeated the ONNX baselines; those values are
stored in the JSON artifact and were close to the AMD-run baseline values.

## Windows Intel Arc B580 Sidecar Result

This run was captured on 2026-06-25, Asia/Tokyo, on Windows 11 with an
Intel(R) Arc(TM) B580 Graphics device using driver `32.0.101.8826`. The local
TTS.cpp build had `tts-server.exe` but did not have a native shared library
(`tts.dll`), so this result uses the managed sidecar HTTP transport rather than
the native TTS.cpp C API used in the Linux table above. It is useful as local
device evidence and a sidecar-path benchmark, but should not be mixed directly
with the native-binding rows.

The run used `--warmup_runs 1 --runs 3`, `tts-cpp-jp-bert`,
`synthesize-front`, `accurate` Vulkan precision, and the default base64 BERT
payload format. The JSON artifact is:

```text
C:\Users\kevin\run-logs\aivis-style-bert-vits2-b580-sidecar-warm.json
```

The captured TTS.cpp device evidence was:

```text
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Arc(TM) B580 Graphics (Intel Corporation) | uma: 0 | fp16: 0 | bf16: 0 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: none
```

| text length | chars | ONNX CPU RTF | ggml Vulkan Intel Arc B580 sidecar RTF | ggml/ONNX CPU RTF ratio |
| --- | ---: | ---: | ---: | ---: |
| short | 6 | `0.444` | `0.184` | `0.414` |
| medium | 10 | `0.377` | `0.156` | `0.413` |
| long | 41 | `0.276` | `0.069` | `0.251` |
| overall mean | - | `0.366` | `0.136` | `0.373` |

Interpretation:

- The Windows Intel Arc B580 sidecar path is active and faster than ONNX CPU in
  this run, with overall ggml/Vulkan RTF at `37.3%` of ONNX CPU.
- Short text remains more overhead-sensitive than long text. The long text
  amortizes sidecar and payload overhead best, reaching `0.069` RTF.
- Because this run uses sidecar HTTP transport, the numbers include local JSON,
  base64 BERT payload, WAV decode, and sidecar request overhead.

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
  --output_json /tmp/aivis-style-bert-vits2-benchmark-amd780m-final.json \
  > /tmp/aivis-style-bert-vits2-benchmark-amd780m-final.log 2>&1
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
  --output_json /tmp/aivis-style-bert-vits2-benchmark-rtx3060-final.json \
  > /tmp/aivis-style-bert-vits2-benchmark-rtx3060-final.log 2>&1
```
