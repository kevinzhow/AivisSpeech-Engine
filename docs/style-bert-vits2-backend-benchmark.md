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
  GGUF JP-BERT from Hugging Face `kevinzhow/style-bert-vits2-gguf`.
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
| ggml Metal | Apple M1 Pro, TTS.cpp `a120f9e`, ggml `a78c352b`, `BUILD_SHARED_LIBS=ON`, `GGML_METAL=ON`, `GGML_METAL_NO_RESIDENCY=1`, Style-Bert AOT simdgroup-half Metal ConvTranspose1D and fused decoder Conv1D epilogue enabled by default |

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
| ggml Vulkan iGPU | AivisSpeech ggml backend through native TTS.cpp C API | `.aivm` / Safetensors + `kevinzhow/style-bert-vits2-gguf` synthesis GGUF + JP-BERT GGUF | AMD Radeon 780M, `GGML_VK_VISIBLE_DEVICES=0` |
| ggml Vulkan NVIDIA | AivisSpeech ggml backend through native TTS.cpp C API | `.aivm` / Safetensors + `kevinzhow/style-bert-vits2-gguf` synthesis GGUF + JP-BERT GGUF | RTX 3060, `GGML_VK_VISIBLE_DEVICES=1` |
| ggml Metal native | AivisSpeech ggml backend through native TTS.cpp C API | `.aivm` / Safetensors metadata + `kevinzhow/style-bert-vits2-gguf` synthesis GGUF + JP-BERT GGUF | Apple M1 Pro Metal |

The ONNX CUDA run was accepted only after checking the loaded ONNX session's
actual providers. This matters because ONNX Runtime can expose
`CUDAExecutionProvider` but still fall back to CPU if compatible CUDA/cuDNN
runtime libraries are missing.

## Linux Vulkan Results

| text length | ONNX CPU RTF | ONNX CUDA RTF | ggml Vulkan AMD 780M RTF | ggml Vulkan RTX 3060 RTF |
| --- | ---: | ---: | ---: | ---: |
| short | `0.350` | `0.267` | `0.222` | `0.147` |
| medium | `0.289` | `0.184` | `0.182` | `0.102` |
| long | `0.209` | `0.065` | `0.131` | `0.063` |
| overall mean | `0.283` | `0.172` | `0.178` | `0.104` |

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
the native shared library and Vulkan `tts-server`. The RTF table is from
no-audio benchmark runs so that AAC encoding does not perturb the AMD APU's
CPU/iGPU shared power and memory budget. The ONNX CPU, ONNX CUDA, and AMD 780M
ggml columns came from the AMD 780M JSON report; the RTX 3060 ggml column came
from the RTX 3060 JSON report. The RTX 3060 run also repeated the ONNX
baselines; those values were close to the AMD-run baseline values.

An earlier same-day run generated AAC after every measured sample and made the
AMD 780M ggml/Vulkan column look slower (`0.195` overall). A no-audio control
run on the same checkout produced `0.178` overall, while TTS.cpp native synthesis
time stayed essentially unchanged. The apparent regression came from
interleaved ffmpeg AAC encoding disturbing the APU benchmark, not from the Vulkan
synthesis path. The benchmark harness now queues AAC files and encodes them
after a backend's measured synthesis loop.

## GGML ONNX Plugin EP Parity

This 2026-06-25 rerun compares the non-invasive ONNX Runtime Plugin EP route
against the native Aivis ggml backend. The Plugin EP uses the normal
`tts_backend=onnx` engine path, but registers `AivisGgmlExecutionProvider` and
passes `claim_synthesis_graph=1` plus `claim_jp_bert_graph=1`. Both the
synthesis ONNX/AIVMX graph and the JP-BERT ONNX graph are executed by TTS.cpp
through the external EP.

The benchmark harness now unloads Style-Bert-VITS2's process-global ONNX
JP-BERT session before each backend is constructed. This is required because
`style_bert_vits2.nlp.onnx_bert_models` caches one BERT session per language;
without clearing that cache, a previous ONNX CPU baseline can leave JP-BERT on
`CPUExecutionProvider` even when the later synthesis session uses the Plugin EP.
The harness records `bert_active_providers` and fails a strict EP run if
JP-BERT is not claimed by `AivisGgmlExecutionProvider`.

| device | text length | ONNX CPU RTF | native ggml Vulkan RTF | ggml ONNX Plugin EP Vulkan RTF |
| --- | --- | ---: | ---: | ---: |
| AMD Radeon 780M | short | `0.381` | `0.218` | `0.217` |
| AMD Radeon 780M | medium | `0.313` | `0.178` | `0.180` |
| AMD Radeon 780M | long | `0.231` | `0.131` | `0.131` |
| AMD Radeon 780M | overall mean | `0.309` | `0.176` | `0.176` |
| RTX 3060 | short | `0.354` | `0.142` | `0.145` |
| RTX 3060 | medium | `0.271` | `0.101` | `0.101` |
| RTX 3060 | long | `0.200` | `0.062` | `0.062` |
| RTX 3060 | overall mean | `0.275` | `0.102` | `0.102` |

Provider evidence from both Plugin EP runs:

```json
{
  "active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
  "bert_active_providers": ["AivisGgmlExecutionProvider", "CPUExecutionProvider"],
  "provider_options": {
    "backend": "vulkan",
    "claim_jp_bert_graph": "1",
    "claim_synthesis_graph": "1",
    "eager_load_model": "1",
    "precision": "accurate"
  }
}
```

Interpretation:

- With the JP-BERT session cache fixed, the Plugin EP route is effectively tied
  with native ggml/Vulkan: `0.176008` vs `0.176042` overall RTF on AMD 780M,
  and `0.102431` vs `0.101817` on RTX 3060.
- The earlier Plugin EP probe around `0.274` overall RTF was not a Vulkan
  regression. It measured synthesis through the EP while JP-BERT was still
  served by the previously cached ONNX CPU session.
- The remaining per-run sample-count jitter matches the ONNX frontend's normal
  stochastic synthesis behavior. Peak normalization stayed at `0.999969482`
  for all three backends.

## Accuracy Against ONNX CPU

Accuracy was probed against ONNX CPU on the AMD Radeon 780M host after the
Plugin EP RTF parity run. Two different checks are needed:

- JP-BERT graph parity compares the supported JP-BERT ONNX graph directly:
  ONNX CPU output `[tokens, 1024]` vs Plugin EP output `[tokens, 1024]`.
- Synthesis graph parity compares the supported Style-Bert-VITS2 synthesis graph
  directly with the same `phone/tone/language/BERT/style` input tensors. For
  deterministic model comparison, `sdp_ratio=0`, `noise_scale=0`, and
  `noise_scale_w=0` are used. The normal synthesis settings include
  `RandomNormalLike` inside the ONNX graph, so default-noise waveform samples are
  not valid for pointwise max/rms accuracy.

JP-BERT parity:

| text length | tokens | shape delta | max abs | RMS | relative RMS | SNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | `8` | `0` | `0.132941` | `0.004381` | `0.004555` | `46.83 dB` |
| medium | `13` | `0` | `0.011683` | `0.001546` | `0.001615` | `55.84 dB` |
| long | `43` | `0` | `0.014273` | `0.000924` | `0.000958` | `60.37 dB` |

Synthesis graph parity with deterministic zero-noise inputs after the
TTS.cpp F32 Conv1D fix. This run keeps JP-BERT on ONNX CPU and lets the Plugin
EP claim the synthesis graph, so the numbers isolate the Style-Bert-VITS2
synthesis runner:

| text length | phones | sample delta | max abs | RMS | relative RMS | SNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | `27` | `0` | `0.003418` | `0.000233` | `0.001506` | `56.45 dB` |
| medium | `43` | `0` | `0.001923` | `0.000135` | `0.000753` | `62.47 dB` |
| long | `195` | `0` | `0.030548` | `0.000691` | `0.004917` | `46.17 dB` |

Interpretation:

- JP-BERT is close enough for the current Plugin EP route: shapes match exactly,
  RMS is below `0.005`, and long-text relative RMS is below `0.1%`.
- Synthesis is still not bit-exact against ONNX CPU, but the raw output lengths
  now match for all benchmark texts and deterministic relative RMS stays below
  `0.5%` when JP-BERT is held on ONNX CPU.
- Root cause of the earlier short-text `+512` sample error was upstream of
  `ceil`: Style-Bert-VITS2 was using `ggml_conv_1d`, whose implementation lowers
  through F16 im2col. That F16 input conversion made the text encoder input drift
  by `max_abs=0.011486`, and the duration predictor drift by `max_abs=0.000536`.
  One token then landed at `w=2.00005`, so direct `ceil(w)` expanded it to `3`
  frames while ONNX CPU stayed at `2` frames.
- TTS.cpp now routes Style-Bert-VITS2 Conv1D through the F32
  `ggml_kokoro_conv_1d` path. The same fixture now has text encoder input
  `max_abs=8.01086e-05` and duration predictor `max_abs=2.38419e-06`.
  The small duration `ceil` epsilon remains as a boundary guard, but it is not
  the primary fix.
- When the Plugin EP also claims JP-BERT, raw deterministic sample counts still
  match, but waveform relative RMS is higher (`1.8%` to `3.6%` in the latest
  probe). Remaining waveform work should therefore focus on JP-BERT ggml parity
  separately from synthesis duration parity.
- Therefore the current status is performance parity with native ggml/Vulkan,
  JP-BERT numerical parity is acceptable, and the supported deterministic
  synthesis graph is accurate enough for the current Plugin EP acceptance gate.

## Vulkan Fused ConvTranspose1D Evaluation

Same text set and `warm_steady_state` profile, using the local fused
ConvTranspose1D Vulkan patch on top of TTS.cpp `8e26ac0` / ggml `b6ad57d8`.

| device | Vulkan ConvTranspose1D path | overall RTF | short RTF | medium RTF | long RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| AMD 780M | upstream baseline, no local fused patch | `0.1775` | `0.2181` | `0.1854` | `0.1290` |
| AMD 780M | `scalar` fused | `0.1677` | `0.2076` | `0.1711` | `0.1245` |
| AMD 780M | `phase_k64` fused | `0.1559` | `0.1993` | `0.1590` | `0.1093` |
| AMD 780M | `aot` fused | `0.1560` | `0.1990` | `0.1591` | `0.1101` |
| RTX 3060 | upstream baseline, no local fused patch | `0.1040` | `0.1471` | `0.1015` | `0.0635` |
| RTX 3060 | `scalar` fused | `0.0980` | `0.1415` | `0.0954` | `0.0571` |
| RTX 3060 | `aot` fused | `0.0988` | `0.1420` | `0.0979` | `0.0565` |

Best observed rows: AMD 780M `phase_k64` at `0.1559` overall RTF, RTX 3060
`scalar` at `0.0980` overall RTF.

Decoder fixture parity stayed within the existing deterministic tolerance:

| device/path | decoder output max_abs | decoder output rms |
| --- | ---: | ---: |
| AMD 780M `phase_k64` / `aot` | `0.00140219` | `8.28174e-05` |
| RTX 3060 `aot` | `0.00113783` | `7.45232e-05` |

## Linux Vulkan Audio Preview

Each preview is the `run00` AAC file for that backend and text. These
previews are for qualitative listening only; the RTF table above comes from the
no-audio runs. `--audio_output_dir` records each generated path in
`records[].audio_path`, and AAC encoding runs outside the synthesis timer.
The Ubuntu ONNX CPU and both Linux ggml/Vulkan previews below were refreshed
after the 2026-06-25 ONNX-vs-GGML audio parity fix described in the next
section.

| text length | ONNX CPU | ONNX CUDA | ggml Vulkan AMD 780M | ggml Vulkan RTX 3060 |
| --- | --- | --- | --- | --- |
| short | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_short.m4a) |
| medium | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_medium.m4a) |
| long | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-ryzen8845hs_onnx-cpu_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_onnx-cuda_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-amd780m_ggml-vulkan-native_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/ubuntu-rtx3060_ggml-vulkan-native_long.m4a) |

## ONNX vs GGML Audio Parity Fix

The 2026-06-25 audio validation found that the visible long-sentence file
size and loudness mismatch was not a Vulkan precision regression. The ONNX path
uses Style-Bert-VITS2's `convert_to_16_bit_wav()` behavior: float output is
peak-normalized over the whole utterance before conversion to PCM16. The GGML
native and sidecar paths were returning PCM16 produced from a direct clipped
float conversion, so the waveform was roughly half the ONNX level. That lower
level also caused the engine-level silence threshold trim to remove valid
low-energy boundary samples, which made long outputs shorter.

The fix is to normalize GGML sidecar/native PCM16 at the backend boundary using
the same peak-normalization rule and then skip the engine-level threshold trim
for served `ggml-*` backends. ONNX keeps the existing trim behavior.

Post-fix cold parity validation on the AMD 780M, native binding, `tts-cpp-jp-bert`,
`synthesize-front`, one run per text. This table intentionally omits RTF:
the run used `cold_smoke` (`--warmup_runs 0 --runs 1`) and exists only to
validate output shape, level, and preview audio. Use the `Linux Vulkan Results`
section above for comparable `warm_steady_state` RTF numbers.

| text length | backend | output samples | duration sec | peak abs |
| --- | --- | ---: | ---: | ---: |
| short | ONNX CPU | `45567` | `1.033265` | `0.999969482` |
| short | ggml Vulkan AMD 780M | `45568` | `1.033288` | `0.999969482` |
| medium | ONNX CPU | `76288` | `1.729887` | `0.999969482` |
| medium | ggml Vulkan AMD 780M | `76288` | `1.729887` | `0.999969482` |
| long | ONNX CPU | `336384` | `7.627755` | `0.999969482` |
| long | ggml Vulkan AMD 780M | `336384` | `7.627755` | `0.999969482` |

The short one-sample difference is one output sample at 44.1 kHz. The medium and
long sample counts match exactly. A follow-up long-text three-run validation with the
ONNX-BERT frontend fixed showed ONNX CPU at `336383`, `336384`, `336382` samples
and ggml/Vulkan native at `336384`, `336384`, `336384`; both paths stayed at
`0.999969482` peak.

The refreshed RTX 3060 ggml/Vulkan preview validation used
`GGML_VK_VISIBLE_DEVICES=1` and confirmed TTS.cpp selected
`NVIDIA GeForce RTX 3060`. The updated Linux NVIDIA Vulkan AAC previews reported
`45568`, `76288`, and `336384` samples for short, medium, and long text, with
the same `0.999969482` peak.

The repository preview files for Ubuntu ONNX CPU, AMD 780M ggml/Vulkan, and RTX
3060 ggml/Vulkan in
`docs/res/style-bert-vits2-benchmark-20260625/representative-audio/` were
regenerated from the post-fix parity validation runs. AAC byte sizes are not an
accuracy metric because encoder decisions and stochastic TTS content can differ
even when sample counts and level normalization match.

## Windows Intel Arc B580 Native Binding Result

This run was captured on 2026-06-25, Asia/Tokyo, on Windows 11 with an
Intel(R) Arc(TM) B580 Graphics device using driver `32.0.101.8826`. The run used
`--warmup_runs 1 --runs 3`, `tts-cpp-jp-bert`, `synthesize-front`, `accurate`
Vulkan precision, and native binding transport.

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

The Windows Intel Arc B580 native binding path is faster than ONNX CPU in this
run, with overall ggml/Vulkan RTF at `26.2%` of ONNX CPU.

## macOS Metal Native Result

This run used the TTS.cpp native C API from `libtts.dylib`, TTS.cpp `a120f9e`,
ggml `a78c352b`, and preconverted synthesis / JP-BERT GGUF files from
`kevinzhow/style-bert-vits2-gguf`.

| text length | ONNX CPU RTF | ggml Metal native RTF | Metal/ONNX CPU ratio |
| --- | ---: | ---: | ---: |
| short | `0.297` | `0.215` | `0.725` |
| medium | `0.274` | `0.154` | `0.561` |
| long | `0.245` | `0.126` | `0.514` |
| overall mean | `0.272` | `0.165` | `0.606` |

Native timing breakdown for `ggml-metal-jp-bert-native`:

| text length | frontend seconds | native synthesis seconds | native JP-BERT seconds |
| --- | ---: | ---: | ---: |
| short | `0.055` | `0.151` | `0.055` |
| medium | `0.052` | `0.211` | `0.052` |
| long | `0.076` | `0.874` | `0.074` |

The AAC previews below are representative `run00` outputs from the same
backend and text set. AAC encoding runs after the synthesis timer, so these
files are not included in the RTF values above.

| text length | ONNX CPU | ggml Metal native |
| --- | --- | --- |
| short | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_onnx-cpu_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_onnx-cpu_short.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_ggml-metal-native_short.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_ggml-metal-native_short.m4a) |
| medium | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_onnx-cpu_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_onnx-cpu_medium.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_ggml-metal-native_medium.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_ggml-metal-native_medium.m4a) |
| long | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_onnx-cpu_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_onnx-cpu_long.m4a) | <audio controls preload="none" src="res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_ggml-metal-native_long.m4a"></audio><br>[AAC](res/style-bert-vits2-benchmark-20260625/representative-audio/macos-m1pro_ggml-metal-native_long.m4a) |

A one-text native binding comparison on the same long input measured the active
Metal decoder variants:

| Metal decoder path | RTF | native synthesis seconds | decoder seconds | decoder nodes |
| --- | ---: | ---: | ---: | ---: |
| `STYLE_BERT_VITS2_METAL_CONV_TRANSPOSE_1D_KERNEL=phase_32x32_k128` | `0.141` | `1.454` | `0.993` | `428` |
| AOT simdgroup-half, `STYLE_BERT_VITS2_METAL_FUSED_CONV1D=0` | `0.137` | `1.406` | `0.939` | `428` |
| default AOT simdgroup-half + fused Conv1D epilogue | `0.133` | `1.364` | `0.896` | `248` |

The default fused decoder fixture stayed within tolerance:
`max_abs=0.000594173`, `rms=7.4558e-05`.

## Evaluation Commands

The Linux RTF table was generated without AAC output so encoding would not
perturb the AMD APU measurements. Set these paths for your local checkout before
running the commands:

```bash
export CUDA12_NVIDIA_LIBS="<colon-separated CUDA 12/cuDNN library dirs>"
export AIVM_PATH="<path-to-mao.aivm>"
export AIVMX_PATH="<path-to-mao.aivmx>"
export SYNTHESIS_GGUF_PATH="<path-to-HF-kevinzhow/style-bert-vits2-gguf/voices/mao-full-sdp.gguf>"
export JP_BERT_GGUF_PATH="<path-to-HF-kevinzhow/style-bert-vits2-gguf/frontend/style-bert-vits2-jp-bert.gguf>"
export TTS_CPP_DIR="<path-to-TTS.cpp>"
export TTS_CPP_VULKAN_BUILD_DIR="$TTS_CPP_DIR/build-vulkan-latest-main"
export TTS_CPP_NATIVE_BUILD_DIR="$TTS_CPP_DIR/build-native-binding-shared"
export AIVIS_GGML_ONNX_EP_LIBRARY_PATH="<path-to-libaivis_ggml_onnx_ep.so>"
export BENCHMARK_OUTPUT_DIR="<path-to-benchmark-output-dir>"
```

AMD 780M iGPU:

```bash
LD_LIBRARY_PATH="${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GGML_VK_VISIBLE_DEVICES=0 \
uv run python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path "$AIVM_PATH" \
  --aivmx_path "$AIVMX_PATH" \
  --gguf_path "$SYNTHESIS_GGUF_PATH" \
  --jp_bert_gguf_path "$JP_BERT_GGUF_PATH" \
  --tts_server_path "$TTS_CPP_VULKAN_BUILD_DIR/bin/tts-server" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_BUILD_DIR/src/libtts.so" \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --onnx_ep_backend vulkan \
  --onnx_ep_vulkan_precision accurate \
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
  --output_json "$BENCHMARK_OUTPUT_DIR/amd780m.json" \
  > "$BENCHMARK_OUTPUT_DIR/amd780m.log" 2>&1
```

RTX 3060:

```bash
LD_LIBRARY_PATH="${CUDA12_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GGML_VK_VISIBLE_DEVICES=1 \
uv run python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path "$AIVM_PATH" \
  --aivmx_path "$AIVMX_PATH" \
  --gguf_path "$SYNTHESIS_GGUF_PATH" \
  --jp_bert_gguf_path "$JP_BERT_GGUF_PATH" \
  --tts_server_path "$TTS_CPP_VULKAN_BUILD_DIR/bin/tts-server" \
  --ggml_native_library_path "$TTS_CPP_NATIVE_BUILD_DIR/src/libtts.so" \
  --onnx_ep_library_path "$AIVIS_GGML_ONNX_EP_LIBRARY_PATH" \
  --onnx_ep_backend vulkan \
  --onnx_ep_vulkan_precision accurate \
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
  --output_json "$BENCHMARK_OUTPUT_DIR/rtx3060.json" \
  > "$BENCHMARK_OUTPUT_DIR/rtx3060.log" 2>&1
```

For AAC preview regeneration, add these flags to a separate run:

```text
--audio_output_dir "$BENCHMARK_OUTPUT_DIR/audio-preview"
--output_json "$BENCHMARK_OUTPUT_DIR/audio-preview.json"
```

macOS Metal:

```bash
TTS_CPP_METAL_BUILD_DIR="$TTS_CPP_DIR/build-metal-shared" \
DYLD_LIBRARY_PATH="$TTS_CPP_METAL_BUILD_DIR/src:$TTS_CPP_METAL_BUILD_DIR/ggml/src:$TTS_CPP_METAL_BUILD_DIR/ggml/src/ggml-blas:$TTS_CPP_METAL_BUILD_DIR/ggml/src/ggml-metal:$TTS_CPP_METAL_BUILD_DIR/ggml/src/ggml-cpu" \
GGML_METAL_NO_RESIDENCY=1 \
STYLE_BERT_VITS2_DEBUG_TIMINGS=1 \
uv run --group dev python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path "$AIVM_PATH" \
  --aivmx_path "$AIVMX_PATH" \
  --gguf_path "$SYNTHESIS_GGUF_PATH" \
  --jp_bert_gguf_path "$JP_BERT_GGUF_PATH" \
  --tts_server_path "$TTS_CPP_DIR/build-metal/bin/tts-server" \
  --ggml_native_library_path "$TTS_CPP_METAL_BUILD_DIR/src/libtts.dylib" \
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
  --output_json "$BENCHMARK_OUTPUT_DIR/metal-native-aot-fused-conv1d.json" \
  > "$BENCHMARK_OUTPUT_DIR/metal-native-aot-fused-conv1d.log" 2>&1
```
