# AivisSpeech Engine ggml Vulkan Backend Staged Plan

本文档描述如何利用 TTS.cpp 的 ggml/Vulkan 能力，为 AivisSpeech Engine 增加更广泛设备可用的 GPU 推理后端。

当前 AivisSpeech Engine 的主合成路径是：

- `StyleBertVITS2TTSEngine` 负责文本正规化、g2p、参数映射和模型缓存。
- `.aivmx` 模型直接交给 `style_bert_vits2.TTSModel`。
- 推理执行依赖 ONNX Runtime provider，当前实际选择面是 CPU、CUDA、DirectML。

因此，ggml/Vulkan 的执行逻辑不应作为 AivisSpeech Engine 内部的 ONNX Runtime provider 补丁接入。当前已验证路线仍是并行的显式 opt-in native 推理后端：保留现有 ONNX 路径为默认稳定路径，把 Vulkan 做成显式 opt-in，并且每个阶段都有可回滚边界。

如果后续要尝试 ONNX Runtime Plugin EP 路线，也必须保持完全外置：AivisSpeech Engine 只做通用 Plugin EP shared library 注册和 provider 顺序选择，不承载 ggml 图匹配、TTS.cpp runner、GGUF cache、Vulkan device routing、JP-BERT ONNX 替换或 synthesis 执行逻辑。该路线的边界与阶段记录在 [GGML ONNX Runtime Plugin Provider](./ggml-onnx-plugin-provider.md)。

For ggml/TTS.cpp, AIVM/Safetensors is the preferred source format. Safetensors preserves the original named weight tensors more directly, which matches TTS.cpp's GGUF conversion model. AIVMX/ONNX remains the compatibility format for the existing ONNX backend and for parity baselines, but automatic GGUF conversion should not depend on ONNX graph/export names.

## Design Principles

- Preserve the existing ONNX path until the ggml path proves parity and speed.
- Keep model conversion, backend selection, process management, and graph optimization as separate stages.
- Prefer source-level parity tests over listening-only validation.
- Make fallback explicit. Vulkan initialization failure must not break normal CPU/CUDA/DML synthesis.
- Do not make `--use_gpu` silently mean Vulkan. Vulkan should require a dedicated option until it is production proven.

## Stage 0: Baseline And Interface Boundary

Goal: freeze the current behavior and define the backend seam.

Implementation:

- Add a small internal backend interface for Style-Bert-VITS2 synthesis, for example:
  - `load_model(aivm_model_uuid, aivm_info, metadata)`.
  - `unload_model(aivm_model_uuid)`.
  - `is_model_loaded(aivm_model_uuid)`.
  - `synthesize(params) -> (sample_rate, int16_wave or float_wave)`.
- Wrap the existing `TTSModel` path as `OnnxStyleBertVITS2Backend`.
- Keep `StyleBertVITS2TTSEngine` responsible for API-compatible parameter mapping:
  - `speaker_id`.
  - `style_id`.
  - `style_weight`.
  - `length_scale`.
  - `sdp_ratio`.
  - `pitch_scale`.
  - silence trimming and output post-processing.
- Add tests that prove the wrapper preserves the current argument mapping.

Acceptance:

- Existing unit and e2e tests pass without Vulkan installed.
- No user-visible behavior changes.
- `--use_gpu` continues to mean the current ONNX Runtime provider behavior.

Rollback:

- Remove the wrapper and call `TTSModel` directly again. No model/cache migration is involved.

## Stage 1: TTS.cpp Sidecar MVP

Goal: prove the product integration using TTS.cpp as a local sidecar process before committing to in-process native bindings.

Implementation:

- Build TTS.cpp with `-DGGML_VULKAN=ON`.
- Launch `tts-server` with explicit backend controls:
  - `--backend vulkan`.
  - `TTS_BACKEND_STRICT=1`.
  - optional `TTS_DEVICE=<index>`.
- Add `GgmlVulkanStyleBertVITS2Backend` that talks to the sidecar over localhost HTTP.
- For the first pass, use TTS.cpp `/v1/style-bert-vits2/synthesize-front` with explicit phones/tones/BERT features.
- TTS.cpp has a `style-bert-vits2-jp-bert` GGUF runner and `/v1/style-bert-vits2/jp-bert/features`, but this is a BERT feature endpoint, not a complete text-to-wave Style-Bert-VITS2 frontend. AivisSpeech still has to provide Japanese normalization, g2p, word-to-phone alignment, phone/tone/language IDs, and the phone-level BERT tensor unless TTS.cpp grows a fused text endpoint. The generic `/v1/audio/speech` route is not that fused endpoint today: the local TTS.cpp fix now returns a structured unsupported-route error instead of aborting, but `style_bert_vits2_runner::generate()` still does not synthesize from natural text.
- Keep `/v1/style-bert-vits2/decode` and `/v1/style-bert-vits2/synthesize-latent` as diagnostic routes, not product defaults.
- Add a new explicit startup flag, for example:
  - `--tts_backend onnx|ggml-vulkan`.
  - or `--use_ggml_vulkan`.
- If the sidecar is unavailable, log the exact failure and fall back to ONNX unless strict Vulkan mode is requested.

Acceptance:

- One installed model can synthesize through TTS.cpp Vulkan from the normal `/synthesis` API.
- The same request falls back to ONNX when Vulkan is unavailable.
- Logs show which backend served each synthesis request.
- Sidecar process shutdown is deterministic.

Rollback:

- Disable the flag and the system returns to pure ONNX behavior.
- No AIVMX files are modified.

## Stage 2: AIVM/Safetensors To GGUF Conversion Cache

Goal: make AIVM/Safetensors models usable by TTS.cpp without manual conversion, while keeping AIVMX/ONNX as the existing compatibility path.

Implementation:

- Add an `AivmGgufCache` component under the model management layer.
- Extend model discovery/installation so the engine can register AIVM/Safetensors models for the ggml backend.
- Treat AIVM/Safetensors as the primary registered model format when both AIVM and AIVMX exist for the same UUID.
- Keep same-UUID AIVMX files as the ONNX-compatible input that the existing backend resolves explicitly.
- Cache key should include:
  - AIVM/Safetensors file path.
  - file size.
  - mtime or content hash.
  - AIVM manifest UUID and version.
  - converter version.
  - TTS.cpp GGUF schema version.
- Cache location should be separate from installed user models, for example:
  - `get_save_dir() / "GgufModelCaches"`.
- The preferred converter should extract from AIVM/Safetensors:
  - original named tensors.
  - hyper parameters.
  - style vectors.
  - speaker/style metadata needed for local id mapping.
- TTS.cpp's existing Style-Bert-VITS2 converter should be adapted to consume AIVM/Safetensors plus embedded AIVM metadata directly.
- AIVMX/ONNX to GGUF conversion should be treated as a secondary compatibility route, not the primary implementation route. It is more fragile because ONNX graph/export names may not map cleanly back to the tensor layout expected by TTS.cpp.
- Conversion should happen lazily on first ggml load, with clear progress logs.

Acceptance:

- Installing or updating an AIVM/Safetensors model invalidates only the matching GGUF cache entry.
- Existing AIVMX install and synthesis behavior remains unchanged for the ONNX backend.
- A corrupted/incomplete GGUF cache is deleted and regenerated.
- Cache miss does not affect existing ONNX synthesis.
- The generated GGUF loads in `tts-server` with `TTS_BACKEND=cpu` and `TTS_BACKEND=vulkan`.

Rollback:

- Delete `GgufModelCaches`.
- Disable ggml backend flag.

## Stage 3: Feature Parity Gate

Goal: make the ggml path semantically compatible enough to be useful for real AivisSpeech requests.

Scope:

- Japanese Style-Bert-VITS2 and Style-Bert-VITS2 JP-Extra first.
- Existing AivisSpeech parameter mapping must remain the source of truth.
- Non-Japanese and unsupported model architectures should reject ggml path and use ONNX.

Required parity checks:

- Same `AudioQuery` produces matching:
  - sample rate.
  - non-silence duration within tolerance.
  - leading/trailing silence behavior after AivisSpeech post-processing.
  - style id and speaker id mapping.
  - `style_weight` mapping from `intonationScale`.
  - `length_scale` mapping from `speedScale`.
  - `sdp_ratio` mapping from `tempoDynamicsScale`.
- Compare deterministic paths first:
  - `sdp_ratio=0`.
  - fixed random seed for latent/noise paths.
- Only enable non-zero `sdp_ratio` after TTS.cpp stochastic duration predictor parity is verified.

Acceptance:

- A golden request suite passes ONNX vs ggml comparisons.
- Unsupported requests fall back or return a targeted error, depending on strict mode.
- Tests cover empty text, punctuation-heavy text, long text, and multiple styles in the same AIVMX.

Rollback:

- Keep the ggml backend behind the explicit flag.
- If one feature fails parity, route only that request shape back to ONNX.

## Stage 4: Production Runtime Integration

Goal: turn the sidecar prototype into a maintainable runtime feature.

Two options:

### Option A: Managed Sidecar

Keep `tts-server` as a child process.

Pros:

- Fastest to ship.
- TTS.cpp crashes do not necessarily kill the Python engine.
- Easier to update TTS.cpp independently.

Cons:

- HTTP JSON tensor transfer is expensive.
- Process lifecycle and port management add operational complexity.
- Large BERT/front-half tensors inflate request overhead.

Required work:

- Auto-select a free localhost port.
- Health check endpoint before serving synthesis.
- Process supervision and timeout.
- Request queueing consistent with current `_inference_lock`.
- Structured error propagation to engine logs.

### Option B: In-Process Native Binding

Expose TTS.cpp through a C ABI or pybind module.

Pros:

- Avoids HTTP and JSON tensor overhead.
- Easier to share model handles and avoid repeated serialization.
- Better long-term product shape.

Cons:

- Packaging becomes harder.
- Native crashes can kill the engine process.
- ABI stability must be maintained.

Required work:

- Stable C API for:
  - load GGUF model.
  - select backend/device.
  - synthesize front.
  - synthesize latent.
  - unload model.
  - query backend/device status.
- Python extension packaging for Windows, Linux, and macOS.
- Explicit library loading and error handling.

Recommendation:

- Ship Stage 1 with managed sidecar.
- Move to in-process native binding only after Stage 3 parity and real performance numbers justify the packaging cost.

Acceptance:

- Backend selection is visible through logs and diagnostics.
- Concurrent requests do not corrupt model state.
- Startup and shutdown are clean on Linux, Windows, and macOS.

Rollback:

- Runtime flag disables sidecar/native binding entirely.

- Stage 4 managed-sidecar runtime coverage now includes automatic localhost port selection, `/health` polling before serving synthesis, deterministic shutdown, structured diagnostics, Vulkan/CPU backend environment propagation, one-thread sidecar defaults matching the engine inference lock, and startup failure messages that include the recent TTS.cpp log tail. If Vulkan initialization fails or the process exits before health check, the managed sidecar error includes both the log path and the last sidecar lines, and the ggml backend preserves that detail in its HTTP error. Synthesis-time TTS.cpp HTTP errors also keep the sidecar's returned message; when the Engine is using the default `bert_b64` payload, the error points old sidecar binaries at `--ggml_bert_payload_format json-array`.

## Stage 5: Performance Optimization

Goal: optimize the real bottlenecks after correctness is locked.

Do not start here. The first performance milestone is not shader tuning; it is proving that the integrated AivisSpeech request path actually spends enough time inside ggml/Vulkan to benefit.

Measurement plan:

- Record per-request:
  - frontend preprocessing time.
  - BERT feature extraction time.
  - ggml graph compute time.
  - tensor transfer/serialization time.
  - audio post-processing time.
  - total RTF.
- Compare:
  - ONNX CPU.
  - ONNX CUDA if available.
  - ONNX DirectML on Windows.
  - ggml CPU.
  - ggml Vulkan accurate.
  - ggml Vulkan fast diagnostics only.
- Use `TTS_BACKEND_STRICT=1` in validation so CPU fallback is not mistaken for GPU speed.

Likely optimization targets:

- Avoid JSON transfer for large tensors by moving from sidecar HTTP to native binding.
- Move JP BERT to TTS.cpp in two steps:
  - current sidecar validation: Python keeps normalization/g2p/tokenization and TTS.cpp computes `/jp-bert/features`;
  - final performance target: a fused TTS.cpp endpoint avoids returning BERT features to Python and sending the same tensor back through `/synthesize-front`.
- While the managed sidecar is still used, send large float tensors through TTS.cpp's `*_b64` JSON fields instead of JSON float arrays.
- Keep small integer ID arrays as JSON arrays unless measurement proves otherwise; a local probe showed int32 base64 for `phone_ids`/`tone_ids`/`language_ids` slightly increased request size because the values are small.
- Reduce graph dispatch and layout churn inside TTS.cpp.
- Keep Vulkan accurate mode as default unless fast mode passes waveform parity for Style-Bert-VITS2.
- Add device selection and denylist/allowlist for known slow or inaccurate Vulkan drivers.

Acceptance:

- Vulkan improves real end-to-end RTF on target non-CUDA devices.
- Accuracy gates remain green.
- Performance tests identify the active Vulkan device and fail if the request silently falls back to CPU.

Rollback:

- Keep optimization toggles opt-in until they pass both parity and end-to-end timing gates.

## User-Facing Controls

Initial controls should be explicit and conservative:

- `--tts_backend onnx|ggml-vulkan`.
- `--ggml_vulkan_strict`.
- `--ggml_vulkan_device <index>`.
- `--ggml_model_cache_dir <path>`.
- `--ggml_vulkan_precision accurate|fast`, with `accurate` as default.

Do not repurpose `--use_gpu` at first. After ggml/Vulkan becomes stable, `--use_gpu` can become a high-level alias that selects the best backend by platform, but only if diagnostics clearly report the chosen backend.

## Minimal First Pull Request

The smallest useful PR should include only:

- Backend interface extraction.
- Existing ONNX backend wrapper.
- CLI/config flag that still defaults to ONNX.
- Unit tests proving no behavior change.

The second PR can add the sidecar backend behind the flag. The third PR can add AIVM/Safetensors to GGUF cache conversion. AIVMX/ONNX to GGUF compatibility can be a later PR only if there is a concrete need. Keeping these separate makes review and rollback practical.

## Current Landing Notes

The first implementation in this repository follows the staged plan through the managed sidecar foundation:

- `StyleBertVITS2TTSEngine` now calls a backend interface instead of directly owning the `TTSModel` cache.
- `OnnxStyleBertVITS2Backend` preserves the existing AIVMX/ONNX Runtime path.
- `GgmlVulkanStyleBertVITS2Backend` calls TTS.cpp `/v1/style-bert-vits2/synthesize-front`.
- `FallbackStyleBertVITS2Backend` keeps `ggml-vulkan` non-strict by default: sidecar load or inference failure falls back to ONNX.
- Non-zero `sdp_ratio` is protected by default in the ggml backend. Until stochastic duration parity has stable thresholds, non-strict mode preflights this unsupported request shape and routes it directly to ONNX without first calling the sidecar; strict mode still returns the ggml backend's targeted error. `--ggml_vulkan_allow_nonzero_sdp` is an explicit probe flag.
- `AivmInfosRepository` can scan `.aivm` and `.aivmx`; if both exist for the same UUID, `.aivm` is the registered primary model because Safetensors is the better ggml/GGUF source.
- The ONNX backend separately resolves a same-UUID `.aivmx` compatibility file when the registered model is `.aivm`.
- `AivmGgufCache` lazily converts AIVM/Safetensors to GGUF through an external TTS.cpp converter command and stores results under `GgufModelCaches`.
- AIVM `.aivm` files are Safetensors containers with AIVM metadata in the Safetensors metadata block; the cache passes the `.aivm` path directly as the converter's source model path and writes temporary config/style-vector files from the embedded metadata.
- AIVM/Safetensors updates delete matching default GGUF cache entries. Uninstall deletes all installed `.aivm`/`.aivmx` files with the same manifest UUID and removes matching default GGUF cache entries, so same-UUID compatibility files do not make a model reappear after the next scan.
- There is intentionally no `--ggml_model_format` switch. The managed automatic conversion path requires `.aivm`/Safetensors; `.aivmx` is used by the ONNX backend and by benchmark baselines only.
- `ManagedTtsCppSidecar` can start a local `tts-server`, auto-select a localhost port, poll `/health`, and shut the child process down from the FastAPI shutdown event.
- `ManagedTtsCppSidecar.status` exposes structured runtime diagnostics for backend, device, Vulkan precision, debug timing mode, strict mode, server URL, active model path, active default model, log path, process id, and running state.
- The ggml backend rejects non-Japanese model metadata before sidecar load. In non-strict mode this is handled by the existing ONNX fallback path; in strict mode the caller receives a targeted error.
- The ggml backend logs and exposes structured per-request frontend feature preparation, payload construction, BERT payload format/size, JSON serialization, sidecar HTTP inference, WAV decode, request JSON byte size, response WAV byte size, and optional TTS.cpp JP-BERT feature HTTP timings.
- `GgmlVulkanStyleBertVITS2Backend.diagnostics` exposes internal runtime state without expanding the public HTTP API: configured TTS.cpp backend, server URL, synthesis model name, optional JP-BERT model name, selected synthesis endpoint, BERT payload format, fused text endpoint support state/reason, managed model path, loaded model UUIDs, non-zero SDP gate state, latest sidecar timings, and managed sidecar status.
- `StyleBertVITS2TTSEngine` logs unified per-request synthesis performance telemetry for every backend: actual served backend, engine preparation time, backend inference time, post-processing time, total elapsed time, output duration, sample count, and end-to-end RTF. In non-strict `ggml-vulkan`, ONNX fallback requests are logged as `backend=onnx` rather than as Vulkan.
- The ggml sidecar request now uses TTS.cpp's `bert_b64` input by default (`--ggml_bert_payload_format base64`), avoiding the earlier huge JSON float array for BERT features. `--ggml_bert_payload_format json-array` remains available for older sidecar compatibility.
- TTS.cpp does include a `style-bert-vits2-jp-bert` GGUF feature runner and `/v1/style-bert-vits2/jp-bert/features`, and this engine can use it when `--ggml_jp_bert_model` is set. That is not yet a full text-to-audio Style-Bert-VITS2 frontend: TTS.cpp still requires explicit phones/tones/language IDs and a phone-level BERT feature tensor for `/synthesize-front`, while the generic `generate()` path now reports unsupported for Style-Bert-VITS2 instead of synthesizing text.
- Backend inference for both ONNX and ggml remains inside the existing `StyleBertVITS2TTSEngine` inference lock, so the managed sidecar does not introduce a new concurrent runner mutation path.
- `GgmlVulkanStyleBertVITS2Backend.close()` stops the managed TTS.cpp sidecar; FastAPI lifespan shutdown calls `close()` on registered TTS engines.

Current CLI flags:

- `--tts_backend onnx|ggml-vulkan`
- `--ggml_vulkan_server_url http://127.0.0.1:8080`
- `--ggml_vulkan_model <model-id>`
- `--ggml_jp_bert_model <jp-bert-model-id>`
- `--ggml_vulkan_strict`
- `--ggml_model_cache_dir <path>`
- `--ggml_converter_path <path-to-TTS.cpp/py-gguf/convert_style_bert_vits2_to_gguf>`
- `--ggml_converter_device cpu|cuda|...`
- `--ggml_tts_server_path <path-to-tts-server>`
- `--ggml_tts_server_backend vulkan|cpu`
- `--ggml_model_path <preconverted-gguf-file-or-directory>`
- `--ggml_vulkan_device <TTS_DEVICE>`
- `--ggml_vulkan_precision accurate|fast`
- `--ggml_vulkan_allow_nonzero_sdp`
- `--ggml_synthesis_endpoint synthesize-front|synthesize-symbols`
- `--ggml_bert_payload_format base64|json-array`
- `--ggml_native_library_path <path-to-libtts.so-or-platform-equivalent>`
- `--ggml_tts_server_debug_timings`
- `--ggml_tts_server_log_path <path-to-sidecar-log>`
- benchmark-only: `--expect_ggml_vulkan_mean_rtf_at_most <rtf>`
- benchmark-only: `--expect_ggml_vulkan_per_text_rtf_at_most <rtf>`
- benchmark-only: `--expect_ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most <ratio>`
- benchmark-only: `--expect_ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most <ratio>`

Managed sidecar example:

```bash
uv run python run.py \
  --tts_backend ggml-vulkan \
  --ggml_tts_server_path /path/to/TTS.cpp/build/bin/tts-server \
  --ggml_tts_server_backend vulkan \
  --ggml_converter_path /path/to/TTS.cpp/py-gguf/convert_style_bert_vits2_to_gguf \
  --ggml_vulkan_device 0 \
  --ggml_vulkan_precision accurate \
  --ggml_tts_server_debug_timings \
  --ggml_tts_server_log_path <path-to-sidecar-log>
```

User-managed sidecar remains supported:

```bash
TTS_BACKEND_STRICT=1 \
tts-server \
  --backend vulkan \
  --model-path /path/to/model-or-cache-dir \
  --host 127.0.0.1 \
  --port 8080

uv run python run.py \
  --tts_backend ggml-vulkan \
  --ggml_vulkan_server_url http://127.0.0.1:8080 \
  --ggml_converter_path /path/to/TTS.cpp/py-gguf/convert_style_bert_vits2_to_gguf
```

Optional TTS.cpp JP-BERT feature extraction requires the sidecar to load both GGUF files. With a managed sidecar, point `--ggml_model_path` to a directory containing the synthesis GGUF and the JP-BERT GGUF:

```bash
uv run python run.py \
  --tts_backend ggml-vulkan \
  --ggml_tts_server_path /path/to/TTS.cpp/build/bin/tts-server \
  --ggml_tts_server_backend vulkan \
  --ggml_model_path /path/to/gguf-model-directory \
  --ggml_vulkan_model mao-full-sdp \
  --ggml_jp_bert_model style-bert-vits2-jp-bert \
  --ggml_bert_payload_format base64
```

Native binding is the experimental in-process transport for the same ggml backend. It loads TTS.cpp's shared `libtts` C API instead of starting `tts-server`, so there is no localhost HTTP, JSON tensor transfer, base64 BERT payload, or WAV decode step. It still uses Engine-side normalization/g2p/phone alignment, and the current C API supports only `synthesize-front`; `synthesize-symbols` and a fully fused text-to-audio endpoint remain sidecar/future work.

```bash
MESA_VK_DEVICE_SELECT=1002:1900! \
uv run python run.py \
  --tts_backend ggml-vulkan \
  --ggml_tts_server_backend vulkan \
  --ggml_model_path /path/to/gguf-model-directory \
  --ggml_vulkan_model mao-full-sdp \
  --ggml_jp_bert_model style-bert-vits2-jp-bert \
  --ggml_native_library_path /path/to/TTS.cpp/build/src/libtts.so \
  --ggml_vulkan_precision accurate
```

The native binding path requires TTS.cpp to be built as shared libraries, for example:

```bash
cmake -S /path/to/TTS.cpp -B /path/to/TTS.cpp/build-native-binding-shared \
  -DBUILD_SHARED_LIBS=ON \
  -DTTS_BUILD_EXAMPLES=OFF \
  -DGGML_VULKAN=ON
cmake --build /path/to/TTS.cpp/build-native-binding-shared --target tts -j2
```

Known remaining work before calling the whole staged plan complete:

- Stage acceptance coverage snapshot:

| Stage | acceptance area | current evidence | status |
| --- | --- | --- | --- |
| Stage 0 | backend boundary, ONNX behavior preserved, `--use_gpu` unchanged | `StyleBertVITS2Backend`, `OnnxStyleBertVITS2Backend`, existing unit/e2e suite, `test/unit/tts_pipeline/test_style_bert_vits2_tts_engine.py` argument-mapping coverage | Done for the internal interface extraction. |
| Stage 1 | normal `/synthesis` can use TTS.cpp Vulkan, non-strict fallback to ONNX, backend logs, deterministic shutdown | `GgmlVulkanStyleBertVITS2Backend`, `FallbackStyleBertVITS2Backend`, `ManagedTtsCppSidecar`, unit tests for fallback/shutdown/startup failure details, opt-in Vulkan integration smoke | Done for managed sidecar MVP. |
| Stage 2 | AIVM/Safetensors primary source, lazy GGUF cache, same-UUID AIVMX compatibility, cache invalidation/regeneration, CPU/Vulkan load proof | `AivmInfosRepository`, `AivmGgufCache`, `AivmManager` cache deletion, cache unit tests, opt-in `.aivm` -> GGUF -> sidecar integration test | Done for the external TTS.cpp converter route. |
| Stage 3 | deterministic duration/silence parity, unsupported request routing, empty/punctuation/long/multi-style coverage | opt-in local golden-suite integration test, non-Japanese/unknown-architecture unit gates, non-zero SDP fallback gate | Partial. Duration/silence parity is gated; waveform-distance, broader fixtures, and stable non-zero SDP parity remain open. |
| Stage 4 | visible diagnostics, clean sidecar startup/shutdown, request serialization consistent with engine lock, structured errors | `ManagedTtsCppSidecar.status`, backend `diagnostics`, FastAPI shutdown close path, startup log-tail propagation tests, sidecar run under engine inference lock | Done for Option A managed sidecar. Cross-platform packaging/runtime validation still needs release testing. |
| Stage 5 | real end-to-end RTF improvement with device evidence, active bottleneck attribution, no silent CPU fallback | engine telemetry, benchmark harness, Vulkan log gates, payload/timing summary fields, native-binding transport, local AMD 780M benchmark evidence | Partial. Native binding removes HTTP/base64/WAV transport overhead; fully fused TTS.cpp text/frontend routing and broader device matrices remain open. |

- Completion boundary: this repository now has a functional opt-in ggml/Vulkan backend through a managed TTS.cpp sidecar or experimental native binding, automatic AIVM/Safetensors-to-GGUF cache conversion, deterministic parity gates, diagnostics, and benchmark tooling. The whole staged plan is not complete until Stage 5's open performance work is closed: a fused TTS.cpp text/frontend route that keeps tokenization, JP-BERT, and synthesis tensors inside TTS.cpp, plus broader device/performance validation and final parity conclusions for fast mode and non-zero SDP.
- A strict managed-sidecar smoke test has been proven locally with the `まお` AIVMX/AIVM pair and a preconverted `mao-full-sdp.gguf`: the engine produced a float32 waveform and the sidecar log showed `POST /v1/style-bert-vits2/synthesize-front 200` on `AMD Radeon 780M Graphics (RADV PHOENIX)`.
- That smoke test is now captured as an opt-in pytest integration test:

```bash
uv run --group dev pytest test/integration/test_style_bert_vits2_ggml_vulkan.py -q
# default result: skipped unless AIVIS_GGML_VULKAN_TEST=1 is set
```

When enabled, the integration test now runs both ONNX and ggml/Vulkan for a small deterministic AivisSpeech golden suite. It defaults to `tempoDynamicsScale=0`. Non-zero SDP probes must set both `AIVIS_GGML_TEST_TEMPO_DYNAMICS_SCALE=1.0` and `AIVIS_GGML_TEST_ALLOW_NONZERO_SDP=1`. Precision matrix probes can set `AIVIS_GGML_TEST_VULKAN_PRECISIONS=accurate,fast`; by default only `accurate` is a hard parity gate, while `fast` is diagnostic unless `AIVIS_GGML_TEST_REQUIRED_PRECISIONS=accurate,fast` is set. Frontend matrix probes can set `AIVIS_GGML_TEST_FRONTENDS=onnx-bert,tts-cpp-jp-bert`; the TTS.cpp JP-BERT mode requires `AIVIS_GGML_TEST_JP_BERT_GGUF_PATH=/path/to/style-bert-vits2-jp-bert.gguf` and asserts that `/v1/style-bert-vits2/jp-bert/features` was served. Synthesis endpoint probes can set `AIVIS_GGML_TEST_SYNTHESIS_ENDPOINTS=synthesize-front,synthesize-symbols`; the log gate checks the exact endpoint route that was selected. The test can write a parity report with `AIVIS_GGML_TEST_PARITY_REPORT_PATH=/tmp/report.json`. The test asserts:

- non-empty float32 output from both backends.
- matching total duration within 50 ms.
- matching non-silence duration within 50 ms.
- matching leading/trailing silence after AivisSpeech post-processing within 50 ms.
- matching empty-text output behavior.
- punctuation-heavy and long-text request coverage.
- multiple-style coverage when the installed AIVM manifest exposes more than one style.
- strict sidecar execution with Vulkan device evidence visible in the TTS.cpp log when `AIVIS_GGML_TEST_TTS_BACKEND=vulkan`; `AIVIS_GGML_TEST_EXPECT_LOG_CONTAINS` can additionally require an exact device string.

Local AMD 780M verification command:

```bash
MESA_VK_DEVICE_SELECT=1002:1900! \
AIVIS_GGML_VULKAN_TEST=1 \
AIVIS_GGML_TEST_AIVMX_PATH=<path-to-model.aivmx> \
AIVIS_GGML_TEST_AIVM_PATH=<path-to-model.aivm> \
AIVIS_GGML_TEST_GGUF_PATH=<path-to-mao-full-sdp.gguf> \
AIVIS_GGML_TEST_TTS_SERVER_PATH=<path-to-tts-server> \
AIVIS_GGML_TEST_STYLE_ID=888753760 \
AIVIS_GGML_TEST_VULKAN_PRECISION=accurate \
AIVIS_GGML_TEST_EXPECT_LOG_CONTAINS='AMD Radeon 780M Graphics' \
uv run --group dev pytest test/integration/test_style_bert_vits2_ggml_vulkan.py -q
```

Verified result on 2026-06-24: `1 passed`, with strict Vulkan sidecar logging the AMD 780M device and `/v1/style-bert-vits2/synthesize-front`.
- A precision matrix verification on 2026-06-24 with `AIVIS_GGML_TEST_VULKAN_PRECISIONS=accurate,fast` confirmed `accurate` passed the current 50 ms duration/silence gate across the local golden suite. `fast` remained diagnostic: it passed four of five requests but missed the gate on the 58-character long text by `54.4 ms`, so `accurate` remains the only hard-gated Vulkan precision mode.
- A frontend matrix verification on 2026-06-24 with `AIVIS_GGML_TEST_FRONTENDS=onnx-bert,tts-cpp-jp-bert` confirmed both frontend modes passed the current 50 ms duration/silence gate in `accurate` mode across the local golden suite. The TTS.cpp JP-BERT path served `/v1/style-bert-vits2/jp-bert/features`; its largest observed duration delta was `37.8 ms` on the 58-character long text.
- A synthesis endpoint smoke on 2026-06-24 with `AIVIS_GGML_TEST_SYNTHESIS_ENDPOINTS=synthesize-symbols`, one short text, and one style passed the current 50 ms duration/silence gate. The sidecar log showed `AMD Radeon 780M Graphics (RADV PHOENIX)` and `POST /v1/style-bert-vits2/synthesize-symbols 200`, proving the endpoint is covered by the same opt-in Vulkan integration gate.
- Stage 3 now has an opt-in deterministic duration/silence parity gate for the local `まお` AIVM/AIVMX pair. The default suite covers four texts on the first style, a silent empty-text parity check, and one text on a second style when available.
- Stage 3 also has a backend support gate for the current Japanese-only Style-Bert-VITS2 ggml path. Non-Japanese metadata and unknown model architectures are rejected by the ggml backend. In non-strict mode these load/request failures use ONNX fallback; in strict mode the caller receives the targeted ggml error.
- A waveform-distance gate is intentionally not enabled yet. A local probe showed duration parity within 50 ms, but normalized waveform correlation was still poor between ONNX and ggml for the current TTS.cpp output, so duration/silence parity is the current defensible gate.
- Non-zero SDP now follows the Stage 3 gate: it is disabled on the ggml path by default, non-strict fallback skips the known-unsupported primary request instead of generating a sidecar failure, and direct ggml probing remains available only through explicit local flags.
- Remaining Stage 3 work: waveform-distance thresholds after TTS.cpp output parity improves, more installed-model fixtures, stable non-zero SDP parity thresholds before removing the default fallback, and broader multi-style coverage.
- Stage 2 cache acceptance now has unit coverage for valid cache reuse, incomplete GGUF regeneration, and same-UUID stale entry deletion, including orphan manifest-only entries, without deleting other models' cache entries.
- GGUF cache manifests now store both the opaque `cache_key` and explicit `cache_key_inputs`: resolved AIVM/Safetensors path, file size, file mtime, manifest UUID/version, model architecture, converter version, and GGUF schema version.
- `AivmGgufCache` can run TTS.cpp `py-gguf` converters that keep their dependencies in an adjacent virtualenv. If `--ggml_converter_path` points into a directory with `.venv312` or `.venv`, the cache invokes that Python interpreter and bridges the Engine's current `style_bert_vits2` and `aivmlib` packages into the converter process. The bridge also provides a no-op `pyworld` import stub because conversion needs to import `style_bert_vits2.tts_model` but does not use voice pitch/intonation adjustment. If no adjacent converter virtualenv exists, it runs the converter directly and bridges the adjacent `gguf` package only, avoiding incompatible full-venv `PYTHONPATH` leakage.
- Stage 2 model lifecycle coverage now includes deleting all same-UUID `.aivm`/`.aivmx` install files and default GGUF cache entries on uninstall.
- Local source-format probe on 2026-06-24 confirmed the `まお` AIVM opens through Safetensors directly, with `1165` tensors and metadata keys including `aivm_hyper_parameters`, `aivm_manifest`, and `aivm_style_vectors`.
- Stage 2 now has an opt-in automatic conversion integration test that exercises `.aivm` -> `GgufModelCaches/*.gguf` -> managed TTS.cpp sidecar synthesis without passing a preconverted `--ggml_model_path`:

```bash
MESA_VK_DEVICE_SELECT=1002:1900! \
AIVIS_GGML_VULKAN_CACHE_TEST=1 \
AIVIS_GGML_TEST_AIVMX_PATH=/path/to/model.aivmx \
AIVIS_GGML_TEST_AIVM_PATH=/path/to/model.aivm \
AIVIS_GGML_TEST_CONVERTER_PATH=/path/to/TTS.cpp/py-gguf/convert_style_bert_vits2_to_gguf \
AIVIS_GGML_TEST_TTS_SERVER_PATH=/path/to/TTS.cpp/build/bin/tts-server \
AIVIS_GGML_TEST_EXPECT_LOG_CONTAINS='AMD Radeon 780M Graphics' \
uv run --group dev pytest \
  test/integration/test_style_bert_vits2_ggml_vulkan.py::test_managed_ggml_vulkan_sidecar_converts_aivm_cache_and_synthesizes -q
```

Local run status on 2026-06-24: `1 passed` on the AMD 780M host after installing the converter's `onnx` dependency. The generated cache file was `240M`, used a dot-free TTS.cpp model id stem (`1_2_0` instead of `1.2.0`), wrote manifest `cache_key_inputs`, and the sidecar log showed `AMD Radeon 780M Graphics (RADV PHOENIX)` plus `POST /v1/style-bert-vits2/synthesize-front 200`.
- Stage 5 now has basic engine-level per-request telemetry, including total RTF, plus an executable local benchmark harness. The integration test and benchmark harness both read managed sidecar logs and fail Vulkan/Metal runs when the log does not prove an accelerator device/backend was active. The benchmark stores Vulkan/Metal device evidence in `metadata.sidecar_diagnostics` and parses TTS.cpp `STYLE_BERT_VITS2_*_TIMING` graph timing lines when requested:

```bash
MESA_VK_DEVICE_SELECT=1002:1900! \
uv run python tools/benchmark_style_bert_vits2_ggml_vulkan.py \
  --aivm_path <path-to-model.aivm> \
  --aivmx_path <path-to-model.aivmx> \
  --gguf_path <path-to-mao-full-sdp.gguf> \
  --jp_bert_gguf_path <path-to-style-bert-vits2-jp-bert.gguf> \
  --tts_server_path <path-to-tts-server> \
  --ggml_backend vulkan \
  --ggml_frontend tts-cpp-jp-bert \
  --ggml_vulkan_precision accurate \
  --ggml_debug_timings \
  --ggml_synthesis_endpoint synthesize-front \
  --style_id 888753760 \
  --text 'テストです。' \
  --runs 1 \
  --expect_sidecar_log_contains 'AMD Radeon 780M Graphics' \
  --expect_ggml_vulkan_mean_rtf_at_most 0.2
```

The benchmark requires `.aivm`/Safetensors for the ggml path and `.aivmx` only for the ONNX baseline. It can now run ONNX Runtime CPU/CUDA baselines by repeating `--onnx_baseline cpu --onnx_baseline cuda`, and the JSON report records `metadata.onnx_provider_diagnostics` with available, configured, and loaded-session active providers so CUDA fallback is not misreported as a GPU result. It can run a managed TTS.cpp matrix by repeating `--ggml_backend`, for example `--ggml_backend cpu --ggml_backend vulkan --ggml_backend metal`, can compare frontend modes with `--ggml_frontend onnx-bert --ggml_frontend tts-cpp-jp-bert`, can compare sidecar synthesis endpoints with `--ggml_synthesis_endpoint synthesize-front --ggml_synthesis_endpoint synthesize-symbols`, can compare Vulkan precision modes with repeated `--ggml_vulkan_precision accurate --ggml_vulkan_precision fast`, can select `--ggml_bert_payload_format base64|json-array`, can require graph timing evidence with `--ggml_debug_timings`, and can switch from sidecar HTTP to in-process C API with `--ggml_native_library_path`. The default `base64` mode uses TTS.cpp's `json_float_array` support for a `bert_b64` sibling field, so it remains the same `/synthesize-front` or `/synthesize-symbols` protocol with a smaller BERT tensor representation. Native binding specs are labeled with a `-native` suffix and are restricted to `synthesize-front` until the C API grows a symbol or text endpoint.

The benchmark also has opt-in performance gates for Stage 5 validation: overall ggml accelerator mean RTF, per-text ggml accelerator mean RTF, overall RTF ratio versus ONNX CPU, and per-text RTF ratio versus ONNX CPU. These gates are intentionally benchmark-only and disabled by default; when enabled, the JSON report includes `performance_gates.checks` and the command exits non-zero if any `ggml-vulkan*` or `ggml-metal*` backend violates the configured threshold. Use `--ggml_frontend tts-cpp-jp-bert` when applying a `0.2`-RTF performance gate; include `--ggml_frontend onnx-bert` only for comparison runs, because ONNX-BERT frontend smokes include extra Python frontend work and are not the current best ggml path. `synthesize-symbols` is an intermediate TTS.cpp endpoint that moves phone/tone/language ID mapping into TTS.cpp, but it still requires Engine-side BERT features and still sends the large BERT tensor over the sidecar request; it is not the final fused text-to-audio path.

Parsed sidecar timing evidence is stored under `metadata.sidecar_diagnostics.<backend>.style_bert_vits2_timings`, including per-marker summaries and graph-event totals for `compute_submit_ms`, `read_ms`, and `total_ms`. Native binding runs do not have sidecar logs, so their diagnostics record `transport=native-binding`, the library path, and per-request native timing fields instead of Vulkan/Metal log evidence. Each measured `records[]` entry now includes `backend_timings` for ggml runs: frontend mode, synthesis endpoint, transport, frontend seconds, payload build seconds, JSON encode seconds, sidecar/native synthesis seconds, WAV decode seconds, request JSON bytes, response WAV bytes, BERT token/float counts, the float32 BERT binary lower-bound byte size, selected BERT payload format, BERT payload byte size, total numeric payload byte size, the full JSON request to BERT-binary ratio, phone/symbol payload counts, and JP-BERT feature request timing/bytes when enabled. These payload-size fields are the Stage 5 evidence for whether a run is dominated by sidecar tensor transfer instead of the TTS.cpp graph itself. The JSON report keeps the original aggregate `summary`, and also writes `per_text_summary`, `per_text_backend_timing_summary`, `rtf_ratio_vs_onnx_cpu`, and `performance_gates` so short-sentence overhead, payload overhead, and local performance acceptance failures are not hidden by medium/long averages. It also writes `metadata.benchmark_profile`, which labels one-run no-warmup reports as `cold_smoke` and warm repeated reports as `warm_steady_state`. Remaining Stage 5 work: backend comparison matrices across devices, fused TTS.cpp text/frontend routing, finer TTS.cpp graph timing coverage for text encoder/duration predictor stages, fast-mode parity conclusions from a broader golden suite, and reducing the remaining native tensor copies.

For benchmark reproduction, use the preconverted GGUF bundle
`kevinzhow/style-bert-vits2-gguf`; it contains
`voices/mao-full-sdp.gguf` and `frontend/style-bert-vits2-jp-bert.gguf` for the
local `まお` benchmark. See
[style-bert-vits2-backend-benchmark.md](style-bert-vits2-backend-benchmark.md)
for the 2026-06-24 short/medium/long RTF comparison across ONNX CPU, ONNX CUDA,
ggml Vulkan AMD 780M iGPU, and ggml Vulkan RTX 3060, plus the 2026-06-25 macOS
Metal native binding result on Apple M1 Pro.

The same benchmark document also records the 2026-06-25 local TTS.cpp Vulkan
fused ConvTranspose1D exploration based on TTS.cpp `8e26ac0` / ggml `b6ad57d8`.
That unmerged local patch reduced AMD 780M native Vulkan overall RTF from
`0.1775` to `0.1559` with the f32 `phase_k64` selector. On RTX 3060, the fused
op improved overall RTF from `0.1040` to `0.0980`, but the AOT selector was not
the best default. Both measured Vulkan devices reported `matrix cores: none`, so
the Metal simdgroup-half strategy is documented as inspiration, not as the Linux
Vulkan default route.

Performance interpretation guardrails:

- The TTS.cpp unsupported-route fix for generic `/v1/audio/speech` is a stability and integration correctness change, not a synthesis performance optimization. It prevents Style-Bert-VITS2 requests from aborting the server when the wrong generic route is used.
- The current best integrated performance path is `--ggml_frontend tts-cpp-jp-bert` plus `--ggml_synthesis_endpoint synthesize-front`; use `--ggml_native_library_path` when a shared TTS.cpp library is available, otherwise use sidecar HTTP with `--ggml_bert_payload_format base64`.
- Results labeled `frontend_mode onnx-bert` still use Python-side ONNX BERT feature extraction. They are useful as a comparison baseline, but they should not be used to judge the ggml JP-BERT path.
- One-run `cold_smoke` results, especially the 6-character short sentence, are for route/device validation only. Treat warmed short/medium/long runs as the evidence for performance trend.

A base64 payload-summary verification on 2026-06-24 wrote a local JSON report from a three-text `warm_single_run` probe with `MESA_VK_DEVICE_SELECT=1002:1900!`, `--ggml_backend vulkan`, `--ggml_frontend onnx-bert`, `--ggml_synthesis_endpoint synthesize-front`, and default `--ggml_bert_payload_format base64`. The report included `per_text_backend_timing_summary`, `bert_payload_format=base64`, `numeric_payload_bytes`, and captured `AMD Radeon 780M Graphics (RADV PHOENIX)` in `metadata.sidecar_diagnostics.ggml-vulkan.vulkan_device_evidence`. Because it used one measured run after warmup, treat the RTF values as a quick warmed probe rather than steady-state statistics:

| text length | ONNX CPU RTF | ggml Vulkan RTF | request JSON bytes | BERT binary bytes | BERT payload bytes | numeric payload bytes | JSON / BERT binary | sidecar HTTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short, 6 chars | `0.366` | `0.381` | `147818` | `110592` | `147456` | `147633` | `1.34x` | `0.212s` |
| medium, 17 chars | `0.203` | `0.220` | `344657` | `258048` | `344064` | `344472` | `1.34x` | `0.397s` |
| long, 58 chars | `0.182` | `0.149` | `1350725` | `1011712` | `1348952` | `1350540` | `1.34x` | `1.202s` |

An attempted all-numeric base64 probe also sent `phone_ids_b64`, `tone_ids_b64`, and `language_ids_b64`; it was accepted by TTS.cpp but increased request JSON bytes to `148091`, `345275`, and `1353115` for the same short/medium/long texts. The current default therefore keeps small ID arrays as JSON and only base64-encodes the large BERT float tensor.

A one-run device-evidence and graph-timing smoke on 2026-06-24 confirmed the benchmark's automatic Vulkan log gates with `--ggml_debug_timings`: `metadata.sidecar_diagnostics.ggml-vulkan.vulkan_device_evidence` captured `ggml_vulkan: Found 1 Vulkan devices:` and `AMD Radeon 780M Graphics (RADV PHOENIX)`, while `style_bert_vits2_timings.event_count` was `6`. That smoke still used `frontend_mode onnx-bert` and produced `ggml-vulkan` RTF `0.562` for `テストです。`; parsed TTS.cpp graph-event totals were `compute_submit_ms_sum=112.916`, `read_ms_sum=73.903`, and `total_ms_sum=195.908`. The remaining gap to the pure TTS.cpp ref is expected for the integrated sidecar path because Python frontend work and JSON/HTTP transfer are included.

Latest TTS.cpp JP-BERT/base64 warmed probe on 2026-06-24 wrote a local JSON report with `MESA_VK_DEVICE_SELECT=1002:1900!`, `--ggml_backend vulkan`, `--ggml_frontend tts-cpp-jp-bert`, `--jp_bert_gguf_path <path-to-style-bert-vits2-jp-bert.gguf>`, `--ggml_synthesis_endpoint synthesize-front`, `--ggml_bert_payload_format base64`, `--warmup_runs 1`, `--runs 1`, and `--expect_sidecar_log_contains 'AMD Radeon 780M Graphics'`. Treat it as a quick warmed probe rather than a multi-sample benchmark, but use it as the current routing sanity check because it exercises TTS.cpp JP-BERT instead of ONNX BERT:

| text length | ONNX CPU RTF | ggml Vulkan + TTS.cpp JP-BERT RTF | frontend | JP-BERT HTTP | sidecar synthesis HTTP | request JSON bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short, 6 chars | `0.403` | `0.232` | `0.083s` | `0.082s` | `0.145s` | `147818` |
| medium, 11 chars | `0.314` | `0.198` | `0.098s` | `0.097s` | `0.238s` | `235304` |
| long, 41 chars | `0.231` | `0.135` | `0.126s` | `0.123s` | `0.887s` | `1066403` |
| overall mean | `0.316` | `0.189` | n/a | n/a | n/a | n/a |

This confirms that the current TTS.cpp JP-BERT path is the one that brings integrated AivisSpeech ggml/Vulkan back to the expected `~0.2` RTF range. The short sentence is still slightly above `0.2` in this run because fixed sidecar/frontend overhead dominates one-second audio, while medium and long text are below `0.2`.

Native-binding JP-BERT warmed probe on 2026-06-24 wrote a local JSON report with `MESA_VK_DEVICE_SELECT=1002:1900!`, `--ggml_backend vulkan`, `--ggml_frontend tts-cpp-jp-bert`, `--jp_bert_gguf_path <path-to-style-bert-vits2-jp-bert.gguf>`, `--ggml_synthesis_endpoint synthesize-front`, `--ggml_native_library_path <path-to-libtts.so>`, `--warmup_runs 1`, and `--runs 1`. The TTS.cpp shared library printed Vulkan device discovery for `AMD Radeon 780M Graphics (RADV PHOENIX)` during the run. Treat it as a quick warmed probe, but it is the current best integrated route because it removes HTTP, base64 tensor transfer, and WAV decode from the sidecar path:

| text length | ONNX CPU RTF | ggml Vulkan + native JP-BERT RTF | frontend | native JP-BERT | native synthesis | request JSON bytes | BERT payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| short, 6 chars | `0.369` | `0.219` | `0.078s` | `0.078s` | `0.136s` | `0` | `native-f32` |
| medium, 11 chars | `0.306` | `0.176` | `0.087s` | `0.086s` | `0.213s` | `0` | `native-f32` |
| long, 41 chars | `0.196` | `0.129` | `0.114s` | `0.112s` | `0.856s` | `0` | `native-f32` |
| overall mean | `0.290` | `0.175` | n/a | n/a | n/a | n/a | n/a |

The native binding improvement is real but modest versus sidecar JP-BERT/base64 (`0.189` -> `0.175` overall in these quick warmed probes). It removes process/HTTP/serialization/WAV overhead, but the current C API still copies input arrays into C++ vectors, returns JP-BERT features to Python, and then passes those features back into synthesis. A fully fused TTS.cpp Style-Bert-VITS2 text/frontend endpoint would remove that remaining tensor crossing.

Earlier JSON-array frontend-matrix benchmark result on 2026-06-24 on the local AMD 780M host, with `--expect_sidecar_log_contains 'AMD Radeon 780M Graphics'` passing:

| text length | ONNX CPU mean RTF | ggml Vulkan mean RTF | ggml Vulkan + TTS.cpp JP-BERT mean RTF |
| --- | ---: | ---: | ---: |
| short, 6 chars | `0.385` | `0.421` | `0.270` |
| medium, 33 chars | `0.214` | `0.176` | `0.161` |
| long, 120 chars | `0.229` | `0.162` | `0.163` |

Observed sidecar timings for that earlier JSON-array run:

| frontend mode | short JSON bytes | medium JSON bytes | long JSON bytes | short sidecar HTTP | medium sidecar HTTP | long sidecar HTTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX BERT in Python | `417146` | `2008932` | `6571475` | `0.262s` | `0.779s` | `2.444s` |
| TTS.cpp JP-BERT features | `552299` | `2772447` | `9110069` | `0.170s` | `0.788s` | `2.541s` |

Together these runs confirm:

- The earlier `0.5+` short result was the integrated AivisSpeech sidecar path, not a pure TTS.cpp graph benchmark. Python frontend, HTTP, JSON serialization, and WAV decode are included.
- Later `0.57`/`0.53`-class short results came from one-run smokes for the 6-character short sentence, including `warmup_runs=0` or `frontend_mode onnx-bert` route checks; they should be read as cold/smoke evidence, not as the current TTS.cpp JP-BERT performance path.
- TTS.cpp JP-BERT feature extraction is the current preferred performance path and improves the integrated short/medium/long ggml/Vulkan run versus the ONNX CPU baseline in the latest warmed probe, but it still returns BERT features to Python and then sends those features back to `/synthesize-front`.
- TTS.cpp `/synthesize-symbols` is now selectable for Engine/benchmark probes, but it only replaces Engine-side phone/tone/language ID mapping; it does not remove the BERT tensor transfer.
- The default `bert_b64` path reduces the request JSON / BERT binary ratio from the earlier `3.55x-3.77x` JSON-array range to about `1.34x` on the same short/medium/long warmed probe.
- The implemented native binding removes HTTP/base64/WAV transport overhead. The next real performance step is a fused TTS.cpp Style-Bert-VITS2 endpoint that keeps tokenization, JP-BERT features, and synthesis tensors inside one TTS.cpp call.

Verified short/medium/long benchmark result with non-zero SDP (`--tempo_dynamics_scale 1.0 --ggml_vulkan_allow_nonzero_sdp`) on the same host:

| text length | ONNX CPU mean RTF | ggml Vulkan mean RTF |
| --- | ---: | ---: |
| short, 6 chars | `0.405` | `0.458` |
| medium, 33 chars | `0.234` | `0.180` |
| long, 120 chars | `0.240` | `0.162` |

This keeps the same shape: short text is dominated by fixed integrated-path overhead, while medium and long text are under `0.2` RTF on the AMD 780M Vulkan path. Non-zero SDP remains an opt-in parity probe because stochastic duration differences can exceed the deterministic 50 ms duration gate on individual long-text runs even when mean duration is close.
