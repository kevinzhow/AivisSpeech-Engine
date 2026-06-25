# onnxruntime-ep-aivis-ggml

Standalone ONNX Runtime Plugin Execution Provider package for the Aivis
Style-Bert-VITS2 ggml runtime.

The package is intentionally external to AivisSpeech Engine. Aivis only
registers the shared library and prepends `AivisGgmlExecutionProvider` to the
ONNX provider list when explicitly configured.

## Python Helper Contract

The Python package exposes:

- `get_library_path() -> str`
- `get_ep_name() -> str`
- `get_ep_names() -> list[str]`
- `get_default_provider_options() -> dict[str, str]`

These match ONNX Runtime Plugin EP packaging guidance and let an application
discover the provider library path without knowing the wheel layout.

## Native Provider Contract

The native shared library must export the ONNX Runtime Plugin EP symbols:

- `CreateEpFactories`
- `ReleaseEpFactory`

Provider name:

```text
AivisGgmlExecutionProvider
```

Initial provider options:

- `backend`: `vulkan`, `metal`, or `cpu`
- `device`: backend-local device id
- `precision`: `accurate` or `fast`
- `cache_dir`: compiled ggml/GGUF artifact cache
- `cache_manifest_path`: prepared `manifest.json` from this package
- `gguf_path`: prepared TTS.cpp Style-Bert-VITS2 GGUF path
- `jp_bert_gguf_path`: prepared TTS.cpp Style-Bert-VITS2 JP-BERT GGUF path
- `tts_cpp_library_path`: TTS.cpp shared library exposing
  `tts_style_bert_vits2_*`
- `eager_load_model`: `0` or `1`; when `1`, `CreateEp()` dlopens TTS.cpp and
  loads any configured `gguf_path` and `jp_bert_gguf_path`
- `claim_synthesis_graph`: `0` or `1`; when `1`, claim the supported
  Style-Bert-VITS2 synthesis ONNX/AIVMX graph and run it through TTS.cpp
- `claim_jp_bert_graph`: `0` or `1`; when `1`, claim the supported
  Style-Bert-VITS2 JP-BERT ONNX graph and run it through TTS.cpp
- `n_threads`: TTS.cpp load/runtime thread count, where `0` keeps the runtime
  default

The native EP exposes default `backend=vulkan`, `precision=accurate`,
`eager_load_model=0`, `claim_synthesis_graph=0`, `claim_jp_bert_graph=0`, and
`n_threads=0` options from `GetSupportedDevices()`. During session creation,
ONNX Runtime passes selected options to `CreateEp()` with the
`ep.aivisggmlexecutionprovider.` prefix; the native EP validates
backend/precision, optional cache manifest readiness, and optional TTS.cpp
binding inputs before graph inspection or compile work.

The generic ONNX Runtime session config `ep.context_enable=1` is recognized by
the native EP. During `Compile()`, supported synthesis and JP-BERT graphs return
official `com.microsoft::EPContext` nodes with an Aivis GGML JSON context
payload. `ep.context_embed_mode=1` embeds that payload into the
`ep_cache_context` attribute. `ep.context_embed_mode=0` writes a JSON context
file next to `ep.context_file_path` and records only the relative file name in
`ep_cache_context`.

Loading a precompiled EPContext model is supported with payload-driven lazy
runtime restore. The application still passes the deployment-specific
`tts_cpp_library_path` and the relevant `claim_*_graph=1` flag, but `gguf_path`,
`jp_bert_gguf_path`, and `cache_manifest_path` can be recovered from the
relative paths stored in the EPContext payload. The portable payload does not
store `tts_cpp_library_path` because shared library locations are deployment
specific.

## Current Status

This directory currently provides:

- Python discovery helpers.
- A native Plugin EP with graph signature gates and compute bridges for the
  supported Style-Bert-VITS2 synthesis graph and JP-BERT graph.
- A smoke registration script.
- A Style-Bert-VITS2 synthesis graph signature inspector.
- A deterministic cache manifest preparer for the future ONNX-to-GGML compile
  path.
- An ONNX initializer tensor-pack extractor that writes deterministic raw
  tensor bytes plus per-tensor metadata.
- A conservative ONNX-initializer to TTS.cpp Style-Bert-VITS2 GGUF tensor
  mapping report. The current known Aivis AIVMX synthesis exports map cleanly
  when graph-derived anonymous MatMul weights are resolved by consumer node
  path.
- A converter readiness report that blocks GGUF writing when required model
  metadata, style vectors, tensor packs, or initializer mappings are missing.
- A strict GGUF writer entry point for ready converter plans. It writes
  `model.gguf` only after tensor mapping, external sources, and initializer
  counts are complete.
- A cache manifest validator for deployment gates. It checks manifest version,
  signature/runtime contracts, EPContext-lite metadata, readiness status when
  requested, and portable relative artifact paths.
- An official ORT EPContext payload validator. It checks the generated JSON
  payload contract, graph kind, provider/runtime versions, backend options,
  portable GGUF/cache artifact paths, and rejects deployment-specific
  `tts_cpp_library_path` leakage.
- Opt-in real-artifact integration fixtures for ORT `ModelCompiler` EPContext
  round trips and strict synthesis ONNX-to-GGUF writing.
- A deployment compatibility matrix in every manifest. It records the provider
  version, tested ONNX Runtime Plugin EP API version, TTS.cpp C API contract,
  GGUF schema expectation, signature contract versions, and official
  EPContext support level, payload version, and compiled-model compatibility
  contract.
- Native ORT compiled-model compatibility callbacks. `GetCompiledModelCompatibilityInfo()`
  returns a portable JSON contract for package metadata, and
  `ValidateCompiledModelCompatibilityInfo()` lets ORT score model-package
  variants as optimal, prefer-recompile, unsupported, or not-applicable.
- A native TTS.cpp binding readiness gate. With `eager_load_model=1`,
  `CreateEp()` dlopens `tts_cpp_library_path`, resolves the Style-Bert-VITS2
  and JP-BERT C API symbols, and loads configured GGUF paths. This keeps
  TTS.cpp-specific logic out of AivisSpeech Engine.
- A process-local TTS.cpp runtime registry. Sessions with the same
  library/backend/device/thread/model tuple reuse one loaded TTS.cpp runtime
  instead of reloading the same synthesis and JP-BERT GGUF artifacts for every
  ONNX session.

The native EP reports `AivisGgmlExecutionProvider` to ONNX Runtime through the
Plugin EP ABI. `GetCapability()` inspects ORT graphs for the known Aivis
Style-Bert-VITS2 synthesis shape and the known
`deberta-v2-large-japanese-char-wwm-onnx` JP-BERT shape. Graph claim is opt-in:
`claim_synthesis_graph=1` and `claim_jp_bert_graph=1` are required before the
EP claims either graph. Unclaimed graphs continue on CPU/CUDA/DML fallback
providers.

Provider options are parsed and validated by the native EP. The compute bridge
routes fused graph inputs and outputs by ONNX name because ORT may reorder
fused subgraph inputs/outputs. The synthesis bridge calls TTS.cpp
`tts_style_bert_vits2_synthesize_front_with_style_vec` and writes the ONNX
`output` tensor as `[1, 1, samples]`. The JP-BERT bridge calls
`tts_style_bert_vits2_jp_bert_encode_features` and writes the ONNX `output`
tensor as `[tokens, 1024]`.

The native bridge resolves all required TTS.cpp Style-Bert-VITS2 C API symbols
before graph claim is possible. It also checks optional version symbols when a
newer TTS.cpp build exports them:

- `tts_style_bert_vits2_runtime_abi_version`
- `tts_style_bert_vits2_gguf_schema_version`

Missing version symbols are treated as the legacy contract for compatibility
with current TTS.cpp builds; mismatched exported versions fail during
`CreateEp()` before any graph is claimed.

The cache preparer writes a portable `manifest.json` under a stable cache key
directory. It records source file size/hash, graph signature, a versioned
signature contract, a TTS.cpp runtime contract, a compatibility matrix, an
EPContext-lite artifact layout, provider options, planned artifact names, and
optionally an `initializers.bin` tensor pack. Without `--write-gguf`, the cache
entry remains a planned compile artifact.

When a tensor pack is written, the manifest also includes
`tts_cpp_tensor_mapping`. Direct mappings use tensor names accepted by the local
TTS.cpp Style-Bert-VITS2 GGUF loader, for example
`enc_p.emb.weight` to
`style_bert_vits2.text_encoder.token_embedding.weight`. Text-encoder and flow
module internals use the compact keys produced by the current TTS.cpp encoder,
such as `style_bert_vits2.te.enc.al.0.q.w` and
`style_bert_vits2.fl.0.pre.w`. Complete weight-normalized source pairs
(`weight_g` and `weight_v`) are materialized with PyTorch's default
weight-norm `dim=0` contract and written as the final TTS.cpp `weight` tensor.
Missing pairs remain `requires_transform` blockers.

The cache preparer can also record external converter inputs:

- `config.json` as `style_bert_vits2_config`
- `style_vectors.npy` as `style_vectors`

Only portable source metadata is written: filename, byte size, and SHA256. Local
absolute paths are not stored in the manifest.

`--write-gguf` upgrades the cache entry from planned to ready only when the
converter readiness report has no blockers. It requires `--write-tensor-pack`,
`--config-path`, `--style-vectors-path`, and the optional `convert` Python
extra so the `gguf` writer module is available. Written GGUF artifacts must use
a real `--converter-version`; `unimplemented` is rejected for `--write-gguf`.
If any mapping, weight-norm pair, or metadata is incomplete, the command fails
before creating a partial GGUF.

The Python signature inspector defines the raw synthesis and JP-BERT graph
contracts for the native EP. The current supported Aivis Style-Bert-VITS2 ONNX
export has 11 inputs, 7 outputs, 5334 nodes, 948 initializers, opset 18, and
stable op-sequence / initializer-name hashes. The supported JP-BERT ONNX export
has 2 inputs, 1 output, 3619 raw nodes, 432 raw initializers, and opset 17; the
native gate also accepts observed ORT-optimized variants of that graph. The
inspector prints both match results plus a structural contract hash for cache
manifests.

## Production Hardening Stages

1. Runtime registry and ABI gate: implemented. The native EP now reuses a
   process-local TTS.cpp runtime for identical library/backend/device/thread
   and GGUF inputs. Required C API symbols are resolved before graph claim, and
   optional TTS.cpp ABI/schema version symbols are enforced when present.
2. Signature contract: implemented for cache tooling. The Python inspector and
   manifest now record `aivis-ggml-signature-contract-v1`, structural graph
   hashes, synthesis match results, and JP-BERT match results. Native
   `GetCapability()` still uses ORT graph structural checks because ORT sees
   optimized fused graphs rather than the raw model file.
3. EPContext-lite: implemented as manifest metadata. The manifest records
   `aivis-ggml-ep-context-lite-v1`, relative artifact names, provider options,
   and cache key without storing absolute local paths. This is not yet the
   official ONNX Runtime `EPContext` node path.
4. Official ORT EPContext: generation implemented, lazy artifact restore
   inference implemented. The native EP honors `ep.context_enable` by returning
   real EPContext nodes from `Compile()` and writing/embedding a portable JSON
   payload. When a precompiled EPContext model is loaded, the provider claims
   `source=AivisGgmlExecutionProvider` nodes, restores relative GGUF/cache
   artifact paths from the payload, lazy-loads TTS.cpp through the provided
   `tts_cpp_library_path`, and routes compute through the same synthesis/JP-BERT
   bridge. The generated payload contract is also machine-checkable through
   `validate_ep_context_payload.py`. `Compile()` returns both the ORT-required
   `OrtNodeComputeInfo` and a fused-node-name-matched `EPContext` node, which is
   required by ORT `ModelCompiler` in ONNX Runtime 1.26.
5. Offline compiler lifecycle: partially implemented. The cache manifest
   validator now provides a deployment gate for manifest/runtime/signature and
   portable artifact layout compatibility, while the EPContext payload validator
   gates generated ORT context artifacts. Opt-in integration tests now compile
   real synthesis and JP-BERT graphs into external and embedded EPContext
   models, then load those precompiled models through payload-driven lazy
   restore. A second opt-in fixture writes a real synthesis GGUF with
   `prepare_cache --write-gguf`, and the `compile-cache` command wraps that
   path as the versioned offline compiler entry point with strict tensor
   mapping, GGUF writing, ready-manifest validation, and compiled-model
   compatibility metadata output. The manifest also records an explicit
   compatibility matrix covering ORT Plugin EP API version, TTS.cpp C API
   contract, GGUF schema expectation, model signature contracts, EPContext
   support level, EPContext payload version, and the compiled-model
   compatibility contract used by ORT model package selection. The native EP
   also implements ORT's compiled-model compatibility callbacks: exact
   provider/runtime/signature/GGUF contract matches return optimal, ORT API
   version drift returns prefer-recompile, unrelated EP metadata is
   not-applicable, and Aivis contract drift is unsupported. The dedicated
   GitHub Actions workflow covers public Plugin EP checks and can run the
   real-artifact compiler/EPContext matrix on manual dispatch or the weekly
   scheduled path when a pinned artifact bundle secret is configured. The
   package now also owns the JP-BERT GGUF writer for TTS.cpp's
   Style-Bert-VITS2 JP-BERT schema, so JP-BERT can be generated from the ONNX
   export plus adjacent `config.json`/tokenizer files, or from a Hugging Face
   JP-BERT directory. A real-artifact JP-BERT parity fixture compares Plugin EP
   `[tokens, 1024]` features against ONNX CPU. Remaining work is expanding the
   hosted matrix across more ORT/TTS.cpp/GGUF schema versions and configuring
   the production artifact bundle secrets.

## Native Build

Build against ONNX Runtime public headers matching the runtime version used by
the application:

```bash
cmake -S native -B build/native \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session
cmake --build build/native
```

The output library is:

- Linux: `libaivis_ggml_onnx_ep.so`
- macOS: `libaivis_ggml_onnx_ep.dylib`
- Windows: `aivis_ggml_onnx_ep.dll`

For package discovery, place the built library under:

```text
src/onnxruntime_ep_aivis_ggml/lib/
```

Smoke registration:

```bash
python tools/smoke_register_plugin.py build/native/libaivis_ggml_onnx_ep.so
# or, after installing the package:
aivis-ggml-onnx-ep-smoke-register build/native/libaivis_ggml_onnx_ep.so
```

Smoke registration plus session creation with explicit provider options:

```bash
python tools/smoke_register_plugin.py build/native/libaivis_ggml_onnx_ep.so \
  --session-smoke \
  --provider-option backend=vulkan \
  --provider-option device=0 \
  --provider-option precision=accurate
```

Smoke the native TTS.cpp binding without graph claim:

```bash
python tools/smoke_register_plugin.py build/native/libaivis_ggml_onnx_ep.so \
  --session-smoke \
  --provider-option backend=cpu \
  --provider-option precision=accurate \
  --provider-option eager_load_model=1 \
  --provider-option n_threads=2 \
  --provider-option tts_cpp_library_path=/path/to/libtts.so \
  --provider-option gguf_path=/path/to/model.gguf
```

Smoke a supported JP-BERT ONNX graph through TTS.cpp:

```bash
python tools/smoke_register_plugin.py build/native/libaivis_ggml_onnx_ep.so \
  --session-smoke \
  --provider-option backend=cpu \
  --provider-option precision=accurate \
  --provider-option eager_load_model=1 \
  --provider-option claim_jp_bert_graph=1 \
  --provider-option n_threads=2 \
  --provider-option tts_cpp_library_path=/path/to/libtts.so \
  --provider-option jp_bert_gguf_path=/path/to/style-bert-vits2-jp-bert.gguf
```

For the current Aivis integration path, the same provider list is shared by the
JP-BERT ONNX session and the synthesis ONNX/AIVMX session. To claim both graphs
without adding per-session Aivis logic, pass both `gguf_path` and
`jp_bert_gguf_path`, plus both `claim_synthesis_graph=1` and
`claim_jp_bert_graph=1`.

Inspect a model graph signature:

```bash
python tools/inspect_model_signature.py /path/to/model.aivmx --fail-if-unsupported
# or, after installing the package:
aivis-ggml-onnx-ep-inspect /path/to/model.aivmx --fail-if-unsupported
```

Prepare the GGML cache manifest for a supported synthesis graph:

```bash
python tools/prepare_cache.py /path/to/model.aivmx \
  --cache-dir /path/to/cache \
  --config-path /path/to/config.json \
  --style-vectors-path /path/to/style_vectors.npy \
  --backend vulkan \
  --precision accurate \
  --write-tensor-pack \
  --write-gguf
# or, after installing the package:
aivis-ggml-onnx-ep-prepare-cache /path/to/model.aivmx \
  --cache-dir /path/to/cache \
  --config-path /path/to/config.json \
  --style-vectors-path /path/to/style_vectors.npy \
  --write-tensor-pack \
  --write-gguf
```

Use `--fail-on-unsupported-mapping` when validating that the current mapper has
enough information to become a real converter.

Run the versioned offline compiler for a release-ready synthesis GGUF artifact:

```bash
python tools/compile_cache.py /path/to/model.aivmx \
  --cache-dir /path/to/cache \
  --config-path /path/to/config.json \
  --style-vectors-path /path/to/style_vectors.npy \
  --backend vulkan \
  --precision accurate \
  --converter-version 0.1.0
# or, after installing the package:
aivis-ggml-onnx-ep-compile-cache /path/to/model.aivmx \
  --cache-dir /path/to/cache \
  --config-path /path/to/config.json \
  --style-vectors-path /path/to/style_vectors.npy \
  --backend vulkan \
  --precision accurate \
  --converter-version 0.1.0
```

This command always enables tensor-pack extraction, strict initializer mapping,
GGUF writing, and `--require-ready` manifest validation. Its JSON output
includes the generated `manifest.json`, `model.gguf`, cache key, and ORT
`ep_compatibility_info` payload for model-package metadata.

Run the JP-BERT GGUF compiler for TTS.cpp's Style-Bert-VITS2 JP-BERT runtime:

```bash
python tools/compile_jp_bert.py \
  --onnx-path /path/to/jp_bert/model.onnx \
  --save-path /path/to/cache/style-bert-vits2-jp-bert.gguf \
  --backend vulkan \
  --precision accurate \
  --converter-version 0.1.0
# or, after installing the package:
aivis-ggml-onnx-ep-compile-jp-bert \
  --onnx-path /path/to/jp_bert/model.onnx \
  --save-path /path/to/cache/style-bert-vits2-jp-bert.gguf \
  --backend vulkan \
  --precision accurate \
  --converter-version 0.1.0
```

The ONNX mode reads `config.json`, `vocab.txt`, and
`tokenizer_config.json` from the same directory as `model.onnx`. For source
Hugging Face checkpoints, pass `--bert-dir /path/to/deberta-v2-large-japanese-char-wwm`
instead; the directory must contain `config.json`, `vocab.txt`, tokenizer
metadata, and either `model.safetensors` or `pytorch_model.bin`.
`model.safetensors` is read without PyTorch; legacy `pytorch_model.bin`
requires installing the `convert-pytorch` extra or otherwise making `torch`
available. Unknown tensor names and incomplete first-N-layer artifacts are hard
failures, because the resulting GGUF must match the tensors TTS.cpp will
actually load. Its JSON output includes the generated JP-BERT GGUF metadata
and a `compiled_model_compatibility_info` payload with `graph_kind=jp-bert` for
ORT model-package metadata.

Validate a prepared cache manifest before deployment:

```bash
python tools/validate_cache_manifest.py /path/to/cache/<key>/manifest.json --require-ready
# or, after installing the package:
aivis-ggml-onnx-ep-validate-cache /path/to/cache/<key>/manifest.json --require-ready
```

Validate an official ORT EPContext payload before shipping a precompiled
context model:

```bash
python tools/validate_ep_context_payload.py /path/to/context/model_aivis_ggml_synthesis_0.json --graph-kind synthesis
# or, after installing the package:
aivis-ggml-onnx-ep-validate-ep-context /path/to/context/model_aivis_ggml_synthesis_0.json --graph-kind synthesis
```

Run the opt-in real-artifact EPContext round-trip fixture:

```bash
AIVIS_GGML_ONNX_EP_TEST=1 \
AIVIS_GGML_ONNX_EP_LIBRARY_PATH=/path/to/libaivis_ggml_onnx_ep.so \
AIVIS_GGML_ONNX_EP_TTS_CPP_LIBRARY_PATH=/path/to/libtts.so \
AIVIS_GGML_ONNX_EP_SYNTHESIS_ONNX_PATH=/path/to/model.aivmx \
AIVIS_GGML_ONNX_EP_SYNTHESIS_GGUF_PATH=/path/to/model.gguf \
AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH=/path/to/jp-bert.onnx \
AIVIS_GGML_ONNX_EP_JP_BERT_GGUF_PATH=/path/to/style-bert-vits2-jp-bert.gguf \
uv run pytest test/integration/test_onnxruntime_ep_aivis_ggml.py::test_aivis_ggml_onnx_ep_compiles_and_loads_ep_context_round_trip -q
```

Run the opt-in real synthesis ONNX-to-GGUF writer fixture:

```bash
AIVIS_GGML_ONNX_EP_CONVERT_TEST=1 \
AIVIS_GGML_ONNX_EP_SYNTHESIS_ONNX_PATH=/path/to/model.aivmx \
AIVIS_GGML_ONNX_EP_SYNTHESIS_CONFIG_PATH=/path/to/config.json \
AIVIS_GGML_ONNX_EP_STYLE_VECTORS_PATH=/path/to/style_vectors.npy \
uv run --with gguf pytest test/integration/test_onnxruntime_ep_aivis_ggml.py::test_aivis_ggml_onnx_ep_prepare_cache_writes_real_synthesis_gguf -q
```

Run the opt-in real JP-BERT GGUF writer fixture:

```bash
AIVIS_GGML_ONNX_EP_JP_BERT_CONVERT_TEST=1 \
AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH=/path/to/jp_bert/model.onnx \
uv run --with gguf pytest test/integration/test_onnxruntime_ep_aivis_ggml.py::test_aivis_ggml_onnx_ep_writes_real_jp_bert_gguf -q
```

For Hugging Face checkpoint sources, use
`AIVIS_GGML_ONNX_EP_JP_BERT_DIR=/path/to/deberta-v2-large-japanese-char-wwm`
instead of `AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH`.

Run the opt-in real JP-BERT feature parity fixture:

```bash
AIVIS_GGML_ONNX_EP_JP_BERT_PARITY_TEST=1 \
AIVIS_GGML_ONNX_EP_LIBRARY_PATH=/path/to/libaivis_ggml_onnx_ep.so \
AIVIS_GGML_ONNX_EP_TTS_CPP_LIBRARY_PATH=/path/to/libtts.so \
AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH=/path/to/jp_bert/model.onnx \
AIVIS_GGML_ONNX_EP_JP_BERT_GGUF_PATH=/path/to/jp_bert/model.gguf \
uv run pytest test/integration/test_onnxruntime_ep_aivis_ggml.py::test_aivis_ggml_onnx_ep_jp_bert_matches_onnx_cpu_features -q
```

This fixture creates an ONNX CPU reference session and an
`AivisGgmlExecutionProvider`-only candidate session, then compares the
`[tokens, 1024]` JP-BERT feature tensor. Defaults are a short no-padding token
probe (`1,5,6,2`), `max_abs_diff <= 0.05`, `rmse <= 0.005`, and
`snr_db >= 35`. Override with
`AIVIS_GGML_ONNX_EP_JP_BERT_INPUT_IDS`,
`AIVIS_GGML_ONNX_EP_JP_BERT_ATTENTION_MASK`,
`AIVIS_GGML_ONNX_EP_JP_BERT_MAX_ABS_DIFF`,
`AIVIS_GGML_ONNX_EP_JP_BERT_RMSE`, and
`AIVIS_GGML_ONNX_EP_JP_BERT_MIN_SNR_DB`.

The hosted workflow `.github/workflows/test-onnxruntime-ggml-ep.yml` runs the
public Plugin EP checks on push and pull request. On manual dispatch or the
weekly scheduled run, it can also run the real-artifact compiler and EPContext
matrix when given a bundle URL via the `artifact_bundle_url` input or the
`AIVIS_GGML_ONNX_EP_ARTIFACT_BUNDLE_URL` secret. The bundle must unpack to this
portable layout:

```text
lib/libtts.so
synthesis/model.aivmx
synthesis/model.gguf
synthesis/config.json
synthesis/style_vectors.npy
jp_bert/model.onnx              # optional JP-BERT graph for EPContext tests
jp_bert/model.gguf              # optional; generated when missing and model.onnx exists
jp_bert/config.json             # required when model.gguf must be generated
jp_bert/vocab.txt               # required when model.gguf must be generated
jp_bert/tokenizer_config.json   # required when model.gguf must be generated
```

The workflow builds the native Plugin EP against ONNX Runtime 1.26 headers,
smoke-registers it, runs `compile_cache.py` with the synthesis files, generates
`jp_bert/model.gguf` with `compile_jp_bert.py` when `jp_bert/model.onnx` is
present and the GGUF is missing, runs the JP-BERT writer fixture when
`jp_bert/model.onnx` is present, runs the JP-BERT ONNX CPU parity fixture when
both JP-BERT ONNX and GGUF artifacts are present, and runs the EPContext
round-trip fixture. The optional `artifact_bundle_sha256` input or
`AIVIS_GGML_ONNX_EP_ARTIFACT_BUNDLE_SHA256` secret pins the downloaded bundle.
