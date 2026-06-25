# GGML ONNX Runtime Plugin Provider

This document defines the non-invasive ONNX Runtime Plugin EP route for the
Style-Bert-VITS2 ggml backend.

## Boundary

AivisSpeech Engine owns only generic ONNX Runtime plugin registration and
provider ordering. It must not own ggml graph execution, TTS.cpp model loading,
GGUF cache policy, Vulkan device routing, or Style-Bert-VITS2 graph matching.

The external package owns:

- ONNX Runtime Plugin EP shared library packaging.
- Provider name `AivisGgmlExecutionProvider`.
- Style-Bert-VITS2 synthesis graph signature detection.
- Style-Bert-VITS2 JP-BERT ONNX graph signature detection.
- ONNX initializer extraction and ggml/GGUF cache creation.
- TTS.cpp runner loading and Vulkan/Metal/CPU backend selection.
- TTS.cpp JP-BERT feature extraction for the supported JP-BERT ONNX graph.
- Returning an ONNX-compatible CPU output tensor to the existing ONNX frontend.

The existing `ggml-vulkan` sidecar/native backend remains an explicit
experimental backend. The Plugin EP path is a separate route whose purpose is
to keep the normal ONNX frontend and AIVMX model flow intact.

## Aivis Registration Contract

Aivis exposes generic Plugin EP controls:

```bash
python run.py \
  --tts_backend onnx \
  --onnx_ep_library_path /path/to/libaivis_ggml_onnx_ep.so \
  --onnx_ep_name AivisGgmlExecutionProvider \
  --onnx_ep_option backend=vulkan \
  --onnx_ep_option device=0 \
  --onnx_ep_option precision=accurate \
  --onnx_ep_option cache_dir=/path/to/cache \
  --onnx_ep_option cache_manifest_path=/path/to/cache/<key>/manifest.json \
  --onnx_ep_option gguf_path=/path/to/cache/<key>/model.gguf \
  --onnx_ep_option jp_bert_gguf_path=/path/to/style-bert-vits2-jp-bert.gguf \
  --onnx_ep_option tts_cpp_library_path=/path/to/libtts.so \
  --onnx_ep_option eager_load_model=1 \
  --onnx_ep_option claim_synthesis_graph=1 \
  --onnx_ep_option claim_jp_bert_graph=1 \
  --onnx_ep_strict
```

`--onnx_ep_library_path` registers the external library with ONNX Runtime.
`--onnx_ep_name` is prepended to the provider list for every Style-Bert-VITS2
ONNX session. If omitted while `--onnx_ep_library_path` is set, Aivis uses
`AivisGgmlExecutionProvider`.

`--onnx_ep_strict` is for validation and benchmark runs. Without it, startup
falls back to the existing ONNX provider list if the plugin cannot be
registered or is not reported by ONNX Runtime.

In strict mode, Aivis also checks the loaded ONNX synthesis session after
`style_bert_vits2.TTSModel.load()`. This catches ONNX Runtime Python's default
behavior of retrying with CPU providers if an EP fails during session creation.
Benchmark and validation code must also check the JP-BERT ONNX session
separately. Style-Bert-VITS2 caches ONNX BERT sessions globally by language, so
a previous ONNX CPU/CUDA run can otherwise hide whether
`claim_jp_bert_graph=1` actually selected `AivisGgmlExecutionProvider`.

The `cache_manifest_path`, `gguf_path`, `jp_bert_gguf_path`,
`tts_cpp_library_path`, `eager_load_model`, and `claim_*_graph` options are
owned by the external package. They let the provider validate the prepared GGUF
cache and dlopen TTS.cpp's C API without adding TTS.cpp-specific code to
AivisSpeech Engine. Graph claim is opt-in: without `claim_synthesis_graph=1`
or `claim_jp_bert_graph=1`, the provider registers successfully but leaves
execution on the fallback ONNX providers.

## Provider Behavior

The provider must claim only complete known graphs. It must not claim individual
ONNX ops. Per-op dispatch would cross the ONNX/ggml boundary repeatedly and is
not a viable performance path.

Capability rules:

- Claim the synthesis graph only when the model signature matches the supported
  Aivis Style-Bert-VITS2 export.
- Claim the JP-BERT ONNX graph only when the model signature matches the
  supported `deberta-v2-large-japanese-char-wwm-onnx` export.
- Do not claim unknown architectures, unknown input/output contracts, or
  unsupported dynamic graph shapes.
- Allow ORT to fall back to CPU/CUDA/DML for all unclaimed graphs.

Compile rules:

- Read graph input/output names from the fused ORT graph and route tensors by
  name, not by original ONNX index. ORT may reorder fused inputs and outputs.
- Read ONNX initializers directly from the ORT graph where conversion is needed.
- Build or load a ggml runner keyed by a stable model graph/initializer hash.
- Use provider options for backend/device/precision/cache configuration.
- Return the same output tensor contract as the original graph:
  Style-Bert-VITS2 synthesis returns `output` as `[1, 1, samples]`; JP-BERT
  returns `output` as `[tokens, 1024]`.

## First Implementation Stages

1. Aivis registration only.
   - Register Plugin EP library.
   - Prepend provider to `onnx_providers`.
   - Preserve default behavior when no `--onnx_ep_*` options are set.

2. Empty external Plugin EP smoke.
   - Package `onnxruntime-ep-aivis-ggml`.
   - Export the Plugin EP factory symbols.
   - Report `AivisGgmlExecutionProvider` to ONNX Runtime.
   - Claim no graph by default.
   - Bootstrap on the ORT CPU hardware device only; ggml/Vulkan device routing
     remains a provider option interpreted later by the external runtime.
   - Expose default provider options from `CreateEpDevice()` and validate
     selected session options in native `CreateEp()`.
   - Optionally validate the TTS.cpp native binding by passing
     `tts_cpp_library_path`, `gguf_path`, and `eager_load_model=1`; this loads
     the GGUF in `CreateEp()` while still leaving execution on fallback
     providers.

3. Graph signature gate.
   - Inspect ONNX graph inputs, outputs, node count, domains, and initializer
     names/hashes.
   - The current supported Aivis Style-Bert-VITS2 ONNX synthesis export has 11
     inputs, 7 outputs, 5334 nodes, 948 initializers, opset 18, and stable
     op-sequence / initializer-name hashes.
   - The supported JP-BERT ONNX export has 2 inputs, 1 output, 3619 raw nodes,
     432 raw initializers, and opset 17. The native gate also accepts observed
     ORT-optimized graph sizes for the same model.
   - The external package keeps the Python inspector as the exact raw-model
     signature checker. Native `GetCapability()` mirrors the structural checks
     available from the ORT graph API and logs match/reject results.
   - Claim only the known synthesis and JP-BERT graphs, and only after the
     corresponding TTS.cpp GGUF has been loaded.
   - Add a debug assignment report using ORT graph assignment info.

4. ggml compile path.
   - Prepare a deterministic cache manifest keyed by source hash, graph
     signature, initializer-name hash, and converter version.
   - Keep the manifest portable: no local absolute model paths are stored.
   - Extract ONNX initializers into `initializers.bin` with per-tensor name,
     dtype, shape, byte offset, byte size, and SHA256 metadata.
   - Generate a conservative TTS.cpp tensor mapping report. Directly mapped
     tensors use names accepted by the local TTS.cpp Style-Bert-VITS2 GGUF
     loader. Complete weight-normalized `weight_g` / `weight_v` pairs are
     materialized into final `weight` tensors with PyTorch's default `dim=0`
     contract; missing or unmapped pairs remain readiness blockers.
     Text-encoder and flow internals use the compact TTS.cpp encoder keys
     (`style_bert_vits2.te.enc.*`, `style_bert_vits2.fl.*`) instead of keeping
     PyTorch/ONNX module paths.
   - Record external converter sources such as `config.json` and
     `style_vectors.npy` by filename, size, and SHA256 only. Do not store local
     absolute paths in the manifest.
   - Include a converter readiness report so incomplete initializer mapping,
     missing style vectors, missing config metadata, or tensor-count mismatches
     block GGUF writing before native graph claim is enabled. The current known
     Aivis AIVMX synthesis exports pass this readiness gate when `config.json`,
     `style_vectors.npy`, and the initializer tensor pack are present.
   - Add a strict GGUF writer entry point. It writes `model.gguf` only when the
     readiness gate is clean; otherwise it fails before creating a partial
     artifact. The writer currently depends on the optional `gguf` Python
     package and writes F32 tensors plus TTS.cpp Style-Bert-VITS2 metadata.
   - Keep new ONNX-specific tensor transforms behind the readiness gate; any
     future export drift must fail as an incomplete converter plan instead of
     producing a partial GGUF.
   - Cache compiled artifacts outside the Aivis project tree.
   - Run a CPU ggml backend first to reduce Vulkan-specific variables.

5. ORT Compile/Compute bridge.
   - Use TTS.cpp `synthesize_front_with_style_vec` for the fused synthesis
     graph. This keeps style vector blending in Aivis and avoids duplicating
     style selection logic inside TTS.cpp.
   - Use TTS.cpp `jp_bert_encode_features` for the fused JP-BERT graph. The
     tokenizer, Japanese normalization, g2p, and `word2ph` expansion remain in
     the existing frontend; only the `[tokens, 1024]` BERT tensor compute is
     replaced.
   - Store fused input/output index maps during `Compile()`. Compute must route
     tensors by ONNX name because ORT can reorder fused inputs and outputs.

6. Vulkan/Metal backend.
   - Enable provider options `backend`, `device`, and `precision`.
   - Load the prepared GGUF through TTS.cpp's native Style-Bert-VITS2 C API.
   - Match ONNX CPU output shape and postprocessing expectations.
   - Benchmark short/medium/long sentences against ONNX CPU and ONNX CUDA.

## Acceptance Gates

- Aivis starts without the plugin and behaves exactly like the current ONNX
  path.
- With `--onnx_ep_strict`, startup fails if the plugin cannot be registered or
  selected.
- With the plugin installed, `get_providers()[0]` for the synthesis session is
  `AivisGgmlExecutionProvider`.
- JP-BERT ONNX sessions continue to use CPU/CUDA/DML unless
  `claim_jp_bert_graph=1` and `jp_bert_gguf_path` are provided.
- With `claim_jp_bert_graph=1`, a supported JP-BERT ONNX session is claimed by
  `AivisGgmlExecutionProvider` and returns `[tokens, 1024]` features.
- JP-BERT feature output is compared against ONNX CPU for matching shape, max
  absolute difference, RMSE, and SNR.
- Short, medium, and long synthesis outputs are compared against ONNX CPU for
  sample count, max absolute difference, RMSE, SNR, and RTF.

## Production Hardening Stages

These stages continue from the first implementation route above. They are
intended to make the Plugin EP production-safe without moving TTS.cpp-specific
logic back into AivisSpeech Engine.

1. Runtime registry and ABI gate.
   - Status: implemented in the native Plugin EP.
   - Identical `tts_cpp_library_path`, backend, device, precision, thread
     count, `gguf_path`, and `jp_bert_gguf_path` combinations share a
     process-local TTS.cpp runtime through a weak registry.
   - Fused compute info holds a shared runtime reference, so the runtime stays
     alive until ORT releases the compiled compute path.
   - Required TTS.cpp C API symbols are resolved before graph claim. Optional
     TTS.cpp runtime ABI and GGUF schema version symbols are enforced when a
     newer TTS.cpp build exports them.

2. Signature contract.
   - Status: implemented in cache tooling and the signature inspector.
   - The Python inspector emits `aivis-ggml-signature-contract-v1`, a stable
     structural hash, synthesis match status, and JP-BERT match status.
   - The cache key includes the structural signature hash in addition to source
     hash, op-sequence hash, initializer-name hash, and converter version.
   - Native `GetCapability()` still mirrors structural checks from the ORT graph
     API because ORT may expose optimized fused graphs instead of the raw ONNX
     file hash.

3. EPContext-lite manifest.
   - Status: implemented as portable manifest metadata.
   - `manifest.json` records `aivis-ggml-ep-context-lite-v1`, provider name,
     provider options, cache key, and relative artifact names. It does not
     store local absolute paths.
   - This is a deployment and cache contract only. It is intentionally marked as
     `official_ort_ep_context.enabled=false` until the native EP creates real
     ORT EPContext nodes.

4. Official ONNX Runtime EPContext.
   - Status: generation implemented, lazy artifact restore inference
     implemented.
   - Native `Compile()` honors ORT's `ep.context_enable` flow for supported
     synthesis and JP-BERT graphs by returning `com.microsoft::EPContext` nodes
     instead of `OrtNodeComputeInfo`.
   - `ep.context_embed_mode=1` embeds an Aivis GGML JSON context payload in
     `ep_cache_context`. `ep.context_embed_mode=0` writes that payload beside
     `ep.context_file_path` and stores only a relative file name.
   - Context payload artifact paths must stay relative to the generated context
     model directory. If `cache_manifest_path`, `gguf_path`, or
     `jp_bert_gguf_path` are outside that directory tree, context generation
     fails instead of storing absolute local paths.
   - Loading and executing precompiled EPContext models works when the
     application passes the deployment-specific `tts_cpp_library_path` and the
     relevant claim flag. The provider claims `source=AivisGgmlExecutionProvider`
     EPContext nodes, restores relative `cache_manifest_path`, `gguf_path`, and
     `jp_bert_gguf_path` values from the payload, lazy-loads TTS.cpp, and routes
     compute through the existing synthesis/JP-BERT bridge.
   - The context payload intentionally does not store `tts_cpp_library_path`
     because shared library paths are deployment-specific.
   - `validate_ep_context_payload.py` can validate the generated payload before
     deployment: version/provider/runtime contracts, graph kind, backend
     options, portable artifact paths, and absence of deployment-specific TTS.cpp
     library paths.
   - ORT 1.26 `ModelCompiler` still requires `Compile()` to return a valid
     `OrtNodeComputeInfo` for each graph even when EPContext generation is
     enabled. The generated EPContext node must also use the exact fused node
     name supplied by ORT; ORT replaces fused nodes by name when building the
     compiled model.

5. Offline compiler lifecycle and compatibility matrix.
   - Status: partially implemented.
   - A cache manifest validator now checks manifest version,
     signature/runtime contracts, EPContext-lite metadata, optional ready
     status, and portable relative artifact paths before deployment.
   - An official EPContext payload validator now gates generated ORT context
     artifacts separately from cache manifests.
   - Opt-in real-artifact fixtures now cover ORT `ModelCompiler` EPContext
     round trips for synthesis and JP-BERT graphs, in both external payload and
     embedded payload modes. The precompiled models are loaded without passing
     `gguf_path` or `jp_bert_gguf_path`, proving payload-driven lazy restore.
   - A second opt-in fixture covers strict synthesis ONNX-to-GGUF writing with
     `prepare_cache --write-gguf` and validates the resulting ready manifest.
   - Every cache manifest records an explicit compatibility matrix: provider
     version, tested ONNX Runtime Plugin EP API version, TTS.cpp C API
     contract, GGUF schema expectation, synthesis/JP-BERT signature contracts,
     EPContext support level, EPContext payload version, and compiled-model
     compatibility contract.
   - The native Plugin EP now implements ORT compiled-model compatibility:
     `GetCompiledModelCompatibilityInfo()` emits the provider/runtime/signature
     contract as portable JSON for `ep_compatibility_info`, and
     `ValidateCompiledModelCompatibilityInfo()` scores that metadata as
     optimal, prefer-recompile, unsupported, or not-applicable.
   - `aivis-ggml-onnx-ep-compile-cache` is the versioned offline compiler
     entry point for synthesis artifacts. It wraps tensor-pack extraction,
     strict initializer mapping, GGUF writing, ready-manifest validation, and
     `ep_compatibility_info` generation for ORT model-package metadata.
   - `aivis-ggml-onnx-ep-compile-jp-bert` is the versioned offline compiler
     entry point for JP-BERT artifacts. It writes TTS.cpp-compatible JP-BERT
     GGUF from ONNX initializers or a Hugging Face checkpoint directory and
     emits `compiled_model_compatibility_info` with `graph_kind=jp-bert` for
     ORT model-package metadata.
   - Hosted CI now has a dedicated `Test ONNX Runtime GGML EP` workflow. Push
     and pull request runs cover Python checks, default Plugin EP integration
     tests, native build, and native smoke registration. Manual dispatch and
     weekly scheduled runs can download a real-artifact bundle, run the
     synthesis compiler, generate the JP-BERT GGUF from `jp_bert/model.onnx`
     when the bundle does not already include `jp_bert/model.gguf`, run the
     JP-BERT writer fixture, compare JP-BERT Plugin EP feature output against
     ONNX CPU, and run the EPContext round-trip matrix.
   - The package now owns a TTS.cpp-compatible Style-Bert-VITS2 JP-BERT GGUF
     writer. It maps Hugging Face DeBERTa tensor names into TTS.cpp's compact
     JP-BERT tensor schema, writes the tokenizer/config metadata consumed by
     TTS.cpp, and fails before writing if any tensor that TTS.cpp will load is
     missing. It supports JP-BERT ONNX initializers and Hugging Face checkpoint
     directories (`model.safetensors` or `pytorch_model.bin`).
   - Remaining production work: configure a pinned production artifact bundle
     in repository secrets and expand the hosted real-artifact matrix across
     ORT/TTS.cpp/GGUF schema versions.
   - Gate deployment on an explicit matrix: ORT API version, Plugin EP version,
     TTS.cpp C API version, GGUF schema version, synthesis signature contract,
     and JP-BERT signature contract.
   - Keep unknown exports and incomplete mappings as hard failures before graph
     claim. Do not silently fall back to partial GGUF generation.
