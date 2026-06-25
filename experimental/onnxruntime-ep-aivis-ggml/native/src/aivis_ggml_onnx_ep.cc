#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <new>
#include <stdexcept>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#if defined(_WIN32)
#define AIVIS_GGML_EP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define AIVIS_GGML_EP_EXPORT __attribute__((visibility("default")))
#else
#define AIVIS_GGML_EP_EXPORT
#endif

namespace {

constexpr const char* kEpName = "AivisGgmlExecutionProvider";
constexpr const char* kVendor = "Aivis Project";
constexpr const char* kVersion = "0.1.0";
constexpr const char* kStage = "synthesis-jp-bert-bridge";
constexpr int64_t kExpectedIrVersion = 8;
constexpr int64_t kExpectedOpsetVersion = 18;
constexpr const char* kExpectedGraphName = "main_graph";
constexpr size_t kExpectedOutputCount = 7;
constexpr const char* kExpectedFirstOutputName = "output";
constexpr size_t kExpectedNodeCount = 5334;
constexpr size_t kExpectedOptimizedNodeCount = 5332;
constexpr size_t kExpectedInitializerCount = 948;
constexpr size_t kExpectedOptimizedInitializerCount = 949;
constexpr const char* kDefaultBackend = "vulkan";
constexpr const char* kDefaultPrecision = "accurate";
constexpr int64_t kExpectedJpBertOpsetVersion = 17;
constexpr size_t kExpectedJpBertInputCount = 2;
constexpr size_t kExpectedJpBertOutputCount = 1;
constexpr size_t kExpectedJpBertNodeCount = 3619;
constexpr size_t kExpectedJpBertInitializerCount = 432;
constexpr std::array<size_t, 3> kAcceptedJpBertNodeCounts = {3619, 3092, 3180};
constexpr std::array<size_t, 3> kAcceptedJpBertInitializerCounts = {432, 521, 543};
constexpr const char* kExpectedJpBertOutputName = "output";

constexpr std::array<const char*, 11> kExpectedInputNames = {
    "x_tst",
    "x_tst_lengths",
    "sid",
    "tones",
    "language",
    "bert",
    "style_vec",
    "length_scale",
    "sdp_ratio",
    "noise_scale",
    "noise_scale_w",
};

constexpr std::array<ONNXTensorElementDataType, 11> kExpectedInputTypes = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
};

constexpr std::array<const char*, 4> kRequiredInitializerNames = {
    "enc_p.emb.weight",
    "enc_p.tone_emb.weight",
    "enc_p.language_emb.weight",
    "enc_p.bert_proj.weight",
};

constexpr std::array<const char*, 5> kRequiredOpTypes = {
    "Conv",
    "Gather",
    "MatMul",
    "RandomNormalLike",
    "Tanh",
};

constexpr std::array<const char*, 2> kExpectedJpBertInputNames = {
    "input_ids",
    "attention_mask",
};

constexpr std::array<ONNXTensorElementDataType, 2> kExpectedJpBertInputTypes = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
};

constexpr std::array<const char*, 5> kRequiredJpBertOpTypes = {
    "Gather",
    "LayerNormalization",
    "MatMul",
    "Reshape",
    "Where",
};

OrtStatus* CreateStatus(const OrtApi& api, OrtErrorCode code, const char* message) noexcept {
  return api.CreateStatus(code, message);
}

std::string ToLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

void LogMessage(
    const OrtApi& api,
    const OrtLogger* logger,
    OrtLoggingLevel level,
    const std::string& message) noexcept {
  if (logger == nullptr) {
    return;
  }

  OrtStatus* status = api.Logger_LogMessage(
      logger,
      level,
      message.c_str(),
      ORT_FILE,
      __LINE__,
      __FUNCTION__);
  if (status != nullptr) {
    api.ReleaseStatus(status);
  }
}

bool IsTraceEnabled() {
  const char* value = std::getenv("AIVIS_GGML_EP_TRACE");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void TraceMessage(const std::string& message) {
  if (IsTraceEnabled()) {
    std::cerr << "AIVIS_GGML_EP_TRACE " << message << std::endl;
  }
}

const char* TensorElementTypeName(ONNXTensorElementDataType type) noexcept {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "INT64";
    default:
      return "OTHER";
  }
}

ONNXTensorElementDataType TensorElementType(Ort::ConstValueInfo value_info) {
  Ort::ConstTypeInfo type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetElementType();
}

std::string JoinReasons(const std::vector<std::string>& reasons) {
  std::ostringstream out;
  for (size_t i = 0; i < reasons.size(); ++i) {
    if (i != 0) {
      out << "; ";
    }
    out << reasons[i];
  }
  return out.str();
}

bool IsAcceptedGraphName(const std::string& graph_name) {
  return graph_name == kExpectedGraphName || graph_name.rfind(kEpName, 0) == 0;
}

template <size_t N>
std::vector<size_t> BuildInputIndices(
    const OrtGraph* ort_graph,
    const std::array<const char*, N>& expected_input_names) {
  Ort::ConstGraph graph{ort_graph};
  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  std::vector<size_t> indices;
  indices.reserve(N);
  for (const char* expected_name : expected_input_names) {
    bool found = false;
    for (size_t index = 0; index < inputs.size(); ++index) {
      if (inputs[index].GetName() == expected_name) {
        indices.push_back(index);
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error(std::string("compiled graph input is missing: ") + expected_name);
    }
  }
  return indices;
}

size_t BuildOutputIndex(const OrtGraph* ort_graph, const char* expected_output_name) {
  Ort::ConstGraph graph{ort_graph};
  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index].GetName() == expected_output_name) {
      return index;
    }
  }
  throw std::runtime_error(std::string("compiled graph output is missing: ") + expected_output_name);
}

struct AivisGgmlEpConfig {
  std::string backend = kDefaultBackend;
  std::string device;
  std::string precision = kDefaultPrecision;
  std::string cache_dir;
  std::string cache_manifest_path;
  std::string gguf_path;
  std::string jp_bert_gguf_path;
  std::string tts_cpp_library_path;
  bool eager_load_model = false;
  bool claim_synthesis_graph = false;
  bool claim_jp_bert_graph = false;
  int n_threads = 0;
};

std::string ReadEpOption(
    const OrtSessionOptions* session_options,
    const std::string& option_name,
    const std::string& default_value) {
  if (session_options == nullptr) {
    return default_value;
  }

  Ort::ConstSessionOptions options{session_options};
  const std::string ep_name = kEpName;
  const std::string lower_ep_name = ToLowerAscii(ep_name);
  const std::array<std::string, 4> candidates = {
      "ep." + lower_ep_name + "." + option_name,
      "ep." + ep_name + "." + option_name,
      "ep_factory." + lower_ep_name + "." + option_name,
      "ep_factory." + ep_name + "." + option_name,
  };

  for (const std::string& candidate : candidates) {
    if (options.HasConfigEntry(candidate.c_str())) {
      return options.GetConfigEntry(candidate.c_str());
    }
  }
  return default_value;
}

bool ParseBoolOption(const std::string& value, bool default_value, const char* option_name) {
  if (value.empty()) {
    return default_value;
  }
  const std::string lowered = ToLowerAscii(value);
  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
    return false;
  }
  throw std::invalid_argument(std::string("Invalid boolean value for ") + option_name + ": " + value);
}

int ParseNonNegativeIntOption(const std::string& value, int default_value, const char* option_name) {
  if (value.empty()) {
    return default_value;
  }
  size_t consumed = 0;
  unsigned long parsed = 0;
  try {
    parsed = std::stoul(value, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("Invalid integer value for ") + option_name + ": " + value);
  }
  if (consumed != value.size() || parsed > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string("Invalid integer value for ") + option_name + ": " + value);
  }
  return static_cast<int>(parsed);
}

AivisGgmlEpConfig ReadEpConfig(const OrtSessionOptions* session_options) {
  AivisGgmlEpConfig config;
  config.backend = ReadEpOption(session_options, "backend", kDefaultBackend);
  config.device = ReadEpOption(session_options, "device", "");
  config.precision = ReadEpOption(session_options, "precision", kDefaultPrecision);
  config.cache_dir = ReadEpOption(session_options, "cache_dir", "");
  config.cache_manifest_path = ReadEpOption(session_options, "cache_manifest_path", "");
  config.gguf_path = ReadEpOption(session_options, "gguf_path", "");
  config.jp_bert_gguf_path = ReadEpOption(session_options, "jp_bert_gguf_path", "");
  config.tts_cpp_library_path = ReadEpOption(session_options, "tts_cpp_library_path", "");
  config.eager_load_model = ParseBoolOption(
      ReadEpOption(session_options, "eager_load_model", "0"),
      false,
      "eager_load_model");
  config.claim_synthesis_graph = ParseBoolOption(
      ReadEpOption(session_options, "claim_synthesis_graph", "0"),
      false,
      "claim_synthesis_graph");
  config.claim_jp_bert_graph = ParseBoolOption(
      ReadEpOption(session_options, "claim_jp_bert_graph", "0"),
      false,
      "claim_jp_bert_graph");
  config.n_threads = ParseNonNegativeIntOption(
      ReadEpOption(session_options, "n_threads", "0"),
      0,
      "n_threads");
  return config;
}

bool IsSupportedBackend(const std::string& backend) {
  return backend == "vulkan" || backend == "metal" || backend == "cpu";
}

bool IsSupportedPrecision(const std::string& precision) {
  return precision == "accurate" || precision == "fast";
}

bool PathExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::error_code error;
  return std::filesystem::exists(std::filesystem::path(path), error) && !error;
}

std::string ReadSmallTextFile(const std::string& path, size_t max_bytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("could not open file: " + path);
  }
  std::string data;
  data.resize(max_bytes + 1);
  file.read(data.data(), static_cast<std::streamsize>(data.size()));
  const std::streamsize bytes_read = file.gcount();
  if (bytes_read > static_cast<std::streamsize>(max_bytes)) {
    throw std::runtime_error("file is too large for bootstrap validation: " + path);
  }
  data.resize(static_cast<size_t>(bytes_read));
  return data;
}

bool ManifestContainsReadyCache(const std::string& manifest_text) {
  return manifest_text.find("aivis-ggml-onnx-cache-v1") != std::string::npos &&
         manifest_text.find("\"status\": \"ready\"") != std::string::npos &&
         manifest_text.find("\"can_write_gguf\": true") != std::string::npos;
}

OrtStatus* ValidateEpConfig(const OrtApi& api, const AivisGgmlEpConfig& config) noexcept {
  if (!IsSupportedBackend(config.backend)) {
    return CreateStatus(
        api,
        ORT_INVALID_ARGUMENT,
        "AivisGgmlExecutionProvider option backend must be one of: vulkan, metal, cpu.");
  }
  if (!IsSupportedPrecision(config.precision)) {
    return CreateStatus(
        api,
        ORT_INVALID_ARGUMENT,
        "AivisGgmlExecutionProvider option precision must be one of: accurate, fast.");
  }
  try {
    if (!config.cache_manifest_path.empty()) {
      if (!PathExists(config.cache_manifest_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider option cache_manifest_path does not exist.");
      }
      const std::string manifest = ReadSmallTextFile(config.cache_manifest_path, 1024 * 1024);
      if (!ManifestContainsReadyCache(manifest)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider cache_manifest_path is not a ready Aivis GGML cache manifest.");
      }
    }
    if (config.eager_load_model) {
      if (config.tts_cpp_library_path.empty() || !PathExists(config.tts_cpp_library_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider eager_load_model requires an existing tts_cpp_library_path.");
      }
      if (config.gguf_path.empty() && config.jp_bert_gguf_path.empty()) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider eager_load_model requires gguf_path or jp_bert_gguf_path.");
      }
      if (!config.gguf_path.empty() && !PathExists(config.gguf_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider option gguf_path does not exist.");
      }
      if (!config.jp_bert_gguf_path.empty() && !PathExists(config.jp_bert_gguf_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider option jp_bert_gguf_path does not exist.");
      }
    }
    if (config.claim_synthesis_graph && !config.eager_load_model) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider claim_synthesis_graph requires eager_load_model=1.");
    }
    if (config.claim_synthesis_graph && config.gguf_path.empty()) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider claim_synthesis_graph requires gguf_path.");
    }
    if (config.claim_jp_bert_graph && !config.eager_load_model) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider claim_jp_bert_graph requires eager_load_model=1.");
    }
    if (config.claim_jp_bert_graph && config.jp_bert_gguf_path.empty()) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider claim_jp_bert_graph requires jp_bert_gguf_path.");
    }
  } catch (const std::exception& ex) {
    return CreateStatus(api, ORT_INVALID_ARGUMENT, ex.what());
  }
  return nullptr;
}

std::string ConfigSummary(const AivisGgmlEpConfig& config) {
  std::ostringstream out;
  out << "backend=" << config.backend
      << ", device=" << (config.device.empty() ? "default" : config.device)
      << ", precision=" << config.precision
      << ", cache_dir_set=" << (config.cache_dir.empty() ? "false" : "true")
      << ", cache_manifest_set=" << (config.cache_manifest_path.empty() ? "false" : "true")
      << ", gguf_path_set=" << (config.gguf_path.empty() ? "false" : "true")
      << ", jp_bert_gguf_path_set=" << (config.jp_bert_gguf_path.empty() ? "false" : "true")
      << ", tts_cpp_library_set=" << (config.tts_cpp_library_path.empty() ? "false" : "true")
      << ", eager_load_model=" << (config.eager_load_model ? "true" : "false")
      << ", claim_synthesis_graph=" << (config.claim_synthesis_graph ? "true" : "false")
      << ", claim_jp_bert_graph=" << (config.claim_jp_bert_graph ? "true" : "false")
      << ", n_threads=" << config.n_threads;
  return out.str();
}

class DynamicLibrary final {
 public:
  static std::unique_ptr<DynamicLibrary> Load(const std::string& path) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == nullptr) {
      throw std::runtime_error("could not load dynamic library: " + path);
    }
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      throw std::runtime_error(
          std::string("could not load dynamic library: ") + path +
          (error != nullptr ? std::string(": ") + error : std::string()));
    }
#endif
    return std::unique_ptr<DynamicLibrary>(new DynamicLibrary(handle));
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }

  void* Symbol(const char* name) const {
#if defined(_WIN32)
    void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    dlerror();
    void* symbol = dlsym(handle_, name);
#endif
    if (symbol == nullptr) {
      throw std::runtime_error(std::string("missing TTS.cpp symbol: ") + name);
    }
    return symbol;
  }

 private:
#if defined(_WIN32)
  explicit DynamicLibrary(HMODULE handle) : handle_(handle) {}
  HMODULE handle_;
#else
  explicit DynamicLibrary(void* handle) : handle_(handle) {}
  void* handle_;
#endif
};

struct tts_style_bert_vits2_handle;
struct tts_style_bert_vits2_jp_bert_handle;

struct tts_style_bert_vits2_float_buffer {
  const float* data;
  size_t length;
  uint32_t hidden_size;
  float sample_rate;
};

size_t CheckedElementCount(const std::vector<int64_t>& shape, const std::string& name) {
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error(name + " has a dynamic runtime dimension.");
    }
    if (dim != 0 && count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error(name + " element count overflows size_t.");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

void ExpectTensorElementType(
    Ort::ConstValue value,
    ONNXTensorElementDataType expected_type,
    const std::string& name) {
  const ONNXTensorElementDataType actual_type =
      value.GetTensorTypeAndShapeInfo().GetElementType();
  if (actual_type != expected_type) {
    throw std::runtime_error(
        name + " has element type " + TensorElementTypeName(actual_type) +
        ", expected " + TensorElementTypeName(expected_type) + ".");
  }
}

size_t ExpectTokenMatrix(
    Ort::ConstValue value,
    ONNXTensorElementDataType expected_type,
    const std::string& name) {
  if (value == nullptr) {
    throw std::runtime_error(name + " input is missing.");
  }
  ExpectTensorElementType(value, expected_type, name);
  const std::vector<int64_t> shape = TensorShape(value);
  if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0) {
    throw std::runtime_error(name + " must have shape [1, tokens], got " + ShapeString(shape) + ".");
  }
  return static_cast<size_t>(shape[1]);
}

std::vector<int32_t> CopyInt64TokenInput(
    Ort::ConstValue value,
    size_t expected_tokens,
    const std::string& name) {
  const size_t tokens = ExpectTokenMatrix(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, name);
  if (tokens != expected_tokens) {
    throw std::runtime_error(name + " token count does not match x_tst.");
  }
  const int64_t* source = value.GetTensorData<int64_t>();
  std::vector<int32_t> output(tokens);
  for (size_t i = 0; i < tokens; ++i) {
    if (source[i] < std::numeric_limits<int32_t>::min() ||
        source[i] > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(name + " contains a value outside int32 range.");
    }
    output[i] = static_cast<int32_t>(source[i]);
  }
  return output;
}

int32_t ReadInt64ScalarVectorInput(Ort::ConstValue value, const std::string& name) {
  if (value == nullptr) {
    throw std::runtime_error(name + " input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, name);
  const std::vector<int64_t> shape = TensorShape(value);
  if (CheckedElementCount(shape, name) != 1) {
    throw std::runtime_error(name + " must contain exactly one scalar value.");
  }
  const int64_t raw_value = value.GetTensorData<int64_t>()[0];
  if (raw_value < std::numeric_limits<int32_t>::min() ||
      raw_value > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(name + " is outside int32 range.");
  }
  return static_cast<int32_t>(raw_value);
}

float ReadFloatScalarInput(Ort::ConstValue value, const std::string& name) {
  if (value == nullptr) {
    throw std::runtime_error(name + " input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, name);
  const std::vector<int64_t> shape = TensorShape(value);
  if (CheckedElementCount(shape, name) != 1) {
    throw std::runtime_error(name + " must contain exactly one scalar value.");
  }
  return value.GetTensorData<float>()[0];
}

const float* ExpectBertInput(Ort::ConstValue value, size_t tokens) {
  if (value == nullptr) {
    throw std::runtime_error("bert input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "bert");
  const std::vector<int64_t> shape = TensorShape(value);
  if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1024 ||
      shape[2] != static_cast<int64_t>(tokens)) {
    throw std::runtime_error("bert must have shape [1, 1024, tokens], got " + ShapeString(shape) + ".");
  }
  return value.GetTensorData<float>();
}

const float* ExpectStyleVecInput(Ort::ConstValue value) {
  if (value == nullptr) {
    throw std::runtime_error("style_vec input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "style_vec");
  const std::vector<int64_t> shape = TensorShape(value);
  if (shape.size() != 2 || shape[0] != 1 || shape[1] != 256) {
    throw std::runtime_error("style_vec must have shape [1, 256], got " + ShapeString(shape) + ".");
  }
  return value.GetTensorData<float>();
}

class TtsCppRuntime final {
 public:
  using LastErrorFn = const char* (*)();
  using LoadModelFn = int (*)(
      const char* model_path,
      int n_threads,
      int cpu_only,
      tts_style_bert_vits2_handle** out_handle);
  using FreeModelFn = void (*)(tts_style_bert_vits2_handle* handle);
  using LoadJpBertModelFn = int (*)(
      const char* model_path,
      int n_threads,
      int cpu_only,
      tts_style_bert_vits2_jp_bert_handle** out_handle);
  using FreeJpBertModelFn = void (*)(tts_style_bert_vits2_jp_bert_handle* handle);
  using SynthesizeFrontFn = int (*)(
      tts_style_bert_vits2_handle* handle,
      const int32_t* phone_ids,
      const int32_t* tone_ids,
      const int32_t* language_ids,
      size_t tokens,
      const float* bert,
      size_t bert_length,
      int32_t speaker_id,
      int32_t style_id,
      float style_weight,
      float sdp_ratio,
      float length_scale,
      float noise_scale,
      float noise_w_scale,
      tts_style_bert_vits2_float_buffer* out_audio);
  using SynthesizeFrontWithStyleVecFn = int (*)(
      tts_style_bert_vits2_handle* handle,
      const int32_t* phone_ids,
      const int32_t* tone_ids,
      const int32_t* language_ids,
      size_t tokens,
      const float* bert,
      size_t bert_length,
      const float* style_vec,
      size_t style_vec_length,
      int32_t speaker_id,
      float sdp_ratio,
      float length_scale,
      float noise_scale,
      float noise_w_scale,
      tts_style_bert_vits2_float_buffer* out_audio);
  using EncodeJpBertFeaturesFn = int (*)(
      tts_style_bert_vits2_jp_bert_handle* handle,
      const int32_t* input_ids,
      size_t tokens,
      tts_style_bert_vits2_float_buffer* out_features);

  static std::unique_ptr<TtsCppRuntime> LoadAndMaybeOpenModel(const AivisGgmlEpConfig& config) {
    auto library = DynamicLibrary::Load(config.tts_cpp_library_path);
    auto runtime = std::unique_ptr<TtsCppRuntime>(new TtsCppRuntime(std::move(library)));
    runtime->last_error_ = reinterpret_cast<LastErrorFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_last_error"));
    runtime->load_model_ = reinterpret_cast<LoadModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_load_model"));
    runtime->free_model_ = reinterpret_cast<FreeModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_free_model"));
    runtime->load_jp_bert_model_ = reinterpret_cast<LoadJpBertModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_jp_bert_load_model"));
    runtime->free_jp_bert_model_ = reinterpret_cast<FreeJpBertModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_jp_bert_free_model"));
    runtime->synthesize_front_ = reinterpret_cast<SynthesizeFrontFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_synthesize_front"));
    runtime->synthesize_front_with_style_vec_ = reinterpret_cast<SynthesizeFrontWithStyleVecFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_synthesize_front_with_style_vec"));
    runtime->encode_jp_bert_features_ = reinterpret_cast<EncodeJpBertFeaturesFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_jp_bert_encode_features"));

    const int cpu_only = config.backend == "cpu" ? 1 : 0;
    if (!config.gguf_path.empty()) {
      tts_style_bert_vits2_handle* handle = nullptr;
      if (runtime->load_model_(config.gguf_path.c_str(), config.n_threads, cpu_only, &handle) == 0 || handle == nullptr) {
        const char* detail = runtime->last_error_ != nullptr ? runtime->last_error_() : nullptr;
        throw std::runtime_error(
            std::string("TTS.cpp failed to load Style-Bert-VITS2 GGUF") +
            (detail != nullptr && detail[0] ? std::string(": ") + detail : std::string(".")));
      }
      runtime->model_handle_ = handle;
    }
    if (!config.jp_bert_gguf_path.empty()) {
      tts_style_bert_vits2_jp_bert_handle* handle = nullptr;
      if (runtime->load_jp_bert_model_(config.jp_bert_gguf_path.c_str(), config.n_threads, cpu_only, &handle) == 0 ||
          handle == nullptr) {
        const char* detail = runtime->last_error_ != nullptr ? runtime->last_error_() : nullptr;
        throw std::runtime_error(
            std::string("TTS.cpp failed to load Style-Bert-VITS2 JP-BERT GGUF") +
            (detail != nullptr && detail[0] ? std::string(": ") + detail : std::string(".")));
      }
      runtime->jp_bert_handle_ = handle;
    }
    return runtime;
  }

  TtsCppRuntime(const TtsCppRuntime&) = delete;
  TtsCppRuntime& operator=(const TtsCppRuntime&) = delete;

  ~TtsCppRuntime() {
    if (jp_bert_handle_ != nullptr && free_jp_bert_model_ != nullptr) {
      free_jp_bert_model_(jp_bert_handle_);
    }
    if (model_handle_ != nullptr && free_model_ != nullptr) {
      free_model_(model_handle_);
    }
  }

  bool HasSynthesisModel() const noexcept {
    return model_handle_ != nullptr;
  }

  bool HasJpBertModel() const noexcept {
    return jp_bert_handle_ != nullptr;
  }

  bool SynthesizeFrontWithStyleVec(
      const int32_t* phone_ids,
      const int32_t* tone_ids,
      const int32_t* language_ids,
      size_t tokens,
      const float* bert,
      size_t bert_length,
      const float* style_vec,
      size_t style_vec_length,
      int32_t speaker_id,
      float sdp_ratio,
      float length_scale,
      float noise_scale,
      float noise_w_scale,
      tts_style_bert_vits2_float_buffer& out_audio,
      std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (synthesize_front_with_style_vec_ == nullptr || model_handle_ == nullptr) {
      error = "TTS.cpp Style-Bert-VITS2 direct style_vec synthesis API is not loaded.";
      return false;
    }
    const int ok = synthesize_front_with_style_vec_(
        model_handle_,
        phone_ids,
        tone_ids,
        language_ids,
        tokens,
        bert,
        bert_length,
        style_vec,
        style_vec_length,
        speaker_id,
        sdp_ratio,
        length_scale,
        noise_scale,
        noise_w_scale,
        &out_audio);
    if (ok == 0) {
      const char* detail = last_error_ != nullptr ? last_error_() : nullptr;
      error = detail != nullptr && detail[0] ? detail : "TTS.cpp Style-Bert-VITS2 synthesis failed.";
      return false;
    }
    return true;
  }

  bool EncodeJpBertFeatures(
      const int32_t* input_ids,
      size_t tokens,
      tts_style_bert_vits2_float_buffer& out_features,
      std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (encode_jp_bert_features_ == nullptr || jp_bert_handle_ == nullptr) {
      error = "TTS.cpp Style-Bert-VITS2 JP-BERT API is not loaded.";
      return false;
    }
    const int ok = encode_jp_bert_features_(
        jp_bert_handle_,
        input_ids,
        tokens,
        &out_features);
    if (ok == 0) {
      const char* detail = last_error_ != nullptr ? last_error_() : nullptr;
      error = detail != nullptr && detail[0] ? detail : "TTS.cpp Style-Bert-VITS2 JP-BERT feature extraction failed.";
      return false;
    }
    return true;
  }

 private:
  explicit TtsCppRuntime(std::unique_ptr<DynamicLibrary> library)
      : library_(std::move(library)) {}

  std::unique_ptr<DynamicLibrary> library_;
  LastErrorFn last_error_ = nullptr;
  LoadModelFn load_model_ = nullptr;
  FreeModelFn free_model_ = nullptr;
  LoadJpBertModelFn load_jp_bert_model_ = nullptr;
  FreeJpBertModelFn free_jp_bert_model_ = nullptr;
  SynthesizeFrontFn synthesize_front_ = nullptr;
  SynthesizeFrontWithStyleVecFn synthesize_front_with_style_vec_ = nullptr;
  EncodeJpBertFeaturesFn encode_jp_bert_features_ = nullptr;
  tts_style_bert_vits2_handle* model_handle_ = nullptr;
  tts_style_bert_vits2_jp_bert_handle* jp_bert_handle_ = nullptr;
  std::mutex mutex_;
};

enum class AivisGgmlGraphKind {
  Synthesis,
  JpBert,
};

const char* GraphKindName(AivisGgmlGraphKind graph_kind) noexcept {
  switch (graph_kind) {
    case AivisGgmlGraphKind::Synthesis:
      return "synthesis";
    case AivisGgmlGraphKind::JpBert:
      return "jp-bert";
  }
  return "unknown";
}

struct AivisGgmlNodeComputeInfo final : OrtNodeComputeInfo {
  AivisGgmlNodeComputeInfo(
      const OrtApi& ort_api,
      TtsCppRuntime* runtime,
      AivisGgmlGraphKind graph_kind,
      std::vector<size_t> input_indices,
      size_t primary_output_index)
      : OrtNodeComputeInfo{},
        ort_api_(ort_api),
        runtime_(runtime),
        graph_kind_(graph_kind),
        input_indices_(std::move(input_indices)),
        primary_output_index_(primary_output_index) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr,
      OrtNodeComputeContext* compute_context,
      void** compute_state) noexcept {
    auto* info = static_cast<AivisGgmlNodeComputeInfo*>(this_ptr);
    if (compute_state == nullptr) {
      return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, "Aivis GGML compute state output is null.");
    }

    *compute_state = nullptr;
    try {
      const OrtEpApi* ep_api = info->ort_api_.GetEpApi();
      const char* node_name =
          ep_api != nullptr && compute_context != nullptr
              ? ep_api->NodeComputeContext_NodeName(compute_context)
              : nullptr;
      auto state = std::make_unique<std::string>(node_name != nullptr ? node_name : "");
      *compute_state = state.release();
      return nullptr;
    } catch (const std::bad_alloc&) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Out of memory creating Aivis GGML compute state.");
    } catch (const std::exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Unknown error creating Aivis GGML compute state.");
    }
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* this_ptr,
      void* compute_state,
      OrtKernelContext* kernel_context) noexcept {
    auto* info = static_cast<AivisGgmlNodeComputeInfo*>(this_ptr);
    if (kernel_context == nullptr) {
      return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, "Aivis GGML Compute received a null kernel context.");
    }
    if (info->runtime_ == nullptr) {
      return CreateStatus(
          info->ort_api_,
          ORT_NOT_IMPLEMENTED,
          "Aivis GGML Compute requires an eager-loaded TTS.cpp runtime.");
    }
    if (info->graph_kind_ == AivisGgmlGraphKind::JpBert) {
      return ComputeJpBertImpl(info, compute_state, kernel_context);
    }

    try {
      Ort::KernelContext context(kernel_context);
      const size_t input_count = context.GetInputCount();
      const size_t output_count = context.GetOutputCount();
      if (input_count != kExpectedInputNames.size()) {
        std::ostringstream message;
        message << "Aivis GGML Compute expected " << kExpectedInputNames.size()
                << " inputs, got " << input_count << ".";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, error_message.c_str());
      }
      if (output_count == 0) {
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, "Aivis GGML Compute expected at least one output.");
      }

      if (info->input_indices_.size() != kExpectedInputNames.size()) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML synthesis input index map is invalid.");
      }
      Ort::ConstValue x_tst = context.GetInput(info->input_indices_[0]);
      const size_t tokens = ExpectTokenMatrix(x_tst, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "x_tst");
      std::vector<int32_t> phone_ids = CopyInt64TokenInput(x_tst, tokens, "x_tst");
      const int32_t x_tst_length = ReadInt64ScalarVectorInput(context.GetInput(info->input_indices_[1]), "x_tst_lengths");
      if (x_tst_length < 0 || static_cast<size_t>(x_tst_length) != tokens) {
        throw std::runtime_error("x_tst_lengths does not match x_tst token count.");
      }
      const int32_t speaker_id = ReadInt64ScalarVectorInput(context.GetInput(info->input_indices_[2]), "sid");
      std::vector<int32_t> tone_ids = CopyInt64TokenInput(context.GetInput(info->input_indices_[3]), tokens, "tones");
      std::vector<int32_t> language_ids = CopyInt64TokenInput(context.GetInput(info->input_indices_[4]), tokens, "language");
      const float* bert = ExpectBertInput(context.GetInput(info->input_indices_[5]), tokens);
      const float* style_vec = ExpectStyleVecInput(context.GetInput(info->input_indices_[6]));
      const float length_scale = ReadFloatScalarInput(context.GetInput(info->input_indices_[7]), "length_scale");
      const float sdp_ratio = ReadFloatScalarInput(context.GetInput(info->input_indices_[8]), "sdp_ratio");
      const float noise_scale = ReadFloatScalarInput(context.GetInput(info->input_indices_[9]), "noise_scale");
      const float noise_scale_w = ReadFloatScalarInput(context.GetInput(info->input_indices_[10]), "noise_scale_w");
      if (tokens > std::numeric_limits<size_t>::max() / 1024) {
        throw std::runtime_error("bert element count overflows size_t.");
      }

      tts_style_bert_vits2_float_buffer audio{};
      std::string synthesis_error;
      if (!info->runtime_->SynthesizeFrontWithStyleVec(
              phone_ids.data(),
              tone_ids.data(),
              language_ids.data(),
              tokens,
              bert,
              tokens * 1024,
              style_vec,
              256,
              speaker_id,
              sdp_ratio,
              length_scale,
              noise_scale,
              noise_scale_w,
              audio,
              synthesis_error)) {
        return CreateStatus(info->ort_api_, ORT_FAIL, synthesis_error.c_str());
      }
      if (audio.hidden_size != 1) {
        std::ostringstream message;
        message << "TTS.cpp returned hidden_size=" << audio.hidden_size << ", expected 1.";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_FAIL, error_message.c_str());
      }
      if (audio.length > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp returned an audio tensor too large for ONNX shape.");
      }

      const std::vector<int64_t> output_shape = {1, 1, static_cast<int64_t>(audio.length)};
      Ort::UnownedValue output = context.GetOutput(info->primary_output_index_, output_shape);
      if (output == nullptr) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML Compute could not allocate output tensor.");
      }
      float* output_data = output.GetTensorMutableData<float>();
      if (audio.length > 0) {
        if (audio.data == nullptr) {
          return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp returned null audio data.");
        }
        std::copy(audio.data, audio.data + audio.length, output_data);
      }

      const std::vector<int64_t> placeholder_shape = {1};
      for (size_t output_index = 0; output_index < output_count; ++output_index) {
        if (output_index == info->primary_output_index_) {
          continue;
        }
        Ort::UnownedValue placeholder = context.GetOutput(output_index, placeholder_shape);
        if (placeholder != nullptr) {
          placeholder.GetTensorMutableData<float>()[0] = 0.0f;
        }
      }

      const auto* node_name = static_cast<const std::string*>(compute_state);
      std::ostringstream trace;
      trace << "synthesized samples=" << audio.length << " tokens=" << tokens;
      if (node_name != nullptr && !node_name->empty()) {
        trace << " node='" << *node_name << "'";
      }
      TraceMessage(trace.str());
      return nullptr;
    } catch (const Ort::Exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (const std::bad_alloc&) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Out of memory during Aivis GGML Compute.");
    } catch (const std::exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Unknown error during Aivis GGML Compute.");
    }
  }

  static OrtStatus* ComputeJpBertImpl(
      AivisGgmlNodeComputeInfo* info,
      void* compute_state,
      OrtKernelContext* kernel_context) noexcept {
    if (!info->runtime_->HasJpBertModel()) {
      return CreateStatus(
          info->ort_api_,
          ORT_NOT_IMPLEMENTED,
          "Aivis GGML JP-BERT Compute requires an eager-loaded TTS.cpp JP-BERT runtime.");
    }

    try {
      Ort::KernelContext context(kernel_context);
      const size_t input_count = context.GetInputCount();
      const size_t output_count = context.GetOutputCount();
      if (input_count != kExpectedJpBertInputCount) {
        std::ostringstream message;
        message << "Aivis GGML JP-BERT Compute expected " << kExpectedJpBertInputCount
                << " inputs, got " << input_count << ".";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, error_message.c_str());
      }
      if (output_count != kExpectedJpBertOutputCount) {
        std::ostringstream message;
        message << "Aivis GGML JP-BERT Compute expected " << kExpectedJpBertOutputCount
                << " output, got " << output_count << ".";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, error_message.c_str());
      }

      if (info->input_indices_.size() != kExpectedJpBertInputNames.size()) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML JP-BERT input index map is invalid.");
      }
      Ort::ConstValue input_ids_value = context.GetInput(info->input_indices_[0]);
      const size_t tokens = ExpectTokenMatrix(input_ids_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "input_ids");
      std::vector<int32_t> input_ids = CopyInt64TokenInput(input_ids_value, tokens, "input_ids");
      std::vector<int32_t> attention_mask = CopyInt64TokenInput(context.GetInput(info->input_indices_[1]), tokens, "attention_mask");
      for (int32_t value : attention_mask) {
        if (value != 0 && value != 1) {
          throw std::runtime_error("attention_mask must contain only 0 or 1 values.");
        }
      }

      tts_style_bert_vits2_float_buffer features{};
      std::string error;
      if (!info->runtime_->EncodeJpBertFeatures(
              input_ids.data(),
              tokens,
              features,
              error)) {
        return CreateStatus(info->ort_api_, ORT_FAIL, error.c_str());
      }
      if (features.hidden_size != 1024) {
        std::ostringstream message;
        message << "TTS.cpp JP-BERT returned hidden_size=" << features.hidden_size << ", expected 1024.";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_FAIL, error_message.c_str());
      }
      const size_t expected_values = tokens * static_cast<size_t>(features.hidden_size);
      if (features.length != expected_values) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp JP-BERT feature length does not match metadata.");
      }
      if (tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp JP-BERT returned too many tokens for ONNX shape.");
      }

      const std::vector<int64_t> output_shape = {
          static_cast<int64_t>(tokens),
          static_cast<int64_t>(features.hidden_size),
      };
      Ort::UnownedValue output = context.GetOutput(info->primary_output_index_, output_shape);
      if (output == nullptr) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML JP-BERT Compute could not allocate output tensor.");
      }
      float* output_data = output.GetTensorMutableData<float>();
      if (features.data == nullptr) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp JP-BERT returned null feature data.");
      }
      std::copy(features.data, features.data + features.length, output_data);

      const auto* node_name = static_cast<const std::string*>(compute_state);
      std::ostringstream trace;
      trace << "encoded jp-bert tokens=" << tokens;
      if (node_name != nullptr && !node_name->empty()) {
        trace << " node='" << *node_name << "'";
      }
      TraceMessage(trace.str());
      return nullptr;
    } catch (const Ort::Exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (const std::bad_alloc&) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Out of memory during Aivis GGML JP-BERT Compute.");
    } catch (const std::exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Unknown error during Aivis GGML JP-BERT Compute.");
    }
  }

  static void ORT_API_CALL ReleaseStateImpl(
      OrtNodeComputeInfo* /*this_ptr*/,
      void* compute_state) noexcept {
    delete static_cast<std::string*>(compute_state);
  }

  const OrtApi& ort_api_;
  TtsCppRuntime* runtime_;
  AivisGgmlGraphKind graph_kind_;
  std::vector<size_t> input_indices_;
  size_t primary_output_index_;
};

struct GraphSignatureGateResult {
  bool supported = true;
  std::vector<std::string> reasons;

  void Reject(std::string reason) {
    supported = false;
    reasons.push_back(std::move(reason));
  }
};

GraphSignatureGateResult MatchStyleBertVits2SynthesisGraph(const OrtGraph* ort_graph) {
  GraphSignatureGateResult result;
  Ort::ConstGraph graph{ort_graph};

  const std::string graph_name = graph.GetName();
  if (!IsAcceptedGraphName(graph_name)) {
    std::ostringstream reason;
    reason << "graph name '" << graph_name << "' is not accepted";
    result.Reject(reason.str());
  }

  const int64_t ir_version = graph.GetOnnxIRVersion();
  if (ir_version != kExpectedIrVersion) {
    std::ostringstream reason;
    reason << "IR version " << ir_version << " != " << kExpectedIrVersion;
    result.Reject(reason.str());
  }

  const std::vector<Ort::OperatorSet> opsets = graph.GetOperatorSets();
  const bool has_default_opset = std::any_of(
      opsets.begin(),
      opsets.end(),
      [](const Ort::OperatorSet& opset) {
        return opset.domain == "" && opset.version == kExpectedOpsetVersion;
      });
  if (!has_default_opset) {
    std::ostringstream reason;
    reason << "default-domain opset " << kExpectedOpsetVersion << " is missing";
    result.Reject(reason.str());
  }

  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  if (inputs.size() != kExpectedInputNames.size()) {
    std::ostringstream reason;
    reason << "input count " << inputs.size() << " != " << kExpectedInputNames.size();
    result.Reject(reason.str());
  }

  for (size_t i = 0; i < kExpectedInputNames.size(); ++i) {
    const auto input = std::find_if(
        inputs.begin(),
        inputs.end(),
        [i](const Ort::ConstValueInfo& value_info) {
          return value_info.GetName() == kExpectedInputNames[i];
        });
    if (input == inputs.end()) {
      std::ostringstream reason;
      reason << "input '" << kExpectedInputNames[i] << "' is missing";
      result.Reject(reason.str());
      continue;
    }

    const ONNXTensorElementDataType elem_type = TensorElementType(*input);
    if (elem_type != kExpectedInputTypes[i]) {
      std::ostringstream reason;
      reason << "input '" << kExpectedInputNames[i] << "' type " << TensorElementTypeName(elem_type)
             << " != " << TensorElementTypeName(kExpectedInputTypes[i]);
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  if (outputs.size() != kExpectedOutputCount) {
    std::ostringstream reason;
    reason << "output count " << outputs.size() << " != " << kExpectedOutputCount;
    result.Reject(reason.str());
  }
  const auto audio_output = std::find_if(
      outputs.begin(),
      outputs.end(),
      [](const Ort::ConstValueInfo& value_info) {
        return value_info.GetName() == kExpectedFirstOutputName;
      });
  if (audio_output == outputs.end()) {
    std::ostringstream reason;
    reason << "output '" << kExpectedFirstOutputName << "' is missing";
    result.Reject(reason.str());
  } else {
    const ONNXTensorElementDataType elem_type = TensorElementType(*audio_output);
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::ostringstream reason;
      reason << "output '" << kExpectedFirstOutputName << "' type "
             << TensorElementTypeName(elem_type) << " != FLOAT";
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> initializers = graph.GetInitializers();
  if (initializers.size() != kExpectedInitializerCount &&
      initializers.size() != kExpectedOptimizedInitializerCount) {
    std::ostringstream reason;
    reason << "initializer count " << initializers.size() << " is not one of "
           << kExpectedInitializerCount << " or " << kExpectedOptimizedInitializerCount;
    result.Reject(reason.str());
  }

  std::unordered_set<std::string> initializer_names;
  initializer_names.reserve(initializers.size());
  for (const Ort::ConstValueInfo& initializer : initializers) {
    initializer_names.insert(initializer.GetName());
  }
  for (const char* expected_name : kRequiredInitializerNames) {
    if (initializer_names.find(expected_name) == initializer_names.end()) {
      std::ostringstream reason;
      reason << "required initializer '" << expected_name << "' is missing";
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != kExpectedNodeCount &&
      nodes.size() != kExpectedOptimizedNodeCount) {
    std::ostringstream reason;
    reason << "node count " << nodes.size() << " is not one of "
           << kExpectedNodeCount << " or " << kExpectedOptimizedNodeCount;
    result.Reject(reason.str());
  }

  std::unordered_set<std::string> op_types;
  op_types.reserve(nodes.size());
  for (const Ort::ConstNode& node : nodes) {
    op_types.insert(node.GetOperatorType());
  }
  for (const char* required_op_type : kRequiredOpTypes) {
    if (op_types.find(required_op_type) == op_types.end()) {
      std::ostringstream reason;
      reason << "required op type '" << required_op_type << "' is missing";
      result.Reject(reason.str());
    }
  }
  if (!nodes.empty() && nodes.back().GetOperatorType() != "Tanh") {
    std::ostringstream reason;
    reason << "last node op '" << nodes.back().GetOperatorType() << "' != 'Tanh'";
    result.Reject(reason.str());
  }

  return result;
}

GraphSignatureGateResult MatchStyleBertVits2JpBertGraph(const OrtGraph* ort_graph) {
  GraphSignatureGateResult result;
  Ort::ConstGraph graph{ort_graph};

  const std::string graph_name = graph.GetName();
  if (!IsAcceptedGraphName(graph_name)) {
    std::ostringstream reason;
    reason << "graph name '" << graph_name << "' is not accepted";
    result.Reject(reason.str());
  }

  const int64_t ir_version = graph.GetOnnxIRVersion();
  if (ir_version != kExpectedIrVersion) {
    std::ostringstream reason;
    reason << "IR version " << ir_version << " != " << kExpectedIrVersion;
    result.Reject(reason.str());
  }

  const std::vector<Ort::OperatorSet> opsets = graph.GetOperatorSets();
  const bool has_default_opset = std::any_of(
      opsets.begin(),
      opsets.end(),
      [](const Ort::OperatorSet& opset) {
        return opset.domain == "" && opset.version == kExpectedJpBertOpsetVersion;
      });
  if (!has_default_opset) {
    std::ostringstream reason;
    reason << "default-domain opset " << kExpectedJpBertOpsetVersion << " is missing";
    result.Reject(reason.str());
  }

  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  if (inputs.size() != kExpectedJpBertInputCount) {
    std::ostringstream reason;
    reason << "input count " << inputs.size() << " != " << kExpectedJpBertInputCount;
    result.Reject(reason.str());
  }

  for (size_t i = 0; i < kExpectedJpBertInputNames.size(); ++i) {
    const auto input = std::find_if(
        inputs.begin(),
        inputs.end(),
        [i](const Ort::ConstValueInfo& value_info) {
          return value_info.GetName() == kExpectedJpBertInputNames[i];
        });
    if (input == inputs.end()) {
      std::ostringstream reason;
      reason << "input '" << kExpectedJpBertInputNames[i] << "' is missing";
      result.Reject(reason.str());
      continue;
    }

    const ONNXTensorElementDataType elem_type = TensorElementType(*input);
    if (elem_type != kExpectedJpBertInputTypes[i]) {
      std::ostringstream reason;
      reason << "input '" << kExpectedJpBertInputNames[i] << "' type " << TensorElementTypeName(elem_type)
             << " != " << TensorElementTypeName(kExpectedJpBertInputTypes[i]);
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  if (outputs.size() != kExpectedJpBertOutputCount) {
    std::ostringstream reason;
    reason << "output count " << outputs.size() << " != " << kExpectedJpBertOutputCount;
    result.Reject(reason.str());
  } else {
    const std::string output_name = outputs[0].GetName();
    if (output_name != kExpectedJpBertOutputName) {
      std::ostringstream reason;
      reason << "output[0] name '" << output_name << "' != '" << kExpectedJpBertOutputName << "'";
      result.Reject(reason.str());
    }
    const ONNXTensorElementDataType elem_type = TensorElementType(outputs[0]);
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::ostringstream reason;
      reason << "output[0] type " << TensorElementTypeName(elem_type) << " != FLOAT";
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> initializers = graph.GetInitializers();
  if (std::find(
          kAcceptedJpBertInitializerCounts.begin(),
          kAcceptedJpBertInitializerCounts.end(),
          initializers.size()) == kAcceptedJpBertInitializerCounts.end()) {
    std::ostringstream reason;
    reason << "initializer count " << initializers.size() << " is not accepted";
    result.Reject(reason.str());
  }

  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (std::find(
          kAcceptedJpBertNodeCounts.begin(),
          kAcceptedJpBertNodeCounts.end(),
          nodes.size()) == kAcceptedJpBertNodeCounts.end()) {
    std::ostringstream reason;
    reason << "node count " << nodes.size() << " is not accepted";
    result.Reject(reason.str());
  }

  std::unordered_set<std::string> op_types;
  op_types.reserve(nodes.size());
  for (const Ort::ConstNode& node : nodes) {
    op_types.insert(node.GetOperatorType());
  }
  for (const char* required_op_type : kRequiredJpBertOpTypes) {
    if (op_types.find(required_op_type) == op_types.end()) {
      std::ostringstream reason;
      reason << "required JP-BERT op type '" << required_op_type << "' is missing";
      result.Reject(reason.str());
    }
  }
  if (!nodes.empty() && nodes.back().GetOperatorType() != "Cast") {
    std::ostringstream reason;
    reason << "last node op '" << nodes.back().GetOperatorType() << "' != 'Cast'";
    result.Reject(reason.str());
  }

  return result;
}

struct AivisGgmlEp final : OrtEp {
  AivisGgmlEp(
      const OrtApi& ort_api,
      const OrtEpApi& ep_api,
      const OrtLogger* logger,
      AivisGgmlEpConfig config,
      std::unique_ptr<TtsCppRuntime> runtime)
      : OrtEp{},
        ort_api_{ort_api},
        ep_api_{ep_api},
        logger_{logger},
        config_{std::move(config)},
        runtime_{std::move(runtime)} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* /*this_ptr*/) noexcept {
    return kEpName;
  }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* this_ptr,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept {
    auto* ep = static_cast<AivisGgmlEp*>(this_ptr);
    if (graph == nullptr) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          "AivisGgmlExecutionProvider received a null graph and claimed no nodes.");
      return nullptr;
    }

    try {
      const GraphSignatureGateResult gate = MatchStyleBertVits2SynthesisGraph(graph);
      if (gate.supported) {
        const std::string runtime_state =
            ep->runtime_ != nullptr ? "runtime_ready=true, " : "runtime_ready=false, ";
        if (ep->config_.claim_synthesis_graph) {
          if (ep->runtime_ == nullptr) {
            LogMessage(
                ep->ort_api_,
                ep->logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 synthesis graph, "
                "but claim_synthesis_graph was ignored because TTS.cpp runtime is not loaded.");
            return nullptr;
          }
          if (graph_support_info == nullptr) {
            return CreateStatus(
                ep->ort_api_,
                ORT_INVALID_ARGUMENT,
                "AivisGgmlExecutionProvider graph_support_info is null.");
          }

          const std::vector<Ort::ConstNode> nodes = Ort::ConstGraph{graph}.GetNodes();
          std::vector<const OrtNode*> raw_nodes;
          raw_nodes.reserve(nodes.size());
          for (const Ort::ConstNode& node : nodes) {
            raw_nodes.push_back(node);
          }

          OrtNodeFusionOptions fusion_options{};
          fusion_options.ort_version_supported = ORT_API_VERSION;
          fusion_options.drop_constant_initializers = true;
          OrtStatus* status = ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
              graph_support_info,
              raw_nodes.data(),
              raw_nodes.size(),
              &fusion_options);
          if (status != nullptr) {
            return status;
          }
          TraceMessage("claimed Style-Bert-VITS2 graph nodes=" + std::to_string(raw_nodes.size()));
          LogMessage(
              ep->ort_api_,
              ep->logger_,
              ORT_LOGGING_LEVEL_INFO,
              "AivisGgmlExecutionProvider claimed the Style-Bert-VITS2 synthesis graph. " +
                  runtime_state + ConfigSummary(ep->config_));
          return nullptr;
        }
        LogMessage(
            ep->ort_api_,
            ep->logger_,
            ORT_LOGGING_LEVEL_INFO,
            "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 synthesis graph, "
            "but graph claiming is disabled until compile/compute is implemented. " +
                runtime_state +
                ConfigSummary(ep->config_));
      } else {
        const GraphSignatureGateResult jp_bert_gate = MatchStyleBertVits2JpBertGraph(graph);
        if (jp_bert_gate.supported) {
          const std::string runtime_state =
              ep->runtime_ != nullptr && ep->runtime_->HasJpBertModel()
                  ? "jp_bert_runtime_ready=true, "
                  : "jp_bert_runtime_ready=false, ";
          if (ep->config_.claim_jp_bert_graph) {
            if (ep->runtime_ == nullptr || !ep->runtime_->HasJpBertModel()) {
              LogMessage(
                  ep->ort_api_,
                  ep->logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 JP-BERT graph, "
                  "but claim_jp_bert_graph was ignored because TTS.cpp JP-BERT runtime is not loaded.");
              return nullptr;
            }
            if (graph_support_info == nullptr) {
              return CreateStatus(
                  ep->ort_api_,
                  ORT_INVALID_ARGUMENT,
                  "AivisGgmlExecutionProvider graph_support_info is null.");
            }

            const std::vector<Ort::ConstNode> nodes = Ort::ConstGraph{graph}.GetNodes();
            std::vector<const OrtNode*> raw_nodes;
            raw_nodes.reserve(nodes.size());
            for (const Ort::ConstNode& node : nodes) {
              raw_nodes.push_back(node);
            }

            OrtNodeFusionOptions fusion_options{};
            fusion_options.ort_version_supported = ORT_API_VERSION;
            fusion_options.drop_constant_initializers = true;
            OrtStatus* status = ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
                graph_support_info,
                raw_nodes.data(),
                raw_nodes.size(),
                &fusion_options);
            if (status != nullptr) {
              return status;
            }
            TraceMessage("claimed Style-Bert-VITS2 JP-BERT graph nodes=" + std::to_string(raw_nodes.size()));
            LogMessage(
                ep->ort_api_,
                ep->logger_,
                ORT_LOGGING_LEVEL_INFO,
                "AivisGgmlExecutionProvider claimed the Style-Bert-VITS2 JP-BERT graph. " +
                    runtime_state + ConfigSummary(ep->config_));
            return nullptr;
          }
          LogMessage(
              ep->ort_api_,
              ep->logger_,
              ORT_LOGGING_LEVEL_INFO,
              "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 JP-BERT graph, "
              "but claim_jp_bert_graph is disabled. " +
                  runtime_state +
                  ConfigSummary(ep->config_));
          return nullptr;
        }
        TraceMessage(
            "rejected graph: synthesis={" + JoinReasons(gate.reasons) +
            "}; jp-bert={" + JoinReasons(jp_bert_gate.reasons) + "}");
        LogMessage(
            ep->ort_api_,
            ep->logger_,
            ORT_LOGGING_LEVEL_VERBOSE,
            "AivisGgmlExecutionProvider rejected graph signatures and claimed no nodes: "
            "synthesis={" + JoinReasons(gate.reasons) + "}; jp-bert={" +
                JoinReasons(jp_bert_gate.reasons) + "}");
      }
    } catch (const Ort::Exception& ex) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          std::string("AivisGgmlExecutionProvider could not inspect graph and claimed no nodes: ") + ex.what());
    } catch (const std::exception& ex) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          std::string("AivisGgmlExecutionProvider graph inspection failed and claimed no nodes: ") + ex.what());
    } catch (...) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          "AivisGgmlExecutionProvider graph inspection failed with an unknown error and claimed no nodes.");
    }
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* this_ptr,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept {
    auto* ep = static_cast<AivisGgmlEp*>(this_ptr);
    for (size_t i = 0; i < count; ++i) {
      if (node_compute_infos != nullptr) {
        node_compute_infos[i] = nullptr;
      }
      if (ep_context_nodes != nullptr) {
        ep_context_nodes[i] = nullptr;
      }
    }
    if (!ep->config_.claim_synthesis_graph && !ep->config_.claim_jp_bert_graph) {
      return CreateStatus(
          ep->ort_api_,
          ORT_EP_FAIL,
          "AivisGgmlExecutionProvider Compile was called while graph claiming is disabled.");
    }
    if (node_compute_infos == nullptr) {
      return CreateStatus(
          ep->ort_api_,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider Compile received a null node_compute_infos array.");
    }

    try {
      for (size_t i = 0; i < count; ++i) {
        (void) fused_nodes;
        if (graphs == nullptr || graphs[i] == nullptr) {
          throw std::runtime_error("AivisGgmlExecutionProvider Compile received a null graph.");
        }
        AivisGgmlGraphKind graph_kind = AivisGgmlGraphKind::Synthesis;
        const GraphSignatureGateResult synthesis_gate = MatchStyleBertVits2SynthesisGraph(graphs[i]);
        const GraphSignatureGateResult jp_bert_gate = MatchStyleBertVits2JpBertGraph(graphs[i]);
        std::vector<size_t> input_indices;
        size_t primary_output_index = 0;
        if (synthesis_gate.supported) {
          if (!ep->config_.claim_synthesis_graph) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile received a synthesis graph while claim_synthesis_graph is disabled.");
          }
          if (ep->runtime_ == nullptr || !ep->runtime_->HasSynthesisModel()) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile requires a loaded synthesis runtime.");
          }
          graph_kind = AivisGgmlGraphKind::Synthesis;
          input_indices = BuildInputIndices(graphs[i], kExpectedInputNames);
          primary_output_index = BuildOutputIndex(graphs[i], kExpectedFirstOutputName);
        } else if (jp_bert_gate.supported) {
          if (!ep->config_.claim_jp_bert_graph) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile received a JP-BERT graph while claim_jp_bert_graph is disabled.");
          }
          if (ep->runtime_ == nullptr || !ep->runtime_->HasJpBertModel()) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile requires a loaded JP-BERT runtime.");
          }
          graph_kind = AivisGgmlGraphKind::JpBert;
          input_indices = BuildInputIndices(graphs[i], kExpectedJpBertInputNames);
          primary_output_index = BuildOutputIndex(graphs[i], kExpectedJpBertOutputName);
        } else {
          throw std::runtime_error(
              "AivisGgmlExecutionProvider Compile received an unsupported graph signature: synthesis={" +
              JoinReasons(synthesis_gate.reasons) + "}; jp-bert={" + JoinReasons(jp_bert_gate.reasons) + "}");
        }
        node_compute_infos[i] = new AivisGgmlNodeComputeInfo(
            ep->ort_api_,
            ep->runtime_.get(),
            graph_kind,
            std::move(input_indices),
            primary_output_index);
      }
      TraceMessage("compiled fused graph count=" + std::to_string(count));
      return nullptr;
    } catch (const std::bad_alloc&) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, "Out of memory compiling Aivis GGML graph.");
    } catch (const std::exception& ex) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, "Unknown error compiling Aivis GGML graph.");
    }
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* /*this_ptr*/,
      OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos) noexcept {
    if (node_compute_infos == nullptr) {
      return;
    }
    for (size_t i = 0; i < num_node_compute_infos; ++i) {
      delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
      node_compute_infos[i] = nullptr;
    }
  }

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger* logger_;
  AivisGgmlEpConfig config_;
  std::unique_ptr<TtsCppRuntime> runtime_;
};

struct AivisGgmlEpFactory final : OrtEpFactory {
  AivisGgmlEpFactory(
      const OrtApi& ort_api,
      const OrtEpApi& ep_api,
      const OrtLogger* default_logger,
      std::string registration_name)
      : OrtEpFactory{},
        ort_api_{ort_api},
        ep_api_{ep_api},
        default_logger_{default_logger},
        registration_name_{std::move(registration_name)} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
    GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;
    CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;
    GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
    GetCustomOpDomains = GetCustomOpDomainsImpl;
    InitGraphicsInterop = InitGraphicsInteropImpl;
    DeinitGraphicsInterop = DeinitGraphicsInteropImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kEpName;
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kVendor;
  }

  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return 0;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kVersion;
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      OrtEpDevice** ep_devices,
      size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (p_num_ep_devices == nullptr) {
      return CreateStatus(factory->ort_api_, ORT_INVALID_ARGUMENT, "num_ep_devices output is null.");
    }

    *p_num_ep_devices = 0;
    if (ep_devices == nullptr || max_ep_devices == 0) {
      return nullptr;
    }

    for (size_t i = 0; i < num_devices; ++i) {
      const OrtHardwareDevice* device = devices[i];
      if (device == nullptr ||
          factory->ort_api_.HardwareDevice_Type(device) != OrtHardwareDeviceType_CPU) {
        continue;
      }

      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api_.CreateKeyValuePairs(&ep_options);

      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.stage", kStage);
      factory->ort_api_.AddKeyValuePair(ep_metadata, "registration_name", factory->registration_name_.c_str());
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.supported_backends", "vulkan,metal,cpu");
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.supported_precision", "accurate,fast");
      factory->ort_api_.AddKeyValuePair(ep_options, "backend", kDefaultBackend);
      factory->ort_api_.AddKeyValuePair(ep_options, "precision", kDefaultPrecision);
      factory->ort_api_.AddKeyValuePair(ep_options, "eager_load_model", "0");
      factory->ort_api_.AddKeyValuePair(ep_options, "claim_synthesis_graph", "0");
      factory->ort_api_.AddKeyValuePair(ep_options, "claim_jp_bert_graph", "0");
      factory->ort_api_.AddKeyValuePair(ep_options, "n_threads", "0");

      OrtEpDevice* ep_device = nullptr;
      OrtStatus* status = factory->ep_api_.CreateEpDevice(
          this_ptr,
          device,
          ep_metadata,
          ep_options,
          &ep_device);

      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api_.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      ep_devices[0] = ep_device;
      *p_num_ep_devices = 1;
      return nullptr;
    }

    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* /*devices*/,
      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
      size_t num_devices,
      const OrtSessionOptions* session_options,
      const OrtLogger* logger,
      OrtEp** ep) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (ep == nullptr) {
      return CreateStatus(factory->ort_api_, ORT_INVALID_ARGUMENT, "EP output is null.");
    }

    *ep = nullptr;
    if (num_devices != 1) {
      return CreateStatus(
          factory->ort_api_,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider bootstrap EP expects exactly one device.");
    }

    try {
      const OrtLogger* ep_logger = logger != nullptr ? logger : factory->default_logger_;
      AivisGgmlEpConfig config = ReadEpConfig(session_options);
      OrtStatus* status = ValidateEpConfig(factory->ort_api_, config);
      if (status != nullptr) {
        return status;
      }
      std::unique_ptr<TtsCppRuntime> runtime;
      if (config.eager_load_model) {
        runtime = TtsCppRuntime::LoadAndMaybeOpenModel(config);
        TraceMessage("eager-loaded TTS.cpp GGUF model(s)");
        LogMessage(
            factory->ort_api_,
            ep_logger,
            ORT_LOGGING_LEVEL_INFO,
            "AivisGgmlExecutionProvider eagerly loaded configured TTS.cpp GGUF model(s).");
      }
      LogMessage(
          factory->ort_api_,
          ep_logger,
          ORT_LOGGING_LEVEL_INFO,
          "Creating AivisGgmlExecutionProvider with " + ConfigSummary(config) + ".");
      TraceMessage("CreateEp " + ConfigSummary(config));
      *ep = new AivisGgmlEp(
          factory->ort_api_,
          factory->ep_api_,
          ep_logger,
          std::move(config),
          std::move(runtime));
      return nullptr;
    } catch (const std::bad_alloc&) {
      return CreateStatus(factory->ort_api_, ORT_FAIL, "Out of memory creating Aivis GGML EP.");
    } catch (const std::exception& ex) {
      return CreateStatus(factory->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(factory->ort_api_, ORT_FAIL, "Unknown error creating Aivis GGML EP.");
    }
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    delete static_cast<AivisGgmlEp*>(ep);
  }

  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* /*devices*/,
      size_t /*num_devices*/,
      const char* /*compatibility_info*/,
      OrtCompiledModelCompatibility* model_compatibility) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (model_compatibility == nullptr) {
      return CreateStatus(
          factory->ort_api_,
          ORT_INVALID_ARGUMENT,
          "Compiled model compatibility output is null.");
    }

    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEpFactory* /*this_ptr*/,
      const OrtMemoryInfo* /*memory_info*/,
      const OrtKeyValuePairs* /*allocator_options*/,
      OrtAllocator** allocator) noexcept {
    if (allocator != nullptr) {
      *allocator = nullptr;
    }
    return nullptr;
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* /*this_ptr*/,
      OrtAllocator* /*allocator*/) noexcept {}

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* /*this_ptr*/,
      OrtDataTransferImpl** data_transfer) noexcept {
    if (data_transfer != nullptr) {
      *data_transfer = nullptr;
    }
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryDevice* /*memory_device*/,
      const OrtKeyValuePairs* /*stream_options*/,
      OrtSyncStreamImpl** stream) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (stream != nullptr) {
      *stream = nullptr;
    }
    return CreateStatus(factory->ort_api_, ORT_NOT_IMPLEMENTED, "Aivis GGML EP has no stream support yet.");
  }

  static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
      OrtEpFactory* /*this_ptr*/,
      const OrtHardwareDevice* /*hw*/,
      OrtDeviceEpIncompatibilityDetails* /*details*/) noexcept {
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* /*ep_device*/,
      OrtExternalResourceImporterImpl** out_importer) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (out_importer != nullptr) {
      *out_importer = nullptr;
    }
    return CreateStatus(
        factory->ort_api_,
        ORT_NOT_IMPLEMENTED,
        "Aivis GGML EP has no external resource importer yet.");
  }

  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(
      OrtEpFactory* this_ptr,
      size_t* num_domains) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (num_domains == nullptr) {
      return CreateStatus(factory->ort_api_, ORT_INVALID_ARGUMENT, "num_domains output is null.");
    }
    *num_domains = 0;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(
      OrtEpFactory* this_ptr,
      OrtCustomOpDomain** /*domains*/,
      size_t num_domains) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (num_domains != 0) {
      return CreateStatus(
          factory->ort_api_,
          ORT_INVALID_ARGUMENT,
          "Aivis GGML EP bootstrap has no custom op domains.");
    }
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL InitGraphicsInteropImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* /*ep_device*/,
      const OrtGraphicsInteropConfig* /*config*/) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    return CreateStatus(
        factory->ort_api_,
        ORT_NOT_IMPLEMENTED,
        "Aivis GGML EP has no graphics interop support yet.");
  }

  static OrtStatus* ORT_API_CALL DeinitGraphicsInteropImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* /*ep_device*/) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    return CreateStatus(
        factory->ort_api_,
        ORT_NOT_IMPLEMENTED,
        "Aivis GGML EP has no graphics interop support yet.");
  }

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger* default_logger_;
  std::string registration_name_;
};

}  // namespace

extern "C" {

AIVIS_GGML_EP_EXPORT OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base != nullptr ? ort_api_base->GetApi(ORT_API_VERSION) : nullptr;
  if (num_factories != nullptr) {
    *num_factories = 0;
  }
  if (ort_api == nullptr) {
    return nullptr;
  }
  if (factories == nullptr || num_factories == nullptr) {
    return CreateStatus(*ort_api, ORT_INVALID_ARGUMENT, "Factory output is null.");
  }
  if (max_factories < 1) {
    return CreateStatus(*ort_api, ORT_INVALID_ARGUMENT, "Not enough space to return EP factory.");
  }

  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (ep_api == nullptr) {
    return CreateStatus(*ort_api, ORT_FAIL, "ONNX Runtime did not provide OrtEpApi.");
  }
  Ort::InitApi(ort_api);

  try {
    auto factory = std::make_unique<AivisGgmlEpFactory>(
        *ort_api,
        *ep_api,
        default_logger,
        registration_name != nullptr ? registration_name : "");
    factories[0] = factory.release();
    *num_factories = 1;
    return nullptr;
  } catch (const std::bad_alloc&) {
    return CreateStatus(*ort_api, ORT_FAIL, "Out of memory creating Aivis GGML EP factory.");
  } catch (const std::exception& ex) {
    return CreateStatus(*ort_api, ORT_FAIL, ex.what());
  } catch (...) {
    return CreateStatus(*ort_api, ORT_FAIL, "Unknown error creating Aivis GGML EP factory.");
  }
}

AIVIS_GGML_EP_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<AivisGgmlEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
