#include "shared_utils.hpp"

#include <cstdlib>

namespace sync_app {

std::string streamTypeToString(StreamType t) {
    switch(t) {
    case StreamType::Color:
        return "RGB";
    case StreamType::Depth:
        return "DEPTH";
    case StreamType::IR:
        return "IR";
    case StreamType::PointCloud:
        return "POINT_CLOUD";
    }
    return "UNKNOWN";
}

std::optional<StreamType> streamTypeFromString(const std::string &s) {
    if(s == "RGB" || s == "COLOR" || s == "Color" || s == "color") {
        return StreamType::Color;
    }
    if(s == "DEPTH" || s == "Depth" || s == "depth") {
        return StreamType::Depth;
    }
    if(s == "IR" || s == "Infrared" || s == "infrared" || s == "ir") {
        return StreamType::IR;
    }
    if(s == "POINT_CLOUD" || s == "PointCloud" || s == "point_cloud" || s == "pointcloud") {
        return StreamType::PointCloud;
    }
    return std::nullopt;
}

OBMultiDeviceSyncMode stringToOBSyncMode(const std::string &modeString) {
    static const std::unordered_map<std::string, OBMultiDeviceSyncMode> syncModeMap = {
        { "OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN", OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN },
        { "OB_MULTI_DEVICE_SYNC_MODE_STANDALONE", OB_MULTI_DEVICE_SYNC_MODE_STANDALONE },
        { "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY", OB_MULTI_DEVICE_SYNC_MODE_PRIMARY },
        { "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY", OB_MULTI_DEVICE_SYNC_MODE_SECONDARY },
        { "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED", OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED },
        { "OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING", OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING },
        { "OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING", OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING }
    };
    auto it = syncModeMap.find(modeString);
    if(it != syncModeMap.end()) {
        return it->second;
    }
    return OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN;
}

static std::string obSyncModeToString(OBMultiDeviceSyncMode mode) {
    switch(mode) {
    case OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN:
        return "OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN";
    case OB_MULTI_DEVICE_SYNC_MODE_STANDALONE:
        return "OB_MULTI_DEVICE_SYNC_MODE_STANDALONE";
    case OB_MULTI_DEVICE_SYNC_MODE_PRIMARY:
        return "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY";
    case OB_MULTI_DEVICE_SYNC_MODE_SECONDARY:
        return "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY";
    case OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED:
        return "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED";
    case OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING:
        return "OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING";
    case OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING:
        return "OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING";
    default:
        return "UNKNOWN_SYNC_MODE";
    }
}

OBFormat stringToOBFormat(const std::string &formatString, StreamType type) {
    if(type == StreamType::Color) {
        if(formatString == "RGB") {
            return OB_FORMAT_RGB;
        }
        if(formatString == "BGR") {
            return OB_FORMAT_BGR;
        }
        if(formatString == "MJPG") {
            return OB_FORMAT_MJPG;
        }
        if(formatString == "YUYV") {
            return OB_FORMAT_YUYV;
        }
        return OB_FORMAT_RGB;
    }
    if(formatString == "Y16") {
        return OB_FORMAT_Y16;
    }
    if(formatString == "Y14") {
        return OB_FORMAT_Y14;
    }
    if(formatString == "Z16") {
        return OB_FORMAT_Z16;
    }
    if(formatString == "RLE") {
        return OB_FORMAT_RLE;
    }
    return OB_FORMAT_Y16;
}

std::string normalizeMode(std::string s) {
    size_t begin = 0;
    while(begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        begin++;
    }
    size_t end = s.size();
    while(end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    s = s.substr(begin, end - begin);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string trimString(std::string s) {
    size_t begin = 0;
    while(begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        begin++;
    }
    size_t end = s.size();
    while(end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    return s.substr(begin, end - begin);
}

static std::string shellQuote(const std::string &s) {
    std::string out = "'";
    for(char c: s) {
        if(c == '\'') {
            out += "'\\''";
        }
        else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static std::optional<fs::path> findRepoRootContaining(const fs::path &requiredPath) {
    std::vector<fs::path> seeds;
    try {
        seeds.push_back(fs::current_path());
    }
    catch(...) {
    }
    seeds.push_back(fs::path("."));
    seeds.push_back(fs::path(".."));
    seeds.push_back(fs::path("../.."));

    for(auto seed: seeds) {
        try {
            seed = fs::absolute(seed);
        }
        catch(...) {
        }
        for(fs::path cur = seed; !cur.empty(); cur = cur.parent_path()) {
            if(fs::exists(cur / requiredPath)) {
                return cur;
            }
            if(cur == cur.root_path()) {
                break;
            }
        }
    }
    return std::nullopt;
}

static std::optional<fs::path> findManualLabelRepoRoot() {
    return findRepoRootContaining(fs::path("label") / "main.py");
}

static std::optional<fs::path> findQcRepoRoot() {
    return findRepoRootContaining(fs::path("src") / "qc" / "main.py");
}

static fs::path defaultTempPath(const std::string &filename) {
    try {
        return fs::temp_directory_path() / filename;
    }
    catch(...) {
        return fs::path(filename);
    }
}

static bool ensureParentDirectory(const fs::path &path, std::string *errorMessage) {
    const fs::path parent = path.parent_path();
    if(parent.empty()) {
        return true;
    }
    try {
        fs::create_directories(parent);
    }
    catch(const std::exception &e) {
        if(errorMessage) {
            *errorMessage = "failed to create directory " + parent.string() + ": " + e.what();
        }
        return false;
    }
    return true;
}

static std::string normalizedNasPrefix(std::string value) {
    value = trimString(std::move(value));
    while(value.size() > 1 && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

static void addNasMountsConfig(cJSON *root, const NasConfig &nas) {
    cJSON *mounts = cJSON_CreateObject();
    if(nas.enabled) {
        const std::string prefix = normalizedNasPrefix(nas.uriPrefix);
        const std::string mountPath = trimString(nas.mountPath.string());
        if(!prefix.empty() && !mountPath.empty()) {
            cJSON_AddStringToObject(mounts, prefix.c_str(), mountPath.c_str());
        }
    }
    cJSON_AddItemToObject(root, "nas_mounts", mounts);
}

static bool writeJsonConfigFile(cJSON *root, const fs::path &path, std::string *errorMessage) {
    if(!ensureParentDirectory(path, errorMessage)) {
        return false;
    }
    char *printed = cJSON_Print(root);
    if(!printed) {
        if(errorMessage) {
            *errorMessage = "failed to serialize launch config";
        }
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if(!out) {
        if(errorMessage) {
            *errorMessage = "failed to open launch config for writing: " + path.string();
        }
        cJSON_free(printed);
        return false;
    }
    out << printed << "\n";
    const bool ok = static_cast<bool>(out);
    cJSON_free(printed);
    if(!ok) {
        if(errorMessage) {
            *errorMessage = "failed to write launch config: " + path.string();
        }
        return false;
    }
    return true;
}

bool launchManualLabelFrontend(const AppConfig &cfg,
                               const std::string &operatorHint,
                               std::string *errorMessage) {
    const auto repoRoot = findManualLabelRepoRoot();
    if(!repoRoot.has_value()) {
        if(errorMessage) {
            *errorMessage = "label/main.py not found from current working directory";
        }
        return false;
    }

    const auto &frontend = cfg.frontends.label;
    std::string python = trimString(frontend.pythonExecutable);
    if(python.empty()) {
        python = "python3";
    }
    std::string operatorId = trimString(frontend.operatorId);
    if(operatorId.empty()) {
        operatorId = trimString(operatorHint);
    }
    if(operatorId.empty()) {
        operatorId = "labeler_01";
    }

    const fs::path logPath = frontend.logPath.empty() ? defaultTempPath("orbbec_manual_label.log") : frontend.logPath;
    if(!ensureParentDirectory(logPath, errorMessage)) {
        return false;
    }

    const fs::path launchConfigPath = defaultTempPath("orbbec_manual_label_config.json");
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "backend_url", trimString(cfg.taskBackend.baseUrl).c_str());
    cJSON_AddStringToObject(root, "operator_id", operatorId.c_str());
    if(!frontend.frameCacheDir.empty()) {
        cJSON_AddStringToObject(root, "frame_cache_dir", frontend.frameCacheDir.string().c_str());
    }
    std::string ffmpegExecutable = trimString(frontend.ffmpegExecutable);
    if(ffmpegExecutable.empty()) {
        ffmpegExecutable = "ffmpeg";
    }
    cJSON_AddStringToObject(root, "ffmpeg_executable", ffmpegExecutable.c_str());
    cJSON_AddNumberToObject(root, "lease_seconds", std::max(1, frontend.leaseSeconds));
    cJSON_AddNumberToObject(root, "request_timeout_seconds", std::max(1.0, frontend.requestTimeoutSeconds));
    addNasMountsConfig(root, cfg.taskBackend.nas);
    const bool wroteConfig = writeJsonConfigFile(root, launchConfigPath, errorMessage);
    cJSON_Delete(root);
    if(!wroteConfig) {
        return false;
    }

    std::ostringstream cmd;
    cmd << "cd " << shellQuote(repoRoot->string())
        << " && nohup " << shellQuote(python)
        << " -m label.main --config " << shellQuote(launchConfigPath.string())
        << " >> " << shellQuote(logPath.string())
        << " 2>&1 &";

    const int rc = std::system(cmd.str().c_str());
    if(rc != 0) {
        if(errorMessage) {
            *errorMessage = "failed to start manual label frontend; log: " + logPath.string();
        }
        return false;
    }
    if(errorMessage) {
        *errorMessage = logPath.string();
    }
    return true;
}

bool launchQcFrontend(const AppConfig &cfg,
                      const std::string &operatorHint,
                      std::string *errorMessage) {
    const auto repoRoot = findQcRepoRoot();
    if(!repoRoot.has_value()) {
        if(errorMessage) {
            *errorMessage = "src/qc/main.py not found from current working directory";
        }
        return false;
    }

    const auto &frontend = cfg.frontends.qc;
    std::string python = trimString(frontend.pythonExecutable);
    if(python.empty()) {
        python = "python3";
    }
    std::string operatorId = trimString(frontend.operatorId);
    if(operatorId.empty()) {
        operatorId = trimString(operatorHint);
    }
    if(operatorId.empty()) {
        operatorId = "qc_operator_01";
    }

    const fs::path logPath = frontend.logPath.empty() ? defaultTempPath("orbbec_qc_frontend.log") : frontend.logPath;
    if(!ensureParentDirectory(logPath, errorMessage)) {
        return false;
    }

    std::string workerMachineId = trimString(frontend.workerMachineId);
    if(workerMachineId.empty()) {
        workerMachineId = "qc_" + operatorId;
    }

    const fs::path launchConfigPath = defaultTempPath("orbbec_qc_frontend_config.json");
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "backend_url", trimString(cfg.taskBackend.baseUrl).c_str());
    cJSON_AddStringToObject(root, "operator_id", operatorId.c_str());
    cJSON_AddStringToObject(root, "worker_machine_id", workerMachineId.c_str());
    cJSON_AddNumberToObject(root, "sample_interval", std::max(1, frontend.sampleInterval));
    cJSON_AddNumberToObject(root, "default_lease_minutes", std::max(1, frontend.defaultLeaseMinutes));
    cJSON_AddNumberToObject(root, "crash_lease_extension_minutes", std::max(1, frontend.crashLeaseExtensionMinutes));
    cJSON_AddStringToObject(root, "tmp_dir", frontend.tmpDir.string().c_str());
    cJSON_AddStringToObject(root, "state_dir", frontend.stateDir.string().c_str());
    cJSON_AddNumberToObject(root, "range_merge_gap_frames", std::max(0, frontend.rangeMergeGapFrames));
    cJSON_AddNumberToObject(root, "request_timeout_seconds", std::max(1.0, frontend.requestTimeoutSeconds));
    addNasMountsConfig(root, cfg.taskBackend.nas);
    const bool wroteConfig = writeJsonConfigFile(root, launchConfigPath, errorMessage);
    cJSON_Delete(root);
    if(!wroteConfig) {
        return false;
    }

    std::ostringstream cmd;
    cmd << "cd " << shellQuote(repoRoot->string())
        << " && nohup " << shellQuote(python)
        << " -m src.qc.main --config " << shellQuote(launchConfigPath.string())
        << " >> " << shellQuote(logPath.string())
        << " 2>&1 &";

    const int rc = std::system(cmd.str().c_str());
    if(rc != 0) {
        if(errorMessage) {
            *errorMessage = "failed to start QC frontend; log: " + logPath.string();
        }
        return false;
    }
    if(errorMessage) {
        *errorMessage = logPath.string();
    }
    return true;
}

std::string normalizePresetKey(std::string s) {
    s = trimString(std::move(s));
    std::string out;
    out.reserve(s.size());
    for(unsigned char c: s) {
        if(std::isspace(c) || c == '_' || c == '-') {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

bool isViewerMode(const AppConfig &cfg) {
    return cfg.mode == "viewer" || cfg.mode == "interaction";
}

bool isInteractionMode(const AppConfig &cfg) {
    return cfg.mode == "interaction";
}

static std::optional<FisheyeImageFormat> fisheyeImageFormatFromString(const std::string &s) {
    if(s == "jpg" || s == "jpeg" || s == "JPG" || s == "JPEG") {
        return FisheyeImageFormat::Jpeg;
    }
    if(s == "png" || s == "PNG") {
        return FisheyeImageFormat::Png;
    }
    return std::nullopt;
}

static std::string normalizeDepthEncodingConfig(std::string s) {
    s = normalizePresetKey(std::move(s));
    if(s == "ffv1mkv" || s == "ffv1" || s == "mkv") {
        return "ffv1_mkv";
    }
    return "png";
}

static std::string normalizeExtrinsicHealthRotationMethod(std::string s) {
    s = normalizePresetKey(std::move(s));
    if(s == "depthplane" || s == "depth" || s == "plane" || s == "depthfit" || s == "depthsampling") {
        return "depth_plane";
    }
    return "pnp";
}

static std::string readFileAll(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path.string());
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

static std::optional<std::string> getString(cJSON *obj, const char *key) {
    auto *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(item && cJSON_IsString(item) && item->valuestring) {
        return std::string(item->valuestring);
    }
    return std::nullopt;
}

static std::optional<int> getInt(cJSON *obj, const char *key) {
    auto *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(item && cJSON_IsNumber(item)) {
        return item->valueint;
    }
    return std::nullopt;
}

static std::optional<double> getDouble(cJSON *obj, const char *key) {
    auto *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(item && cJSON_IsNumber(item)) {
        return item->valuedouble;
    }
    return std::nullopt;
}

static std::optional<bool> getBool(cJSON *obj, const char *key) {
    auto *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(item && cJSON_IsBool(item)) {
        return static_cast<bool>(item->valueint);
    }
    return std::nullopt;
}

AppConfig loadConfig(const fs::path &configPath) {
    auto content = readFileAll(configPath);
    cJSON *root  = cJSON_Parse(content.c_str());
    if(!root) {
        throw std::runtime_error("Invalid JSON in config file: " + configPath.string());
    }

    AppConfig cfg;
    const fs::path configBase = fs::absolute(configPath).parent_path();
    auto resolveConfigRelativePath = [&](const std::string &value) {
        fs::path p(value);
        if(p.is_relative()) {
            p = (configBase / p).lexically_normal();
        }
        return p;
    };

    if(auto v = getString(root, "outputDir")) {
        cfg.outputDir = fs::path(*v);
    }
    else {
        cfg.outputDir = fs::path("./collected_data");
    }

    if(auto v = getInt(root, "durationSec")) {
        cfg.durationSec = *v;
    }
    if(auto v = getInt(root, "maxFrames")) {
        cfg.maxFrames = *v;
    }
    if(auto v = getInt(root, "collectFps")) {
        cfg.collectFps = *v;
    }
    if(auto v = getString(root, "mode")) {
        cfg.mode = normalizeMode(*v);
    }
    if(auto v = getInt(root, "viewer_fps")) {
        cfg.viewerFps = *v;
    }
    if(auto v = getString(root, "init_extrinsic_path")) {
        fs::path p = resolveConfigRelativePath(*v);
        cfg.initExtrinsicPath = p.string();
    }
    if(auto v = getDouble(root, "max_depth")) {
        if(*v > 0.0) {
            cfg.maxDepth = static_cast<float>(*v);
        }
    }
    if(auto v = getBool(root, "different_color")) {
        cfg.differentColor = *v;
    }
    if(auto v = getBool(root, "colorful_cloud_points")) {
        cfg.colorfulCloudPoints = *v;
    }

    if(auto *filtersObj = cJSON_GetObjectItemCaseSensitive(root, "filters")) {
        if(cJSON_IsObject(filtersObj)) {
            if(auto v = getString(filtersObj, "preset")) {
                cfg.filters.preset = trimString(*v);
            }
            if(auto v = getInt(filtersObj, "point_cloud_decimation_factor")) {
                cfg.filters.pointCloudDecimationFactor = std::max(0, *v);
            }
            if(auto v = getInt(filtersObj, "decimation_filter_scale")) {
                cfg.filters.decimationFilterScale = std::max(0, *v);
            }
            if(auto v = getInt(filtersObj, "noise_removal_filter_max_size")) {
                cfg.filters.noiseRemovalMaxSize = std::max(0, *v);
            }
            if(auto v = getInt(filtersObj, "noise_removal_filter_min_diff")) {
                cfg.filters.noiseRemovalMinDiff = std::max(0, *v);
            }
            if(auto v = getDouble(filtersObj, "spatial_filter_alpha")) {
                cfg.filters.spatialAlpha = std::max(0.0, *v);
            }
            if(auto v = getInt(filtersObj, "spatial_filter_disp_diff")) {
                cfg.filters.spatialDispDiff = std::max(0, *v);
            }
            if(auto v = getInt(filtersObj, "spatial_filter_magnitude")) {
                cfg.filters.spatialMagnitude = std::max(0, *v);
            }
            if(auto v = getInt(filtersObj, "spatial_filter_radius")) {
                cfg.filters.spatialRadius = std::max(0, *v);
            }
            if(auto v = getDouble(filtersObj, "smooth_threshold")) {
                cfg.filters.smoothThresholdM = std::max(0.0, *v);
            }
            if(auto v = getDouble(filtersObj, "temporal_filter_diff_scale")) {
                cfg.filters.temporalDiffScale = std::max(0.0, *v);
            }
            if(auto v = getDouble(filtersObj, "temporal_filter_weight")) {
                cfg.filters.temporalWeight = std::max(0.0, *v);
            }
            if(auto v = getInt(filtersObj, "hole_filling_filter_mode")) {
                cfg.filters.holeFillingMode = std::max(0, *v);
            }
            if(auto v = getDouble(filtersObj, "conf_threshold")) {
                cfg.filters.confThreshold = std::min(1.0, std::max(0.0, *v));
            }
            if(auto v = getBool(filtersObj, "desk_crop")) {
                cfg.filters.deskCrop = *v;
            }
        }
    }

    if(auto *calibObj = cJSON_GetObjectItemCaseSensitive(root, "calibration")) {
        if(cJSON_IsObject(calibObj)) {
            if(auto v = getInt(calibObj, "samplesPerPair")) {
                cfg.calibration.samplesPerPair = std::max(3, *v);
            }
            else if(auto v = getInt(calibObj, "samples_per_pair")) {
                cfg.calibration.samplesPerPair = std::max(3, *v);
            }
            if(auto *cbObj = cJSON_GetObjectItemCaseSensitive(calibObj, "chessboard")) {
                if(cJSON_IsObject(cbObj)) {
                    if(auto v = getInt(cbObj, "cols")) {
                        cfg.calibration.chessboard.cols = std::max(2, *v);
                    }
                    if(auto v = getInt(cbObj, "rows")) {
                        cfg.calibration.chessboard.rows = std::max(2, *v);
                    }
                    if(auto v = getDouble(cbObj, "squareSize")) {
                        if(*v > 0.0) {
                            cfg.calibration.chessboard.squareSize = static_cast<float>(*v);
                        }
                    }
                    if(auto v = getInt(cbObj, "samplesPerPair")) {
                        cfg.calibration.samplesPerPair = std::max(3, *v);
                    }
                    else if(auto v = getInt(cbObj, "samples_per_pair")) {
                        cfg.calibration.samplesPerPair = std::max(3, *v);
                    }
                }
            }
        }
    }

    cfg.extrinsicHealth.scriptPath = (configBase / "extrinsic_health_check.py").lexically_normal();
    if(auto *healthObj = cJSON_GetObjectItemCaseSensitive(root, "extrinsicHealthCheck")) {
        if(cJSON_IsObject(healthObj)) {
            if(auto v = getBool(healthObj, "enabled")) {
                cfg.extrinsicHealth.enabled = *v;
            }
            if(auto v = getString(healthObj, "pythonExecutable")) {
                cfg.extrinsicHealth.pythonExecutable = trimString(*v);
            }
            else if(auto v = getString(healthObj, "python")) {
                cfg.extrinsicHealth.pythonExecutable = trimString(*v);
            }
            if(cfg.extrinsicHealth.pythonExecutable.empty()) {
                cfg.extrinsicHealth.pythonExecutable = "python3";
            }
            if(auto v = getString(healthObj, "scriptPath")) {
                cfg.extrinsicHealth.scriptPath = resolveConfigRelativePath(*v);
            }
            else if(auto v = getString(healthObj, "script")) {
                cfg.extrinsicHealth.scriptPath = resolveConfigRelativePath(*v);
            }
            if(auto v = getString(healthObj, "tagFamily")) {
                cfg.extrinsicHealth.tagFamily = trimString(*v);
            }
            if(auto v = getDouble(healthObj, "tagSizeM")) {
                cfg.extrinsicHealth.tagSizeM = std::max(0.001, *v);
            }
            if(auto v = getString(healthObj, "rotationMethod")) {
                cfg.extrinsicHealth.rotationMethod = normalizeExtrinsicHealthRotationMethod(*v);
            }
            else if(auto v = getString(healthObj, "rotation_method")) {
                cfg.extrinsicHealth.rotationMethod = normalizeExtrinsicHealthRotationMethod(*v);
            }
            else if(auto v = getString(healthObj, "planeAngleMethod")) {
                cfg.extrinsicHealth.rotationMethod = normalizeExtrinsicHealthRotationMethod(*v);
            }
            if(auto v = getInt(healthObj, "sampleCount")) {
                cfg.extrinsicHealth.sampleCount = std::max(1, std::min(30, *v));
            }
            if(auto v = getInt(healthObj, "sampleIntervalMs")) {
                cfg.extrinsicHealth.sampleIntervalMs = std::max(0, std::min(5000, *v));
            }
            if(auto v = getInt(healthObj, "maxSnapshotAgeMs")) {
                cfg.extrinsicHealth.maxSnapshotAgeMs = std::max(100, *v);
            }
            if(auto v = getInt(healthObj, "jpegQuality")) {
                cfg.extrinsicHealth.jpegQuality = std::max(1, std::min(100, *v));
            }
            if(auto v = getBool(healthObj, "keepDebugSnapshots")) {
                cfg.extrinsicHealth.keepDebugSnapshots = *v;
            }
            if(auto v = getBool(healthObj, "blockOnInconclusive")) {
                cfg.extrinsicHealth.blockOnInconclusive = *v;
            }
            if(auto v = getBool(healthObj, "blockOnWarn")) {
                cfg.extrinsicHealth.blockOnWarn = *v;
            }
            if(auto v = getBool(healthObj, "requireAllCameras")) {
                cfg.extrinsicHealth.requireAllCameras = *v;
            }
            if(auto v = getInt(healthObj, "minSharedCamerasPerTag")) {
                cfg.extrinsicHealth.minSharedCamerasPerTag = std::max(2, *v);
            }
            if(auto v = getInt(healthObj, "minTagInlierObservations")) {
                cfg.extrinsicHealth.minTagInlierObservations = std::max(2, *v);
            }
            if(auto v = getInt(healthObj, "minCheckedCameras")) {
                cfg.extrinsicHealth.minCheckedCameras = std::max(2, *v);
            }
            if(auto v = getInt(healthObj, "minTagsPerCamera")) {
                cfg.extrinsicHealth.minTagsPerCamera = std::max(1, *v);
            }
            if(auto v = getInt(healthObj, "minPassingSnapshots")) {
                cfg.extrinsicHealth.minPassingSnapshots = std::max(1, *v);
            }
            if(auto v = getInt(healthObj, "minFailingSnapshots")) {
                cfg.extrinsicHealth.minFailingSnapshots = std::max(1, *v);
            }
            if(auto v = getDouble(healthObj, "singleTagReprojLimitPx")) {
                cfg.extrinsicHealth.singleTagReprojLimitPx = std::max(0.1, *v);
            }
            if(auto v = getDouble(healthObj, "fusionTransThreshM")) {
                cfg.extrinsicHealth.fusionTransThreshM = std::max(0.001, *v);
            }
            if(auto v = getDouble(healthObj, "fusionRotThreshDeg")) {
                cfg.extrinsicHealth.fusionRotThreshDeg = std::max(0.1, *v);
            }
            if(auto v = getDouble(healthObj, "warnTransThreshM")) {
                cfg.extrinsicHealth.warnTransThreshM = std::max(0.0, *v);
            }
            if(auto v = getDouble(healthObj, "warnRotThreshDeg")) {
                cfg.extrinsicHealth.warnRotThreshDeg = std::max(0.0, *v);
            }
            if(auto v = getDouble(healthObj, "warnReprojThreshPx")) {
                cfg.extrinsicHealth.warnReprojThreshPx = std::max(0.0, *v);
            }
            if(auto v = getDouble(healthObj, "failTransThreshM")) {
                cfg.extrinsicHealth.failTransThreshM = std::max(0.0, *v);
            }
            if(auto v = getDouble(healthObj, "failRotThreshDeg")) {
                cfg.extrinsicHealth.failRotThreshDeg = std::max(0.0, *v);
            }
            if(auto v = getDouble(healthObj, "failReprojThreshPx")) {
                cfg.extrinsicHealth.failReprojThreshPx = std::max(0.0, *v);
            }
        }
    }

    if(auto v = getBool(root, "enableSync")) {
        cfg.enableSync = *v;
    }
    if(auto v = getInt(root, "queueCapacity")) {
        cfg.queueCapacity = *v;
    }
    if(auto v = getInt(root, "writerThreads")) {
        cfg.writerThreads = *v;
    }
    if(auto v = getInt(root, "recordQueueCapacity")) {
        cfg.recordQueueCapacity = std::max(0, *v);
    }
    if(auto v = getInt(root, "coordQueueCapacity")) {
        cfg.coordQueueCapacity = std::max(0, *v);
    }
    if(auto v = getInt(root, "writeQueueCapacity")) {
        cfg.writeQueueCapacity = std::max(0, *v);
    }
    if(auto v = getInt(root, "depthAlignWorkers")) {
        cfg.depthAlignWorkers = std::max(0, *v);
    }
    if(auto v = getInt(root, "depthAlignQueueCapacity")) {
        cfg.depthAlignQueueCapacity = std::max(0, *v);
    }
    if(auto v = getDouble(root, "cameraStreamTimeoutSec")) {
        if(*v > 0.0) {
            cfg.cameraStreamTimeoutSec = *v;
        }
    }
    else if(auto v = getDouble(root, "camera_frame_timeout_sec")) {
        if(*v > 0.0) {
            cfg.cameraStreamTimeoutSec = *v;
        }
    }
    else if(auto v = getDouble(root, "cameraFrameTimeoutSec")) {
        if(*v > 0.0) {
            cfg.cameraStreamTimeoutSec = *v;
        }
    }
    if(auto v = getDouble(root, "colorExposureMs")) {
        cfg.colorExposureMs = static_cast<float>(std::max(0.0, *v));
    }
    else if(auto v = getDouble(root, "exposure_ms")) {
        cfg.colorExposureMs = static_cast<float>(std::max(0.0, *v));
    }
    else if(auto v = getDouble(root, "exposureMs")) {
        cfg.colorExposureMs = static_cast<float>(std::max(0.0, *v));
    }
    if(auto v = getInt(root, "colorBrightness")) {
        cfg.colorBrightness = *v;
    }
    else if(auto v = getInt(root, "brightness")) {
        cfg.colorBrightness = *v;
    }
    if(auto v = getInt(root, "colorCloudRgbFrameOffset")) {
        cfg.colorCloudRgbFrameOffset = std::max(-5, std::min(5, *v));
    }

    if(auto *voiceObj = cJSON_GetObjectItemCaseSensitive(root, "voiceFeedback")) {
        if(cJSON_IsObject(voiceObj)) {
            if(auto v = getBool(voiceObj, "enabled")) {
                cfg.voiceFeedback.enabled = *v;
            }
            if(auto v = getString(voiceObj, "speakerDevice")) {
                cfg.voiceFeedback.speakerDevice = trimString(*v);
            }
            else if(auto v = getString(voiceObj, "device")) {
                cfg.voiceFeedback.speakerDevice = trimString(*v);
            }
            else if(auto v = getString(voiceObj, "outputDevice")) {
                cfg.voiceFeedback.speakerDevice = trimString(*v);
            }
            if(cfg.voiceFeedback.speakerDevice.empty()) {
                cfg.voiceFeedback.speakerDevice = "default";
            }
            if(auto v = getString(voiceObj, "command")) {
                cfg.voiceFeedback.command = trimString(*v);
            }
            if(auto v = getString(voiceObj, "voice")) {
                cfg.voiceFeedback.voice = trimString(*v);
            }
            else if(auto v = getString(voiceObj, "ttsVoice")) {
                cfg.voiceFeedback.voice = trimString(*v);
            }
            if(cfg.voiceFeedback.voice.empty()) {
                cfg.voiceFeedback.voice = "zh-CN-XiaoxiaoNeural";
            }
            if(auto v = getString(voiceObj, "rate")) {
                cfg.voiceFeedback.rate = trimString(*v);
            }
            else if(auto v = getString(voiceObj, "ttsRate")) {
                cfg.voiceFeedback.rate = trimString(*v);
            }
            if(auto v = getString(voiceObj, "pitch")) {
                cfg.voiceFeedback.pitch = trimString(*v);
            }
            else if(auto v = getString(voiceObj, "ttsPitch")) {
                cfg.voiceFeedback.pitch = trimString(*v);
            }
            if(auto v = getBool(voiceObj, "naturalOnly")) {
                cfg.voiceFeedback.naturalOnly = *v;
            }
            else if(auto v = getBool(voiceObj, "disableMechanicalFallback")) {
                cfg.voiceFeedback.naturalOnly = *v;
            }
            if(auto *messagesObj = cJSON_GetObjectItemCaseSensitive(voiceObj, "messages")) {
                if(cJSON_IsObject(messagesObj)) {
                    cJSON *item = nullptr;
                    cJSON_ArrayForEach(item, messagesObj) {
                        if(item && item->string && cJSON_IsString(item) && item->valuestring) {
                            cfg.voiceFeedback.messages[item->string] = item->valuestring;
                        }
                    }
                }
            }
        }
    }

    if(auto *backendObj = cJSON_GetObjectItemCaseSensitive(root, "taskBackend")) {
        if(cJSON_IsObject(backendObj)) {
            if(auto v = getBool(backendObj, "enabled")) {
                cfg.taskBackend.enabled = *v;
            }
            if(auto v = getString(backendObj, "baseUrl")) {
                cfg.taskBackend.baseUrl = trimString(*v);
            }
            else if(auto v = getString(backendObj, "url")) {
                cfg.taskBackend.baseUrl = trimString(*v);
            }
            if(auto v = getInt(backendObj, "timeoutMs")) {
                cfg.taskBackend.timeoutMs = std::max(500, *v);
            }
            if(auto *nasObj = cJSON_GetObjectItemCaseSensitive(backendObj, "nas")) {
                if(cJSON_IsObject(nasObj)) {
                    if(auto v = getBool(nasObj, "enabled")) {
                        cfg.taskBackend.nas.enabled = *v;
                    }
                    if(auto v = getBool(nasObj, "deleteLocalAfterUpload")) {
                        cfg.taskBackend.nas.deleteLocalAfterUpload = *v;
                    }
                    if(auto v = getString(nasObj, "serverIp")) {
                        cfg.taskBackend.nas.serverIp = trimString(*v);
                    }
                    if(auto v = getString(nasObj, "shareName")) {
                        cfg.taskBackend.nas.shareName = trimString(*v);
                    }
                    if(auto v = getString(nasObj, "sharePath")) {
                        cfg.taskBackend.nas.sharePath = trimString(*v);
                    }
                    if(auto v = getString(nasObj, "mountPath")) {
                        cfg.taskBackend.nas.mountPath = resolveConfigRelativePath(trimString(*v));
                    }
                    else if(auto v = getString(nasObj, "root")) {
                        cfg.taskBackend.nas.mountPath = resolveConfigRelativePath(trimString(*v));
                    }
                    if(auto v = getString(nasObj, "uriPrefix")) {
                        cfg.taskBackend.nas.uriPrefix = trimString(*v);
                    }
                }
            }
        }
    }
    if(const char *v = std::getenv("ORBBEC_TASK_BACKEND_URL")) {
        const std::string value = trimString(v);
        if(!value.empty()) {
            cfg.taskBackend.baseUrl = value;
        }
    }
    if(const char *v = std::getenv("ORBBEC_TASK_BACKEND_TIMEOUT_MS")) {
        try {
            cfg.taskBackend.timeoutMs = std::max(500, std::stoi(v));
        }
        catch(...) {
        }
    }
    if(const char *v = std::getenv("ORBBEC_TASK_BACKEND_ENABLED")) {
        std::string value = normalizePresetKey(v);
        cfg.taskBackend.enabled = !(value == "0" || value == "false" || value == "no" || value == "off");
    }
    if(cfg.taskBackend.baseUrl.empty()) {
        cfg.taskBackend.baseUrl = "http://127.0.0.1:8765";
    }

    if(auto *frontendsObj = cJSON_GetObjectItemCaseSensitive(root, "frontends")) {
        if(cJSON_IsObject(frontendsObj)) {
            if(auto *labelObj = cJSON_GetObjectItemCaseSensitive(frontendsObj, "label")) {
                if(cJSON_IsObject(labelObj)) {
                    if(auto v = getString(labelObj, "pythonExecutable")) {
                        cfg.frontends.label.pythonExecutable = trimString(*v);
                    }
                    else if(auto v = getString(labelObj, "python")) {
                        cfg.frontends.label.pythonExecutable = trimString(*v);
                    }
                    if(auto v = getString(labelObj, "operatorId")) {
                        cfg.frontends.label.operatorId = trimString(*v);
                    }
                    if(auto v = getString(labelObj, "logPath")) {
                        const std::string value = trimString(*v);
                        cfg.frontends.label.logPath = value.empty() ? fs::path() : resolveConfigRelativePath(value);
                    }
                    if(auto v = getString(labelObj, "frameCacheDir")) {
                        const std::string value = trimString(*v);
                        cfg.frontends.label.frameCacheDir = value.empty() ? fs::path() : resolveConfigRelativePath(value);
                    }
                    if(auto v = getString(labelObj, "ffmpegExecutable")) {
                        cfg.frontends.label.ffmpegExecutable = trimString(*v);
                    }
                    if(auto v = getInt(labelObj, "leaseSeconds")) {
                        cfg.frontends.label.leaseSeconds = std::max(1, *v);
                    }
                    if(auto v = getDouble(labelObj, "requestTimeoutSeconds")) {
                        cfg.frontends.label.requestTimeoutSeconds = std::max(1.0, *v);
                    }
                    else if(auto v = getInt(labelObj, "requestTimeoutMs")) {
                        cfg.frontends.label.requestTimeoutSeconds = std::max(1.0, static_cast<double>(*v) / 1000.0);
                    }
                }
            }
            if(auto *qcObj = cJSON_GetObjectItemCaseSensitive(frontendsObj, "qc")) {
                if(cJSON_IsObject(qcObj)) {
                    if(auto v = getString(qcObj, "pythonExecutable")) {
                        cfg.frontends.qc.pythonExecutable = trimString(*v);
                    }
                    else if(auto v = getString(qcObj, "python")) {
                        cfg.frontends.qc.pythonExecutable = trimString(*v);
                    }
                    if(auto v = getString(qcObj, "operatorId")) {
                        cfg.frontends.qc.operatorId = trimString(*v);
                    }
                    if(auto v = getString(qcObj, "logPath")) {
                        const std::string value = trimString(*v);
                        cfg.frontends.qc.logPath = value.empty() ? fs::path() : resolveConfigRelativePath(value);
                    }
                    if(auto v = getInt(qcObj, "sampleInterval")) {
                        cfg.frontends.qc.sampleInterval = std::max(1, *v);
                    }
                    if(auto v = getInt(qcObj, "defaultLeaseMinutes")) {
                        cfg.frontends.qc.defaultLeaseMinutes = std::max(1, *v);
                    }
                    if(auto v = getInt(qcObj, "crashLeaseExtensionMinutes")) {
                        cfg.frontends.qc.crashLeaseExtensionMinutes = std::max(1, *v);
                    }
                    if(auto v = getString(qcObj, "tmpDir")) {
                        const std::string value = trimString(*v);
                        cfg.frontends.qc.tmpDir = value.empty() ? fs::path("./tmp") : resolveConfigRelativePath(value);
                    }
                    if(auto v = getString(qcObj, "stateDir")) {
                        const std::string value = trimString(*v);
                        cfg.frontends.qc.stateDir = value.empty() ? fs::path("./qc_state") : resolveConfigRelativePath(value);
                    }
                    if(auto v = getString(qcObj, "workerMachineId")) {
                        cfg.frontends.qc.workerMachineId = trimString(*v);
                    }
                    if(auto v = getInt(qcObj, "rangeMergeGapFrames")) {
                        cfg.frontends.qc.rangeMergeGapFrames = std::max(0, *v);
                    }
                    if(auto v = getDouble(qcObj, "requestTimeoutSeconds")) {
                        cfg.frontends.qc.requestTimeoutSeconds = std::max(1.0, *v);
                    }
                    else if(auto v = getInt(qcObj, "requestTimeoutMs")) {
                        cfg.frontends.qc.requestTimeoutSeconds = std::max(1.0, static_cast<double>(*v) / 1000.0);
                    }
                }
            }
        }
    }

    if(auto *saveObj = cJSON_GetObjectItemCaseSensitive(root, "save")) {
        if(cJSON_IsObject(saveObj)) {
            if(auto v = getString(saveObj, "colorExt")) {
                cfg.save.colorExt = *v;
            }
            if(auto v = getInt(saveObj, "jpegQuality")) {
                cfg.save.jpegQuality = *v;
            }
            if(auto v = getInt(saveObj, "pngCompression")) {
                cfg.save.pngCompression = *v;
            }
            if(auto v = getBool(saveObj, "rgbH265")) {
                cfg.save.rgbH265 = *v;
            }
            if(auto v = getBool(saveObj, "saveRaw")) {
                cfg.save.saveRaw = *v;
            }
            if(auto v = getString(saveObj, "rgbEncoding")) {
                const std::string mode = normalizePresetKey(*v);
                cfg.save.rgbH265 = (mode == "h265" || mode == "hevc");
            }
            if(auto v = getString(saveObj, "depthEncoding")) {
                cfg.save.depthEncoding = normalizeDepthEncodingConfig(*v);
            }
            if(auto v = getString(saveObj, "depthFormat")) {
                cfg.save.depthEncoding = normalizeDepthEncodingConfig(*v);
            }
            if(auto v = getBool(saveObj, "depthFfv1Mkv")) {
                cfg.save.depthEncoding = *v ? "ffv1_mkv" : "png";
            }
            if(auto v = getString(saveObj, "h265Ext")) {
                cfg.save.h265Ext = *v;
            }
            if(auto v = getString(saveObj, "h265EncoderMode")) {
                cfg.save.h265EncoderMode = trimString(*v);
            }
            if(auto v = getString(saveObj, "h265Codec")) {
                cfg.save.h265Codec = trimString(*v);
            }
            if(auto v = getString(saveObj, "h265HwDevice")) {
                cfg.save.h265HwDevice = trimString(*v);
            }
            if(auto v = getString(saveObj, "h265LogLevel")) {
                cfg.save.h265LogLevel = trimString(*v);
            }
            if(auto v = getString(saveObj, "h265Preset")) {
                cfg.save.h265Preset = *v;
            }
            if(auto v = getInt(saveObj, "h265Crf")) {
                cfg.save.h265Crf = std::max(0, std::min(51, *v));
            }
            if(auto v = getInt(saveObj, "h265Threads")) {
                cfg.save.h265Threads = std::max(0, *v);
            }
            if(auto v = getInt(saveObj, "h265QueueCapacity")) {
                cfg.save.h265QueueCapacity = std::max(1, *v);
            }
            if(auto v = getInt(saveObj, "depthFfv1QueueCapacity")) {
                cfg.save.depthFfv1QueueCapacity = std::max(1, *v);
            }
            if(auto *threadsObj = cJSON_GetObjectItemCaseSensitive(saveObj, "h265ThreadsByCamera")) {
                if(cJSON_IsObject(threadsObj)) {
                    cJSON *item = nullptr;
                    cJSON_ArrayForEach(item, threadsObj) {
                        if(item && item->string && cJSON_IsNumber(item)) {
                            cfg.save.h265ThreadsByCamera[item->string] = std::max(0, item->valueint);
                        }
                    }
                }
            }
        }
    }

    cfg.ego.cameraParamsPath = (configBase / "../../camera_info/ego_camera_params.json").lexically_normal();
    if(auto *egoObj = cJSON_GetObjectItemCaseSensitive(root, "ego")) {
        if(cJSON_IsObject(egoObj)) {
            if(auto v = getBool(egoObj, "enabled")) {
                cfg.ego.enabled = *v;
            }
            if(auto v = getString(egoObj, "host")) {
                cfg.ego.host = trimString(*v);
            }
            else if(auto v = getString(egoObj, "bindHost")) {
                cfg.ego.host = trimString(*v);
            }
            if(cfg.ego.host.empty()) {
                cfg.ego.host = "127.0.0.1";
            }
            if(auto v = getInt(egoObj, "port")) {
                cfg.ego.port = std::max(1, std::min(65535, *v));
            }
            if(auto v = getBool(egoObj, "timeCalibrate")) {
                cfg.ego.timeCalibrate = *v;
            }
            else if(auto v = getBool(egoObj, "timecalibrate")) {
                cfg.ego.timeCalibrate = *v;
            }
            else if(auto v = getBool(egoObj, "timeSync")) {
                cfg.ego.timeCalibrate = *v;
            }
            if(auto v = getInt(egoObj, "timeCalibrateSampleCount")) {
                cfg.ego.timeCalibrateSampleCount = std::max(1, std::min(200, *v));
            }
            else if(auto v = getInt(egoObj, "timeSyncSampleCount")) {
                cfg.ego.timeCalibrateSampleCount = std::max(1, std::min(200, *v));
            }
            if(auto v = getInt(egoObj, "timeCalibrateTimeoutMs")) {
                cfg.ego.timeCalibrateTimeoutMs = std::max(100, *v);
            }
            else if(auto v = getDouble(egoObj, "timeCalibrateTimeoutSec")) {
                cfg.ego.timeCalibrateTimeoutMs = std::max(100, static_cast<int>(*v * 1000.0 + 0.5));
            }
            else if(auto v = getInt(egoObj, "timeSyncTimeoutMs")) {
                cfg.ego.timeCalibrateTimeoutMs = std::max(100, *v);
            }
            if(auto v = getBool(egoObj, "softAlignToOrbbecFirstFrame")) {
                cfg.ego.softAlignToOrbbecFirstFrame = *v;
            }
            else if(auto v = getBool(egoObj, "softAlignOrbbecFirstFrame")) {
                cfg.ego.softAlignToOrbbecFirstFrame = *v;
            }
            else if(auto v = getBool(egoObj, "softAlignFirstFrame")) {
                cfg.ego.softAlignToOrbbecFirstFrame = *v;
            }
            if(auto v = getInt(egoObj, "stopTimeoutMs")) {
                cfg.ego.stopTimeoutMs = std::max(100, *v);
            }
            else if(auto v = getDouble(egoObj, "stopTimeoutSec")) {
                cfg.ego.stopTimeoutMs = std::max(100, static_cast<int>(*v * 1000.0 + 0.5));
            }
            if(auto v = getInt(egoObj, "maxBufferedFrames")) {
                cfg.ego.maxBufferedFrames = static_cast<size_t>(std::max(1, *v));
            }
            if(auto v = getString(egoObj, "cameraId")) {
                cfg.ego.cameraId = trimString(*v);
            }
            if(cfg.ego.cameraId.empty()) {
                cfg.ego.cameraId = "ego";
            }
            if(auto v = getString(egoObj, "cameraParamsPath")) {
                cfg.ego.cameraParamsPath = resolveConfigRelativePath(*v);
            }
            else if(auto v = getString(egoObj, "cameraParamsFile")) {
                cfg.ego.cameraParamsPath = resolveConfigRelativePath(*v);
            }
            else if(auto v = getString(egoObj, "calibrationDir")) {
                cfg.ego.cameraParamsPath = resolveConfigRelativePath(*v) / "camera_params.json";
            }
            else if(auto v = getString(egoObj, "cameraInfoDir")) {
                cfg.ego.cameraParamsPath = resolveConfigRelativePath(*v) / "camera_params.json";
            }
        }
    }

    cJSON *touchObj = cJSON_GetObjectItemCaseSensitive(root, "touch");
    if(!touchObj) {
        touchObj = cJSON_GetObjectItemCaseSensitive(root, "tactile");
    }
    if(touchObj && cJSON_IsObject(touchObj)) {
        if(auto v = getBool(touchObj, "enabled")) {
            cfg.touch.enabled = *v;
        }
        if(auto v = getBool(touchObj, "required")) {
            cfg.touch.required = *v;
        }
        if(auto v = getString(touchObj, "streamId")) {
            cfg.touch.streamId = trimString(*v);
        }
        else if(auto v = getString(touchObj, "id")) {
            cfg.touch.streamId = trimString(*v);
        }
        if(auto v = getString(touchObj, "handSide")) {
            cfg.touch.handSide = trimString(*v);
        }
        else if(auto v = getString(touchObj, "side")) {
            cfg.touch.handSide = trimString(*v);
        }
        if(cfg.touch.handSide.empty()) {
            cfg.touch.handSide = "right";
        }
        if(auto v = getInt(touchObj, "sensorType")) {
            cfg.touch.sensorType = std::max(0, *v);
        }
        else if(auto v = getInt(touchObj, "sensor_type")) {
            cfg.touch.sensorType = std::max(0, *v);
        }
        if(auto v = getInt(touchObj, "targetFps")) {
            cfg.touch.targetFps = std::max(1, *v);
        }
        if(auto v = getInt(touchObj, "maxBufferedSamples")) {
            cfg.touch.maxBufferedSamples = static_cast<size_t>(std::max(1, *v));
        }
        if(auto v = getString(touchObj, "directoryName")) {
            cfg.touch.save.directoryName = trimString(*v);
        }
        else if(auto v = getString(touchObj, "outputDirName")) {
            cfg.touch.save.directoryName = trimString(*v);
        }
        if(cfg.touch.save.directoryName.empty()) {
            cfg.touch.save.directoryName = "touch";
        }
        if(auto v = getInt(touchObj, "csvFloatPrecision")) {
            cfg.touch.save.csvFloatPrecision = std::max(0, *v);
        }

        cJSON *serialObj = cJSON_GetObjectItemCaseSensitive(touchObj, "serial");
        if(serialObj && cJSON_IsObject(serialObj)) {
            if(auto v = getString(serialObj, "portPath")) {
                cfg.touch.serial.portPath = trimString(*v);
            }
            else if(auto v = getString(serialObj, "port")) {
                cfg.touch.serial.portPath = trimString(*v);
            }
            else if(auto v = getString(serialObj, "device")) {
                cfg.touch.serial.portPath = trimString(*v);
            }
            if(auto v = getInt(serialObj, "baudRate")) {
                cfg.touch.serial.baudRate = std::max(1, *v);
            }
            else if(auto v = getInt(serialObj, "baud")) {
                cfg.touch.serial.baudRate = std::max(1, *v);
            }
            if(auto v = getInt(serialObj, "timeoutMs")) {
                cfg.touch.serial.timeoutMs = std::max(1, *v);
            }
        }
        if(auto v = getString(touchObj, "portPath")) {
            cfg.touch.serial.portPath = trimString(*v);
        }
        else if(auto v = getString(touchObj, "port")) {
            cfg.touch.serial.portPath = trimString(*v);
        }
        else if(auto v = getString(touchObj, "device")) {
            cfg.touch.serial.portPath = trimString(*v);
        }
        if(auto v = getInt(touchObj, "baudRate")) {
            cfg.touch.serial.baudRate = std::max(1, *v);
        }
        else if(auto v = getInt(touchObj, "baud")) {
            cfg.touch.serial.baudRate = std::max(1, *v);
        }
        if(auto v = getInt(touchObj, "timeoutMs")) {
            cfg.touch.serial.timeoutMs = std::max(1, *v);
        }

        cJSON *devicesObj = cJSON_GetObjectItemCaseSensitive(touchObj, "devices");
        if(!devicesObj) {
            devicesObj = cJSON_GetObjectItemCaseSensitive(touchObj, "hands");
        }
        if(devicesObj && cJSON_IsArray(devicesObj)) {
            cfg.touch.devices.clear();
            cJSON *deviceObj = nullptr;
            cJSON_ArrayForEach(deviceObj, devicesObj) {
                if(!deviceObj || !cJSON_IsObject(deviceObj)) {
                    continue;
                }
                TactileDeviceConfig dev;
                dev.serial = cfg.touch.serial;
                if(auto v = getString(deviceObj, "streamId")) {
                    dev.streamId = trimString(*v);
                }
                else if(auto v = getString(deviceObj, "id")) {
                    dev.streamId = trimString(*v);
                }
                if(auto v = getString(deviceObj, "handSide")) {
                    dev.handSide = trimString(*v);
                }
                else if(auto v = getString(deviceObj, "side")) {
                    dev.handSide = trimString(*v);
                }
                if(auto v = getInt(deviceObj, "sensorType")) {
                    dev.sensorType = std::max(0, *v);
                }
                else if(auto v = getInt(deviceObj, "sensor_type")) {
                    dev.sensorType = std::max(0, *v);
                }
                cJSON *deviceSerialObj = cJSON_GetObjectItemCaseSensitive(deviceObj, "serial");
                if(deviceSerialObj && cJSON_IsObject(deviceSerialObj)) {
                    if(auto v = getString(deviceSerialObj, "portPath")) {
                        dev.serial.portPath = trimString(*v);
                    }
                    else if(auto v = getString(deviceSerialObj, "port")) {
                        dev.serial.portPath = trimString(*v);
                    }
                    else if(auto v = getString(deviceSerialObj, "device")) {
                        dev.serial.portPath = trimString(*v);
                    }
                    if(auto v = getInt(deviceSerialObj, "baudRate")) {
                        dev.serial.baudRate = std::max(1, *v);
                    }
                    else if(auto v = getInt(deviceSerialObj, "baud")) {
                        dev.serial.baudRate = std::max(1, *v);
                    }
                    if(auto v = getInt(deviceSerialObj, "timeoutMs")) {
                        dev.serial.timeoutMs = std::max(1, *v);
                    }
                }
                if(auto v = getString(deviceObj, "portPath")) {
                    dev.serial.portPath = trimString(*v);
                }
                else if(auto v = getString(deviceObj, "port")) {
                    dev.serial.portPath = trimString(*v);
                }
                else if(auto v = getString(deviceObj, "device")) {
                    dev.serial.portPath = trimString(*v);
                }
                if(auto v = getInt(deviceObj, "baudRate")) {
                    dev.serial.baudRate = std::max(1, *v);
                }
                else if(auto v = getInt(deviceObj, "baud")) {
                    dev.serial.baudRate = std::max(1, *v);
                }
                if(auto v = getInt(deviceObj, "timeoutMs")) {
                    dev.serial.timeoutMs = std::max(1, *v);
                }
                cfg.touch.devices.push_back(std::move(dev));
            }
        }
    }

    if(auto *fisheyeObj = cJSON_GetObjectItemCaseSensitive(root, "fisheye")) {
        if(cJSON_IsObject(fisheyeObj)) {
            if(auto v = getBool(fisheyeObj, "enabled")) {
                cfg.fisheye.enabled = *v;
            }
            if(auto v = getInt(fisheyeObj, "targetFps")) {
                cfg.fisheye.targetFps = std::max(1, *v);
            }
            if(auto v = getInt(fisheyeObj, "maxBufferedSets")) {
                cfg.fisheye.maxBufferedSets = static_cast<size_t>(std::max(0, *v));
            }

            if(auto *saveObj = cJSON_GetObjectItemCaseSensitive(fisheyeObj, "save")) {
                if(cJSON_IsObject(saveObj)) {
                    if(auto v = getString(saveObj, "format")) {
                        if(const auto fmt = fisheyeImageFormatFromString(*v)) {
                            cfg.fisheye.save.format = *fmt;
                        }
                    }
                    if(auto v = getInt(saveObj, "jpegQuality")) {
                        cfg.fisheye.save.jpegQuality = std::max(0, std::min(100, *v));
                    }
                    if(auto v = getInt(saveObj, "pngCompression")) {
                        cfg.fisheye.save.pngCompression = std::max(0, std::min(9, *v));
                    }
                }
            }

            if(auto *camerasObj = cJSON_GetObjectItemCaseSensitive(fisheyeObj, "cameras")) {
                if(cJSON_IsArray(camerasObj)) {
                    cfg.fisheye.cameras.clear();
                    cJSON *camObj = nullptr;
                    cJSON_ArrayForEach(camObj, camerasObj) {
                        if(!cJSON_IsObject(camObj)) {
                            continue;
                        }

                        FisheyeCameraConfig camera;
                        if(auto v = getString(camObj, "cameraId")) {
                            camera.cameraId = *v;
                        }
                        if(auto v = getString(camObj, "handRole")) {
                            camera.handRole = *v;
                        }
                        if(auto v = getString(camObj, "uniqueId")) {
                            camera.uniqueId = trimString(*v);
                        }
                        if(auto v = getString(camObj, "devicePath")) {
                            camera.devicePath = *v;
                        }
                        if(auto v = getString(camObj, "preferredDeviceHint")) {
                            camera.preferredDeviceHint = *v;
                        }
                        if(auto v = getInt(camObj, "deviceIndex")) {
                            camera.deviceIndex = *v;
                        }
                        if(auto v = getInt(camObj, "width")) {
                            camera.width = std::max(1, *v);
                        }
                        if(auto v = getInt(camObj, "height")) {
                            camera.height = std::max(1, *v);
                        }
                        if(auto v = getInt(camObj, "fps")) {
                            camera.fps = std::max(1, *v);
                        }
                        if(auto v = getBool(camObj, "preferMjpeg")) {
                            camera.preferMjpeg = *v;
                        }

                        cfg.fisheye.cameras.push_back(std::move(camera));
                    }
                }
            }
        }
    }

    auto *devices = cJSON_GetObjectItemCaseSensitive(root, "devices");
    if(!devices || !cJSON_IsArray(devices)) {
        cJSON_Delete(root);
        throw std::runtime_error("Missing or invalid 'devices' in config");
    }

    cJSON *dev                = nullptr;
    int    defaultDeviceIndex = 0;
    cJSON_ArrayForEach(dev, devices) {
        if(!cJSON_IsObject(dev)) {
            continue;
        }
        auto snOpt = getString(dev, "sn");
        if(!snOpt.has_value()) {
            continue;
        }
        DeviceConfig dc;
        dc.sn = *snOpt;
        if(auto idx = getString(dev, "index")) {
            dc.index = *idx;
        }
        else if(auto idx2 = getString(dev, "device_index")) {
            dc.index = *idx2;
        }
        else {
            std::ostringstream oss;
            oss << std::setw(2) << std::setfill('0') << defaultDeviceIndex;
            dc.index = oss.str();
        }
        defaultDeviceIndex++;
        memset(&dc.syncConfig, 0, sizeof(dc.syncConfig));
        dc.syncConfig.syncMode = OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN;

        if(auto *syncObj = cJSON_GetObjectItemCaseSensitive(dev, "syncConfig")) {
            if(cJSON_IsObject(syncObj)) {
                dc.hasSyncConfig = true;
                if(auto modeStr = getString(syncObj, "syncMode")) {
                    dc.syncConfig.syncMode = stringToOBSyncMode(*modeStr);
                }
                if(auto v = getInt(syncObj, "depthDelayUs")) {
                    dc.syncConfig.depthDelayUs = *v;
                }
                if(auto v = getInt(syncObj, "colorDelayUs")) {
                    dc.syncConfig.colorDelayUs = *v;
                }
                if(auto v = getInt(syncObj, "trigger2ImageDelayUs")) {
                    dc.syncConfig.trigger2ImageDelayUs = *v;
                }
                if(auto v = getBool(syncObj, "triggerOutEnable")) {
                    dc.syncConfig.triggerOutEnable = *v;
                }
                if(auto v = getInt(syncObj, "triggerOutDelayUs")) {
                    dc.syncConfig.triggerOutDelayUs = *v;
                }
                if(auto v = getInt(syncObj, "framesPerTrigger")) {
                    dc.syncConfig.framesPerTrigger = *v;
                }
            }
        }

        if(auto *streams = cJSON_GetObjectItemCaseSensitive(dev, "streams")) {
            if(cJSON_IsArray(streams)) {
                cJSON *st = nullptr;
                cJSON_ArrayForEach(st, streams) {
                    if(!cJSON_IsObject(st)) {
                        continue;
                    }
                    auto typeStr = getString(st, "type");
                    if(!typeStr.has_value()) {
                        continue;
                    }
                    auto typeOpt = streamTypeFromString(*typeStr);
                    if(!typeOpt.has_value()) {
                        continue;
                    }
                    StreamConfig sc;
                    sc.type   = *typeOpt;
                    sc.enable = getBool(st, "enable").value_or(true);
                    sc.width  = getInt(st, "width").value_or(0);
                    sc.height = getInt(st, "height").value_or(0);
                    sc.fps    = getInt(st, "fps").value_or(0);
                    sc.format = getString(st, "format").value_or(std::string());
                    dc.streams.push_back(std::move(sc));
                }
            }
        }

        cfg.devices.push_back(std::move(dc));
    }

    const bool legacyViewer      = cfg.mode.empty() ? getBool(root, "viewer_mode").value_or(false) : false;
    const bool legacyInteractive = cfg.mode.empty() ? getBool(root, "interactive").value_or(false) : false;

    cJSON_Delete(root);

    if(cfg.devices.empty()) {
        throw std::runtime_error("No valid devices in config");
    }

    if(cfg.mode.empty()) {
        if(legacyInteractive) {
            cfg.mode = "interaction";
        }
        else if(legacyViewer) {
            cfg.mode = "viewer";
        }
        else {
            cfg.mode = "collection";
        }
    }

    if(cfg.mode != "viewer" && cfg.mode != "interaction" && cfg.mode != "collection" && cfg.mode != "calibration" && cfg.mode != "label") {
        throw std::runtime_error("Invalid mode in config: " + cfg.mode + ", expected viewer/interaction/collection/calibration/label");
    }
    return cfg;
}

std::vector<DeviceRuntime> selectDevicesWithPipeline(const std::shared_ptr<ob::DeviceList> &deviceList, const AppConfig &cfg) {
    std::unordered_map<std::string, std::shared_ptr<ob::Device>> bySn;
    for(uint32_t i = 0; i < deviceList->deviceCount(); i++) {
        auto dev = deviceList->getDevice(i);
        bySn.emplace(std::string(dev->getDeviceInfo()->serialNumber()), dev);
    }

    std::vector<DeviceRuntime> out;
    out.reserve(cfg.devices.size());
    int index = 0;
    for(const auto &dc: cfg.devices) {
        auto it = bySn.find(dc.sn);
        if(it == bySn.end()) {
            std::cerr << "Configured device not found: " << dc.sn << std::endl;
            continue;
        }
        DeviceRuntime rt;
        rt.cfg         = dc;
        rt.dev         = it->second;
        try {
            if(rt.dev->isGlobalTimestampSupported()) {
                rt.dev->enableGlobalTimestamp(true);
                std::cerr << "[sync] global timestamp enabled sn=" << dc.sn << std::endl;
            }
            else {
                std::cerr << "[sync] global timestamp unsupported sn=" << dc.sn << std::endl;
            }
        }
        catch(const std::exception &e) {
            std::cerr << "[sync] global timestamp enable failed sn=" << dc.sn << ": " << e.what() << std::endl;
        }
        rt.pipe        = std::make_shared<ob::Pipeline>(rt.dev);
        rt.deviceIndex = index++;
        out.push_back(std::move(rt));
    }
    return out;
}

void applySyncConfig(std::vector<DeviceRuntime> &devices) {
    for(size_t i = 0; i < devices.size(); i++) {
        auto &rt = devices[i];
        auto  cur = rt.dev->getMultiDeviceSyncConfig();
        auto  cfg = rt.cfg.hasSyncConfig ? rt.cfg.syncConfig : cur;
        if(!rt.cfg.hasSyncConfig) {
            const bool isPrimary = (i == 0);
            cfg.syncMode         = isPrimary ? OB_MULTI_DEVICE_SYNC_MODE_PRIMARY : OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED;
            cfg.triggerOutEnable = isPrimary;
            cfg.triggerOutDelayUs = 0;
            if(cfg.framesPerTrigger <= 0) {
                cfg.framesPerTrigger = 1;
            }
        }

        const bool isTriggerSource = (cfg.syncMode == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY);
        cfg.triggerOutEnable      = isTriggerSource ? cfg.triggerOutEnable : false;
        if(cfg.framesPerTrigger <= 0) {
            cfg.framesPerTrigger = 1;
        }

        cur.syncMode             = cfg.syncMode;
        cur.depthDelayUs         = cfg.depthDelayUs;
        cur.colorDelayUs         = cfg.colorDelayUs;
        cur.trigger2ImageDelayUs = cfg.trigger2ImageDelayUs;
        cur.triggerOutEnable     = cfg.triggerOutEnable;
        cur.triggerOutDelayUs    = cfg.triggerOutDelayUs;
        cur.framesPerTrigger     = cfg.framesPerTrigger;

        rt.dev->setMultiDeviceSyncConfig(cur);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        try {
            const auto applied = rt.dev->getMultiDeviceSyncConfig();
            std::cerr << "[sync] applied sn=" << rt.cfg.sn
                      << " index=" << rt.cfg.index
                      << " mode=" << obSyncModeToString(applied.syncMode)
                      << " triggerOutEnable=" << (applied.triggerOutEnable ? "true" : "false")
                      << " triggerOutDelayUs=" << applied.triggerOutDelayUs
                      << " framesPerTrigger=" << applied.framesPerTrigger
                      << std::endl;
        }
        catch(const std::exception &e) {
            std::cerr << "[sync] failed to read back sync config sn=" << rt.cfg.sn << " error=" << e.what() << std::endl;
        }
    }
}

void splitPrimarySecondary(const std::vector<DeviceRuntime> &all, std::vector<DeviceRuntime> &primary, std::vector<DeviceRuntime> &secondary) {
    for(const auto &rt: all) {
        auto cfg = rt.dev->getMultiDeviceSyncConfig();
        if(cfg.syncMode == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY) {
            primary.push_back(rt);
        }
        else {
            secondary.push_back(rt);
        }
    }
}

bool hasSoftwareTrigger(const std::vector<DeviceRuntime> &all) {
    for(const auto &rt: all) {
        auto cfg = rt.dev->getMultiDeviceSyncConfig();
        if(cfg.syncMode == OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING) {
            return true;
        }
    }
    return false;
}

static int preferredVideoProfileFormatScore(OBSensorType sensorType, OBFormat format) {
    if(sensorType == OB_SENSOR_COLOR) {
        if(format == OB_FORMAT_MJPG) {
            return 0;
        }
        if(format == OB_FORMAT_RGB) {
            return 1;
        }
        if(format == OB_FORMAT_BGR) {
            return 2;
        }
        if(format == OB_FORMAT_YUYV || format == OB_FORMAT_YUY2 || format == OB_FORMAT_UYVY) {
            return 3;
        }
        return 10;
    }
    if(sensorType == OB_SENSOR_DEPTH) {
        if(format == OB_FORMAT_Y14) {
            return 0;
        }
        if(format == OB_FORMAT_Y16 || format == OB_FORMAT_Z16) {
            return 1;
        }
        if(format == OB_FORMAT_RLE) {
            return 2;
        }
        return 10;
    }
    if(sensorType == OB_SENSOR_IR || sensorType == OB_SENSOR_IR_LEFT || sensorType == OB_SENSOR_IR_RIGHT) {
        if(format == OB_FORMAT_Y8) {
            return 0;
        }
        if(format == OB_FORMAT_Y16) {
            return 1;
        }
        if(format == OB_FORMAT_MJPG) {
            return 2;
        }
        return 10;
    }
    return 10;
}

static std::shared_ptr<ob::VideoStreamProfile> pickBestVideoProfile(const std::shared_ptr<ob::StreamProfileList> &list,
                                                                    OBSensorType sensorType,
                                                                    int width,
                                                                    int height,
                                                                    int fps) {
    if(!list || list->getCount() == 0) {
        return nullptr;
    }
    std::shared_ptr<ob::VideoStreamProfile> best;
    int bestScore = std::numeric_limits<int>::max();
    for(uint32_t i = 0; i < list->getCount(); ++i) {
        auto profile = list->getProfile(i)->as<ob::VideoStreamProfile>();
        if(!profile) {
            continue;
        }
        const int pw = static_cast<int>(profile->getWidth());
        const int ph = static_cast<int>(profile->getHeight());
        const int pf = static_cast<int>(profile->getFps());
        if(width > 0 && pw != width) {
            continue;
        }
        if(height > 0 && ph != height) {
            continue;
        }
        int score = preferredVideoProfileFormatScore(sensorType, profile->getFormat());
        if(fps > 0) {
            score += std::abs(pf - fps) * 10;
            if(pf < fps) {
                score += 2;
            }
        }
        if(score < bestScore) {
            bestScore = score;
            best = profile;
        }
    }
    return best;
}

std::shared_ptr<ob::VideoStreamProfile> pickDepthProfileForPointCloud(const std::shared_ptr<ob::Pipeline> &pipe,
                                                                      const std::vector<StreamConfig> &streams) {
    StreamConfig pointCloudSc{};
    bool         found = false;
    for(const auto &sc: streams) {
        if(sc.enable && sc.type == StreamType::PointCloud) {
            pointCloudSc = sc;
            found        = true;
            break;
        }
    }
    auto list = pipe->getStreamProfileList(OB_SENSOR_DEPTH);
    if(!list || list->getCount() == 0) {
        return nullptr;
    }

    if(found) {
        if(auto profile = pickBestVideoProfile(list, OB_SENSOR_DEPTH, pointCloudSc.width, pointCloudSc.height, pointCloudSc.fps)) {
            return profile;
        }
    }

    if(auto profile = pickBestVideoProfile(list, OB_SENSOR_DEPTH, 640, 400, 30)) {
        return profile;
    }
    if(auto profile = pickBestVideoProfile(list, OB_SENSOR_DEPTH, 1280, 800, 30)) {
        return profile;
    }
    if(auto profile = pickBestVideoProfile(list, OB_SENSOR_DEPTH, 320, 200, 30)) {
        return profile;
    }
    return pickBestVideoProfile(list, OB_SENSOR_DEPTH, 0, 0, 30);
}

std::shared_ptr<ob::VideoStreamProfile> pickDefaultVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe, OBSensorType sensorType) {
    std::shared_ptr<ob::StreamProfileList> list;
    try {
        list = pipe->getStreamProfileList(sensorType);
    }
    catch(...) {
        return nullptr;
    }
    if(!list || list->getCount() == 0) {
        return nullptr;
    }
    if(sensorType == OB_SENSOR_COLOR) {
        if(auto profile = pickBestVideoProfile(list, sensorType, 640, 400, 30)) {
            return profile;
        }
        if(auto profile = pickBestVideoProfile(list, sensorType, 1280, 800, 30)) {
            return profile;
        }
        if(auto profile = pickBestVideoProfile(list, sensorType, 1280, 720, 30)) {
            return profile;
        }
        if(auto profile = pickBestVideoProfile(list, sensorType, 1920, 1080, 30)) {
            return profile;
        }
        if(auto profile = pickBestVideoProfile(list, sensorType, 640, 480, 30)) {
            return profile;
        }
    }
    if(sensorType == OB_SENSOR_DEPTH) {
        if(auto profile = pickBestVideoProfile(list, sensorType, 640, 400, 30)) {
            return profile;
        }
        if(auto profile = pickBestVideoProfile(list, sensorType, 1280, 800, 30)) {
            return profile;
        }
        if(auto profile = pickBestVideoProfile(list, sensorType, 320, 200, 30)) {
            return profile;
        }
    }
    return pickBestVideoProfile(list, sensorType, 0, 0, 30);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr denoiseCloudSor(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in, int meanK, double stddevMulThresh) {
    auto out = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if(!in || in->empty()) {
        return out;
    }
    if(meanK <= 0 || stddevMulThresh <= 0.0) {
        *out = *in;
        return out;
    }
    if(in->size() < static_cast<size_t>(meanK)) {
        *out = *in;
        return out;
    }

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(in);
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stddevMulThresh);
    sor.filter(*out);
    return out;
}

void applyEdgeSmoothing(std::shared_ptr<ob::Frame> &frame, double thresholdM) {
    if(!frame || !(thresholdM > 0.0)) {
        return;
    }

    auto depth = frame->as<ob::DepthFrame>();
    if(!depth) {
        return;
    }

    if(depth->getFormat() != OB_FORMAT_Y16 && depth->getFormat() != OB_FORMAT_Y14 && depth->getFormat() != OB_FORMAT_Z16) {
        return;
    }

    const float scaleMm = depth->getValueScale();
    if(!(scaleMm > 0.0f)) {
        return;
    }

    const double thresholdU16 = (thresholdM * 1000.0) / static_cast<double>(scaleMm);
    if(!(thresholdU16 > 0.0)) {
        return;
    }

    const int    width    = static_cast<int>(depth->getWidth());
    const int    height   = static_cast<int>(depth->getHeight());
    void        *data     = frame->data();
    const size_t dataSize = frame->dataSize();
    const size_t stride   = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;

    if(stride < static_cast<size_t>(width) * sizeof(uint16_t)) {
        return;
    }

    try {
        cv::Mat depthMat(height, width, CV_16U, data, stride);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat grad;
        cv::morphologyEx(depthMat, grad, cv::MORPH_GRADIENT, kernel);

        cv::Mat mask;
        cv::compare(grad, thresholdU16, mask, cv::CMP_GT);

        cv::Mat nonZero;
        cv::compare(depthMat, 0, nonZero, cv::CMP_GT);
        cv::bitwise_and(mask, nonZero, mask);

        depthMat.setTo(0, mask);
    }
    catch(...) {
    }
}

std::shared_ptr<ob::Frame> refineDepthFrameForPointCloud(const std::shared_ptr<ob::Frame> &depthFrame,
                                                         OrbbecDepthFilterChain &chain,
                                                         float minDepthM,
                                                         float maxDepthM,
                                                         const DepthPointCloudFiltersConfig &cfg) {
    if(!depthFrame) {
        return nullptr;
    }

    std::shared_ptr<ob::Frame> out = depthFrame;

    if(cfg.decimationFilterScale > 0) {
        if(!chain.decimation) {
            chain.decimation = std::make_shared<ob::DecimationFilter>();
        }
        const int scale = std::min(8, std::max(1, cfg.decimationFilterScale));
        chain.decimation->setScaleValue(static_cast<uint8_t>(scale));
        try {
            out = chain.decimation->process(out);
        }
        catch(...) {
            return nullptr;
        }
        if(!out) {
            return nullptr;
        }
    }

    if(maxDepthM > 0.0f && minDepthM >= 0.0f && maxDepthM > minDepthM) {
        if(!chain.threshold) {
            chain.threshold = std::make_shared<ob::ThresholdFilter>();
        }
        const double minMmD = static_cast<double>(minDepthM) * 1000.0;
        const double maxMmD = static_cast<double>(maxDepthM) * 1000.0;
        const auto   minMm  = static_cast<uint16_t>(std::min(16000.0, std::max(0.0, minMmD)));
        const auto   maxMm  = static_cast<uint16_t>(std::min(16000.0, std::max(0.0, maxMmD)));
        if(minMm < maxMm) {
            chain.threshold->setValueRange(minMm, maxMm);
            try {
                out = chain.threshold->process(out);
            }
            catch(...) {
                return nullptr;
            }
            if(!out) {
                return nullptr;
            }
        }
    }

    if(cfg.noiseRemovalMaxSize > 0 || cfg.noiseRemovalMinDiff > 0) {
        if(!chain.noiseRemoval) {
            chain.noiseRemoval = std::make_shared<ob::NoiseRemovalFilter>();
        }
        auto params = chain.noiseRemoval->getFilterParams();
        if(cfg.noiseRemovalMaxSize > 0) {
            params.max_size = static_cast<uint16_t>(std::min(10000, std::max(1, cfg.noiseRemovalMaxSize)));
        }
        if(cfg.noiseRemovalMinDiff > 0) {
            params.disp_diff = static_cast<uint16_t>(std::min(65535, std::max(1, cfg.noiseRemovalMinDiff)));
        }
        chain.noiseRemoval->setFilterParams(params);
        try {
            out = chain.noiseRemoval->process(out);
        }
        catch(...) {
            return nullptr;
        }
        if(!out) {
            return nullptr;
        }
    }

    if(cfg.spatialAlpha > 0.0 || cfg.spatialDispDiff > 0 || cfg.spatialMagnitude > 0 || cfg.spatialRadius > 0) {
        if(!chain.spatial) {
            chain.spatial = std::make_shared<ob::SpatialAdvancedFilter>();
        }
        auto params = chain.spatial->getFilterParams();
        if(cfg.spatialAlpha > 0.0) {
            params.alpha = static_cast<float>(std::min(1.0, std::max(0.1, cfg.spatialAlpha)));
        }
        if(cfg.spatialDispDiff > 0) {
            params.disp_diff = static_cast<uint16_t>(std::min(65535, std::max(1, cfg.spatialDispDiff)));
        }
        if(cfg.spatialMagnitude > 0) {
            params.magnitude = static_cast<uint8_t>(std::min(10, std::max(1, cfg.spatialMagnitude)));
        }
        if(cfg.spatialRadius > 0) {
            params.radius = static_cast<uint16_t>(std::min(100, std::max(1, cfg.spatialRadius)));
        }
        chain.spatial->setFilterParams(params);
        try {
            out = chain.spatial->process(out);
        }
        catch(...) {
            return nullptr;
        }
        if(!out) {
            return nullptr;
        }
    }

    if(cfg.temporalDiffScale > 0.0 || cfg.temporalWeight > 0.0) {
        if(!chain.temporal) {
            chain.temporal = std::make_shared<ob::TemporalFilter>();
        }
        if(cfg.temporalDiffScale > 0.0) {
            chain.temporal->setDiffScale(static_cast<float>(std::min(1.0, std::max(0.001, cfg.temporalDiffScale))));
        }
        if(cfg.temporalWeight > 0.0) {
            chain.temporal->setWeight(static_cast<float>(std::min(1.0, std::max(0.001, cfg.temporalWeight))));
        }
        try {
            out = chain.temporal->process(out);
        }
        catch(...) {
            return nullptr;
        }
        if(!out) {
            return nullptr;
        }
    }

    if(cfg.holeFillingMode > 0) {
        if(!chain.holeFilling) {
            chain.holeFilling = std::make_shared<ob::HoleFillingFilter>();
        }
        OBHoleFillingMode mode = OB_HOLE_FILL_TOP;
        if(cfg.holeFillingMode == 2) {
            mode = OB_HOLE_FILL_NEAREST;
        }
        else if(cfg.holeFillingMode == 3) {
            mode = OB_HOLE_FILL_FAREST;
        }
        chain.holeFilling->setFilterMode(mode);
        try {
            out = chain.holeFilling->process(out);
        }
        catch(...) {
            return nullptr;
        }
        if(!out) {
            return nullptr;
        }
    }

    return out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr removeDominantPlaneRansac(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                                                              int maxIterations,
                                                              double distanceThreshold,
                                                              size_t minInliers) {
    auto out = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if(!in || in->empty()) {
        return out;
    }
    if(maxIterations <= 0 || distanceThreshold <= 0.0 || minInliers == 0) {
        *out = *in;
        return out;
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setInputCloud(in);

    pcl::PointIndices::Ptr     inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    seg.segment(*inliers, *coefficients);

    if(!inliers || inliers->indices.size() < minInliers) {
        *out = *in;
        return out;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(in);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*out);
    return out;
}

cv::Mat visualizeObFrame(const std::shared_ptr<const ob::Frame> &frame) {
    if(frame == nullptr) {
        return cv::Mat();
    }

    cv::Mat rstMat;
    switch(frame->getType()) {
    case OB_FRAME_COLOR:
    case OB_FRAME_COLOR_LEFT:
    case OB_FRAME_COLOR_RIGHT: {
        auto videoFrame = frame->as<const ob::VideoFrame>();
        switch(videoFrame->getFormat()) {
        case OB_FORMAT_MJPG: {
            const auto *dataPtr = reinterpret_cast<const uint8_t *>(videoFrame->getData());
            const size_t n      = static_cast<size_t>(videoFrame->getDataSize());
            if(!dataPtr || n == 0) {
                break;
            }
            cv::Mat rawMat(1, static_cast<int>(n), CV_8UC1, const_cast<uint8_t *>(dataPtr));
            try {
                rstMat = cv::imdecode(rawMat, cv::IMREAD_COLOR);
            }
            catch(...) {
                rstMat.release();
            }
        } break;
        case OB_FORMAT_NV21: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight() * 3 / 2), static_cast<int>(videoFrame->getWidth()), CV_8UC1, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_NV21);
        } break;
        case OB_FORMAT_YUYV:
        case OB_FORMAT_YUY2: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC2, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_YUY2);
        } break;
        case OB_FORMAT_BGR: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC3, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_BGR2RGB);
        } break;
        case OB_FORMAT_RGB: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC3, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_RGB2BGR);
        } break;
        case OB_FORMAT_RGBA: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC4, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_RGBA2BGR);
        } break;
        case OB_FORMAT_BGRA: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC4, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_BGRA2RGB);
        } break;
        case OB_FORMAT_UYVY: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC2, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_UYVY);
        } break;
        case OB_FORMAT_I420: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight() * 3 / 2), static_cast<int>(videoFrame->getWidth()), CV_8UC1, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_I420);
        } break;
        case OB_FORMAT_Y8: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC1, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_GRAY2BGR);
        } break;
        case OB_FORMAT_Y16: {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_16UC1, videoFrame->getData());
            cv::Mat gray8;
            rawMat.convertTo(gray8, CV_8UC1, 255.0 / 65535.0);
            cv::cvtColor(gray8, rstMat, cv::COLOR_GRAY2BGR);
        } break;
        default:
            break;
        }
    } break;
    case OB_FRAME_DEPTH: {
        auto videoFrame = frame->as<const ob::VideoFrame>();
        if(videoFrame->getFormat() == OB_FORMAT_Y16 || videoFrame->getFormat() == OB_FORMAT_Y14 || videoFrame->getFormat() == OB_FORMAT_Z16
           || videoFrame->getFormat() == OB_FORMAT_Y12C4) {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_16UC1, videoFrame->getData());
            float   scale = videoFrame->as<ob::DepthFrame>()->getValueScale();

            cv::Mat cvtMat;
            rawMat.convertTo(cvtMat, CV_32F, scale * 0.032f);
            cv::pow(cvtMat, 0.6f, cvtMat);
            cvtMat.convertTo(cvtMat, CV_8UC1, 10);
            cv::applyColorMap(cvtMat, rstMat, cv::COLORMAP_JET);
        }
    } break;
    case OB_FRAME_IR:
    case OB_FRAME_IR_LEFT:
    case OB_FRAME_IR_RIGHT: {
        auto videoFrame = frame->as<const ob::VideoFrame>();
        if(videoFrame->getFormat() == OB_FORMAT_Y16) {
            cv::Mat cvtMat;
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_16UC1, videoFrame->getData());
            rawMat.convertTo(cvtMat, CV_8UC1, 1.0 / 16.0f);
            cv::cvtColor(cvtMat, rstMat, cv::COLOR_GRAY2BGR);
        }
        else if(videoFrame->getFormat() == OB_FORMAT_Y8) {
            cv::Mat rawMat(static_cast<int>(videoFrame->getHeight()), static_cast<int>(videoFrame->getWidth()), CV_8UC1, videoFrame->getData());
            cv::cvtColor(rawMat, rstMat, cv::COLOR_GRAY2BGR);
        }
        else if(videoFrame->getFormat() == OB_FORMAT_MJPG) {
            cv::Mat rawMat(1, static_cast<int>(videoFrame->getDataSize()), CV_8UC1, videoFrame->getData());
            rstMat = cv::imdecode(rawMat, 1);
            cv::cvtColor(rstMat, rstMat, cv::COLOR_GRAY2BGR);
        }
    } break;
    default:
        break;
    }
    return rstMat;
}

}  // namespace sync_app
