// Copyright (c) Orbbec Inc. All Rights Reserved.
// Licensed under the MIT License.

#include <libobsensor/ObSensor.hpp>
#include "libobsensor/hpp/Filter.hpp"
#include "libobsensor/hpp/Utils.hpp"
#include <opencv2/opencv.hpp>
#include "utils/cJSON.h"
#include "utils/utils_opencv.hpp"

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/memory.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/SVD>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

enum class StreamType { Color, Depth, IR, PointCloud };

static std::string streamTypeToString(StreamType t) {
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

static std::optional<StreamType> streamTypeFromString(const std::string &s) {
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

static OBMultiDeviceSyncMode stringToOBSyncMode(const std::string &modeString) {
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

static OBFormat stringToOBFormat(const std::string &formatString, StreamType type) {
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
    return OB_FORMAT_Y16;
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

struct StreamConfig {
    StreamType   type;
    bool         enable = true;
    int          width  = 0;
    int          height = 0;
    int          fps    = 0;
    std::string  format;
};

struct DeviceConfig {
    std::string            sn;
    std::string            index;
    OBMultiDeviceSyncConfig syncConfig{};
    bool                    hasSyncConfig = false;
    std::vector<StreamConfig> streams;
};

struct SaveOptions {
    std::string colorExt        = "jpg";
    int         jpegQuality     = 90;
    int         pngCompression  = 1;
};

struct ChessboardConfig {
    int   cols = 9;
    int   rows = 6;
    float squareSize = 0.025f;
};

struct CalibrationConfig {
    ChessboardConfig chessboard;
};

struct DepthPointCloudFiltersConfig {
    std::string preset;
    int    pointCloudDecimationFactor = 0;
    int    decimationFilterScale      = 0;

    int    noiseRemovalMaxSize        = 0;
    int    noiseRemovalMinDiff        = 0;

    double spatialAlpha               = 0.0;
    int    spatialDispDiff            = 0;
    int    spatialMagnitude           = 0;
    int    spatialRadius              = 0;
    double smoothThresholdM           = 0.0;

    double temporalDiffScale          = 0.0;
    double temporalWeight             = 0.0;

    int    holeFillingMode            = 0;
    double confThreshold              = 0.0;
    bool   deskCrop                   = false;
};

struct AppConfig {
    fs::path              outputDir;
    int                   durationSec   = 0;
    int                   maxFrames     = 0;
    int                   collectFps    = 0;
    std::string           mode;
    int                   viewerFps     = 30;
    std::string           initExtrinsicPath;
    float                 maxDepth = 6.0f;
    bool                  differentColor = false;
    CalibrationConfig     calibration;
    bool                  enableSync    = true;
    int                   queueCapacity = 512;
    int                   writerThreads = 0;
    SaveOptions           save;
    DepthPointCloudFiltersConfig filters;
    std::vector<DeviceConfig> devices;
};

static std::string normalizeMode(std::string s) {
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

static std::string trimString(std::string s) {
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

static std::string normalizePresetKey(std::string s) {
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

static bool isViewerMode(const AppConfig &cfg) {
    return cfg.mode == "viewer" || cfg.mode == "interaction";
}

static bool isInteractionMode(const AppConfig &cfg) {
    return cfg.mode == "interaction";
}

static AppConfig loadConfig(const fs::path &configPath) {
    auto content = readFileAll(configPath);
    cJSON *root  = cJSON_Parse(content.c_str());
    if(!root) {
        throw std::runtime_error("Invalid JSON in config file: " + configPath.string());
    }

    AppConfig cfg;

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
        fs::path p = fs::path(*v);
        if(p.is_relative()) {
            const fs::path base = fs::absolute(configPath).parent_path();
            p = (base / p).lexically_normal();
        }
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
                }
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
        }
    }

    auto *devices = cJSON_GetObjectItemCaseSensitive(root, "devices");
    if(!devices || !cJSON_IsArray(devices)) {
        cJSON_Delete(root);
        throw std::runtime_error("Missing or invalid 'devices' in config");
    }

    cJSON *dev = nullptr;
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

    if(cfg.mode != "viewer" && cfg.mode != "interaction" && cfg.mode != "collection" && cfg.mode != "calibration") {
        throw std::runtime_error("Invalid mode in config: " + cfg.mode + ", expected viewer/interaction/collection/calibration");
    }
    return cfg;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr denoiseCloudSor(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                                                           int meanK,
                                                           double stddevMulThresh) {
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

struct OrbbecDepthFilterChain {
    std::shared_ptr<ob::DecimationFilter>     decimation;
    std::shared_ptr<ob::ThresholdFilter>      threshold;
    std::shared_ptr<ob::NoiseRemovalFilter>   noiseRemoval;
    std::shared_ptr<ob::SpatialAdvancedFilter> spatial;
    std::shared_ptr<ob::TemporalFilter>       temporal;
    std::shared_ptr<ob::HoleFillingFilter>    holeFilling;
};

static void applyEdgeSmoothing(std::shared_ptr<ob::Frame> &frame, double thresholdM) {
    if(!frame || !(thresholdM > 0.0)) {
        return;
    }

    auto depth = frame->as<ob::DepthFrame>();
    if(!depth) {
        return;
    }

    if(depth->getFormat() != OB_FORMAT_Y16) {
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
        // Ignore OpenCV errors to prevent crash
    }
}

static std::shared_ptr<ob::Frame> refineDepthFrameForPointCloud(const std::shared_ptr<ob::Frame> &depthFrame,
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

static pcl::PointCloud<pcl::PointXYZ>::Ptr removeDominantPlaneRansac(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                                                                     int maxIterations,
                                                                     double distanceThreshold,
                                                                     double minInlierRatio,
                                                                     int minInliersAbs) {
    auto out = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if(!in || in->empty()) {
        return out;
    }
    if(maxIterations <= 0 || distanceThreshold <= 0.0 || minInlierRatio <= 0.0) {
        *out = *in;
        return out;
    }
    if(in->size() < static_cast<size_t>(std::max(50, minInliersAbs))) {
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

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    seg.segment(*inliers, *coefficients);

    const int minInliers = std::max(minInliersAbs, static_cast<int>(static_cast<double>(in->size()) * minInlierRatio));
    if(!inliers || static_cast<int>(inliers->indices.size()) < minInliers) {
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

struct SaveTask {
    StreamType                    type;
    std::string                   deviceSn;
    uint64_t                      timestampUs = 0;
    std::shared_ptr<ob::Frame>    frame;
    std::shared_ptr<ob::FrameSet> frameSet;
};

class AsyncSaver {
public:
    AsyncSaver(fs::path baseDir,
               SaveOptions options,
               int queueCapacity,
               int writerThreads,
               float maxDepthM,
               DepthPointCloudFiltersConfig filters)
        : baseDir_(std::move(baseDir)),
          options_(std::move(options)),
          queueCapacity_(queueCapacity),
          writerThreads_(writerThreads),
          maxDepthM_(maxDepthM),
          filters_(std::move(filters)) {}

    void prepareDirectories(const std::vector<DeviceConfig> &devices) {
        fs::create_directories(baseDir_);
        for(const auto &dev: devices) {
            for(const auto &st: dev.streams) {
                if(!st.enable) {
                    continue;
                }
                fs::create_directories(baseDir_ / dev.sn / streamTypeToString(st.type));
            }
        }
    }

    void start() {
        if(writerThreads_ <= 0) {
            writerThreads_ = std::max(1u, std::thread::hardware_concurrency());
        }
        running_.store(true);
        for(int i = 0; i < writerThreads_; i++) {
            workers_.emplace_back([this]() { workerLoop(); });
        }
    }

    void stop() {
        running_.store(false);
        cv_.notify_all();
        for(auto &t: workers_) {
            if(t.joinable()) {
                t.join();
            }
        }
        workers_.clear();
    }

    void push(SaveTask task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(static_cast<int>(queue_.size()) >= queueCapacity_) {
                queue_.pop_front();
                dropped_.fetch_add(1);
            }
            queue_.push_back(std::move(task));
        }
        cv_.notify_one();
    }

    uint64_t droppedCount() const { return dropped_.load(); }

    ~AsyncSaver() { stop(); }

private:
    void workerLoop() {
        while(true) {
            SaveTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return !queue_.empty() || !running_.load(); });
                if(queue_.empty() && !running_.load()) {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            saveTask(task);
        }
    }

    void saveTask(const SaveTask &task) {
        try {
            const auto typeName = streamTypeToString(task.type);
            const fs::path dir  = baseDir_ / task.deviceSn / typeName;

            std::string ext;
            if(task.type == StreamType::Color) {
                ext = options_.colorExt;
                if(ext.empty()) {
                    ext = "jpg";
                }
                if(ext[0] != '.') {
                    ext = "." + ext;
                }
            }
            else {
                ext = (task.type == StreamType::PointCloud) ? ".ply" : ".png";
            }

            const auto fileName =
                typeName + "_" + std::to_string(task.timestampUs) + ext;
            const fs::path fullPath = dir / fileName;

            if(task.type == StreamType::Color) {
                saveColor(task.frame, fullPath.string());
            }
            else if(task.type == StreamType::Depth) {
                saveY16(task.frame, fullPath.string());
            }
            else if(task.type == StreamType::IR) {
                saveY16(task.frame, fullPath.string());
            }
            else if(task.type == StreamType::PointCloud) {
                savePointCloud(task.deviceSn, task.frameSet, fullPath.string());
            }
        }
        catch(const std::exception &e) {
            std::lock_guard<std::mutex> lock(errMutex_);
            std::cerr << "Save error: " << e.what() << std::endl;
        }
    }

    void saveColor(const std::shared_ptr<ob::Frame> &frame, const std::string &path) {
        auto colorFrame = frame->as<ob::ColorFrame>();
        if(!colorFrame) {
            return;
        }

        thread_local auto converter = std::make_shared<ob::FormatConvertFilter>();
        if(colorFrame->format() != OB_FORMAT_BGR) {
            if(colorFrame->format() != OB_FORMAT_RGB) {
                if(colorFrame->format() == OB_FORMAT_MJPG) {
                    converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
                }
                else if(colorFrame->format() == OB_FORMAT_UYVY) {
                    converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
                }
                else if(colorFrame->format() == OB_FORMAT_YUYV) {
                    converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
                }
                else {
                    return;
                }
                colorFrame = converter->process(colorFrame)->as<ob::ColorFrame>();
                if(!colorFrame) {
                    return;
                }
            }
            converter->setFormatConvertType(FORMAT_RGB_TO_BGR);
            colorFrame = converter->process(colorFrame)->as<ob::ColorFrame>();
            if(!colorFrame) {
                return;
            }
        }

        const int width  = static_cast<int>(colorFrame->width());
        const int height = static_cast<int>(colorFrame->height());
        cv::Mat bgr(height, width, CV_8UC3, const_cast<void *>(colorFrame->data()));

        std::vector<int> params;
        if(endsWith(path, ".jpg") || endsWith(path, ".jpeg")) {
            params = { cv::IMWRITE_JPEG_QUALITY, options_.jpegQuality };
        }
        else if(endsWith(path, ".png")) {
            params = { cv::IMWRITE_PNG_COMPRESSION, options_.pngCompression };
        }
        cv::imwrite(path, bgr, params);
    }

    void saveY16(const std::shared_ptr<ob::Frame> &frame, const std::string &path) {
        auto vf = frame->as<ob::VideoFrame>();
        if(!vf) {
            return;
        }
        const int width  = static_cast<int>(vf->width());
        const int height = static_cast<int>(vf->height());
        cv::Mat img(height, width, CV_16UC1, const_cast<void *>(vf->data()));
        std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, options_.pngCompression };
        cv::imwrite(path, img, params);
    }

    void savePointCloud(const std::string &deviceSn, const std::shared_ptr<ob::FrameSet> &frameSet, const std::string &path) {
        if(!frameSet) {
            return;
        }
        auto depthFrame = frameSet->depthFrame();
        if(!depthFrame) {
            return;
        }

        std::lock_guard<std::mutex> lock(filterStateMutex_);
        auto itPc = pointCloudFilters_.find(deviceSn);
        if(itPc == pointCloudFilters_.end()) {
            auto pc = std::make_shared<ob::PointCloudFilter>();
            pc->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
            pc->setDecimationFactor(filters_.pointCloudDecimationFactor > 0 ? filters_.pointCloudDecimationFactor : 1);
            itPc = pointCloudFilters_.emplace(deviceSn, std::move(pc)).first;
        }

        const bool enableDepthRefine =
            (filters_.decimationFilterScale > 0) || (filters_.noiseRemovalMaxSize > 0) || (filters_.noiseRemovalMinDiff > 0) || (filters_.spatialAlpha > 0.0)
            || (filters_.spatialDispDiff > 0) || (filters_.spatialMagnitude > 0) || (filters_.spatialRadius > 0) || (filters_.temporalDiffScale > 0.0)
            || (filters_.temporalWeight > 0.0) || (filters_.holeFillingMode > 0);

        std::shared_ptr<ob::Frame> pcFrame;
        if(enableDepthRefine) {
            itPc->second->setCreatePointFormat(OB_FORMAT_POINT);
            std::shared_ptr<ob::Frame> depthForPc = depthFrame;

            auto itChain = depthFilterChains_.find(deviceSn);
            if(itChain == depthFilterChains_.end()) {
                itChain = depthFilterChains_.emplace(deviceSn, OrbbecDepthFilterChain{}).first;
            }
            depthForPc = refineDepthFrameForPointCloud(depthForPc, itChain->second, 0.2f, maxDepthM_, filters_);
            if(!depthForPc) {
                return;
            }
            pcFrame = itPc->second->process(depthForPc);
        }
        else {
            thread_local auto align = std::make_shared<ob::Align>(OB_STREAM_COLOR);
            std::shared_ptr<ob::Frame> aligned = frameSet;
            auto colorFrame = frameSet->colorFrame();
            if(colorFrame) {
                aligned = align->process(frameSet);
                itPc->second->setCreatePointFormat(OB_FORMAT_RGB_POINT);
            }
            else {
                itPc->second->setCreatePointFormat(OB_FORMAT_POINT);
            }
            pcFrame = itPc->second->process(aligned);
        }
        if(!pcFrame) {
            return;
        }
        ob::PointCloudHelper::savePointcloudToPly(path.c_str(), pcFrame, false, false, 50);
    }

    static bool endsWith(const std::string &s, const std::string &suffix) {
        if(s.size() < suffix.size()) {
            return false;
        }
        return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(), [](char a, char b) {
            return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
        });
    }

private:
    fs::path                     baseDir_;
    SaveOptions                  options_;
    int                          queueCapacity_;
    int                          writerThreads_;

    std::atomic<bool>            running_{false};
    std::mutex                   mutex_;
    std::condition_variable      cv_;
    std::deque<SaveTask>         queue_;
    std::vector<std::thread>     workers_;
    std::atomic<uint64_t>        dropped_{0};

    std::mutex                   filterStateMutex_;
    float                        maxDepthM_ = 0.0f;
    DepthPointCloudFiltersConfig filters_;
    std::unordered_map<std::string, std::shared_ptr<ob::PointCloudFilter>> pointCloudFilters_;
    std::unordered_map<std::string, OrbbecDepthFilterChain>                depthFilterChains_;

    std::mutex                   errMutex_;
};

class MultiDeviceRecorder {
public:
    explicit MultiDeviceRecorder(AppConfig cfg)
        : cfg_(std::move(cfg)),
          saver_(cfg_.outputDir, cfg_.save, cfg_.queueCapacity, cfg_.writerThreads, cfg_.maxDepth, cfg_.filters) {}

    int run() {
        auto deviceList = ctx_.queryDeviceList();
        if(deviceList->deviceCount() == 0) {
            std::cerr << "No devices found" << std::endl;
            return 1;
        }

        auto selected = selectDevices(deviceList);
        if(selected.empty()) {
            std::cerr << "No configured devices found in current device list" << std::endl;
            return 1;
        }

        if(cfg_.enableSync) {
            applySyncConfig(selected);
        }

        saver_.prepareDirectories(cfg_.devices);
        saver_.start();

        std::vector<DeviceRuntime> primary;
        std::vector<DeviceRuntime> secondary;
        splitPrimarySecondary(selected, primary, secondary);

        startPipelines(secondary);
        startPipelines(primary);

        if(cfg_.enableSync) {
            ctx_.enableDeviceClockSync(60000);
        }

        std::thread triggerThread;
        if(cfg_.enableSync && hasSoftwareTrigger(selected)) {
            triggerThread = std::thread([this, &selected]() { triggerLoop(selected); });
        }

        auto startTs = std::chrono::steady_clock::now();
        uint64_t lastReportSec = 0;
        while(true) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedSec =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(now - startTs).count());

            if(cfg_.durationSec > 0 && static_cast<int>(elapsedSec) >= cfg_.durationSec) {
                break;
            }
            if(cfg_.maxFrames > 0 && savedFrames_.load() >= cfg_.maxFrames) {
                break;
            }
            if(elapsedSec / 5 > lastReportSec) {
                lastReportSec = elapsedSec / 5;
                std::cout << "Progress: " << elapsedSec << "s, savedFrames=" << savedFrames_.load()
                          << ", droppedWriteTasks=" << saver_.droppedCount() << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        stopping_.store(true);
        if(triggerThread.joinable()) {
            triggerThread.join();
        }

        stopPipelines(primary);
        stopPipelines(secondary);
        saver_.stop();
        return 0;
    }

private:
    struct DeviceRuntime {
        DeviceConfig              cfg;
        std::shared_ptr<ob::Device>   dev;
        std::shared_ptr<ob::Pipeline> pipe;
    };

    std::vector<DeviceRuntime> selectDevices(const std::shared_ptr<ob::DeviceList> &deviceList) {
        std::unordered_map<std::string, std::shared_ptr<ob::Device>> bySn;
        for(uint32_t i = 0; i < deviceList->deviceCount(); i++) {
            auto dev  = deviceList->getDevice(i);
            auto sn   = dev->getDeviceInfo()->serialNumber();
            bySn.emplace(std::string(sn), dev);
        }

        std::vector<DeviceRuntime> out;
        out.reserve(cfg_.devices.size());
        for(const auto &dc: cfg_.devices) {
            auto it = bySn.find(dc.sn);
            if(it == bySn.end()) {
                std::cerr << "Configured device not found: " << dc.sn << std::endl;
                continue;
            }
            DeviceRuntime rt;
            rt.cfg  = dc;
            rt.dev  = it->second;
            rt.pipe = std::make_shared<ob::Pipeline>(rt.dev);
            out.push_back(std::move(rt));
        }
        return out;
    }

    void applySyncConfig(std::vector<DeviceRuntime> &devices) {
        for(size_t i = 0; i < devices.size(); i++) {
            auto &rt = devices[i];
            auto cur = rt.dev->getMultiDeviceSyncConfig();
            auto cfg = rt.cfg.hasSyncConfig ? rt.cfg.syncConfig : cur;
            if(!rt.cfg.hasSyncConfig) {
                cfg.syncMode = (i == 0) ? OB_MULTI_DEVICE_SYNC_MODE_PRIMARY : OB_MULTI_DEVICE_SYNC_MODE_SECONDARY;
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
        }
    }

    void splitPrimarySecondary(const std::vector<DeviceRuntime> &all,
                               std::vector<DeviceRuntime> &primary,
                               std::vector<DeviceRuntime> &secondary) {
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

    static bool hasSoftwareTrigger(const std::vector<DeviceRuntime> &all) {
        for(const auto &rt: all) {
            auto cfg = rt.dev->getMultiDeviceSyncConfig();
            if(cfg.syncMode == OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING) {
                return true;
            }
        }
        return false;
    }

    void triggerLoop(std::vector<DeviceRuntime> &devices) {
        int fps = cfg_.collectFps;
        if(fps <= 0) {
            for(const auto &rt: devices) {
                auto cfg = rt.dev->getMultiDeviceSyncConfig();
                if(cfg.syncMode != OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING) {
                    continue;
                }
                for(const auto &sc: rt.cfg.streams) {
                    if(sc.enable && sc.fps > 0) {
                        fps = sc.fps;
                        break;
                    }
                }
                if(fps > 0) {
                    break;
                }
            }
            if(fps <= 0) {
                fps = 30;
            }
        }
        const auto interval = std::chrono::microseconds(static_cast<int64_t>(1000000.0 / fps));
        while(!stopping_.load()) {
            for(auto &rt: devices) {
                auto cfg = rt.dev->getMultiDeviceSyncConfig();
                if(cfg.syncMode == OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING) {
                    rt.dev->triggerCapture();
                }
            }
            std::this_thread::sleep_for(interval);
        }
    }

    std::shared_ptr<ob::VideoStreamProfile> pickProfile(const std::shared_ptr<ob::Pipeline> &pipe,
                                                        OBSensorType sensorType,
                                                        const StreamConfig &sc) {
        auto list = pipe->getStreamProfileList(sensorType);
        if(!list || list->getCount() == 0) {
            return nullptr;
        }

        const auto format = stringToOBFormat(sc.format, sc.type);
        try {
            if(sc.width > 0 || sc.height > 0 || sc.fps > 0) {
                auto profile = list->getVideoStreamProfile(sc.width > 0 ? sc.width : OB_WIDTH_ANY,
                                                           sc.height > 0 ? sc.height : OB_HEIGHT_ANY,
                                                           format != OB_FORMAT_UNKNOWN ? format : OB_FORMAT_ANY,
                                                           sc.fps > 0 ? sc.fps : OB_FPS_ANY);
                if(profile) {
                    return profile;
                }
            }
        }
        catch(...) {
        }

        for(uint32_t i = 0; i < list->getCount(); i++) {
            auto p  = list->getProfile(i);
            auto vp = p->as<ob::VideoStreamProfile>();
            if(!vp) {
                continue;
            }
            if(sc.width > 0 && static_cast<int>(vp->getWidth()) != sc.width) {
                continue;
            }
            if(sc.height > 0 && static_cast<int>(vp->getHeight()) != sc.height) {
                continue;
            }
            if(sc.fps > 0 && static_cast<int>(vp->getFps()) != sc.fps) {
                continue;
            }
            if(format != OB_FORMAT_UNKNOWN && p->getFormat() != format) {
                continue;
            }
            return vp;
        }

        return list->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
    }

    void startPipelines(std::vector<DeviceRuntime> &devices) {
        for(auto &rt: devices) {
            auto config = std::make_shared<ob::Config>();
            std::unordered_set<OBSensorType> enabledSensors;
            for(const auto &sc: rt.cfg.streams) {
                if(!sc.enable) {
                    continue;
                }
                OBSensorType sensorType;
                switch(sc.type) {
                case StreamType::Color:
                    sensorType = OB_SENSOR_COLOR;
                    break;
                case StreamType::Depth:
                    sensorType = OB_SENSOR_DEPTH;
                    break;
                case StreamType::IR:
                    sensorType = OB_SENSOR_IR;
                    break;
                case StreamType::PointCloud:
                    sensorType = OB_SENSOR_DEPTH;
                    break;
                }

                if(enabledSensors.find(sensorType) != enabledSensors.end()) {
                    continue;
                }
                auto profile = pickProfile(rt.pipe, sensorType, sc);
                if(profile) {
                    config->enableStream(profile);
                    enabledSensors.insert(sensorType);
                }
            }

            const auto deviceSn = rt.cfg.sn;
            rt.pipe->start(config, [this, deviceSn, streams = rt.cfg.streams](std::shared_ptr<ob::FrameSet> frameSet) {
                onFrameSet(deviceSn, streams, frameSet);
            });
        }
    }

    void stopPipelines(std::vector<DeviceRuntime> &devices) {
        for(auto &rt: devices) {
            try {
                rt.pipe->stop();
            }
            catch(...) {
            }
        }
    }

    void onFrameSet(const std::string &deviceSn,
                    const std::vector<StreamConfig> &streams,
                    const std::shared_ptr<ob::FrameSet> &frameSet) {
        if(!frameSet) {
            return;
        }

        for(const auto &sc: streams) {
            if(!sc.enable) {
                continue;
            }
            uint64_t ts = 0;
            std::shared_ptr<ob::Frame> frame;

            if(sc.type == StreamType::PointCloud) {
                auto depth = frameSet->depthFrame();
                if(depth) {
                    try {
                        ts = depth->globalTimeStampUs();
                    }
                    catch(...) {
                    }
                    if(ts == 0) {
                        ts = depth->timeStampUs();
                    }
                }
            }
            else {
                if(sc.type == StreamType::Color) {
                    frame = frameSet->colorFrame();
                }
                else if(sc.type == StreamType::Depth) {
                    frame = frameSet->depthFrame();
                }
                else if(sc.type == StreamType::IR) {
                    frame = frameSet->irFrame();
                }
                if(!frame) {
                    continue;
                }
                try {
                    ts = frame->globalTimeStampUs();
                }
                catch(...) {
                }
                if(ts == 0) {
                    ts = frame->timeStampUs();
                }
            }

            if(ts == 0) {
                ts = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count());
            }

            if(cfg_.collectFps > 0) {
                const uint64_t interval = static_cast<uint64_t>(1000000.0 / cfg_.collectFps);
                const auto key          = deviceSn + ":" + streamTypeToString(sc.type);
                {
                    std::lock_guard<std::mutex> lock(rateMutex_);
                    const auto last = lastSavedTs_.find(key);
                    if(last != lastSavedTs_.end() && ts > last->second && (ts - last->second) < interval) {
                        continue;
                    }
                    lastSavedTs_[key] = ts;
                }
            }

            if(sc.type == StreamType::PointCloud) {
                saver_.push(SaveTask{ sc.type, deviceSn, ts, nullptr, frameSet });
            }
            else {
                saver_.push(SaveTask{ sc.type, deviceSn, ts, frame, nullptr });
            }
            savedFrames_.fetch_add(1);
        }
    }

private:
    AppConfig                             cfg_;
    ob::Context                           ctx_;
    AsyncSaver                            saver_;
    std::atomic<bool>                     stopping_{false};
    std::atomic<int64_t>                  savedFrames_{0};
    std::mutex                            rateMutex_;
    std::unordered_map<std::string, uint64_t> lastSavedTs_;
};

class MultiDeviceViewer {
public:
    explicit MultiDeviceViewer(AppConfig cfg)
        : cfg_(std::move(cfg)) {}

    int run() {
        auto deviceList = ctx_.queryDeviceList();
        if(deviceList->deviceCount() == 0) {
            std::cerr << "No devices found" << std::endl;
            return 1;
        }

        auto selected = selectDevices(deviceList);
        if(selected.empty()) {
            std::cerr << "No configured devices found in current device list" << std::endl;
            return 1;
        }

        if(isInteractionMode(cfg_)) {
            loadInitExtrinsicsIfNeeded();
        }

        if(cfg_.enableSync) {
            applySyncConfig(selected);
        }

        std::vector<DeviceRuntime> primary;
        std::vector<DeviceRuntime> secondary;
        splitPrimarySecondary(selected, primary, secondary);

        startPipelines(secondary);
        startPipelines(primary);

        if(cfg_.enableSync) {
            ctx_.enableDeviceClockSync(60000);
        }

        int fps = cfg_.viewerFps;
        if(fps <= 0) {
            fps = 30;
        }
        viewerIntervalUs_ = static_cast<uint64_t>(1000000.0 / fps);

        if(!isInteractionMode(cfg_)) {
            ob_smpl::CVWindow win("PointCloudViewer", 1600, 900, ob_smpl::ARRANGE_GRID);
            win.setShowInfo(true);
            win.setShowSyncTimeInfo(true);

            winPtr_ = &win;
            while(win.run()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            winPtr_ = nullptr;
        }
        else {
            const std::string winName = "PointCloudViewer";
            int               winW    = 1600;
            int               winH    = 900;

            InteractiveViewState viewState;
            viewState.windowName = winName;
            viewState.width      = winW;
            viewState.height     = winH;

            uint64_t lastRenderUs = 0;
            bool exiting = false;
            while(!exiting) {
                if(cv::getWindowProperty(winName, cv::WND_PROP_VISIBLE) < 1) {
                    cv::namedWindow(winName, cv::WINDOW_NORMAL);
                    cv::resizeWindow(winName, winW, winH);
                    cv::setMouseCallback(winName, &MultiDeviceViewer::mouseCallbackThunk, &viewState);
                }

                const auto nowUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                             std::chrono::steady_clock::now().time_since_epoch())
                                                             .count());
                if(lastRenderUs == 0 || nowUs - lastRenderUs >= viewerIntervalUs_) {
                    lastRenderUs = nowUs;
                    auto frameMap = snapshotLatestDepthFrames();
                    auto canvas   = renderUnifiedPointCloud(frameMap, viewState);
                    cv::imshow(winName, canvas);
                }

                const int key = cv::waitKey(1);
                if(key == 'r' || key == 'R') {
                    viewState.resetView();
                }
                else if(key == 'q' || key == 'Q') {
                    exiting = true;
                }
            }
        }

        stopPipelines(primary);
        stopPipelines(secondary);
        return 0;
    }

private:
    struct DeviceRuntime {
        DeviceConfig                cfg;
        std::shared_ptr<ob::Device> dev;
        std::shared_ptr<ob::Pipeline> pipe;
        int                         deviceIndex = 0;
    };

    std::vector<DeviceRuntime> selectDevices(const std::shared_ptr<ob::DeviceList> &deviceList) {
        std::unordered_map<std::string, std::shared_ptr<ob::Device>> bySn;
        for(uint32_t i = 0; i < deviceList->deviceCount(); i++) {
            auto dev = deviceList->getDevice(i);
            bySn.emplace(std::string(dev->getDeviceInfo()->serialNumber()), dev);
        }

        std::vector<DeviceRuntime> out;
        out.reserve(cfg_.devices.size());
        int index = 0;
        for(const auto &dc: cfg_.devices) {
            auto it = bySn.find(dc.sn);
            if(it == bySn.end()) {
                std::cerr << "Configured device not found: " << dc.sn << std::endl;
                continue;
            }
            DeviceRuntime rt;
            rt.cfg         = dc;
            rt.dev         = it->second;
            rt.pipe        = std::make_shared<ob::Pipeline>(rt.dev);
            rt.deviceIndex = index++;
            out.push_back(std::move(rt));
        }
        return out;
    }

    void applySyncConfig(std::vector<DeviceRuntime> &devices) {
        for(size_t i = 0; i < devices.size(); i++) {
            auto &rt = devices[i];
            auto cur = rt.dev->getMultiDeviceSyncConfig();
            auto cfg = rt.cfg.hasSyncConfig ? rt.cfg.syncConfig : cur;
            if(!rt.cfg.hasSyncConfig) {
                cfg.syncMode = (i == 0) ? OB_MULTI_DEVICE_SYNC_MODE_PRIMARY : OB_MULTI_DEVICE_SYNC_MODE_SECONDARY;
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
        }
    }

    void splitPrimarySecondary(const std::vector<DeviceRuntime> &all,
                               std::vector<DeviceRuntime> &primary,
                               std::vector<DeviceRuntime> &secondary) {
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

    static OBFormat pointCloudSourceFormat(const StreamConfig &sc) {
        (void)sc;
        return OB_FORMAT_Y16;
    }

    std::shared_ptr<ob::VideoStreamProfile> pickDepthProfileForPointCloud(const std::shared_ptr<ob::Pipeline> &pipe,
                                                                          const std::vector<StreamConfig> &streams) {
        StreamConfig pointCloudSc{};
        bool found = false;
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
            const auto fmt = pointCloudSourceFormat(pointCloudSc);
            try {
                auto profile = list->getVideoStreamProfile(pointCloudSc.width > 0 ? pointCloudSc.width : OB_WIDTH_ANY,
                                                           pointCloudSc.height > 0 ? pointCloudSc.height : OB_HEIGHT_ANY,
                                                           fmt,
                                                           pointCloudSc.fps > 0 ? pointCloudSc.fps : OB_FPS_ANY);
                if(profile) {
                    return profile;
                }
            }
            catch(...) {
            }
        }

        return list->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
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
        try {
            return list->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
        }
        catch(...) {
            return nullptr;
        }
    }

    void startPipelines(std::vector<DeviceRuntime> &devices) {
        for(auto &rt: devices) {
            if(isInteractionMode(cfg_)) {
                const std::string preset = cfg_.filters.preset;
                if(!preset.empty() && preset != "0") {
                    try {
                        const auto list = rt.dev ? rt.dev->getAvailablePresetList() : nullptr;
                        if(list && list->getCount() > 0) {
                            const std::string want = normalizePresetKey(preset);
                            const char *match      = nullptr;
                            for(uint32_t i = 0; i < list->getCount(); i++) {
                                const char *name = list->getName(i);
                                if(!name) {
                                    continue;
                                }
                                if(normalizePresetKey(std::string(name)) == want) {
                                    match = name;
                                    break;
                                }
                            }
                            if(match) {
                                rt.dev->loadPreset(match);
                            }
                            else {
                                rt.dev->loadPreset(preset.c_str());
                            }
                        }
                    }
                    catch(...) {
                    }
                }
            }

            auto config = std::make_shared<ob::Config>();
            auto depthProfile = pickDepthProfileForPointCloud(rt.pipe, rt.cfg.streams);
            if(depthProfile) {
                config->enableStream(depthProfile);
            }

            const bool wantIrMask = isInteractionMode(cfg_) && cfg_.filters.confThreshold > 0.0;
            if(wantIrMask) {
                config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);

                auto irL = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_LEFT);
                auto irR = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_RIGHT);
                if(irL && irR) {
                    config->enableStream(irL);
                    config->enableStream(irR);
                }
                else {
                    auto ir = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR);
                    if(ir) {
                        config->enableStream(ir);
                    }
                }
            }

            const auto deviceSn    = rt.cfg.sn;
            const auto deviceIndex = rt.deviceIndex;
            const auto camIndex    = rt.cfg.index;
            rt.pipe->start(config, [this, deviceSn, camIndex, deviceIndex](std::shared_ptr<ob::FrameSet> frameSet) {
                onFrameSet(deviceSn, camIndex, deviceIndex, frameSet);
            });
        }
    }

    void stopPipelines(std::vector<DeviceRuntime> &devices) {
        for(auto &rt: devices) {
            try {
                rt.pipe->stop();
            }
            catch(...) {
            }
        }
    }

    void onFrameSet(const std::string &deviceSn,
                    const std::string &camIndex,
                    int deviceIndex,
                    const std::shared_ptr<ob::FrameSet> &frameSet) {
        if(!frameSet) {
            return;
        }
        auto depth = frameSet->depthFrame();
        if(!depth) {
            return;
        }
        uint64_t ts = 0;
        try {
            ts = depth->globalTimeStampUs();
        }
        catch(...) {
        }
        if(ts == 0) {
            ts = depth->timeStampUs();
        }
        if(ts == 0) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(viewerMutex_);
            const auto last = lastShownTs_.find(deviceIndex);
            if(last != lastShownTs_.end() && ts > last->second && (ts - last->second) < viewerIntervalUs_) {
                return;
            }
            lastShownTs_[deviceIndex] = ts;
        }

        auto *win = winPtr_.load();
        if(!isInteractionMode(cfg_)) {
            if(!win) {
                return;
            }
            win->pushFramesToView(depth, deviceIndex);
        }
        else {
            std::shared_ptr<ob::IRFrame> irLeft;
            std::shared_ptr<ob::IRFrame> irRight;
            if(cfg_.filters.confThreshold > 0.0) {
                try {
                    auto fL = frameSet->getFrame(OB_FRAME_IR_LEFT);
                    if(fL) {
                        irLeft = fL->as<ob::IRFrame>();
                    }
                }
                catch(...) {
                }
                try {
                    auto fR = frameSet->getFrame(OB_FRAME_IR_RIGHT);
                    if(fR) {
                        irRight = fR->as<ob::IRFrame>();
                    }
                }
                catch(...) {
                }
                if(!irLeft || !irRight) {
                    try {
                        auto ir = frameSet->irFrame();
                        if(ir) {
                            irLeft  = ir;
                            irRight = ir;
                        }
                    }
                    catch(...) {
                    }
                }
            }

            cacheLatestFrame(deviceIndex, deviceSn, camIndex, ts, depth, irLeft, irRight);
        }
    }

private:
    struct ExtrinsicCamToWorld {
        bool       valid = false;
        cv::Matx33f R    = cv::Matx33f::eye();
        cv::Vec3f   t{ 0.0f, 0.0f, 0.0f };
    };

    struct CachedStereoFrame {
        uint64_t                          tsUs = 0;
        std::shared_ptr<ob::DepthFrame>   depth;
        std::shared_ptr<ob::IRFrame>      irLeft;
        std::shared_ptr<ob::IRFrame>      irRight;
        std::string                       sn;
        std::string                       camIndex;
    };

    struct InteractiveViewState {
        std::string windowName;
        int         width  = 1600;
        int         height = 900;

        bool     rotating   = false;
        bool     panning    = false;
        cv::Point lastMouse = { 0, 0 };

        float yawRad   = 0.0f;
        float pitchRad = 0.0f;
        float distance = 1.5f;
        cv::Vec3f target{ 0.0f, 0.0f, 1.0f };

        void resetView() {
            rotating   = false;
            panning    = false;
            yawRad     = 0.0f;
            pitchRad   = 0.0f;
            distance   = 1.5f;
            target     = cv::Vec3f(0.0f, 0.0f, 1.0f);
            lastMouse  = cv::Point(0, 0);
        }
    };

    static cv::Vec3f normalizeVec3(const cv::Vec3f &v) {
        const float n = std::sqrt(v.dot(v));
        if(n <= 1e-8f) {
            return cv::Vec3f(0, 0, 0);
        }
        return v * (1.0f / n);
    }

    static cv::Vec3f crossVec3(const cv::Vec3f &a, const cv::Vec3f &b) {
        return cv::Vec3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
    }

    static void computeCameraBasis(const InteractiveViewState &s, cv::Vec3f &right, cv::Vec3f &up, cv::Vec3f &forward, cv::Vec3f &camPos) {
        const float cy = std::cos(s.yawRad);
        const float sy = std::sin(s.yawRad);
        const float cp = std::cos(s.pitchRad);
        const float sp = std::sin(s.pitchRad);

        forward = normalizeVec3(cv::Vec3f(sy * cp, -sp, cy * cp));
        const cv::Vec3f worldUp(0.0f, -1.0f, 0.0f);
        right = normalizeVec3(crossVec3(worldUp, forward));
        up    = crossVec3(forward, right);
        camPos = s.target - forward * s.distance;
    }

    static void mouseCallbackThunk(int event, int x, int y, int flags, void *userdata) {
        auto *s = reinterpret_cast<InteractiveViewState *>(userdata);
        if(!s) {
            return;
        }

        if(event == cv::EVENT_LBUTTONDOWN) {
            s->rotating  = true;
            s->panning   = false;
            s->lastMouse = cv::Point(x, y);
            return;
        }
        if(event == cv::EVENT_LBUTTONUP) {
            s->rotating = false;
            return;
        }
        if(event == cv::EVENT_RBUTTONDOWN) {
            s->panning   = true;
            s->rotating  = false;
            s->lastMouse = cv::Point(x, y);
            return;
        }
        if(event == cv::EVENT_RBUTTONUP) {
            s->panning = false;
            return;
        }
        if(event == cv::EVENT_MOUSEWHEEL) {
            const int delta = cv::getMouseWheelDelta(flags);
            if(delta != 0) {
                const float k = std::exp(-static_cast<float>(delta) * 0.001f);
                s->distance   = std::min(20.0f, std::max(0.2f, s->distance * k));
            }
            return;
        }
        if(event == cv::EVENT_MOUSEMOVE) {
            const int dx = x - s->lastMouse.x;
            const int dy = y - s->lastMouse.y;
            s->lastMouse = cv::Point(x, y);
            if(s->rotating) {
                s->yawRad += static_cast<float>(dx) * 0.005f;
                s->pitchRad += static_cast<float>(dy) * 0.005f;
                s->pitchRad = std::min(1.55f, std::max(-1.55f, s->pitchRad));
                return;
            }
            if(s->panning) {
                cv::Vec3f right, up, forward, camPos;
                computeCameraBasis(*s, right, up, forward, camPos);
                const float scale = s->distance * 0.001f;
                s->target -= right * (static_cast<float>(dx) * scale);
                s->target += up * (static_cast<float>(dy) * scale);
                return;
            }
        }
    }

    void cacheLatestFrame(int deviceIndex,
                          const std::string &sn,
                          const std::string &camIndex,
                          uint64_t tsUs,
                          const std::shared_ptr<ob::DepthFrame> &depth,
                          const std::shared_ptr<ob::IRFrame> &irLeft,
                          const std::shared_ptr<ob::IRFrame> &irRight) {
        std::lock_guard<std::mutex> lock(latestDepthMutex_);
        latestDepth_[deviceIndex] = CachedStereoFrame{ tsUs, depth, irLeft, irRight, sn, camIndex };
    }

    std::unordered_map<int, CachedStereoFrame> snapshotLatestDepthFrames() {
        std::lock_guard<std::mutex> lock(latestDepthMutex_);
        return latestDepth_;
    }

    static bool parseVec3(cJSON *arr, cv::Vec3f &out) {
        if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
            return false;
        }
        const auto *a0 = cJSON_GetArrayItem(arr, 0);
        const auto *a1 = cJSON_GetArrayItem(arr, 1);
        const auto *a2 = cJSON_GetArrayItem(arr, 2);
        if(!a0 || !a1 || !a2 || !cJSON_IsNumber(a0) || !cJSON_IsNumber(a1) || !cJSON_IsNumber(a2)) {
            return false;
        }
        out = cv::Vec3f(static_cast<float>(a0->valuedouble), static_cast<float>(a1->valuedouble), static_cast<float>(a2->valuedouble));
        return true;
    }

    static bool parseMat3(cJSON *arr, cv::Matx33f &out) {
        if(!arr || !cJSON_IsArray(arr)) {
            return false;
        }
        const int n = cJSON_GetArraySize(arr);
        if(n == 9) {
            float v[9];
            for(int i = 0; i < 9; i++) {
                auto *it = cJSON_GetArrayItem(arr, i);
                if(!it || !cJSON_IsNumber(it)) {
                    return false;
                }
                v[i] = static_cast<float>(it->valuedouble);
            }
            out = cv::Matx33f(v[0], v[1], v[2],
                              v[3], v[4], v[5],
                              v[6], v[7], v[8]);
            return true;
        }
        if(n == 3) {
            auto *r0 = cJSON_GetArrayItem(arr, 0);
            auto *r1 = cJSON_GetArrayItem(arr, 1);
            auto *r2 = cJSON_GetArrayItem(arr, 2);
            if(r0 && r1 && r2 && cJSON_IsArray(r0) && cJSON_IsArray(r1) && cJSON_IsArray(r2)
               && cJSON_GetArraySize(r0) == 3 && cJSON_GetArraySize(r1) == 3 && cJSON_GetArraySize(r2) == 3) {
                float v[9];
                cJSON *rows[3] = { r0, r1, r2 };
                for(int y = 0; y < 3; y++) {
                    for(int x = 0; x < 3; x++) {
                        auto *it = cJSON_GetArrayItem(rows[y], x);
                        if(!it || !cJSON_IsNumber(it)) {
                            return false;
                        }
                        v[y * 3 + x] = static_cast<float>(it->valuedouble);
                    }
                }
                out = cv::Matx33f(v[0], v[1], v[2],
                                  v[3], v[4], v[5],
                                  v[6], v[7], v[8]);
                return true;
            }
        }
        return false;
    }

    void loadInitExtrinsicsIfNeeded() {
        if(cfg_.initExtrinsicPath.empty()) {
            std::cerr << "mode=interaction but init_extrinsic_path is empty, fallback to default spacing" << std::endl;
            return;
        }
        fs::path p = fs::path(cfg_.initExtrinsicPath);
        std::string content;
        try {
            content = readFileAll(p);
        }
        catch(const std::exception &e) {
            std::cerr << "Failed to read init_extrinsic_path: " << p.string() << ", error=" << e.what() << std::endl;
            return;
        }

        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            std::cerr << "Invalid JSON in init_extrinsic_path: " << p.string() << std::endl;
            return;
        }

        for(cJSON *item = root->child; item != nullptr; item = item->next) {
            if(!item->string || !cJSON_IsObject(item)) {
                continue;
            }
            const std::string camId = item->string;
            auto *rotArr            = cJSON_GetObjectItemCaseSensitive(item, "rotation");
            auto *tArr              = cJSON_GetObjectItemCaseSensitive(item, "translation");
            cv::Vec3f tCw;
            if(!parseVec3(tArr, tCw)) {
                continue;
            }

            cv::Matx33f Rcw = cv::Matx33f::eye();
            if(!parseMat3(rotArr, Rcw)) {
                std::cerr << "Invalid rotation matrix for camera " << camId << " in init_extrinsic_path (expect 3x3 array)" << std::endl;
                continue;
            }

            const cv::Matx33f Rwc = Rcw.t();
            const cv::Vec3f   tWc = -(Rwc * tCw);

            ExtrinsicCamToWorld ex;
            ex.valid = true;
            ex.R     = Rwc;
            ex.t     = tWc;
            extrinsicsCamToWorld_[camId] = ex;
        }

        cJSON_Delete(root);
    }

    static float bilinearSampleNormalized(const std::shared_ptr<ob::IRFrame> &ir, float x, float y) {
        if(!ir) {
            return 0.0f;
        }
        const int w = static_cast<int>(ir->getWidth());
        const int h = static_cast<int>(ir->getHeight());
        if(w <= 1 || h <= 1) {
            return 0.0f;
        }
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        if(x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) {
            return 0.0f;
        }

        const float fx = x - static_cast<float>(x0);
        const float fy = y - static_cast<float>(y0);

        const uint8_t *raw = reinterpret_cast<const uint8_t *>(ir->data());
        if(!raw) {
            return 0.0f;
        }
        const auto dataSize = static_cast<size_t>(ir->dataSize());
        const size_t strideBytes = h > 0 ? (dataSize / static_cast<size_t>(h)) : 0;
        if(strideBytes == 0) {
            return 0.0f;
        }

        const auto fmt = ir->getFormat();
        float maxVal = 255.0f;
        if(fmt != OB_FORMAT_Y8) {
            uint8_t bitSize = 0;
            try {
                bitSize = ir->getPixelAvailableBitSize();
            }
            catch(...) {
            }
            if(bitSize > 0 && bitSize < 16) {
                maxVal = static_cast<float>((1u << bitSize) - 1u);
            }
            else {
                maxVal = 65535.0f;
            }
        }
        if(maxVal <= 0.0f) {
            return 0.0f;
        }

        auto sampleU8 = [&](int sx, int sy) -> float {
            const size_t off = static_cast<size_t>(sy) * strideBytes + static_cast<size_t>(sx);
            if(off >= dataSize) {
                return 0.0f;
            }
            return static_cast<float>(raw[off]);
        };
        auto sampleU16 = [&](int sx, int sy) -> float {
            const size_t off = static_cast<size_t>(sy) * strideBytes + static_cast<size_t>(sx) * sizeof(uint16_t);
            if(off + sizeof(uint16_t) > dataSize) {
                return 0.0f;
            }
            const auto *p = reinterpret_cast<const uint16_t *>(raw + off);
            return static_cast<float>(*p);
        };

        float v00 = 0.0f, v10 = 0.0f, v01 = 0.0f, v11 = 0.0f;
        if(fmt == OB_FORMAT_Y8) {
            v00 = sampleU8(x0, y0);
            v10 = sampleU8(x1, y0);
            v01 = sampleU8(x0, y1);
            v11 = sampleU8(x1, y1);
        }
        else {
            v00 = sampleU16(x0, y0);
            v10 = sampleU16(x1, y0);
            v01 = sampleU16(x0, y1);
            v11 = sampleU16(x1, y1);
        }

        const float v0 = v00 + fx * (v10 - v00);
        const float v1 = v01 + fx * (v11 - v01);
        const float v  = v0 + fy * (v1 - v0);
        const float out = v / maxVal;
        if(!std::isfinite(out)) {
            return 0.0f;
        }
        return std::min(1.0f, std::max(0.0f, out));
    }

    static std::shared_ptr<ob::Frame> applyIrConfidenceMaskToDepth(const std::shared_ptr<ob::DepthFrame> &depthFrame,
                                                                   const std::shared_ptr<ob::IRFrame> &irLeft,
                                                                   const std::shared_ptr<ob::IRFrame> &irRight,
                                                                   double threshold01) {
        if(!depthFrame) {
            return nullptr;
        }
        if(threshold01 <= 0.0) {
            return depthFrame;
        }
        if(!irLeft || !irRight) {
            return depthFrame;
        }

        std::shared_ptr<ob::DepthFrame> maskedDepth;
        try {
            auto cloned = ob::FrameFactory::createFrameFromOtherFrame(depthFrame, true);
            maskedDepth = cloned->as<ob::DepthFrame>();
        }
        catch(...) {
            return depthFrame;
        }
        if(!maskedDepth) {
            return depthFrame;
        }

        const int dw = static_cast<int>(maskedDepth->getWidth());
        const int dh = static_cast<int>(maskedDepth->getHeight());
        if(dw <= 0 || dh <= 0) {
            return depthFrame;
        }

        auto *depthRaw = reinterpret_cast<uint8_t *>(maskedDepth->data());
        if(!depthRaw) {
            return depthFrame;
        }
        const size_t depthSize = static_cast<size_t>(maskedDepth->dataSize());
        const size_t depthStrideBytes = dh > 0 ? (depthSize / static_cast<size_t>(dh)) : 0;
        if(depthStrideBytes < static_cast<size_t>(dw) * sizeof(uint16_t)) {
            return depthFrame;
        }

        const float scale = maskedDepth->getValueScale();
        if(!(scale > 0.0f)) {
            return depthFrame;
        }

        std::shared_ptr<ob::StreamProfile> depthProfile;
        std::shared_ptr<ob::StreamProfile> irLProfile;
        std::shared_ptr<ob::StreamProfile> irRProfile;
        try {
            depthProfile = maskedDepth->getStreamProfile();
            irLProfile   = irLeft->getStreamProfile();
            irRProfile   = irRight->getStreamProfile();
        }
        catch(...) {
            return depthFrame;
        }
        if(!depthProfile || !irLProfile || !irRProfile) {
            return depthFrame;
        }

        std::shared_ptr<ob::VideoStreamProfile> depthVsp;
        std::shared_ptr<ob::VideoStreamProfile> irLVsp;
        std::shared_ptr<ob::VideoStreamProfile> irRVsp;
        try {
            depthVsp = depthProfile->as<ob::VideoStreamProfile>();
            irLVsp   = irLProfile->as<ob::VideoStreamProfile>();
            irRVsp   = irRProfile->as<ob::VideoStreamProfile>();
        }
        catch(...) {
            return depthFrame;
        }
        if(!depthVsp || !irLVsp || !irRVsp) {
            return depthFrame;
        }

        const auto depthIntrinsic = depthVsp->getIntrinsic();
        const auto depthDist      = depthVsp->getDistortion();
        const auto irLIntrinsic   = irLVsp->getIntrinsic();
        const auto irLDist        = irLVsp->getDistortion();
        const auto irRIntrinsic   = irRVsp->getIntrinsic();
        const auto irRDist        = irRVsp->getDistortion();

        OBExtrinsic extrD2L{};
        OBExtrinsic extrD2R{};
        try {
            extrD2L = depthProfile->getExtrinsicTo(irLProfile);
            extrD2R = depthProfile->getExtrinsicTo(irRProfile);
        }
        catch(...) {
            return depthFrame;
        }

        const float th = static_cast<float>(std::min(1.0, std::max(0.0, threshold01)));
        for(int y = 0; y < dh; y++) {
            auto *rowU16 = reinterpret_cast<uint16_t *>(depthRaw + static_cast<size_t>(y) * depthStrideBytes);
            for(int x = 0; x < dw; x++) {
                const uint16_t d = rowU16[x];
                if(d == 0) {
                    continue;
                }
                const float depthMm = static_cast<float>(d) * scale;
                if(!(depthMm > 0.0f)) {
                    rowU16[x] = 0;
                    continue;
                }

                const OBPoint2f src{ static_cast<float>(x), static_cast<float>(y) };
                OBPoint2f pL{}, pR{};
                bool okL = false;
                bool okR = false;
                try {
                    okL = ob::CoordinateTransformHelper::transformation2dto2d(src, depthMm, depthIntrinsic, depthDist, irLIntrinsic, irLDist, extrD2L, &pL);
                }
                catch(...) {
                    okL = false;
                }
                try {
                    okR = ob::CoordinateTransformHelper::transformation2dto2d(src, depthMm, depthIntrinsic, depthDist, irRIntrinsic, irRDist, extrD2R, &pR);
                }
                catch(...) {
                    okR = false;
                }
                if(!okL || !okR) {
                    rowU16[x] = 0;
                    continue;
                }

                const float cL = bilinearSampleNormalized(irLeft, pL.x, pL.y);
                const float cR = bilinearSampleNormalized(irRight, pR.x, pR.y);
                const float conf = std::min(cL, cR);
                if(conf < th) {
                    rowU16[x] = 0;
                }
            }
        }

        return maskedDepth;
    }

    cv::Mat renderUnifiedPointCloud(const std::unordered_map<int, CachedStereoFrame> &frames, const InteractiveViewState &viewState) {
        cv::Mat canvas(viewState.height, viewState.width, CV_8UC3, cv::Scalar(0, 0, 0));
        std::vector<float> zbuf(static_cast<size_t>(viewState.width) * static_cast<size_t>(viewState.height),
                                std::numeric_limits<float>::infinity());

        cv::Vec3f right, up, forward, camPos;
        computeCameraBasis(viewState, right, up, forward, camPos);

        const float fx = 900.0f;
        const float fy = 900.0f;
        const float cx = static_cast<float>(viewState.width) * 0.5f;
        const float cy = static_cast<float>(viewState.height) * 0.5f;

        const std::vector<cv::Vec3b> palette = {
            cv::Vec3b(0, 80, 255), cv::Vec3b(0, 255, 80), cv::Vec3b(255, 80, 0), cv::Vec3b(255, 255, 0),
            cv::Vec3b(255, 0, 255), cv::Vec3b(0, 255, 255),
        };

        const float spacingM = 0.25f;

        for(const auto &kv: frames) {
            const int deviceIndex = kv.first;
            const auto &cached    = kv.second;
            if(!cached.depth) {
                continue;
            }

            std::shared_ptr<ob::Frame> depthForPc = cached.depth;
            if(cfg_.filters.confThreshold > 0.0) {
                depthForPc = applyIrConfidenceMaskToDepth(cached.depth, cached.irLeft, cached.irRight, cfg_.filters.confThreshold);
            }
            auto itDepthFilters = depthFilterChains_.find(deviceIndex);
            if(itDepthFilters == depthFilterChains_.end()) {
                itDepthFilters = depthFilterChains_.emplace(deviceIndex, OrbbecDepthFilterChain{}).first;
            }
            depthForPc = refineDepthFrameForPointCloud(depthForPc, itDepthFilters->second, 0.2f, cfg_.maxDepth, cfg_.filters);
            if(!depthForPc) {
                continue;
            }

            if(cfg_.filters.smoothThresholdM > 0.0) {
                if(depthForPc == cached.depth) {
                    try {
                        depthForPc = ob::FrameFactory::createFrameFromOtherFrame(depthForPc, true);
                    }
                    catch(...) {
                        continue;
                    }
                }
                applyEdgeSmoothing(depthForPc, cfg_.filters.smoothThresholdM);
            }

            auto it = pointCloudFilters_.find(deviceIndex);
            if(it == pointCloudFilters_.end()) {
                auto f = std::make_shared<ob::PointCloudFilter>();
                f->setCreatePointFormat(OB_FORMAT_POINT);
                f->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
                f->setDecimationFactor(cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 1);
                it = pointCloudFilters_.emplace(deviceIndex, std::move(f)).first;
            }

            std::shared_ptr<ob::Frame> pcFrame;
            try {
                pcFrame = it->second->process(depthForPc);
            }
            catch(...) {
                continue;
            }
            if(!pcFrame) {
                continue;
            }

            auto pointsFrame = pcFrame->as<ob::PointsFrame>();
            if(!pointsFrame) {
                continue;
            }

            const float scaleMm = pointsFrame->getCoordinateValueScale();
            const auto  data    = reinterpret_cast<const OBPoint *>(pointsFrame->data());
            const auto  count   = pointsFrame->dataSize() / sizeof(OBPoint);
            if(!data || count == 0) {
                continue;
            }

            auto cloudCam = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            cloudCam->points.reserve(static_cast<size_t>(count));
            for(size_t i = 0; i < count; i++) {
                const OBPoint &p = data[i];
                if(!std::isfinite(p.z) || p.z <= 0.0f) {
                    continue;
                }
                const float z = p.z * scaleMm * 0.001f;
                if(!std::isfinite(z) || z <= 0.0f) {
                    continue;
                }
                const float x = p.x * scaleMm * 0.001f;
                const float y = p.y * scaleMm * 0.001f;
                if(!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                cloudCam->points.emplace_back(x, y, z);
            }
            cloudCam->width    = static_cast<uint32_t>(cloudCam->points.size());
            cloudCam->height   = 1;
            cloudCam->is_dense = false;
            if(cfg_.filters.deskCrop) {
                cloudCam = removeDominantPlaneRansac(cloudCam, 100, 0.015, 0.25, 1500);
            }

            const cv::Vec3b color =
                cfg_.differentColor ? palette[static_cast<size_t>(deviceIndex) % palette.size()] : cv::Vec3b(255, 255, 255);
            cv::Matx33f      Rcam = cv::Matx33f::eye();
            cv::Vec3f        tcam(deviceIndex * spacingM, 0.0f, 0.0f);
            if(!cached.camIndex.empty()) {
                const auto itEx = extrinsicsCamToWorld_.find(cached.camIndex);
                if(itEx != extrinsicsCamToWorld_.end() && itEx->second.valid) {
                    Rcam = itEx->second.R;
                    tcam = itEx->second.t;
                }
            }

            for(size_t i = 0; i < cloudCam->points.size(); i++) {
                const auto &p = cloudCam->points[i];
                const cv::Vec3f pCam(p.x, p.y, p.z);
                const cv::Vec3f pw = Rcam * pCam + tcam;
                const float     x  = pw[0];
                const float     y  = pw[1];
                const float     z  = pw[2];
                if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    continue;
                }

                const cv::Vec3f v  = pw - camPos;
                const float      xc = v.dot(right);
                const float      yc = v.dot(up);
                const float      zc = v.dot(forward);
                if(zc <= 0.05f) {
                    continue;
                }

                const int u = static_cast<int>(fx * (xc / zc) + cx);
                const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
                if(u < 0 || u >= viewState.width || vpx < 0 || vpx >= viewState.height) {
                    continue;
                }

                const size_t idx = static_cast<size_t>(vpx) * static_cast<size_t>(viewState.width) + static_cast<size_t>(u);
                if(zc < zbuf[idx]) {
                    zbuf[idx]                = zc;
                    canvas.at<cv::Vec3b>(vpx, u) = color;
                }
            }
        }

        cv::putText(canvas, "LMB: rotate  RMB: pan  Wheel: zoom  R: reset  Ctrl+C: exit", cv::Point(12, 28), cv::FONT_HERSHEY_DUPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        return canvas;
    }

    AppConfig                                  cfg_;
    ob::Context                                ctx_;
    std::atomic<ob_smpl::CVWindow *>           winPtr_{nullptr};
    std::mutex                                 viewerMutex_;
    std::unordered_map<int, uint64_t>          lastShownTs_;
    uint64_t                                   viewerIntervalUs_ = 33333;
    std::mutex                                 latestDepthMutex_;
    std::unordered_map<int, CachedStereoFrame> latestDepth_;
    std::unordered_map<int, std::shared_ptr<ob::PointCloudFilter>> pointCloudFilters_;
    std::unordered_map<int, OrbbecDepthFilterChain>                depthFilterChains_;
    std::unordered_map<std::string, ExtrinsicCamToWorld> extrinsicsCamToWorld_;
};

class MultiDeviceCalibrator {
public:
    explicit MultiDeviceCalibrator(AppConfig cfg)
        : cfg_(std::move(cfg)) {}

    int run() {
        auto deviceList = ctx_.queryDeviceList();
        if(deviceList->deviceCount() == 0) {
            std::cerr << "No devices found" << std::endl;
            return 1;
        }

        auto selected = selectDevices(deviceList);
        if(selected.empty()) {
            std::cerr << "No configured devices found in current device list" << std::endl;
            return 1;
        }

        if(cfg_.enableSync) {
            applySyncConfig(selected);
            ctx_.enableDeviceClockSync(60000);
        }

        const bool ansi = ob_smpl::supportAnsiEscape();
        bool quit       = false;

        while(!quit) {
            std::cout << "==========calibration mode(press \"q\" for quitting)==========" << std::endl;
            std::cout << "choose calibration method:" << std::endl;
            std::cout << "    1. chessboard" << std::endl;
            std::cout << "    2. block" << std::endl;
            std::cout << "    3. icp" << std::endl;

            const auto input = readLine();
            if(isQuit(input)) {
                quit = true;
                break;
            }
            if(input == "1") {
                runChessboardCalibration(selected, ansi, quit);
            }
            else if(input == "2") {
                runBlockCalibration(ansi, quit);
            }
            else if(input == "3") {
                runIcpRefinement(selected, ansi);
            }
        }

        return 0;
    }

private:
    struct DeviceRuntime {
        DeviceConfig                cfg;
        std::shared_ptr<ob::Device> dev;
    };

    struct ColorSample {
        uint64_t ts1 = 0;
        uint64_t ts2 = 0;
        cv::Mat  img1;
        cv::Mat  img2;
    };

    struct EdgeExtrinsic {
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d   t{ 0.0, 0.0, 0.0 };
        bool        valid = false;
    };

    static std::string readLine() {
        std::string line;
        std::getline(std::cin, line);
        size_t b = 0;
        while(b < line.size() && std::isspace(static_cast<unsigned char>(line[b]))) {
            b++;
        }
        size_t e = line.size();
        while(e > b && std::isspace(static_cast<unsigned char>(line[e - 1]))) {
            e--;
        }
        line = line.substr(b, e - b);
        std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return line;
    }

    static bool isQuit(const std::string &s) {
        return s == "q" || s == "quit" || s == "exit";
    }

    static std::string colorize(const std::string &s, const char *code, bool ansi) {
        if(!ansi) {
            return s;
        }
        return std::string("\033[") + code + "m" + s + "\033[0m";
    }

    static std::string normalizePairKey(std::string a, std::string b) {
        if(a > b) {
            std::swap(a, b);
        }
        return a + "|" + b;
    }

    static std::string edgeKey(const std::string &from, const std::string &to) {
        return from + "->" + to;
    }

    std::vector<DeviceRuntime> selectDevices(const std::shared_ptr<ob::DeviceList> &deviceList) {
        std::unordered_map<std::string, std::shared_ptr<ob::Device>> bySn;
        for(uint32_t i = 0; i < deviceList->deviceCount(); i++) {
            auto dev = deviceList->getDevice(i);
            bySn.emplace(std::string(dev->getDeviceInfo()->serialNumber()), dev);
        }

        std::vector<DeviceRuntime> out;
        out.reserve(cfg_.devices.size());
        for(const auto &dc: cfg_.devices) {
            auto it = bySn.find(dc.sn);
            if(it == bySn.end()) {
                std::cerr << "Configured device not found: " << dc.sn << std::endl;
                continue;
            }
            DeviceRuntime rt;
            rt.cfg = dc;
            rt.dev = it->second;
            out.push_back(std::move(rt));
        }
        return out;
    }

    void applySyncConfig(std::vector<DeviceRuntime> &devices) {
        for(size_t i = 0; i < devices.size(); i++) {
            auto &rt = devices[i];
            auto cur = rt.dev->getMultiDeviceSyncConfig();
            auto cfg = rt.cfg.hasSyncConfig ? rt.cfg.syncConfig : cur;
            if(!rt.cfg.hasSyncConfig) {
                cfg.syncMode = (i == 0) ? OB_MULTI_DEVICE_SYNC_MODE_PRIMARY : OB_MULTI_DEVICE_SYNC_MODE_SECONDARY;
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
        }
    }

    bool hasCameraIndex(const std::vector<DeviceRuntime> &devices, const std::string &idx) {
        for(const auto &rt: devices) {
            if(rt.cfg.index == idx) {
                return true;
            }
        }
        return false;
    }

    const DeviceRuntime *findByIndex(const std::vector<DeviceRuntime> &devices, const std::string &idx) {
        for(const auto &rt: devices) {
            if(rt.cfg.index == idx) {
                return &rt;
            }
        }
        return nullptr;
    }

    static uint64_t frameTimestampUs(const std::shared_ptr<ob::Frame> &frame) {
        if(!frame) {
            return 0;
        }
        uint64_t ts = 0;
        try {
            ts = frame->globalTimeStampUs();
        }
        catch(...) {
        }
        if(ts == 0) {
            ts = frame->timeStampUs();
        }
        return ts;
    }

    static cv::Mat colorFrameToBgr(const std::shared_ptr<ob::Frame> &frame) {
        auto colorFrame = frame ? frame->as<ob::ColorFrame>() : nullptr;
        if(!colorFrame) {
            return cv::Mat();
        }

        thread_local auto converter = std::make_shared<ob::FormatConvertFilter>();
        if(colorFrame->format() != OB_FORMAT_BGR) {
            if(colorFrame->format() != OB_FORMAT_RGB) {
                if(colorFrame->format() == OB_FORMAT_MJPG) {
                    converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
                }
                else if(colorFrame->format() == OB_FORMAT_UYVY) {
                    converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
                }
                else if(colorFrame->format() == OB_FORMAT_YUYV) {
                    converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
                }
                else {
                    return cv::Mat();
                }
                colorFrame = converter->process(colorFrame)->as<ob::ColorFrame>();
                if(!colorFrame) {
                    return cv::Mat();
                }
            }
            converter->setFormatConvertType(FORMAT_RGB_TO_BGR);
            colorFrame = converter->process(colorFrame)->as<ob::ColorFrame>();
            if(!colorFrame) {
                return cv::Mat();
            }
        }

        const int width  = static_cast<int>(colorFrame->width());
        const int height = static_cast<int>(colorFrame->height());
        cv::Mat bgr(height, width, CV_8UC3, const_cast<void *>(colorFrame->data()));
        return bgr.clone();
    }

    static cv::Mat toCameraMatrix(const OBCameraIntrinsic &in) {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = static_cast<double>(in.fx);
        K.at<double>(1, 1) = static_cast<double>(in.fy);
        K.at<double>(0, 2) = static_cast<double>(in.cx);
        K.at<double>(1, 2) = static_cast<double>(in.cy);
        return K;
    }

    static cv::Mat toDistCoeffs(const OBCameraDistortion &d) {
        cv::Mat coeffs = cv::Mat::zeros(1, 8, CV_64F);
        coeffs.at<double>(0, 0) = static_cast<double>(d.k1);
        coeffs.at<double>(0, 1) = static_cast<double>(d.k2);
        coeffs.at<double>(0, 2) = static_cast<double>(d.p1);
        coeffs.at<double>(0, 3) = static_cast<double>(d.p2);
        coeffs.at<double>(0, 4) = static_cast<double>(d.k3);
        coeffs.at<double>(0, 5) = static_cast<double>(d.k4);
        coeffs.at<double>(0, 6) = static_cast<double>(d.k5);
        coeffs.at<double>(0, 7) = static_cast<double>(d.k6);
        return coeffs;
    }

    struct PairPipelines {
        std::shared_ptr<ob::Pipeline> p1;
        std::shared_ptr<ob::Pipeline> p2;
        cv::Mat K1, D1, K2, D2;
        int colorW = 0;
        int colorH = 0;
    };

    static std::shared_ptr<ob::VideoStreamProfile> defaultVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe, OBSensorType sensorType) {
        auto list = pipe->getStreamProfileList(sensorType);
        if(!list || list->getCount() == 0) {
            return nullptr;
        }
        return list->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
    }

    bool startPairPipelines(const DeviceRuntime &a, const DeviceRuntime &b, PairPipelines &out) {
        out.p1 = std::make_shared<ob::Pipeline>(a.dev);
        out.p2 = std::make_shared<ob::Pipeline>(b.dev);

        auto c1 = defaultVideoProfile(out.p1, OB_SENSOR_COLOR);
        auto d1 = defaultVideoProfile(out.p1, OB_SENSOR_DEPTH);
        auto c2 = defaultVideoProfile(out.p2, OB_SENSOR_COLOR);
        auto d2 = defaultVideoProfile(out.p2, OB_SENSOR_DEPTH);
        if(!c1 || !d1 || !c2 || !d2) {
            return false;
        }

        out.colorW = static_cast<int>(c1->getWidth());
        out.colorH = static_cast<int>(c1->getHeight());

        auto cfg1 = std::make_shared<ob::Config>();
        cfg1->enableStream(c1);
        cfg1->enableStream(d1);
        auto cfg2 = std::make_shared<ob::Config>();
        cfg2->enableStream(c2);
        cfg2->enableStream(d2);

        const auto mode1 = a.dev->getMultiDeviceSyncConfig().syncMode;
        const auto mode2 = b.dev->getMultiDeviceSyncConfig().syncMode;
        const bool p1IsPrimary   = (mode1 == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY);
        const bool p2IsSecondary = (mode2 == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY || mode2 == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED);

        if(p1IsPrimary && p2IsSecondary) {
            out.p2->start(cfg2);
            out.p1->start(cfg1);
        }
        else {
            out.p1->start(cfg1);
            out.p2->start(cfg2);
        }

        const auto cp1 = out.p1->getCameraParamWithProfile(c1->getWidth(), c1->getHeight(), d1->getWidth(), d1->getHeight());
        const auto cp2 = out.p2->getCameraParamWithProfile(c2->getWidth(), c2->getHeight(), d2->getWidth(), d2->getHeight());
        out.K1 = toCameraMatrix(cp1.rgbIntrinsic);
        out.D1 = toDistCoeffs(cp1.rgbDistortion);
        out.K2 = toCameraMatrix(cp2.rgbIntrinsic);
        out.D2 = toDistCoeffs(cp2.rgbDistortion);

        return true;
    }

    static void stopPairPipelines(PairPipelines &pp) {
        try {
            if(pp.p1) {
                pp.p1->stop();
            }
        }
        catch(...) {
        }
        try {
            if(pp.p2) {
                pp.p2->stop();
            }
        }
        catch(...) {
        }
        pp.p1.reset();
        pp.p2.reset();
    }

    static bool tryGetColorFrame(const std::shared_ptr<ob::Pipeline> &p, uint32_t timeoutMs, cv::Mat &img, uint64_t &tsUs) {
        auto fs = p->waitForFrameset(timeoutMs);
        if(!fs) {
            return false;
        }
        auto frame = fs->colorFrame();
        if(!frame) {
            return false;
        }
        tsUs = frameTimestampUs(frame);
        img  = colorFrameToBgr(frame);
        return !img.empty() && tsUs != 0;
    }

    static bool captureSyncedPair(const PairPipelines &pp, uint64_t maxWaitMs, uint64_t maxDiffUs, ColorSample &out) {
        std::optional<cv::Mat>  img1;
        std::optional<cv::Mat>  img2;
        std::optional<uint64_t> ts1;
        std::optional<uint64_t> ts2;

        const auto start = std::chrono::steady_clock::now();
        while(true) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count());
            if(elapsedMs > maxWaitMs) {
                return false;
            }

            if(!img1.has_value()) {
                cv::Mat im;
                uint64_t ts = 0;
                if(tryGetColorFrame(pp.p1, 50, im, ts)) {
                    img1 = std::move(im);
                    ts1  = ts;
                }
            }
            if(!img2.has_value()) {
                cv::Mat im;
                uint64_t ts = 0;
                if(tryGetColorFrame(pp.p2, 50, im, ts)) {
                    img2 = std::move(im);
                    ts2  = ts;
                }
            }

            if(img1.has_value() && img2.has_value() && ts1.has_value() && ts2.has_value()) {
                const uint64_t a = *ts1;
                const uint64_t b = *ts2;
                const uint64_t diff = (a > b) ? (a - b) : (b - a);
                if(diff <= maxDiffUs) {
                    out.ts1  = a;
                    out.ts2  = b;
                    out.img1 = img1->clone();
                    out.img2 = img2->clone();
                    return true;
                }
                if(a < b) {
                    img1.reset();
                    ts1.reset();
                }
                else {
                    img2.reset();
                    ts2.reset();
                }
            }
        }
    }

    bool computeRelativeExtrinsic(const PairPipelines &pp,
                                  const std::vector<ColorSample> &samples,
                                  cv::Matx33d &R12,
                                  cv::Vec3d &t12) {
        const auto &cb = cfg_.calibration.chessboard;
        const cv::Size patternSize(cb.cols, cb.rows);

        std::vector<std::vector<cv::Point3f>> objPts;
        std::vector<std::vector<cv::Point2f>> imgPts1;
        std::vector<std::vector<cv::Point2f>> imgPts2;

        std::vector<cv::Point3f> obj;
        obj.reserve(static_cast<size_t>(cb.cols * cb.rows));
        for(int y = 0; y < cb.rows; y++) {
            for(int x = 0; x < cb.cols; x++) {
                obj.emplace_back(static_cast<float>(x) * cb.squareSize, static_cast<float>(y) * cb.squareSize, 0.0f);
            }
        }

        for(const auto &s: samples) {
            if(s.img1.empty() || s.img2.empty()) {
                continue;
            }
            cv::Mat g1, g2;
            cv::cvtColor(s.img1, g1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(s.img2, g2, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners1;
            std::vector<cv::Point2f> corners2;
            const bool ok1 = cv::findChessboardCorners(g1, patternSize, corners1, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            const bool ok2 = cv::findChessboardCorners(g2, patternSize, corners2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if(!ok1 || !ok2) {
                continue;
            }

            cv::cornerSubPix(g1, corners1, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(g2, corners2, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

            objPts.push_back(obj);
            imgPts1.push_back(std::move(corners1));
            imgPts2.push_back(std::move(corners2));
        }

        if(objPts.size() < 3) {
            return false;
        }

        cv::Mat K1 = pp.K1.clone();
        cv::Mat D1 = pp.D1.clone();
        cv::Mat K2 = pp.K2.clone();
        cv::Mat D2 = pp.D2.clone();

        int flags = cv::CALIB_FIX_INTRINSIC;
        if((D1.total() == 8 && (D1.rows == 1 || D1.cols == 1)) && (D2.total() == 8 && (D2.rows == 1 || D2.cols == 1))) {
            flags |= cv::CALIB_RATIONAL_MODEL;
        }

        cv::Mat R, T, E, F;
        const cv::Size imageSize(pp.colorW, pp.colorH);
        cv::stereoCalibrate(objPts, imgPts1, imgPts2, K1, D1, K2, D2, imageSize, R, T, E, F,
                            flags,
                            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-6));

        cv::Mat Rd, Td;
        R.convertTo(Rd, CV_64F);
        T.convertTo(Td, CV_64F);

        R12 = cv::Matx33d(Rd.at<double>(0, 0), Rd.at<double>(0, 1), Rd.at<double>(0, 2),
                          Rd.at<double>(1, 0), Rd.at<double>(1, 1), Rd.at<double>(1, 2),
                          Rd.at<double>(2, 0), Rd.at<double>(2, 1), Rd.at<double>(2, 2));
        t12 = cv::Vec3d(Td.at<double>(0, 0), Td.at<double>(1, 0), Td.at<double>(2, 0));

        return true;
    }

    static bool isPrimary(const std::shared_ptr<ob::Device> &dev) {
        if(!dev) {
            return false;
        }
        try {
            return dev->getMultiDeviceSyncConfig().syncMode == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY;
        }
        catch(...) {
            return false;
        }
    }

    struct IcpDevice {
        std::string                        index;
        std::shared_ptr<ob::Device>        dev;
        std::shared_ptr<ob::Pipeline>      pipe;
        std::shared_ptr<ob::PointCloudFilter> pcFilter;
        OrbbecDepthFilterChain                depthFilters;
    };

    bool startIcpPipelines(std::vector<IcpDevice> &devs) {
        for(auto &d: devs) {
            d.pipe     = std::make_shared<ob::Pipeline>(d.dev);
            d.pcFilter = std::make_shared<ob::PointCloudFilter>();
            d.pcFilter->setCreatePointFormat(OB_FORMAT_POINT);
            d.pcFilter->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
            d.pcFilter->setDecimationFactor(cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 1);
        }

        std::vector<IcpDevice *> secondary;
        std::vector<IcpDevice *> primary;
        for(auto &d: devs) {
            if(isPrimary(d.dev)) {
                primary.push_back(&d);
            }
            else {
                secondary.push_back(&d);
            }
        }

        auto startOne = [](IcpDevice &d) -> bool {
            auto depthProfile = defaultVideoProfile(d.pipe, OB_SENSOR_DEPTH);
            if(!depthProfile) {
                return false;
            }
            auto cfg = std::make_shared<ob::Config>();
            cfg->enableStream(depthProfile);
            d.pipe->start(cfg);
            return true;
        };

        for(auto *d: secondary) {
            if(!startOne(*d)) {
                return false;
            }
        }
        for(auto *d: primary) {
            if(!startOne(*d)) {
                return false;
            }
        }
        return true;
    }

    static void stopIcpPipelines(std::vector<IcpDevice> &devs) {
        for(auto &d: devs) {
            try {
                if(d.pipe) {
                    d.pipe->stop();
                }
            }
            catch(...) {
            }
            d.pipe.reset();
            d.pcFilter.reset();
        }
    }

    static bool tryGetDepthFrame(const std::shared_ptr<ob::Pipeline> &p, uint32_t timeoutMs, std::shared_ptr<ob::Frame> &depth, uint64_t &tsUs) {
        auto fs = p->waitForFrameset(timeoutMs);
        if(!fs) {
            return false;
        }
        auto frame = fs->depthFrame();
        if(!frame) {
            return false;
        }
        tsUs = frameTimestampUs(frame);
        if(tsUs == 0) {
            return false;
        }
        depth = frame;
        return true;
    }

    bool captureSyncedDepthSet(std::vector<IcpDevice> &devs,
                               uint64_t maxWaitMs,
                               uint64_t maxDiffUs,
                               std::unordered_map<std::string, std::pair<uint64_t, std::shared_ptr<ob::Frame>>> &out) {
        struct Slot {
            std::shared_ptr<ob::Frame> depth;
            uint64_t                   tsUs  = 0;
            bool                       ready = false;
        };
        std::vector<Slot> slots(devs.size());

        const auto start = std::chrono::steady_clock::now();
        while(true) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count());
            if(elapsedMs > maxWaitMs) {
                return false;
            }

            for(size_t i = 0; i < devs.size(); i++) {
                if(slots[i].ready) {
                    continue;
                }
                std::shared_ptr<ob::Frame> depth;
                uint64_t tsUs = 0;
                if(tryGetDepthFrame(devs[i].pipe, 50, depth, tsUs)) {
                    slots[i].depth = std::move(depth);
                    slots[i].tsUs  = tsUs;
                    slots[i].ready = true;
                }
            }

            bool allReady = true;
            for(const auto &s: slots) {
                if(!s.ready) {
                    allReady = false;
                    break;
                }
            }
            if(!allReady) {
                continue;
            }

            uint64_t minTs = std::numeric_limits<uint64_t>::max();
            uint64_t maxTs = 0;
            size_t minIdx  = 0;
            for(size_t i = 0; i < slots.size(); i++) {
                const auto ts = slots[i].tsUs;
                if(ts < minTs) {
                    minTs  = ts;
                    minIdx = i;
                }
                if(ts > maxTs) {
                    maxTs = ts;
                }
            }

            if(maxTs - minTs <= maxDiffUs) {
                out.clear();
                for(size_t i = 0; i < devs.size(); i++) {
                    out.emplace(devs[i].index, std::make_pair(slots[i].tsUs, slots[i].depth));
                }
                return true;
            }

            slots[minIdx] = Slot{};
        }
    }

    static Eigen::Matrix4f toEigenWorldToCam(const EdgeExtrinsic &ex) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(0, 0) = static_cast<float>(ex.R(0, 0));
        T(0, 1) = static_cast<float>(ex.R(0, 1));
        T(0, 2) = static_cast<float>(ex.R(0, 2));
        T(1, 0) = static_cast<float>(ex.R(1, 0));
        T(1, 1) = static_cast<float>(ex.R(1, 1));
        T(1, 2) = static_cast<float>(ex.R(1, 2));
        T(2, 0) = static_cast<float>(ex.R(2, 0));
        T(2, 1) = static_cast<float>(ex.R(2, 1));
        T(2, 2) = static_cast<float>(ex.R(2, 2));
        T(0, 3) = static_cast<float>(ex.t[0]);
        T(1, 3) = static_cast<float>(ex.t[1]);
        T(2, 3) = static_cast<float>(ex.t[2]);
        return T;
    }

    static Eigen::Matrix3f projectRotationToSO3(const Eigen::Matrix3f &Rin) {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(Rin, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        Eigen::Matrix3f R = U * V.transpose();
        if(R.determinant() < 0.0f) {
            U.col(2) *= -1.0f;
            R = U * V.transpose();
        }
        return R;
    }

    static Eigen::Matrix4f makeRigid(const Eigen::Matrix4f &Tin) {
        Eigen::Matrix4f T = Tin;
        T.block<3, 3>(0, 0) = projectRotationToSO3(T.block<3, 3>(0, 0));
        T(3, 0) = 0.0f;
        T(3, 1) = 0.0f;
        T(3, 2) = 0.0f;
        T(3, 3) = 1.0f;
        return T;
    }

    static EdgeExtrinsic fromEigenWorldToCam(const Eigen::Matrix4f &T) {
        const Eigen::Matrix4f Tr = makeRigid(T);
        EdgeExtrinsic ex;
        ex.valid = true;
        ex.R = cv::Matx33d(static_cast<double>(Tr(0, 0)), static_cast<double>(Tr(0, 1)), static_cast<double>(Tr(0, 2)),
                           static_cast<double>(Tr(1, 0)), static_cast<double>(Tr(1, 1)), static_cast<double>(Tr(1, 2)),
                           static_cast<double>(Tr(2, 0)), static_cast<double>(Tr(2, 1)), static_cast<double>(Tr(2, 2)));
        ex.t = cv::Vec3d(static_cast<double>(Tr(0, 3)), static_cast<double>(Tr(1, 3)), static_cast<double>(Tr(2, 3)));
        return ex;
    }

    static Eigen::Matrix4f invertRigid(const Eigen::Matrix4f &T) {
        const Eigen::Matrix4f Tr = makeRigid(T);
        const Eigen::Matrix3f R  = Tr.block<3, 3>(0, 0);
        const Eigen::Vector3f t  = Tr.block<3, 1>(0, 3);
        Eigen::Matrix4f inv = Eigen::Matrix4f::Identity();
        inv.block<3, 3>(0, 0) = R.transpose();
        inv.block<3, 1>(0, 3) = -R.transpose() * t;
        return inv;
    }

    static pcl::PointCloud<pcl::PointXYZ>::Ptr pointsFrameToCloudCam(const std::shared_ptr<ob::PointsFrame> &pointsFrame,
                                                                     float minZ,
                                                                     float maxZ) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        (void)minZ;
        (void)maxZ;
        if(!pointsFrame) {
            return cloud;
        }

        const float scaleMm = pointsFrame->getCoordinateValueScale();
        const auto  data    = reinterpret_cast<const OBPoint *>(pointsFrame->data());
        const auto  count   = pointsFrame->dataSize() / sizeof(OBPoint);
        if(!data || count == 0) {
            return cloud;
        }

        cloud->points.reserve(static_cast<size_t>(count));
        for(size_t i = 0; i < count; i++) {
            const OBPoint &p = data[i];
            if(!std::isfinite(p.z) || p.z <= 0.0f) {
                continue;
            }
            const float z = p.z * scaleMm * 0.001f;
            if(!std::isfinite(z) || z <= 0.0f) {
                continue;
            }
            const float x = p.x * scaleMm * 0.001f;
            const float y = p.y * scaleMm * 0.001f;
            if(!std::isfinite(x) || !std::isfinite(y)) {
                continue;
            }
            cloud->points.emplace_back(x, y, z);
        }
        cloud->width  = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    bool refineExtrinsicsWithPclIcp(std::vector<DeviceRuntime> &devices,
                                    std::unordered_map<std::string, EdgeExtrinsic> &worldToCam,
                                    const std::string &rootIdx,
                                    int &outItersUsed) {
        outItersUsed = 0;
        const int frames = 10;
        const uint64_t maxWaitMs = 4000;
        const uint64_t maxDiffUs = 20000;
        const float minZ = 0.2f;
        const float maxZ = (cfg_.maxDepth > 0.0f) ? cfg_.maxDepth : 6.0f;
        const float maxCorrespondenceDist = 0.08f;
        const int maxOuterIters = 300;
        const int icpInnerIters = 1;
        const double stopRotDeg = 0.001;
        const double stopTransM = 0.0005;

        const auto itRoot = worldToCam.find(rootIdx);
        if(itRoot == worldToCam.end() || !itRoot->second.valid) {
            return false;
        }

        for(auto &kv: worldToCam) {
            if(!kv.second.valid) {
                continue;
            }
            kv.second = fromEigenWorldToCam(toEigenWorldToCam(kv.second));
        }

        std::vector<IcpDevice> devs;
        devs.reserve(devices.size());
        for(auto &rt: devices) {
            IcpDevice d;
            d.index = rt.cfg.index;
            d.dev   = rt.dev;
            devs.push_back(std::move(d));
        }

        if(!startIcpPipelines(devs)) {
            stopIcpPipelines(devs);
            return false;
        }

        std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudsW;
        cloudsW.reserve(devs.size());
        for(const auto &d: devs) {
            cloudsW.emplace(d.index, pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>());
        }

        for(int f = 0; f < frames; f++) {
            std::unordered_map<std::string, std::pair<uint64_t, std::shared_ptr<ob::Frame>>> depthFrames;
            if(!captureSyncedDepthSet(devs, maxWaitMs, maxDiffUs, depthFrames)) {
                break;
            }

            for(auto &d: devs) {
                const auto it = depthFrames.find(d.index);
                if(it == depthFrames.end()) {
                    continue;
                }
                auto depth = it->second.second;
                if(!depth) {
                    continue;
                }

                std::shared_ptr<ob::Frame> depthForPc = depth;
                depthForPc = refineDepthFrameForPointCloud(depthForPc, d.depthFilters, minZ, maxZ, cfg_.filters);
                if(!depthForPc) {
                    continue;
                }

                std::shared_ptr<ob::Frame> pcFrame;
                try {
                    pcFrame = d.pcFilter->process(depthForPc);
                }
                catch(...) {
                    continue;
                }
                if(!pcFrame) {
                    continue;
                }
                auto pointsFrame = pcFrame->as<ob::PointsFrame>();
                if(!pointsFrame) {
                    continue;
                }

                const auto itEx = worldToCam.find(d.index);
                if(itEx == worldToCam.end() || !itEx->second.valid) {
                    continue;
                }
                const Eigen::Matrix4f Tcw = toEigenWorldToCam(itEx->second);
                const Eigen::Matrix4f Twc = invertRigid(Tcw);

                auto cloudCam = pointsFrameToCloudCam(pointsFrame, minZ, maxZ);
                auto cloudW   = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                pcl::transformPointCloud(*cloudCam, *cloudW, Twc);

                auto &acc = cloudsW[d.index];
                *acc += *cloudW;
            }
        }

        stopIcpPipelines(devs);

        for(auto &kv: cloudsW) {
            if(cfg_.filters.deskCrop) {
                kv.second = removeDominantPlaneRansac(kv.second, 100, 0.015, 0.25, 1500);
            }
            kv.second = denoiseCloudSor(kv.second, 20, 1.0);
        }

        const auto mergeCloudsExcluding = [&](const std::string &excludeIdx) -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
            auto merged = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for(const auto &dc: cfg_.devices) {
                if(dc.index == excludeIdx) {
                    continue;
                }
                const auto it = cloudsW.find(dc.index);
                if(it == cloudsW.end() || !it->second || it->second->empty()) {
                    continue;
                }
                *merged += *it->second;
            }
            merged->width  = static_cast<uint32_t>(merged->points.size());
            merged->height = 1;
            merged->is_dense = false;
            return merged;
        };
        const auto calcRotDegFromR = [](const Eigen::Matrix3f &Rin) -> double {
            const Eigen::Matrix3f R = projectRotationToSO3(Rin);
            double tr = static_cast<double>(R.trace());
            double c = (tr - 1.0) * 0.5;
            c = std::min(1.0, std::max(-1.0, c));
            const double angle = std::acos(c);
            return angle * (180.0 / 3.14159265358979323846);
        };

        const auto tryParseIndex = [](const std::string &s) -> std::optional<int> {
            if(s.empty()) {
                return std::nullopt;
            }
            for(const char c: s) {
                if(!std::isdigit(static_cast<unsigned char>(c))) {
                    return std::nullopt;
                }
            }
            try {
                return std::stoi(s);
            }
            catch(...) {
                return std::nullopt;
            }
        };

        std::vector<std::string> ordered;
        ordered.reserve(cfg_.devices.size());
        for(const auto &dc: cfg_.devices) {
            ordered.push_back(dc.index);
        }
        std::sort(ordered.begin(), ordered.end(), [&](const std::string &a, const std::string &b) {
            const auto ia = tryParseIndex(a);
            const auto ib = tryParseIndex(b);
            if(ia.has_value() && ib.has_value()) {
                if(*ia != *ib) {
                    return *ia < *ib;
                }
                return a < b;
            }
            return a < b;
        });
        ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
        const auto itRootPos = std::find(ordered.begin(), ordered.end(), rootIdx);
        if(itRootPos != ordered.end() && itRootPos != ordered.begin()) {
            const std::string r = *itRootPos;
            ordered.erase(itRootPos);
            ordered.insert(ordered.begin(), r);
        }

        if(ordered.empty() || ordered.front() != rootIdx) {
            return true;
        }

        int itersUsed = 0;
        for(int iter = 0; iter < maxOuterIters; iter++) {
            bool allSmall = true;
            for(size_t i = 0; i < ordered.size(); i++) {
                const std::string &curIdx = ordered[i];
                const auto itSrc = cloudsW.find(curIdx);
                if(itSrc == cloudsW.end() || !itSrc->second || itSrc->second->size() < 100) {
                    continue;
                }

                auto target = mergeCloudsExcluding(curIdx);
                if(!target || target->size() < 200) {
                    continue;
                }

                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                icp.setInputSource(itSrc->second);
                icp.setInputTarget(target);
                icp.setMaximumIterations(icpInnerIters);
                icp.setMaxCorrespondenceDistance(maxCorrespondenceDist);
                icp.setTransformationEpsilon(1e-10);
                icp.setEuclideanFitnessEpsilon(1e-9);
                icp.setUseReciprocalCorrespondences(true);
                icp.setRANSACOutlierRejectionThreshold(maxCorrespondenceDist * 0.5f);

                auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                icp.align(*aligned);
                if(!icp.hasConverged()) {
                    continue;
                }

                const Eigen::Matrix4f delta = makeRigid(icp.getFinalTransformation());
                cloudsW[curIdx] = aligned;

                const auto itEx = worldToCam.find(curIdx);
                if(itEx != worldToCam.end() && itEx->second.valid) {
                    const Eigen::Matrix4f Tcw_old = toEigenWorldToCam(itEx->second);
                    const Eigen::Matrix4f Twc_old = invertRigid(Tcw_old);
                    const Eigen::Matrix4f Twc_new = delta * Twc_old;
                    const Eigen::Matrix4f Tcw_new = invertRigid(Twc_new);
                    worldToCam[curIdx] = fromEigenWorldToCam(Tcw_new);
                }

                const double rotDeg = calcRotDegFromR(delta.block<3, 3>(0, 0));
                const double transM = static_cast<double>(delta.block<3, 1>(0, 3).norm());
                if(rotDeg > stopRotDeg || transM > stopTransM) {
                    allSmall = false;
                }
            }
            itersUsed = iter + 1;
            if(allSmall) {
                break;
            }
        }
        outItersUsed = itersUsed;

        const auto itRootFinal = worldToCam.find(rootIdx);
        if(itRootFinal != worldToCam.end() && itRootFinal->second.valid) {
            const Eigen::Matrix4f Tcw_root = toEigenWorldToCam(itRootFinal->second);
            const Eigen::Matrix4f Trw = invertRigid(Tcw_root);
            for(const auto &dc: cfg_.devices) {
                auto it = worldToCam.find(dc.index);
                if(it == worldToCam.end() || !it->second.valid) {
                    continue;
                }
                const Eigen::Matrix4f Tcw = toEigenWorldToCam(it->second);
                const Eigen::Matrix4f Tcr = makeRigid(Tcw * Trw);
                it->second = fromEigenWorldToCam(Tcr);
            }

            auto itRootSet = worldToCam.find(rootIdx);
            if(itRootSet != worldToCam.end()) {
                itRootSet->second.valid = true;
                itRootSet->second.R     = cv::Matx33d::eye();
                itRootSet->second.t     = cv::Vec3d(0.0, 0.0, 0.0);
            }
        }

        return true;
    }

    static bool parseVec3d(cJSON *arr, cv::Vec3d &out) {
        if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
            return false;
        }
        const auto *a0 = cJSON_GetArrayItem(arr, 0);
        const auto *a1 = cJSON_GetArrayItem(arr, 1);
        const auto *a2 = cJSON_GetArrayItem(arr, 2);
        if(!a0 || !a1 || !a2 || !cJSON_IsNumber(a0) || !cJSON_IsNumber(a1) || !cJSON_IsNumber(a2)) {
            return false;
        }
        out = cv::Vec3d(a0->valuedouble, a1->valuedouble, a2->valuedouble);
        return true;
    }

    static bool parseMat3d(cJSON *arr, cv::Matx33d &out) {
        if(!arr || !cJSON_IsArray(arr)) {
            return false;
        }
        const int n = cJSON_GetArraySize(arr);
        if(n == 9) {
            double v[9];
            for(int i = 0; i < 9; i++) {
                auto *it = cJSON_GetArrayItem(arr, i);
                if(!it || !cJSON_IsNumber(it)) {
                    return false;
                }
                v[i] = it->valuedouble;
            }
            out = cv::Matx33d(v[0], v[1], v[2],
                              v[3], v[4], v[5],
                              v[6], v[7], v[8]);
            return true;
        }
        if(n == 3) {
            auto *r0 = cJSON_GetArrayItem(arr, 0);
            auto *r1 = cJSON_GetArrayItem(arr, 1);
            auto *r2 = cJSON_GetArrayItem(arr, 2);
            if(r0 && r1 && r2 && cJSON_IsArray(r0) && cJSON_IsArray(r1) && cJSON_IsArray(r2)
               && cJSON_GetArraySize(r0) == 3 && cJSON_GetArraySize(r1) == 3 && cJSON_GetArraySize(r2) == 3) {
                double v[9];
                cJSON *rows[3] = { r0, r1, r2 };
                for(int y = 0; y < 3; y++) {
                    for(int x = 0; x < 3; x++) {
                        auto *it = cJSON_GetArrayItem(rows[y], x);
                        if(!it || !cJSON_IsNumber(it)) {
                            return false;
                        }
                        v[y * 3 + x] = it->valuedouble;
                    }
                }
                out = cv::Matx33d(v[0], v[1], v[2],
                                  v[3], v[4], v[5],
                                  v[6], v[7], v[8]);
                return true;
            }
        }
        return false;
    }

    bool loadWorldToCamFromFile(const fs::path &path, std::unordered_map<std::string, EdgeExtrinsic> &worldToCam) {
        std::string content;
        try {
            content = readFileAll(path);
        }
        catch(const std::exception &e) {
            std::cerr << "Failed to read init_extrinsic_path: " << path.string() << ", error=" << e.what() << std::endl;
            return false;
        }

        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            std::cerr << "Invalid JSON in init_extrinsic_path: " << path.string() << std::endl;
            return false;
        }

        worldToCam.clear();
        for(cJSON *item = root->child; item != nullptr; item = item->next) {
            if(!item->string || !cJSON_IsObject(item)) {
                continue;
            }
            const std::string camId = item->string;
            auto *rotArr            = cJSON_GetObjectItemCaseSensitive(item, "rotation");
            auto *tArr              = cJSON_GetObjectItemCaseSensitive(item, "translation");

            cv::Vec3d tCw;
            cv::Matx33d Rcw = cv::Matx33d::eye();
            if(!parseVec3d(tArr, tCw) || !parseMat3d(rotArr, Rcw)) {
                continue;
            }
            EdgeExtrinsic ex;
            ex.valid = true;
            ex.R     = Rcw;
            ex.t     = tCw;
            worldToCam.emplace(camId, ex);
        }

        cJSON_Delete(root);
        return true;
    }

    bool writeWorldToCamToFile(const std::unordered_map<std::string, EdgeExtrinsic> &worldToCam) {
        if(cfg_.initExtrinsicPath.empty()) {
            std::cerr << "init_extrinsic_path is empty" << std::endl;
            return false;
        }

        cJSON *root = cJSON_CreateObject();
        for(const auto &dc: cfg_.devices) {
            const auto it = worldToCam.find(dc.index);
            if(it == worldToCam.end() || !it->second.valid) {
                continue;
            }
            const auto &ex = it->second;

            cJSON *camObj = cJSON_CreateObject();
            cJSON *rot    = cJSON_CreateArray();
            for(int y = 0; y < 3; y++) {
                cJSON *row = cJSON_CreateArray();
                for(int x = 0; x < 3; x++) {
                    cJSON_AddItemToArray(row, cJSON_CreateNumber(ex.R(y, x)));
                }
                cJSON_AddItemToArray(rot, row);
            }
            cJSON *t = cJSON_CreateArray();
            cJSON_AddItemToArray(t, cJSON_CreateNumber(ex.t[0]));
            cJSON_AddItemToArray(t, cJSON_CreateNumber(ex.t[1]));
            cJSON_AddItemToArray(t, cJSON_CreateNumber(ex.t[2]));

            cJSON_AddItemToObject(camObj, "rotation", rot);
            cJSON_AddItemToObject(camObj, "translation", t);
            cJSON_AddItemToObject(root, dc.index.c_str(), camObj);
        }

        char *printed = cJSON_Print(root);
        cJSON_Delete(root);
        if(!printed) {
            return false;
        }

        fs::path p(cfg_.initExtrinsicPath);
        if(p.is_relative()) {
            p = fs::absolute(p);
        }
        const fs::path absP = fs::absolute(p);

        if(absP.has_parent_path()) {
            try {
                fs::create_directories(absP.parent_path());
            }
            catch(const std::exception &e) {
                std::cerr << "Failed to create directories: " << absP.parent_path().string() << ", error=" << e.what() << std::endl;
                cJSON_free(printed);
                return false;
            }
        }

        std::ofstream f(absP, std::ios::binary | std::ios::trunc);
        if(!f.is_open()) {
            std::cerr << "Failed to open for writing: " << absP.string() << std::endl;
            cJSON_free(printed);
            return false;
        }
        f << printed;
        f.flush();
        const bool ok = f.good();
        f.close();
        cJSON_free(printed);
        return ok;
    }

    bool writeCoarseExtrinsics(const std::string &rootIdx,
                               const std::unordered_map<std::string, EdgeExtrinsic> &edges) {
        std::unordered_map<std::string, EdgeExtrinsic> worldToCam;
        std::unordered_set<std::string> visited;
        std::deque<std::string> q;

        EdgeExtrinsic rootEx;
        rootEx.valid = true;
        worldToCam[rootIdx] = rootEx;
        visited.insert(rootIdx);
        q.push_back(rootIdx);

        while(!q.empty()) {
            const auto u = q.front();
            q.pop_front();
            const auto exU = worldToCam[u];

            for(const auto &kv: edges) {
                const auto &k = kv.first;
                const auto &e = kv.second;
                if(!e.valid) {
                    continue;
                }
                const auto pos = k.find("->");
                if(pos == std::string::npos) {
                    continue;
                }
                const auto from = k.substr(0, pos);
                const auto to   = k.substr(pos + 2);
                if(from != u) {
                    continue;
                }
                if(visited.find(to) != visited.end()) {
                    continue;
                }

                EdgeExtrinsic exV;
                exV.valid = true;
                exV.R     = e.R * exU.R;
                exV.t     = e.R * exU.t + e.t;
                worldToCam[to] = exV;
                visited.insert(to);
                q.push_back(to);
            }
        }

        for(const auto &dc: cfg_.devices) {
            if(worldToCam.find(dc.index) == worldToCam.end()) {
                std::cerr << "Missing calibration path from " << rootIdx << " to " << dc.index << std::endl;
                return false;
            }
        }

        return writeWorldToCamToFile(worldToCam);
    }

    void runIcpRefinement(std::vector<DeviceRuntime> &devices, bool ansi) {
        if(cfg_.initExtrinsicPath.empty()) {
            std::cout << colorize("init_extrinsic_path is empty", "31", ansi) << std::endl;
            return;
        }

        fs::path p(cfg_.initExtrinsicPath);
        if(p.is_relative()) {
            p = fs::absolute(p);
        }
        const fs::path absP = fs::absolute(p);

        std::unordered_map<std::string, EdgeExtrinsic> worldToCam;
        if(!loadWorldToCamFromFile(absP, worldToCam)) {
            std::cout << colorize("failed to read coarse extrinsics from file", "31", ansi) << std::endl;
            return;
        }

        for(const auto &dc: cfg_.devices) {
            const auto it = worldToCam.find(dc.index);
            if(it == worldToCam.end() || !it->second.valid) {
                std::cout << colorize("missing camera extrinsic in file: " + dc.index, "31", ansi) << std::endl;
                return;
            }
        }

        const std::string rootIdx = hasCameraIndex(devices, "00") ? std::string("00") : devices.front().cfg.index;
        int itersUsed = 0;
        if(!refineExtrinsicsWithPclIcp(devices, worldToCam, rootIdx, itersUsed)) {
            std::cout << colorize("ICP refinement failed", "31", ansi) << std::endl;
            return;
        }
        std::cout << colorize("ICP iterations: " + std::to_string(itersUsed), "36", ansi) << std::endl;

        if(writeWorldToCamToFile(worldToCam)) {
            std::cout << colorize("ICP refined extrinsics saved to: " + absP.string(), "32", ansi) << std::endl;
        }
        else {
            std::cout << colorize("failed to save ICP refined extrinsics", "31", ansi) << std::endl;
        }
    }

    void runBlockCalibration(bool ansi, bool &quit) {
        std::cout << colorize("block calibration is not implemented yet", "33", ansi) << std::endl;
        const auto input = readLine();
        if(isQuit(input)) {
            quit = true;
        }
    }

    void runChessboardCalibration(std::vector<DeviceRuntime> &devices, bool ansi, bool &quit) {
        std::unordered_set<std::string> calibratedPairKeys;
        std::vector<std::pair<std::string, std::string>> calibratedPairs;
        std::unordered_map<std::string, EdgeExtrinsic> edges;

        const std::string rootIdx = hasCameraIndex(devices, "00") ? std::string("00") : devices.front().cfg.index;

        while(!quit) {
            std::cout << "==========chessboard calibration(press \"q\" for quitting)==========" << std::endl;
            if(!calibratedPairs.empty()) {
                std::cout << "camera calibrated:" << std::endl;
                for(const auto &p: calibratedPairs) {
                    std::cout << "    " << colorize("(" + p.first + "," + p.second + ")", "32", ansi) << std::endl;
                }
            }
            std::cout << colorize("press \"c\" to save coarse calibration result", "36", ansi) << std::endl;
            std::cout << "    index of first camera(e.g. \"00\"): ";
            std::cout.flush();
            const auto first = readLine();
            if(isQuit(first)) {
                quit = true;
                break;
            }
            if(first == "c") {
                if(edges.empty()) {
                    std::cout << colorize("no calibrated pairs yet", "33", ansi) << std::endl;
                    continue;
                }
                if(writeCoarseExtrinsics(rootIdx, edges)) {
                    const fs::path absP = fs::absolute(fs::path(cfg_.initExtrinsicPath));
                    std::cout << colorize("coarse calibration saved to: " + absP.string(), "32", ansi) << std::endl;
                }
                else {
                    std::cout << colorize("failed to calculate/save coarse calibration (missing paths?)", "31", ansi) << std::endl;
                }
                continue;
            }
            if(!hasCameraIndex(devices, first)) {
                std::cout << colorize("invalid camera index: " + first, "31", ansi) << std::endl;
                continue;
            }
            std::cout << "    index of second camera(e.g. \"01\"): ";
            std::cout.flush();
            const auto second = readLine();
            if(isQuit(second)) {
                quit = true;
                break;
            }
            if(!hasCameraIndex(devices, second)) {
                std::cout << colorize("invalid camera index: " + second, "31", ansi) << std::endl;
                continue;
            }
            if(first == second) {
                std::cout << colorize("two camera indices are the same", "31", ansi) << std::endl;
                continue;
            }

            const auto np = normalizePairKey(first, second);
            if(calibratedPairKeys.find(np) != calibratedPairKeys.end()) {
                std::cout << colorize("this camera pair has been calibrated, please choose another pair", "33", ansi) << std::endl;
                continue;
            }

            const auto *rt1 = findByIndex(devices, first);
            const auto *rt2 = findByIndex(devices, second);
            if(!rt1 || !rt2) {
                std::cout << colorize("failed to locate devices for indices", "31", ansi) << std::endl;
                continue;
            }

            if(!runPairSampling(*rt1, *rt2, ansi, quit, edges)) {
                if(quit) {
                    break;
                }
                std::cout << colorize("pair calibration failed", "31", ansi) << std::endl;
                continue;
            }

            calibratedPairKeys.insert(np);
            calibratedPairs.push_back({ first, second });
        }
    }

    bool runPairSampling(const DeviceRuntime &rt1,
                         const DeviceRuntime &rt2,
                         bool ansi,
                         bool &quit,
                         std::unordered_map<std::string, EdgeExtrinsic> &edges) {
        PairPipelines pp;
        if(!startPairPipelines(rt1, rt2, pp)) {
            stopPairPipelines(pp);
            return false;
        }

        std::vector<ColorSample> samples;
        while(!quit) {
            std::cout << "==========chessboard calibration(press \"q\" for quitting)==========" << std::endl;
            std::cout << "calibrating " << rt1.cfg.index << " and " << rt2.cfg.index << std::endl;
            std::cout << "    press \"1\" to sample images" << std::endl;
            std::cout << "    press \"0\" to calculate relative extrinsic" << std::endl;
            const char key = ob_smpl::waitForKeyPressed(0);
            if(key == 'q' || key == 'Q') {
                quit = true;
                break;
            }
            if(key == '1') {
                ColorSample s;
                if(captureSyncedPair(pp, 2000, 20000, s)) {
                    samples.push_back(ColorSample{ s.ts1, s.ts2, std::move(s.img1), std::move(s.img2) });
                    std::cout << "captured image:" << std::endl;
                    std::cout << "    " << rt1.cfg.index << ": " << s.ts1 << ",  " << rt2.cfg.index << ": " << s.ts2 << std::endl;
                }
                else {
                    std::cout << colorize("capture timeout, please try again", "33", ansi) << std::endl;
                }
            }
            else if(key == '0') {
                cv::Matx33d R12;
                cv::Vec3d   t12;
                if(!computeRelativeExtrinsic(pp, samples, R12, t12)) {
                    std::cout << colorize("not enough valid chessboard samples", "33", ansi) << std::endl;
                    continue;
                }

                EdgeExtrinsic e12;
                e12.valid = true;
                e12.R     = R12;
                e12.t     = t12;
                edges[edgeKey(rt1.cfg.index, rt2.cfg.index)] = e12;

                EdgeExtrinsic e21;
                e21.valid = true;
                e21.R     = R12.t();
                e21.t     = -(e21.R * t12);
                edges[edgeKey(rt2.cfg.index, rt1.cfg.index)] = e21;

                break;
            }
        }

        stopPairPipelines(pp);
        return !quit;
    }

private:
    AppConfig cfg_;
    ob::Context ctx_;
};

int main(int argc, char **argv) {
    try {
        fs::path configPath;
        if(argc > 1) {
            configPath = fs::path(argv[1]);
        }
        else if(fs::exists("config.json")) {
            configPath = fs::path("config.json");
        }
        else {
            configPath = fs::path("src/sync/config.json");
        }
        auto cfg = loadConfig(configPath);
        if(cfg.mode == "calibration") {
            MultiDeviceCalibrator calibrator(cfg);
            return calibrator.run();
        }
        const bool runViewer = isViewerMode(cfg);

        if(runViewer) {
            MultiDeviceViewer viewer(cfg);
            return viewer.run();
        }
        else {
            std::cout << "OutputDir: " << cfg.outputDir.string() << std::endl;
            std::cout << "DurationSec: " << cfg.durationSec << std::endl;
            std::cout << "EnableSync: " << (cfg.enableSync ? "true" : "false") << std::endl;
            MultiDeviceRecorder recorder(cfg);
            return recorder.run();
        }
    }
    catch(const ob::Error &e) {
        std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs() << "\nmessage:" << e.getMessage()
                  << "\ntype:" << e.getExceptionType() << std::endl;
        return 1;
    }
    catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
