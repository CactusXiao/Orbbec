#pragma once

#include <libobsensor/ObSensor.hpp>
#include "libobsensor/hpp/Filter.hpp"
#include "libobsensor/hpp/Utils.hpp"

#include <opencv2/opencv.hpp>

#include "fisheyes.hpp"
#include "utils/cJSON.h"

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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

namespace sync_app {

namespace fs = std::filesystem;

enum class StreamType { Color, Depth, IR, PointCloud };

std::string streamTypeToString(StreamType t);
std::optional<StreamType> streamTypeFromString(const std::string &s);

OBMultiDeviceSyncMode stringToOBSyncMode(const std::string &modeString);
OBFormat stringToOBFormat(const std::string &formatString, StreamType type);

struct StreamConfig {
    StreamType   type;
    bool         enable = true;
    int          width  = 0;
    int          height = 0;
    int          fps    = 0;
    std::string  format;
};

struct DeviceConfig {
    std::string             sn;
    std::string             index;
    OBMultiDeviceSyncConfig syncConfig{};
    bool                    hasSyncConfig = false;
    std::vector<StreamConfig> streams;
};

struct SaveOptions {
    std::string colorExt       = "jpg";
    int         jpegQuality    = 90;
    int         pngCompression = 1;
    bool        rgbH265        = false;
    std::string h265Ext        = "mp4";
    std::string h265Preset     = "medium";
    int         h265Crf        = 23;
    int         h265Threads    = 0;
    std::unordered_map<std::string, int> h265ThreadsByCamera;
};

struct ChessboardConfig {
    int   cols       = 9;
    int   rows       = 6;
    float squareSize = 0.025f;
};

struct CalibrationConfig {
    ChessboardConfig chessboard;
};

struct DepthPointCloudFiltersConfig {
    std::string preset;
    int         pointCloudDecimationFactor = 0;
    int         decimationFilterScale      = 0;
    int         noiseRemovalMaxSize        = 0;
    int         noiseRemovalMinDiff        = 0;
    double      spatialAlpha               = 0.0;
    int         spatialDispDiff            = 0;
    int         spatialMagnitude           = 0;
    int         spatialRadius              = 0;
    double      smoothThresholdM           = 0.0;
    double      temporalDiffScale          = 0.0;
    double      temporalWeight             = 0.0;
    int         holeFillingMode            = 0;
    double      confThreshold              = 0.0;
    bool        deskCrop                   = false;
};

struct AppConfig {
    fs::path                  outputDir;
    int                       durationSec   = 0;
    int                       maxFrames     = 0;
    int                       collectFps    = 0;
    std::string               mode;
    int                       viewerFps     = 30;
    std::string               initExtrinsicPath;
    float                     maxDepth       = 6.0f;
    bool                      differentColor = false;
    bool                      colorfulCloudPoints = false;
    CalibrationConfig         calibration;
    bool                      enableSync     = true;
    int                       queueCapacity  = 512;
    int                       writerThreads  = 0;
    SaveOptions               save;
    FisheyeModuleConfig       fisheye;
    DepthPointCloudFiltersConfig filters;
    std::vector<DeviceConfig> devices;
};

std::string normalizeMode(std::string s);
std::string trimString(std::string s);
std::string normalizePresetKey(std::string s);

bool isViewerMode(const AppConfig &cfg);
bool isInteractionMode(const AppConfig &cfg);

AppConfig loadConfig(const fs::path &configPath);

struct DeviceRuntime {
    DeviceConfig                  cfg;
    std::shared_ptr<ob::Device>   dev;
    std::shared_ptr<ob::Pipeline> pipe;
    int                           deviceIndex = 0;
};

std::vector<DeviceRuntime> selectDevicesWithPipeline(const std::shared_ptr<ob::DeviceList> &deviceList, const AppConfig &cfg);
void applySyncConfig(std::vector<DeviceRuntime> &devices);
void splitPrimarySecondary(const std::vector<DeviceRuntime> &all, std::vector<DeviceRuntime> &primary, std::vector<DeviceRuntime> &secondary);
bool hasSoftwareTrigger(const std::vector<DeviceRuntime> &all);

std::shared_ptr<ob::VideoStreamProfile> pickDepthProfileForPointCloud(const std::shared_ptr<ob::Pipeline> &pipe,
                                                                      const std::vector<StreamConfig> &streams);
std::shared_ptr<ob::VideoStreamProfile> pickDefaultVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe, OBSensorType sensorType);

struct OrbbecDepthFilterChain {
    std::shared_ptr<ob::DecimationFilter>      decimation;
    std::shared_ptr<ob::ThresholdFilter>       threshold;
    std::shared_ptr<ob::NoiseRemovalFilter>    noiseRemoval;
    std::shared_ptr<ob::SpatialAdvancedFilter> spatial;
    std::shared_ptr<ob::TemporalFilter>        temporal;
    std::shared_ptr<ob::HoleFillingFilter>     holeFilling;
};

pcl::PointCloud<pcl::PointXYZ>::Ptr denoiseCloudSor(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in, int meanK, double stddevMulThresh);
void applyEdgeSmoothing(std::shared_ptr<ob::Frame> &frame, double thresholdM);
std::shared_ptr<ob::Frame> refineDepthFrameForPointCloud(const std::shared_ptr<ob::Frame> &depthFrame,
                                                         OrbbecDepthFilterChain &chain,
                                                         float minDepthM,
                                                         float maxDepthM,
                                                         const DepthPointCloudFiltersConfig &filters);
pcl::PointCloud<pcl::PointXYZ>::Ptr removeDominantPlaneRansac(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                                                              int maxIterations,
                                                              double distanceThreshold,
                                                              size_t minInliers);

cv::Mat visualizeObFrame(const std::shared_ptr<const ob::Frame> &frame);

}  // namespace sync_app
