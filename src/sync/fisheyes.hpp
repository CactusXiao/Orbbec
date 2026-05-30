#pragma once

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sync_app {

enum class FisheyeImageFormat {
    Jpeg,
    Png,
};

struct FisheyeCameraConfig {
    std::string cameraId = "cam0";
    std::string handRole;
    std::string uniqueId;
    std::string devicePath;
    std::string preferredDeviceHint;
    int         deviceIndex = -1;
    int         width = 1280;
    int         height = 720;
    int         fps = 60;
    bool        preferMjpeg = true;
};

struct FisheyeSaveOptions {
    FisheyeImageFormat format = FisheyeImageFormat::Jpeg;
    int                jpegQuality = 95;
    int                pngCompression = 1;
};

struct FisheyeModuleConfig {
    bool                           enabled = false;
    int                            targetFps = 60;
    size_t                         maxBufferedSets = 4096;
    std::vector<FisheyeCameraConfig> cameras = {
        FisheyeCameraConfig{ "cam0", "", "", "", "", -1, 1280, 720, 60, true },
        FisheyeCameraConfig{ "cam1", "", "", "", "", -1, 1280, 720, 60, true },
    };
    FisheyeSaveOptions             save;
};

struct FisheyeFrame {
    std::string cameraId;
    int         deviceIndex = -1;
    uint64_t    captureTimestampUs = 0;
    double      captureTimestampSec = 0.0;
    cv::Mat     bgr;
};

struct FisheyeFrameSet {
    uint64_t                 sequence = 0;
    uint64_t                 representativeTimestampUs = 0;
    double                   representativeTimestampSec = 0.0;
    std::vector<FisheyeFrame> frames;
};

struct FisheyeSavedSample {
    uint64_t                 sequence = 0;
    uint64_t                 representativeTimestampUs = 0;
    double                   representativeTimestampSec = 0.0;
    std::vector<std::string> relativePaths;
};

struct FisheyeDeviceInfo {
    std::string devicePath;
    std::string stablePath;
    std::string cardName;
    std::string busInfo;
    std::string uniqueId;
    std::string usbVendorId;
    std::string usbProductId;
    std::string usbSerial;
    std::string usbManufacturer;
    std::string usbProduct;
    std::string usbPortPath;
    bool        supportsMjpeg = false;
    bool        supportsYuyv = false;
};

struct FisheyeDatasetIndex {
    std::vector<std::string> cameraOrder;
    std::vector<FisheyeSavedSample> samples;
};

struct FisheyeNearestMatch {
    size_t                    sampleIndex = 0;
    uint64_t                  absDiffUs = 0;
    const FisheyeSavedSample *sample = nullptr;
};

std::string fisheyeImageFormatToString(FisheyeImageFormat format);
std::string fisheyeImageExtension(FisheyeImageFormat format);
std::string formatFisheyeTimestampUs(uint64_t timestampUs);
std::optional<FisheyeNearestMatch> findNearestFisheyeSample(const FisheyeDatasetIndex &index, uint64_t targetTimestampUs);
std::vector<FisheyeDeviceInfo> listAvailableFisheyeDevices();
std::vector<FisheyeDeviceInfo> listPreferredFisheyeDevices(const std::vector<std::string> &preferredLabels);

class IFisheyeModule {
public:
    virtual ~IFisheyeModule() = default;

    virtual std::string pluginId() const = 0;
    virtual bool start(const FisheyeModuleConfig &config, std::string *errorMessage = nullptr) = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual FisheyeModuleConfig config() const = 0;
    virtual bool waitUntilReady(std::chrono::milliseconds timeout) = 0;
    virtual std::optional<FisheyeFrameSet> snapshotLatest(std::string *errorMessage = nullptr) = 0;
};

std::unique_ptr<IFisheyeModule> createOpenCvFisheyeModule();

class FisheyeRecorder {
public:
    explicit FisheyeRecorder(std::unique_ptr<IFisheyeModule> module = createOpenCvFisheyeModule());
    ~FisheyeRecorder();

    bool start(const FisheyeModuleConfig &config, std::string *errorMessage = nullptr);
    void stop();
    bool isRunning() const;
    bool waitUntilReady(std::chrono::milliseconds timeout);

    std::optional<FisheyeFrameSet> snapshotLatest(std::string *errorMessage = nullptr);
    std::optional<FisheyeFrameSet> captureNext(std::string *errorMessage = nullptr);
    bool captureFor(std::chrono::milliseconds duration,
                    const std::atomic_bool *cancel = nullptr,
                    std::string *errorMessage = nullptr);

    static bool saveFrameSets(const std::vector<FisheyeFrameSet> &frameSets,
                              const std::filesystem::path &saveRoot,
                              const FisheyeSaveOptions &options,
                              FisheyeDatasetIndex *indexOut = nullptr,
                              std::string *errorMessage = nullptr);

    static bool loadDatasetIndexCsv(const std::filesystem::path &csvPath,
                                    FisheyeDatasetIndex *indexOut,
                                    std::string *errorMessage = nullptr);

private:
    std::unique_ptr<IFisheyeModule>       module_;
    FisheyeModuleConfig                   config_{};
    std::atomic_bool                      running_{ false };
    uint64_t                              nextSequence_ = 0;
    bool                                  nextCaptureTimeValid_ = false;
    std::chrono::steady_clock::time_point nextCaptureTime_{};
};

}  // namespace sync_app
