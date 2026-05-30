#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sync_app {

constexpr size_t kTactileChannelCount = 48;
constexpr size_t kTactileRegionCount = 6;
constexpr size_t kTactileChannelsPerRegion = 8;

struct TactileCalibrationEntry {
    int    regionIndex = 0;
    int    pointIndex = 0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double rsquare = 0.0;
    int    validCount = 0;
};

struct TactileSerialConfig {
    std::string portPath;
    int         baudRate = 115200;
    int         timeoutMs = 1000;
    std::string requestCommand = "A\r\n";
    bool        clearInputBufferBeforeRequest = true;
};

struct TactileSaveOptions {
    int         csvFloatPrecision = 6;
    std::string sampleDirectoryName = "samples";
};

struct TactileModuleConfig {
    bool                  enabled = false;
    int                   targetFps = 60;
    size_t                maxBufferedSamples = 4096;
    bool                  applyCalibration = true;
    std::filesystem::path calibrationPath;
    TactileSerialConfig   serial;
    TactileSaveOptions    save;
};

struct TactileFrame {
    uint64_t              captureTimestampUs = 0;
    double                captureTimestampSec = 0.0;
    std::vector<uint16_t> rawAdc;
    std::vector<double>   calibratedValues;
    std::vector<double>   outputValues;
};

struct TactileSample {
    uint64_t     sequence = 0;
    uint64_t     representativeTimestampUs = 0;
    double       representativeTimestampSec = 0.0;
    TactileFrame frame;
};

struct TactileSavedSample {
    uint64_t    sequence = 0;
    uint64_t    representativeTimestampUs = 0;
    double      representativeTimestampSec = 0.0;
    std::string relativePath;
};

struct TactileDatasetIndex {
    std::vector<TactileSavedSample> samples;
};

struct TactileNearestMatch {
    size_t                    sampleIndex = 0;
    uint64_t                  absDiffUs = 0;
    const TactileSavedSample *sample = nullptr;
};

struct TactileSerialPortInfo {
    std::string devicePath;
    std::string stablePath;
};

std::string formatTactileTimestampUs(uint64_t timestampUs);
std::vector<std::string> tactileRegionNamesEn();
std::optional<TactileNearestMatch> findNearestTactileSample(const TactileDatasetIndex &index, uint64_t targetTimestampUs);
std::vector<TactileSerialPortInfo> listAvailableTactileSerialPorts();

class ITactileModule {
public:
    virtual ~ITactileModule() = default;

    virtual std::string pluginId() const = 0;
    virtual bool start(const TactileModuleConfig &config, std::string *errorMessage = nullptr) = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual TactileModuleConfig config() const = 0;
    virtual bool waitUntilReady(std::chrono::milliseconds timeout) = 0;
    virtual std::optional<TactileSample> snapshotLatest(std::string *errorMessage = nullptr) = 0;
};

std::unique_ptr<ITactileModule> createPosixSerialTactileModule();

class TactileRecorder {
public:
    explicit TactileRecorder(std::unique_ptr<ITactileModule> module = createPosixSerialTactileModule());
    ~TactileRecorder();

    bool start(const TactileModuleConfig &config, std::string *errorMessage = nullptr);
    void stop();
    bool isRunning() const;
    bool waitUntilReady(std::chrono::milliseconds timeout);

    std::optional<TactileSample> snapshotLatest(std::string *errorMessage = nullptr);
    std::optional<TactileSample> captureNext(std::string *errorMessage = nullptr);
    bool captureFor(std::chrono::milliseconds duration,
                    const std::atomic_bool *cancel = nullptr,
                    std::string *errorMessage = nullptr);

    std::vector<TactileSample> bufferedSamplesCopy() const;
    std::vector<TactileSample> takeBufferedSamples();
    void clearBuffered();

    bool saveBufferedSession(const std::filesystem::path &saveRoot,
                             TactileDatasetIndex *indexOut = nullptr,
                             std::string *errorMessage = nullptr) const;

    static bool saveSamples(const std::vector<TactileSample> &samples,
                            const std::filesystem::path &saveRoot,
                            const TactileSaveOptions &options,
                            TactileDatasetIndex *indexOut = nullptr,
                            std::string *errorMessage = nullptr);

    static bool loadDatasetIndexCsv(const std::filesystem::path &csvPath,
                                    TactileDatasetIndex *indexOut,
                                    std::string *errorMessage = nullptr);

private:
    std::unique_ptr<ITactileModule>       module_;
    TactileModuleConfig                   config_{};
    std::atomic_bool                      running_{ false };
    mutable std::mutex                    bufferMtx_;
    std::vector<TactileSample>            buffered_;
    uint64_t                              nextSequence_ = 0;
    bool                                  nextCaptureTimeValid_ = false;
    std::chrono::steady_clock::time_point nextCaptureTime_{};
};

}  // namespace sync_app
