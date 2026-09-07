#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <limits>
#include <utility>
#include <string>
#include <vector>

namespace sync_app {

constexpr size_t kTactileChannelCount = 48;
constexpr size_t kTactileRegionCount = 6;
constexpr size_t kTactileChannelsPerRegion = 8;
constexpr size_t kJqShroomPressureChannelCount = 256;

struct TactileSerialConfig {
    std::string portPath;
    int         baudRate = 921600;
    int         timeoutMs = 1000;
};

struct TactileSaveOptions {
    int         csvFloatPrecision = 6;
    std::string sampleDirectoryName = "samples";
    std::string directoryName = "touch";
};

struct TactileDeviceConfig {
    std::string         streamId;
    std::string         handSide;
    int                 sensorType = 0;
    TactileSerialConfig serial;
};

struct TactileModuleConfig {
    bool                  enabled = false;
    bool                  required = true;
    std::string           streamId;
    std::string           handSide = "right";
    int                   sensorType = 2;
    int                   targetFps = 60;
    size_t                maxBufferedSamples = 8192;
    // CSV paths are resolved relative to config.json by loadConfig.
    std::vector<std::filesystem::path> calibrationPaths{
        "tactile/5 (1).csv", "tactile/5 (2).csv", "tactile/5 (3).csv"
    };
    TactileSerialConfig   serial;
    TactileSaveOptions    save;
    std::vector<TactileDeviceConfig> devices;
};

struct TactileFrame {
    uint64_t              captureTimestampUs = 0;
    double                captureTimestampSec = 0.0;
    std::string           side;
    int                   sensorType = 0;
    uint64_t              packet1TimestampUs = 0;
    uint64_t              packet2TimestampUs = 0;
    uint64_t              packetGapUs = 0;
    std::vector<uint8_t>  imuRaw;
    float                 imuW = 0.0f;
    float                 imuX = 0.0f;
    float                 imuY = 0.0f;
    float                 imuZ = 0.0f;
    bool                  imuValid = false;
    std::string           qualityFlag = "ok";
    std::vector<uint16_t> rawAdc;
    double                calibratedRegionForceN = std::numeric_limits<double>::quiet_NaN();
    bool                  forceOutOfRange = false;
    // N; NaN for channels not covered by the calibration CSVs.
    std::vector<double>   calibratedValues;
    std::vector<double>   outputValues;
};

// Uses the supplied image's Hill fit on the sum ADC of the CSV sensor channels.
// Channel numbers in the source CSV are one-based hardware sensor IDs.
class TactileForceCalibration {
public:
    // Parameters displayed in tactile/微信图片_2026-09-07_165536_559.jpg.
    static constexpr double kHillMaxAdc = 1257.0210;
    static constexpr double kHillHalfForceN = 18.2084;
    static constexpr double kHillExponent = 1.1685;
    bool load(const std::vector<std::filesystem::path> &paths, std::string *errorMessage = nullptr);
    double forceN(double adcSum) const;
    bool apply(TactileFrame &frame, std::string *errorMessage = nullptr) const;
    const std::vector<size_t> &channelIndices() const { return channelIndices_; }
private:
    std::vector<size_t> channelIndices_;
    double maxMeasuredAdcSum_ = 0.0;
};

void writeTactileMeasurementCsvHeader(std::ostream &out);
void writeTactileMeasurementCsvValues(std::ostream &out, const TactileFrame &frame, int precision);

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
    void resetCaptureCursorToLatest();
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
    uint64_t                              lastCaptureTimestampUs_ = 0;
};

}  // namespace sync_app
