#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace sync_app {

struct EgoModuleConfig {
    bool enabled = false;
    std::string host = "127.0.0.1";
    int port = 50051;
    int stopTimeoutMs = 60000;
    size_t maxBufferedFrames = 8192;
    std::filesystem::path cameraParamsPath;
    std::string cameraId = "ego";
    int64_t timestampOffsetUs = 0;
    bool autoTimestampOffset = true;
    int64_t autoTimestampOffsetMinUs = 1000000;
};

struct EgoFrame {
    uint64_t sequence = 0;
    int sourceFrameIndex = -1;
    int videoFrameIndex = -1;
    uint64_t refTimestampUs = 0;
    uint64_t rawRefTimestampUs = 0;
    int64_t clockOffsetUs = 0;
    uint64_t rgbTimestampUs = 0;
    uint64_t acquireStartTimestampUs = 0;
    uint64_t acquireEndTimestampUs = 0;
    uint64_t picoFrameTimestampNs = 0;
    uint64_t xrHeadTimestampUs = 0;
    uint64_t gazeTimestampUs = 0;
};

class EgoRecorder {
public:
    EgoRecorder();
    ~EgoRecorder();

    EgoRecorder(const EgoRecorder &) = delete;
    EgoRecorder &operator=(const EgoRecorder &) = delete;

    bool start(const EgoModuleConfig &config, std::string *errorMessage = nullptr);
    void stop();
    bool isRunning() const;
    bool isConnected() const;
    bool waitUntilReady(std::chrono::milliseconds timeout);
    EgoModuleConfig config() const;
    std::string lastHelloSummary() const;

    bool beginSession(const std::filesystem::path &episodeDir,
                      const std::string &sessionName,
                      std::string *errorMessage = nullptr);
    bool requestStopSession(std::string *errorMessage = nullptr);
    bool stopSessionAndWait(std::chrono::milliseconds timeout, std::string *errorMessage = nullptr);
    bool isSessionActive() const;

    bool popFrame(EgoFrame &out, std::chrono::milliseconds timeout);
    bool hasPendingFrames() const;
    int videoFrameIndexForSourceFrame(int sourceFrameIndex) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sync_app
