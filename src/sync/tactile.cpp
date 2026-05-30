#include "tactile.hpp"

#include <array>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <system_error>
#include <thread>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace sync_app {

namespace {

uint64_t systemClockNowUs() {
    const auto now = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

double timestampUsToSec(uint64_t timestampUs) {
    return static_cast<double>(timestampUs) / 1000000.0;
}

uint64_t parseTimestampSecToUs(const std::string &s) {
    try {
        const double tsSec = std::stod(s);
        if(tsSec <= 0.0) {
            return 0;
        }
        return static_cast<uint64_t>(tsSec * 1000000.0 + 0.5);
    }
    catch(...) {
        return 0;
    }
}

std::vector<std::string> splitCsvLineSimple(const std::string &line) {
    std::vector<std::string> parts;
    std::string current;
    for(char ch : line) {
        if(ch == ',') {
            parts.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    parts.push_back(current);
    return parts;
}

std::vector<std::string> splitTokensFlexible(const std::string &line) {
    std::string normalized = line;
    for(char &ch : normalized) {
        if(ch == ',' || ch == '\t') {
            ch = ' ';
        }
    }

    std::istringstream iss(normalized);
    std::vector<std::string> parts;
    std::string token;
    while(iss >> token) {
        parts.push_back(token);
    }
    return parts;
}

const std::vector<std::string> &regionNames() {
    static const std::vector<std::string> kNames = {
        "Thumb",
        "Index",
        "Middle",
        "Ring",
        "Pinky",
        "Palm",
    };
    return kNames;
}

class CalibrationModel {
public:
    bool load(const std::filesystem::path &path, std::string *errorMessage) {
        params_.clear();
        sourcePath_ = path;

        std::ifstream ifs(path);
        if(!ifs.is_open()) {
            if(errorMessage) {
                *errorMessage = "Failed to open tactile calibration file: " + path.string();
            }
            return false;
        }

        std::string line;
        size_t lineNo = 0;
        while(std::getline(ifs, line)) {
            ++lineNo;
            if(line.empty()) {
                continue;
            }

            const auto firstNonWs = line.find_first_not_of(" \t\r\n");
            if(firstNonWs == std::string::npos || line[firstNonWs] == '#') {
                continue;
            }

            const auto parts = splitTokensFlexible(line);
            if(parts.size() < 5) {
                continue;
            }

            try {
                TactileCalibrationEntry entry;
                entry.regionIndex = static_cast<int>(std::lround(std::stod(parts[0])));
                entry.pointIndex = static_cast<int>(std::lround(std::stod(parts[1])));
                entry.a = std::stod(parts[2]);
                entry.b = std::stod(parts[3]);
                entry.c = std::stod(parts[4]);
                entry.rsquare = parts.size() > 5 ? std::stod(parts[5]) : 0.0;
                entry.validCount = parts.size() > 6 ? static_cast<int>(std::lround(std::stod(parts[6]))) : 0;
                params_[{ entry.regionIndex, entry.pointIndex }] = entry;
            }
            catch(const std::exception &) {
                if(errorMessage) {
                    *errorMessage = "Failed to parse tactile calibration file at line " + std::to_string(lineNo);
                }
                return false;
            }
        }

        if(params_.empty()) {
            if(errorMessage) {
                *errorMessage = "Tactile calibration file contains no valid entries: " + path.string();
            }
            return false;
        }
        return true;
    }

    bool empty() const {
        return params_.empty();
    }

    double apply(size_t channelIndex, uint16_t adcValue) const {
        const int regionIndex = static_cast<int>(channelIndex / kTactileChannelsPerRegion) + 1;
        const int pointIndex = static_cast<int>(channelIndex % kTactileChannelsPerRegion) + 1;

        const auto it = params_.find({ regionIndex, pointIndex });
        if(it == params_.end()) {
            return static_cast<double>(adcValue);
        }
        return compute(static_cast<double>(adcValue), it->second);
    }

private:
    static double compute(double x, const TactileCalibrationEntry &entry) {
        if(x > entry.c) {
            x = entry.c;
        }
        const double denominator = entry.c - entry.a * x;
        if(denominator <= 0.0) {
            return entry.b > 0.0 ? (entry.b * entry.c) : 0.0;
        }
        return entry.b * entry.c * x / denominator;
    }

    std::filesystem::path                         sourcePath_;
    std::map<std::pair<int, int>, TactileCalibrationEntry> params_;
};

struct SerialFrameResult {
    bool                  ok = false;
    std::vector<uint16_t> rawAdc;
    std::string           error;
};

#if defined(__unix__) || defined(__APPLE__)
class PosixSerialPort {
public:
    ~PosixSerialPort() {
        close();
    }

    bool openDevice(const TactileSerialConfig &config, std::string *errorMessage) {
        close();

        fd_ = ::open(config.portPath.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if(fd_ < 0) {
            if(errorMessage) {
                *errorMessage = "Failed to open tactile serial port " + config.portPath + ": " + std::strerror(errno);
            }
            return false;
        }

        termios tty{};
        if(::tcgetattr(fd_, &tty) != 0) {
            if(errorMessage) {
                *errorMessage = "tcgetattr failed for tactile serial port " + config.portPath + ": " + std::strerror(errno);
            }
            close();
            return false;
        }

        ::cfmakeraw(&tty);
        tty.c_cflag |= static_cast<tcflag_t>(CLOCAL | CREAD);
        tty.c_cflag &= static_cast<tcflag_t>(~PARENB);
        tty.c_cflag &= static_cast<tcflag_t>(~CSTOPB);
        tty.c_cflag &= static_cast<tcflag_t>(~CSIZE);
        tty.c_cflag |= CS8;
#ifdef CRTSCTS
        tty.c_cflag &= static_cast<tcflag_t>(~CRTSCTS);
#endif
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;

        const speed_t speed = toSpeed(config.baudRate);
        if(speed == 0) {
            if(errorMessage) {
                *errorMessage = "Unsupported tactile serial baud rate: " + std::to_string(config.baudRate);
            }
            close();
            return false;
        }

        if(::cfsetispeed(&tty, speed) != 0 || ::cfsetospeed(&tty, speed) != 0) {
            if(errorMessage) {
                *errorMessage = "Failed to apply tactile serial baud rate on " + config.portPath + ": " + std::strerror(errno);
            }
            close();
            return false;
        }

        if(::tcsetattr(fd_, TCSANOW, &tty) != 0) {
            if(errorMessage) {
                *errorMessage = "tcsetattr failed for tactile serial port " + config.portPath + ": " + std::strerror(errno);
            }
            close();
            return false;
        }

        ::tcflush(fd_, TCIOFLUSH);
        timeoutMs_ = std::max(1, config.timeoutMs);
        return true;
    }

    void close() {
        if(fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool isOpen() const {
        return fd_ >= 0;
    }

    bool clearInputBuffer(std::string *errorMessage) {
        if(fd_ < 0) {
            if(errorMessage) {
                *errorMessage = "Tactile serial port is not open";
            }
            return false;
        }
        if(::tcflush(fd_, TCIFLUSH) != 0) {
            if(errorMessage) {
                *errorMessage = "Failed to flush tactile serial input buffer: " + std::string(std::strerror(errno));
            }
            return false;
        }
        return true;
    }

    bool writeAll(const std::string &data, std::string *errorMessage) {
        return writeAll(data.data(), data.size(), errorMessage);
    }

    bool writeAll(const void *data, size_t size, std::string *errorMessage) {
        if(fd_ < 0) {
            if(errorMessage) {
                *errorMessage = "Tactile serial port is not open";
            }
            return false;
        }

        const auto *bytes = static_cast<const uint8_t *>(data);
        size_t written = 0;
        while(written < size) {
            const ssize_t rc = ::write(fd_, bytes + written, size - written);
            if(rc > 0) {
                written += static_cast<size_t>(rc);
                continue;
            }
            if(rc < 0 && errno == EINTR) {
                continue;
            }
            if(errorMessage) {
                *errorMessage = "Failed to write tactile serial request: " + std::string(std::strerror(errno));
            }
            return false;
        }

        if(::tcdrain(fd_) != 0) {
            if(errorMessage) {
                *errorMessage = "tcdrain failed after tactile serial write: " + std::string(std::strerror(errno));
            }
            return false;
        }
        return true;
    }

    bool readExact(void *buffer, size_t size, std::string *errorMessage) {
        if(fd_ < 0) {
            if(errorMessage) {
                *errorMessage = "Tactile serial port is not open";
            }
            return false;
        }

        auto *bytes = static_cast<uint8_t *>(buffer);
        size_t totalRead = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs_);

        while(totalRead < size) {
            const auto now = std::chrono::steady_clock::now();
            if(now >= deadline) {
                if(errorMessage) {
                    *errorMessage = "Timed out while reading tactile serial frame";
                }
                return false;
            }

            const auto remainingMs = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            timeval tv{};
            tv.tv_sec = static_cast<time_t>(remainingMs.count() / 1000);
            tv.tv_usec = static_cast<suseconds_t>((remainingMs.count() % 1000) * 1000);

            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);

            const int ready = ::select(fd_ + 1, &readfds, nullptr, nullptr, &tv);
            if(ready > 0) {
                const ssize_t rc = ::read(fd_, bytes + totalRead, size - totalRead);
                if(rc > 0) {
                    totalRead += static_cast<size_t>(rc);
                    continue;
                }
                if(rc < 0 && errno == EINTR) {
                    continue;
                }
                if(errorMessage) {
                    *errorMessage = "Failed to read tactile serial frame: " + std::string(std::strerror(errno));
                }
                return false;
            }
            if(ready == 0) {
                if(errorMessage) {
                    *errorMessage = "Timed out while reading tactile serial frame";
                }
                return false;
            }
            if(errno == EINTR) {
                continue;
            }
            if(errorMessage) {
                *errorMessage = "select failed while reading tactile serial frame: " + std::string(std::strerror(errno));
            }
            return false;
        }

        return true;
    }

private:
    static speed_t toSpeed(int baudRate) {
        switch(baudRate) {
        case 9600:
            return B9600;
        case 19200:
            return B19200;
        case 38400:
            return B38400;
        case 57600:
            return B57600;
        case 115200:
            return B115200;
#ifdef B230400
        case 230400:
            return B230400;
#endif
        default:
            return static_cast<speed_t>(0);
        }
    }

    int fd_ = -1;
    int timeoutMs_ = 1000;
};
#endif

SerialFrameResult requestSerialFrame(
#if defined(__unix__) || defined(__APPLE__)
    PosixSerialPort &port,
#endif
    const TactileSerialConfig &config) {
    SerialFrameResult result;
    result.rawAdc.resize(kTactileChannelCount, 0);

#if defined(__unix__) || defined(__APPLE__)
    std::string ioError;
    if(config.clearInputBufferBeforeRequest && !port.clearInputBuffer(&ioError)) {
        result.error = ioError;
        return result;
    }
    if(!port.writeAll(config.requestCommand, &ioError)) {
        result.error = ioError;
        return result;
    }

    std::array<uint8_t, kTactileChannelCount * 2> frameBytes{};
    if(!port.readExact(frameBytes.data(), frameBytes.size(), &ioError)) {
        result.error = ioError;
        return result;
    }

    for(size_t i = 0; i < kTactileChannelCount; ++i) {
        const size_t offset = i * 2;
        result.rawAdc[i] = static_cast<uint16_t>(frameBytes[offset] | (static_cast<uint16_t>(frameBytes[offset + 1]) << 8));
    }
    result.ok = true;
#else
    (void)config;
    result.error = "Tactile serial module currently requires a POSIX platform";
#endif
    return result;
}

std::vector<TactileSerialPortInfo> enumerateSerialPorts() {
    std::vector<TactileSerialPortInfo> out;
    std::set<std::string> seen;

    auto pushPath = [&](const std::filesystem::path &path, const std::string &stablePath) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(path, ec);
        const std::string devicePath = canonical.empty() ? path.string() : canonical.string();
        if(devicePath.empty() || !seen.insert(devicePath).second) {
            return;
        }

        TactileSerialPortInfo info;
        info.devicePath = devicePath;
        info.stablePath = stablePath;
        out.push_back(std::move(info));
    };

#if defined(__linux__)
    const std::filesystem::path byIdDir("/dev/serial/by-id");
    if(std::filesystem::exists(byIdDir)) {
        for(const auto &entry : std::filesystem::directory_iterator(byIdDir)) {
            pushPath(entry.path(), entry.path().string());
        }
    }

    for(const auto &entry : std::filesystem::directory_iterator("/dev")) {
        const auto name = entry.path().filename().string();
        if(name.rfind("ttyUSB", 0) == 0 || name.rfind("ttyACM", 0) == 0) {
            pushPath(entry.path(), "");
        }
    }
#elif defined(__APPLE__)
    for(const auto &entry : std::filesystem::directory_iterator("/dev")) {
        const auto name = entry.path().filename().string();
        if(name.rfind("tty.usb", 0) == 0 || name.rfind("cu.usb", 0) == 0) {
            pushPath(entry.path(), "");
        }
    }
#endif

    std::sort(out.begin(), out.end(), [](const TactileSerialPortInfo &a, const TactileSerialPortInfo &b) {
        const std::string ka = !a.stablePath.empty() ? a.stablePath : a.devicePath;
        const std::string kb = !b.stablePath.empty() ? b.stablePath : b.devicePath;
        return ka < kb;
    });
    return out;
}

class PosixSerialTactileModule final : public ITactileModule {
public:
    ~PosixSerialTactileModule() override {
        stop();
    }

    std::string pluginId() const override {
        return "posix_serial";
    }

    bool start(const TactileModuleConfig &config, std::string *errorMessage) override {
        stop();

        if(config.serial.portPath.empty()) {
            if(errorMessage) {
                *errorMessage = "Tactile serial port path is empty";
            }
            return false;
        }

        auto state = std::make_shared<State>();
        state->config = config;

        if(config.applyCalibration && !config.calibrationPath.empty()) {
            if(!state->calibration.load(config.calibrationPath, errorMessage)) {
                return false;
            }
        }

#if defined(__unix__) || defined(__APPLE__)
        if(!state->port.openDevice(config.serial, errorMessage)) {
            return false;
        }
#else
        if(errorMessage) {
            *errorMessage = "Tactile serial module currently requires a POSIX platform";
        }
        return false;
#endif

        {
            std::lock_guard<std::mutex> lock(moduleMtx_);
            config_ = config;
            state_ = state;
            running_.store(true);
        }

        state->worker = std::thread([this, state]() { captureLoop(*state); });
        return true;
    }

    void stop() override {
        std::shared_ptr<State> state;
        {
            std::lock_guard<std::mutex> lock(moduleMtx_);
            running_.store(false);
            state = state_;
            state_.reset();
        }

        if(state) {
            state->stopRequested.store(true);
            if(state->worker.joinable()) {
                state->worker.join();
            }
#if defined(__unix__) || defined(__APPLE__)
            state->port.close();
#endif
        }

        {
            std::lock_guard<std::mutex> lock(moduleMtx_);
            readyCv_.notify_all();
        }
    }

    bool isRunning() const override {
        return running_.load();
    }

    TactileModuleConfig config() const override {
        std::lock_guard<std::mutex> lock(moduleMtx_);
        return config_;
    }

    bool waitUntilReady(std::chrono::milliseconds timeout) override {
        std::unique_lock<std::mutex> lock(moduleMtx_);
        return readyCv_.wait_for(lock, timeout, [&]() {
            return !running_.load() || (state_ && state_->ready.load());
        });
    }

    std::optional<TactileSample> snapshotLatest(std::string *errorMessage) override {
        std::shared_ptr<State> state;
        {
            std::lock_guard<std::mutex> lock(moduleMtx_);
            if(!running_.load() || !state_) {
                if(errorMessage) {
                    *errorMessage = "Tactile module is not running";
                }
                return std::nullopt;
            }
            state = state_;
        }

        std::lock_guard<std::mutex> frameLock(state->sampleMtx);
        if(!state->ready.load() || state->latest.frame.rawAdc.size() != kTactileChannelCount) {
            if(errorMessage) {
                *errorMessage = state->lastError.empty() ? "Tactile module has no ready frame yet" : state->lastError;
            }
            return std::nullopt;
        }

        TactileSample sample = state->latest;
        sample.sequence = 0;
        return sample;
    }

private:
    struct State {
        TactileModuleConfig config;
        CalibrationModel    calibration;
#if defined(__unix__) || defined(__APPLE__)
        PosixSerialPort     port;
#endif
        std::thread         worker;
        std::atomic_bool    stopRequested{ false };
        std::atomic_bool    ready{ false };
        std::mutex          sampleMtx;
        TactileSample       latest;
        std::string         lastError;
    };

    static TactileSample buildSample(const SerialFrameResult &frame,
                                     const TactileModuleConfig &config,
                                     const CalibrationModel &calibration) {
        TactileSample sample;
        sample.representativeTimestampUs = systemClockNowUs();
        sample.representativeTimestampSec = timestampUsToSec(sample.representativeTimestampUs);
        sample.frame.captureTimestampUs = sample.representativeTimestampUs;
        sample.frame.captureTimestampSec = sample.representativeTimestampSec;
        sample.frame.rawAdc = frame.rawAdc;
        sample.frame.calibratedValues.reserve(frame.rawAdc.size());
        sample.frame.outputValues.reserve(frame.rawAdc.size());

        for(size_t i = 0; i < frame.rawAdc.size(); ++i) {
            const double calibrated = (!config.applyCalibration || calibration.empty())
                                          ? static_cast<double>(frame.rawAdc[i])
                                          : calibration.apply(i, frame.rawAdc[i]);
            sample.frame.calibratedValues.push_back(calibrated);
            sample.frame.outputValues.push_back(calibrated);
        }
        return sample;
    }

    void captureLoop(State &state) {
        while(!state.stopRequested.load()) {
            const auto frame = requestSerialFrame(
#if defined(__unix__) || defined(__APPLE__)
                state.port,
#endif
                state.config.serial);
            if(!frame.ok) {
                {
                    std::lock_guard<std::mutex> lock(state.sampleMtx);
                    state.lastError = frame.error.empty() ? "Unknown tactile serial read error" : frame.error;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            TactileSample sample = buildSample(frame, state.config, state.calibration);
            bool becameReady = false;
            {
                std::lock_guard<std::mutex> lock(state.sampleMtx);
                state.latest = std::move(sample);
                state.lastError.clear();
                if(!state.ready.load()) {
                    state.ready.store(true);
                    becameReady = true;
                }
            }

            if(becameReady) {
                std::lock_guard<std::mutex> lock(moduleMtx_);
                readyCv_.notify_all();
            }
        }
    }

    mutable std::mutex              moduleMtx_;
    std::condition_variable         readyCv_;
    TactileModuleConfig             config_{};
    std::shared_ptr<State>          state_;
    std::atomic_bool                running_{ false };
};

bool saveSingleSampleCsv(const TactileSample &sample,
                         const std::filesystem::path &path,
                         const TactileSaveOptions &options,
                         std::string *errorMessage) {
    std::ofstream ofs(path);
    if(!ofs.is_open()) {
        if(errorMessage) {
            *errorMessage = "Failed to open tactile sample csv for writing: " + path.string();
        }
        return false;
    }

    ofs << "channel_index,region_id,region_name,point_id,raw_adc,calibrated_value,output_value\n";
    ofs << std::fixed << std::setprecision(std::max(0, options.csvFloatPrecision));
    const auto &names = regionNames();
    for(size_t i = 0; i < sample.frame.rawAdc.size(); ++i) {
        const size_t regionIndex = i / kTactileChannelsPerRegion;
        const size_t pointIndex = i % kTactileChannelsPerRegion;
        const double calibrated = i < sample.frame.calibratedValues.size() ? sample.frame.calibratedValues[i] : 0.0;
        const double output = i < sample.frame.outputValues.size() ? sample.frame.outputValues[i] : calibrated;
        ofs << i
            << "," << (regionIndex + 1)
            << "," << names[regionIndex]
            << "," << (pointIndex + 1)
            << "," << sample.frame.rawAdc[i]
            << "," << calibrated
            << "," << output
            << "\n";
    }
    return true;
}

}  // namespace

std::string formatTactileTimestampUs(uint64_t timestampUs) {
    std::ostringstream oss;
    oss << (timestampUs / 1000000ULL) << "." << std::setw(6) << std::setfill('0') << (timestampUs % 1000000ULL);
    return oss.str();
}

std::vector<std::string> tactileRegionNamesEn() {
    return regionNames();
}

std::optional<TactileNearestMatch> findNearestTactileSample(const TactileDatasetIndex &index, uint64_t targetTimestampUs) {
    if(index.samples.empty()) {
        return std::nullopt;
    }

    auto absDiff = [](uint64_t a, uint64_t b) {
        return a > b ? (a - b) : (b - a);
    };

    auto it = std::lower_bound(index.samples.begin(), index.samples.end(), targetTimestampUs,
                               [](const TactileSavedSample &sample, uint64_t tsUs) {
                                   return sample.representativeTimestampUs < tsUs;
                               });

    size_t bestIndex = 0;
    uint64_t bestDiff = std::numeric_limits<uint64_t>::max();
    if(it != index.samples.end()) {
        bestIndex = static_cast<size_t>(std::distance(index.samples.begin(), it));
        bestDiff = absDiff(index.samples[bestIndex].representativeTimestampUs, targetTimestampUs);
    }
    if(it != index.samples.begin()) {
        const size_t candidate = static_cast<size_t>(std::distance(index.samples.begin(), it - 1));
        const uint64_t diff = absDiff(index.samples[candidate].representativeTimestampUs, targetTimestampUs);
        if(diff <= bestDiff) {
            bestIndex = candidate;
            bestDiff = diff;
        }
    }

    return TactileNearestMatch{ bestIndex, bestDiff, &index.samples[bestIndex] };
}

std::vector<TactileSerialPortInfo> listAvailableTactileSerialPorts() {
    return enumerateSerialPorts();
}

std::unique_ptr<ITactileModule> createPosixSerialTactileModule() {
    return std::make_unique<PosixSerialTactileModule>();
}

TactileRecorder::TactileRecorder(std::unique_ptr<ITactileModule> module)
    : module_(std::move(module)) {
}

TactileRecorder::~TactileRecorder() {
    stop();
}

bool TactileRecorder::start(const TactileModuleConfig &config, std::string *errorMessage) {
    stop();

    if(!module_) {
        module_ = createPosixSerialTactileModule();
    }
    if(!module_) {
        if(errorMessage) {
            *errorMessage = "Failed to create tactile module";
        }
        return false;
    }
    if(!module_->start(config, errorMessage)) {
        return false;
    }

    config_ = config;
    running_.store(true);
    nextSequence_ = 0;
    nextCaptureTimeValid_ = false;
    clearBuffered();
    return true;
}

void TactileRecorder::stop() {
    running_.store(false);
    nextCaptureTimeValid_ = false;
    if(module_) {
        module_->stop();
    }
}

bool TactileRecorder::isRunning() const {
    return running_.load() && module_ && module_->isRunning();
}

bool TactileRecorder::waitUntilReady(std::chrono::milliseconds timeout) {
    return module_ && module_->waitUntilReady(timeout);
}

std::optional<TactileSample> TactileRecorder::snapshotLatest(std::string *errorMessage) {
    if(!isRunning()) {
        if(errorMessage) {
            *errorMessage = "Tactile recorder is not running";
        }
        return std::nullopt;
    }
    return module_->snapshotLatest(errorMessage);
}

std::optional<TactileSample> TactileRecorder::captureNext(std::string *errorMessage) {
    if(!isRunning()) {
        if(errorMessage) {
            *errorMessage = "Tactile recorder is not running";
        }
        return std::nullopt;
    }

    const int targetFps = std::max(1, config_.targetFps);
    const auto period = std::chrono::microseconds(static_cast<int64_t>(1000000.0 / static_cast<double>(targetFps)));

    const auto now = std::chrono::steady_clock::now();
    if(!nextCaptureTimeValid_) {
        nextCaptureTime_ = now;
        nextCaptureTimeValid_ = true;
    }
    if(now < nextCaptureTime_) {
        std::this_thread::sleep_until(nextCaptureTime_);
    }

    auto sample = module_->snapshotLatest(errorMessage);
    if(!sample) {
        return std::nullopt;
    }
    sample->sequence = nextSequence_++;

    {
        std::lock_guard<std::mutex> lock(bufferMtx_);
        buffered_.push_back(*sample);
        if(config_.maxBufferedSamples > 0 && buffered_.size() > config_.maxBufferedSamples) {
            buffered_.erase(buffered_.begin());
        }
    }

    nextCaptureTime_ += period;
    if(nextCaptureTime_ < std::chrono::steady_clock::now()) {
        nextCaptureTime_ = std::chrono::steady_clock::now();
    }
    return sample;
}

bool TactileRecorder::captureFor(std::chrono::milliseconds duration,
                                 const std::atomic_bool *cancel,
                                 std::string *errorMessage) {
    if(duration.count() <= 0) {
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() + duration;
    while(std::chrono::steady_clock::now() < deadline) {
        if(cancel && cancel->load()) {
            return true;
        }
        if(!captureNext(errorMessage)) {
            return false;
        }
    }
    return true;
}

std::vector<TactileSample> TactileRecorder::bufferedSamplesCopy() const {
    std::lock_guard<std::mutex> lock(bufferMtx_);
    return buffered_;
}

std::vector<TactileSample> TactileRecorder::takeBufferedSamples() {
    std::lock_guard<std::mutex> lock(bufferMtx_);
    std::vector<TactileSample> out;
    out.swap(buffered_);
    return out;
}

void TactileRecorder::clearBuffered() {
    std::lock_guard<std::mutex> lock(bufferMtx_);
    buffered_.clear();
}

bool TactileRecorder::saveBufferedSession(const std::filesystem::path &saveRoot,
                                          TactileDatasetIndex *indexOut,
                                          std::string *errorMessage) const {
    return saveSamples(bufferedSamplesCopy(), saveRoot, config_.save, indexOut, errorMessage);
}

bool TactileRecorder::saveSamples(const std::vector<TactileSample> &samples,
                                  const std::filesystem::path &saveRoot,
                                  const TactileSaveOptions &options,
                                  TactileDatasetIndex *indexOut,
                                  std::string *errorMessage) {
    if(samples.empty()) {
        if(errorMessage) {
            *errorMessage = "No tactile samples to save";
        }
        return false;
    }
    if(saveRoot.empty()) {
        if(errorMessage) {
            *errorMessage = "Tactile save path is empty";
        }
        return false;
    }

    const std::filesystem::path samplesDir = saveRoot / options.sampleDirectoryName;
    try {
        std::filesystem::create_directories(samplesDir);
    }
    catch(const std::filesystem::filesystem_error &ex) {
        if(errorMessage) {
            *errorMessage = ex.what();
        }
        return false;
    }

    TactileDatasetIndex localIndex;
    localIndex.samples.reserve(samples.size());

    const std::filesystem::path csvPath = saveRoot / "timestamps.csv";
    const std::filesystem::path csvTmpPath = saveRoot / "timestamps.csv.tmp";
    std::ofstream csv(csvTmpPath);
    if(!csv.is_open()) {
        if(errorMessage) {
            *errorMessage = "Failed to open tactile timestamps csv for writing: " + csvTmpPath.string();
        }
        return false;
    }
    csv << "frame_id,timestamp_s,tactile_file\n";

    for(size_t i = 0; i < samples.size(); ++i) {
        const auto &sample = samples[i];
        const std::string tsString = formatTactileTimestampUs(sample.representativeTimestampUs);
        const std::filesystem::path relativePath = std::filesystem::path(options.sampleDirectoryName) / (tsString + ".csv");
        if(!saveSingleSampleCsv(sample, saveRoot / relativePath, options, errorMessage)) {
            return false;
        }

        TactileSavedSample saved;
        saved.sequence = sample.sequence;
        saved.representativeTimestampUs = sample.representativeTimestampUs;
        saved.representativeTimestampSec = sample.representativeTimestampSec;
        saved.relativePath = relativePath.generic_string();
        localIndex.samples.push_back(saved);

        csv << i << "," << tsString << "," << saved.relativePath << "\n";
    }
    csv.close();

    try {
        std::filesystem::rename(csvTmpPath, csvPath);
    }
    catch(const std::filesystem::filesystem_error &ex) {
        if(errorMessage) {
            *errorMessage = ex.what();
        }
        return false;
    }

    if(indexOut) {
        *indexOut = std::move(localIndex);
    }
    return true;
}

bool TactileRecorder::loadDatasetIndexCsv(const std::filesystem::path &csvPath,
                                          TactileDatasetIndex *indexOut,
                                          std::string *errorMessage) {
    if(!indexOut) {
        if(errorMessage) {
            *errorMessage = "indexOut must not be null";
        }
        return false;
    }

    std::ifstream ifs(csvPath);
    if(!ifs.is_open()) {
        if(errorMessage) {
            *errorMessage = "Failed to open tactile timestamps csv: " + csvPath.string();
        }
        return false;
    }

    std::string headerLine;
    if(!std::getline(ifs, headerLine)) {
        if(errorMessage) {
            *errorMessage = "Tactile timestamps csv is empty";
        }
        return false;
    }

    const auto headers = splitCsvLineSimple(headerLine);
    if(headers.size() != 3 || headers[0] != "frame_id" || headers[1] != "timestamp_s" || headers[2] != "tactile_file") {
        if(errorMessage) {
            *errorMessage = "Unexpected tactile timestamps csv header";
        }
        return false;
    }

    TactileDatasetIndex parsed;
    std::string line;
    while(std::getline(ifs, line)) {
        if(line.empty()) {
            continue;
        }

        const auto cols = splitCsvLineSimple(line);
        if(cols.size() != headers.size()) {
            if(errorMessage) {
                *errorMessage = "Unexpected tactile timestamps csv row width";
            }
            return false;
        }

        TactileSavedSample sample;
        try {
            sample.sequence = static_cast<uint64_t>(std::stoull(cols[0]));
        }
        catch(...) {
            if(errorMessage) {
                *errorMessage = "Invalid tactile frame_id in timestamps csv";
            }
            return false;
        }
        sample.representativeTimestampUs = parseTimestampSecToUs(cols[1]);
        sample.representativeTimestampSec = timestampUsToSec(sample.representativeTimestampUs);
        sample.relativePath = cols[2];
        parsed.samples.push_back(std::move(sample));
    }

    *indexOut = std::move(parsed);
    return true;
}

}  // namespace sync_app
