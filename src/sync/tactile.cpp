#include "tactile.hpp"

#include <array>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
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

std::string sideForSensorType(int sensorType) {
    if(sensorType == 1) {
        return "left";
    }
    if(sensorType == 2) {
        return "right";
    }
    return "";
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

struct SerialFrameResult {
    bool                  ok = false;
    std::string           side;
    int                   sensorType = 0;
    uint64_t              timestampUs = 0;
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

    ssize_t readSome(void *buffer, size_t capacity, int timeoutMs, std::string *errorMessage) {
        if(fd_ < 0) {
            if(errorMessage) {
                *errorMessage = "Tactile serial port is not open";
            }
            return -1;
        }
        if(capacity == 0) {
            return 0;
        }

        timeval tv{};
        const int boundedTimeoutMs = std::max(1, timeoutMs);
        tv.tv_sec = static_cast<time_t>(boundedTimeoutMs / 1000);
        tv.tv_usec = static_cast<suseconds_t>((boundedTimeoutMs % 1000) * 1000);

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(fd_, &readfds);

        while(true) {
            const int ready = ::select(fd_ + 1, &readfds, nullptr, nullptr, &tv);
            if(ready > 0) {
                const ssize_t rc = ::read(fd_, buffer, capacity);
                if(rc >= 0) {
                    return rc;
                }
                if(errno == EINTR) {
                    continue;
                }
                if(errorMessage) {
                    *errorMessage = "Failed to read tactile serial stream: " + std::string(std::strerror(errno));
                }
                return -1;
            }
            if(ready == 0) {
                return 0;
            }
            if(errno == EINTR) {
                continue;
            }
            if(errorMessage) {
                *errorMessage = "select failed while reading tactile serial stream: " + std::string(std::strerror(errno));
            }
            return -1;
        }
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
#ifdef B460800
        case 460800:
            return B460800;
#endif
#ifdef B500000
        case 500000:
            return B500000;
#endif
#ifdef B576000
        case 576000:
            return B576000;
#endif
#ifdef B921600
        case 921600:
            return B921600;
#endif
#ifdef B1000000
        case 1000000:
            return B1000000;
#endif
        default:
            return static_cast<speed_t>(0);
        }
    }

    int fd_ = -1;
};
#endif

struct JqPendingPacket {
    std::vector<uint8_t> payload;
    uint64_t             timestampUs = 0;
};

struct JqParserState {
    std::vector<uint8_t> buffer;
    std::map<int, JqPendingPacket> firstPacketBySensor;
    size_t parseErrors = 0;
    size_t droppedPacket1 = 0;
    size_t droppedPacket2 = 0;
};

float readFloat32Le(const uint8_t *bytes) {
    uint32_t raw = static_cast<uint32_t>(bytes[0])
                 | (static_cast<uint32_t>(bytes[1]) << 8)
                 | (static_cast<uint32_t>(bytes[2]) << 16)
                 | (static_cast<uint32_t>(bytes[3]) << 24);
    float out = 0.0f;
    std::memcpy(&out, &raw, sizeof(float));
    return out;
}

std::optional<SerialFrameResult> parseOneJqFrame(JqParserState &state, const TactileModuleConfig &config) {
    static constexpr std::array<uint8_t, 4> kHeader = { 0xAA, 0x55, 0x03, 0x99 };

    for(;;) {
        if(state.buffer.size() < kHeader.size() + 2) {
            return std::nullopt;
        }

        auto headerIt = std::search(state.buffer.begin(), state.buffer.end(), kHeader.begin(), kHeader.end());
        if(headerIt == state.buffer.end()) {
            if(state.buffer.size() > kHeader.size() - 1) {
                state.buffer.erase(state.buffer.begin(), state.buffer.end() - static_cast<std::ptrdiff_t>(kHeader.size() - 1));
            }
            return std::nullopt;
        }
        if(headerIt != state.buffer.begin()) {
            state.buffer.erase(state.buffer.begin(), headerIt);
        }
        if(state.buffer.size() < kHeader.size() + 2) {
            return std::nullopt;
        }

        const uint8_t packetId = state.buffer[4];
        const uint8_t sensorType = state.buffer[5];
        size_t payloadLen = 0;
        if(packetId == 0x01) {
            payloadLen = 128;
        }
        else if(packetId == 0x02) {
            payloadLen = 144;
        }
        else {
            state.parseErrors++;
            state.buffer.erase(state.buffer.begin());
            continue;
        }

        const size_t packetLen = 6 + payloadLen;
        if(state.buffer.size() < packetLen) {
            return std::nullopt;
        }

        std::vector<uint8_t> payload(state.buffer.begin() + 6, state.buffer.begin() + static_cast<std::ptrdiff_t>(packetLen));
        state.buffer.erase(state.buffer.begin(), state.buffer.begin() + static_cast<std::ptrdiff_t>(packetLen));
        const uint64_t arrivalUs = systemClockNowUs();

        if(config.sensorType > 0 && static_cast<int>(sensorType) != config.sensorType) {
            continue;
        }

        if(packetId == 0x01) {
            JqPendingPacket pending;
            pending.payload = std::move(payload);
            pending.timestampUs = arrivalUs;
            state.firstPacketBySensor[static_cast<int>(sensorType)] = std::move(pending);
            continue;
        }

        auto itFirst = state.firstPacketBySensor.find(static_cast<int>(sensorType));
        if(itFirst == state.firstPacketBySensor.end()) {
            state.droppedPacket1++;
            continue;
        }

        JqPendingPacket first = std::move(itFirst->second);
        state.firstPacketBySensor.erase(itFirst);
        if(first.payload.size() != 128 || payload.size() < 144) {
            state.parseErrors++;
            continue;
        }

        SerialFrameResult result;
        result.ok = true;
        result.sensorType = static_cast<int>(sensorType);
        result.side = !config.handSide.empty() ? config.handSide : sideForSensorType(result.sensorType);
        result.packet1TimestampUs = first.timestampUs;
        result.packet2TimestampUs = arrivalUs;
        result.packetGapUs = arrivalUs >= first.timestampUs ? (arrivalUs - first.timestampUs) : 0;
        result.timestampUs = std::max(result.packet1TimestampUs, result.packet2TimestampUs);
        result.rawAdc.reserve(kJqShroomPressureChannelCount);
        for(uint8_t value : first.payload) {
            result.rawAdc.push_back(static_cast<uint16_t>(value));
        }
        for(size_t i = 0; i < 128; ++i) {
            result.rawAdc.push_back(static_cast<uint16_t>(payload[i]));
        }
        result.imuRaw.assign(payload.begin() + 128, payload.begin() + 144);
        if(result.imuRaw.size() == 16) {
            result.imuW = readFloat32Le(result.imuRaw.data());
            result.imuX = readFloat32Le(result.imuRaw.data() + 4);
            result.imuY = readFloat32Le(result.imuRaw.data() + 8);
            result.imuZ = readFloat32Le(result.imuRaw.data() + 12);
            result.imuValid = true;
        }
        return result;
    }
}

SerialFrameResult readJqShroomFrame(
#if defined(__unix__) || defined(__APPLE__)
    PosixSerialPort &port,
#endif
    JqParserState &parserState,
    const TactileModuleConfig &config) {
    SerialFrameResult result;
#if defined(__unix__) || defined(__APPLE__)
    std::array<uint8_t, 4096> chunk{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(1, config.serial.timeoutMs));
    while(std::chrono::steady_clock::now() < deadline) {
        if(auto parsed = parseOneJqFrame(parserState, config)) {
            return *parsed;
        }

        std::string ioError;
        const int remainingMs = static_cast<int>(std::max<int64_t>(
            1,
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count()));
        const ssize_t n = port.readSome(chunk.data(), chunk.size(), remainingMs, &ioError);
        if(n < 0) {
            result.error = ioError;
            return result;
        }
        if(n == 0) {
            continue;
        }
        parserState.buffer.insert(parserState.buffer.end(), chunk.begin(), chunk.begin() + n);
        if(parserState.buffer.size() > 8192) {
            parserState.buffer.erase(parserState.buffer.begin(),
                                     parserState.buffer.end() - static_cast<std::ptrdiff_t>(4096));
            parserState.parseErrors++;
        }
    }
    result.error = "Timed out while reading JQ/Shroom tactile frame";
#else
    (void)parserState;
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
        return "jq_shroom_serial";
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
        if(!state->calibration.load(config.calibrationPaths, errorMessage)) {
            return false;
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
        if(!state->ready.load() || state->latest.frame.rawAdc.empty()) {
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
        TactileForceCalibration calibration;
#if defined(__unix__) || defined(__APPLE__)
        PosixSerialPort     port;
#endif
        std::thread         worker;
        std::atomic_bool    stopRequested{ false };
        std::atomic_bool    ready{ false };
        std::mutex          sampleMtx;
        TactileSample       latest;
        std::string         lastError;
        JqParserState       jqParser;
    };

    static TactileSample buildSample(const SerialFrameResult &frame,
                                     const TactileModuleConfig &config) {
        TactileSample sample;
        sample.representativeTimestampUs = frame.timestampUs != 0 ? frame.timestampUs : systemClockNowUs();
        sample.representativeTimestampSec = timestampUsToSec(sample.representativeTimestampUs);
        sample.frame.captureTimestampUs = sample.representativeTimestampUs;
        sample.frame.captureTimestampSec = sample.representativeTimestampSec;
        sample.frame.side = frame.side.empty() ? config.handSide : frame.side;
        sample.frame.sensorType = frame.sensorType;
        sample.frame.packet1TimestampUs = frame.packet1TimestampUs;
        sample.frame.packet2TimestampUs = frame.packet2TimestampUs;
        sample.frame.packetGapUs = frame.packetGapUs;
        sample.frame.imuRaw = frame.imuRaw;
        sample.frame.imuW = frame.imuW;
        sample.frame.imuX = frame.imuX;
        sample.frame.imuY = frame.imuY;
        sample.frame.imuZ = frame.imuZ;
        sample.frame.imuValid = frame.imuValid;
        sample.frame.qualityFlag = frame.qualityFlag.empty() ? "ok" : frame.qualityFlag;
        sample.frame.rawAdc = frame.rawAdc;
        return sample;
    }

    void captureLoop(State &state) {
        while(!state.stopRequested.load()) {
            const auto frame = readJqShroomFrame(
#if defined(__unix__) || defined(__APPLE__)
                state.port,
#endif
                state.jqParser,
                state.config);
            if(!frame.ok) {
                {
                    std::lock_guard<std::mutex> lock(state.sampleMtx);
                    state.lastError = frame.error.empty() ? "Unknown tactile serial read error" : frame.error;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            TactileSample sample = buildSample(frame, state.config);
            std::string calibrationError;
            if(!state.calibration.apply(sample.frame, &calibrationError)) {
                std::lock_guard<std::mutex> lock(state.sampleMtx);
                state.lastError = calibrationError;
                continue;
            }
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

    ofs << "channel_index,region_id,region_name,point_id,raw_adc,calibrated_value,output_value,force_unit,calibrated_region_force_n,force_out_of_range\n";
    ofs << std::fixed << std::setprecision(std::max(0, options.csvFloatPrecision));
    const auto &names = regionNames();
    for(size_t i = 0; i < sample.frame.rawAdc.size(); ++i) {
        const size_t regionIndex = i / kTactileChannelsPerRegion;
        const size_t pointIndex = i % kTactileChannelsPerRegion;
        const std::string regionName = regionIndex < names.size()
            ? names[regionIndex]
            : ("SensorBlock" + std::to_string(regionIndex + 1));
        const double calibrated = i < sample.frame.calibratedValues.size() ? sample.frame.calibratedValues[i] : std::numeric_limits<double>::quiet_NaN();
        const double output = i < sample.frame.outputValues.size() ? sample.frame.outputValues[i] : calibrated;
        ofs << i
            << "," << (regionIndex + 1)
            << "," << regionName
            << "," << (pointIndex + 1)
            << "," << sample.frame.rawAdc[i]
            << "," << calibrated
            << "," << output
            << ",N," << sample.frame.calibratedRegionForceN
            << "," << (sample.frame.forceOutOfRange ? 1 : 0)
            << "\n";
    }
    return true;
}

}  // namespace

bool TactileForceCalibration::load(const std::vector<std::filesystem::path> &paths, std::string *errorMessage) {
    channelIndices_.clear();
    maxMeasuredAdcSum_ = 0.0;
    std::vector<size_t> channels;
    // CSVs define channel membership and the observed range, not new fit parameters.
    double maxMeasuredAdcSum = 0.0;
    auto fail = [&](const std::string &message) {
        if(errorMessage) *errorMessage = message;
        return false;
    };
    if(paths.empty()) return fail("No tactile force calibration CSV configured");
    for(const auto &path : paths) {
        std::ifstream in(path);
        std::string line;
        if(!in || !std::getline(in, line)) return fail("Cannot read tactile calibration: " + path.string());
        const auto header = splitCsvLineSimple(line);
        if(header.size() < 3 || header[1].find("(N)") == std::string::npos) {
            return fail("Expected time,force(N),sensor#... calibration CSV: " + path.string());
        }
        std::vector<size_t> fileChannels;
        try {
            for(size_t i = 2; i < header.size(); ++i) {
                const auto hash = header[i].find('#');
                if(hash == std::string::npos) throw std::runtime_error("Missing sensor number");
                std::istringstream idStream(header[i].substr(hash + 1));
                int id = 0;
                if(!(idStream >> id) || !(idStream >> std::ws).eof()
                   || id < 1 || id > static_cast<int>(kJqShroomPressureChannelCount)) {
                    throw std::runtime_error("Invalid sensor number");
                }
                fileChannels.push_back(static_cast<size_t>(id - 1));
            }
            if(std::set<size_t>(fileChannels.begin(), fileChannels.end()).size() != fileChannels.size()) {
                throw std::runtime_error("Duplicate sensor number");
            }
            auto sorted = fileChannels;
            std::sort(sorted.begin(), sorted.end());
            if(channels.empty()) channels = sorted;
            if(channels != sorted) throw std::runtime_error("Calibration sensor sets differ");
        }
        catch(const std::exception &ex) {
            return fail(path.string() + ": " + ex.what());
        }
        size_t lineNo = 1;
        size_t positiveCount = 0;
        while(std::getline(in, line)) {
            ++lineNo;
            if(line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const auto columns = splitCsvLineSimple(line);
            try {
                if(columns.size() != header.size()) throw std::runtime_error("Wrong row width");
                auto number = [](const std::string &value) {
                    size_t consumed = 0;
                    double result = std::stod(value, &consumed);
                    if(!std::isfinite(result)
                       || value.find_first_not_of(" \t\r", consumed) != std::string::npos) {
                        throw std::runtime_error("Invalid finite number");
                    }
                    return result;
                };
                const double force = number(columns[1]);
                double adcSum = 0.0;
                for(size_t i = 2; i < columns.size(); ++i) {
                    double adc = number(columns[i]);
                    if(adc < 0.0 || adc > 255.0) throw std::runtime_error("ADC outside uint8 range");
                    adcSum += adc;
                }
                // No-contact samples anchor the origin; zero-ADC force readings
                // cannot identify a response and must not create an offset.
                if(adcSum > 0.0) {
                    if(force < 0.0) throw std::runtime_error("Negative force with nonzero ADC");
                    maxMeasuredAdcSum = std::max(maxMeasuredAdcSum, adcSum);
                    if(force > 0.0) ++positiveCount;
                }
            }
            catch(const std::exception &ex) {
                return fail(path.string() + ":" + std::to_string(lineNo) + ": " + ex.what());
            }
        }
        if(!in.eof()) return fail("Error reading tactile calibration: " + path.string());
        if(positiveCount == 0) return fail("No positive force measurements: " + path.string());
    }
    channelIndices_ = std::move(channels);
    maxMeasuredAdcSum_ = maxMeasuredAdcSum;
    return true;
}

double TactileForceCalibration::forceN(double adcSum) const {
    // The inverse is defined for 0 <= ADC < a. Do not clamp valid high
    // readings: the requested Hill curve intentionally grows near saturation.
    if(channelIndices_.empty() || !std::isfinite(adcSum) || adcSum < 0.0 || adcSum >= kHillMaxAdc) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if(adcSum == 0.0) return 0.0;
    const double force = kHillHalfForceN * std::pow(adcSum / (kHillMaxAdc - adcSum), 1.0 / kHillExponent);
    return std::isfinite(force) ? force : std::numeric_limits<double>::quiet_NaN();
}

bool TactileForceCalibration::apply(TactileFrame &frame, std::string *errorMessage) const {
    frame.calibratedValues.assign(frame.rawAdc.size(), std::numeric_limits<double>::quiet_NaN());
    frame.outputValues = frame.calibratedValues;
    frame.calibratedRegionForceN = std::numeric_limits<double>::quiet_NaN();
    frame.forceOutOfRange = false;
    if(channelIndices_.empty() || channelIndices_.back() >= frame.rawAdc.size()) {
        if(errorMessage) *errorMessage = "Missing tactile force calibration or incomplete ADC frame";
        return false;
    }
    double adcSum = 0.0;
    for(size_t i : channelIndices_) adcSum += frame.rawAdc[i];
    frame.calibratedRegionForceN = forceN(adcSum);
    frame.forceOutOfRange = adcSum > maxMeasuredAdcSum_ || !std::isfinite(frame.calibratedRegionForceN);
    for(size_t i : channelIndices_) {
        frame.calibratedValues[i] = adcSum > 0.0 ? frame.calibratedRegionForceN * frame.rawAdc[i] / adcSum : 0.0;
    }
    frame.outputValues = frame.calibratedValues;
    return true;
}

void writeTactileMeasurementCsvHeader(std::ostream &out) {
    out << ",calibrated_region_force_n,force_out_of_range";
    for(size_t i = 0; i < kJqShroomPressureChannelCount; ++i) {
        out << ",force_" << std::setw(3) << std::setfill('0') << i << "_n";
    }
    for(size_t i = 0; i < kJqShroomPressureChannelCount; ++i) {
        out << ",raw_adc_" << std::setw(3) << std::setfill('0') << i;
    }
    out << std::setfill(' ');
}

void writeTactileMeasurementCsvValues(std::ostream &out, const TactileFrame &frame, int precision) {
    const auto flags = out.flags();
    const auto oldPrecision = out.precision();
    out << std::fixed << std::setprecision(std::max(0, precision));
    out << "," << frame.calibratedRegionForceN << "," << (frame.forceOutOfRange ? 1 : 0);
    for(size_t i = 0; i < kJqShroomPressureChannelCount; ++i) {
        out << ",";
        if(i < frame.outputValues.size()) out << frame.outputValues[i];
        else out << "nan";
    }
    for(size_t i = 0; i < kJqShroomPressureChannelCount; ++i) {
        out << ",";
        if(i < frame.rawAdc.size()) out << frame.rawAdc[i];
    }
    out.flags(flags);
    out.precision(oldPrecision);
}

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
    lastCaptureTimestampUs_ = 0;
    clearBuffered();
    return true;
}

void TactileRecorder::stop() {
    running_.store(false);
    lastCaptureTimestampUs_ = 0;
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

void TactileRecorder::resetCaptureCursorToLatest() {
    lastCaptureTimestampUs_ = 0;
    if(!isRunning()) {
        return;
    }
    auto sample = module_->snapshotLatest(nullptr);
    if(sample) {
        lastCaptureTimestampUs_ = sample->representativeTimestampUs;
    }
}

std::optional<TactileSample> TactileRecorder::captureNext(std::string *errorMessage) {
    if(!isRunning()) {
        if(errorMessage) {
            *errorMessage = "Tactile recorder is not running";
        }
        return std::nullopt;
    }

    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(std::max(1, config_.serial.timeoutMs));
    std::string lastError;
    while(isRunning() && std::chrono::steady_clock::now() < deadline) {
        std::string snapshotError;
        auto sample = module_->snapshotLatest(&snapshotError);
        if(sample && sample->representativeTimestampUs != lastCaptureTimestampUs_) {
            lastCaptureTimestampUs_ = sample->representativeTimestampUs;
            sample->sequence = nextSequence_++;

            {
                std::lock_guard<std::mutex> lock(bufferMtx_);
                buffered_.push_back(*sample);
                if(config_.maxBufferedSamples > 0 && buffered_.size() > config_.maxBufferedSamples) {
                    buffered_.erase(buffered_.begin());
                }
            }
            return sample;
        }

        if(!snapshotError.empty()) {
            lastError = std::move(snapshotError);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if(errorMessage) {
        if(!isRunning()) {
            *errorMessage = "Tactile recorder is not running";
        }
        else {
            *errorMessage = lastError.empty() ? "Timed out waiting for next tactile frame" : lastError;
        }
    }
    return std::nullopt;
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
