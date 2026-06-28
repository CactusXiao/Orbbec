#include "ego.hpp"

#include "utils/cJSON.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace sync_app {

namespace {

constexpr uint32_t kPacketMagic = 0x50454731; // "PEG1"
constexpr uint8_t  kPacketVersion = 1;
constexpr size_t   kPacketHeaderBytes = 20;

enum PacketType : uint8_t {
    PKT_HELLO = 1,
    PKT_START = 2,
    PKT_STOP = 3,
    PKT_CAMERA_JSON = 4,
    PKT_METADATA_ROW = 5,
    PKT_TIMESTAMP_ROW = 6,
    PKT_HEVC_SAMPLE = 7,
    PKT_SESSION_END = 8,
    PKT_ERROR = 9,
    PKT_METADATA_HEADER = 10,
    PKT_TIMESTAMP_HEADER = 11,
};

struct Packet {
    uint8_t type = 0;
    std::string headerJson;
    std::vector<uint8_t> payload;
};

uint64_t unixUsNow() {
    const auto now = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

std::string sanitizeSessionName(std::string value) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while(!value.empty() && isSpace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while(!value.empty() && isSpace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    if(value.empty()) {
        value = "ego_session";
    }

    std::string out;
    out.reserve(value.size());
    for(char ch: value) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if(std::isalnum(c) || ch == '-' || ch == '_' || ch == '.') {
            out.push_back(ch);
        }
        else {
            out.push_back('_');
        }
    }
    return out.empty() ? "ego_session" : out;
}

std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for(char ch: s) {
        switch(ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::string jsonString(const std::string &s) {
    return "\"" + jsonEscape(s) + "\"";
}

std::string readTextFile(const std::filesystem::path &path) {
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs.is_open()) {
        return "";
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

bool writeTextFile(const std::filesystem::path &path, const std::string &content) {
    std::ofstream ofs(path, std::ios::binary);
    if(!ofs.is_open()) {
        return false;
    }
    ofs << content;
    return static_cast<bool>(ofs);
}

std::vector<std::string> splitCsvSimple(const std::string &line) {
    std::vector<std::string> out;
    std::string current;
    bool inQuotes = false;
    for(size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if(ch == '"') {
            if(inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            }
            else {
                inQuotes = !inQuotes;
            }
            continue;
        }
        if(ch == ',' && !inQuotes) {
            out.push_back(current);
            current.clear();
            continue;
        }
        if(ch != '\r' && ch != '\n') {
            current.push_back(ch);
        }
    }
    out.push_back(current);
    return out;
}

int parseIntOr(const std::string &s, int fallback) {
    try {
        size_t idx = 0;
        const int v = std::stoi(s, &idx);
        return idx == 0 ? fallback : v;
    }
    catch(...) {
        return fallback;
    }
}

uint64_t parseUint64Or(const std::string &s, uint64_t fallback = 0) {
    try {
        size_t idx = 0;
        const auto v = std::stoull(s, &idx);
        return idx == 0 ? fallback : static_cast<uint64_t>(v);
    }
    catch(...) {
        return fallback;
    }
}

std::optional<int64_t> jsonInt64Field(const std::string &json, const char *key) {
    cJSON *root = cJSON_Parse(json.c_str());
    if(!root) {
        return std::nullopt;
    }
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    std::optional<int64_t> out;
    if(item && cJSON_IsNumber(item)) {
        out = static_cast<int64_t>(item->valuedouble);
    }
    cJSON_Delete(root);
    return out;
}

std::optional<bool> jsonBoolField(const std::string &json, const char *key) {
    cJSON *root = cJSON_Parse(json.c_str());
    if(!root) {
        return std::nullopt;
    }
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    std::optional<bool> out;
    if(item) {
        if(cJSON_IsBool(item)) {
            out = cJSON_IsTrue(item);
        }
        else if(cJSON_IsNumber(item)) {
            out = item->valuedouble != 0.0;
        }
        else if(cJSON_IsString(item) && item->valuestring) {
            std::string v = item->valuestring;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if(v == "true" || v == "1" || v == "yes") {
                out = true;
            }
            else if(v == "false" || v == "0" || v == "no") {
                out = false;
            }
        }
    }
    cJSON_Delete(root);
    return out;
}

std::string packetTypeName(uint8_t type) {
    switch(type) {
    case PKT_HELLO:
        return "HELLO";
    case PKT_START:
        return "START";
    case PKT_STOP:
        return "STOP";
    case PKT_CAMERA_JSON:
        return "CAMERA_JSON";
    case PKT_METADATA_ROW:
        return "METADATA_ROW";
    case PKT_TIMESTAMP_ROW:
        return "TIMESTAMP_ROW";
    case PKT_HEVC_SAMPLE:
        return "HEVC_SAMPLE";
    case PKT_SESSION_END:
        return "SESSION_END";
    case PKT_ERROR:
        return "ERROR";
    case PKT_METADATA_HEADER:
        return "METADATA_HEADER";
    case PKT_TIMESTAMP_HEADER:
        return "TIMESTAMP_HEADER";
    default:
        return std::to_string(static_cast<int>(type));
    }
}

void putU16Be(uint8_t *p, uint16_t v) {
    p[0] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[1] = static_cast<uint8_t>(v & 0xFF);
}

void putU32Be(uint8_t *p, uint32_t v) {
    p[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[3] = static_cast<uint8_t>(v & 0xFF);
}

void putU64Be(uint8_t *p, uint64_t v) {
    for(int i = 7; i >= 0; --i) {
        p[7 - i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFF);
    }
}

uint16_t getU16Be(const uint8_t *p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

uint32_t getU32Be(const uint8_t *p) {
    return (static_cast<uint32_t>(p[0]) << 24)
           | (static_cast<uint32_t>(p[1]) << 16)
           | (static_cast<uint32_t>(p[2]) << 8)
           | static_cast<uint32_t>(p[3]);
}

uint64_t getU64Be(const uint8_t *p) {
    uint64_t v = 0;
    for(size_t i = 0; i < 8; ++i) {
        v = (v << 8) | p[i];
    }
    return v;
}

bool sendAll(int fd, const uint8_t *data, size_t size) {
    size_t sent = 0;
    while(sent < size) {
#if defined(MSG_NOSIGNAL)
        const int flags = MSG_NOSIGNAL;
#else
        const int flags = 0;
#endif
        const ssize_t n = ::send(fd, data + sent, size - sent, flags);
        if(n <= 0) {
            if(errno == EINTR) {
                continue;
            }
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool recvExact(int fd, uint8_t *data, size_t size) {
    size_t got = 0;
    while(got < size) {
        const ssize_t n = ::recv(fd, data + got, size - got, 0);
        if(n <= 0) {
            if(n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        got += static_cast<size_t>(n);
    }
    return true;
}

bool sendPacketFd(int fd, uint8_t type, const std::string &headerJson, const std::vector<uint8_t> &payload) {
    std::array<uint8_t, kPacketHeaderBytes> header{};
    putU32Be(header.data(), kPacketMagic);
    header[4] = kPacketVersion;
    header[5] = type;
    putU16Be(header.data() + 6, 0);
    putU32Be(header.data() + 8, static_cast<uint32_t>(headerJson.size()));
    putU64Be(header.data() + 12, static_cast<uint64_t>(payload.size()));
    if(!sendAll(fd, header.data(), header.size())) {
        return false;
    }
    if(!headerJson.empty() && !sendAll(fd, reinterpret_cast<const uint8_t *>(headerJson.data()), headerJson.size())) {
        return false;
    }
    if(!payload.empty() && !sendAll(fd, payload.data(), payload.size())) {
        return false;
    }
    return true;
}

bool recvPacketFd(int fd, Packet &packet, std::string *errorMessage) {
    std::array<uint8_t, kPacketHeaderBytes> header{};
    if(!recvExact(fd, header.data(), header.size())) {
        if(errorMessage) {
            *errorMessage = "socket closed";
        }
        return false;
    }
    if(getU32Be(header.data()) != kPacketMagic) {
        if(errorMessage) {
            *errorMessage = "invalid packet magic";
        }
        return false;
    }
    if(header[4] != kPacketVersion) {
        if(errorMessage) {
            *errorMessage = "unsupported protocol version";
        }
        return false;
    }
    const uint32_t headerLen = getU32Be(header.data() + 8);
    const uint64_t payloadLen = getU64Be(header.data() + 12);
    if(headerLen > 1024 * 1024 || payloadLen > 512ULL * 1024ULL * 1024ULL) {
        if(errorMessage) {
            *errorMessage = "packet too large";
        }
        return false;
    }
    packet = Packet{};
    packet.type = header[5];
    packet.headerJson.resize(headerLen);
    if(headerLen > 0 && !recvExact(fd, reinterpret_cast<uint8_t *>(packet.headerJson.data()), headerLen)) {
        if(errorMessage) {
            *errorMessage = "failed to read packet header json";
        }
        return false;
    }
    if(packet.headerJson.empty()) {
        packet.headerJson = "{}";
    }
    packet.payload.resize(static_cast<size_t>(payloadLen));
    if(payloadLen > 0 && !recvExact(fd, packet.payload.data(), static_cast<size_t>(payloadLen))) {
        if(errorMessage) {
            *errorMessage = "failed to read packet payload";
        }
        return false;
    }
    return true;
}

void closeFd(int &fd) {
    if(fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        fd = -1;
    }
}

bool loadFixedEgoCameraParams(const std::filesystem::path &path,
                              const std::string &cameraId,
                              cJSON **outRoot,
                              std::string *errorMessage) {
    if(outRoot) {
        *outRoot = nullptr;
    }
    try {
        if(path.empty()) {
            if(errorMessage) {
                *errorMessage = "Ego camera params path is empty";
            }
            return false;
        }
        if(!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
            if(errorMessage) {
                *errorMessage = "Ego camera params file not found: " + path.string();
            }
            return false;
        }
    }
    catch(const std::exception &ex) {
        if(errorMessage) {
            *errorMessage = std::string("Failed to check ego camera params file: ") + ex.what();
        }
        return false;
    }

    const std::string content = readTextFile(path);
    if(content.empty()) {
        if(errorMessage) {
            *errorMessage = "Ego camera params file is empty or unreadable: " + path.string();
        }
        return false;
    }
    cJSON *root = cJSON_Parse(content.c_str());
    if(!root || !cJSON_IsObject(root)) {
        if(root) {
            cJSON_Delete(root);
        }
        if(errorMessage) {
            *errorMessage = "Invalid ego camera params JSON: " + path.string();
        }
        return false;
    }
    const std::string key = cameraId.empty() ? "ego" : cameraId;
    cJSON *camObj = cJSON_GetObjectItemCaseSensitive(root, key.c_str());
    if(!camObj || !cJSON_IsObject(camObj)) {
        cJSON_Delete(root);
        if(errorMessage) {
            *errorMessage = "Ego camera params JSON must contain top-level object: " + key;
        }
        return false;
    }
    cJSON *rgbObj = cJSON_GetObjectItemCaseSensitive(camObj, "RGB");
    cJSON *intrObj = rgbObj ? cJSON_GetObjectItemCaseSensitive(rgbObj, "intrinsic") : nullptr;
    cJSON *fxObj = intrObj ? cJSON_GetObjectItemCaseSensitive(intrObj, "fx") : nullptr;
    cJSON *fyObj = intrObj ? cJSON_GetObjectItemCaseSensitive(intrObj, "fy") : nullptr;
    if(!rgbObj || !cJSON_IsObject(rgbObj) || !intrObj || !cJSON_IsObject(intrObj)
       || !fxObj || !cJSON_IsNumber(fxObj) || !fyObj || !cJSON_IsNumber(fyObj)) {
        cJSON_Delete(root);
        if(errorMessage) {
            *errorMessage = "Ego camera params JSON must contain " + key + ".RGB.intrinsic.fx/fy";
        }
        return false;
    }
    if(outRoot) {
        *outRoot = root;
    }
    else {
        cJSON_Delete(root);
    }
    return true;
}

bool validateEgoCameraParamsFile(const EgoModuleConfig &config, std::string *errorMessage) {
    cJSON *root = nullptr;
    if(!loadFixedEgoCameraParams(config.cameraParamsPath, config.cameraId, &root, errorMessage)) {
        return false;
    }
    cJSON_Delete(root);
    return true;
}

bool writeEgoCameraParamsJson(const std::filesystem::path &egoDir,
                              const EgoModuleConfig &config,
                              std::string *errorMessage) {
    cJSON *root = nullptr;
    if(!loadFixedEgoCameraParams(config.cameraParamsPath, config.cameraId, &root, errorMessage)) {
        return false;
    }
    char *printed = cJSON_Print(root);
    bool ok = false;
    if(printed) {
        ok = writeTextFile(egoDir / "camera_params.json", printed);
        cJSON_free(printed);
    }
    cJSON_Delete(root);
    if(!ok && errorMessage) {
        *errorMessage = "Failed to write ego camera_params.json under " + egoDir.string();
    }
    return ok;
}

std::string headerJsonWithServerTime(const std::string &sessionName) {
    std::ostringstream oss;
    oss << "{\"session_name\":" << jsonString(sessionName)
        << ",\"server_unix_us\":" << unixUsNow() << "}";
    return oss.str();
}

}  // namespace

class EgoRecorder::Impl {
public:
    ~Impl() {
        stop();
    }

    bool start(const EgoModuleConfig &config, std::string *errorMessage) {
        stop();
        config_ = config;
        if(config_.port <= 0 || config_.port > 65535) {
            if(errorMessage) {
                *errorMessage = "Invalid ego TCP port";
            }
            return false;
        }
        if(!validateEgoCameraParamsFile(config_, errorMessage)) {
            return false;
        }

#if !(defined(__unix__) || defined(__APPLE__))
        if(errorMessage) {
            *errorMessage = "Ego TCP recorder is only implemented for POSIX platforms";
        }
        return false;
#else
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if(fd < 0) {
            if(errorMessage) {
                *errorMessage = std::string("socket failed: ") + std::strerror(errno);
            }
            return false;
        }

        int opt = 1;
        (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#if defined(SO_NOSIGPIPE)
        (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &opt, sizeof(opt));
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(config_.port));
        if(config_.host.empty() || config_.host == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        }
        else if(::inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) != 1) {
            closeFd(fd);
            if(errorMessage) {
                *errorMessage = "Invalid ego host address: " + config_.host;
            }
            return false;
        }

        if(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
            const std::string msg = std::string("bind failed: ") + std::strerror(errno);
            closeFd(fd);
            if(errorMessage) {
                *errorMessage = msg;
            }
            return false;
        }
        if(::listen(fd, 1) != 0) {
            const std::string msg = std::string("listen failed: ") + std::strerror(errno);
            closeFd(fd);
            if(errorMessage) {
                *errorMessage = msg;
            }
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            listenerFd_ = fd;
            stopRequested_.store(false);
            running_.store(true);
            connected_ = false;
            helloReceived_ = false;
            lastHelloJson_.clear();
        }

        acceptThread_ = std::thread([this]() { acceptLoop(); });
        std::cerr << "[ego] listening on " << config_.host << ":" << config_.port << std::endl;
        return true;
#endif
    }

    void stop() {
        stopRequested_.store(true);
        requestStopSession(nullptr);
        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            closeFd(listenerFd_);
            closeFd(clientFd_);
            connected_ = false;
            helloReceived_ = false;
        }
        readyCv_.notify_all();
        sessionCv_.notify_all();
        frameCv_.notify_all();
        std::thread acceptThread;
        std::thread readerThread;
        {
            std::lock_guard<std::mutex> lock(threadMtx_);
            if(acceptThread_.joinable()) {
                acceptThread = std::move(acceptThread_);
            }
            if(readerThread_.joinable()) {
                readerThread = std::move(readerThread_);
            }
        }
        if(acceptThread.joinable()) {
            acceptThread.join();
        }
        if(readerThread.joinable()) {
            readerThread.join();
        }
        {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            closeSessionLocked("server_stop");
        }
        {
            std::lock_guard<std::mutex> lock(frameMtx_);
            frameQueue_.clear();
        }
        running_.store(false);
    }

    bool isRunning() const {
        return running_.load();
    }

    bool isConnected() const {
        std::lock_guard<std::mutex> lock(stateMtx_);
        return connected_ && helloReceived_;
    }

    bool waitUntilReady(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(stateMtx_);
        return readyCv_.wait_for(lock, timeout, [&]() {
            return stopRequested_.load() || (connected_ && helloReceived_);
        });
    }

    EgoModuleConfig config() const {
        return config_;
    }

    std::string lastHelloSummary() const {
        std::lock_guard<std::mutex> lock(stateMtx_);
        return lastHelloJson_;
    }

    bool beginSession(const std::filesystem::path &episodeDir,
                      const std::string &sessionName,
                      std::string *errorMessage) {
        if(!isConnected()) {
            if(errorMessage) {
                *errorMessage = "PICO ego client is not connected";
            }
            return false;
        }

        const std::string safeName = sanitizeSessionName(sessionName);
        auto writer = std::make_unique<SessionWriter>();
        if(!writer->open(episodeDir, safeName, config_, errorMessage)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> frameLock(frameMtx_);
            frameQueue_.clear();
            nextFrameSequence_ = 0;
        }

        {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            if(session_) {
                if(errorMessage) {
                    *errorMessage = "Ego session is already active";
                }
                return false;
            }
            session_ = std::move(writer);
        }

        const std::string header = headerJsonWithServerTime(safeName);
        if(!sendPacket(PKT_START, header, {}, errorMessage)) {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            closeSessionLocked("start_send_failed");
            return false;
        }
        std::cerr << "[ego] START sent session=" << safeName << std::endl;
        return true;
    }

    bool requestStopSession(std::string *errorMessage) {
        if(!isSessionActive()) {
            return true;
        }
        const std::string header = "{\"server_unix_us\":" + std::to_string(unixUsNow()) + "}";
        if(!sendPacket(PKT_STOP, header, {}, errorMessage)) {
            return false;
        }
        std::cerr << "[ego] STOP sent" << std::endl;
        return true;
    }

    bool stopSessionAndWait(std::chrono::milliseconds timeout, std::string *errorMessage) {
        (void)requestStopSession(errorMessage);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        {
            std::unique_lock<std::mutex> lock(sessionMtx_);
            sessionCv_.wait_until(lock, deadline, [&]() {
                return !session_;
            });
            if(session_) {
                std::cerr << "[ego] warning: stop timed out, finalizing ego session locally" << std::endl;
                closeSessionLocked("stop_timeout");
            }
        }
        return true;
    }

    bool isSessionActive() const {
        std::lock_guard<std::mutex> lock(sessionMtx_);
        return static_cast<bool>(session_);
    }

    bool popFrame(EgoFrame &out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(frameMtx_);
        if(!frameCv_.wait_for(lock, timeout, [&]() {
               return !frameQueue_.empty() || stopRequested_.load();
           })) {
            return false;
        }
        if(frameQueue_.empty()) {
            return false;
        }
        out = frameQueue_.front();
        frameQueue_.pop_front();
        return true;
    }

    bool hasPendingFrames() const {
        std::lock_guard<std::mutex> lock(frameMtx_);
        return !frameQueue_.empty();
    }

    bool hasFramePayload(int sourceFrameIndex) const;
    bool commitFrame(int sourceFrameIndex, std::string *errorMessage);
    void discardFramesBefore(int sourceFrameIndex);

private:
    class SessionWriter {
        struct BufferedHevcSample {
            int frameIndex = -1;
            bool codecConfig = false;
            std::string headerJson;
            std::vector<uint8_t> payload;
            uint64_t receivedUnixUs = 0;
        };

    public:
        ~SessionWriter() {
            close("destroyed");
        }

        bool open(const std::filesystem::path &episodeDir,
                  const std::string &sessionName,
                  const EgoModuleConfig &config,
                  std::string *errorMessage) {
            sessionName_ = sessionName;
            egoDir_ = episodeDir / "ego";
            rgbDir_ = egoDir_ / "RGB";
            videoPath_ = rgbDir_ / "rgb.h265";
            metadataPath_ = egoDir_ / "metadata.csv";
            timestampsPath_ = egoDir_ / "timestamps.csv";
            cameraPath_ = egoDir_ / "camera.json";
            sessionJsonPath_ = egoDir_ / "session.json";
            networkLogPath_ = egoDir_ / "network_log.jsonl";
            cameraParamsSourcePath_ = config.cameraParamsPath;
            metadataTmpPath_ = metadataPath_;
            metadataTmpPath_ += ".tmp";
            timestampsTmpPath_ = timestampsPath_;
            timestampsTmpPath_ += ".tmp";

            try {
                std::filesystem::create_directories(rgbDir_);
                if(!writeEgoCameraParamsJson(egoDir_, config, errorMessage)) {
                    return false;
                }
            }
            catch(const std::exception &ex) {
                if(errorMessage) {
                    *errorMessage = ex.what();
                }
                return false;
            }

            video_.open(videoPath_, std::ios::binary | std::ios::out | std::ios::trunc);
            metadata_.open(metadataTmpPath_, std::ios::binary | std::ios::out | std::ios::trunc);
            timestamps_.open(timestampsTmpPath_, std::ios::binary | std::ios::out | std::ios::trunc);
            networkLog_.open(networkLogPath_, std::ios::binary | std::ios::out | std::ios::trunc);
            if(!video_.is_open() || !metadata_.is_open() || !timestamps_.is_open() || !networkLog_.is_open()) {
                if(errorMessage) {
                    *errorMessage = "Failed to open ego output files under " + egoDir_.string();
                }
                close("open_failed");
                return false;
            }
            startedUnixUs_ = unixUsNow();
            logRaw("{\"event\":\"session_open\",\"server_unix_us\":" + std::to_string(startedUnixUs_) + "}");
            return true;
        }

        void writeCameraJson(const std::vector<uint8_t> &payload) {
            std::ofstream ofs(cameraPath_, std::ios::binary | std::ios::out | std::ios::trunc);
            if(ofs.is_open() && !payload.empty()) {
                ofs.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
                cameraJsonReceived_ = true;
            }
            logEventWithBytes("camera_json", payload.size());
        }

        void writeMetadataHeader(const std::vector<uint8_t> &payload) {
            metadata_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            metadata_.flush();
            logEventWithBytes("metadata_header", payload.size());
        }

        void writeMetadataRow(const std::vector<uint8_t> &payload) {
            metadata_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            metadataRows_++;
        }

        void writeTimestampHeader(const std::vector<uint8_t> &payload) {
            timestamps_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            timestamps_.flush();
            timestampHeader_ = splitCsvSimple(std::string(reinterpret_cast<const char *>(payload.data()), payload.size()));
            if(!timestampHeader_.empty() && timestampHeader_.back().empty()) {
                timestampHeader_.pop_back();
            }
            timestampIndex_.clear();
            for(size_t i = 0; i < timestampHeader_.size(); ++i) {
                timestampIndex_[timestampHeader_[i]] = i;
            }
            logEventWithBytes("timestamp_header", payload.size());
        }

        std::optional<EgoFrame> writeTimestampRow(const std::vector<uint8_t> &payload, uint64_t sequence) {
            timestamps_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            timestampRows_++;
            const std::string line(reinterpret_cast<const char *>(payload.data()), payload.size());
            const auto cols = splitCsvSimple(line);
            auto col = [&](const std::string &name) -> std::string {
                auto it = timestampIndex_.find(name);
                if(it == timestampIndex_.end() || it->second >= cols.size()) {
                    return "";
                }
                return cols[it->second];
            };
            EgoFrame frame;
            frame.sequence = sequence;
            frame.sourceFrameIndex = parseIntOr(col("frame_index"), -1);
            frame.refTimestampUs = parseUint64Or(col("ref_timestamp_us"));
            frame.rgbTimestampUs = parseUint64Or(col("pico_rgb_timestamp_us"));
            frame.acquireStartTimestampUs = parseUint64Or(col("pico_rgb_acquire_start_timestamp_us"));
            frame.acquireEndTimestampUs = parseUint64Or(col("pico_rgb_acquire_end_timestamp_us"));
            frame.picoFrameTimestampNs = parseUint64Or(col("pico_frame_timestamp_ns"));
            frame.xrHeadTimestampUs = parseUint64Or(col("pico_xr_head_timestamp_us"));
            frame.gazeTimestampUs = parseUint64Or(col("pico_gaze_timestamp_us"));
            if(frame.refTimestampUs == 0) {
                frame.refTimestampUs = frame.rgbTimestampUs;
            }
            if(frame.refTimestampUs == 0) {
                return std::nullopt;
            }
            return frame;
        }

        void writeHevcSample(const std::string &headerJson, const std::vector<uint8_t> &payload) {
            BufferedHevcSample sample;
            const auto frameIndex = jsonInt64Field(headerJson, "frame_index");
            if(frameIndex.has_value()) {
                sample.frameIndex = static_cast<int>(*frameIndex);
            }
            sample.codecConfig = jsonBoolField(headerJson, "is_codec_config").value_or(sample.frameIndex < 0);
            sample.headerJson = headerJson;
            sample.payload = payload;
            sample.receivedUnixUs = unixUsNow();

            hevcSamplesReceived_++;
            hevcBytesReceived_ += sample.payload.size();
            logHevcSampleEvent("hevc_sample_received", sample, std::nullopt);

            if(sample.codecConfig || sample.frameIndex < 0) {
                if(wroteCodecConfig_ || hevcFramesCommitted_ > 0) {
                    hevcSamplesDropped_++;
                    hevcBytesDropped_ += sample.payload.size();
                    logHevcSampleEvent("hevc_codec_config_late_dropped", sample, std::nullopt);
                    return;
                }
                codecConfigSamples_.push_back(std::move(sample));
                return;
            }

            if(committedHevcSourceFrames_.find(sample.frameIndex) != committedHevcSourceFrames_.end()) {
                hevcSamplesDropped_++;
                hevcBytesDropped_ += sample.payload.size();
                logHevcSampleEvent("hevc_sample_duplicate_dropped", sample, std::nullopt);
                return;
            }
            pendingHevcByFrame_[sample.frameIndex].push_back(std::move(sample));
        }

        bool hasHevcFrame(int sourceFrameIndex) const {
            if(sourceFrameIndex < 0) {
                return false;
            }
            auto it = pendingHevcByFrame_.find(sourceFrameIndex);
            return it != pendingHevcByFrame_.end() && !it->second.empty();
        }

        bool commitHevcFrame(int sourceFrameIndex, std::string *errorMessage) {
            hevcFrameCommitRequests_++;
            if(sourceFrameIndex < 0) {
                if(errorMessage) {
                    *errorMessage = "invalid ego source frame index";
                }
                return false;
            }
            if(committedHevcSourceFrames_.find(sourceFrameIndex) != committedHevcSourceFrames_.end()) {
                if(errorMessage) {
                    *errorMessage = "ego source frame already committed: " + std::to_string(sourceFrameIndex);
                }
                return false;
            }
            auto it = pendingHevcByFrame_.find(sourceFrameIndex);
            if(it == pendingHevcByFrame_.end() || it->second.empty()) {
                hevcFramesMissing_++;
                if(errorMessage) {
                    *errorMessage = "missing HEVC payload for ego source frame " + std::to_string(sourceFrameIndex);
                }
                logRaw("{\"event\":\"hevc_frame_missing\",\"server_unix_us\":" + std::to_string(unixUsNow())
                       + ",\"source_frame_index\":" + std::to_string(sourceFrameIndex) + "}");
                return false;
            }

            if(!wroteCodecConfig_) {
                if(codecConfigSamples_.empty()) {
                    logRaw("{\"event\":\"hevc_codec_config_missing_at_first_commit\",\"server_unix_us\":"
                           + std::to_string(unixUsNow()) + "}");
                }
                for(const auto &sample: codecConfigSamples_) {
                    if(!writeBufferedHevcSample(sample, "hevc_codec_config_committed", errorMessage)) {
                        return false;
                    }
                }
                codecConfigSamples_.clear();
                wroteCodecConfig_ = true;
            }

            for(const auto &sample: it->second) {
                if(!writeBufferedHevcSample(sample, "hevc_sample_committed", errorMessage)) {
                    return false;
                }
            }
            pendingHevcByFrame_.erase(it);
            committedHevcSourceFrames_.insert(sourceFrameIndex);
            hevcFramesCommitted_++;
            logRaw("{\"event\":\"hevc_frame_committed\",\"server_unix_us\":" + std::to_string(unixUsNow())
                   + ",\"source_frame_index\":" + std::to_string(sourceFrameIndex)
                   + ",\"committed_frame_index\":" + std::to_string(hevcFramesCommitted_ - 1) + "}");
            return true;
        }

        void discardHevcFramesBefore(int sourceFrameIndex) {
            if(sourceFrameIndex <= 0 || pendingHevcByFrame_.empty()) {
                return;
            }
            uint64_t droppedSamples = 0;
            uint64_t droppedBytes = 0;
            for(auto it = pendingHevcByFrame_.begin(); it != pendingHevcByFrame_.end();) {
                if(it->first >= sourceFrameIndex) {
                    break;
                }
                for(const auto &sample: it->second) {
                    droppedSamples++;
                    droppedBytes += sample.payload.size();
                }
                it = pendingHevcByFrame_.erase(it);
            }
            if(droppedSamples > 0) {
                hevcSamplesDropped_ += droppedSamples;
                hevcBytesDropped_ += droppedBytes;
                logRaw("{\"event\":\"hevc_samples_discarded\",\"server_unix_us\":" + std::to_string(unixUsNow())
                       + ",\"before_source_frame_index\":" + std::to_string(sourceFrameIndex)
                       + ",\"samples\":" + std::to_string(droppedSamples)
                       + ",\"bytes\":" + std::to_string(droppedBytes) + "}");
            }
        }

        void markError(const std::string &message) {
            lastError_ = message;
            logRaw("{\"event\":\"client_error\",\"server_unix_us\":" + std::to_string(unixUsNow())
                   + ",\"message\":" + jsonString(message) + "}");
        }

        void markSessionEnd(const std::string &headerJson) {
            clientSummaryJson_ = headerJson.empty() ? "{}" : headerJson;
            logRaw("{\"event\":\"session_end\",\"server_unix_us\":" + std::to_string(unixUsNow())
                   + ",\"client_summary\":" + clientSummaryJson_ + "}");
        }

        void close(const std::string &reason) {
            if(closed_) {
                return;
            }
            closed_ = true;
            endedUnixUs_ = unixUsNow();
            discardAllPendingHevcSamples("session_close");
            logRaw("{\"event\":\"session_close\",\"server_unix_us\":" + std::to_string(endedUnixUs_)
                   + ",\"reason\":" + jsonString(reason) + "}");

            for(auto *ofs: { &video_, &metadata_, &timestamps_, &networkLog_ }) {
                if(ofs && ofs->is_open()) {
                    ofs->flush();
                    ofs->close();
                }
            }
            try {
                if(std::filesystem::exists(metadataTmpPath_)) {
                    std::filesystem::rename(metadataTmpPath_, metadataPath_);
                }
            }
            catch(...) {
            }
            try {
                if(std::filesystem::exists(timestampsTmpPath_)) {
                    std::filesystem::rename(timestampsTmpPath_, timestampsPath_);
                }
            }
            catch(...) {
            }
            writeSessionJson(reason);
        }

    private:
        void logEventWithBytes(const std::string &event, size_t bytes) {
            std::ostringstream oss;
            oss << "{\"event\":" << jsonString(event)
                << ",\"server_unix_us\":" << unixUsNow()
                << ",\"bytes\":" << bytes << "}";
            logRaw(oss.str());
        }

        void logRaw(const std::string &line) {
            if(networkLog_.is_open()) {
                networkLog_ << line << "\n";
            }
        }

        void appendHeaderJson(std::ostringstream &oss, const std::string &headerJson) {
            if(cJSON *root = cJSON_Parse(headerJson.c_str())) {
                char *printed = cJSON_PrintUnformatted(root);
                if(printed) {
                    oss << printed;
                    cJSON_free(printed);
                }
                else {
                    oss << jsonString(headerJson);
                }
                cJSON_Delete(root);
            }
            else {
                oss << jsonString(headerJson);
            }
        }

        void logHevcSampleEvent(const std::string &event,
                                const BufferedHevcSample &sample,
                                std::optional<uint64_t> videoOffset) {
            std::ostringstream oss;
            oss << "{\"event\":" << jsonString(event)
                << ",\"server_unix_us\":" << unixUsNow()
                << ",\"source_frame_index\":" << sample.frameIndex
                << ",\"is_codec_config\":" << (sample.codecConfig ? "true" : "false")
                << ",\"payload_size\":" << sample.payload.size();
            if(videoOffset.has_value()) {
                oss << ",\"video_offset\":" << *videoOffset;
            }
            oss << ",\"header\":";
            appendHeaderJson(oss, sample.headerJson);
            oss << "}";
            logRaw(oss.str());
        }

        bool writeBufferedHevcSample(const BufferedHevcSample &sample,
                                     const std::string &event,
                                     std::string *errorMessage) {
            if(!video_.is_open()) {
                if(errorMessage) {
                    *errorMessage = "ego video file is not open";
                }
                return false;
            }
            const auto pos = video_.tellp();
            if(pos == std::ofstream::pos_type(-1)) {
                if(errorMessage) {
                    *errorMessage = "failed to query ego video offset";
                }
                return false;
            }
            const uint64_t offset = static_cast<uint64_t>(pos);
            if(!sample.payload.empty()) {
                video_.write(reinterpret_cast<const char *>(sample.payload.data()), static_cast<std::streamsize>(sample.payload.size()));
                if(!video_) {
                    if(errorMessage) {
                        *errorMessage = "failed to write ego HEVC payload";
                    }
                    return false;
                }
            }
            videoBytes_ += sample.payload.size();
            hevcSamplesCommitted_++;
            logHevcSampleEvent(event, sample, offset);
            return true;
        }

        void discardAllPendingHevcSamples(const std::string &reason) {
            uint64_t droppedSamples = 0;
            uint64_t droppedBytes = 0;
            for(const auto &kv: pendingHevcByFrame_) {
                for(const auto &sample: kv.second) {
                    droppedSamples++;
                    droppedBytes += sample.payload.size();
                }
            }
            pendingHevcByFrame_.clear();
            codecConfigSamples_.clear();
            if(droppedSamples > 0) {
                hevcSamplesDropped_ += droppedSamples;
                hevcBytesDropped_ += droppedBytes;
                logRaw("{\"event\":\"hevc_pending_samples_discarded\",\"server_unix_us\":" + std::to_string(unixUsNow())
                       + ",\"reason\":" + jsonString(reason)
                       + ",\"samples\":" + std::to_string(droppedSamples)
                       + ",\"bytes\":" + std::to_string(droppedBytes) + "}");
            }
        }

        void writeSessionJson(const std::string &reason) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << "{\n"
                << "  \"session_name\": " << jsonString(sessionName_) << ",\n"
                << "  \"session_dir\": " << jsonString(egoDir_.string()) << ",\n"
                << "  \"started_unix_us\": " << startedUnixUs_ << ",\n"
                << "  \"ended_unix_us\": " << endedUnixUs_ << ",\n"
                << "  \"duration_seconds\": " << std::setprecision(6)
                << (endedUnixUs_ > startedUnixUs_ ? static_cast<double>(endedUnixUs_ - startedUnixUs_) / 1000000.0 : 0.0) << ",\n"
                << "  \"video_h265\": " << jsonString(videoPath_.string()) << ",\n"
                << "  \"metadata_csv\": " << jsonString(metadataPath_.string()) << ",\n"
                << "  \"timestamps_csv\": " << jsonString(timestampsPath_.string()) << ",\n"
                << "  \"camera_json\": " << jsonString(cameraPath_.string()) << ",\n"
                << "  \"network_log_jsonl\": " << jsonString(networkLogPath_.string()) << ",\n"
                << "  \"video_bytes\": " << videoBytes_ << ",\n"
                << "  \"hevc_samples\": " << hevcSamplesCommitted_ << ",\n"
                << "  \"hevc_samples_received\": " << hevcSamplesReceived_ << ",\n"
                << "  \"hevc_samples_committed\": " << hevcSamplesCommitted_ << ",\n"
                << "  \"hevc_samples_dropped\": " << hevcSamplesDropped_ << ",\n"
                << "  \"hevc_bytes_received\": " << hevcBytesReceived_ << ",\n"
                << "  \"hevc_bytes_committed\": " << videoBytes_ << ",\n"
                << "  \"hevc_bytes_dropped\": " << hevcBytesDropped_ << ",\n"
                << "  \"hevc_frame_commit_requests\": " << hevcFrameCommitRequests_ << ",\n"
                << "  \"hevc_frames_committed\": " << hevcFramesCommitted_ << ",\n"
                << "  \"hevc_frames_missing\": " << hevcFramesMissing_ << ",\n"
                << "  \"metadata_rows\": " << metadataRows_ << ",\n"
                << "  \"timestamp_rows\": " << timestampRows_ << ",\n"
                << "  \"camera_json_received\": " << (cameraJsonReceived_ ? "true" : "false") << ",\n"
                << "  \"last_error\": " << jsonString(lastError_) << ",\n"
                << "  \"close_reason\": " << jsonString(reason) << ",\n"
                << "  \"client_summary\": " << (clientSummaryJson_.empty() ? "{}" : clientSummaryJson_) << ",\n"
                << "  \"timestamp_standard\": \"unix_epoch_microseconds_utc\",\n"
                << "  \"transport\": \"adb_reverse_tcp\",\n"
                << "  \"camera_params_json\": \"camera_params.json\",\n"
                << "  \"camera_params_source_json\": " << jsonString(cameraParamsSourcePath_.generic_string()) << "\n"
                << "}\n";
            (void)writeTextFile(sessionJsonPath_, oss.str());
        }

        std::string sessionName_;
        std::filesystem::path egoDir_;
        std::filesystem::path rgbDir_;
        std::filesystem::path videoPath_;
        std::filesystem::path metadataPath_;
        std::filesystem::path timestampsPath_;
        std::filesystem::path cameraPath_;
        std::filesystem::path sessionJsonPath_;
        std::filesystem::path networkLogPath_;
        std::filesystem::path cameraParamsSourcePath_;
        std::filesystem::path metadataTmpPath_;
        std::filesystem::path timestampsTmpPath_;
        std::ofstream video_;
        std::ofstream metadata_;
        std::ofstream timestamps_;
        std::ofstream networkLog_;
        std::vector<std::string> timestampHeader_;
        std::unordered_map<std::string, size_t> timestampIndex_;
        uint64_t startedUnixUs_ = 0;
        uint64_t endedUnixUs_ = 0;
        uint64_t videoBytes_ = 0;
        uint64_t hevcSamplesReceived_ = 0;
        uint64_t hevcSamplesCommitted_ = 0;
        uint64_t hevcSamplesDropped_ = 0;
        uint64_t hevcBytesReceived_ = 0;
        uint64_t hevcBytesDropped_ = 0;
        uint64_t hevcFrameCommitRequests_ = 0;
        uint64_t hevcFramesCommitted_ = 0;
        uint64_t hevcFramesMissing_ = 0;
        uint64_t metadataRows_ = 0;
        uint64_t timestampRows_ = 0;
        bool cameraJsonReceived_ = false;
        bool closed_ = false;
        bool wroteCodecConfig_ = false;
        std::vector<BufferedHevcSample> codecConfigSamples_;
        std::map<int, std::vector<BufferedHevcSample>> pendingHevcByFrame_;
        std::set<int> committedHevcSourceFrames_;
        std::string lastError_;
        std::string clientSummaryJson_ = "{}";
    };

    bool sendPacket(uint8_t type,
                    const std::string &headerJson,
                    const std::vector<uint8_t> &payload,
                    std::string *errorMessage) {
        int fd = -1;
        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            fd = clientFd_;
        }
        if(fd < 0) {
            if(errorMessage) {
                *errorMessage = "No connected PICO ego client";
            }
            return false;
        }
        std::lock_guard<std::mutex> sendLock(sendMtx_);
        if(!sendPacketFd(fd, type, headerJson.empty() ? "{}" : headerJson, payload)) {
            if(errorMessage) {
                *errorMessage = "Failed to send ego packet";
            }
            return false;
        }
        return true;
    }

    void acceptLoop() {
        while(!stopRequested_.load()) {
            int listener = -1;
            {
                std::lock_guard<std::mutex> lock(stateMtx_);
                listener = listenerFd_;
            }
            if(listener < 0) {
                break;
            }
            sockaddr_in addr{};
            socklen_t addrLen = sizeof(addr);
            int fd = ::accept(listener, reinterpret_cast<sockaddr *>(&addr), &addrLen);
            if(fd < 0) {
                if(stopRequested_.load()) {
                    break;
                }
                if(errno == EINTR) {
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            if(stopRequested_.load()) {
                closeFd(fd);
                break;
            }

            {
                std::lock_guard<std::mutex> lock(stateMtx_);
                closeFd(clientFd_);
                clientFd_ = fd;
                connected_ = true;
                helloReceived_ = false;
                lastHelloJson_.clear();
            }
            readyCv_.notify_all();
            std::thread oldReader;
            {
                std::lock_guard<std::mutex> lock(threadMtx_);
                if(readerThread_.joinable()) {
                    oldReader = std::move(readerThread_);
                }
            }
            if(oldReader.joinable()) {
                oldReader.join();
            }
            if(stopRequested_.load()) {
                std::lock_guard<std::mutex> lock(stateMtx_);
                if(clientFd_ == fd) {
                    closeFd(clientFd_);
                    connected_ = false;
                    helloReceived_ = false;
                }
                break;
            }
            {
                std::lock_guard<std::mutex> lock(threadMtx_);
                readerThread_ = std::thread([this, fd]() { readerLoop(fd); });
            }
            std::cerr << "[ego] client connected" << std::endl;
        }
    }

    void readerLoop(int fd) {
        while(!stopRequested_.load()) {
            Packet packet;
            std::string error;
            if(!recvPacketFd(fd, packet, &error)) {
                if(!stopRequested_.load()) {
                    std::cerr << "[ego] client disconnected: " << error << std::endl;
                }
                break;
            }
            handlePacket(packet);
        }

        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            if(clientFd_ == fd) {
                closeFd(clientFd_);
                connected_ = false;
                helloReceived_ = false;
            }
        }
        readyCv_.notify_all();
    }

    void handlePacket(const Packet &packet) {
        if(packet.type == PKT_HELLO) {
            {
                std::lock_guard<std::mutex> lock(stateMtx_);
                helloReceived_ = true;
                lastHelloJson_ = packet.headerJson;
            }
            readyCv_.notify_all();
            std::cerr << "[ego] HELLO " << packet.headerJson << std::endl;
            return;
        }

        if(packet.type == PKT_ERROR) {
            const std::string message = packet.headerJson;
            std::lock_guard<std::mutex> lock(sessionMtx_);
            if(session_) {
                session_->markError(message);
            }
            std::cerr << "[ego] client ERROR " << message << std::endl;
            return;
        }

        std::unique_lock<std::mutex> lock(sessionMtx_);
        if(!session_) {
            if(packet.type != PKT_HELLO) {
                std::cerr << "[ego] ignoring " << packetTypeName(packet.type) << "; no active session" << std::endl;
            }
            return;
        }

        switch(packet.type) {
        case PKT_CAMERA_JSON:
            session_->writeCameraJson(packet.payload);
            break;
        case PKT_METADATA_HEADER:
            session_->writeMetadataHeader(packet.payload);
            break;
        case PKT_METADATA_ROW:
            session_->writeMetadataRow(packet.payload);
            break;
        case PKT_TIMESTAMP_HEADER:
            session_->writeTimestampHeader(packet.payload);
            break;
        case PKT_TIMESTAMP_ROW: {
            const uint64_t seq = nextFrameSequence_++;
            auto frame = session_->writeTimestampRow(packet.payload, seq);
            if(frame.has_value()) {
                pushFrame(*frame);
            }
            break;
        }
        case PKT_HEVC_SAMPLE:
            session_->writeHevcSample(packet.headerJson, packet.payload);
            break;
        case PKT_SESSION_END:
            session_->markSessionEnd(packet.headerJson);
            closeSessionLocked("client_session_end");
            lock.unlock();
            sessionCv_.notify_all();
            break;
        default:
            std::cerr << "[ego] unknown packet type=" << static_cast<int>(packet.type) << std::endl;
            break;
        }
    }

    void pushFrame(EgoFrame frame) {
        {
            std::lock_guard<std::mutex> lock(frameMtx_);
            frameQueue_.push_back(frame);
            while(frameQueue_.size() > std::max<size_t>(1, config_.maxBufferedFrames)) {
                frameQueue_.pop_front();
            }
        }
        frameCv_.notify_all();
    }

    void closeSessionLocked(const std::string &reason) {
        if(!session_) {
            return;
        }
        session_->close(reason);
        session_.reset();
        sessionCv_.notify_all();
    }

    EgoModuleConfig config_{};
    std::atomic_bool running_{ false };
    std::atomic_bool stopRequested_{ false };

    mutable std::mutex stateMtx_;
    std::condition_variable readyCv_;
    int listenerFd_ = -1;
    int clientFd_ = -1;
    bool connected_ = false;
    bool helloReceived_ = false;
    std::string lastHelloJson_;

    std::mutex sendMtx_;
    std::mutex threadMtx_;
    std::thread acceptThread_;
    std::thread readerThread_;

    mutable std::mutex sessionMtx_;
    std::condition_variable sessionCv_;
    std::unique_ptr<SessionWriter> session_;

    mutable std::mutex frameMtx_;
    std::condition_variable frameCv_;
    std::deque<EgoFrame> frameQueue_;
    uint64_t nextFrameSequence_ = 0;
};

bool EgoRecorder::Impl::hasFramePayload(int sourceFrameIndex) const {
    std::lock_guard<std::mutex> lock(sessionMtx_);
    return session_ && session_->hasHevcFrame(sourceFrameIndex);
}

bool EgoRecorder::Impl::commitFrame(int sourceFrameIndex, std::string *errorMessage) {
    std::lock_guard<std::mutex> lock(sessionMtx_);
    if(!session_) {
        if(errorMessage) {
            *errorMessage = "No active ego session";
        }
        return false;
    }
    return session_->commitHevcFrame(sourceFrameIndex, errorMessage);
}

void EgoRecorder::Impl::discardFramesBefore(int sourceFrameIndex) {
    std::lock_guard<std::mutex> lock(sessionMtx_);
    if(session_) {
        session_->discardHevcFramesBefore(sourceFrameIndex);
    }
}

EgoRecorder::EgoRecorder()
    : impl_(std::make_unique<Impl>()) {
}

EgoRecorder::~EgoRecorder() = default;

bool EgoRecorder::start(const EgoModuleConfig &config, std::string *errorMessage) {
    return impl_->start(config, errorMessage);
}

void EgoRecorder::stop() {
    impl_->stop();
}

bool EgoRecorder::isRunning() const {
    return impl_->isRunning();
}

bool EgoRecorder::isConnected() const {
    return impl_->isConnected();
}

bool EgoRecorder::waitUntilReady(std::chrono::milliseconds timeout) {
    return impl_->waitUntilReady(timeout);
}

EgoModuleConfig EgoRecorder::config() const {
    return impl_->config();
}

std::string EgoRecorder::lastHelloSummary() const {
    return impl_->lastHelloSummary();
}

bool EgoRecorder::beginSession(const std::filesystem::path &episodeDir,
                               const std::string &sessionName,
                               std::string *errorMessage) {
    return impl_->beginSession(episodeDir, sessionName, errorMessage);
}

bool EgoRecorder::requestStopSession(std::string *errorMessage) {
    return impl_->requestStopSession(errorMessage);
}

bool EgoRecorder::stopSessionAndWait(std::chrono::milliseconds timeout, std::string *errorMessage) {
    return impl_->stopSessionAndWait(timeout, errorMessage);
}

bool EgoRecorder::isSessionActive() const {
    return impl_->isSessionActive();
}

bool EgoRecorder::popFrame(EgoFrame &out, std::chrono::milliseconds timeout) {
    return impl_->popFrame(out, timeout);
}

bool EgoRecorder::hasPendingFrames() const {
    return impl_->hasPendingFrames();
}

bool EgoRecorder::hasFramePayload(int sourceFrameIndex) const {
    return impl_->hasFramePayload(sourceFrameIndex);
}

bool EgoRecorder::commitFrame(int sourceFrameIndex, std::string *errorMessage) {
    return impl_->commitFrame(sourceFrameIndex, errorMessage);
}

void EgoRecorder::discardFramesBefore(int sourceFrameIndex) {
    impl_->discardFramesBefore(sourceFrameIndex);
}

}  // namespace sync_app
