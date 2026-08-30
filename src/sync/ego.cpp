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
#include <limits>
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
    PKT_TIME_SYNC_REQUEST = 12,
    PKT_TIME_SYNC_RESPONSE = 13,
};

struct Packet {
    uint8_t type = 0;
    std::string headerJson;
    std::vector<uint8_t> payload;
};

struct TimeSyncResponse {
    std::string calibrationId;
    int seq = -1;
    int64_t serverSendUnixUs = 0;
    int64_t clientRecvUnixUs = 0;
    int64_t clientSendUnixUs = 0;
    int64_t serverRecvUnixUs = 0;
};

struct TimeCalibrationSample {
    int seq = -1;
    int64_t serverSendUnixUs = 0;
    int64_t clientRecvUnixUs = 0;
    int64_t clientSendUnixUs = 0;
    int64_t serverRecvUnixUs = 0;
    int64_t rttUs = 0;
    int64_t clientProcessingUs = 0;
    int64_t picoToHostOffsetUs = 0;
};

struct TimeCalibrationResult {
    std::string calibrationId;
    uint64_t createdUnixUs = 0;
    std::string reason;
    std::string method = "ntp_best_half_median_offset_v1";
    int sampleCount = 0;
    int responseCount = 0;
    int acceptedCount = 0;
    int bestSampleCount = 0;
    int64_t picoToHostOffsetUs = 0;
    int64_t hostToPicoOffsetUs = 0;
    int64_t minRttUs = 0;
    int64_t medianRttUs = 0;
    int64_t maxRttUs = 0;
    std::vector<TimeCalibrationSample> samples;
};

uint64_t unixUsNow() {
    const auto now = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

uint64_t addSignedUs(uint64_t value, int64_t delta) {
    if(delta >= 0) {
        const auto udelta = static_cast<uint64_t>(delta);
        if(value > std::numeric_limits<uint64_t>::max() - udelta) {
            return std::numeric_limits<uint64_t>::max();
        }
        return value + udelta;
    }
    const auto magnitude = static_cast<uint64_t>(-(delta + 1)) + 1;
    return value > magnitude ? value - magnitude : 0;
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

std::string pathConfigKey(const std::filesystem::path &path) {
    return path.lexically_normal().generic_string();
}

bool sameEgoServerConfig(const EgoModuleConfig &a, const EgoModuleConfig &b) {
    return a.host == b.host
           && a.port == b.port
           && pathConfigKey(a.cameraParamsPath) == pathConfigKey(b.cameraParamsPath);
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

bool getJsonInt64(cJSON *root, const char *key, int64_t &out) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if(!item) {
        return false;
    }
    if(cJSON_IsNumber(item)) {
        out = static_cast<int64_t>(item->valuedouble);
        return true;
    }
    if(cJSON_IsString(item) && item->valuestring) {
        try {
            size_t idx = 0;
            const auto value = std::stoll(item->valuestring, &idx);
            if(idx > 0) {
                out = value;
                return true;
            }
        }
        catch(...) {
        }
    }
    return false;
}

std::optional<std::string> getJsonString(cJSON *root, const char *key) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if(item && cJSON_IsString(item) && item->valuestring) {
        return std::string(item->valuestring);
    }
    return std::nullopt;
}

bool parseTimeSyncResponseJson(const std::string &json, TimeSyncResponse &out) {
    cJSON *root = cJSON_Parse(json.c_str());
    if(!root) {
        return false;
    }
    const auto calibrationId = getJsonString(root, "calibration_id");
    int64_t seq = -1;
    bool ok = calibrationId.has_value()
              && getJsonInt64(root, "seq", seq)
              && getJsonInt64(root, "server_send_unix_us", out.serverSendUnixUs)
              && getJsonInt64(root, "client_recv_unix_us", out.clientRecvUnixUs)
              && getJsonInt64(root, "client_send_unix_us", out.clientSendUnixUs);
    if(ok) {
        out.calibrationId = *calibrationId;
        out.seq = static_cast<int>(seq);
    }
    cJSON_Delete(root);
    return ok && !out.calibrationId.empty() && out.seq >= 0;
}

int64_t medianInt64(std::vector<int64_t> values) {
    if(values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if(values.size() % 2 == 1) {
        return values[mid];
    }
    return (values[mid - 1] + values[mid]) / 2;
}

bool buildTimeCalibrationResult(const std::string &calibrationId,
                                int sampleCount,
                                const std::vector<TimeSyncResponse> &responses,
                                const std::string &reason,
                                TimeCalibrationResult &out,
                                std::string *errorMessage) {
    std::vector<TimeCalibrationSample> validSamples;
    validSamples.reserve(responses.size());
    for(const auto &response: responses) {
        const int64_t rttUs = (response.serverRecvUnixUs - response.serverSendUnixUs)
                              - (response.clientSendUnixUs - response.clientRecvUnixUs);
        const int64_t clientProcessingUs = response.clientSendUnixUs - response.clientRecvUnixUs;
        if(rttUs < 0 || clientProcessingUs < 0) {
            continue;
        }
        TimeCalibrationSample sample;
        sample.seq = response.seq;
        sample.serverSendUnixUs = response.serverSendUnixUs;
        sample.clientRecvUnixUs = response.clientRecvUnixUs;
        sample.clientSendUnixUs = response.clientSendUnixUs;
        sample.serverRecvUnixUs = response.serverRecvUnixUs;
        sample.rttUs = rttUs;
        sample.clientProcessingUs = clientProcessingUs;
        sample.picoToHostOffsetUs = ((response.serverSendUnixUs + response.serverRecvUnixUs)
                                     - (response.clientRecvUnixUs + response.clientSendUnixUs))
                                    / 2;
        validSamples.push_back(sample);
    }

    if(validSamples.empty()) {
        if(errorMessage) {
            *errorMessage = "timecalibrate failed: no valid time sync responses";
        }
        return false;
    }

    std::vector<TimeCalibrationSample> samplesByRtt = validSamples;
    std::sort(samplesByRtt.begin(), samplesByRtt.end(), [](const auto &a, const auto &b) {
        if(a.rttUs != b.rttUs) {
            return a.rttUs < b.rttUs;
        }
        return a.seq < b.seq;
    });
    const int bestSampleCount = std::max<int>(1, static_cast<int>((samplesByRtt.size() + 1) / 2));
    std::vector<int64_t> rtts;
    std::vector<int64_t> bestOffsets;
    rtts.reserve(validSamples.size());
    bestOffsets.reserve(static_cast<size_t>(bestSampleCount));
    for(const auto &sample: validSamples) {
        rtts.push_back(sample.rttUs);
    }
    for(int i = 0; i < bestSampleCount; ++i) {
        bestOffsets.push_back(samplesByRtt[static_cast<size_t>(i)].picoToHostOffsetUs);
    }

    out = TimeCalibrationResult{};
    out.calibrationId = calibrationId;
    out.createdUnixUs = unixUsNow();
    out.reason = reason;
    out.sampleCount = sampleCount;
    out.responseCount = static_cast<int>(responses.size());
    out.acceptedCount = static_cast<int>(validSamples.size());
    out.bestSampleCount = bestSampleCount;
    out.picoToHostOffsetUs = medianInt64(bestOffsets);
    out.hostToPicoOffsetUs = -out.picoToHostOffsetUs;
    out.minRttUs = *std::min_element(rtts.begin(), rtts.end());
    out.medianRttUs = medianInt64(rtts);
    out.maxRttUs = *std::max_element(rtts.begin(), rtts.end());
    out.samples = std::move(validSamples);
    return true;
}

std::string timeCalibrationToJson(const TimeCalibrationResult &calibration, int indent = 0) {
    const std::string pad(static_cast<size_t>(std::max(0, indent)), ' ');
    const std::string pad2 = pad + "  ";
    const std::string pad4 = pad + "    ";
    std::ostringstream oss;
    oss << pad << "{\n"
        << pad2 << "\"calibration_id\": " << jsonString(calibration.calibrationId) << ",\n"
        << pad2 << "\"created_unix_us\": " << calibration.createdUnixUs << ",\n"
        << pad2 << "\"reason\": " << jsonString(calibration.reason) << ",\n"
        << pad2 << "\"method\": " << jsonString(calibration.method) << ",\n"
        << pad2 << "\"sample_count\": " << calibration.sampleCount << ",\n"
        << pad2 << "\"response_count\": " << calibration.responseCount << ",\n"
        << pad2 << "\"accepted_count\": " << calibration.acceptedCount << ",\n"
        << pad2 << "\"best_sample_count\": " << calibration.bestSampleCount << ",\n"
        << pad2 << "\"pico_to_host_offset_us\": " << calibration.picoToHostOffsetUs << ",\n"
        << pad2 << "\"host_to_pico_offset_us\": " << calibration.hostToPicoOffsetUs << ",\n"
        << pad2 << "\"min_rtt_us\": " << calibration.minRttUs << ",\n"
        << pad2 << "\"median_rtt_us\": " << calibration.medianRttUs << ",\n"
        << pad2 << "\"max_rtt_us\": " << calibration.maxRttUs << ",\n"
        << pad2 << "\"formula\": \"host_unix_us = pico_unix_us + pico_to_host_offset_us\",\n"
        << pad2 << "\"samples\": [";
    for(size_t i = 0; i < calibration.samples.size(); ++i) {
        const auto &sample = calibration.samples[i];
        oss << (i == 0 ? "\n" : ",\n")
            << pad4 << "{"
            << "\"seq\": " << sample.seq
            << ", \"server_send_unix_us\": " << sample.serverSendUnixUs
            << ", \"client_recv_unix_us\": " << sample.clientRecvUnixUs
            << ", \"client_send_unix_us\": " << sample.clientSendUnixUs
            << ", \"server_recv_unix_us\": " << sample.serverRecvUnixUs
            << ", \"rtt_us\": " << sample.rttUs
            << ", \"client_processing_us\": " << sample.clientProcessingUs
            << ", \"pico_to_host_offset_us\": " << sample.picoToHostOffsetUs
            << "}";
    }
    if(!calibration.samples.empty()) {
        oss << "\n" << pad2;
    }
    oss << "]\n" << pad << "}";
    return oss.str();
}

std::string timeCalibrationSummary(const TimeCalibrationResult &calibration) {
    std::ostringstream oss;
    oss << "timecalibrate ok samples=" << calibration.sampleCount
        << " accepted=" << calibration.acceptedCount
        << " pico_to_host_offset_us=" << calibration.picoToHostOffsetUs
        << " min_rtt_us=" << calibration.minRttUs
        << " median_rtt_us=" << calibration.medianRttUs
        << " max_rtt_us=" << calibration.maxRttUs;
    return oss.str();
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
    case PKT_TIME_SYNC_REQUEST:
        return "TIME_SYNC_REQUEST";
    case PKT_TIME_SYNC_RESPONSE:
        return "TIME_SYNC_RESPONSE";
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
        if(running_.load() && sameEgoServerConfig(config_, config)) {
            config_ = config;
            return true;
        }
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
        timeSyncCv_.notify_all();
        sessionCv_.notify_all();
        frameCv_.notify_all();
        hevcCv_.notify_all();
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
        clearHevcSamples(false);
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
        return beginSessionInternal(episodeDir, sessionName, true, errorMessage);
    }

    bool beginPreviewSession(const std::string &sessionName,
                             std::string *errorMessage) {
        return beginSessionInternal({}, sessionName, false, errorMessage);
    }

    bool beginSessionInternal(const std::filesystem::path &episodeDir,
                              const std::string &sessionName,
                              bool persistToDisk,
                              std::string *errorMessage) {
        if(!isConnected()) {
            if(errorMessage) {
                *errorMessage = "PICO ego client is not connected";
            }
            return false;
        }

        const std::string safeName = sanitizeSessionName(sessionName);
        {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            if(session_) {
                if(errorMessage) {
                    *errorMessage = "Ego session is already active";
                }
                return false;
            }
        }

        std::optional<TimeCalibrationResult> timeCalibration;
        if(config_.timeCalibrate) {
            TimeCalibrationResult calibration;
            if(!performTimeCalibration(calibration, errorMessage)) {
                return false;
            }
            timeCalibration = std::move(calibration);
        }

        auto writer = std::make_unique<SessionWriter>();
        if(!writer->open(episodeDir, safeName, config_, timeCalibration ? &(*timeCalibration) : nullptr, persistToDisk, errorMessage)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> frameLock(frameMtx_);
            frameQueue_.clear();
            nextFrameSequence_ = 0;
        }
        clearHevcSamples(false);
        {
            std::lock_guard<std::mutex> hevcLock(hevcMtx_);
            nextHevcSequence_ = 0;
        }

        {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            if(session_) {
                if(errorMessage) {
                    *errorMessage = "Ego session is already active";
                }
                return false;
            }
            pendingTimestampFramesBySourceFrame_.clear();
            session_ = std::move(writer);
        }

        const std::string header = headerJsonWithServerTime(safeName);
        if(!sendPacket(PKT_START, header, {}, errorMessage)) {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            closeSessionLocked("start_send_failed");
            return false;
        }
        std::cerr << "[ego] START sent session=" << safeName
                  << " storage=" << (persistToDisk ? "disk" : "memory") << std::endl;
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
                if(errorMessage) {
                    *errorMessage = "Timed out waiting for PICO ego SESSION_END after " + std::to_string(timeout.count()) + " ms";
                }
                std::cerr << "[ego] warning: " << (errorMessage ? *errorMessage : "stop timed out")
                          << "; finalizing ego session locally" << std::endl;
                closeSessionLocked("stop_timeout");
                return false;
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

    bool popHevcSample(EgoHevcSample &out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(hevcMtx_);
        if(!hevcCv_.wait_for(lock, timeout, [&]() {
               return !hevcQueue_.empty() || stopRequested_.load();
           })) {
            return false;
        }
        if(hevcQueue_.empty()) {
            return false;
        }
        out = std::move(hevcQueue_.front());
        hevcQueue_.pop_front();
        return true;
    }

    void clearHevcSamples(bool keepCodecConfig) {
        {
            std::lock_guard<std::mutex> lock(hevcMtx_);
            hevcQueue_.clear();
            hevcAwaitingKeyFrame_ = false;
            if(!keepCodecConfig) {
                latestCodecConfigSamples_.clear();
            }
            else {
                for(const auto &sample: latestCodecConfigSamples_) {
                    hevcQueue_.push_back(sample);
                }
            }
        }
        hevcCv_.notify_all();
    }

    void requestHevcKeyFrameResync() {
        {
            std::lock_guard<std::mutex> lock(hevcMtx_);
            hevcQueue_.clear();
            hevcAwaitingKeyFrame_ = true;
        }
        hevcCv_.notify_all();
    }

    int videoFrameIndexForSourceFrame(int sourceFrameIndex) const;

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
                  const TimeCalibrationResult *timeCalibration,
                  bool persistToDisk,
                  std::string *errorMessage) {
            sessionName_ = sessionName;
            persistToDisk_ = persistToDisk;
            maxFrameMappings_ = std::max<size_t>(1, config.maxBufferedFrames);
            egoDir_ = episodeDir / "ego";
            rgbDir_ = egoDir_ / "RGB";
            videoPath_ = rgbDir_ / "rgb.h265";
            metadataPath_ = egoDir_ / "metadata.csv";
            timestampsPath_ = egoDir_ / "timestamps.csv";
            cameraPath_ = egoDir_ / "camera.json";
            sessionJsonPath_ = egoDir_ / "session.json";
            networkLogPath_ = egoDir_ / "network_log.jsonl";
            timeCalibrationPath_ = egoDir_ / "time_calibration.json";
            cameraParamsSourcePath_ = config.cameraParamsPath;
            timeCalibrateEnabled_ = config.timeCalibrate;
            softAlignToOrbbecFirstFrame_ = config.softAlignToOrbbecFirstFrame;
            if(timeCalibration) {
                timeCalibration_ = *timeCalibration;
                timeCalibrationOffsetUs_ = timeCalibration->picoToHostOffsetUs;
                timeCalibrationStatus_ = "ok";
            }
            else {
                timeCalibration_.reset();
                timeCalibrationOffsetUs_ = 0;
                timeCalibrationStatus_ = config.timeCalibrate ? "missing" : "disabled";
            }
            metadataTmpPath_ = metadataPath_;
            metadataTmpPath_ += ".tmp";
            timestampsTmpPath_ = timestampsPath_;
            timestampsTmpPath_ += ".tmp";

            startedUnixUs_ = unixUsNow();
            if(!persistToDisk_) {
                return true;
            }

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
            logRaw("{\"event\":\"session_open\",\"server_unix_us\":" + std::to_string(startedUnixUs_) + "}");
            writeTimeCalibrationSnapshot();
            return true;
        }

        void writeCameraJson(const std::vector<uint8_t> &payload) {
            if(!persistToDisk_) {
                return;
            }
            std::ofstream ofs(cameraPath_, std::ios::binary | std::ios::out | std::ios::trunc);
            if(ofs.is_open() && !payload.empty()) {
                ofs.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
                cameraJsonReceived_ = true;
            }
            logEventWithBytes("camera_json", payload.size());
        }

        void writeMetadataHeader(const std::vector<uint8_t> &payload) {
            if(persistToDisk_) {
                metadata_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
                metadata_.flush();
            }
            logEventWithBytes("metadata_header", payload.size());
        }

        void writeMetadataRow(const std::vector<uint8_t> &payload) {
            if(persistToDisk_) {
                metadata_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            }
            metadataRows_++;
        }

        void writeTimestampHeader(const std::vector<uint8_t> &payload) {
            if(persistToDisk_) {
                timestamps_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
                timestamps_.flush();
            }
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
            if(persistToDisk_) {
                timestamps_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            }
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
            frame.rawRefTimestampUs = parseUint64Or(col("ref_timestamp_us"));
            frame.rgbTimestampUs = parseUint64Or(col("pico_rgb_timestamp_us"));
            frame.acquireStartTimestampUs = parseUint64Or(col("pico_rgb_acquire_start_timestamp_us"));
            frame.acquireEndTimestampUs = parseUint64Or(col("pico_rgb_acquire_end_timestamp_us"));
            frame.picoFrameTimestampNs = parseUint64Or(col("pico_frame_timestamp_ns"));
            frame.xrHeadTimestampUs = parseUint64Or(col("pico_xr_head_timestamp_us"));
            frame.gazeTimestampUs = parseUint64Or(col("pico_gaze_timestamp_us"));
            if(frame.rawRefTimestampUs == 0) {
                frame.rawRefTimestampUs = frame.rgbTimestampUs;
            }
            if(frame.rawRefTimestampUs == 0) {
                return std::nullopt;
            }
            frame.timeCalibrationOffsetUs = timeCalibrationOffsetUs_;
            frame.timeCalibrationStatus = timeCalibrationStatus_;
            frame.refTimestampUs = timeCalibrationStatus_ == "ok"
                ? addSignedUs(frame.rawRefTimestampUs, timeCalibrationOffsetUs_)
                : frame.rawRefTimestampUs;
            return frame;
        }

        int writeHevcSample(const std::string &headerJson, const std::vector<uint8_t> &payload) {
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

            int videoFrameIndex = -1;
            if(!sample.codecConfig && sample.frameIndex >= 0) {
                auto it = videoFrameBySourceFrame_.find(sample.frameIndex);
                if(it == videoFrameBySourceFrame_.end()) {
                    videoFrameIndex = static_cast<int>(hevcVideoFrames_);
                    videoFrameBySourceFrame_[sample.frameIndex] = videoFrameIndex;
                    videoFrameSourceOrder_.push_back(sample.frameIndex);
                    while(!persistToDisk_ && videoFrameSourceOrder_.size() > maxFrameMappings_) {
                        videoFrameBySourceFrame_.erase(videoFrameSourceOrder_.front());
                        videoFrameSourceOrder_.pop_front();
                    }
                    hevcVideoFrames_++;
                }
                else {
                    videoFrameIndex = it->second;
                }
            }

            uint64_t offset = 0;
            if(persistToDisk_) {
                const auto pos = video_.tellp();
                offset = pos == std::ofstream::pos_type(-1) ? videoBytes_ : static_cast<uint64_t>(pos);
                if(!payload.empty()) {
                    video_.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
                }
                if(video_) {
                    videoBytes_ += payload.size();
                    hevcSamplesWritten_++;
                    video_.flush();
                }
                else {
                    lastError_ = "failed to write ego HEVC payload";
                }
            }
            logHevcSampleEvent("hevc_sample", sample, offset, videoFrameIndex);
            return videoFrameIndex >= 0 ? sample.frameIndex : -1;
        }

        int videoFrameIndexForSourceFrame(int sourceFrameIndex) const {
            if(sourceFrameIndex < 0) {
                return -1;
            }
            auto it = videoFrameBySourceFrame_.find(sourceFrameIndex);
            if(it == videoFrameBySourceFrame_.end()) {
                return -1;
            }
            return it->second;
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
            if(!persistToDisk_) {
                return;
            }
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

        void writeTimeCalibrationSnapshot() {
            if(!timeCalibration_) {
                return;
            }
            const std::string json = timeCalibrationToJson(*timeCalibration_);
            (void)writeTextFile(timeCalibrationPath_, json + "\n");
            logRaw("{\"event\":\"time_calibration_snapshot\",\"server_unix_us\":" + std::to_string(unixUsNow())
                   + ",\"pico_to_host_offset_us\":" + std::to_string(timeCalibration_->picoToHostOffsetUs)
                   + ",\"accepted_count\":" + std::to_string(timeCalibration_->acceptedCount)
                   + ",\"median_rtt_us\":" + std::to_string(timeCalibration_->medianRttUs) + "}");
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
                                uint64_t videoOffset,
                                int videoFrameIndex) {
            std::ostringstream oss;
            oss << "{\"event\":" << jsonString(event)
                << ",\"server_unix_us\":" << unixUsNow()
                << ",\"source_frame_index\":" << sample.frameIndex
                << ",\"is_codec_config\":" << (sample.codecConfig ? "true" : "false")
                << ",\"payload_size\":" << sample.payload.size()
                << ",\"video_offset\":" << videoOffset;
            if(videoFrameIndex >= 0) {
                oss << ",\"video_frame_index\":" << videoFrameIndex;
            }
            oss << ",\"header\":";
            appendHeaderJson(oss, sample.headerJson);
            oss << "}";
            logRaw(oss.str());
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
                << "  \"time_calibration_json\": " << (timeCalibration_ ? jsonString(timeCalibrationPath_.string()) : jsonString("")) << ",\n"
                << "  \"time_calibrate_enabled\": " << (timeCalibrateEnabled_ ? "true" : "false") << ",\n"
                << "  \"time_calibration_status\": " << jsonString(timeCalibrationStatus_) << ",\n"
                << "  \"pico_to_host_offset_us\": " << timeCalibrationOffsetUs_ << ",\n"
                << "  \"video_bytes\": " << videoBytes_ << ",\n"
                << "  \"hevc_samples\": " << hevcSamplesWritten_ << ",\n"
                << "  \"hevc_samples_received\": " << hevcSamplesReceived_ << ",\n"
                << "  \"hevc_samples_written\": " << hevcSamplesWritten_ << ",\n"
                << "  \"hevc_video_frames\": " << hevcVideoFrames_ << ",\n"
                << "  \"hevc_bytes_received\": " << hevcBytesReceived_ << ",\n"
                << "  \"hevc_bytes_written\": " << videoBytes_ << ",\n"
                << "  \"metadata_rows\": " << metadataRows_ << ",\n"
                << "  \"timestamp_rows\": " << timestampRows_ << ",\n"
                << "  \"camera_json_received\": " << (cameraJsonReceived_ ? "true" : "false") << ",\n"
                << "  \"last_error\": " << jsonString(lastError_) << ",\n"
                << "  \"close_reason\": " << jsonString(reason) << ",\n"
                << "  \"client_summary\": " << (clientSummaryJson_.empty() ? "{}" : clientSummaryJson_) << ",\n"
                << "  \"timestamp_standard\": \"unix_epoch_microseconds_utc\",\n"
                << "  \"transport\": \"adb_reverse_tcp\",\n"
                << "  \"camera_params_json\": \"camera_params.json\",\n"
                << "  \"camera_params_source_json\": " << jsonString(cameraParamsSourcePath_.generic_string()) << ",\n"
                << "  \"soft_align_to_orbbec_first_frame\": " << (softAlignToOrbbecFirstFrame_ ? "true" : "false") << "\n"
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
        std::filesystem::path timeCalibrationPath_;
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
        uint64_t hevcSamplesWritten_ = 0;
        uint64_t hevcVideoFrames_ = 0;
        uint64_t hevcBytesReceived_ = 0;
        uint64_t metadataRows_ = 0;
        uint64_t timestampRows_ = 0;
        bool cameraJsonReceived_ = false;
        bool closed_ = false;
        bool timeCalibrateEnabled_ = false;
        bool softAlignToOrbbecFirstFrame_ = false;
        int64_t timeCalibrationOffsetUs_ = 0;
        std::string timeCalibrationStatus_ = "disabled";
        std::optional<TimeCalibrationResult> timeCalibration_;
        std::unordered_map<int, int> videoFrameBySourceFrame_;
        std::deque<int> videoFrameSourceOrder_;
        size_t maxFrameMappings_ = 1;
        bool persistToDisk_ = true;
        std::string lastError_;
        std::string clientSummaryJson_ = "{}";
    };

    std::string makeTimeCalibrationId() {
        std::lock_guard<std::mutex> lock(timeSyncMtx_);
        std::ostringstream oss;
        oss << unixUsNow() << "_" << std::hex << reinterpret_cast<uintptr_t>(this)
            << "_" << std::dec << (++timeSyncSerial_);
        return oss.str();
    }

    bool performTimeCalibration(TimeCalibrationResult &out, std::string *errorMessage) {
        const int sampleCount = std::max(1, std::min(200, config_.timeCalibrateSampleCount));
        const auto timeout = std::chrono::milliseconds(std::max(100, config_.timeCalibrateTimeoutMs));
        if(!isConnected()) {
            if(errorMessage) {
                *errorMessage = "timecalibrate failed: no PICO client connected";
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(sessionMtx_);
            if(session_) {
                if(errorMessage) {
                    *errorMessage = "timecalibrate failed: stop the active ego session first";
                }
                return false;
            }
        }

        const std::string calibrationId = makeTimeCalibrationId();
        {
            std::lock_guard<std::mutex> lock(timeSyncMtx_);
            timeSyncCalibrationId_ = calibrationId;
            timeSyncResponses_.clear();
        }

        auto clearCalibration = [&]() {
            std::lock_guard<std::mutex> lock(timeSyncMtx_);
            if(timeSyncCalibrationId_ == calibrationId) {
                timeSyncCalibrationId_.clear();
                timeSyncResponses_.clear();
            }
        };

        std::vector<TimeSyncResponse> responses;
        responses.reserve(static_cast<size_t>(sampleCount));
        for(int seq = 0; seq < sampleCount; ++seq) {
            const uint64_t serverSendUnixUs = unixUsNow();
            std::ostringstream header;
            header << "{\"calibration_id\":" << jsonString(calibrationId)
                   << ",\"seq\":" << seq
                   << ",\"server_send_unix_us\":" << serverSendUnixUs << "}";
            if(!sendPacket(PKT_TIME_SYNC_REQUEST, header.str(), {}, errorMessage)) {
                clearCalibration();
                return false;
            }

            const auto deadline = std::chrono::steady_clock::now() + timeout;
            std::unique_lock<std::mutex> lock(timeSyncMtx_);
            while(timeSyncResponses_.find(seq) == timeSyncResponses_.end()) {
                if(stopRequested_.load() || !isConnected()) {
                    lock.unlock();
                    clearCalibration();
                    if(errorMessage) {
                        *errorMessage = "timecalibrate failed: PICO client disconnected";
                    }
                    return false;
                }
                if(timeSyncCv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                    if(timeSyncResponses_.find(seq) == timeSyncResponses_.end()) {
                        lock.unlock();
                        clearCalibration();
                        if(errorMessage) {
                            *errorMessage = "timecalibrate failed: timeout waiting for seq=" + std::to_string(seq);
                        }
                        return false;
                    }
                }
            }
            responses.push_back(timeSyncResponses_.at(seq));
        }
        clearCalibration();

        if(!buildTimeCalibrationResult(calibrationId, sampleCount, responses, "auto_start", out, errorMessage)) {
            return false;
        }
        std::cerr << "[ego] " << timeCalibrationSummary(out) << std::endl;
        return true;
    }

    void handleTimeSyncResponse(const std::string &headerJson) {
        TimeSyncResponse response;
        if(!parseTimeSyncResponseJson(headerJson, response)) {
            return;
        }
        response.serverRecvUnixUs = static_cast<int64_t>(unixUsNow());
        std::lock_guard<std::mutex> lock(timeSyncMtx_);
        if(response.calibrationId != timeSyncCalibrationId_) {
            return;
        }
        timeSyncResponses_[response.seq] = response;
        timeSyncCv_.notify_all();
    }

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
        timeSyncCv_.notify_all();
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

        if(packet.type == PKT_TIME_SYNC_RESPONSE) {
            handleTimeSyncResponse(packet.headerJson);
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
                queueTimestampFrameWhenVideoReadyLocked(*frame);
            }
            break;
        }
        case PKT_HEVC_SAMPLE: {
            const int mappedSourceFrameIndex = session_->writeHevcSample(packet.headerJson, packet.payload);
            pushHevcSample(packet);
            releasePendingTimestampFrameLocked(mappedSourceFrameIndex);
            break;
        }
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

    void queueTimestampFrameWhenVideoReadyLocked(EgoFrame frame) {
        if(!session_ || frame.sourceFrameIndex < 0) {
            return;
        }
        const int videoFrameIndex = session_->videoFrameIndexForSourceFrame(frame.sourceFrameIndex);
        if(videoFrameIndex >= 0) {
            frame.videoFrameIndex = videoFrameIndex;
            pushFrame(std::move(frame));
            return;
        }
        pendingTimestampFramesBySourceFrame_[frame.sourceFrameIndex] = std::move(frame);
        while(pendingTimestampFramesBySourceFrame_.size() > std::max<size_t>(1, config_.maxBufferedFrames)) {
            pendingTimestampFramesBySourceFrame_.erase(pendingTimestampFramesBySourceFrame_.begin());
        }
    }

    void releasePendingTimestampFrameLocked(int sourceFrameIndex) {
        if(sourceFrameIndex < 0 || !session_) {
            return;
        }
        auto it = pendingTimestampFramesBySourceFrame_.find(sourceFrameIndex);
        if(it == pendingTimestampFramesBySourceFrame_.end()) {
            return;
        }
        EgoFrame frame = it->second;
        pendingTimestampFramesBySourceFrame_.erase(it);
        frame.videoFrameIndex = session_->videoFrameIndexForSourceFrame(sourceFrameIndex);
        pushFrame(std::move(frame));
    }

    void releasePendingTimestampFramesOnCloseLocked(const std::string &reason) {
        if(!session_ || pendingTimestampFramesBySourceFrame_.empty()) {
            return;
        }
        std::vector<int> readySourceFrames;
        readySourceFrames.reserve(pendingTimestampFramesBySourceFrame_.size());
        for(const auto &kv: pendingTimestampFramesBySourceFrame_) {
            if(session_->videoFrameIndexForSourceFrame(kv.first) >= 0) {
                readySourceFrames.push_back(kv.first);
            }
        }
        std::sort(readySourceFrames.begin(), readySourceFrames.end(), [&](int a, int b) {
            const auto ita = pendingTimestampFramesBySourceFrame_.find(a);
            const auto itb = pendingTimestampFramesBySourceFrame_.find(b);
            if(ita != pendingTimestampFramesBySourceFrame_.end() && itb != pendingTimestampFramesBySourceFrame_.end()
               && ita->second.sequence != itb->second.sequence) {
                return ita->second.sequence < itb->second.sequence;
            }
            return a < b;
        });

        size_t released = 0;
        for(const int sourceFrameIndex: readySourceFrames) {
            auto it = pendingTimestampFramesBySourceFrame_.find(sourceFrameIndex);
            if(it == pendingTimestampFramesBySourceFrame_.end()) {
                continue;
            }
            EgoFrame frame = std::move(it->second);
            pendingTimestampFramesBySourceFrame_.erase(it);
            frame.videoFrameIndex = session_->videoFrameIndexForSourceFrame(sourceFrameIndex);
            pushFrame(std::move(frame));
            released++;
        }
        std::cerr << "[ego] released pending timestamp frames on close reason=" << reason
                  << " released=" << released
                  << " still_waiting_for_hevc=" << pendingTimestampFramesBySourceFrame_.size() << std::endl;
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

    void pushHevcSample(const Packet &packet) {
        EgoHevcSample sample;
        const auto frameIndex = jsonInt64Field(packet.headerJson, "frame_index");
        if(frameIndex.has_value()) {
            sample.sourceFrameIndex = static_cast<int>(*frameIndex);
        }
        sample.codecConfig = jsonBoolField(packet.headerJson, "is_codec_config").value_or(sample.sourceFrameIndex < 0);
        sample.keyFrame = jsonBoolField(packet.headerJson, "is_keyframe").value_or(false);
        sample.receivedUnixUs = unixUsNow();
        sample.headerJson = packet.headerJson;
        sample.payload = packet.payload;

        {
            std::lock_guard<std::mutex> lock(hevcMtx_);
            sample.sequence = nextHevcSequence_++;
            if(sample.codecConfig && !sample.payload.empty()) {
                latestCodecConfigSamples_.clear();
                latestCodecConfigSamples_.push_back(sample);
            }

            if(hevcAwaitingKeyFrame_) {
                if(sample.codecConfig || !sample.keyFrame || latestCodecConfigSamples_.empty()) {
                    return;
                }
                bool resetMarked = false;
                for(const auto &configSample : latestCodecConfigSamples_) {
                    EgoHevcSample queuedConfig = configSample;
                    if(!resetMarked) {
                        queuedConfig.decoderReset = true;
                        resetMarked = true;
                    }
                    hevcQueue_.push_back(std::move(queuedConfig));
                }
                if(!resetMarked) {
                    sample.decoderReset = true;
                }
                hevcQueue_.push_back(std::move(sample));
                hevcAwaitingKeyFrame_ = false;
                hevcCv_.notify_all();
                return;
            }

            hevcQueue_.push_back(std::move(sample));
            if(hevcQueue_.size() > kMaxLiveHevcSamples) {
                auto newestKeyFrame = hevcQueue_.end();
                for(auto it = hevcQueue_.begin(); it != hevcQueue_.end(); ++it) {
                    if(!it->codecConfig && it->keyFrame) {
                        newestKeyFrame = it;
                    }
                }

                if(newestKeyFrame == hevcQueue_.end() || latestCodecConfigSamples_.empty()) {
                    hevcQueue_.clear();
                    hevcAwaitingKeyFrame_ = true;
                }
                else {
                    std::deque<EgoHevcSample> decodableTail;
                    for(auto it = newestKeyFrame; it != hevcQueue_.end(); ++it) {
                        if(!it->codecConfig) {
                            decodableTail.push_back(std::move(*it));
                        }
                    }
                    hevcQueue_.clear();
                    bool resetMarked = false;
                    for(const auto &configSample : latestCodecConfigSamples_) {
                        EgoHevcSample queuedConfig = configSample;
                        if(!resetMarked) {
                            queuedConfig.decoderReset = true;
                            resetMarked = true;
                        }
                        hevcQueue_.push_back(std::move(queuedConfig));
                    }
                    if(!resetMarked && !decodableTail.empty()) {
                        decodableTail.front().decoderReset = true;
                    }
                    while(!decodableTail.empty()) {
                        hevcQueue_.push_back(std::move(decodableTail.front()));
                        decodableTail.pop_front();
                    }
                }
            }
        }
        hevcCv_.notify_all();
    }

    void closeSessionLocked(const std::string &reason) {
        if(!session_) {
            return;
        }
        releasePendingTimestampFramesOnCloseLocked(reason);
        session_->close(reason);
        session_.reset();
        pendingTimestampFramesBySourceFrame_.clear();
        clearHevcSamples(false);
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
    std::mutex timeSyncMtx_;
    std::condition_variable timeSyncCv_;
    std::string timeSyncCalibrationId_;
    std::unordered_map<int, TimeSyncResponse> timeSyncResponses_;
    uint64_t timeSyncSerial_ = 0;

    std::mutex threadMtx_;
    std::thread acceptThread_;
    std::thread readerThread_;

    mutable std::mutex sessionMtx_;
    std::condition_variable sessionCv_;
    std::unique_ptr<SessionWriter> session_;
    std::unordered_map<int, EgoFrame> pendingTimestampFramesBySourceFrame_;

    mutable std::mutex frameMtx_;
    std::condition_variable frameCv_;
    std::deque<EgoFrame> frameQueue_;
    uint64_t nextFrameSequence_ = 0;

    mutable std::mutex hevcMtx_;
    std::condition_variable hevcCv_;
    std::deque<EgoHevcSample> hevcQueue_;
    std::vector<EgoHevcSample> latestCodecConfigSamples_;
    uint64_t nextHevcSequence_ = 0;
    bool hevcAwaitingKeyFrame_ = false;
    static constexpr size_t kMaxLiveHevcSamples = 96;
};

int EgoRecorder::Impl::videoFrameIndexForSourceFrame(int sourceFrameIndex) const {
    std::lock_guard<std::mutex> lock(sessionMtx_);
    if(!session_) {
        return -1;
    }
    return session_->videoFrameIndexForSourceFrame(sourceFrameIndex);
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

bool EgoRecorder::beginPreviewSession(const std::string &sessionName,
                                      std::string *errorMessage) {
    return impl_->beginPreviewSession(sessionName, errorMessage);
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

int EgoRecorder::videoFrameIndexForSourceFrame(int sourceFrameIndex) const {
    return impl_->videoFrameIndexForSourceFrame(sourceFrameIndex);
}

bool EgoRecorder::popHevcSample(EgoHevcSample &out, std::chrono::milliseconds timeout) {
    return impl_->popHevcSample(out, timeout);
}

void EgoRecorder::clearHevcSamples(bool keepCodecConfig) {
    impl_->clearHevcSamples(keepCodecConfig);
}

void EgoRecorder::requestHevcKeyFrameResync() {
    impl_->requestHevcKeyFrameResync();
}

}  // namespace sync_app
