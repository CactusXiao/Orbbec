#include "fisheyes.hpp"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <set>
#include <unordered_map>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace sync_app {

namespace {

struct FisheyeDeviceCandidate {
    std::string devPath;
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

std::vector<int> fisheyeEncodeParams(const FisheyeSaveOptions &options) {
    if(options.format == FisheyeImageFormat::Png) {
        return { cv::IMWRITE_PNG_COMPRESSION, std::max(0, std::min(9, options.pngCompression)) };
    }
    return { cv::IMWRITE_JPEG_QUALITY, std::max(0, std::min(100, options.jpegQuality)) };
}

bool saveBgrImage(const cv::Mat &bgr, const std::filesystem::path &path, const FisheyeSaveOptions &options, std::string *errorMessage) {
    if(bgr.empty()) {
        if(errorMessage) {
            *errorMessage = "Fisheye frame is empty";
        }
        return false;
    }
    try {
        if(cv::imwrite(path.string(), bgr, fisheyeEncodeParams(options))) {
            return true;
        }
        if(errorMessage) {
            *errorMessage = "cv::imwrite failed for " + path.string();
        }
        return false;
    }
    catch(const cv::Exception &ex) {
        if(errorMessage) {
            *errorMessage = ex.what();
        }
        return false;
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

int parseVideoIndex(const std::string &path) {
    const std::string prefix = "/dev/video";
    if(path.rfind(prefix, 0) != 0) {
        return -1;
    }
    try {
        return std::stoi(path.substr(prefix.size()));
    }
    catch(...) {
        return -1;
    }
}

std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trimCopy(std::string s) {
    auto isSpace = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    while(!s.empty() && isSpace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while(!s.empty() && isSpace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

bool fisheyeDebugEnabled() {
    return std::getenv("SYNC_FISHEYE_DEBUG") != nullptr;
}

std::vector<std::string> splitHintTokens(const std::string &s) {
    std::vector<std::string> out;
    std::string current;
    for(char ch : s) {
        if(ch == ',' || ch == ';' || ch == '|' || std::isspace(static_cast<unsigned char>(ch))) {
            current = trimCopy(current);
            if(!current.empty()) {
                out.push_back(toLowerCopy(current));
            }
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    current = trimCopy(current);
    if(!current.empty()) {
        out.push_back(toLowerCopy(current));
    }
    return out;
}

const std::vector<std::string> &defaultFisheyeIdentityTokens() {
    static const std::vector<std::string> tokens = {
        "dcx-250107-xh",
        "decxin",
        "fisheye",
    };
    return tokens;
}

const std::vector<std::string> &knownNonFisheyeIdentityTokens() {
    static const std::vector<std::string> tokens = {
        "orbbec",
        "obsensor",
        "realsense",
        "intel",
        "depthai",
        "luxonis",
        "oak",
        "azure",
        "kinect",
    };
    return tokens;
}

const std::vector<std::string> &extraFisheyeIdentityTokens() {
    static const std::vector<std::string> tokens = []() {
        const char *env = std::getenv("SYNC_FISHEYE_HINTS");
        if(!env) {
            return std::vector<std::string>{};
        }
        return splitHintTokens(env);
    }();
    return tokens;
}

bool containsAnyIdentityToken(const std::string &haystack, const std::vector<std::string> &tokens) {
    for(const auto &token : tokens) {
        if(!token.empty() && haystack.find(token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string candidateIdentityString(const FisheyeDeviceCandidate &candidate) {
    std::string identity;
    identity.reserve(candidate.devPath.size() + candidate.stablePath.size() + candidate.cardName.size() + candidate.busInfo.size()
                     + candidate.uniqueId.size() + candidate.usbVendorId.size() + candidate.usbProductId.size()
                     + candidate.usbSerial.size() + candidate.usbManufacturer.size() + candidate.usbProduct.size()
                     + candidate.usbPortPath.size() + 16);
    identity += toLowerCopy(candidate.devPath);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.stablePath);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.cardName);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.busInfo);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.uniqueId);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.usbVendorId);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.usbProductId);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.usbSerial);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.usbManufacturer);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.usbProduct);
    identity.push_back(' ');
    identity += toLowerCopy(candidate.usbPortPath);
    return identity;
}

bool hasConfiguredUniqueId(const FisheyeCameraConfig &camera) {
    return !camera.uniqueId.empty();
}

bool isExplicitCameraSelection(const FisheyeCameraConfig &camera) {
    return !camera.devicePath.empty() || camera.deviceIndex >= 0 || !camera.preferredDeviceHint.empty();
}

bool candidateLooksLikeKnownNonFisheye(const FisheyeDeviceCandidate &candidate) {
    return containsAnyIdentityToken(candidateIdentityString(candidate), knownNonFisheyeIdentityTokens());
}

bool candidateHasKnownFisheyeUsbId(const FisheyeDeviceCandidate &candidate) {
    const std::string vendor = toLowerCopy(candidate.usbVendorId);
    const std::string product = toLowerCopy(candidate.usbProductId);
    return vendor == "1d6c" && product == "0103";
}

bool candidateLooksLikeFisheye(const FisheyeDeviceCandidate &candidate) {
    const std::string identity = candidateIdentityString(candidate);
    if(identity.empty()) {
        return false;
    }
    if(candidateLooksLikeKnownNonFisheye(candidate)) {
        return false;
    }
    if(candidateHasKnownFisheyeUsbId(candidate)) {
        return true;
    }
    if(containsAnyIdentityToken(identity, defaultFisheyeIdentityTokens())) {
        return true;
    }
    if(containsAnyIdentityToken(identity, extraFisheyeIdentityTokens())) {
        return true;
    }
    return false;
}

std::string makeUsbSignature(const FisheyeDeviceCandidate &candidate) {
    if(!candidate.usbVendorId.empty() || !candidate.usbProductId.empty()) {
        return toLowerCopy(candidate.usbVendorId) + ":" + toLowerCopy(candidate.usbProductId);
    }
    if(!candidate.usbManufacturer.empty() || !candidate.usbProduct.empty()) {
        return toLowerCopy(candidate.usbManufacturer) + "|" + toLowerCopy(candidate.usbProduct);
    }
    return "";
}

std::string makeCandidateUniqueId(const FisheyeDeviceCandidate &candidate) {
    if(!candidate.usbVendorId.empty() && !candidate.usbProductId.empty() && !candidate.usbSerial.empty()) {
        return "usb:" + toLowerCopy(candidate.usbVendorId) + ":" + toLowerCopy(candidate.usbProductId) + ":" + candidate.usbSerial;
    }
    if(!candidate.stablePath.empty()) {
        return "v4l:" + candidate.stablePath;
    }
    const std::string usbSignature = makeUsbSignature(candidate);
    if(!usbSignature.empty() && !candidate.usbPortPath.empty()) {
        return "usb:" + usbSignature + ":" + candidate.usbPortPath;
    }
    if(!usbSignature.empty()) {
        return "usb:" + usbSignature + ":" + candidate.cardName;
    }
    return "video:" + candidate.devPath;
}

std::string stripLeadingZeros(std::string s) {
    if(s.empty()) {
        return s;
    }
    size_t pos = 0;
    while(pos + 1 < s.size() && s[pos] == '0') {
        ++pos;
    }
    return s.substr(pos);
}

std::string normalizeCameraLabel(std::string label) {
    label = toLowerCopy(trimCopy(label));
    if(label.empty()) {
        return "";
    }
    bool allDigits = true;
    for(char ch : label) {
        if(ch < '0' || ch > '9') {
            allDigits = false;
            break;
        }
    }
    if(allDigits) {
        return "camera" + stripLeadingZeros(label);
    }
    constexpr const char *kPrefix = "camera";
    if(label.rfind(kPrefix, 0) == 0) {
        const std::string digits = label.substr(6);
        if(!digits.empty() && std::all_of(digits.begin(), digits.end(), [](char ch) {
               return ch >= '0' && ch <= '9';
           })) {
            return "camera" + stripLeadingZeros(digits);
        }
    }
    return label;
}

bool containsNormalizedCameraLabel(const std::string &text, const std::string &normalizedLabel) {
    if(text.empty() || normalizedLabel.empty()) {
        return false;
    }
    const std::string lower = toLowerCopy(text);
    for(size_t i = 0; i + 6 < lower.size(); ++i) {
        if(lower.compare(i, 6, "camera") != 0) {
            continue;
        }
        if(i > 0) {
            const unsigned char prev = static_cast<unsigned char>(lower[i - 1]);
            if(std::isalnum(prev) != 0) {
                continue;
            }
        }
        size_t j = i + 6;
        while(j < lower.size() && lower[j] >= '0' && lower[j] <= '9') {
            ++j;
        }
        if(j == i + 6) {
            continue;
        }
        if(j < lower.size()) {
            const unsigned char next = static_cast<unsigned char>(lower[j]);
            if(std::isalnum(next) != 0) {
                continue;
            }
        }
        const std::string token = normalizeCameraLabel(lower.substr(i, j - i));
        if(token == normalizedLabel) {
            return true;
        }
    }
    return false;
}

bool candidateMatchesCameraLabel(const FisheyeDeviceCandidate &candidate, const std::string &label) {
    const std::string normalizedLabel = normalizeCameraLabel(label);
    if(normalizedLabel.empty()) {
        return false;
    }
    return containsNormalizedCameraLabel(candidate.usbProduct, normalizedLabel)
           || containsNormalizedCameraLabel(candidate.stablePath, normalizedLabel)
           || containsNormalizedCameraLabel(candidate.devPath, normalizedLabel)
           || containsNormalizedCameraLabel(candidate.cardName, normalizedLabel)
           || containsNormalizedCameraLabel(candidate.busInfo, normalizedLabel)
           || containsNormalizedCameraLabel(candidate.uniqueId, normalizedLabel);
}

FisheyeDeviceInfo toDeviceInfo(const FisheyeDeviceCandidate &candidate) {
    FisheyeDeviceInfo info;
    info.devicePath      = candidate.devPath;
    info.stablePath      = candidate.stablePath;
    info.cardName        = candidate.cardName;
    info.busInfo         = candidate.busInfo;
    info.uniqueId        = candidate.uniqueId;
    info.usbVendorId     = candidate.usbVendorId;
    info.usbProductId    = candidate.usbProductId;
    info.usbSerial       = candidate.usbSerial;
    info.usbManufacturer = candidate.usbManufacturer;
    info.usbProduct      = candidate.usbProduct;
    info.usbPortPath     = candidate.usbPortPath;
    info.supportsMjpeg   = candidate.supportsMjpeg;
    info.supportsYuyv    = candidate.supportsYuyv;
    return info;
}

#if defined(__linux__)
bool readSysfsTextFile(const std::filesystem::path &path, std::string &out) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if(!ifs.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = trimCopy(ss.str());
    return !out.empty();
}

void enrichCandidateUsbInfo(const std::string &devPath, FisheyeDeviceCandidate &out) {
    const int videoIndex = parseVideoIndex(devPath);
    if(videoIndex < 0) {
        return;
    }

    std::error_code ec;
    auto cur = std::filesystem::weakly_canonical("/sys/class/video4linux/video" + std::to_string(videoIndex) + "/device", ec);
    if(ec || cur.empty()) {
        return;
    }

    while(!cur.empty()) {
        std::string vendor;
        std::string productId;
        if(readSysfsTextFile(cur / "idVendor", vendor) && readSysfsTextFile(cur / "idProduct", productId)) {
            out.usbVendorId = vendor;
            out.usbProductId = productId;
            std::string value;
            if(readSysfsTextFile(cur / "serial", value)) {
                out.usbSerial = value;
            }
            if(readSysfsTextFile(cur / "manufacturer", value)) {
                out.usbManufacturer = value;
            }
            if(readSysfsTextFile(cur / "product", value)) {
                out.usbProduct = value;
            }
            out.usbPortPath = cur.filename().string();
            return;
        }
        auto parent = cur.parent_path();
        if(parent == cur) {
            break;
        }
        cur = parent;
    }
}

bool queryCaptureCandidate(const std::string &devPath, FisheyeDeviceCandidate &out) {
    const int fd = ::open(devPath.c_str(), O_RDWR | O_NONBLOCK);
    if(fd < 0) {
        return false;
    }

    v4l2_capability cap{};
    if(::ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
        ::close(fd);
        return false;
    }

    const __u32 caps = (cap.device_caps != 0) ? cap.device_caps : cap.capabilities;
    const bool canCapture = (caps & V4L2_CAP_VIDEO_CAPTURE) || (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE);
    const bool hasStreaming = (caps & V4L2_CAP_STREAMING) || (caps & V4L2_CAP_READWRITE);
    if(!canCapture || !hasStreaming) {
        ::close(fd);
        return false;
    }

    out.devPath = devPath;
    out.cardName = reinterpret_cast<const char *>(cap.card);
    out.busInfo = reinterpret_cast<const char *>(cap.bus_info);

    v4l2_fmtdesc fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for(fmt.index = 0; ::ioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0; ++fmt.index) {
        if(fmt.pixelformat == V4L2_PIX_FMT_MJPEG) {
            out.supportsMjpeg = true;
        }
        else if(fmt.pixelformat == V4L2_PIX_FMT_YUYV) {
            out.supportsYuyv = true;
        }
    }

    ::close(fd);
    enrichCandidateUsbInfo(devPath, out);
    out.uniqueId = makeCandidateUniqueId(out);
    return true;
}

std::vector<FisheyeDeviceCandidate> enumerateVideoCaptureCandidates() {
    std::vector<FisheyeDeviceCandidate> out;
    std::unordered_map<std::string, std::string> stablePathByCanonical;

    const std::filesystem::path byIdDir("/dev/v4l/by-id");
    if(std::filesystem::exists(byIdDir)) {
        for(const auto &entry : std::filesystem::directory_iterator(byIdDir)) {
            std::error_code ec;
            const auto canonical = std::filesystem::weakly_canonical(entry.path(), ec);
            if(ec || canonical.empty()) {
                continue;
            }
            const std::string base = entry.path().filename().string();
            const std::string baseLower = toLowerCopy(base);
            if(baseLower.find("video-index") != std::string::npos && baseLower.find("video-index0") == std::string::npos) {
                continue;
            }
            stablePathByCanonical[canonical.string()] = entry.path().string();
        }
    }

    std::set<std::string> seen;
    for(const auto &entry : std::filesystem::directory_iterator("/dev")) {
        const auto filename = entry.path().filename().string();
        if(filename.rfind("video", 0) != 0) {
            continue;
        }
        const std::string devPath = entry.path().string();
        if(!seen.insert(devPath).second) {
            continue;
        }

        FisheyeDeviceCandidate candidate;
        if(!queryCaptureCandidate(devPath, candidate)) {
            continue;
        }

        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(entry.path(), ec);
        if(!ec) {
            auto it = stablePathByCanonical.find(canonical.string());
            if(it != stablePathByCanonical.end()) {
                candidate.stablePath = it->second;
            }
        }
        candidate.uniqueId = makeCandidateUniqueId(candidate);
        out.push_back(std::move(candidate));
    }

    std::sort(out.begin(), out.end(), [](const FisheyeDeviceCandidate &a, const FisheyeDeviceCandidate &b) {
        const std::string ka = !a.stablePath.empty() ? a.stablePath : a.devPath;
        const std::string kb = !b.stablePath.empty() ? b.stablePath : b.devPath;
        return ka < kb;
    });
    return out;
}
#else
std::vector<FisheyeDeviceCandidate> enumerateVideoCaptureCandidates() {
    return {};
}
#endif

std::vector<FisheyeDeviceCandidate> enumerateFisheyeDeviceCandidates() {
    const auto allCandidates = enumerateVideoCaptureCandidates();
    std::vector<FisheyeDeviceCandidate> out;
    out.reserve(allCandidates.size());
    std::set<std::string> positiveUsbSignatures;

    for(const auto &candidate : allCandidates) {
        if(candidateLooksLikeFisheye(candidate)) {
            const std::string usbSignature = makeUsbSignature(candidate);
            if(!usbSignature.empty()) {
                positiveUsbSignatures.insert(usbSignature);
            }
        }
    }

    for(const auto &candidate : allCandidates) {
        const bool explicitPositive = candidateLooksLikeFisheye(candidate);
        const bool knownNegative = candidateLooksLikeKnownNonFisheye(candidate);
        const std::string usbSignature = makeUsbSignature(candidate);
        const bool sameFamilyAsPositive = !usbSignature.empty() && positiveUsbSignatures.find(usbSignature) != positiveUsbSignatures.end();
        if(explicitPositive || (!knownNegative && sameFamilyAsPositive)) {
            out.push_back(candidate);
            continue;
        }
        if(fisheyeDebugEnabled()) {
            std::cerr << "[fisheye] reject candidate path=" << candidate.devPath
                      << " stable=" << candidate.stablePath
                      << " card=" << candidate.cardName
                      << " bus=" << candidate.busInfo
                      << " vendor=" << candidate.usbVendorId
                      << " productId=" << candidate.usbProductId
                      << " manufacturer=" << candidate.usbManufacturer
                      << " product=" << candidate.usbProduct
                      << " serial=" << candidate.usbSerial
                      << " uniqueId=" << candidate.uniqueId
                      << " reason=" << (knownNegative ? "known_non_fisheye" : "missing_fisheye_identity")
                      << std::endl;
        }
    }
    if(fisheyeDebugEnabled() && out.empty() && !allCandidates.empty()) {
        std::cerr << "[fisheye] no device matched known fisheye identity tokens; "
                  << "set SYNC_FISHEYE_HINTS with additional model/vendor keywords if needed"
                  << std::endl;
    }
    std::sort(out.begin(), out.end(), [](const FisheyeDeviceCandidate &a, const FisheyeDeviceCandidate &b) {
        const std::string ka = !a.uniqueId.empty() ? a.uniqueId : (!a.stablePath.empty() ? a.stablePath : a.devPath);
        const std::string kb = !b.uniqueId.empty() ? b.uniqueId : (!b.stablePath.empty() ? b.stablePath : b.devPath);
        return ka < kb;
    });
    return out;
}

std::optional<FisheyeDeviceCandidate> resolveCameraCandidate(const FisheyeCameraConfig &camera,
                                                             const std::vector<FisheyeDeviceCandidate> &candidates,
                                                             std::set<std::string> &usedPaths) {
    auto tryTake = [&](const FisheyeDeviceCandidate &candidate) -> bool {
        return usedPaths.insert(candidate.devPath).second;
    };

    if(!camera.devicePath.empty()) {
        for(const auto &candidate : candidates) {
            if(candidate.devPath == camera.devicePath || candidate.stablePath == camera.devicePath) {
                if(tryTake(candidate)) {
                    return candidate;
                }
                return std::nullopt;
            }
        }
    }

    if(camera.deviceIndex >= 0) {
        const std::string targetPath = "/dev/video" + std::to_string(camera.deviceIndex);
        for(const auto &candidate : candidates) {
            if(candidate.devPath == targetPath) {
                if(tryTake(candidate)) {
                    return candidate;
                }
                return std::nullopt;
            }
        }
    }

    if(!camera.preferredDeviceHint.empty()) {
        const std::string hintLower = toLowerCopy(camera.preferredDeviceHint);
        for(const auto &candidate : candidates) {
            const std::string stableLower = toLowerCopy(candidate.stablePath);
            const std::string cardLower = toLowerCopy(candidate.cardName);
            const std::string busLower = toLowerCopy(candidate.busInfo);
            const std::string uniqueLower = toLowerCopy(candidate.uniqueId);
            const std::string serialLower = toLowerCopy(candidate.usbSerial);
            const std::string productLower = toLowerCopy(candidate.usbProduct);
            const std::string manufacturerLower = toLowerCopy(candidate.usbManufacturer);
            if(stableLower.find(hintLower) == std::string::npos
               && cardLower.find(hintLower) == std::string::npos
               && busLower.find(hintLower) == std::string::npos
               && uniqueLower.find(hintLower) == std::string::npos
               && serialLower.find(hintLower) == std::string::npos
               && productLower.find(hintLower) == std::string::npos
               && manufacturerLower.find(hintLower) == std::string::npos) {
                continue;
            }
            if(tryTake(candidate)) {
                return candidate;
            }
        }
    }

    for(const auto &candidate : candidates) {
        const std::string stableLower = toLowerCopy(candidate.stablePath);
        const std::string cardLower = toLowerCopy(candidate.cardName);
        if(stableLower.find("orbbec") != std::string::npos || cardLower.find("orbbec") != std::string::npos) {
            continue;
        }
        if(tryTake(candidate)) {
            return candidate;
        }
    }

    return std::nullopt;
}

class OpenCvFisheyeModule final : public IFisheyeModule {
public:
    ~OpenCvFisheyeModule() override {
        stop();
    }

    std::string pluginId() const override {
        return "opencv_videocapture";
    }

    bool start(const FisheyeModuleConfig &config, std::string *errorMessage) override {
        stop();

        if(config.cameras.empty()) {
            if(errorMessage) {
                *errorMessage = "No fisheye cameras configured";
            }
            return false;
        }

        const auto allCandidates = enumerateVideoCaptureCandidates();
        const auto fisheyeCandidates = enumerateFisheyeDeviceCandidates();
        std::set<std::string> usedPaths;

        std::vector<std::shared_ptr<CameraState>> states;
        states.reserve(config.cameras.size());
        std::vector<std::string> skippedErrors;

        for(const auto &camera : config.cameras) {
            const auto &candidatePool = isExplicitCameraSelection(camera) ? allCandidates : fisheyeCandidates;
            auto resolved = resolveCameraCandidate(camera, candidatePool, usedPaths);
            if(!resolved) {
                skippedErrors.push_back("resolve failed for " + camera.cameraId);
                continue;
            }

            auto state = std::make_shared<CameraState>();
            state->config = camera;
            state->resolvedPath = !resolved->stablePath.empty() ? resolved->stablePath : resolved->devPath;
            state->resolvedVideoPath = resolved->devPath;
            state->cardName = resolved->cardName;
            state->busInfo = resolved->busInfo;
            state->uniqueId = resolved->uniqueId;

            std::string openError;
            if(!openCamera(state->cap, *resolved, camera, &openError)) {
                skippedErrors.push_back("open failed for " + camera.cameraId + ": " + openError);
                continue;
            }
            std::cerr << "[fisheye] opened cameraId=" << camera.cameraId
                      << " path=" << state->resolvedPath
                      << " videoNode=" << state->resolvedVideoPath
                      << " card=" << state->cardName
                      << " bus=" << state->busInfo
                      << " uniqueId=" << state->uniqueId << std::endl;
            states.push_back(std::move(state));
        }

        if(states.empty()) {
            if(errorMessage) {
                if(skippedErrors.empty()) {
                    *errorMessage = "No fisheye devices could be opened";
                }
                else {
                    std::ostringstream oss;
                    oss << "No fisheye devices could be opened";
                    for(const auto &msg : skippedErrors) {
                        oss << "; " << msg;
                    }
                    *errorMessage = oss.str();
                }
            }
            return false;
        }

        for(const auto &msg : skippedErrors) {
            std::cerr << "[fisheye] warning: " << msg << std::endl;
        }

        std::vector<std::shared_ptr<CameraState>> startStates;
        {
            std::lock_guard<std::mutex> lock(moduleMtx_);
            config_ = config;
            readyCount_ = 0;
            cameras_ = std::move(states);
            startStates = cameras_;
            running_.store(true);
        }

        for(auto &camera : startStates) {
            camera->worker = std::thread([this, state = camera]() { captureLoop(*state); });
        }

        return true;
    }

    void stop() override {
        std::vector<std::shared_ptr<CameraState>> states;
        {
            std::lock_guard<std::mutex> lock(moduleMtx_);
            if(cameras_.empty()) {
                running_.store(false);
                readyCount_ = 0;
                return;
            }
            running_.store(false);
            readyCount_ = 0;
            states.swap(cameras_);
        }

        for(auto &state : states) {
            state->stopRequested.store(true);
        }
        for(auto &state : states) {
            if(state->worker.joinable()) {
                state->worker.join();
            }
            state->cap.release();
        }
    }

    bool isRunning() const override {
        return running_.load();
    }

    FisheyeModuleConfig config() const override {
        std::lock_guard<std::mutex> lock(moduleMtx_);
        return config_;
    }

    bool waitUntilReady(std::chrono::milliseconds timeout) override {
        std::unique_lock<std::mutex> lock(moduleMtx_);
        return readyCv_.wait_for(lock, timeout, [&]() {
            return !running_.load() || (!cameras_.empty() && readyCount_ == cameras_.size());
        });
    }

    std::optional<FisheyeFrameSet> snapshotLatest(std::string *errorMessage) override {
        std::vector<std::shared_ptr<CameraState>> cameraPtrs;
        {
            std::lock_guard<std::mutex> moduleLock(moduleMtx_);
            if(!running_.load() || cameras_.empty()) {
                if(errorMessage) {
                    *errorMessage = "Fisheye module is not running";
                }
                return std::nullopt;
            }
            cameraPtrs.reserve(cameras_.size());
            for(const auto &camera : cameras_) {
                cameraPtrs.push_back(camera);
            }
        }

        std::vector<FisheyeFrame> frames;
        uint64_t tsSum = 0;

        frames.reserve(cameraPtrs.size());
        for(const auto &camera : cameraPtrs) {
            std::lock_guard<std::mutex> frameLock(camera->frameMtx);
            if(!camera->ready || camera->latestBgr.empty() || camera->latestTimestampUs == 0) {
                if(errorMessage) {
                    *errorMessage = "Fisheye camera not ready: " + camera->config.cameraId;
                }
                return std::nullopt;
            }

            FisheyeFrame frame;
            frame.cameraId = camera->config.cameraId;
            frame.deviceIndex = parseVideoIndex(camera->resolvedVideoPath);
            frame.captureTimestampUs = camera->latestTimestampUs;
            frame.captureTimestampSec = timestampUsToSec(camera->latestTimestampUs);
            frame.bgr = camera->latestBgr.clone();
            if(frame.bgr.empty()) {
                if(errorMessage) {
                    *errorMessage = "Failed to copy fisheye frame for " + camera->config.cameraId;
                }
                return std::nullopt;
            }
            tsSum += frame.captureTimestampUs;
            frames.push_back(std::move(frame));
        }

        FisheyeFrameSet frameSet;
        frameSet.frames = std::move(frames);
        frameSet.representativeTimestampUs = tsSum / std::max<uint64_t>(1, frameSet.frames.size());
        frameSet.representativeTimestampSec = timestampUsToSec(frameSet.representativeTimestampUs);
        return frameSet;
    }

private:
    struct CameraState {
        FisheyeCameraConfig config;
        cv::VideoCapture    cap;
        std::thread         worker;
        std::atomic_bool    stopRequested{ false };
        std::mutex          frameMtx;
        cv::Mat             latestBgr;
        uint64_t            latestTimestampUs = 0;
        bool                ready = false;
        std::string         resolvedPath;
        std::string         resolvedVideoPath;
        std::string         cardName;
        std::string         busInfo;
        std::string         uniqueId;
    };

    static bool openCamera(cv::VideoCapture &cap,
                           const FisheyeDeviceCandidate &candidate,
                           const FisheyeCameraConfig &camera,
                           std::string *errorMessage) {
        const std::string openPath = !candidate.stablePath.empty() ? candidate.stablePath : candidate.devPath;
        cap.open(openPath, cv::CAP_V4L2);
        if(!cap.isOpened()) {
            cap.open(candidate.devPath, cv::CAP_V4L2);
        }
        if(!cap.isOpened()) {
            cap.open(openPath);
        }
        if(!cap.isOpened()) {
            cap.open(candidate.devPath);
        }
        if(!cap.isOpened()) {
            if(errorMessage) {
                *errorMessage = "OpenCV failed to open " + openPath;
            }
            return false;
        }

        cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
        if(camera.preferMjpeg) {
            const double fourcc = static_cast<double>(cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            (void)cap.set(cv::CAP_PROP_FOURCC, fourcc);
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, camera.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, camera.height);
        cap.set(cv::CAP_PROP_FPS, camera.fps);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        const auto actualFourcc = static_cast<uint32_t>(cap.get(cv::CAP_PROP_FOURCC));
        const bool actualMjpeg = actualFourcc == static_cast<uint32_t>(cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        if(camera.preferMjpeg && candidate.supportsMjpeg && !actualMjpeg) {
            std::cerr << "[fisheye] warning: " << openPath << " did not switch to MJPG, actual fourcc=0x"
                      << std::hex << actualFourcc << std::dec << std::endl;
        }
        std::cerr << "[fisheye] configured " << openPath
                  << " width=" << cap.get(cv::CAP_PROP_FRAME_WIDTH)
                  << " height=" << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
                  << " fps=" << cap.get(cv::CAP_PROP_FPS)
                  << " fourcc=0x" << std::hex << actualFourcc << std::dec
                  << std::endl;
        return true;
    }

    void captureLoop(CameraState &state) {
        while(!state.stopRequested.load()) {
            cv::Mat frame;
            if(!state.cap.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            const uint64_t tsUs = systemClockNowUs();
            bool becameReady = false;
            {
                std::lock_guard<std::mutex> lock(state.frameMtx);
                state.latestBgr = frame.clone();
                state.latestTimestampUs = tsUs;
                if(!state.ready) {
                    state.ready = true;
                    becameReady = true;
                }
            }

            if(becameReady) {
                std::lock_guard<std::mutex> lock(moduleMtx_);
                readyCount_++;
                readyCv_.notify_all();
            }
        }
    }

    mutable std::mutex                         moduleMtx_;
    std::condition_variable                    readyCv_;
    FisheyeModuleConfig                        config_{};
    std::vector<std::shared_ptr<CameraState>>  cameras_;
    size_t                                     readyCount_ = 0;
    std::atomic_bool                           running_{ false };
};

}  // namespace

std::string fisheyeImageFormatToString(FisheyeImageFormat format) {
    switch(format) {
    case FisheyeImageFormat::Jpeg:
        return "jpg";
    case FisheyeImageFormat::Png:
        return "png";
    }
    return "jpg";
}

std::string fisheyeImageExtension(FisheyeImageFormat format) {
    return "." + fisheyeImageFormatToString(format);
}

std::string formatFisheyeTimestampUs(uint64_t timestampUs) {
    std::ostringstream oss;
    oss << (timestampUs / 1000000ULL) << "." << std::setw(6) << std::setfill('0') << (timestampUs % 1000000ULL);
    return oss.str();
}

std::optional<FisheyeNearestMatch> findNearestFisheyeSample(const FisheyeDatasetIndex &index, uint64_t targetTimestampUs) {
    if(index.samples.empty()) {
        return std::nullopt;
    }

    auto absDiff = [](uint64_t a, uint64_t b) {
        return a > b ? (a - b) : (b - a);
    };

    auto it = std::lower_bound(index.samples.begin(), index.samples.end(), targetTimestampUs,
                               [](const FisheyeSavedSample &sample, uint64_t tsUs) {
                                   return sample.representativeTimestampUs < tsUs;
                               });

    size_t bestIndex = 0;
    uint64_t bestDiff = std::numeric_limits<uint64_t>::max();
    if(it != index.samples.end()) {
        const size_t idx = static_cast<size_t>(std::distance(index.samples.begin(), it));
        bestIndex = idx;
        bestDiff = absDiff(index.samples[idx].representativeTimestampUs, targetTimestampUs);
    }
    if(it != index.samples.begin()) {
        const size_t idx = static_cast<size_t>(std::distance(index.samples.begin(), it - 1));
        const uint64_t diff = absDiff(index.samples[idx].representativeTimestampUs, targetTimestampUs);
        if(diff <= bestDiff) {
            bestIndex = idx;
            bestDiff = diff;
        }
    }

    return FisheyeNearestMatch{ bestIndex, bestDiff, &index.samples[bestIndex] };
}

std::vector<FisheyeDeviceInfo> listAvailableFisheyeDevices() {
    const auto candidates = enumerateFisheyeDeviceCandidates();
    std::vector<FisheyeDeviceInfo> out;
    out.reserve(candidates.size());
    for(const auto &candidate : candidates) {
        out.push_back(toDeviceInfo(candidate));
    }
    return out;
}

std::vector<FisheyeDeviceInfo> listPreferredFisheyeDevices(const std::vector<std::string> &preferredLabels) {
    const auto allCandidates = enumerateVideoCaptureCandidates();
    const auto fisheyeCandidates = enumerateFisheyeDeviceCandidates();

    std::vector<std::optional<FisheyeDeviceCandidate>> selected(preferredLabels.size());
    std::set<std::string> usedPaths;

    for(size_t i = 0; i < preferredLabels.size(); ++i) {
        auto it = std::find_if(allCandidates.begin(), allCandidates.end(), [&](const FisheyeDeviceCandidate &candidate) {
            return usedPaths.find(candidate.devPath) == usedPaths.end() && candidateMatchesCameraLabel(candidate, preferredLabels[i]);
        });
        if(it != allCandidates.end()) {
            usedPaths.insert(it->devPath);
            selected[i] = *it;
        }
    }

    for(size_t i = 0; i < selected.size(); ++i) {
        if(selected[i].has_value()) {
            continue;
        }
        for(const auto &candidate : fisheyeCandidates) {
            if(usedPaths.insert(candidate.devPath).second) {
                selected[i] = candidate;
                break;
            }
        }
    }

    std::vector<FisheyeDeviceInfo> out;
    out.reserve(selected.size());
    for(const auto &candidate : selected) {
        if(candidate.has_value()) {
            out.push_back(toDeviceInfo(*candidate));
        }
    }

    return out;
}

std::unique_ptr<IFisheyeModule> createOpenCvFisheyeModule() {
    return std::make_unique<OpenCvFisheyeModule>();
}

FisheyeRecorder::FisheyeRecorder(std::unique_ptr<IFisheyeModule> module)
    : module_(std::move(module)) {
}

FisheyeRecorder::~FisheyeRecorder() {
    stop();
}

bool FisheyeRecorder::start(const FisheyeModuleConfig &config, std::string *errorMessage) {
    stop();

    if(!module_) {
        module_ = createOpenCvFisheyeModule();
    }
    if(!module_) {
        if(errorMessage) {
            *errorMessage = "Failed to create fisheye module";
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
    return true;
}

void FisheyeRecorder::stop() {
    running_.store(false);
    nextCaptureTimeValid_ = false;
    if(module_) {
        module_->stop();
    }
}

bool FisheyeRecorder::isRunning() const {
    return running_.load() && module_ && module_->isRunning();
}

bool FisheyeRecorder::waitUntilReady(std::chrono::milliseconds timeout) {
    return module_ && module_->waitUntilReady(timeout);
}

std::optional<FisheyeFrameSet> FisheyeRecorder::snapshotLatest(std::string *errorMessage) {
    if(!isRunning()) {
        if(errorMessage) {
            *errorMessage = "Fisheye recorder is not running";
        }
        return std::nullopt;
    }
    return module_->snapshotLatest(errorMessage);
}

std::optional<FisheyeFrameSet> FisheyeRecorder::captureNext(std::string *errorMessage) {
    if(!isRunning()) {
        if(errorMessage) {
            *errorMessage = "Fisheye recorder is not running";
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

    auto frameSet = module_->snapshotLatest(errorMessage);
    if(!frameSet) {
        return std::nullopt;
    }
    frameSet->sequence = nextSequence_++;

    nextCaptureTime_ += period;
    if(nextCaptureTime_ < std::chrono::steady_clock::now()) {
        nextCaptureTime_ = std::chrono::steady_clock::now();
    }
    return frameSet;
}

bool FisheyeRecorder::captureFor(std::chrono::milliseconds duration,
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

bool FisheyeRecorder::saveFrameSets(const std::vector<FisheyeFrameSet> &frameSets,
                                    const std::filesystem::path &saveRoot,
                                    const FisheyeSaveOptions &options,
                                    FisheyeDatasetIndex *indexOut,
                                    std::string *errorMessage) {
    if(frameSets.empty()) {
        if(errorMessage) {
            *errorMessage = "No fisheye frames to save";
        }
        return false;
    }
    if(saveRoot.empty()) {
        if(errorMessage) {
            *errorMessage = "Fisheye save path is empty";
        }
        return false;
    }

    const auto &firstSet = frameSets.front();
    if(firstSet.frames.empty()) {
        if(errorMessage) {
            *errorMessage = "First fisheye frame set is empty";
        }
        return false;
    }

    std::vector<std::string> cameraOrder;
    cameraOrder.reserve(firstSet.frames.size());
    for(const auto &frame : firstSet.frames) {
        cameraOrder.push_back(frame.cameraId);
    }

    try {
        std::filesystem::create_directories(saveRoot);
        for(const auto &cameraId : cameraOrder) {
            std::filesystem::create_directories(saveRoot / cameraId);
        }
    }
    catch(const std::filesystem::filesystem_error &ex) {
        if(errorMessage) {
            *errorMessage = ex.what();
        }
        return false;
    }

    FisheyeDatasetIndex localIndex;
    localIndex.cameraOrder = cameraOrder;
    localIndex.samples.reserve(frameSets.size());

    const std::string ext = fisheyeImageExtension(options.format);
    const std::filesystem::path csvPath = saveRoot / "timestamps.csv";
    const std::filesystem::path csvTmpPath = saveRoot / "timestamps.csv.tmp";
    std::ofstream csv(csvTmpPath);
    if(!csv.is_open()) {
        if(errorMessage) {
            *errorMessage = "Failed to open fisheye timestamps csv for writing: " + csvTmpPath.string();
        }
        return false;
    }

    csv << "frame_id,timestamp_s";
    for(const auto &cameraId : cameraOrder) {
        csv << "," << cameraId << "_file";
    }
    csv << "\n";

    for(size_t setIdx = 0; setIdx < frameSets.size(); ++setIdx) {
        const auto &frameSet = frameSets[setIdx];
        if(frameSet.frames.size() != cameraOrder.size()) {
            if(errorMessage) {
                *errorMessage = "Inconsistent fisheye camera count in frame set " + std::to_string(setIdx);
            }
            return false;
        }

        const std::string tsString = formatFisheyeTimestampUs(frameSet.representativeTimestampUs);
        FisheyeSavedSample sample;
        sample.sequence = frameSet.sequence;
        sample.representativeTimestampUs = frameSet.representativeTimestampUs;
        sample.representativeTimestampSec = frameSet.representativeTimestampSec;
        sample.relativePaths.reserve(frameSet.frames.size());

        csv << setIdx << "," << tsString;
        for(size_t frameIdx = 0; frameIdx < frameSet.frames.size(); ++frameIdx) {
            const auto &frame = frameSet.frames[frameIdx];
            if(frame.cameraId != cameraOrder[frameIdx]) {
                if(errorMessage) {
                    *errorMessage = "Fisheye camera order changed at frame set " + std::to_string(setIdx);
                }
                return false;
            }
            const std::filesystem::path relative = std::filesystem::path(frame.cameraId) / (tsString + ext);
            if(!saveBgrImage(frame.bgr, saveRoot / relative, options, errorMessage)) {
                return false;
            }
            sample.relativePaths.push_back(relative.generic_string());
            csv << "," << sample.relativePaths.back();
        }
        csv << "\n";
        localIndex.samples.push_back(std::move(sample));
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

bool FisheyeRecorder::loadDatasetIndexCsv(const std::filesystem::path &csvPath,
                                         FisheyeDatasetIndex *indexOut,
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
            *errorMessage = "Failed to open fisheye timestamps csv: " + csvPath.string();
        }
        return false;
    }

    std::string headerLine;
    if(!std::getline(ifs, headerLine)) {
        if(errorMessage) {
            *errorMessage = "Fisheye timestamps csv is empty";
        }
        return false;
    }

    const auto headers = splitCsvLineSimple(headerLine);
    if(headers.size() < 3 || headers[0] != "frame_id" || headers[1] != "timestamp_s") {
        if(errorMessage) {
            *errorMessage = "Unexpected fisheye timestamps csv header";
        }
        return false;
    }

    FisheyeDatasetIndex parsed;
    for(size_t i = 2; i < headers.size(); ++i) {
        std::string cameraId = headers[i];
        const std::string suffix = "_file";
        if(cameraId.size() > suffix.size() && cameraId.compare(cameraId.size() - suffix.size(), suffix.size(), suffix) == 0) {
            cameraId.resize(cameraId.size() - suffix.size());
        }
        parsed.cameraOrder.push_back(std::move(cameraId));
    }

    std::string line;
    while(std::getline(ifs, line)) {
        if(line.empty()) {
            continue;
        }
        const auto cols = splitCsvLineSimple(line);
        if(cols.size() != headers.size()) {
            if(errorMessage) {
                *errorMessage = "Unexpected fisheye timestamps csv row width";
            }
            return false;
        }

        FisheyeSavedSample sample;
        try {
            sample.sequence = static_cast<uint64_t>(std::stoull(cols[0]));
        }
        catch(...) {
            if(errorMessage) {
                *errorMessage = "Invalid fisheye frame_id in timestamps csv";
            }
            return false;
        }
        sample.representativeTimestampUs = parseTimestampSecToUs(cols[1]);
        sample.representativeTimestampSec = timestampUsToSec(sample.representativeTimestampUs);
        for(size_t i = 2; i < cols.size(); ++i) {
            sample.relativePaths.push_back(cols[i]);
        }
        parsed.samples.push_back(std::move(sample));
    }

    *indexOut = std::move(parsed);
    return true;
}

}  // namespace sync_app
