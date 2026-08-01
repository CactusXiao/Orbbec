#include "interactive_visualization.hpp"
#include "fisheyes.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_set>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <execinfo.h>
#endif

namespace sync_app {

static std::atomic<const char *> g_iviz_stage{"init"};

static bool isCtrlModifierKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return key == static_cast<int>(0xFFE3) || key == static_cast<int>(0xFFE4) || lo16 == static_cast<int>(0xFFE3)
           || lo16 == static_cast<int>(0xFFE4) || lo8 == 0xE3 || lo8 == 0xE4 || key == 227 || key == 228;
}

static bool isCtrlReleaseKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == 0x007F || lo8 == 0x7F || key == 127;
}

static bool isCtrlZoomInKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == '+' || lo16 == '=' || lo16 == static_cast<int>(0xFFAB) || lo8 == '+' || lo8 == '=' || lo8 == 0xAB;
}

static bool isCtrlZoomOutKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == '-' || lo16 == '_' || lo16 == static_cast<int>(0xFFAD) || lo8 == '-' || lo8 == '_' || lo8 == 0xAD;
}

static bool g_ivizCtrlShortcutListening = false;

static inline void ivizSetStage(const char *s) {
    g_iviz_stage.store(s, std::memory_order_relaxed);
}

static void ivizInstallCrashHandlerOnce() {
    static std::atomic<bool> installed{false};
    bool expected = false;
    if(!installed.compare_exchange_strong(expected, true)) {
        return;
    }

#if defined(__linux__)
    auto handler = +[](int sig) {
        const char *stage = g_iviz_stage.load(std::memory_order_relaxed);
        char buf[512];
        int n = snprintf(buf, sizeof(buf), "\n[interactive_visualization] signal=%d stage=%s\n", sig, stage ? stage : "(null)");
        if(n > 0) {
            (void)write(2, buf, static_cast<size_t>(n));
        }

        void *bt[64];
        const int sz = backtrace(bt, 64);
        backtrace_symbols_fd(bt, sz, 2);
        _exit(128 + sig);
    };

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
#endif
}

struct CvMouseState {
    int x = 0;
    int y = 0;
    bool clicked = false;
    int clickX = 0;
    int clickY = 0;
    int wheelDelta = 0;
};

static bool uiButton(cv::Mat &img, const cv::Rect &r, const std::string &label, CvMouseState &ms) {
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar bg = hover ? cv::Scalar(60, 60, 60) : cv::Scalar(40, 40, 40);
    cv::rectangle(img, r, bg, cv::FILLED);
    cv::rectangle(img, r, cv::Scalar(120, 120, 120), 1);
    cv::putText(img, label, cv::Point(r.x + 10, r.y + r.height / 2 + 6), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static bool uiCheckbox(cv::Mat &img, const cv::Rect &r, bool checked, const std::string &label, CvMouseState &ms) {
    const cv::Rect box(r.x, r.y + 6, 18, 18);
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::rectangle(img, box, cv::Scalar(200, 200, 200), 1);
    if(checked) {
        cv::rectangle(img, box, cv::Scalar(80, 200, 80), cv::FILLED);
        cv::rectangle(img, box, cv::Scalar(200, 200, 200), 1);
    }
    cv::putText(img, label, cv::Point(r.x + 28, r.y + 20), cv::FONT_HERSHEY_DUPLEX, 0.55, hover ? cv::Scalar(255, 255, 255) : cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static std::string ellipsizeTextToWidth(const std::string &text, int maxWidthPx, int fontFace, double fontScale, int thickness) {
    if(maxWidthPx <= 0 || text.empty()) {
        return "";
    }
    int baseline = 0;
    if(cv::getTextSize(text, fontFace, fontScale, thickness, &baseline).width <= maxWidthPx) {
        return text;
    }
    const std::string suffix = "...";
    if(cv::getTextSize(suffix, fontFace, fontScale, thickness, &baseline).width > maxWidthPx) {
        return "";
    }
    std::string clipped = text;
    while(!clipped.empty()) {
        clipped.pop_back();
        if(cv::getTextSize(clipped + suffix, fontFace, fontScale, thickness, &baseline).width <= maxWidthPx) {
            return clipped + suffix;
        }
    }
    return suffix;
}

struct ExtrinsicCamToWorld {
    bool      valid = false;
    cv::Matx33f R   = cv::Matx33f::eye();
    cv::Vec3f   t   = cv::Vec3f(0, 0, 0);
};

struct CachedFrameBundle {
    uint64_t tsUs = 0;
    std::shared_ptr<ob::DepthFrame> depth;
    std::shared_ptr<ob::ColorFrame> color;
    std::shared_ptr<ob::IRFrame> irLeft;
    std::shared_ptr<ob::IRFrame> irRight;
    std::shared_ptr<ob::IRFrame> ir;
    std::string sn;
    std::string camIndex;
};

struct ExtrinsicHealthPose {
    bool      valid = false;
    cv::Matx33f R   = cv::Matx33f::eye();
    cv::Vec3f   t   = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct ExtrinsicHealthTagObservation {
    std::string          cameraId;
    int                  deviceIndex = -1;
    double               rmsePx = 0.0;
    bool                 inlier = false;
    ExtrinsicHealthPose  worldFromTag;
};

struct ExtrinsicHealthTagResult {
    int                                      tagId = -1;
    std::vector<ExtrinsicHealthTagObservation> observations;
    ExtrinsicHealthPose                     fusedWorldFromTag;
    bool                                     hasFused = false;
    std::vector<std::string>                 fusedInlierCameras;
};

struct ExtrinsicHealthSampleResult {
    bool                                      valid = false;
    std::string                               status;
    std::string                               summary;
    std::string                               debugDir;
    std::vector<std::string>                  detailLines;
    std::map<int, ExtrinsicHealthTagResult>   tags;
};

enum class ExtrinsicHealthVizMode {
    AllTags,
    SingleTag
};

struct ExtrinsicHealthCameraRequest {
    int                         deviceIndex = -1;
    std::string                 sn;
    std::string                 camIndex;
    std::shared_ptr<ob::Device> device;
    std::shared_ptr<ob::DepthFrame> depth;
    std::shared_ptr<ob::ColorFrame> color;
    OBCameraParam               rgbDepthParam{};
    bool                        rgbDepthParamValid = false;
    uint64_t                    tsUs = 0;
};

struct ExtrinsicHealthCapturedCamera {
    int           deviceIndex = -1;
    std::string   sn;
    std::string   camIndex;
    cv::Mat       bgr;
    cv::Mat       depthAlignedRgb16;
    OBCameraParam rgbDepthParam{};
    bool          rgbDepthParamValid = false;
    uint64_t      rgbTsUs = 0;
    uint64_t      depthTsUs = 0;
    float         depthValueScaleMm = 1.0f;
    OBFormat      colorFormat = OB_FORMAT_UNKNOWN;
    OBFormat      depthFormat = OB_FORMAT_UNKNOWN;
};

static uint64_t interactiveAlignedMaxAbsDiffUs(uint64_t stepUs) {
    const uint64_t halfWinUs = stepUs / 2;
    const uint64_t tolUs = std::max<uint64_t>(2000, stepUs / 10);
    return halfWinUs + tolUs;
}

static bool pickNearestFrameBundle(const std::deque<CachedFrameBundle> &items, uint64_t centerUs, uint64_t maxAbsDiffUs, size_t &picked) {
    if(items.empty()) {
        return false;
    }
    auto absDiff = [](uint64_t a, uint64_t b) {
        return a > b ? (a - b) : (b - a);
    };
    auto it = std::lower_bound(items.begin(), items.end(), centerUs, [](const CachedFrameBundle &item, uint64_t ts) { return item.tsUs < ts; });
    size_t cand0 = (it == items.end()) ? (items.size() - 1) : static_cast<size_t>(std::distance(items.begin(), it));
    size_t cand1 = (cand0 > 0) ? (cand0 - 1) : cand0;
    const uint64_t d0 = absDiff(items[cand0].tsUs, centerUs);
    const uint64_t d1 = absDiff(items[cand1].tsUs, centerUs);
    const size_t chosen = (d1 <= d0) ? cand1 : cand0;
    if(absDiff(items[chosen].tsUs, centerUs) > maxAbsDiffUs) {
        return false;
    }
    picked = chosen;
    return true;
}

struct GtCameraFramePacket {
    std::string camIndex;
    int         deviceIndex = -1;
    cv::Mat     rgbBgr;
    cv::Mat     depthAlignedRgb16;
    float       rgbScaleX = 1.0f;
    float       rgbScaleY = 1.0f;
    float       depthScaleX = 1.0f;
    float       depthScaleY = 1.0f;
    float       depthUnitMm = 1.0f;
    OBCameraIntrinsic rgbIntrinsic{};
    cv::Matx33f Rcw = cv::Matx33f::eye();
    cv::Vec3f   tcw = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Matx33f Rwc = cv::Matx33f::eye();
    cv::Vec3f   twc = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct GtInferenceRequest {
    uint64_t                       frameId = 0;
    uint64_t                       captureTsUs = 0;
    std::vector<GtCameraFramePacket> cameras;
};

struct GtJoint3d {
    bool        valid = false;
    std::string side;
    int         jointIndex = -1;
    cv::Vec3f   position = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Vec3b   color = cv::Vec3b(255, 255, 255);
    float       reprojectionErrorPx = 0.0f;
};

struct GtHand3d {
    int                    trackId = -1;
    std::string            side;
    bool                   visible = false;
    int                    validJointCount = 0;
    float                  avgReprojectionErrorPx = 0.0f;
    std::vector<GtJoint3d> joints;
};

struct GtInferenceResult {
    bool                 valid = false;
    uint64_t             frameId = 0;
    uint64_t             captureTsUs = 0;
    std::vector<GtHand3d> hands;
    int                  visibleHands = 0;
    double               workerFps = 0.0;
    std::string          status;
};

struct EgoAprilTagCameraPacket {
    std::string camIndex;
    int         deviceIndex = -1;
    cv::Mat     rgbBgr;
    float       rgbScaleX = 1.0f;
    float       rgbScaleY = 1.0f;
    OBCameraIntrinsic rgbIntrinsic{};
    OBCameraDistortion rgbDistortion{};
    cv::Matx33f Rwc = cv::Matx33f::eye();
    cv::Vec3f   twc = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct EgoAprilTagRequest {
    uint64_t frameId = 0;
    int      egoVideoFrameIndex = -1;
    fs::path egoVideoPath;
    fs::path egoCameraParamsPath;
    std::vector<EgoAprilTagCameraPacket> cameras;
};

struct EgoAprilTagLine {
    cv::Vec3f p0 = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Vec3f p1 = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Vec3b color = cv::Vec3b(255, 255, 255);
};

struct EgoAprilTagResult {
    bool                       valid = false;
    uint64_t                   frameId = 0;
    int                        egoVideoFrameIndex = -1;
    int                        tagCount = 0;
    int                        referenceTagCount = 0;
    double                     rmsePx = 0.0;
    double                     workerFps = 0.0;
    std::string                status;
    std::vector<EgoAprilTagLine> lines;
};

static bool isDigits(const std::string &s) {
    if(s.empty()) {
        return false;
    }
    for(char c: s) {
        if(!(c >= '0' && c <= '9')) {
            return false;
        }
    }
    return true;
}

static std::string stripLeadingZeros(const std::string &s) {
    if(s.empty()) {
        return s;
    }
    size_t i = 0;
    while(i + 1 < s.size() && s[i] == '0') {
        i++;
    }
    return s.substr(i);
}

static std::string padLeftZeros(const std::string &s, size_t width) {
    if(s.size() >= width) {
        return s;
    }
    return std::string(width - s.size(), '0') + s;
}

template <class MapT>
static const typename MapT::mapped_type *findByCamKeyVariants(const MapT &m, const std::string &camKey) {
    auto it = m.find(camKey);
    if(it != m.end()) {
        return &it->second;
    }
    const std::string stripped = stripLeadingZeros(camKey);
    it = m.find(stripped);
    if(it != m.end()) {
        return &it->second;
    }
    if(isDigits(camKey)) {
        const std::string pad2 = padLeftZeros(stripped, 2);
        it = m.find(pad2);
        if(it != m.end()) {
            return &it->second;
        }
        const std::string pad3 = padLeftZeros(stripped, 3);
        it = m.find(pad3);
        if(it != m.end()) {
            return &it->second;
        }
        const std::string pad4 = padLeftZeros(stripped, 4);
        it = m.find(pad4);
        if(it != m.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

static bool containsCamKeyVariant(const std::unordered_set<std::string> &items, const std::string &camKey) {
    if(items.find(camKey) != items.end()) {
        return true;
    }
    const std::string stripped = stripLeadingZeros(camKey);
    if(items.find(stripped) != items.end()) {
        return true;
    }
    if(isDigits(camKey)) {
        if(items.find(padLeftZeros(stripped, 2)) != items.end()) {
            return true;
        }
        if(items.find(padLeftZeros(stripped, 3)) != items.end()) {
            return true;
        }
        if(items.find(padLeftZeros(stripped, 4)) != items.end()) {
            return true;
        }
    }
    return false;
}

static size_t matByteSize(const cv::Mat &m) {
    return m.empty() ? 0u : (m.total() * m.elemSize());
}

static cv::Mat resizeMatKeepingAspect(const cv::Mat &src, int maxSide, int interpolation, float &scaleX, float &scaleY) {
    scaleX = 1.0f;
    scaleY = 1.0f;
    if(src.empty() || maxSide <= 0) {
        return src.clone();
    }
    const int longSide = std::max(src.cols, src.rows);
    if(longSide <= maxSide) {
        return src.clone();
    }
    const double scale = static_cast<double>(maxSide) / static_cast<double>(longSide);
    const int newW = std::max(1, static_cast<int>(std::lround(static_cast<double>(src.cols) * scale)));
    const int newH = std::max(1, static_cast<int>(std::lround(static_cast<double>(src.rows) * scale)));
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(newW, newH), 0.0, 0.0, interpolation);
    scaleX = static_cast<float>(src.cols) / static_cast<float>(newW);
    scaleY = static_cast<float>(src.rows) / static_cast<float>(newH);
    return dst;
}

static cv::Mat extractDepth16Mat(const std::shared_ptr<ob::DepthFrame> &depthFrame) {
    if(!depthFrame) {
        return cv::Mat();
    }
    const auto fmt = depthFrame->getFormat();
    if(fmt != OB_FORMAT_Y16 && fmt != OB_FORMAT_Y14 && fmt != OB_FORMAT_Z16 && fmt != OB_FORMAT_Y12C4) {
        return cv::Mat();
    }
    const int width = static_cast<int>(depthFrame->getWidth());
    const int height = static_cast<int>(depthFrame->getHeight());
    if(width <= 0 || height <= 0) {
        return cv::Mat();
    }
    void *raw = depthFrame->data();
    const size_t dataSize = static_cast<size_t>(depthFrame->dataSize());
    if(!raw || dataSize < static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uint16_t)) {
        return cv::Mat();
    }
    const size_t strideBytes = dataSize / static_cast<size_t>(height);
    if(strideBytes < static_cast<size_t>(width) * sizeof(uint16_t)) {
        return cv::Mat();
    }
    cv::Mat depth(height, width, CV_16UC1, raw, strideBytes);
    return depth.clone();
}

static cv::Mat buildDepthAlignedToRgb(const std::shared_ptr<ob::DepthFrame> &depthFrame, const OBCameraParam &cameraParam, int rgbWidth, int rgbHeight) {
    if(!depthFrame || rgbWidth <= 0 || rgbHeight <= 0) {
        return cv::Mat();
    }

    const cv::Mat depth16 = extractDepth16Mat(depthFrame);
    if(depth16.empty()) {
        return cv::Mat();
    }

    const float scaleMm = depthFrame->getValueScale();
    if(!(scaleMm > 0.0f)) {
        return cv::Mat();
    }

    cv::Mat aligned(rgbHeight, rgbWidth, CV_16UC1, cv::Scalar(0));
    for(int y = 0; y < depth16.rows; ++y) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        for(int x = 0; x < depth16.cols; ++x) {
            const uint16_t d = row[x];
            if(d == 0) {
                continue;
            }

            const float depthMm = static_cast<float>(d) * scaleMm;
            if(!(depthMm > 0.0f)) {
                continue;
            }

            OBPoint2f src{};
            src.x = static_cast<float>(x);
            src.y = static_cast<float>(y);
            OBPoint2f dst{};
            bool ok = false;
            try {
                ok = ob::CoordinateTransformHelper::transformation2dto2d(src,
                                                                         depthMm,
                                                                         cameraParam.depthIntrinsic,
                                                                         cameraParam.depthDistortion,
                                                                         cameraParam.rgbIntrinsic,
                                                                         cameraParam.rgbDistortion,
                                                                         cameraParam.transform,
                                                                         &dst);
            }
            catch(...) {
                ok = false;
            }
            if(!ok) {
                continue;
            }

            const int u = static_cast<int>(std::lround(dst.x));
            const int v = static_cast<int>(std::lround(dst.y));
            if(u < 0 || u >= rgbWidth || v < 0 || v >= rgbHeight) {
                continue;
            }

            uint16_t &slot = aligned.at<uint16_t>(v, u);
            if(slot == 0 || d < slot) {
                slot = d;
            }
        }
    }
    return aligned;
}

static cv::Mat buildDepthAlignedToRgbViaSdk(const std::shared_ptr<ob::Device> &device,
                                            const std::shared_ptr<ob::DepthFrame> &depthFrame,
                                            int                                     rgbWidth,
                                            int                                     rgbHeight) {
    if(!device || !depthFrame || rgbWidth <= 0 || rgbHeight <= 0) {
        return cv::Mat();
    }
    try {
        auto alignedFrame = ob::CoordinateTransformHelper::transformationDepthFrameToColorCamera(device,
                                                                                                  std::static_pointer_cast<ob::Frame>(depthFrame),
                                                                                                  static_cast<uint32_t>(rgbWidth),
                                                                                                  static_cast<uint32_t>(rgbHeight));
        if(!alignedFrame) {
            return cv::Mat();
        }
        auto alignedDepth = alignedFrame->as<ob::DepthFrame>();
        if(!alignedDepth) {
            return cv::Mat();
        }
        return extractDepth16Mat(alignedDepth);
    }
    catch(...) {
        return cv::Mat();
    }
}

static void putU32Le(uint8_t *dst, uint32_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xFFu);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    dst[2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    dst[3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

static void putU64Le(uint8_t *dst, uint64_t value) {
    for(int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xFFu);
    }
}

static uint32_t getU32Le(const uint8_t *src) {
    return static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) | (static_cast<uint32_t>(src[2]) << 16)
           | (static_cast<uint32_t>(src[3]) << 24);
}

static uint64_t getU64Le(const uint8_t *src) {
    uint64_t value = 0;
    for(int i = 0; i < 8; ++i) {
        value |= (static_cast<uint64_t>(src[i]) << (8 * i));
    }
    return value;
}

#if !defined(_WIN32)
static bool writeAllFd(int fd, const void *data, size_t size) {
    const auto *ptr = reinterpret_cast<const uint8_t *>(data);
    size_t      off = 0;
    while(off < size) {
        const ssize_t n = ::write(fd, ptr + off, size - off);
        if(n < 0) {
            if(errno == EINTR) {
                continue;
            }
            return false;
        }
        if(n == 0) {
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

static bool readAllFd(int fd, void *data, size_t size) {
    auto  *ptr = reinterpret_cast<uint8_t *>(data);
    size_t off = 0;
    while(off < size) {
        const ssize_t n = ::read(fd, ptr + off, size - off);
        if(n < 0) {
            if(errno == EINTR) {
                continue;
            }
            return false;
        }
        if(n == 0) {
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}
#endif

static cv::Scalar toScalar(const cv::Vec3b &bgr) {
    return cv::Scalar(bgr[0], bgr[1], bgr[2]);
}

static cv::Vec3b brightenColor(const cv::Vec3b &base, float amount01) {
    const float t = std::max(0.0f, std::min(1.0f, amount01));
    cv::Vec3b    out = base;
    for(int i = 0; i < 3; ++i) {
        const float v = static_cast<float>(base[i]) + (255.0f - static_cast<float>(base[i])) * t;
        out[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, v)));
    }
    return out;
}

static cv::Vec3b gtJointColorFor(const std::string &side, int jointIndex) {
    static const std::array<cv::Vec3b, 5> leftBases = { cv::Vec3b(255, 120, 0), cv::Vec3b(255, 180, 40), cv::Vec3b(255, 220, 80),
                                                         cv::Vec3b(220, 255, 120), cv::Vec3b(180, 255, 180) };
    static const std::array<cv::Vec3b, 5> rightBases = { cv::Vec3b(0, 70, 255), cv::Vec3b(0, 120, 255), cv::Vec3b(0, 180, 255),
                                                          cv::Vec3b(60, 220, 255), cv::Vec3b(120, 255, 255) };
    if(jointIndex <= 0) {
        return side == "Left" ? cv::Vec3b(255, 255, 255) : cv::Vec3b(210, 210, 255);
    }
    const int finger = std::min(4, std::max(0, (jointIndex - 1) / 4));
    const int step   = std::min(3, std::max(0, (jointIndex - 1) % 4));
    const auto &base = side == "Left" ? leftBases[static_cast<size_t>(finger)] : rightBases[static_cast<size_t>(finger)];
    return brightenColor(base, 0.12f * static_cast<float>(step));
}

static cv::Vec3b gtSkeletonColorForSide(const std::string &side) {
    return side == "Left" ? cv::Vec3b(190, 235, 255) : cv::Vec3b(255, 210, 160);
}

static std::string jsonStringLocal(const std::string &s);

class LiveGtJointWorker {
public:
    LiveGtJointWorker() = default;

    ~LiveGtJointWorker() {
        stop();
    }

    void setScriptPath(fs::path scriptPath) {
        std::lock_guard<std::mutex> lock(mtx_);
        scriptPath_ = std::move(scriptPath);
    }

    void ensureRunning() {
        std::lock_guard<std::mutex> lock(mtx_);
        if(workerThread_.joinable()) {
            return;
        }
        stopRequested_ = false;
        hasPendingRequest_ = false;
        latestResult_ = GtInferenceResult{};
        statusLine_ = "GT worker starting";
        workerThread_ = std::thread([this]() { workerLoop(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopRequested_ = true;
            hasPendingRequest_ = false;
            cv_.notify_all();
        }
        if(workerThread_.joinable()) {
            workerThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopRequested_ = false;
            latestResult_ = GtInferenceResult{};
            statusLine_ = "GT disabled";
        }
    }

    void submitLatest(GtInferenceRequest req) {
        ensureRunning();
        std::lock_guard<std::mutex> lock(mtx_);
        pendingRequest_ = std::move(req);
        hasPendingRequest_ = true;
        cv_.notify_one();
    }

    void setIdleStatus(const std::string &status, bool clearResult) {
        std::lock_guard<std::mutex> lock(mtx_);
        statusLine_ = status;
        if(clearResult) {
            latestResult_ = GtInferenceResult{};
            latestResult_.status = status;
            latestResult_.workerFps = smoothedFps_;
        }
    }

    GtInferenceResult latestResult() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return latestResult_;
    }

    bool hasPendingRequest() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return hasPendingRequest_;
    }

    std::string statusLine() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return statusLine_;
    }

private:
#if !defined(_WIN32)
    bool launchChild() {
        fs::path scriptPath;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            scriptPath = scriptPath_;
        }
        if(scriptPath.empty() || !fs::exists(scriptPath)) {
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "GT worker script not found";
            return false;
        }

        int stdinPipe[2] = { -1, -1 };
        int stdoutPipe[2] = { -1, -1 };
        if(::pipe(stdinPipe) != 0 || ::pipe(stdoutPipe) != 0) {
            if(stdinPipe[0] >= 0) {
                ::close(stdinPipe[0]);
            }
            if(stdinPipe[1] >= 0) {
                ::close(stdinPipe[1]);
            }
            if(stdoutPipe[0] >= 0) {
                ::close(stdoutPipe[0]);
            }
            if(stdoutPipe[1] >= 0) {
                ::close(stdoutPipe[1]);
            }
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "GT worker pipe creation failed";
            return false;
        }

        const pid_t pid = ::fork();
        if(pid < 0) {
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            ::close(stdoutPipe[0]);
            ::close(stdoutPipe[1]);
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "GT worker fork failed";
            return false;
        }

        if(pid == 0) {
            ::dup2(stdinPipe[0], STDIN_FILENO);
            ::dup2(stdoutPipe[1], STDOUT_FILENO);
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            ::close(stdoutPipe[0]);
            ::close(stdoutPipe[1]);
            ::execlp("python3", "python3", scriptPath.string().c_str(), "--stdio", nullptr);
            ::execlp("python", "python", scriptPath.string().c_str(), "--stdio", nullptr);
            _exit(127);
        }

        ::close(stdinPipe[0]);
        ::close(stdoutPipe[1]);
        childPid_ = pid;
        childStdinFd_ = stdinPipe[1];
        childStdoutFd_ = stdoutPipe[0];
        {
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "GT worker ready";
        }
        return true;
    }

    void closeChild() {
        if(childStdinFd_ >= 0) {
            ::close(childStdinFd_);
            childStdinFd_ = -1;
        }
        if(childStdoutFd_ >= 0) {
            ::close(childStdoutFd_);
            childStdoutFd_ = -1;
        }
        if(childPid_ > 0) {
            ::kill(childPid_, SIGTERM);
            int status = 0;
            ::waitpid(childPid_, &status, 0);
            childPid_ = -1;
        }
    }

    std::string buildRequestJson(const GtInferenceRequest &req, uint64_t &payloadBytesOut) const {
        payloadBytesOut = 0;
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(9);
        oss << "{\"type\":\"frame_batch\",\"frame_id\":" << req.frameId << ",\"timestamp_us\":" << req.captureTsUs << ",\"cameras\":[";
        uint64_t payloadOffset = 0;
        for(size_t i = 0; i < req.cameras.size(); ++i) {
            const auto &cam = req.cameras[i];
            const uint64_t rgbBytes = static_cast<uint64_t>(matByteSize(cam.rgbBgr));
            const uint64_t depthBytes = static_cast<uint64_t>(matByteSize(cam.depthAlignedRgb16));
            if(i != 0) {
                oss << ",";
            }
            oss << "{\"camera_id\":\"" << cam.camIndex << "\",\"device_index\":" << cam.deviceIndex << ",\"rgb_width\":" << cam.rgbBgr.cols << ",\"rgb_height\":" << cam.rgbBgr.rows
                << ",\"orig_rgb_width\":" << static_cast<int>(std::lround(static_cast<double>(cam.rgbBgr.cols) * static_cast<double>(cam.rgbScaleX))) << ",\"orig_rgb_height\":"
                << static_cast<int>(std::lround(static_cast<double>(cam.rgbBgr.rows) * static_cast<double>(cam.rgbScaleY))) << ",\"rgb_scale_x\":" << cam.rgbScaleX << ",\"rgb_scale_y\":"
                << cam.rgbScaleY << ",\"depth_width\":" << cam.depthAlignedRgb16.cols << ",\"depth_height\":" << cam.depthAlignedRgb16.rows << ",\"depth_scale_x\":" << cam.depthScaleX
                << ",\"depth_scale_y\":" << cam.depthScaleY << ",\"rgb_offset\":" << payloadOffset << ",\"rgb_size\":" << rgbBytes << ",\"depth_offset\":" << (payloadOffset + rgbBytes)
                << ",\"depth_size\":" << depthBytes << ",\"depth_unit_mm\":" << cam.depthUnitMm << ",\"intrinsic\":{\"fx\":" << cam.rgbIntrinsic.fx << ",\"fy\":" << cam.rgbIntrinsic.fy
                << ",\"cx\":" << cam.rgbIntrinsic.cx
                << ",\"cy\":" << cam.rgbIntrinsic.cy << "},\"Rcw\":[" << cam.Rcw(0, 0) << "," << cam.Rcw(0, 1) << "," << cam.Rcw(0, 2) << "," << cam.Rcw(1, 0) << "," << cam.Rcw(1, 1)
                << "," << cam.Rcw(1, 2) << "," << cam.Rcw(2, 0) << "," << cam.Rcw(2, 1) << "," << cam.Rcw(2, 2) << "],\"tcw\":[" << cam.tcw[0] << "," << cam.tcw[1] << ","
                << cam.tcw[2] << "],\"Rwc\":[" << cam.Rwc(0, 0) << "," << cam.Rwc(0, 1) << "," << cam.Rwc(0, 2) << "," << cam.Rwc(1, 0) << "," << cam.Rwc(1, 1) << ","
                << cam.Rwc(1, 2) << "," << cam.Rwc(2, 0) << "," << cam.Rwc(2, 1) << "," << cam.Rwc(2, 2) << "],\"twc\":[" << cam.twc[0] << "," << cam.twc[1] << "," << cam.twc[2]
                << "]}";
            payloadOffset += rgbBytes + depthBytes;
        }
        oss << "]}";
        payloadBytesOut = payloadOffset;
        return oss.str();
    }

    bool sendRequest(const GtInferenceRequest &req) {
        if(childStdinFd_ < 0) {
            return false;
        }
        uint64_t payloadBytes = 0;
        const std::string json = buildRequestJson(req, payloadBytes);
        uint8_t header[12];
        putU32Le(header, static_cast<uint32_t>(json.size()));
        putU64Le(header + 4, payloadBytes);
        if(!writeAllFd(childStdinFd_, header, sizeof(header))) {
            return false;
        }
        if(!json.empty() && !writeAllFd(childStdinFd_, json.data(), json.size())) {
            return false;
        }
        for(const auto &cam : req.cameras) {
            if(!cam.rgbBgr.empty() && !writeAllFd(childStdinFd_, cam.rgbBgr.data, matByteSize(cam.rgbBgr))) {
                return false;
            }
            if(!cam.depthAlignedRgb16.empty() && !writeAllFd(childStdinFd_, cam.depthAlignedRgb16.data, matByteSize(cam.depthAlignedRgb16))) {
                return false;
            }
        }
        return true;
    }

    bool readResponse(GtInferenceResult &out) {
        if(childStdoutFd_ < 0) {
            return false;
        }
        uint8_t header[12];
        if(!readAllFd(childStdoutFd_, header, sizeof(header))) {
            return false;
        }
        const uint32_t jsonBytes = getU32Le(header);
        const uint64_t payloadBytes = getU64Le(header + 4);
        if(payloadBytes != 0) {
            return false;
        }
        std::string json(jsonBytes, '\0');
        if(jsonBytes > 0 && !readAllFd(childStdoutFd_, json.data(), jsonBytes)) {
            return false;
        }

        cJSON *root = cJSON_Parse(json.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            return false;
        }

        out = GtInferenceResult{};
        auto *okItem = cJSON_GetObjectItemCaseSensitive(root, "ok");
        out.valid = okItem && cJSON_IsBool(okItem) ? (okItem->valueint != 0) : false;
        auto *frameIdItem = cJSON_GetObjectItemCaseSensitive(root, "frame_id");
        if(frameIdItem && cJSON_IsNumber(frameIdItem)) {
            out.frameId = static_cast<uint64_t>(frameIdItem->valuedouble);
        }
        auto *tsItem = cJSON_GetObjectItemCaseSensitive(root, "timestamp_us");
        if(tsItem && cJSON_IsNumber(tsItem)) {
            out.captureTsUs = static_cast<uint64_t>(tsItem->valuedouble);
        }
        auto *fpsItem = cJSON_GetObjectItemCaseSensitive(root, "fps");
        if(fpsItem && cJSON_IsNumber(fpsItem)) {
            out.workerFps = fpsItem->valuedouble;
        }
        auto *statusItem = cJSON_GetObjectItemCaseSensitive(root, "status");
        if(statusItem && cJSON_IsString(statusItem) && statusItem->valuestring) {
            out.status = statusItem->valuestring;
        }
        auto *visibleHandsItem = cJSON_GetObjectItemCaseSensitive(root, "visible_hands");
        if(visibleHandsItem && cJSON_IsNumber(visibleHandsItem)) {
            out.visibleHands = visibleHandsItem->valueint;
        }
        auto *handsItem = cJSON_GetObjectItemCaseSensitive(root, "hands");
        if(handsItem && cJSON_IsArray(handsItem)) {
            const int handCount = cJSON_GetArraySize(handsItem);
            out.hands.reserve(static_cast<size_t>(handCount));
            for(int hi = 0; hi < handCount; ++hi) {
                auto *handObj = cJSON_GetArrayItem(handsItem, hi);
                if(!handObj || !cJSON_IsObject(handObj)) {
                    continue;
                }

                GtHand3d hand;
                auto *trackIdItem = cJSON_GetObjectItemCaseSensitive(handObj, "track_id");
                if(trackIdItem && cJSON_IsNumber(trackIdItem)) {
                    hand.trackId = trackIdItem->valueint;
                }
                auto *sideItem = cJSON_GetObjectItemCaseSensitive(handObj, "side");
                if(sideItem && cJSON_IsString(sideItem) && sideItem->valuestring) {
                    hand.side = sideItem->valuestring;
                }
                auto *visibleItem = cJSON_GetObjectItemCaseSensitive(handObj, "visible");
                hand.visible = visibleItem && cJSON_IsBool(visibleItem) ? (visibleItem->valueint != 0) : false;
                auto *countItem = cJSON_GetObjectItemCaseSensitive(handObj, "valid_joint_count");
                if(countItem && cJSON_IsNumber(countItem)) {
                    hand.validJointCount = countItem->valueint;
                }
                auto *avgErrItem = cJSON_GetObjectItemCaseSensitive(handObj, "avg_reproj_error_px");
                if(avgErrItem && cJSON_IsNumber(avgErrItem)) {
                    hand.avgReprojectionErrorPx = static_cast<float>(avgErrItem->valuedouble);
                }

                hand.joints.assign(21, GtJoint3d{});
                auto *jointsItem = cJSON_GetObjectItemCaseSensitive(handObj, "joints");
                if(jointsItem && cJSON_IsArray(jointsItem)) {
                    const int jointCount = cJSON_GetArraySize(jointsItem);
                    for(int ji = 0; ji < jointCount; ++ji) {
                        auto *jointObj = cJSON_GetArrayItem(jointsItem, ji);
                        if(!jointObj || !cJSON_IsObject(jointObj)) {
                            continue;
                        }
                        auto *indexItem = cJSON_GetObjectItemCaseSensitive(jointObj, "index");
                        if(!indexItem || !cJSON_IsNumber(indexItem)) {
                            continue;
                        }
                        const int jointIndex = indexItem->valueint;
                        if(jointIndex < 0 || jointIndex >= 21) {
                            continue;
                        }

                        GtJoint3d joint;
                        joint.side = hand.side;
                        joint.jointIndex = jointIndex;
                        auto *validItem = cJSON_GetObjectItemCaseSensitive(jointObj, "valid");
                        joint.valid = validItem && cJSON_IsBool(validItem) ? (validItem->valueint != 0) : false;
                        if(joint.valid) {
                            auto *xyzItem = cJSON_GetObjectItemCaseSensitive(jointObj, "xyz");
                            if(xyzItem && cJSON_IsArray(xyzItem) && cJSON_GetArraySize(xyzItem) == 3) {
                                auto *xItem = cJSON_GetArrayItem(xyzItem, 0);
                                auto *yItem = cJSON_GetArrayItem(xyzItem, 1);
                                auto *zItem = cJSON_GetArrayItem(xyzItem, 2);
                                if(xItem && yItem && zItem && cJSON_IsNumber(xItem) && cJSON_IsNumber(yItem) && cJSON_IsNumber(zItem)) {
                                    joint.position = cv::Vec3f(static_cast<float>(xItem->valuedouble),
                                                               static_cast<float>(yItem->valuedouble),
                                                               static_cast<float>(zItem->valuedouble));
                                }
                                else {
                                    joint.valid = false;
                                }
                            }
                            else {
                                joint.valid = false;
                            }
                        }
                        joint.color = gtJointColorFor(joint.side, joint.jointIndex);
                        auto *errItem = cJSON_GetObjectItemCaseSensitive(jointObj, "reproj_error_px");
                        if(errItem && cJSON_IsNumber(errItem)) {
                            joint.reprojectionErrorPx = static_cast<float>(errItem->valuedouble);
                        }
                        hand.joints[static_cast<size_t>(jointIndex)] = std::move(joint);
                    }
                }

                out.hands.push_back(std::move(hand));
            }
        }

        if(out.visibleHands <= 0) {
            for(const auto &hand : out.hands) {
                if(hand.visible) {
                    out.visibleHands++;
                }
            }
        }

        cJSON_Delete(root);
        return true;
    }
#endif

    void workerLoop() {
#if defined(_WIN32)
        std::lock_guard<std::mutex> lock(mtx_);
        statusLine_ = "GT worker unsupported on Windows";
#else
        for(;;) {
            GtInferenceRequest req;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() { return stopRequested_ || hasPendingRequest_; });
                if(stopRequested_) {
                    break;
                }
                req = std::move(pendingRequest_);
                hasPendingRequest_ = false;
            }

            if(childPid_ <= 0 && !launchChild()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                continue;
            }

            GtInferenceResult response;
            if(!sendRequest(req) || !readResponse(response)) {
                closeChild();
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    latestResult_ = GtInferenceResult{};
                    latestResult_.status = "GT worker disconnected";
                    latestResult_.workerFps = smoothedFps_;
                    statusLine_ = "GT worker disconnected";
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(mtx_);
                const auto now = std::chrono::steady_clock::now();
                if(lastResponseTime_.time_since_epoch().count() != 0) {
                    const double dt = std::chrono::duration<double>(now - lastResponseTime_).count();
                    if(dt > 1e-6) {
                        const double instantFps = 1.0 / dt;
                        smoothedFps_ = smoothedFps_ > 0.0 ? (0.8 * smoothedFps_ + 0.2 * instantFps) : instantFps;
                    }
                }
                lastResponseTime_ = now;
                response.workerFps = response.workerFps > 0.0 ? response.workerFps : smoothedFps_;
                latestResult_ = std::move(response);
                statusLine_ = latestResult_.status.empty() ? "GT worker ready" : latestResult_.status;
            }
        }

        closeChild();
#endif
    }

    mutable std::mutex      mtx_;
    std::condition_variable cv_;
    bool                    stopRequested_ = false;
    bool                    hasPendingRequest_ = false;
    GtInferenceRequest      pendingRequest_;
    GtInferenceResult       latestResult_;
    std::string             statusLine_ = "GT disabled";
    fs::path                scriptPath_;
    std::thread             workerThread_;
    double                  smoothedFps_ = 0.0;
    std::chrono::steady_clock::time_point lastResponseTime_{};
#if !defined(_WIN32)
    pid_t childPid_ = -1;
    int   childStdinFd_ = -1;
    int   childStdoutFd_ = -1;
#endif
};

class LiveEgoAprilTagWorker {
public:
    LiveEgoAprilTagWorker() = default;

    ~LiveEgoAprilTagWorker() {
        stop();
    }

    void setScriptPath(fs::path scriptPath) {
        std::lock_guard<std::mutex> lock(mtx_);
        scriptPath_ = std::move(scriptPath);
    }

    void ensureRunning() {
        std::lock_guard<std::mutex> lock(mtx_);
        if(workerThread_.joinable()) {
            return;
        }
        stopRequested_ = false;
        hasPendingRequest_ = false;
        latestResult_ = EgoAprilTagResult{};
        statusLine_ = "PICO tags worker starting";
        workerThread_ = std::thread([this]() { workerLoop(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopRequested_ = true;
            hasPendingRequest_ = false;
            cv_.notify_all();
        }
        if(workerThread_.joinable()) {
            workerThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopRequested_ = false;
            latestResult_ = EgoAprilTagResult{};
            statusLine_ = "PICO tags off";
        }
    }

    void submitLatest(EgoAprilTagRequest req) {
        ensureRunning();
        std::lock_guard<std::mutex> lock(mtx_);
        pendingRequest_ = std::move(req);
        hasPendingRequest_ = true;
        cv_.notify_one();
    }

    void setIdleStatus(const std::string &status, bool clearResult) {
        std::lock_guard<std::mutex> lock(mtx_);
        statusLine_ = status;
        if(clearResult) {
            latestResult_ = EgoAprilTagResult{};
            latestResult_.status = status;
            latestResult_.workerFps = smoothedFps_;
        }
    }

    EgoAprilTagResult latestResult() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return latestResult_;
    }

    bool hasPendingRequest() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return hasPendingRequest_;
    }

    std::string statusLine() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return statusLine_;
    }

private:
#if !defined(_WIN32)
    bool launchChild() {
        fs::path scriptPath;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            scriptPath = scriptPath_;
        }
        if(scriptPath.empty() || !fs::exists(scriptPath)) {
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "PICO tags worker script not found";
            return false;
        }

        int stdinPipe[2] = { -1, -1 };
        int stdoutPipe[2] = { -1, -1 };
        if(::pipe(stdinPipe) != 0 || ::pipe(stdoutPipe) != 0) {
            if(stdinPipe[0] >= 0) {
                ::close(stdinPipe[0]);
            }
            if(stdinPipe[1] >= 0) {
                ::close(stdinPipe[1]);
            }
            if(stdoutPipe[0] >= 0) {
                ::close(stdoutPipe[0]);
            }
            if(stdoutPipe[1] >= 0) {
                ::close(stdoutPipe[1]);
            }
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "PICO tags worker pipe creation failed";
            return false;
        }

        const pid_t pid = ::fork();
        if(pid < 0) {
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            ::close(stdoutPipe[0]);
            ::close(stdoutPipe[1]);
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "PICO tags worker fork failed";
            return false;
        }

        if(pid == 0) {
            ::dup2(stdinPipe[0], STDIN_FILENO);
            ::dup2(stdoutPipe[1], STDOUT_FILENO);
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            ::close(stdoutPipe[0]);
            ::close(stdoutPipe[1]);
            ::execlp("python3", "python3", scriptPath.string().c_str(), nullptr);
            ::execlp("python", "python", scriptPath.string().c_str(), nullptr);
            _exit(127);
        }

        ::close(stdinPipe[0]);
        ::close(stdoutPipe[1]);
        childPid_ = pid;
        childStdinFd_ = stdinPipe[1];
        childStdoutFd_ = stdoutPipe[0];
        {
            std::lock_guard<std::mutex> lock(mtx_);
            statusLine_ = "PICO tags worker ready";
        }
        return true;
    }

    void closeChild() {
        if(childStdinFd_ >= 0) {
            ::close(childStdinFd_);
            childStdinFd_ = -1;
        }
        if(childStdoutFd_ >= 0) {
            ::close(childStdoutFd_);
            childStdoutFd_ = -1;
        }
        if(childPid_ > 0) {
            ::kill(childPid_, SIGTERM);
            int status = 0;
            ::waitpid(childPid_, &status, 0);
            childPid_ = -1;
        }
    }

    std::string buildRequestJson(const EgoAprilTagRequest &req, uint64_t &payloadBytesOut) const {
        payloadBytesOut = 0;
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(9);
        oss << "{\"type\":\"frame\",\"frame_id\":" << req.frameId
            << ",\"ego_video_frame_index\":" << req.egoVideoFrameIndex
            << ",\"ego_video_path\":" << jsonStringLocal(req.egoVideoPath.string())
            << ",\"ego_camera_params_path\":" << jsonStringLocal(req.egoCameraParamsPath.string())
            << ",\"tag_family\":\"tag36h11\",\"tag_size_m\":0.096,\"cameras\":[";
        uint64_t payloadOffset = 0;
        for(size_t i = 0; i < req.cameras.size(); ++i) {
            const auto &cam = req.cameras[i];
            const uint64_t rgbBytes = static_cast<uint64_t>(matByteSize(cam.rgbBgr));
            if(i != 0) {
                oss << ",";
            }
            oss << "{\"camera_id\":" << jsonStringLocal(cam.camIndex)
                << ",\"device_index\":" << cam.deviceIndex
                << ",\"rgb_width\":" << cam.rgbBgr.cols
                << ",\"rgb_height\":" << cam.rgbBgr.rows
                << ",\"rgb_scale_x\":" << cam.rgbScaleX
                << ",\"rgb_scale_y\":" << cam.rgbScaleY
                << ",\"rgb_offset\":" << payloadOffset
                << ",\"rgb_size\":" << rgbBytes
                << ",\"intrinsic\":{\"fx\":" << cam.rgbIntrinsic.fx
                << ",\"fy\":" << cam.rgbIntrinsic.fy
                << ",\"cx\":" << cam.rgbIntrinsic.cx
                << ",\"cy\":" << cam.rgbIntrinsic.cy
                << "},\"distortion\":{\"k1\":" << cam.rgbDistortion.k1
                << ",\"k2\":" << cam.rgbDistortion.k2
                << ",\"k3\":" << cam.rgbDistortion.k3
                << ",\"k4\":" << cam.rgbDistortion.k4
                << ",\"k5\":" << cam.rgbDistortion.k5
                << ",\"k6\":" << cam.rgbDistortion.k6
                << ",\"p1\":" << cam.rgbDistortion.p1
                << ",\"p2\":" << cam.rgbDistortion.p2
                << "},\"Rwc\":[" << cam.Rwc(0, 0) << "," << cam.Rwc(0, 1) << "," << cam.Rwc(0, 2)
                << "," << cam.Rwc(1, 0) << "," << cam.Rwc(1, 1) << "," << cam.Rwc(1, 2)
                << "," << cam.Rwc(2, 0) << "," << cam.Rwc(2, 1) << "," << cam.Rwc(2, 2)
                << "],\"twc\":[" << cam.twc[0] << "," << cam.twc[1] << "," << cam.twc[2] << "]}";
            payloadOffset += rgbBytes;
        }
        oss << "]}";
        payloadBytesOut = payloadOffset;
        return oss.str();
    }

    bool sendRequest(const EgoAprilTagRequest &req) {
        if(childStdinFd_ < 0) {
            return false;
        }
        uint64_t payloadBytes = 0;
        const std::string json = buildRequestJson(req, payloadBytes);
        uint8_t header[12];
        putU32Le(header, static_cast<uint32_t>(json.size()));
        putU64Le(header + 4, payloadBytes);
        if(!writeAllFd(childStdinFd_, header, sizeof(header))) {
            return false;
        }
        if(!json.empty() && !writeAllFd(childStdinFd_, json.data(), json.size())) {
            return false;
        }
        for(const auto &cam : req.cameras) {
            if(!cam.rgbBgr.empty() && !writeAllFd(childStdinFd_, cam.rgbBgr.data, matByteSize(cam.rgbBgr))) {
                return false;
            }
        }
        return true;
    }

    bool readResponse(EgoAprilTagResult &out) {
        if(childStdoutFd_ < 0) {
            return false;
        }
        uint8_t header[12];
        if(!readAllFd(childStdoutFd_, header, sizeof(header))) {
            return false;
        }
        const uint32_t jsonBytes = getU32Le(header);
        const uint64_t payloadBytes = getU64Le(header + 4);
        if(payloadBytes != 0) {
            return false;
        }
        std::string json(jsonBytes, '\0');
        if(jsonBytes > 0 && !readAllFd(childStdoutFd_, json.data(), jsonBytes)) {
            return false;
        }

        cJSON *root = cJSON_Parse(json.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            return false;
        }

        out = EgoAprilTagResult{};
        auto *okItem = cJSON_GetObjectItemCaseSensitive(root, "ok");
        out.valid = okItem && cJSON_IsBool(okItem) ? (okItem->valueint != 0) : false;
        auto *frameIdItem = cJSON_GetObjectItemCaseSensitive(root, "frame_id");
        if(frameIdItem && cJSON_IsNumber(frameIdItem)) {
            out.frameId = static_cast<uint64_t>(frameIdItem->valuedouble);
        }
        auto *egoFrameItem = cJSON_GetObjectItemCaseSensitive(root, "ego_video_frame_index");
        if(egoFrameItem && cJSON_IsNumber(egoFrameItem)) {
            out.egoVideoFrameIndex = egoFrameItem->valueint;
        }
        auto *tagCountItem = cJSON_GetObjectItemCaseSensitive(root, "tag_count");
        if(tagCountItem && cJSON_IsNumber(tagCountItem)) {
            out.tagCount = tagCountItem->valueint;
        }
        auto *refCountItem = cJSON_GetObjectItemCaseSensitive(root, "reference_tag_count");
        if(refCountItem && cJSON_IsNumber(refCountItem)) {
            out.referenceTagCount = refCountItem->valueint;
        }
        auto *rmseItem = cJSON_GetObjectItemCaseSensitive(root, "rmse_px");
        if(rmseItem && cJSON_IsNumber(rmseItem)) {
            out.rmsePx = rmseItem->valuedouble;
        }
        auto *fpsItem = cJSON_GetObjectItemCaseSensitive(root, "worker_fps");
        if(fpsItem && cJSON_IsNumber(fpsItem)) {
            out.workerFps = fpsItem->valuedouble;
        }
        auto *statusItem = cJSON_GetObjectItemCaseSensitive(root, "status");
        if(statusItem && cJSON_IsString(statusItem) && statusItem->valuestring) {
            out.status = statusItem->valuestring;
        }

        auto clampByte = [](double v) -> uint8_t {
            if(!std::isfinite(v)) {
                return 255;
            }
            return static_cast<uint8_t>(std::max(0.0, std::min(255.0, std::round(v))));
        };

        auto *linesItem = cJSON_GetObjectItemCaseSensitive(root, "lines");
        if(linesItem && cJSON_IsArray(linesItem)) {
            const int lineCount = cJSON_GetArraySize(linesItem);
            out.lines.reserve(static_cast<size_t>(lineCount));
            for(int i = 0; i < lineCount; ++i) {
                auto *arr = cJSON_GetArrayItem(linesItem, i);
                if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) < 9) {
                    continue;
                }
                double v[9] = {};
                bool ok = true;
                for(int j = 0; j < 9; ++j) {
                    auto *item = cJSON_GetArrayItem(arr, j);
                    if(!item || !cJSON_IsNumber(item)) {
                        ok = false;
                        break;
                    }
                    v[j] = item->valuedouble;
                }
                if(!ok) {
                    continue;
                }
                EgoAprilTagLine line;
                line.p0 = cv::Vec3f(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]));
                line.p1 = cv::Vec3f(static_cast<float>(v[3]), static_cast<float>(v[4]), static_cast<float>(v[5]));
                line.color = cv::Vec3b(clampByte(v[6]), clampByte(v[7]), clampByte(v[8]));
                out.lines.push_back(line);
            }
        }

        cJSON_Delete(root);
        return true;
    }
#endif

    void workerLoop() {
#if defined(_WIN32)
        std::lock_guard<std::mutex> lock(mtx_);
        statusLine_ = "PICO tags worker unsupported on Windows";
#else
        for(;;) {
            EgoAprilTagRequest req;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() { return stopRequested_ || hasPendingRequest_; });
                if(stopRequested_) {
                    break;
                }
                req = std::move(pendingRequest_);
                hasPendingRequest_ = false;
            }

            if(childPid_ <= 0 && !launchChild()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                continue;
            }

            EgoAprilTagResult response;
            if(!sendRequest(req) || !readResponse(response)) {
                closeChild();
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    latestResult_ = EgoAprilTagResult{};
                    latestResult_.status = "PICO tags worker disconnected";
                    latestResult_.workerFps = smoothedFps_;
                    statusLine_ = "PICO tags worker disconnected";
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(mtx_);
                const auto now = std::chrono::steady_clock::now();
                if(lastResponseTime_.time_since_epoch().count() != 0) {
                    const double dt = std::chrono::duration<double>(now - lastResponseTime_).count();
                    if(dt > 1e-6) {
                        const double instantFps = 1.0 / dt;
                        smoothedFps_ = smoothedFps_ > 0.0 ? (0.8 * smoothedFps_ + 0.2 * instantFps) : instantFps;
                    }
                }
                lastResponseTime_ = now;
                response.workerFps = response.workerFps > 0.0 ? response.workerFps : smoothedFps_;
                latestResult_ = std::move(response);
                statusLine_ = latestResult_.status.empty() ? "PICO tags worker ready" : latestResult_.status;
            }
        }

        closeChild();
#endif
    }

    mutable std::mutex      mtx_;
    std::condition_variable cv_;
    bool                    stopRequested_ = false;
    bool                    hasPendingRequest_ = false;
    EgoAprilTagRequest      pendingRequest_;
    EgoAprilTagResult       latestResult_;
    std::string             statusLine_ = "PICO tags off";
    fs::path                scriptPath_;
    std::thread             workerThread_;
    double                  smoothedFps_ = 0.0;
    std::chrono::steady_clock::time_point lastResponseTime_{};
#if !defined(_WIN32)
    pid_t childPid_ = -1;
    int   childStdinFd_ = -1;
    int   childStdoutFd_ = -1;
#endif
};

class LivePicoRgbFrameSource {
public:
    LivePicoRgbFrameSource() = default;

    ~LivePicoRgbFrameSource() {
        close();
    }

    void close() {
        stopRequested_.store(true);
#if !defined(_WIN32)
        closeDecoderPipesOnly();
#endif
        if(workerThread_.joinable()) {
            workerThread_.join();
        }
#if !defined(_WIN32)
        closeDecoder();
#endif
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopRequested_.store(false);
            recorder_ = nullptr;
            latestFrame_.release();
            latestFrameIndex_ = -1;
            statusLine_ = "PICO RGB off";
        }
    }

    void start(EgoRecorder *recorder) {
        if(!recorder) {
            publishStatus("PICO RGB recorder missing");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if(workerThread_.joinable() && recorder_ == recorder && workerRunning_.load() && !stopRequested_.load()) {
                return;
            }
        }

        close();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopRequested_.store(false);
            recorder_ = recorder;
            latestFrame_.release();
            latestFrameIndex_ = -1;
            statusLine_ = "PICO RGB live decoder starting";
        }
        workerThread_ = std::thread([this, recorder]() { workerLoop(recorder); });
    }

    bool latest(cv::Mat &out, int *frameIndex, std::string *status, int sinceFrameIndex = -1) const {
        std::lock_guard<std::mutex> lock(mtx_);
        if(frameIndex) {
            *frameIndex = latestFrameIndex_;
        }
        if(status) {
            *status = statusLine_;
        }
        if(latestFrame_.empty() || latestFrameIndex_ == sinceFrameIndex) {
            out.release();
            return false;
        }
        out = latestFrame_.clone();
        return true;
    }

private:
    void publishStatus(const std::string &status) {
        std::lock_guard<std::mutex> lock(mtx_);
        statusLine_ = status;
    }

    void publishFrame(const cv::Mat &frame) {
        if(frame.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        latestFrame_ = frame;
        latestFrameIndex_++;
        statusLine_ = "PICO RGB live frame " + std::to_string(latestFrameIndex_);
    }

    void workerLoop(EgoRecorder *recorder) {
        struct RunningGuard {
            std::atomic_bool &running;
            ~RunningGuard() {
                running.store(false);
            }
        };
        workerRunning_.store(true);
        RunningGuard guard{ workerRunning_ };
#if defined(_WIN32)
        (void)recorder;
        publishStatus("PICO RGB live preview is unsupported on Windows");
#else
        if(!launchDecoder()) {
            publishStatus("PICO RGB ffmpeg start failed");
            return;
        }
        recorder->clearHevcSamples(true);
        publishStatus("PICO RGB waiting for live H265");
        auto restartDecoder = [&]() {
            publishStatus("PICO RGB decoder restarting");
            closeDecoder();
            if(stopRequested_.load()) {
                return false;
            }
            if(!launchDecoder()) {
                publishStatus("PICO RGB ffmpeg start failed");
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                return false;
            }
            recorder->clearHevcSamples(true);
            publishStatus("PICO RGB waiting for live H265");
            return true;
        };

        while(!stopRequested_.load()) {
            EgoHevcSample sample;
            if(!recorder->popHevcSample(sample, std::chrono::milliseconds(20))) {
                continue;
            }
            if(sample.payload.empty()) {
                continue;
            }
            if(sample.codecConfig) {
                if(!writeSample(sample) && !restartDecoder()) {
                    break;
                }
                continue;
            }

            int droppedSamples = 0;
            bool restartFailed = false;
            EgoHevcSample newer;
            while(recorder->popHevcSample(newer, std::chrono::milliseconds(0))) {
                if(newer.payload.empty()) {
                    continue;
                }
                if(newer.codecConfig) {
                    if(!writeSample(newer) && !restartDecoder()) {
                        restartFailed = true;
                        break;
                    }
                    continue;
                }
                sample = std::move(newer);
                droppedSamples++;
            }
            if(restartFailed) {
                if(stopRequested_.load()) {
                    break;
                }
                continue;
            }
            if(stopRequested_.load()) {
                break;
            }
            if(droppedSamples > 0 && latestFrameIndex() < 0) {
                publishStatus("PICO RGB dropping delayed startup frames");
            }
            if(!writeSample(sample) && !restartDecoder()) {
                break;
            }
            if(!sample.codecConfig && latestFrameIndex() < 0) {
                publishStatus("PICO RGB decoding live stream");
            }
        }

        closeDecoder();
#endif
    }

    int latestFrameIndex() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return latestFrameIndex_;
    }

#if !defined(_WIN32)
    bool launchDecoder() {
        closeDecoder();

        std::signal(SIGPIPE, SIG_IGN);
        int stdinPipe[2] = { -1, -1 };
        int stdoutPipe[2] = { -1, -1 };
        if(::pipe(stdinPipe) != 0) {
            return false;
        }
        if(::pipe(stdoutPipe) != 0) {
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            return false;
        }

        const pid_t pid = ::fork();
        if(pid < 0) {
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            ::close(stdoutPipe[0]);
            ::close(stdoutPipe[1]);
            return false;
        }
        if(pid == 0) {
            ::dup2(stdinPipe[0], STDIN_FILENO);
            ::dup2(stdoutPipe[1], STDOUT_FILENO);
            const int devNull = ::open("/dev/null", O_WRONLY);
            if(devNull >= 0) {
                ::dup2(devNull, STDERR_FILENO);
                ::close(devNull);
            }
            ::close(stdinPipe[0]);
            ::close(stdinPipe[1]);
            ::close(stdoutPipe[0]);
            ::close(stdoutPipe[1]);
            ::execlp("ffmpeg",
                     "ffmpeg",
                     "-hide_banner",
                     "-loglevel",
                     "error",
                     "-probesize",
                     "32",
                     "-analyzeduration",
                     "0",
                     "-fflags",
                     "nobuffer",
                     "-flags",
                     "low_delay",
                     "-f",
                     "hevc",
                     "-i",
                     "pipe:0",
                     "-an",
                     "-pix_fmt",
                     "rgb24",
                     "-f",
                     "image2pipe",
                     "-flush_packets",
                     "1",
                     "-vcodec",
                     "ppm",
                     "pipe:1",
                     static_cast<char *>(nullptr));
            _exit(127);
        }

        ::close(stdinPipe[0]);
        ::close(stdoutPipe[1]);
        {
            std::lock_guard<std::mutex> lock(processMtx_);
            decoderPid_ = pid;
            decoderStdinFd_ = stdinPipe[1];
            decoderStdoutFd_ = stdoutPipe[0];
        }
        readerThread_ = std::thread([this]() { readerLoop(); });
        return true;
    }

    void closeDecoderPipesOnly() {
        int stdinFd = -1;
        int stdoutFd = -1;
        pid_t pid = -1;
        {
            std::lock_guard<std::mutex> lock(processMtx_);
            stdinFd = decoderStdinFd_;
            stdoutFd = decoderStdoutFd_;
            pid = decoderPid_;
            decoderStdinFd_ = -1;
            decoderStdoutFd_ = -1;
        }
        if(stdinFd >= 0) {
            ::close(stdinFd);
        }
        if(stdoutFd >= 0) {
            ::close(stdoutFd);
        }
        if(pid > 0) {
            ::kill(pid, SIGTERM);
        }
    }

    void closeDecoder() {
        int stdinFd = -1;
        int stdoutFd = -1;
        pid_t pid = -1;
        {
            std::lock_guard<std::mutex> lock(processMtx_);
            stdinFd = decoderStdinFd_;
            stdoutFd = decoderStdoutFd_;
            pid = decoderPid_;
            decoderStdinFd_ = -1;
            decoderStdoutFd_ = -1;
            decoderPid_ = -1;
        }
        if(stdinFd >= 0) {
            ::close(stdinFd);
        }
        if(pid > 0) {
            ::kill(pid, SIGTERM);
        }
        if(stdoutFd >= 0) {
            ::close(stdoutFd);
        }
        if(readerThread_.joinable()) {
            readerThread_.join();
        }
        if(pid > 0) {
            int status = 0;
            (void)::waitpid(pid, &status, 0);
        }
    }

    bool writeSample(const EgoHevcSample &sample) {
        int fd = -1;
        {
            std::lock_guard<std::mutex> lock(processMtx_);
            fd = decoderStdinFd_;
        }
        return fd >= 0 && writeAllFd(fd, sample.payload.data(), sample.payload.size());
    }

    static bool nextPpmToken(const std::vector<uint8_t> &buffer, size_t &pos, std::string &token) {
        token.clear();
        for(;;) {
            while(pos < buffer.size() && std::isspace(static_cast<unsigned char>(buffer[pos]))) {
                ++pos;
            }
            if(pos < buffer.size() && buffer[pos] == '#') {
                while(pos < buffer.size() && buffer[pos] != '\n') {
                    ++pos;
                }
                continue;
            }
            break;
        }
        if(pos >= buffer.size()) {
            return false;
        }
        const size_t start = pos;
        while(pos < buffer.size() && !std::isspace(static_cast<unsigned char>(buffer[pos]))) {
            ++pos;
        }
        if(pos == buffer.size()) {
            return false;
        }
        token.assign(reinterpret_cast<const char *>(buffer.data() + start), pos - start);
        return !token.empty();
    }

    static bool tryPopPpmFrame(std::vector<uint8_t> &buffer, cv::Mat &out) {
        out.release();
        if(buffer.size() < 3) {
            return false;
        }
        size_t start = std::string::npos;
        for(size_t i = 0; i + 1 < buffer.size(); ++i) {
            if(buffer[i] == 'P' && buffer[i + 1] == '6') {
                start = i;
                break;
            }
        }
        if(start == std::string::npos) {
            buffer.clear();
            return false;
        }
        if(start > 0) {
            buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(start));
        }

        size_t pos = 2;
        std::string widthToken;
        std::string heightToken;
        std::string maxToken;
        if(!nextPpmToken(buffer, pos, widthToken) || !nextPpmToken(buffer, pos, heightToken) || !nextPpmToken(buffer, pos, maxToken)) {
            return false;
        }
        if(pos >= buffer.size()) {
            return false;
        }
        const uint8_t separator = buffer[pos];
        ++pos;
        if(separator == '\r' && pos < buffer.size() && buffer[pos] == '\n') {
            ++pos;
        }

        int width = 0;
        int height = 0;
        int maxValue = 0;
        try {
            width = std::stoi(widthToken);
            height = std::stoi(heightToken);
            maxValue = std::stoi(maxToken);
        }
        catch(...) {
            buffer.erase(buffer.begin());
            return false;
        }
        if(width <= 0 || height <= 0 || width > 8192 || height > 8192 || maxValue != 255) {
            buffer.erase(buffer.begin());
            return false;
        }
        const size_t bytesNeeded = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
        if(buffer.size() - pos < bytesNeeded) {
            return false;
        }
        cv::Mat rgb(height, width, CV_8UC3, buffer.data() + pos);
        cv::cvtColor(rgb, out, cv::COLOR_RGB2BGR);
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(pos + bytesNeeded));
        return !out.empty();
    }

    void readerLoop() {
        int fd = -1;
        {
            std::lock_guard<std::mutex> lock(processMtx_);
            fd = decoderStdoutFd_;
        }
        if(fd < 0) {
            return;
        }

        std::vector<uint8_t> buffer;
        buffer.reserve(2 * 1024 * 1024);
        std::array<uint8_t, 32768> chunk{};
        while(!stopRequested_.load()) {
            const ssize_t n = ::read(fd, chunk.data(), chunk.size());
            if(n < 0) {
                if(errno == EINTR) {
                    continue;
                }
                break;
            }
            if(n == 0) {
                break;
            }
            buffer.insert(buffer.end(), chunk.begin(), chunk.begin() + n);
            cv::Mat frame;
            while(tryPopPpmFrame(buffer, frame)) {
                publishFrame(frame);
            }
            if(buffer.size() > 12 * 1024 * 1024) {
                buffer.clear();
                publishStatus("PICO RGB decoder buffer reset");
            }
        }
    }
#endif

    mutable std::mutex mtx_;
    std::thread workerThread_;
    std::atomic_bool stopRequested_{ false };
    std::atomic_bool workerRunning_{ false };
    EgoRecorder *recorder_ = nullptr;
    cv::Mat latestFrame_;
    int latestFrameIndex_ = -1;
    std::string statusLine_ = "PICO RGB off";
#if !defined(_WIN32)
    std::mutex processMtx_;
    std::thread readerThread_;
    pid_t decoderPid_ = -1;
    int decoderStdinFd_ = -1;
    int decoderStdoutFd_ = -1;
#endif
};

struct InteractiveViewState {
    int width = 1600;
    int height = 900;
    cv::Rect pcRect{0, 0, 1600, 900};
    cv::Point cursor{0, 0};

    bool rotating = false;
    bool panning = false;
    cv::Point lastMouse{0, 0};

    float yawRad = 0.0f;
    float pitchRad = 0.0f;
    float distance = 1.5f;
    cv::Vec3f target{0.0f, 0.0f, 1.0f};

    void resetView() {
        rotating = false;
        panning = false;
        yawRad = 0.0f;
        pitchRad = 0.0f;
        distance = 1.5f;
        target = cv::Vec3f(0.0f, 0.0f, 1.0f);
        cursor = cv::Point(0, 0);
        lastMouse = cv::Point(0, 0);
    }
};

static cv::Vec3f normalizeVec3(const cv::Vec3f &v) {
    const float n = std::sqrt(v.dot(v));
    if(n <= 1e-8f) {
        return cv::Vec3f(0, 0, 0);
    }
    return v * (1.0f / n);
}

static cv::Vec3f crossVec3(const cv::Vec3f &a, const cv::Vec3f &b) {
    return cv::Vec3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

static void computeCameraBasis(const InteractiveViewState &s, cv::Vec3f &right, cv::Vec3f &up, cv::Vec3f &forward, cv::Vec3f &camPos) {
    const float cy = std::cos(s.yawRad);
    const float sy = std::sin(s.yawRad);
    const float cp = std::cos(s.pitchRad);
    const float sp = std::sin(s.pitchRad);

    forward = normalizeVec3(cv::Vec3f(sy * cp, -sp, cy * cp));
    const cv::Vec3f worldUp(0.0f, -1.0f, 0.0f);
    right = normalizeVec3(crossVec3(forward,worldUp));
    up = crossVec3(right,forward);
    camPos = s.target - forward * s.distance;
}

static void mouseCallbackPointCloud(int event, int x, int y, int flags, void *userdata) {
    ivizSetStage("pc_mouse_enter");
    auto *s = reinterpret_cast<InteractiveViewState *>(userdata);
    if(!s) {
        return;
    }

    if(event == cv::EVENT_LBUTTONUP) {
        ivizSetStage("pc_mouse_lup");
        s->rotating = false;
    }
    if(event == cv::EVENT_RBUTTONUP) {
        ivizSetStage("pc_mouse_rup");
        s->panning = false;
    }

    const bool inside = s->pcRect.contains(cv::Point(x, y));
    if(!inside) {
        if(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
            return;
        }
        if(event == cv::EVENT_MOUSEMOVE && !(s->rotating || s->panning)) {
            return;
        }
    }

    if(event == cv::EVENT_LBUTTONDOWN) {
        ivizSetStage("pc_mouse_ldown");
        s->rotating = true;
        s->panning = false;
        s->lastMouse = cv::Point(x, y);
        return;
    }
    if(event == cv::EVENT_RBUTTONDOWN) {
        ivizSetStage("pc_mouse_rdown");
        s->panning = true;
        s->rotating = false;
        s->lastMouse = cv::Point(x, y);
        return;
    }
    if(event == cv::EVENT_MOUSEWHEEL && inside) {
        ivizSetStage("pc_mouse_wheel");
        const int delta = cv::getMouseWheelDelta(flags);
        if(delta != 0) {
            const float scale = delta > 0 ? 0.90f : 1.10f;
            s->distance = std::min(20.0f, std::max(0.2f, s->distance * scale));
        }
        return;
    }
    if(event == cv::EVENT_MOUSEMOVE) {
        ivizSetStage("pc_mouse_move");
        const int dx = x - s->lastMouse.x;
        const int dy = y - s->lastMouse.y;
        s->lastMouse = cv::Point(x, y);
        if(s->rotating) {
            ivizSetStage("pc_mouse_rotate");
            s->yawRad += static_cast<float>(dx) * 0.005f;
            s->pitchRad += static_cast<float>(dy) * 0.005f;
            s->pitchRad = std::min(1.55f, std::max(-1.55f, s->pitchRad));
            return;
        }
        if(s->panning) {
            ivizSetStage("pc_mouse_pan");
            cv::Vec3f right, up, forward, camPos;
            computeCameraBasis(*s, right, up, forward, camPos);
            const float scale = s->distance * 0.001f;
            s->target -= right * (static_cast<float>(dx) * scale);
            s->target += up * (static_cast<float>(dy) * scale);
            return;
        }
    }
}

struct MainMouseContext {
    InteractiveViewState *view = nullptr;
    CvMouseState         *ui   = nullptr;
};

static void mouseCallbackMain(int event, int x, int y, int flags, void *userdata) {
    ivizSetStage("main_mouse_enter");
    auto *ctx = reinterpret_cast<MainMouseContext *>(userdata);
    if(!ctx) {
        return;
    }
    if(ctx->ui) {
        ctx->ui->x = x;
        ctx->ui->y = y;
        if(event == cv::EVENT_LBUTTONDOWN) {
            ctx->ui->clicked = true;
            ctx->ui->clickX = x;
            ctx->ui->clickY = y;
        }
        else if(event == cv::EVENT_MOUSEWHEEL) {
            ctx->ui->wheelDelta += cv::getMouseWheelDelta(flags);
        }
    }
    if(ctx->view) {
        ctx->view->cursor = cv::Point(x, y);
        mouseCallbackPointCloud(event, x, y, flags, ctx->view);
    }
}

static FisheyeModuleConfig buildAutoFisheyeConfigFromDevices(const std::vector<FisheyeDeviceInfo> &devices) {
    FisheyeModuleConfig cfg;
    cfg.enabled = true;
    cfg.targetFps = 60;
    cfg.maxBufferedSets = 512;
    cfg.cameras.clear();
    cfg.cameras.reserve(devices.size());
    for(size_t i = 0; i < devices.size(); ++i) {
        FisheyeCameraConfig camera;
        camera.cameraId = "fisheye_" + std::to_string(i);
        camera.devicePath = !devices[i].stablePath.empty() ? devices[i].stablePath : devices[i].devicePath;
        camera.deviceIndex = -1;
        camera.width = 1280;
        camera.height = 720;
        camera.fps = 60;
        camera.preferMjpeg = devices[i].supportsMjpeg;
        cfg.cameras.push_back(std::move(camera));
    }
    return cfg;
}

static const std::vector<std::string> &preferredFisheyeCameraLabels() {
    static const std::vector<std::string> labels = { "1", "5" };
    return labels;
}

static std::string readFileAllLocal(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

static std::string jsonEscapeLocal(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for(char ch : s) {
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

static std::string jsonStringLocal(const std::string &s) {
    return "\"" + jsonEscapeLocal(s) + "\"";
}

static bool readTextFileLocal(const fs::path &p, std::string &out) {
    std::ifstream ifs(p, std::ios::in | std::ios::binary);
    if(!ifs) {
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

static bool writeTextFileLocal(const fs::path &p, const std::string &content) {
    std::ofstream ofs(p, std::ios::out | std::ios::binary);
    if(!ofs) {
        return false;
    }
    ofs << content;
    return static_cast<bool>(ofs);
}

static std::string shellQuoteLocal(const std::string &s) {
    std::string out = "'";
    for(char c : s) {
        if(c == '\'') {
            out += "'\\''";
        }
        else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static int runCommandCaptureLocal(const std::string &command, std::string &output) {
    output.clear();
#if defined(_WIN32)
    FILE *pipe = _popen(command.c_str(), "r");
#else
    FILE *pipe = popen(command.c_str(), "r");
#endif
    if(!pipe) {
        return -1;
    }
    char buf[512];
    while(true) {
        const size_t n = fread(buf, 1, sizeof(buf), pipe);
        if(n > 0) {
            output.append(buf, n);
        }
        if(n < sizeof(buf)) {
            break;
        }
    }
#if defined(_WIN32)
    const int status = _pclose(pipe);
    return status == 0 ? 0 : -1;
#else
    const int status = pclose(pipe);
    if(status == -1) {
        return -1;
    }
    if(WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return status == 0 ? 0 : -1;
#endif
}

static std::string sanitizePathComponentLocal(std::string value) {
    value = trimString(std::move(value));
    if(value.empty()) {
        return "item";
    }
    std::string out;
    out.reserve(value.size());
    for(char ch : value) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if(std::isalnum(c) || ch == '-' || ch == '_' || ch == '.') {
            out.push_back(ch);
        }
        else {
            out.push_back('_');
        }
    }
    return out.empty() ? "item" : out;
}

static std::string formatFrameIndexLocal(size_t i) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << i;
    return oss.str();
}

static uint64_t frameTimestampUsLocal(const std::shared_ptr<ob::Frame> &frame) {
    if(!frame) {
        return 0;
    }
    uint64_t ts = 0;
    try {
        ts = frame->globalTimeStampUs();
    }
    catch(...) {
    }
    if(ts == 0) {
        try {
            ts = frame->timeStampUs();
        }
        catch(...) {
        }
    }
    return ts;
}

static void jsonAddNumberLocal(cJSON *obj, const char *key, double v) {
    cJSON_AddItemToObject(obj, key, cJSON_CreateNumber(v));
}

static void jsonAddStringLocal(cJSON *obj, const char *key, const std::string &v) {
    cJSON_AddItemToObject(obj, key, cJSON_CreateString(v.c_str()));
}

static void jsonAddIntrinsicLocal(cJSON *obj, const OBCameraIntrinsic &in) {
    jsonAddNumberLocal(obj, "fx", in.fx);
    jsonAddNumberLocal(obj, "fy", in.fy);
    jsonAddNumberLocal(obj, "cx", in.cx);
    jsonAddNumberLocal(obj, "cy", in.cy);
    jsonAddNumberLocal(obj, "width", in.width);
    jsonAddNumberLocal(obj, "height", in.height);
}

static void jsonAddDistortionLocal(cJSON *obj, const OBCameraDistortion &d) {
    jsonAddNumberLocal(obj, "k1", d.k1);
    jsonAddNumberLocal(obj, "k2", d.k2);
    jsonAddNumberLocal(obj, "k3", d.k3);
    jsonAddNumberLocal(obj, "k4", d.k4);
    jsonAddNumberLocal(obj, "k5", d.k5);
    jsonAddNumberLocal(obj, "k6", d.k6);
    jsonAddNumberLocal(obj, "p1", d.p1);
    jsonAddNumberLocal(obj, "p2", d.p2);
    jsonAddNumberLocal(obj, "model", static_cast<int>(d.model));
}

static void jsonAddExtrinsicLocal(cJSON *obj, const float rot[9], const float trans[3]) {
    cJSON *rotArr = cJSON_CreateArray();
    for(int r = 0; r < 3; ++r) {
        cJSON *row = cJSON_CreateArray();
        for(int c = 0; c < 3; ++c) {
            cJSON_AddItemToArray(row, cJSON_CreateNumber(rot[r * 3 + c]));
        }
        cJSON_AddItemToArray(rotArr, row);
    }
    cJSON_AddItemToObject(obj, "rotation", rotArr);

    cJSON *tArr = cJSON_CreateArray();
    for(int i = 0; i < 3; ++i) {
        cJSON_AddItemToArray(tArr, cJSON_CreateNumber(trans[i]));
    }
    cJSON_AddItemToObject(obj, "translation", tArr);
}

static bool parseVec3(cJSON *arr, cv::Vec3f &out) {
    if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
        return false;
    }
    const auto *a0 = cJSON_GetArrayItem(arr, 0);
    const auto *a1 = cJSON_GetArrayItem(arr, 1);
    const auto *a2 = cJSON_GetArrayItem(arr, 2);
    if(!a0 || !a1 || !a2 || !cJSON_IsNumber(a0) || !cJSON_IsNumber(a1) || !cJSON_IsNumber(a2)) {
        return false;
    }
    out = cv::Vec3f(static_cast<float>(a0->valuedouble), static_cast<float>(a1->valuedouble), static_cast<float>(a2->valuedouble));
    return true;
}

static bool parseMat3(cJSON *arr, cv::Matx33f &out) {
    if(!arr || !cJSON_IsArray(arr)) {
        return false;
    }
    const int n = cJSON_GetArraySize(arr);
    if(n == 9) {
        float v[9];
        for(int i = 0; i < 9; i++) {
            auto *it = cJSON_GetArrayItem(arr, i);
            if(!it || !cJSON_IsNumber(it)) {
                return false;
            }
            v[i] = static_cast<float>(it->valuedouble);
        }
        out = cv::Matx33f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
        return true;
    }
    if(n == 3) {
        auto *r0 = cJSON_GetArrayItem(arr, 0);
        auto *r1 = cJSON_GetArrayItem(arr, 1);
        auto *r2 = cJSON_GetArrayItem(arr, 2);
        if(r0 && r1 && r2 && cJSON_IsArray(r0) && cJSON_IsArray(r1) && cJSON_IsArray(r2) && cJSON_GetArraySize(r0) == 3 && cJSON_GetArraySize(r1) == 3
           && cJSON_GetArraySize(r2) == 3) {
            float v[9];
            cJSON *rows[3] = { r0, r1, r2 };
            for(int y = 0; y < 3; y++) {
                for(int x = 0; x < 3; x++) {
                    auto *it = cJSON_GetArrayItem(rows[y], x);
                    if(!it || !cJSON_IsNumber(it)) {
                        return false;
                    }
                    v[y * 3 + x] = static_cast<float>(it->valuedouble);
                }
            }
            out = cv::Matx33f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
            return true;
        }
    }
    return false;
}

static bool parseExtrinsicObject(cJSON *obj, cv::Matx33f &R, cv::Vec3f &t) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    auto *rotArr = cJSON_GetObjectItemCaseSensitive(obj, "rotation");
    auto *tArr = cJSON_GetObjectItemCaseSensitive(obj, "translation");
    return parseMat3(rotArr, R) && parseVec3(tArr, t);
}

static bool parseExtrinsicHealthPose(cJSON *obj, ExtrinsicHealthPose &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    cv::Matx33f R = cv::Matx33f::eye();
    cv::Vec3f t(0.0f, 0.0f, 0.0f);
    if(!parseExtrinsicObject(obj, R, t)) {
        return false;
    }
    out.R = R;
    out.t = t;
    out.valid = true;
    return true;
}

static void invertRigid(const cv::Matx33f &R, const cv::Vec3f &t, cv::Matx33f &Rinv, cv::Vec3f &tinv) {
    Rinv = R.t();
    tinv = -(Rinv * t);
}

static cv::Vec3f translationMmToM(const cv::Vec3f &tMm) {
    return cv::Vec3f(tMm[0] * 0.001f, tMm[1] * 0.001f, tMm[2] * 0.001f);
}

static float bilinearSampleNormalized(const std::shared_ptr<ob::IRFrame> &ir, float x, float y) {
    if(!ir) {
        return 0.0f;
    }
    const int w = static_cast<int>(ir->getWidth());
    const int h = static_cast<int>(ir->getHeight());
    if(w <= 1 || h <= 1) {
        return 0.0f;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    if(x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) {
        return 0.0f;
    }

    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);

    const uint8_t *raw = reinterpret_cast<const uint8_t *>(ir->data());
    if(!raw) {
        return 0.0f;
    }
    const auto dataSize = static_cast<size_t>(ir->dataSize());
    const size_t strideBytes = h > 0 ? (dataSize / static_cast<size_t>(h)) : 0;
    if(strideBytes == 0) {
        return 0.0f;
    }

    const auto fmt = ir->getFormat();
    float maxVal = 255.0f;
    if(fmt != OB_FORMAT_Y8) {
        uint8_t bitSize = 0;
        try {
            bitSize = ir->getPixelAvailableBitSize();
        }
        catch(...) {
        }
        if(bitSize > 0 && bitSize < 16) {
            maxVal = static_cast<float>((1u << bitSize) - 1u);
        }
        else {
            maxVal = 65535.0f;
        }
    }
    if(maxVal <= 0.0f) {
        return 0.0f;
    }

    auto sampleU8 = [&](int sx, int sy) -> float {
        const size_t off = static_cast<size_t>(sy) * strideBytes + static_cast<size_t>(sx);
        if(off >= dataSize) {
            return 0.0f;
        }
        return static_cast<float>(raw[off]);
    };
    auto sampleU16 = [&](int sx, int sy) -> float {
        const size_t off = static_cast<size_t>(sy) * strideBytes + static_cast<size_t>(sx) * sizeof(uint16_t);
        if(off + sizeof(uint16_t) > dataSize) {
            return 0.0f;
        }
        const auto *p = reinterpret_cast<const uint16_t *>(raw + off);
        return static_cast<float>(*p);
    };

    float v00 = 0.0f, v10 = 0.0f, v01 = 0.0f, v11 = 0.0f;
    if(fmt == OB_FORMAT_Y8) {
        v00 = sampleU8(x0, y0);
        v10 = sampleU8(x1, y0);
        v01 = sampleU8(x0, y1);
        v11 = sampleU8(x1, y1);
    }
    else {
        v00 = sampleU16(x0, y0);
        v10 = sampleU16(x1, y0);
        v01 = sampleU16(x0, y1);
        v11 = sampleU16(x1, y1);
    }

    const float v0  = v00 + fx * (v10 - v00);
    const float v1  = v01 + fx * (v11 - v01);
    const float v   = v0 + fy * (v1 - v0);
    const float out = v / maxVal;
    if(!std::isfinite(out)) {
        return 0.0f;
    }
    return std::min(1.0f, std::max(0.0f, out));
}

static std::shared_ptr<ob::Frame> applyIrConfidenceMaskToDepth(const std::shared_ptr<ob::DepthFrame> &depthFrame,
                                                               const std::shared_ptr<ob::IRFrame> &irLeft,
                                                               const std::shared_ptr<ob::IRFrame> &irRight,
                                                               double threshold01) {
    if(!depthFrame) {
        return nullptr;
    }
    if(threshold01 <= 0.0) {
        return depthFrame;
    }
    if(!irLeft || !irRight) {
        return depthFrame;
    }

    std::shared_ptr<ob::DepthFrame> maskedDepth;
    try {
        auto cloned = ob::FrameFactory::createFrameFromOtherFrame(depthFrame, true);
        maskedDepth = cloned->as<ob::DepthFrame>();
    }
    catch(...) {
        return depthFrame;
    }
    if(!maskedDepth) {
        return depthFrame;
    }

    const int dw = static_cast<int>(maskedDepth->getWidth());
    const int dh = static_cast<int>(maskedDepth->getHeight());
    if(dw <= 0 || dh <= 0) {
        return depthFrame;
    }

    auto *depthRaw = reinterpret_cast<uint8_t *>(maskedDepth->data());
    if(!depthRaw) {
        return depthFrame;
    }
    const size_t depthSize = static_cast<size_t>(maskedDepth->dataSize());
    const size_t depthStrideBytes = dh > 0 ? (depthSize / static_cast<size_t>(dh)) : 0;
    if(depthStrideBytes < static_cast<size_t>(dw) * sizeof(uint16_t)) {
        return depthFrame;
    }

    const float scale = maskedDepth->getValueScale();
    if(!(scale > 0.0f)) {
        return depthFrame;
    }

    std::shared_ptr<ob::StreamProfile> depthProfile;
    std::shared_ptr<ob::StreamProfile> irLProfile;
    std::shared_ptr<ob::StreamProfile> irRProfile;
    try {
        depthProfile = maskedDepth->getStreamProfile();
        irLProfile   = irLeft->getStreamProfile();
        irRProfile   = irRight->getStreamProfile();
    }
    catch(...) {
        return depthFrame;
    }
    if(!depthProfile || !irLProfile || !irRProfile) {
        return depthFrame;
    }

    std::shared_ptr<ob::VideoStreamProfile> depthVsp;
    std::shared_ptr<ob::VideoStreamProfile> irLVsp;
    std::shared_ptr<ob::VideoStreamProfile> irRVsp;
    try {
        depthVsp = depthProfile->as<ob::VideoStreamProfile>();
        irLVsp   = irLProfile->as<ob::VideoStreamProfile>();
        irRVsp   = irRProfile->as<ob::VideoStreamProfile>();
    }
    catch(...) {
        return depthFrame;
    }
    if(!depthVsp || !irLVsp || !irRVsp) {
        return depthFrame;
    }

    const auto depthIntrinsic = depthVsp->getIntrinsic();
    const auto depthDist      = depthVsp->getDistortion();
    const auto irLIntrinsic   = irLVsp->getIntrinsic();
    const auto irLDist        = irLVsp->getDistortion();
    const auto irRIntrinsic   = irRVsp->getIntrinsic();
    const auto irRDist        = irRVsp->getDistortion();

    OBExtrinsic extrD2L{};
    OBExtrinsic extrD2R{};
    try {
        extrD2L = depthProfile->getExtrinsicTo(irLProfile);
        extrD2R = depthProfile->getExtrinsicTo(irRProfile);
    }
    catch(...) {
        return depthFrame;
    }

    const float th = static_cast<float>(std::min(1.0, std::max(0.0, threshold01)));
    for(int y = 0; y < dh; y++) {
        auto *rowU16 = reinterpret_cast<uint16_t *>(depthRaw + static_cast<size_t>(y) * depthStrideBytes);
        for(int x = 0; x < dw; x++) {
            const uint16_t d = rowU16[x];
            if(d == 0) {
                continue;
            }
            const float depthMm = static_cast<float>(d) * scale;
            if(!(depthMm > 0.0f)) {
                rowU16[x] = 0;
                continue;
            }

            const OBPoint2f src{ static_cast<float>(x), static_cast<float>(y) };
            OBPoint2f pL{}, pR{};
            bool okL = false;
            bool okR = false;
            try {
                okL = ob::CoordinateTransformHelper::transformation2dto2d(src, depthMm, depthIntrinsic, depthDist, irLIntrinsic, irLDist, extrD2L, &pL);
            }
            catch(...) {
                okL = false;
            }
            try {
                okR = ob::CoordinateTransformHelper::transformation2dto2d(src, depthMm, depthIntrinsic, depthDist, irRIntrinsic, irRDist, extrD2R, &pR);
            }
            catch(...) {
                okR = false;
            }
            if(!okL || !okR) {
                rowU16[x] = 0;
                continue;
            }

            const float cL   = bilinearSampleNormalized(irLeft, pL.x, pL.y);
            const float cR   = bilinearSampleNormalized(irRight, pR.x, pR.y);
            const float conf = std::min(cL, cR);
            if(conf < th) {
                rowU16[x] = 0;
            }
        }
    }

    return maskedDepth;
}

class InteractiveVisualizationApp {
public:
    explicit InteractiveVisualizationApp(AppConfig cfg, const std::atomic_bool *cancel, EgoRecorder *sharedEgoRecorder = nullptr)
        : cfg_(std::move(cfg)),
          cancel_(cancel),
          egoRecorder_(sharedEgoRecorder ? *sharedEgoRecorder : ownedEgoRecorder_),
          ownsEgoRecorder_(sharedEgoRecorder == nullptr) {}

    InteractiveExit run() {
        ivizInstallCrashHandlerOnce();
        ivizSetStage("run_enter");
        auto deviceList = ctx_.queryDeviceList();
        ivizSetStage("deviceList_ok");
        if(!deviceList || deviceList->deviceCount() == 0) {
            std::cerr << "No device connected" << std::endl;
            return InteractiveExit::ReturnMenu;
        }

        ivizSetStage("selectDevicesWithPipeline");
        devices_ = selectDevicesWithPipeline(deviceList, cfg_);
        ivizSetStage("selectDevicesWithPipeline_ok");
        if(devices_.empty()) {
            std::cerr << "No configured devices found" << std::endl;
            return InteractiveExit::ReturnMenu;
        }

        ivizSetStage("applySyncConfig");
        if(cfg_.enableSync) {
            applySyncConfig(devices_);
        }
        ivizSetStage("applySyncConfig_ok");
        std::vector<DeviceRuntime> primary;
        std::vector<DeviceRuntime> secondary;
        ivizSetStage("splitPrimarySecondary");
        splitPrimarySecondary(devices_, primary, secondary);
        ivizSetStage("splitPrimarySecondary_ok");

        int fps = cfg_.viewerFps;
        if(fps <= 0) {
            fps = 30;
        }
        viewerIntervalUs_ = static_cast<uint64_t>(1000000.0 / fps);

        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            cameraEnabled_.assign(devices_.size(), 1);
            frameCountByDevice_.assign(devices_.size(), 0);
        }
        extrinsicCameraTagPoseVisible_.assign(devices_.size(), 1);
        ivizSetStage("loadInitExtrinsics");
        loadInitExtrinsicsIfNeeded();
        ivizSetStage("loadInitExtrinsics_ok");
        initializeFisheyes();
        if(cfg_.demo.active && cfg_.demo.interaction.handGt) {
            setGtVisualizationEnabled(true);
        }

        const std::string winName = "Interaction";
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 1720, 900);
        ivizSetStage("window_created");

        struct ExitGuard {
            InteractiveVisualizationApp *self = nullptr;
            std::string win;
            bool winCreated = false;
            ~ExitGuard() {
                if(!self) {
                    return;
                }
                try {
                    self->stopAllPipelines();
                }
                catch(...) {
                }
                if(winCreated) {
                    try {
                        cv::destroyWindow(win);
                    }
                    catch(...) {
                    }
                }
            }
        } guard;
        guard.self = this;
        guard.win = winName;
        guard.winCreated = true;

        InteractiveViewState viewState;
        CvMouseState uiMs;
        MainMouseContext mouseCtx;
        mouseCtx.view = &viewState;
        mouseCtx.ui   = &uiMs;
        cv::setMouseCallback(winName, mouseCallbackMain, &mouseCtx);

        const uint64_t ignoreInputUntilUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                                      std::chrono::steady_clock::now().time_since_epoch())
                                                                      .count())
                                             + 300000;

        StreamMode desiredMode = computeDesiredStreamMode();
        {
            std::string m = "DepthOnly";
            if(desiredMode == StreamMode::DepthColor) {
                m = "Depth+Color";
            }
            else if(desiredMode == StreamMode::DepthIrLeft) {
                m = "Depth+IR_Left";
            }
            else if(desiredMode == StreamMode::DepthIrRight) {
                m = "Depth+IR_Right";
            }
            else if(desiredMode == StreamMode::DepthIrStereo) {
                m = "Depth+IR_LR";
            }
            else if(desiredMode == StreamMode::DepthIrAny) {
                m = "Depth+IR";
            }
            std::cerr << "Interaction start stream mode: " << m << std::endl;
        }
        ivizSetStage("startPipelines_secondary");
        startSelectedPipelines(secondary, desiredMode);
        ivizSetStage("startPipelines_primary");
        startSelectedPipelines(primary, desiredMode);
        ivizSetStage("startPipelines_ok");

        if(cfg_.enableSync && devices_.size() > 1) {
            try {
                ivizSetStage("enableDeviceClockSync");
                ctx_.enableDeviceClockSync(60000);
                clockSyncEnabled_ = true;
            }
            catch(...) {
                clockSyncEnabled_ = false;
            }
        }

        uint64_t lastRenderUs = 0;
        cv::Mat  lastPc;
        InteractiveExit exitReason = InteractiveExit::Quit;
        bool running = true;
        while(running) {
            ivizSetStage("loop_waitKey");
            const int key = cv::waitKeyEx(1);
            if(key > 0) {
                if(isCtrlModifierKeyEvent(key)) {
                    g_ivizCtrlShortcutListening = true;
                }
                else if(g_ivizCtrlShortcutListening && isCtrlReleaseKeyEvent(key)) {
                    g_ivizCtrlShortcutListening = false;
                }
            }

            const uint64_t loopNowUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                                 std::chrono::steady_clock::now().time_since_epoch())
                                                                 .count());

            CvMouseState frameMs = uiMs;
            uiMs.clicked = false;
            uiMs.wheelDelta = 0;
            ivizSetStage("loop_layout");

            if(loopNowUs < ignoreInputUntilUs) {
                frameMs.clicked = false;
                frameMs.wheelDelta = 0;
            }
            bool keyZoomed = false;
            if(key > 0) {
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld = g_ivizCtrlShortcutListening || ctrlFromMask;
                if(ctrlHeld) {
                    if(isCtrlZoomInKeyEvent(key)) {
                        viewState.distance = std::min(20.0f, std::max(0.2f, viewState.distance * 0.9f));
                        keyZoomed = true;
                    }
                    else if(isCtrlZoomOutKeyEvent(key)) {
                        viewState.distance = std::min(20.0f, std::max(0.2f, viewState.distance * 1.1f));
                        keyZoomed = true;
                    }
                }
            }
            if(key == 'r' || key == 'R') {
                viewState.resetView();
            }
            else if(key == 'q' || key == 'Q' || key == 27) {
                exitReason = InteractiveExit::Quit;
                break;
            }

            if(cancel_ && cancel_->load()) {
                exitReason = InteractiveExit::Quit;
                break;
            }

            int winW = 1720;
            int winH = 900;
            try {
                const auto rect = cv::getWindowImageRect(winName);
                if(rect.width > 0 && rect.height > 0) {
                    winW = rect.width;
                    winH = rect.height;
                }
            }
            catch(...) {
            }

            layout_.update(winW, winH);
            viewState.pcRect = layout_.pcRect;
            viewState.width  = layout_.pcRect.width;
            viewState.height = layout_.pcRect.height;
            ivizSetStage("loop_draw_ui");

            cv::Mat canvas(winH, winW, CV_8UC3, cv::Scalar(20, 20, 20));

            bool streamsDirty = false;
            streamsDirty |= drawLeftPanel(canvas, frameMs);
            streamsDirty |= drawImagePanel(canvas, frameMs);
            if(drawControls(canvas, frameMs, viewState, exitReason)) {
                running = false;
            }

            if(streamsDirty) {
                ivizSetStage("loop_refreshPipelines");
                refreshPipelinesForCurrentMode();
                uiMs.wheelDelta = 0;
            }

            if(streamsDirty || keyZoomed || lastRenderUs == 0 || loopNowUs - lastRenderUs >= viewerIntervalUs_) {
                lastRenderUs = loopNowUs;
                auto frameSnapshot = snapshotProcessingFrames();
                maybeStartPendingExtrinsicHealthCheck(frameSnapshot);
                submitGtInferenceIfNeeded(frameSnapshot, loopNowUs);
                submitEgoAprilTagsIfNeeded(frameSnapshot, loopNowUs);
                updatePicoRgbPreviewIfNeeded(loopNowUs);
                ivizSetStage("loop_render_pointcloud");
                lastPc = renderUnifiedPointCloud(frameSnapshot, viewState);
            }
            if(!lastPc.empty() && layout_.pcRect.width > 0 && layout_.pcRect.height > 0) {
                ivizSetStage("loop_copy_pointcloud");
                if(lastPc.cols == layout_.pcRect.width && lastPc.rows == layout_.pcRect.height) {
                    lastPc.copyTo(canvas(layout_.pcRect));
                }
                else {
                    cv::Mat resized;
                    cv::resize(lastPc, resized, layout_.pcRect.size());
                    resized.copyTo(canvas(layout_.pcRect));
                }
            }
            drawFisheyeOverlay(canvas, layout_.pcRect, snapshotFisheyeFrames());

            ivizSetStage("loop_imshow");
            try {
                cv::imshow(winName, canvas);
            }
            catch(...) {
                exitReason = InteractiveExit::Quit;
                break;
            }
            uiMs.clicked = false;
        }

        {
            const char *reason = "Quit";
            if(exitReason == InteractiveExit::ReturnMenu) {
                reason = "ReturnMenu";
            }
            else if(exitReason == InteractiveExit::ReturnConfig) {
                reason = "ReturnConfig";
            }
            else if(exitReason == InteractiveExit::StartCollection) {
                reason = "StartCollection";
            }
            std::cerr << "[interactive_visualization] exit=" << reason << std::endl;
        }

        return exitReason;
    }

private:
    struct LiveDeskPlane {
        bool  valid   = false;
        float a       = 0.0f;
        float b       = 0.0f;
        float c       = 0.0f;
        float d       = 0.0f;
        float invNorm = 1.0f;
    };

    static LiveDeskPlane fitLiveDeskPlaneSampled(const OBPoint *data, size_t count, float scaleMm) {
        LiveDeskPlane plane;
        if(!data || count < 200) {
            return plane;
        }

        const size_t kMaxSamples = 40000;
        const size_t step = std::max<size_t>(4, count / kMaxSamples);
        const float  sM = scaleMm * 0.001f;
        auto sample = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        sample->points.reserve(std::min(kMaxSamples, count / step + 1));
        for(size_t i = 0; i < count; i += step) {
            const OBPoint &p = data[i];
            if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= 0.0f) {
                continue;
            }
            const float x = p.x * sM;
            const float y = p.y * sM;
            const float z = p.z * sM;
            if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 0.0f) {
                continue;
            }
            sample->points.emplace_back(x, y, z);
        }
        sample->width = static_cast<uint32_t>(sample->points.size());
        sample->height = 1;
        sample->is_dense = false;
        if(sample->points.size() < 200) {
            return plane;
        }

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(false);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(35);
        seg.setDistanceThreshold(0.015);
        seg.setInputCloud(sample);

        pcl::PointIndices::Ptr     inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        try {
            seg.segment(*inliers, *coefficients);
        }
        catch(...) {
            return plane;
        }

        const size_t minInliers = std::min<size_t>(1500, std::max<size_t>(100, sample->points.size() / 20));
        if(!inliers || inliers->indices.size() < minInliers || !coefficients || coefficients->values.size() < 4) {
            return plane;
        }

        const float a = coefficients->values[0];
        const float b = coefficients->values[1];
        const float c = coefficients->values[2];
        const float d = coefficients->values[3];
        const float n = std::sqrt(a * a + b * b + c * c);
        if(!std::isfinite(n) || n <= 1e-6f) {
            return plane;
        }
        plane.valid = true;
        plane.a = a;
        plane.b = b;
        plane.c = c;
        plane.d = d;
        plane.invNorm = 1.0f / n;
        return plane;
    }

    static bool isPointOnLiveDeskPlane(const LiveDeskPlane &plane, float x, float y, float z) {
        if(!plane.valid) {
            return false;
        }
        const float dist = std::fabs((plane.a * x + plane.b * y + plane.c * z + plane.d) * plane.invNorm);
        return std::isfinite(dist) && dist <= 0.015f;
    }

    enum class StreamMode {
        DepthOnly,
        DepthColor,
        DepthIrLeft,
        DepthIrRight,
        DepthIrStereo,
        DepthIrAny
    };

    enum class ImageType { Depth, RGB, IRLeft, IRRight };

    enum class RgbImageSource { Orbbec, Pico };

    struct Layout {
        cv::Rect camsRect;
        cv::Rect pcRect;
        cv::Rect imgRect;
        cv::Rect ctlRect;

        void update(int winW, int winH) {
            const int bottomH = 110;
            const int usableH = std::max(1, winH - bottomH);

            int leftW  = std::max(116, std::min(150, winW / 11));
            int rightW = std::max(320, std::min(400, winW / 5));
            int pcW    = winW - leftW - rightW;
            if(pcW < 1) {
                pcW = 1;
                int remain = winW - pcW;
                if(remain < 1) {
                    remain = 1;
                }
                leftW = std::min(leftW, remain / 2);
                rightW = std::max(1, remain - leftW);
            }
            if(leftW + pcW + rightW != winW) {
                rightW = std::max(1, winW - leftW - pcW);
            }

            camsRect = cv::Rect(0, 0, leftW, usableH);
            pcRect   = cv::Rect(leftW, 0, pcW, usableH);
            imgRect  = cv::Rect(leftW + pcW, 0, rightW, usableH);
            ctlRect  = cv::Rect(0, usableH, winW, bottomH);
        }
    };

    fs::path resolveGtWorkerScriptPath() const {
        std::vector<fs::path> candidates;
        if(!cfg_.initExtrinsicPath.empty()) {
            const fs::path base = fs::absolute(fs::path(cfg_.initExtrinsicPath)).parent_path();
            candidates.push_back(base / "hand_joint_gt_worker.py");
        }
        candidates.push_back(fs::current_path() / "hand_joint_gt_worker.py");
        candidates.push_back(fs::current_path() / "src" / "sync" / "hand_joint_gt_worker.py");
        candidates.push_back(fs::current_path() / ".." / "src" / "sync" / "hand_joint_gt_worker.py");
        for(const auto &p : candidates) {
            std::error_code ec;
            const fs::path absP = fs::absolute(p, ec);
            if(!ec && fs::exists(absP)) {
                return absP;
            }
            if(fs::exists(p)) {
                return p;
            }
        }
        return candidates.empty() ? fs::path() : candidates.front();
    }

    fs::path resolveEgoAprilTagWorkerScriptPath() const {
        std::vector<fs::path> candidates;
        if(!cfg_.initExtrinsicPath.empty()) {
            const fs::path base = fs::absolute(fs::path(cfg_.initExtrinsicPath)).parent_path();
            candidates.push_back(base / "ego_apriltag_overlay_worker.py");
        }
        candidates.push_back(fs::current_path() / "ego_apriltag_overlay_worker.py");
        candidates.push_back(fs::current_path() / "src" / "sync" / "ego_apriltag_overlay_worker.py");
        candidates.push_back(fs::current_path() / ".." / "src" / "sync" / "ego_apriltag_overlay_worker.py");
        for(const auto &p : candidates) {
            std::error_code ec;
            const fs::path absP = fs::absolute(p, ec);
            if(!ec && fs::exists(absP)) {
                return absP;
            }
            if(fs::exists(p)) {
                return p;
            }
        }
        return candidates.empty() ? fs::path() : candidates.front();
    }

    void setGtVisualizationEnabled(bool enabled) {
        if(showGtJoints_ == enabled) {
            return;
        }
        showGtJoints_ = enabled;
        if(showGtJoints_) {
            gtWorker_.setScriptPath(resolveGtWorkerScriptPath());
            gtWorker_.ensureRunning();
            gtWorker_.setIdleStatus("GT waiting for RGB frames", true);
        }
        else {
            gtWorker_.stop();
        }
    }

    fs::path resolveEgoPreviewEpisodeDir() const {
        fs::path base = cfg_.outputDir.empty() ? (fs::current_path() / "interaction_ego_preview") : cfg_.outputDir;
        if(base.is_relative()) {
            base = (fs::current_path() / base).lexically_normal();
        }
        return base / "interaction_ego_preview";
    }

    EgoModuleConfig buildInteractionEgoConfig() const {
        EgoModuleConfig egoCfg = cfg_.ego;
        egoCfg.enabled = true;
        if(egoCfg.maxBufferedFrames == 0) {
            egoCfg.maxBufferedFrames = 4096;
        }
        return egoCfg;
    }

    std::string picoServerListeningStatus() const {
        std::ostringstream oss;
        oss << "PICO server listening on " << cfg_.ego.host << ":" << cfg_.ego.port
            << "; start/connect PICO ego client";
        return oss.str();
    }

    void setEgoAprilTagStatus(const std::string &status, bool clearResult = true) {
        egoTagStatusLine_ = status;
        egoTagWorker_.setIdleStatus(status, clearResult);
    }

    bool wantsPicoRgbPreview() const {
        return imageType_ == ImageType::RGB && rgbImageSource_ == RgbImageSource::Pico;
    }

    bool wantsEgoPreviewSession() const {
        return showEgoAprilTags_ || wantsPicoRgbPreview();
    }

    void clearPicoRgbPreview() {
        picoRgbFrameSource_.close();
        latestPicoRgbFrame_.release();
        latestPicoRgbVideoFrameIndex_ = -1;
        picoRgbStatusLine_ = "PICO RGB off";
    }

    void stopEgoPreviewSessionIfUnused() {
        if(wantsEgoPreviewSession()) {
            return;
        }
        clearPicoRgbPreview();
        if(egoRecorder_.isSessionActive()) {
            std::string err;
            (void)egoRecorder_.stopSessionAndWait(std::chrono::milliseconds(std::min(2000, std::max(100, cfg_.ego.stopTimeoutMs))), &err);
        }
    }

    bool ensureEgoAprilTagPreviewSession(uint64_t nowUs) {
        if(!wantsEgoPreviewSession()) {
            return false;
        }
        if(!cfg_.ego.enabled) {
            setEgoAprilTagStatus("PICO ego is disabled in config");
            return false;
        }

        std::string err;
        if(!egoRecorder_.isRunning()) {
            if(!ownsEgoRecorder_) {
                setEgoAprilTagStatus("PICO server is not ready from menu");
                return false;
            }
            const EgoModuleConfig egoCfg = buildInteractionEgoConfig();
            if(!egoRecorder_.start(egoCfg, &err)) {
                setEgoAprilTagStatus("PICO server start failed: " + err);
                return false;
            }
            setEgoAprilTagStatus(picoServerListeningStatus());
        }

        if(!egoRecorder_.isConnected()) {
            setEgoAprilTagStatus(picoServerListeningStatus());
            return false;
        }

        if(egoRecorder_.isSessionActive()) {
            return true;
        }

        if(lastEgoSessionAttemptUs_ != 0 && nowUs > lastEgoSessionAttemptUs_ && nowUs - lastEgoSessionAttemptUs_ < 1000000) {
            setEgoAprilTagStatus("PICO connected; waiting to start preview session");
            return false;
        }
        const uint64_t attemptUs = nowUs != 0
            ? nowUs
            : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
        lastEgoSessionAttemptUs_ = attemptUs;
        setEgoAprilTagStatus("PICO connected; starting preview session");

        egoPreviewEpisodeDir_ = resolveEgoPreviewEpisodeDir();
        try {
            fs::create_directories(egoPreviewEpisodeDir_);
        }
        catch(const std::exception &ex) {
            setEgoAprilTagStatus(std::string("PICO preview dir failed: ") + ex.what());
            return false;
        }
        if(!egoRecorder_.beginSession(egoPreviewEpisodeDir_, "interaction_preview", &err)) {
            setEgoAprilTagStatus("PICO preview session failed: " + err);
            return false;
        }

        egoVideoPath_ = egoPreviewEpisodeDir_ / "ego" / "RGB" / "rgb.h265";
        egoCameraParamsPath_ = egoPreviewEpisodeDir_ / "ego" / "camera_params.json";
        latestEgoFrame_.reset();
        lastEgoTagSubmitUs_ = 0;
        setEgoAprilTagStatus("PICO preview session active; waiting for RGB frames");
        return true;
    }

    void setEgoAprilTagOverlayEnabled(bool enabled) {
        if(showEgoAprilTags_ == enabled) {
            return;
        }
        if(!enabled) {
            showEgoAprilTags_ = false;
            egoTagWorker_.stop();
            stopEgoPreviewSessionIfUnused();
            if(ownsEgoRecorder_ && !wantsEgoPreviewSession()) {
                egoRecorder_.stop();
            }
            if(!wantsPicoRgbPreview()) {
                latestEgoFrame_.reset();
            }
            lastEgoTagSubmitUs_ = 0;
            if(!wantsEgoPreviewSession()) {
                lastEgoSessionAttemptUs_ = 0;
            }
            egoTagStatusLine_ = "PICO tags off";
            return;
        }

        showEgoAprilTags_ = true;
        egoVideoPath_.clear();
        egoCameraParamsPath_.clear();
        latestEgoFrame_.reset();
        lastEgoTagSubmitUs_ = 0;
        lastEgoSessionAttemptUs_ = 0;
        egoTagWorker_.setScriptPath(resolveEgoAprilTagWorkerScriptPath());
        egoTagWorker_.ensureRunning();
        setEgoAprilTagStatus("Starting PICO AprilTag overlay");
        (void)ensureEgoAprilTagPreviewSession(0);
    }

    fs::path resolveInteractionExtrinsicHealthRoot() const {
        fs::path base = cfg_.outputDir.empty() ? fs::current_path() : cfg_.outputDir;
        if(base.is_relative()) {
            base = (fs::current_path() / base).lexically_normal();
        }
        return base / ".interaction_extrinsic_health";
    }

    void setExtrinsicHealthStatus(const std::string &status) {
        std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
        extrinsicHealthStatusLine_ = status;
    }

    bool isExtrinsicHealthRunning() const {
        std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
        return extrinsicHealthRunning_;
    }

    void joinCompletedExtrinsicHealthThreadIfNeeded() {
        bool shouldJoin = false;
        {
            std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
            shouldJoin = extrinsicHealthThread_.joinable() && !extrinsicHealthRunning_;
        }
        if(shouldJoin) {
            extrinsicHealthThread_.join();
        }
    }

    ExtrinsicHealthSampleResult latestExtrinsicHealthResult() const {
        std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
        return latestExtrinsicHealthResult_;
    }

    std::string buildExtrinsicHealthStatusLine() const {
        std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
        if(extrinsicHealthStatusLine_.empty()) {
            return cfg_.extrinsicHealth.enabled ? "Extrinsic check idle" : "Extrinsic check disabled";
        }
        return extrinsicHealthStatusLine_;
    }

    void requestExtrinsicHealthCheck() {
        joinCompletedExtrinsicHealthThreadIfNeeded();
        const auto &health = cfg_.extrinsicHealth;
        if(!health.enabled) {
            setExtrinsicHealthStatus("Extrinsic check disabled in config");
            return;
        }
        if(cfg_.initExtrinsicPath.empty()) {
            setExtrinsicHealthStatus("Extrinsic check needs init_extrinsic_path");
            return;
        }
        if(health.scriptPath.empty() || !fs::exists(health.scriptPath)) {
            setExtrinsicHealthStatus("Extrinsic check script not found: " + health.scriptPath.string());
            return;
        }
        if(isExtrinsicHealthRunning()) {
            setExtrinsicHealthStatus("Extrinsic check already running");
            return;
        }
        extrinsicHealthForceColor_ = true;
        extrinsicHealthPending_ = true;
        setExtrinsicHealthStatus("Extrinsic check pending RGB/depth sample");
    }

    bool buildExtrinsicHealthCameraRequests(const std::unordered_map<int, CachedFrameBundle> &frames,
                                            std::vector<ExtrinsicHealthCameraRequest> &requests,
                                            std::string &message) const {
        requests.clear();
        for(const auto &kv : frames) {
            const int deviceIndex = kv.first;
            const auto &cached = kv.second;
            if(!isCameraEnabled(deviceIndex) || !cached.depth || !cached.color) {
                continue;
            }

            auto itValid = rgbDepthParamsValid_.find(deviceIndex);
            auto itParam = rgbDepthParamsByDevice_.find(deviceIndex);
            if(itValid == rgbDepthParamsValid_.end() || itParam == rgbDepthParamsByDevice_.end() || !itValid->second) {
                continue;
            }
            if(!(itParam->second.rgbIntrinsic.fx > 0.0f) || !(itParam->second.rgbIntrinsic.fy > 0.0f)
               || !(itParam->second.depthIntrinsic.fx > 0.0f) || !(itParam->second.depthIntrinsic.fy > 0.0f)) {
                continue;
            }
            if(cached.camIndex.empty()) {
                continue;
            }

            const DeviceRuntime *rt = findDeviceRuntimeByIndex(deviceIndex);
            if(!rt || !rt->dev) {
                continue;
            }

            ExtrinsicHealthCameraRequest req;
            req.deviceIndex = deviceIndex;
            req.sn = cached.sn;
            req.camIndex = cached.camIndex;
            req.device = rt->dev;
            req.depth = cached.depth;
            req.color = cached.color;
            req.rgbDepthParam = itParam->second;
            req.rgbDepthParamValid = true;
            req.tsUs = cached.tsUs;
            requests.push_back(std::move(req));
        }

        std::sort(requests.begin(), requests.end(), [](const auto &a, const auto &b) {
            if(a.camIndex == b.camIndex) {
                return a.deviceIndex < b.deviceIndex;
            }
            return a.camIndex < b.camIndex;
        });

        if(requests.size() < 2) {
            message = "Extrinsic check needs at least 2 calibrated RGB/depth cameras";
            return false;
        }
        message.clear();
        return true;
    }

    void maybeStartPendingExtrinsicHealthCheck(const std::unordered_map<int, CachedFrameBundle> &frames) {
        joinCompletedExtrinsicHealthThreadIfNeeded();
        if(!extrinsicHealthPending_ || isExtrinsicHealthRunning()) {
            return;
        }

        std::vector<ExtrinsicHealthCameraRequest> requests;
        std::string message;
        if(!buildExtrinsicHealthCameraRequests(frames, requests, message)) {
            if(!message.empty()) {
                setExtrinsicHealthStatus(message);
            }
            return;
        }

        extrinsicHealthPending_ = false;
        startExtrinsicHealthCheck(std::move(requests));
    }

    void startExtrinsicHealthCheck(std::vector<ExtrinsicHealthCameraRequest> requests) {
        joinCompletedExtrinsicHealthThreadIfNeeded();
        {
            std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
            if(extrinsicHealthRunning_) {
                extrinsicHealthStatusLine_ = "Extrinsic check already running";
                return;
            }
            extrinsicHealthRunning_ = true;
            extrinsicHealthStatusLine_ = "Extrinsic check running";
        }
        extrinsicHealthThread_ = std::thread([this, requests = std::move(requests)]() mutable {
            try {
                runExtrinsicHealthCheckOnce(std::move(requests));
            }
            catch(const std::exception &ex) {
                finishExtrinsicHealthError(std::string("unexpected failure: ") + ex.what());
            }
            catch(...) {
                finishExtrinsicHealthError("unexpected failure");
            }
        });
    }

    bool writeInteractionExtrinsicHealthConfigJson(const fs::path &path) const {
        const auto &h = cfg_.extrinsicHealth;
        cJSON *root = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "tagFamily", h.tagFamily.c_str());
        cJSON_AddNumberToObject(root, "tagSizeM", h.tagSizeM);
        cJSON_AddStringToObject(root, "rotationMethod", h.rotationMethod.c_str());
        cJSON_AddBoolToObject(root, "requireAllCameras", h.requireAllCameras);
        cJSON_AddNumberToObject(root, "minSharedCamerasPerTag", h.minSharedCamerasPerTag);
        cJSON_AddNumberToObject(root, "minTagInlierObservations", h.minTagInlierObservations);
        cJSON_AddNumberToObject(root, "minCheckedCameras", h.minCheckedCameras);
        cJSON_AddNumberToObject(root, "minTagsPerCamera", h.minTagsPerCamera);
        cJSON_AddNumberToObject(root, "minPassingSnapshots", 1);
        cJSON_AddNumberToObject(root, "minFailingSnapshots", 1);
        cJSON_AddNumberToObject(root, "singleTagReprojLimitPx", h.singleTagReprojLimitPx);
        cJSON_AddNumberToObject(root, "fusionTransThreshM", h.fusionTransThreshM);
        cJSON_AddNumberToObject(root, "fusionRotThreshDeg", h.fusionRotThreshDeg);
        cJSON_AddNumberToObject(root, "warnTransThreshM", h.warnTransThreshM);
        cJSON_AddNumberToObject(root, "warnRotThreshDeg", h.warnRotThreshDeg);
        cJSON_AddNumberToObject(root, "warnReprojThreshPx", h.warnReprojThreshPx);
        cJSON_AddNumberToObject(root, "failTransThreshM", h.failTransThreshM);
        cJSON_AddNumberToObject(root, "failRotThreshDeg", h.failRotThreshDeg);
        cJSON_AddNumberToObject(root, "failReprojThreshPx", h.failReprojThreshPx);
        char *printed = cJSON_Print(root);
        bool ok = false;
        if(printed) {
            ok = writeTextFileLocal(path, printed);
            cJSON_free(printed);
        }
        cJSON_Delete(root);
        return ok;
    }

    bool writeInteractionCameraParamsJson(const fs::path &dest,
                                          const std::vector<ExtrinsicHealthCapturedCamera> &cameras,
                                          std::string &message) const {
        cJSON *root = cJSON_CreateObject();
        cJSON *viewerObj = cJSON_CreateObject();
        cJSON_AddNumberToObject(viewerObj, "colorCloudRgbFrameOffset", cfg_.colorCloudRgbFrameOffset);
        cJSON_AddItemToObject(root, "viewer", viewerObj);

        for(const auto &cam : cameras) {
            cJSON *camObj = cJSON_CreateObject();
            jsonAddStringLocal(camObj, "sn", cam.sn);

            cJSON *rgbObj = cJSON_CreateObject();
            jsonAddStringLocal(rgbObj, "storageEncoding", "image");
            jsonAddStringLocal(rgbObj, "filePattern", "%05d.jpg");
            jsonAddNumberLocal(rgbObj, "width", cam.bgr.cols);
            jsonAddNumberLocal(rgbObj, "height", cam.bgr.rows);
            jsonAddNumberLocal(rgbObj, "fps", cfg_.viewerFps > 0 ? cfg_.viewerFps : 30);
            jsonAddNumberLocal(rgbObj, "format", static_cast<int>(cam.colorFormat));
            cJSON *rgbIntr = cJSON_CreateObject();
            jsonAddIntrinsicLocal(rgbIntr, cam.rgbDepthParam.rgbIntrinsic);
            cJSON_AddItemToObject(rgbObj, "intrinsic", rgbIntr);
            cJSON *rgbDist = cJSON_CreateObject();
            jsonAddDistortionLocal(rgbDist, cam.rgbDepthParam.rgbDistortion);
            cJSON_AddItemToObject(rgbObj, "distortion", rgbDist);
            cJSON_AddItemToObject(camObj, "RGB", rgbObj);

            cJSON *depthObj = cJSON_CreateObject();
            jsonAddStringLocal(depthObj, "storageEncoding", "png");
            jsonAddStringLocal(depthObj, "filePattern", "%05d.png");
            jsonAddNumberLocal(depthObj, "width", cam.depthAlignedRgb16.cols);
            jsonAddNumberLocal(depthObj, "height", cam.depthAlignedRgb16.rows);
            jsonAddNumberLocal(depthObj, "fps", cfg_.viewerFps > 0 ? cfg_.viewerFps : 30);
            jsonAddNumberLocal(depthObj, "format", static_cast<int>(cam.depthFormat));
            cJSON *depthIntr = cJSON_CreateObject();
            jsonAddIntrinsicLocal(depthIntr, cam.rgbDepthParam.depthIntrinsic);
            cJSON_AddItemToObject(depthObj, "intrinsic", depthIntr);
            cJSON *depthDist = cJSON_CreateObject();
            jsonAddDistortionLocal(depthDist, cam.rgbDepthParam.depthDistortion);
            cJSON_AddItemToObject(depthObj, "distortion", depthDist);
            cJSON_AddItemToObject(camObj, "Depth", depthObj);

            if(cam.rgbDepthParamValid) {
                cJSON *rgbDepthObj = cJSON_CreateObject();
                cJSON *depthIntr2 = cJSON_CreateObject();
                jsonAddIntrinsicLocal(depthIntr2, cam.rgbDepthParam.depthIntrinsic);
                cJSON_AddItemToObject(rgbDepthObj, "depth_intrinsic", depthIntr2);
                cJSON *depthDist2 = cJSON_CreateObject();
                jsonAddDistortionLocal(depthDist2, cam.rgbDepthParam.depthDistortion);
                cJSON_AddItemToObject(rgbDepthObj, "depth_distortion", depthDist2);
                cJSON *rgbIntr2 = cJSON_CreateObject();
                jsonAddIntrinsicLocal(rgbIntr2, cam.rgbDepthParam.rgbIntrinsic);
                cJSON_AddItemToObject(rgbDepthObj, "rgb_intrinsic", rgbIntr2);
                cJSON *rgbDist2 = cJSON_CreateObject();
                jsonAddDistortionLocal(rgbDist2, cam.rgbDepthParam.rgbDistortion);
                cJSON_AddItemToObject(rgbDepthObj, "rgb_distortion", rgbDist2);

                cJSON *d2cObj = cJSON_CreateObject();
                jsonAddExtrinsicLocal(d2cObj, cam.rgbDepthParam.transform.rot, cam.rgbDepthParam.transform.trans);
                cJSON_AddItemToObject(rgbDepthObj, "d2c_extrinsic", d2cObj);

                float rct[9];
                float tct[3];
                for(int r = 0; r < 3; ++r) {
                    for(int c = 0; c < 3; ++c) {
                        rct[c * 3 + r] = cam.rgbDepthParam.transform.rot[r * 3 + c];
                    }
                }
                for(int r = 0; r < 3; ++r) {
                    float v = 0.0f;
                    for(int c = 0; c < 3; ++c) {
                        v += rct[r * 3 + c] * cam.rgbDepthParam.transform.trans[c];
                    }
                    tct[r] = -v;
                }
                cJSON *c2dObj = cJSON_CreateObject();
                jsonAddExtrinsicLocal(c2dObj, rct, tct);
                cJSON_AddItemToObject(rgbDepthObj, "c2d_extrinsic", c2dObj);
                cJSON_AddItemToObject(camObj, "rgb_to_depth", rgbDepthObj);
            }

            cJSON_AddItemToObject(root, cam.camIndex.c_str(), camObj);
        }

        char *printed = cJSON_Print(root);
        bool ok = false;
        if(printed) {
            ok = writeTextFileLocal(dest / "camera_params.json", printed);
            cJSON_free(printed);
        }
        cJSON_Delete(root);
        if(!ok) {
            message = "failed to write camera_params.json";
        }
        return ok;
    }

    bool writeInteractionExtrinsicsJson(const fs::path &dest, std::string &message) const {
        if(cfg_.initExtrinsicPath.empty()) {
            message = "init_extrinsic_path is empty";
            return false;
        }
        std::string content;
        if(!readTextFileLocal(fs::path(cfg_.initExtrinsicPath), content)) {
            message = "failed to read init extrinsics: " + cfg_.initExtrinsicPath;
            return false;
        }
        cJSON *root = cJSON_Parse(content.c_str());
        if(!root) {
            message = "invalid init extrinsics json";
            return false;
        }
        char *printed = cJSON_Print(root);
        bool ok = false;
        if(printed) {
            ok = writeTextFileLocal(dest / "extrinsics.json", printed);
            cJSON_free(printed);
        }
        cJSON_Delete(root);
        if(!ok) {
            message = "failed to write extrinsics.json";
        }
        return ok;
    }

    bool writeInteractionExtrinsicHealthManifest(const fs::path &checkDir,
                                                 const std::vector<ExtrinsicHealthCapturedCamera> &cameras,
                                                 std::string &message) const {
        const std::string sampleName = "sample_" + formatFrameIndexLocal(0);
        const fs::path sampleDir = checkDir / sampleName;
        try {
            fs::create_directories(sampleDir);
        }
        catch(const std::exception &ex) {
            message = "cannot create sample directory: " + std::string(ex.what());
            return false;
        }

        cJSON *manifest = cJSON_CreateObject();
        cJSON_AddStringToObject(manifest, "camera_params_json", "camera_params.json");
        cJSON_AddStringToObject(manifest, "extrinsics_json", "extrinsics.json");
        cJSON *samples = cJSON_CreateArray();
        cJSON *sampleObj = cJSON_CreateObject();
        cJSON_AddNumberToObject(sampleObj, "index", 0);
        cJSON *camArray = cJSON_CreateArray();

        const int jpegQuality = std::max(1, std::min(100, cfg_.extrinsicHealth.jpegQuality));
        int writtenCameras = 0;
        for(const auto &cam : cameras) {
            const std::string fileName = cam.camIndex + ".jpg";
            const std::string depthFileName = cam.camIndex + ".depth.png";
            const fs::path imagePath = sampleDir / fileName;
            const fs::path depthPath = sampleDir / depthFileName;
            const std::vector<int> jpgParams = { cv::IMWRITE_JPEG_QUALITY, jpegQuality };
            if(!cv::imwrite(imagePath.string(), cam.bgr, jpgParams)) {
                continue;
            }
            if(!cv::imwrite(depthPath.string(), cam.depthAlignedRgb16)) {
                continue;
            }

            cJSON *camObj = cJSON_CreateObject();
            cJSON_AddStringToObject(camObj, "id", cam.camIndex.c_str());
            cJSON_AddStringToObject(camObj, "sn", cam.sn.c_str());
            cJSON_AddStringToObject(camObj, "image", (sampleName + "/" + fileName).c_str());
            cJSON_AddStringToObject(camObj, "depth", (sampleName + "/" + depthFileName).c_str());
            cJSON_AddBoolToObject(camObj, "depth_aligned_to_rgb", true);
            cJSON_AddNumberToObject(camObj, "timestamp_us", static_cast<double>(cam.rgbTsUs));
            cJSON_AddNumberToObject(camObj, "depth_timestamp_us", static_cast<double>(cam.depthTsUs));
            cJSON_AddNumberToObject(camObj, "age_ms", 0.0);
            cJSON_AddNumberToObject(camObj, "depth_age_ms", 0.0);
            cJSON_AddNumberToObject(camObj, "depth_value_scale_mm", cam.depthValueScaleMm);
            cJSON_AddItemToArray(camArray, camObj);
            writtenCameras++;
        }
        cJSON_AddItemToObject(sampleObj, "cameras", camArray);
        cJSON_AddItemToArray(samples, sampleObj);
        cJSON_AddItemToObject(manifest, "samples", samples);
        if(writtenCameras < 2) {
            cJSON_Delete(manifest);
            message = "failed to write enough extrinsic check sample images";
            return false;
        }

        char *printed = cJSON_Print(manifest);
        bool ok = false;
        if(printed) {
            ok = writeTextFileLocal(checkDir / "manifest.json", printed);
            cJSON_free(printed);
        }
        cJSON_Delete(manifest);
        if(!ok) {
            message = "failed to write manifest.json";
        }
        return ok;
    }

    static std::string readJsonStringDefault(cJSON *obj, const char *name) {
        cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, name);
        if(item && cJSON_IsString(item) && item->valuestring) {
            return item->valuestring;
        }
        return "";
    }

    static double readJsonNumberDefault(cJSON *obj, const char *name, double fallback = 0.0) {
        cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, name);
        if(item && cJSON_IsNumber(item)) {
            return item->valuedouble;
        }
        return fallback;
    }

    static int readJsonIntDefault(cJSON *obj, const char *name, int fallback = 0) {
        return static_cast<int>(std::llround(readJsonNumberDefault(obj, name, static_cast<double>(fallback))));
    }

    static std::vector<std::string> readJsonStringArray(cJSON *obj, const char *name) {
        std::vector<std::string> out;
        cJSON *arr = cJSON_GetObjectItemCaseSensitive(obj, name);
        if(!arr || !cJSON_IsArray(arr)) {
            return out;
        }
        cJSON *item = nullptr;
        cJSON_ArrayForEach(item, arr) {
            if(item && cJSON_IsString(item) && item->valuestring) {
                out.emplace_back(item->valuestring);
            }
        }
        return out;
    }

    static std::string joinStrings(const std::vector<std::string> &items, const std::string &sep) {
        std::ostringstream oss;
        for(size_t i = 0; i < items.size(); ++i) {
            if(i > 0) {
                oss << sep;
            }
            oss << items[i];
        }
        return oss.str();
    }

    static void collectExtrinsicHealthResultDetails(cJSON *root, ExtrinsicHealthSampleResult &result) {
        cJSON *counts = cJSON_GetObjectItemCaseSensitive(root, "camera_counts");
        if(counts && cJSON_IsObject(counts)) {
            std::ostringstream oss;
            oss << "Extrinsic camera status:"
                << " total=" << readJsonIntDefault(counts, "total")
                << " pass=" << readJsonIntDefault(counts, "pass")
                << " warn=" << readJsonIntDefault(counts, "warn")
                << " fail=" << readJsonIntDefault(counts, "fail")
                << " inconclusive=" << readJsonIntDefault(counts, "inconclusive");
            result.detailLines.push_back(oss.str());
        }
        const auto fail = readJsonStringArray(root, "fail_cameras");
        const auto warn = readJsonStringArray(root, "warn_cameras");
        const auto inconclusive = readJsonStringArray(root, "inconclusive_cameras");
        if(!fail.empty()) {
            result.detailLines.push_back("fail cameras: " + joinStrings(fail, ","));
        }
        if(!warn.empty()) {
            result.detailLines.push_back("warn cameras: " + joinStrings(warn, ","));
        }
        if(!inconclusive.empty()) {
            result.detailLines.push_back("inconclusive cameras: " + joinStrings(inconclusive, ","));
        }
    }

    static bool parseInteractionExtrinsicHealthResult(const fs::path &path,
                                                      const std::unordered_map<std::string, int> &deviceIndexByCamera,
                                                      const std::string &debugDir,
                                                      ExtrinsicHealthSampleResult &result) {
        std::string content;
        if(!readTextFileLocal(path, content)) {
            return false;
        }
        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            return false;
        }

        result = ExtrinsicHealthSampleResult{};
        result.valid = true;
        result.debugDir = debugDir;
        result.status = readJsonStringDefault(root, "status");
        result.summary = readJsonStringDefault(root, "summary_line");
        collectExtrinsicHealthResultDetails(root, result);

        cJSON *samples = cJSON_GetObjectItemCaseSensitive(root, "samples");
        cJSON *sample = (samples && cJSON_IsArray(samples) && cJSON_GetArraySize(samples) > 0) ? cJSON_GetArrayItem(samples, 0) : nullptr;
        cJSON *tags = sample ? cJSON_GetObjectItemCaseSensitive(sample, "tags") : nullptr;
        if(tags && cJSON_IsObject(tags)) {
            for(cJSON *tagObj = tags->child; tagObj != nullptr; tagObj = tagObj->next) {
                if(!tagObj || !tagObj->string || !cJSON_IsObject(tagObj)) {
                    continue;
                }
                int tagId = -1;
                try {
                    tagId = std::stoi(tagObj->string);
                }
                catch(...) {
                    continue;
                }

                ExtrinsicHealthTagResult tag;
                tag.tagId = tagId;
                cJSON *fused = cJSON_GetObjectItemCaseSensitive(tagObj, "fused");
                if(fused && cJSON_IsObject(fused)) {
                    cJSON *pose = cJSON_GetObjectItemCaseSensitive(fused, "pose");
                    if(parseExtrinsicHealthPose(pose, tag.fusedWorldFromTag)) {
                        tag.hasFused = true;
                    }
                    tag.fusedInlierCameras = readJsonStringArray(fused, "inlier_cameras");
                }

                cJSON *observations = cJSON_GetObjectItemCaseSensitive(tagObj, "observations");
                if(observations && cJSON_IsArray(observations)) {
                    cJSON *obsObj = nullptr;
                    cJSON_ArrayForEach(obsObj, observations) {
                        if(!obsObj || !cJSON_IsObject(obsObj)) {
                            continue;
                        }
                        ExtrinsicHealthTagObservation obs;
                        obs.cameraId = readJsonStringDefault(obsObj, "camera_id");
                        obs.rmsePx = readJsonNumberDefault(obsObj, "rmse_px");
                        cJSON *inlierItem = cJSON_GetObjectItemCaseSensitive(obsObj, "inlier");
                        obs.inlier = inlierItem && cJSON_IsBool(inlierItem) ? (inlierItem->valueint != 0) : false;
                        auto itDevice = findByCamKeyVariants(deviceIndexByCamera, obs.cameraId);
                        if(itDevice) {
                            obs.deviceIndex = *itDevice;
                        }
                        cJSON *worldPose = cJSON_GetObjectItemCaseSensitive(obsObj, "T_world_from_tag");
                        if(!parseExtrinsicHealthPose(worldPose, obs.worldFromTag)) {
                            continue;
                        }
                        tag.observations.push_back(std::move(obs));
                    }
                }
                result.tags[tagId] = std::move(tag);
            }
        }

        cJSON_Delete(root);
        return !result.status.empty();
    }

    std::unordered_map<std::string, int> buildCameraDeviceIndexMap() const {
        std::unordered_map<std::string, int> out;
        for(const auto &rt : devices_) {
            out[rt.cfg.index] = rt.deviceIndex;
        }
        return out;
    }

    std::vector<ExtrinsicHealthCapturedCamera> captureExtrinsicHealthCameras(std::vector<ExtrinsicHealthCameraRequest> requests,
                                                                             std::string &message) const {
        std::vector<ExtrinsicHealthCapturedCamera> out;
        out.reserve(requests.size());
        for(auto &req : requests) {
            cv::Mat bgr = visualizeObFrame(req.color);
            if(bgr.empty()) {
                continue;
            }

            cv::Mat depthAligned = buildDepthAlignedToRgbViaSdk(req.device, req.depth, bgr.cols, bgr.rows);
            if(depthAligned.empty() && req.rgbDepthParamValid) {
                depthAligned = buildDepthAlignedToRgb(req.depth, req.rgbDepthParam, bgr.cols, bgr.rows);
            }
            if(depthAligned.empty() || depthAligned.type() != CV_16UC1 || depthAligned.size() != bgr.size()) {
                continue;
            }

            ExtrinsicHealthCapturedCamera cam;
            cam.deviceIndex = req.deviceIndex;
            cam.sn = req.sn;
            cam.camIndex = req.camIndex;
            cam.bgr = std::move(bgr);
            cam.depthAlignedRgb16 = std::move(depthAligned);
            cam.rgbDepthParam = req.rgbDepthParam;
            cam.rgbDepthParamValid = req.rgbDepthParamValid;
            cam.rgbTsUs = frameTimestampUsLocal(std::static_pointer_cast<ob::Frame>(req.color));
            if(cam.rgbTsUs == 0) {
                cam.rgbTsUs = req.tsUs;
            }
            cam.depthTsUs = frameTimestampUsLocal(std::static_pointer_cast<ob::Frame>(req.depth));
            if(cam.depthTsUs == 0) {
                cam.depthTsUs = req.tsUs;
            }
            try {
                cam.depthValueScaleMm = req.depth ? req.depth->getValueScale() : 1.0f;
            }
            catch(...) {
                cam.depthValueScaleMm = 1.0f;
            }
            try {
                cam.colorFormat = req.color ? req.color->getFormat() : OB_FORMAT_UNKNOWN;
            }
            catch(...) {
                cam.colorFormat = OB_FORMAT_UNKNOWN;
            }
            try {
                cam.depthFormat = req.depth ? req.depth->getFormat() : OB_FORMAT_UNKNOWN;
            }
            catch(...) {
                cam.depthFormat = OB_FORMAT_UNKNOWN;
            }
            out.push_back(std::move(cam));
        }

        if(out.size() < 2) {
            message = "Extrinsic check could not capture enough RGB/depth samples";
        }
        else {
            message.clear();
        }
        return out;
    }

    void finishExtrinsicHealthCheck(const ExtrinsicHealthSampleResult &result, const std::string &statusLine, bool running = false) {
        std::lock_guard<std::mutex> lock(extrinsicHealthMtx_);
        latestExtrinsicHealthResult_ = result;
        extrinsicHealthStatusLine_ = statusLine;
        extrinsicHealthRunning_ = running;
    }

    void finishExtrinsicHealthError(const std::string &message) {
        ExtrinsicHealthSampleResult result;
        result.valid = false;
        result.status = "error";
        result.summary = message;
        finishExtrinsicHealthCheck(result, "Extrinsic check error: " + message, false);
        std::cerr << "[interaction][extrinsic_check] error " << message << std::endl;
    }

    void runExtrinsicHealthCheckOnce(std::vector<ExtrinsicHealthCameraRequest> requests) {
        std::string message;
        auto cameras = captureExtrinsicHealthCameras(std::move(requests), message);
        if(cameras.size() < 2) {
            finishExtrinsicHealthError(message.empty() ? "not enough RGB/depth samples" : message);
            return;
        }

        const auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
        const auto nowMs = now.time_since_epoch().count();
        const fs::path checkDir = resolveInteractionExtrinsicHealthRoot()
                                / ("sample_" + sanitizePathComponentLocal(std::to_string(nowMs)));
        try {
            fs::create_directories(checkDir);
        }
        catch(const std::exception &ex) {
            finishExtrinsicHealthError("cannot create debug directory: " + std::string(ex.what()));
            return;
        }

        if(!writeInteractionCameraParamsJson(checkDir, cameras, message)
           || !writeInteractionExtrinsicsJson(checkDir, message)
           || !writeInteractionExtrinsicHealthConfigJson(checkDir / "health_config.json")
           || !writeInteractionExtrinsicHealthManifest(checkDir, cameras, message)) {
            finishExtrinsicHealthError(message.empty() ? "failed to write snapshot files" : message);
            return;
        }

        const fs::path resultJson = checkDir / "result.json";
        const auto &health = cfg_.extrinsicHealth;
        const std::string command = shellQuoteLocal(health.pythonExecutable)
                                  + " " + shellQuoteLocal(health.scriptPath.string())
                                  + " --snapshot-dir " + shellQuoteLocal(checkDir.string())
                                  + " --config-json " + shellQuoteLocal((checkDir / "health_config.json").string())
                                  + " --result-json " + shellQuoteLocal(resultJson.string())
                                  + " 2>&1";

        std::string output;
        const int exitCode = runCommandCaptureLocal(command, output);
        ExtrinsicHealthSampleResult result;
        const bool parsed = parseInteractionExtrinsicHealthResult(resultJson, buildCameraDeviceIndexMap(), checkDir.string(), result);
        if(!parsed) {
            result.valid = false;
            result.status = "error";
            result.summary = trimString(output);
            if(result.summary.empty()) {
                result.summary = "exit_code=" + std::to_string(exitCode);
            }
            result.debugDir = checkDir.string();
        }
        if(result.summary.empty()) {
            result.summary = trimString(output);
        }
        if(result.summary.empty()) {
            result.summary = "exit_code=" + std::to_string(exitCode);
        }

        bool keepDebug = health.keepDebugSnapshots || result.status == "fail" || result.status == "error" || !parsed;
        if(!keepDebug) {
            try {
                fs::remove_all(checkDir);
                result.debugDir.clear();
            }
            catch(...) {
            }
        }

        std::string statusLine = "Extrinsic check " + result.summary;
        if(!result.debugDir.empty()) {
            statusLine += " debug=" + result.debugDir;
        }
        finishExtrinsicHealthCheck(result, statusLine, false);
        std::cerr << "[interaction][extrinsic_check] status=" << (result.status.empty() ? "(missing)" : result.status)
                  << " exit=" << exitCode
                  << " dir=" << checkDir
                  << " summary=" << result.summary << std::endl;
    }

    std::vector<int> extrinsicHealthTagIds(const ExtrinsicHealthSampleResult &result) const {
        std::vector<int> ids;
        ids.reserve(result.tags.size());
        for(const auto &kv : result.tags) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    int selectedExtrinsicTagId(const ExtrinsicHealthSampleResult &result) {
        const auto ids = extrinsicHealthTagIds(result);
        if(ids.empty()) {
            selectedExtrinsicTagId_ = -1;
            return -1;
        }
        if(std::find(ids.begin(), ids.end(), selectedExtrinsicTagId_) == ids.end()) {
            selectedExtrinsicTagId_ = ids.front();
        }
        return selectedExtrinsicTagId_;
    }

    void stepSelectedExtrinsicTag(int delta) {
        auto result = latestExtrinsicHealthResult();
        auto ids = extrinsicHealthTagIds(result);
        if(ids.empty()) {
            selectedExtrinsicTagId_ = -1;
            return;
        }
        auto it = std::find(ids.begin(), ids.end(), selectedExtrinsicTagId_);
        int index = it == ids.end() ? 0 : static_cast<int>(std::distance(ids.begin(), it));
        index = (index + delta) % static_cast<int>(ids.size());
        if(index < 0) {
            index += static_cast<int>(ids.size());
        }
        selectedExtrinsicTagId_ = ids[static_cast<size_t>(index)];
    }

    GtInferenceRequest buildGtInferenceRequest(const std::unordered_map<int, CachedFrameBundle> &frames, uint64_t frameId) const {
        GtInferenceRequest req;
        req.frameId = frameId;
        for(const auto &kv : frames) {
            const int deviceIndex = kv.first;
            const auto &cached = kv.second;
            if(!isCameraEnabled(deviceIndex) || !cached.color) {
                continue;
            }

            auto itValid = rgbDepthParamsValid_.find(deviceIndex);
            auto itParam = rgbDepthParamsByDevice_.find(deviceIndex);
            if(itValid == rgbDepthParamsValid_.end() || itParam == rgbDepthParamsByDevice_.end() || !itValid->second) {
                continue;
            }
            if(!(itParam->second.rgbIntrinsic.fx > 0.0f) || !(itParam->second.rgbIntrinsic.fy > 0.0f)) {
                continue;
            }

            const auto *rgbPose = findByCamKeyVariants(rgbExtrinsicsCamToWorld_, cached.camIndex);
            if(!rgbPose || !rgbPose->valid) {
                continue;
            }

            GtCameraFramePacket cam;
            cam.camIndex = cached.camIndex;
            cam.deviceIndex = deviceIndex;
            cam.rgbIntrinsic = itParam->second.rgbIntrinsic;
            invertRigid(rgbPose->R, rgbPose->t, cam.Rcw, cam.tcw);
            cam.Rwc = rgbPose->R;
            cam.twc = rgbPose->t;
            cam.depthUnitMm = cached.depth ? cached.depth->getValueScale() : 1.0f;
            cv::Mat rgb = visualizeObFrame(cached.color);
            if(rgb.empty()) {
                continue;
            }
            cam.rgbBgr = resizeMatKeepingAspect(rgb, gtWorkerMaxImageSide_, cv::INTER_AREA, cam.rgbScaleX, cam.rgbScaleY);
            if(cam.rgbBgr.empty()) {
                continue;
            }
            cam.depthAlignedRgb16.release();
            cam.depthScaleX = 1.0f;
            cam.depthScaleY = 1.0f;

            req.captureTsUs = std::max(req.captureTsUs, cached.tsUs);
            req.cameras.push_back(std::move(cam));
        }
        std::sort(req.cameras.begin(), req.cameras.end(), [](const GtCameraFramePacket &a, const GtCameraFramePacket &b) {
            if(a.camIndex == b.camIndex) {
                return a.deviceIndex < b.deviceIndex;
            }
            return a.camIndex < b.camIndex;
        });
        return req;
    }

    void submitGtInferenceIfNeeded(const std::unordered_map<int, CachedFrameBundle> &frames, uint64_t frameId) {
        if(!showGtJoints_) {
            return;
        }
        if(gtWorker_.hasPendingRequest()) {
            return;
        }
        GtInferenceRequest req = buildGtInferenceRequest(frames, frameId);
        if(req.cameras.empty()) {
            gtWorker_.setIdleStatus("GT waiting for calibrated RGB cameras", true);
            return;
        }
        if(req.cameras.size() < 2) {
            gtWorker_.setIdleStatus("GT needs at least 2 calibrated RGB views", true);
            return;
        }
        gtWorker_.submitLatest(std::move(req));
    }

    void pollEgoFrames() {
        if(!wantsEgoPreviewSession() || !egoRecorder_.isRunning()) {
            return;
        }
        EgoFrame frame;
        int      popped = 0;
        while(popped < 32 && egoRecorder_.popFrame(frame, std::chrono::milliseconds(0))) {
            latestEgoFrame_ = frame;
            popped++;
        }
    }

    EgoAprilTagRequest buildEgoAprilTagRequest(const std::unordered_map<int, CachedFrameBundle> &frames, uint64_t frameId) const {
        EgoAprilTagRequest req;
        req.frameId = frameId;
        req.egoVideoPath = egoVideoPath_;
        req.egoCameraParamsPath = egoCameraParamsPath_;
        if(latestEgoFrame_.has_value()) {
            req.egoVideoFrameIndex = latestEgoFrame_->videoFrameIndex;
        }

        for(const auto &kv : frames) {
            const int deviceIndex = kv.first;
            const auto &cached = kv.second;
            if(!isCameraEnabled(deviceIndex) || !cached.color) {
                continue;
            }

            auto itValid = rgbDepthParamsValid_.find(deviceIndex);
            auto itParam = rgbDepthParamsByDevice_.find(deviceIndex);
            if(itValid == rgbDepthParamsValid_.end() || itParam == rgbDepthParamsByDevice_.end() || !itValid->second) {
                continue;
            }
            if(!(itParam->second.rgbIntrinsic.fx > 0.0f) || !(itParam->second.rgbIntrinsic.fy > 0.0f)) {
                continue;
            }

            const auto *rgbPose = findByCamKeyVariants(rgbExtrinsicsCamToWorld_, cached.camIndex);
            if(!rgbPose || !rgbPose->valid) {
                continue;
            }

            cv::Mat rgb = visualizeObFrame(cached.color);
            if(rgb.empty()) {
                continue;
            }

            EgoAprilTagCameraPacket cam;
            cam.camIndex = cached.camIndex;
            cam.deviceIndex = deviceIndex;
            cam.rgbIntrinsic = itParam->second.rgbIntrinsic;
            cam.rgbDistortion = itParam->second.rgbDistortion;
            cam.Rwc = rgbPose->R;
            cam.twc = rgbPose->t;
            cam.rgbBgr = resizeMatKeepingAspect(rgb, egoTagWorkerMaxImageSide_, cv::INTER_AREA, cam.rgbScaleX, cam.rgbScaleY);
            if(cam.rgbBgr.empty()) {
                continue;
            }
            if(!cam.rgbBgr.isContinuous()) {
                cam.rgbBgr = cam.rgbBgr.clone();
            }
            req.cameras.push_back(std::move(cam));
        }

        std::sort(req.cameras.begin(), req.cameras.end(), [](const EgoAprilTagCameraPacket &a, const EgoAprilTagCameraPacket &b) {
            if(a.camIndex == b.camIndex) {
                return a.deviceIndex < b.deviceIndex;
            }
            return a.camIndex < b.camIndex;
        });
        return req;
    }

    void submitEgoAprilTagsIfNeeded(const std::unordered_map<int, CachedFrameBundle> &frames, uint64_t frameId) {
        if(!showEgoAprilTags_) {
            return;
        }
        if(!ensureEgoAprilTagPreviewSession(frameId)) {
            return;
        }
        pollEgoFrames();
        if(egoTagWorker_.hasPendingRequest()) {
            return;
        }
        if(lastEgoTagSubmitUs_ != 0 && frameId > lastEgoTagSubmitUs_ && frameId - lastEgoTagSubmitUs_ < 120000) {
            return;
        }
        if(!latestEgoFrame_.has_value()) {
            setEgoAprilTagStatus("PICO preview session active; waiting for PICO RGB frames");
            return;
        }
        if(latestEgoFrame_->videoFrameIndex < 0) {
            setEgoAprilTagStatus("PICO RGB received; waiting for H265 video frame");
            return;
        }
        if(egoVideoPath_.empty() || egoCameraParamsPath_.empty()) {
            setEgoAprilTagStatus("PICO tags missing preview paths");
            return;
        }

        EgoAprilTagRequest req = buildEgoAprilTagRequest(frames, frameId);
        if(req.cameras.empty()) {
            setEgoAprilTagStatus("PICO tags need calibrated Orbbec RGB");
            return;
        }
        lastEgoTagSubmitUs_ = frameId;
        egoTagWorker_.submitLatest(std::move(req));
    }

    void updatePicoRgbPreviewIfNeeded(uint64_t frameId) {
        if(!wantsPicoRgbPreview()) {
            return;
        }
        if(!ensureEgoAprilTagPreviewSession(frameId)) {
            picoRgbStatusLine_ = buildEgoAprilTagStatusLine();
            return;
        }
        picoRgbFrameSource_.start(&egoRecorder_);

        cv::Mat frame;
        int decodedFrameIndex = -1;
        std::string status;
        if(picoRgbFrameSource_.latest(frame, &decodedFrameIndex, &status, latestPicoRgbVideoFrameIndex_)) {
            latestPicoRgbFrame_ = std::move(frame);
            latestPicoRgbVideoFrameIndex_ = decodedFrameIndex;
        }
        picoRgbStatusLine_ = status.empty()
                            ? ("PICO RGB live frame " + std::to_string(latestPicoRgbVideoFrameIndex_))
                            : status;
    }

    std::string buildGtStatusLine() const {
        if(!showGtJoints_) {
            return "GT hands off";
        }
        const auto result = gtWorker_.latestResult();
        const std::string workerStatus = gtWorker_.statusLine();
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(1);
        oss << "GT hands " << result.visibleHands;
        bool firstVisible = true;
        for(const auto &hand : result.hands) {
            if(!hand.visible) {
                continue;
            }
            oss << (firstVisible ? " | " : " ");
            firstVisible = false;
            oss << hand.side << ":" << hand.validJointCount;
        }
        if(result.workerFps > 0.0) {
            oss << " | " << result.workerFps << " Hz";
        }
        if(!workerStatus.empty()) {
            oss << " | " << workerStatus;
        }
        return oss.str();
    }

    std::string buildEgoAprilTagStatusLine() const {
        if(!showEgoAprilTags_) {
            return egoTagStatusLine_.empty() ? "PICO tags off" : egoTagStatusLine_;
        }
        const auto result = egoTagWorker_.latestResult();
        const std::string workerStatus = egoTagWorker_.statusLine();
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(1);
        oss << "PICO tags";
        if(result.referenceTagCount > 0 || result.tagCount > 0) {
            oss << " " << result.tagCount << " | ref " << result.referenceTagCount;
        }
        if(result.rmsePx > 0.0) {
            oss << " | " << result.rmsePx << "px";
        }
        if(result.workerFps > 0.0) {
            oss << " | " << result.workerFps << " Hz";
        }
        if(!workerStatus.empty()) {
            oss << " | " << workerStatus;
        }
        return oss.str();
    }

    void drawGtJointsOnCanvas(cv::Mat &canvas,
                              const std::vector<float> &zbuf,
                              const GtInferenceResult &gtResult,
                              const cv::Vec3f &right,
                              const cv::Vec3f &up,
                              const cv::Vec3f &forward,
                              const cv::Vec3f &camPos,
                              float fx,
                              float fy,
                              float cx,
                              float cy) const {
        (void)zbuf;
        struct ProjectedJoint {
            cv::Point  pt;
            cv::Scalar color;
            int        radius = 0;
            float      depth = 0.0f;
            bool       valid = false;
        };

        static const std::array<std::pair<int, int>, 23> kSkeletonEdges = {
            std::make_pair(0, 1),   std::make_pair(1, 2),   std::make_pair(2, 3),   std::make_pair(3, 4),   std::make_pair(0, 5),   std::make_pair(5, 6),
            std::make_pair(6, 7),   std::make_pair(7, 8),   std::make_pair(0, 9),   std::make_pair(9, 10),  std::make_pair(10, 11), std::make_pair(11, 12),
            std::make_pair(0, 13),  std::make_pair(13, 14), std::make_pair(14, 15), std::make_pair(15, 16), std::make_pair(0, 17),  std::make_pair(17, 18),
            std::make_pair(18, 19), std::make_pair(19, 20), std::make_pair(5, 9),   std::make_pair(9, 13),  std::make_pair(13, 17)
        };

        auto projectJoint = [&](const GtJoint3d &joint, ProjectedJoint &out) -> bool {
            if(!joint.valid) {
                return false;
            }
            const cv::Vec3f v = joint.position - camPos;
            const float xc = v.dot(right);
            const float yc = v.dot(up);
            const float zc = v.dot(forward);
            if(zc <= 0.05f) {
                return false;
            }
            const int u = static_cast<int>(fx * (xc / zc) + cx);
            const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
            if(u < 0 || u >= canvas.cols || vpx < 0 || vpx >= canvas.rows) {
                return false;
            }
            out.pt = cv::Point(u, vpx);
            out.color = toScalar(joint.color);
            out.depth = zc;
            out.radius = std::max(4, std::min(10, static_cast<int>(std::lround((0.012f * fx) / std::max(0.1f, zc)))));
            out.valid = true;
            return true;
        };

        for(const auto &hand : gtResult.hands) {
            if(!hand.visible || hand.joints.size() < 21) {
                continue;
            }

            std::array<ProjectedJoint, 21> projected{};
            for(size_t i = 0; i < hand.joints.size() && i < projected.size(); ++i) {
                projectJoint(hand.joints[i], projected[i]);
            }

            const cv::Scalar skeletonColor = toScalar(gtSkeletonColorForSide(hand.side));
            for(const auto &edge : kSkeletonEdges) {
                const auto &a = projected[static_cast<size_t>(edge.first)];
                const auto &b = projected[static_cast<size_t>(edge.second)];
                if(!a.valid || !b.valid) {
                    continue;
                }
                cv::line(canvas, a.pt, b.pt, cv::Scalar(12, 12, 12), 4, cv::LINE_AA);
                cv::line(canvas, a.pt, b.pt, skeletonColor, 2, cv::LINE_AA);
            }

            std::vector<ProjectedJoint> ordered;
            ordered.reserve(projected.size());
            for(const auto &pj : projected) {
                if(pj.valid) {
                    ordered.push_back(pj);
                }
            }
            std::sort(ordered.begin(), ordered.end(), [](const ProjectedJoint &a, const ProjectedJoint &b) { return a.depth > b.depth; });
            for(const auto &pj : ordered) {
                cv::circle(canvas, pj.pt, pj.radius + 2, cv::Scalar(16, 16, 16), cv::FILLED, cv::LINE_AA);
                cv::circle(canvas, pj.pt, pj.radius, pj.color, cv::FILLED, cv::LINE_AA);
                cv::circle(canvas, pj.pt, std::max(1, pj.radius / 2), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }
    }

    void drawEgoAprilTagsOnCanvas(cv::Mat &canvas,
                                  const EgoAprilTagResult &result,
                                  const cv::Vec3f &right,
                                  const cv::Vec3f &up,
                                  const cv::Vec3f &forward,
                                  const cv::Vec3f &camPos,
                                  float fx,
                                  float fy,
                                  float cx,
                                  float cy) const {
        if(result.lines.empty()) {
            return;
        }

        auto project = [&](const cv::Vec3f &p, cv::Point &pt, float &depth) -> bool {
            const cv::Vec3f v = p - camPos;
            const float xc = v.dot(right);
            const float yc = v.dot(up);
            const float zc = v.dot(forward);
            if(zc <= 0.05f) {
                return false;
            }
            const int u = static_cast<int>(fx * (xc / zc) + cx);
            const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
            if(u < 0 || u >= canvas.cols || vpx < 0 || vpx >= canvas.rows) {
                return false;
            }
            pt = cv::Point(u, vpx);
            depth = zc;
            return true;
        };

        for(const auto &line : result.lines) {
            cv::Point a;
            cv::Point b;
            float depthA = 0.0f;
            float depthB = 0.0f;
            if(!project(line.p0, a, depthA) || !project(line.p1, b, depthB)) {
                continue;
            }
            const cv::Scalar color = toScalar(line.color);
            cv::line(canvas, a, b, cv::Scalar(8, 8, 8), 5, cv::LINE_AA);
            cv::line(canvas, a, b, color, 2, cv::LINE_AA);
        }
    }

    bool isExtrinsicCameraTagPoseVisible(int deviceIndex) const {
        if(!showExtrinsicCameraTagPoses_) {
            return false;
        }
        if(deviceIndex < 0 || deviceIndex >= static_cast<int>(extrinsicCameraTagPoseVisible_.size())) {
            return true;
        }
        return extrinsicCameraTagPoseVisible_[static_cast<size_t>(deviceIndex)] != 0;
    }

    static cv::Vec3b extrinsicHealthObservationColor(int deviceIndex, int order) {
        static const std::vector<cv::Vec3b> palette = {
            cv::Vec3b(0, 80, 255),
            cv::Vec3b(0, 220, 90),
            cv::Vec3b(255, 110, 0),
            cv::Vec3b(230, 220, 40),
            cv::Vec3b(255, 60, 230),
            cv::Vec3b(0, 220, 255),
            cv::Vec3b(180, 120, 255),
            cv::Vec3b(80, 255, 220),
        };
        const int idx = deviceIndex >= 0 ? deviceIndex : order;
        return palette[static_cast<size_t>(std::max(0, idx)) % palette.size()];
    }

    bool projectWorldPoint(const cv::Vec3f &p,
                           const cv::Vec3f &right,
                           const cv::Vec3f &up,
                           const cv::Vec3f &forward,
                           const cv::Vec3f &camPos,
                           float fx,
                           float fy,
                           float cx,
                           float cy,
                           const cv::Size &size,
                           cv::Point &pt) const {
        const cv::Vec3f v = p - camPos;
        const float xc = v.dot(right);
        const float yc = v.dot(up);
        const float zc = v.dot(forward);
        if(zc <= 0.05f) {
            return false;
        }
        const int u = static_cast<int>(fx * (xc / zc) + cx);
        const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
        if(u < 0 || u >= size.width || vpx < 0 || vpx >= size.height) {
            return false;
        }
        pt = cv::Point(u, vpx);
        return true;
    }

    cv::Vec3f transformExtrinsicHealthPosePoint(const ExtrinsicHealthPose &pose, const cv::Vec3f &local) const {
        return pose.R * local + pose.t;
    }

    void drawExtrinsicHealthPose(cv::Mat &canvas,
                                 const ExtrinsicHealthPose &pose,
                                 int tagId,
                                 const std::string &labelSuffix,
                                 const cv::Vec3b &color,
                                 bool fused,
                                 const cv::Vec3f &right,
                                 const cv::Vec3f &up,
                                 const cv::Vec3f &forward,
                                 const cv::Vec3f &camPos,
                                 float fx,
                                 float fy,
                                 float cx,
                                 float cy) const {
        if(!pose.valid) {
            return;
        }
        const float tagSize = static_cast<float>(std::max(0.001, cfg_.extrinsicHealth.tagSizeM));
        const float h = tagSize * 0.5f;
        const float axisLen = std::max(0.035f, tagSize * 0.75f);
        const std::array<cv::Vec3f, 4> localCorners = {
            cv::Vec3f(-h, -h, 0.0f),
            cv::Vec3f( h, -h, 0.0f),
            cv::Vec3f( h,  h, 0.0f),
            cv::Vec3f(-h,  h, 0.0f),
        };

        std::array<cv::Point, 4> projectedCorners{};
        std::array<bool, 4> cornerVisible{};
        for(size_t i = 0; i < localCorners.size(); ++i) {
            const cv::Vec3f pw = transformExtrinsicHealthPosePoint(pose, localCorners[i]);
            cornerVisible[i] = projectWorldPoint(pw, right, up, forward, camPos, fx, fy, cx, cy, canvas.size(), projectedCorners[i]);
        }

        const cv::Scalar outline = toScalar(color);
        const int thickness = fused ? 3 : 2;
        for(size_t i = 0; i < projectedCorners.size(); ++i) {
            const size_t j = (i + 1) % projectedCorners.size();
            if(cornerVisible[i] && cornerVisible[j]) {
                cv::line(canvas, projectedCorners[i], projectedCorners[j], cv::Scalar(8, 8, 8), thickness + 3, cv::LINE_AA);
                cv::line(canvas, projectedCorners[i], projectedCorners[j], outline, thickness, cv::LINE_AA);
            }
        }

        cv::Point originPt;
        const cv::Vec3f origin = transformExtrinsicHealthPosePoint(pose, cv::Vec3f(0.0f, 0.0f, 0.0f));
        const bool originVisible = projectWorldPoint(origin, right, up, forward, camPos, fx, fy, cx, cy, canvas.size(), originPt);
        if(originVisible) {
            const std::array<std::pair<cv::Vec3f, cv::Scalar>, 3> axes = {
                std::make_pair(cv::Vec3f(axisLen, 0.0f, 0.0f), cv::Scalar(60, 60, 255)),
                std::make_pair(cv::Vec3f(0.0f, axisLen, 0.0f), cv::Scalar(60, 255, 60)),
                std::make_pair(cv::Vec3f(0.0f, 0.0f, axisLen), cv::Scalar(255, 120, 60)),
            };
            for(const auto &axis : axes) {
                cv::Point axisPt;
                const cv::Vec3f axisWorld = transformExtrinsicHealthPosePoint(pose, axis.first);
                if(projectWorldPoint(axisWorld, right, up, forward, camPos, fx, fy, cx, cy, canvas.size(), axisPt)) {
                    cv::line(canvas, originPt, axisPt, cv::Scalar(5, 5, 5), thickness + 3, cv::LINE_AA);
                    cv::line(canvas, originPt, axisPt, axis.second, thickness, cv::LINE_AA);
                }
            }

            cv::circle(canvas, originPt, fused ? 5 : 4, cv::Scalar(8, 8, 8), cv::FILLED, cv::LINE_AA);
            cv::circle(canvas, originPt, fused ? 4 : 3, outline, cv::FILLED, cv::LINE_AA);
            const std::string label = "tag " + std::to_string(tagId) + labelSuffix;
            cv::putText(canvas, label, originPt + cv::Point(7, -7), cv::FONT_HERSHEY_DUPLEX, fused ? 0.46 : 0.40,
                        cv::Scalar(8, 8, 8), 3, cv::LINE_AA);
            cv::putText(canvas, label, originPt + cv::Point(7, -7), cv::FONT_HERSHEY_DUPLEX, fused ? 0.46 : 0.40,
                        fused ? cv::Scalar(255, 255, 255) : outline, 1, cv::LINE_AA);
        }
    }

    void drawExtrinsicHealthOnCanvas(cv::Mat &canvas,
                                     ExtrinsicHealthSampleResult &result,
                                     const cv::Vec3f &right,
                                     const cv::Vec3f &up,
                                     const cv::Vec3f &forward,
                                     const cv::Vec3f &camPos,
                                     float fx,
                                     float fy,
                                     float cx,
                                     float cy) {
        if(!showExtrinsicHealthOverlay_ || !result.valid || result.tags.empty()) {
            return;
        }

        if(!showExtrinsicCameraTagPoses_ && !showExtrinsicFusedTagPoses_) {
            return;
        }

        if(extrinsicHealthVizMode_ == ExtrinsicHealthVizMode::SingleTag) {
            const int tagId = selectedExtrinsicTagId(result);
            auto it = result.tags.find(tagId);
            if(it == result.tags.end()) {
                return;
            }
            if(showExtrinsicCameraTagPoses_) {
                int order = 0;
                for(const auto &obs : it->second.observations) {
                    if(!isExtrinsicCameraTagPoseVisible(obs.deviceIndex)) {
                        continue;
                    }
                    const cv::Vec3b color = extrinsicHealthObservationColor(obs.deviceIndex, order++);
                    drawExtrinsicHealthPose(canvas, obs.worldFromTag, tagId, " " + obs.cameraId, color, false,
                                            right, up, forward, camPos, fx, fy, cx, cy);
                }
            }
            if(showExtrinsicFusedTagPoses_ && it->second.hasFused) {
                drawExtrinsicHealthPose(canvas, it->second.fusedWorldFromTag, tagId, " fused", cv::Vec3b(255, 255, 255), true,
                                        right, up, forward, camPos, fx, fy, cx, cy);
            }
            return;
        }

        int order = 0;
        for(const auto &kv : result.tags) {
            if(showExtrinsicCameraTagPoses_) {
                for(const auto &obs : kv.second.observations) {
                    if(!isExtrinsicCameraTagPoseVisible(obs.deviceIndex)) {
                        continue;
                    }
                    const cv::Vec3b color = extrinsicHealthObservationColor(obs.deviceIndex, order++);
                    drawExtrinsicHealthPose(canvas, obs.worldFromTag, kv.first, " " + obs.cameraId, color, false,
                                            right, up, forward, camPos, fx, fy, cx, cy);
                }
            }
            if(showExtrinsicFusedTagPoses_ && kv.second.hasFused) {
                drawExtrinsicHealthPose(canvas, kv.second.fusedWorldFromTag, kv.first, " fused", cv::Vec3b(255, 255, 255), true,
                                        right, up, forward, camPos, fx, fy, cx, cy);
            }
        }
    }

    void stopAllPipelines() {
        extrinsicHealthPending_ = false;
        if(extrinsicHealthThread_.joinable()) {
            extrinsicHealthThread_.join();
        }
        gtWorker_.stop();
        egoTagWorker_.stop();
        clearPicoRgbPreview();
        if(egoRecorder_.isSessionActive()) {
            std::string err;
            (void)egoRecorder_.stopSessionAndWait(std::chrono::milliseconds(std::min(2000, std::max(100, cfg_.ego.stopTimeoutMs))), &err);
        }
        if(ownsEgoRecorder_) {
            egoRecorder_.stop();
        }
        fisheyeRecorder_.stop();
        for(auto &rt: devices_) {
            try {
                rt.pipe->stop();
            }
            catch(...) {
            }
        }
        {
            std::lock_guard<std::mutex> lock(framesMtx_);
            frames_.clear();
            frameQueues_.clear();
        }
    }

    bool isCameraEnabled(int deviceIndex) const {
        std::lock_guard<std::mutex> lock(stateMtx_);
        if(deviceIndex < 0 || deviceIndex >= static_cast<int>(cameraEnabled_.size())) {
            return false;
        }
        return cameraEnabled_[deviceIndex] != 0;
    }

    const DeviceRuntime *findDeviceRuntimeByIndex(int deviceIndex) const {
        for(const auto &rt : devices_) {
            if(rt.deviceIndex == deviceIndex) {
                return &rt;
            }
        }
        return nullptr;
    }

    void setCameraEnabled(int deviceIndex, bool enabled) {
        std::lock_guard<std::mutex> lock(stateMtx_);
        if(deviceIndex < 0 || deviceIndex >= static_cast<int>(cameraEnabled_.size())) {
            return;
        }
        cameraEnabled_[deviceIndex] = enabled ? 1 : 0;
    }

    StreamMode computeDesiredStreamMode() const {
        if(extrinsicHealthForceColor_ || extrinsicHealthPending_ || showExtrinsicHealthOverlay_) {
            return StreamMode::DepthColor;
        }
        if(showEgoAprilTags_) {
            return StreamMode::DepthColor;
        }
        if(showGtJoints_) {
            return StreamMode::DepthColor;
        }
        if(cfg_.colorfulCloudPoints) {
            return StreamMode::DepthColor;
        }
        if(imageType_ == ImageType::RGB) {
            return StreamMode::DepthColor;
        }
        if(imageType_ == ImageType::IRLeft) {
            return StreamMode::DepthIrLeft;
        }
        if(imageType_ == ImageType::IRRight) {
            return StreamMode::DepthIrRight;
        }
        return StreamMode::DepthOnly;
    }

    StreamMode normalizeDesiredModeForDevice(DeviceRuntime &rt, StreamMode desired) {
        if(desired == StreamMode::DepthColor) {
            return pickDefaultVideoProfile(rt.pipe, OB_SENSOR_COLOR) ? StreamMode::DepthColor : StreamMode::DepthOnly;
        }
        if(desired == StreamMode::DepthIrLeft) {
            if(pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_LEFT)) {
                return StreamMode::DepthIrLeft;
            }
            if(pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR)) {
                return StreamMode::DepthIrAny;
            }
            return StreamMode::DepthOnly;
        }
        if(desired == StreamMode::DepthIrRight) {
            if(pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_RIGHT)) {
                return StreamMode::DepthIrRight;
            }
            if(pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR)) {
                return StreamMode::DepthIrAny;
            }
            return StreamMode::DepthOnly;
        }
        if(desired == StreamMode::DepthIrStereo) {
            if(pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_LEFT) && pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_RIGHT)) {
                return StreamMode::DepthIrStereo;
            }
            if(pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR)) {
                return StreamMode::DepthIrAny;
            }
            return StreamMode::DepthOnly;
        }
        return StreamMode::DepthOnly;
    }

    void startSelectedPipelines(std::vector<DeviceRuntime> &devices, StreamMode desiredMode) {
        for(auto &rt: devices) {
            const int deviceIndex = rt.deviceIndex;
            if(!isCameraEnabled(deviceIndex)) {
                continue;
            }
            const auto normalized = normalizeDesiredModeForDevice(rt, desiredMode);
            const auto started    = startPipeline(rt, normalized);
            pipelineModeByDevice_[deviceIndex] = started;
        }
    }

    StreamMode startPipeline(DeviceRuntime &rt, StreamMode desiredMode) {
        auto config = std::make_shared<ob::Config>();
        std::unordered_set<OBSensorType> enabledSensors;

        auto depthProfile = pickDepthProfileForPointCloud(rt.pipe, rt.cfg.streams);
        if(!depthProfile) {
            std::cerr << "No depth profile for device: " << rt.cfg.sn << std::endl;
            return StreamMode::DepthOnly;
        }
        config->enableStream(depthProfile);
        enabledSensors.insert(OB_SENSOR_DEPTH);
        std::shared_ptr<ob::VideoStreamProfile> colorProfile;

        StreamMode actualMode = StreamMode::DepthOnly;
        if(desiredMode == StreamMode::DepthColor) {
            colorProfile = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_COLOR);
            if(colorProfile) {
                config->enableStream(colorProfile);
                enabledSensors.insert(OB_SENSOR_COLOR);
                actualMode = StreamMode::DepthColor;
            }
        }
        else if(desiredMode == StreamMode::DepthIrStereo) {
            auto irL = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_LEFT);
            auto irR = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_RIGHT);
            if(irL && irR) {
                config->enableStream(irL);
                config->enableStream(irR);
                enabledSensors.insert(OB_SENSOR_IR_LEFT);
                enabledSensors.insert(OB_SENSOR_IR_RIGHT);
                actualMode = StreamMode::DepthIrStereo;
            }
            else {
                auto ir = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR);
                if(ir) {
                    config->enableStream(ir);
                    enabledSensors.insert(OB_SENSOR_IR);
                    actualMode = StreamMode::DepthIrAny;
                }
            }
        }
        else if(desiredMode == StreamMode::DepthIrLeft) {
            auto irL = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_LEFT);
            if(irL) {
                config->enableStream(irL);
                enabledSensors.insert(OB_SENSOR_IR_LEFT);
                actualMode = StreamMode::DepthIrLeft;
            }
            else {
                auto ir = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR);
                if(ir) {
                    config->enableStream(ir);
                    enabledSensors.insert(OB_SENSOR_IR);
                    actualMode = StreamMode::DepthIrAny;
                }
            }
        }
        else if(desiredMode == StreamMode::DepthIrRight) {
            auto irR = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR_RIGHT);
            if(irR) {
                config->enableStream(irR);
                enabledSensors.insert(OB_SENSOR_IR_RIGHT);
                actualMode = StreamMode::DepthIrRight;
            }
            else {
                auto ir = pickDefaultVideoProfile(rt.pipe, OB_SENSOR_IR);
                if(ir) {
                    config->enableStream(ir);
                    enabledSensors.insert(OB_SENSOR_IR);
                    actualMode = StreamMode::DepthIrAny;
                }
            }
        }

        if(cfg_.enableSync && enabledSensors.size() > 1) {
            config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
        }
        else {
            config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
        }

        if(!cfg_.filters.preset.empty() && cfg_.filters.preset != "0") {
            try {
                const auto list = rt.dev ? rt.dev->getAvailablePresetList() : nullptr;
                if(list && list->getCount() > 0) {
                    const std::string want = normalizePresetKey(cfg_.filters.preset);
                    const char *match = nullptr;
                    for(uint32_t i = 0; i < list->getCount(); i++) {
                        const char *name = list->getName(i);
                        if(!name) {
                            continue;
                        }
                        if(normalizePresetKey(std::string(name)) == want) {
                            match = name;
                            break;
                        }
                    }
                    if(match) {
                        rt.dev->loadPreset(match);
                    }
                    else {
                        rt.dev->loadPreset(cfg_.filters.preset.c_str());
                    }
                }
            }
            catch(...) {
            }
        }

        const auto deviceSn    = rt.cfg.sn;
        const auto camIndex    = rt.cfg.index;
        const auto deviceIndex = rt.deviceIndex;
        try {
            if(cfg_.enableSync && enabledSensors.size() > 1) {
                try {
                    rt.pipe->enableFrameSync();
                }
                catch(...) {
                }
            }
            rt.pipe->start(config, [this, deviceSn, camIndex, deviceIndex](std::shared_ptr<ob::FrameSet> frameSet) {
                onFrameSet(deviceSn, camIndex, deviceIndex, frameSet);
            });
            if(actualMode == StreamMode::DepthColor && colorProfile) {
                try {
                    const auto cp = rt.pipe->getCameraParamWithProfile(colorProfile->getWidth(),
                                                                        colorProfile->getHeight(),
                                                                        depthProfile->getWidth(),
                                                                        depthProfile->getHeight());
                    rgbDepthParamsByDevice_[deviceIndex] = cp;
                    rgbDepthParamsValid_[deviceIndex] = true;
                }
                catch(...) {
                    rgbDepthParamsValid_[deviceIndex] = false;
                }
            }
            else {
                rgbDepthParamsValid_[deviceIndex] = false;
            }
        }
        catch(const ob::Error &e) {
            std::cerr << "Pipeline start failed: " << deviceSn << ", " << e.getMessage() << std::endl;
            actualMode = StreamMode::DepthOnly;
            rgbDepthParamsValid_[deviceIndex] = false;
        }
        catch(const std::exception &e) {
            std::cerr << "Pipeline start failed: " << deviceSn << ", " << e.what() << std::endl;
            actualMode = StreamMode::DepthOnly;
            rgbDepthParamsValid_[deviceIndex] = false;
        }
        return actualMode;
    }

    void restartPipeline(int deviceIndex, bool enable) {
        for(auto &rt: devices_) {
            if(rt.deviceIndex != deviceIndex) {
                continue;
            }
            try {
                rt.pipe->stop();
            }
            catch(...) {
            }
            if(enable) {
                const auto desired = normalizeDesiredModeForDevice(rt, computeDesiredStreamMode());
                pipelineModeByDevice_[deviceIndex] = startPipeline(rt, desired);
            }
            else {
                pipelineModeByDevice_.erase(deviceIndex);
                rgbDepthParamsValid_.erase(deviceIndex);
                rgbDepthParamsByDevice_.erase(deviceIndex);
            }
            break;
        }
        {
            std::lock_guard<std::mutex> lock(framesMtx_);
            frames_.erase(deviceIndex);
            frameQueues_.erase(deviceIndex);
        }
    }

    void refreshPipelinesForCurrentMode() {
        const auto desiredBase = computeDesiredStreamMode();
        for(auto &rt: devices_) {
            const int deviceIndex = rt.deviceIndex;
            if(!isCameraEnabled(deviceIndex)) {
                continue;
            }
            const auto desired = normalizeDesiredModeForDevice(rt, desiredBase);
            auto it = pipelineModeByDevice_.find(deviceIndex);
            if(it == pipelineModeByDevice_.end() || it->second != desired) {
                restartPipeline(deviceIndex, true);
            }
        }
    }

    void onFrameSet(const std::string &deviceSn, const std::string &camIndex, int deviceIndex, const std::shared_ptr<ob::FrameSet> &frameSet) {
        if(!frameSet) {
            return;
        }
        int localCount = 0;
        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            if(deviceIndex < 0 || deviceIndex >= static_cast<int>(cameraEnabled_.size()) || cameraEnabled_[deviceIndex] == 0) {
                return;
            }
            if(deviceIndex >= 0 && deviceIndex < static_cast<int>(frameCountByDevice_.size())) {
                frameCountByDevice_[deviceIndex]++;
                localCount = frameCountByDevice_[deviceIndex];
            }
        }

        auto depth = frameSet->depthFrame();
        if(!depth) {
            return;
        }
        uint64_t ts = 0;
        const bool allowGlobalTs = clockSyncEnabled_ && localCount >= 10;
        if(allowGlobalTs) {
            try {
                ts = depth->globalTimeStampUs();
            }
            catch(...) {
            }
        }
        if(ts == 0) {
            ts = depth->timeStampUs();
        }
        if(ts == 0) {
            return;
        }

        CachedFrameBundle b;
        b.tsUs = ts;
        b.depth = depth;
        b.color = frameSet->colorFrame();
        {
            auto fL = frameSet->getFrame(OB_FRAME_IR_LEFT);
            b.irLeft = fL ? fL->as<ob::IRFrame>() : nullptr;
        }
        {
            auto fR = frameSet->getFrame(OB_FRAME_IR_RIGHT);
            b.irRight = fR ? fR->as<ob::IRFrame>() : nullptr;
        }
        b.ir = frameSet->irFrame();
        b.sn = deviceSn;
        b.camIndex = camIndex;
        {
            std::lock_guard<std::mutex> lock(framesMtx_);
            frames_[deviceIndex] = std::move(b);
            auto &queue = frameQueues_[deviceIndex];
            queue.push_back(frames_[deviceIndex]);
            while(queue.size() > kMaxAlignedFrameQueueSize_) {
                queue.pop_front();
            }
        }
    }

    std::unordered_map<int, CachedFrameBundle> snapshotFrames() {
        std::lock_guard<std::mutex> lock(framesMtx_);
        return frames_;
    }

    std::unordered_map<int, CachedFrameBundle> snapshotProcessingFrames() {
        const bool useHardAligned = cfg_.enableSync && clockSyncEnabled_;
        if(!useHardAligned) {
            return snapshotFrames();
        }

        std::vector<int> enabledDeviceIndices;
        {
            std::lock_guard<std::mutex> lock(stateMtx_);
            enabledDeviceIndices.reserve(cameraEnabled_.size());
            for(size_t i = 0; i < cameraEnabled_.size(); ++i) {
                if(cameraEnabled_[i] != 0) {
                    enabledDeviceIndices.push_back(static_cast<int>(i));
                }
            }
        }
        if(enabledDeviceIndices.size() < 2) {
            return snapshotFrames();
        }

        std::lock_guard<std::mutex> lock(framesMtx_);
        uint64_t centerUs = std::numeric_limits<uint64_t>::max();
        for(int deviceIndex : enabledDeviceIndices) {
            auto itQueue = frameQueues_.find(deviceIndex);
            if(itQueue == frameQueues_.end() || itQueue->second.empty()) {
                return frames_;
            }
            centerUs = std::min(centerUs, itQueue->second.back().tsUs);
        }
        if(centerUs == std::numeric_limits<uint64_t>::max()) {
            return frames_;
        }

        const uint64_t stepUs = viewerIntervalUs_ > 0 ? viewerIntervalUs_ : 33333;
        const uint64_t maxAbsDiffUs = interactiveAlignedMaxAbsDiffUs(stepUs);
        std::unordered_map<int, CachedFrameBundle> aligned;
        aligned.reserve(enabledDeviceIndices.size());
        for(int deviceIndex : enabledDeviceIndices) {
            auto itQueue = frameQueues_.find(deviceIndex);
            if(itQueue == frameQueues_.end() || itQueue->second.empty()) {
                return frames_;
            }
            size_t picked = 0;
            if(!pickNearestFrameBundle(itQueue->second, centerUs, maxAbsDiffUs, picked)) {
                return frames_;
            }
            aligned.emplace(deviceIndex, itQueue->second[picked]);
        }
        return aligned;
    }

    void initializeFisheyes() {
        fisheyeRecorder_.stop();
        fisheyeVisible_.clear();
        fisheyeLabels_.clear();
        fisheyeStatusLine_.clear();

        const auto devices = listPreferredFisheyeDevices(preferredFisheyeCameraLabels());
        if(devices.empty()) {
            fisheyeStatusLine_ = "No fisheye detected";
            return;
        }

        const auto cfg = buildAutoFisheyeConfigFromDevices(devices);
        std::string err;
        if(!fisheyeRecorder_.start(cfg, &err)) {
            fisheyeStatusLine_ = "Fisheye unavailable: " + err;
            return;
        }
        if(!fisheyeRecorder_.waitUntilReady(std::chrono::seconds(2))) {
            fisheyeRecorder_.stop();
            fisheyeStatusLine_ = "Fisheye start timed out";
            return;
        }

        fisheyeVisible_.assign(cfg.cameras.size(), 0);
        fisheyeLabels_.reserve(cfg.cameras.size());
        for(const auto &camera : cfg.cameras) {
            fisheyeLabels_.push_back(camera.cameraId);
        }
        std::ostringstream oss;
        oss << "Fisheye connected: " << fisheyeLabels_.size();
        fisheyeStatusLine_ = oss.str();
    }

    std::optional<FisheyeFrameSet> snapshotFisheyeFrames() {
        if(!fisheyeRecorder_.isRunning()) {
            return std::nullopt;
        }
        std::string err;
        auto snap = fisheyeRecorder_.snapshotLatest(&err);
        if(!snap) {
            if(!err.empty()) {
                fisheyeStatusLine_ = err;
            }
            return std::nullopt;
        }
        return snap;
    }

    void drawFisheyeOverlay(cv::Mat &canvas, const cv::Rect &pcRect, const std::optional<FisheyeFrameSet> &snapshot) {
        if(!snapshot || snapshot->frames.empty() || pcRect.width <= 0 || pcRect.height <= 0) {
            return;
        }
        std::vector<size_t> visibleIndices;
        for(size_t i = 0; i < snapshot->frames.size() && i < fisheyeVisible_.size(); ++i) {
            if(fisheyeVisible_[i] != 0) {
                visibleIndices.push_back(i);
            }
        }
        if(visibleIndices.empty()) {
            return;
        }

        const int maxTileW = std::min(220, std::max(120, pcRect.width / 4));
        const int gap = 8;
        const int cols = (visibleIndices.size() > 2) ? 2 : 1;
        int x = pcRect.x + 10;
        int y = pcRect.y + 10;
        int col = 0;
        int rowMaxH = 0;
        for(size_t order = 0; order < visibleIndices.size(); ++order) {
            const auto &frame = snapshot->frames[visibleIndices[order]];
            if(frame.bgr.empty()) {
                continue;
            }
            const int tileW = maxTileW;
            const int tileH = std::max(72, static_cast<int>(static_cast<double>(frame.bgr.rows) * (static_cast<double>(tileW) / static_cast<double>(frame.bgr.cols))));
            const cv::Rect tile(x, y, std::min(tileW, pcRect.x + pcRect.width - x - 10), std::min(tileH + 28, pcRect.y + pcRect.height - y - 10));
            if(tile.width <= 40 || tile.height <= 40) {
                break;
            }
            cv::rectangle(canvas, tile, cv::Scalar(15, 15, 15), cv::FILLED);
            cv::rectangle(canvas, tile, cv::Scalar(120, 120, 120), 1);
            cv::putText(canvas, frame.cameraId, cv::Point(tile.x + 8, tile.y + 18), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
            cv::Rect imageR(tile.x + 4, tile.y + 24, tile.width - 8, tile.height - 28);
            if(imageR.width > 8 && imageR.height > 8) {
                cv::Mat resized;
                cv::resize(frame.bgr, resized, imageR.size(), 0, 0, cv::INTER_AREA);
                resized.copyTo(canvas(imageR));
            }
            rowMaxH = std::max(rowMaxH, tile.height);
            col++;
            if(col >= cols) {
                col = 0;
                x = pcRect.x + 10;
                y += rowMaxH + gap;
                rowMaxH = 0;
            }
            else {
                x += tile.width + gap;
            }
        }
    }

    void loadInitExtrinsicsIfNeeded() {
        depthExtrinsicsCamToWorld_.clear();
        rgbExtrinsicsCamToWorld_.clear();
        if(cfg_.initExtrinsicPath.empty()) {
            return;
        }
        fs::path p = fs::path(cfg_.initExtrinsicPath);
        std::string content;
        try {
            content = readFileAllLocal(p);
        }
        catch(...) {
            return;
        }

        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            return;
        }

        for(cJSON *item = root->child; item != nullptr; item = item->next) {
            if(!item->string || !cJSON_IsObject(item)) {
                continue;
            }
            const std::string camId = item->string;
            auto *rotArr            = cJSON_GetObjectItemCaseSensitive(item, "rotation");
            auto *tArr              = cJSON_GetObjectItemCaseSensitive(item, "translation");
            cv::Vec3f tCwRgb;
            if(!parseVec3(tArr, tCwRgb)) {
                continue;
            }

            cv::Matx33f RcwRgb = cv::Matx33f::eye();
            if(!parseMat3(rotArr, RcwRgb)) {
                continue;
            }

            cv::Matx33f RcwDepth = RcwRgb;
            cv::Vec3f tCwDepth = tCwRgb;

            auto *rgbDepthObj = cJSON_GetObjectItemCaseSensitive(item, "rgb_to_depth");
            if(rgbDepthObj && cJSON_IsObject(rgbDepthObj)) {
                cv::Matx33f Rrgb2depth = cv::Matx33f::eye();
                cv::Vec3f trgb2depth(0.0f, 0.0f, 0.0f);
                bool hasRgb2Depth = false;
                auto *c2dObj = cJSON_GetObjectItemCaseSensitive(rgbDepthObj, "c2d_extrinsic");
                if(parseExtrinsicObject(c2dObj, Rrgb2depth, trgb2depth)) {
                    trgb2depth = translationMmToM(trgb2depth);
                    hasRgb2Depth = true;
                }
                else {
                    cv::Matx33f Rdepth2rgb = cv::Matx33f::eye();
                    cv::Vec3f tdepth2rgb(0.0f, 0.0f, 0.0f);
                    auto *d2cObj = cJSON_GetObjectItemCaseSensitive(rgbDepthObj, "d2c_extrinsic");
                    if(parseExtrinsicObject(d2cObj, Rdepth2rgb, tdepth2rgb)) {
                        tdepth2rgb = translationMmToM(tdepth2rgb);
                        invertRigid(Rdepth2rgb, tdepth2rgb, Rrgb2depth, trgb2depth);
                        hasRgb2Depth = true;
                    }
                }
                if(hasRgb2Depth) {
                    RcwDepth = Rrgb2depth * RcwRgb;
                    tCwDepth = Rrgb2depth * tCwRgb + trgb2depth;
                }
            }

            ExtrinsicCamToWorld rgbPose;
            invertRigid(RcwRgb, tCwRgb, rgbPose.R, rgbPose.t);
            rgbPose.valid = true;
            rgbExtrinsicsCamToWorld_[camId] = rgbPose;

            ExtrinsicCamToWorld depthPose;
            invertRigid(RcwDepth, tCwDepth, depthPose.R, depthPose.t);
            depthPose.valid = true;
            depthExtrinsicsCamToWorld_[camId] = depthPose;
        }

        cJSON_Delete(root);
    }

    cv::Mat renderUnifiedPointCloud(const std::unordered_map<int, CachedFrameBundle> &frames, const InteractiveViewState &viewState) {
        cv::Mat canvas(viewState.height, viewState.width, CV_8UC3, cv::Scalar(0, 0, 0));
        std::vector<float> zbuf(static_cast<size_t>(viewState.width) * static_cast<size_t>(viewState.height), std::numeric_limits<float>::infinity());
        const GtInferenceResult gtResult = showGtJoints_ ? gtWorker_.latestResult() : GtInferenceResult{};
        const EgoAprilTagResult egoTagResult = showEgoAprilTags_ ? egoTagWorker_.latestResult() : EgoAprilTagResult{};
        ExtrinsicHealthSampleResult extrinsicResult = showExtrinsicHealthOverlay_ ? latestExtrinsicHealthResult() : ExtrinsicHealthSampleResult{};
        const bool showExtrinsicResult = showExtrinsicHealthOverlay_ && extrinsicResult.valid && !extrinsicResult.tags.empty();
        std::unordered_set<std::string> extrinsicSingleTagCameraIds;
        if(showExtrinsicResult && extrinsicHealthVizMode_ == ExtrinsicHealthVizMode::SingleTag) {
            const int tagId = selectedExtrinsicTagId(extrinsicResult);
            auto itTag = extrinsicResult.tags.find(tagId);
            if(itTag != extrinsicResult.tags.end()) {
                for(const auto &obs : itTag->second.observations) {
                    if(!obs.cameraId.empty()) {
                        extrinsicSingleTagCameraIds.insert(obs.cameraId);
                    }
                }
            }
        }

        cv::Vec3f right, up, forward, camPos;
        computeCameraBasis(viewState, right, up, forward, camPos);

        const float fx = 900.0f;
        const float fy = 900.0f;
        const float cx = static_cast<float>(viewState.width) * 0.5f;
        const float cy = static_cast<float>(viewState.height) * 0.5f;

        const std::vector<cv::Vec3b> palette = { cv::Vec3b(0, 80, 255), cv::Vec3b(0, 255, 80), cv::Vec3b(255, 80, 0), cv::Vec3b(255, 255, 0),
                                                 cv::Vec3b(255, 0, 255), cv::Vec3b(0, 255, 255) };

        const float spacingM = 0.25f;

        for(const auto &kv: frames) {
            const int deviceIndex = kv.first;
            const auto &cached = kv.second;
            if(!cached.depth) {
                continue;
            }
            if(!isCameraEnabled(deviceIndex)) {
                continue;
            }
            const bool extrinsicSingleTagMode = showExtrinsicResult
                                             && extrinsicHealthVizMode_ == ExtrinsicHealthVizMode::SingleTag;
            if(extrinsicSingleTagMode) {
                if(cached.camIndex.empty() || !containsCamKeyVariant(extrinsicSingleTagCameraIds, cached.camIndex)) {
                    continue;
                }
            }

            std::shared_ptr<ob::Frame> depthForPc = cached.depth;
            if(cfg_.filters.confThreshold > 0.0 && cached.irLeft && cached.irRight) {
                depthForPc = applyIrConfidenceMaskToDepth(cached.depth, cached.irLeft, cached.irRight, cfg_.filters.confThreshold);
            }
            auto itDepthFilters = depthFilterChains_.find(deviceIndex);
            if(itDepthFilters == depthFilterChains_.end()) {
                itDepthFilters = depthFilterChains_.emplace(deviceIndex, OrbbecDepthFilterChain{}).first;
            }
            depthForPc = refineDepthFrameForPointCloud(depthForPc, itDepthFilters->second, 0.2f, cfg_.maxDepth, cfg_.filters);
            if(!depthForPc) {
                continue;
            }

            if(cfg_.filters.smoothThresholdM > 0.0) {
                if(depthForPc == cached.depth) {
                    try {
                        depthForPc = ob::FrameFactory::createFrameFromOtherFrame(depthForPc, true);
                    }
                    catch(...) {
                        continue;
                    }
                }
                applyEdgeSmoothing(depthForPc, cfg_.filters.smoothThresholdM);
            }

            auto it = pointCloudFilters_.find(deviceIndex);
            if(it == pointCloudFilters_.end()) {
                auto f = std::make_shared<ob::PointCloudFilter>();
                f->setCreatePointFormat(OB_FORMAT_POINT);
                f->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
                f->setDecimationFactor(cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 1);
                it = pointCloudFilters_.emplace(deviceIndex, std::move(f)).first;
            }

            std::shared_ptr<ob::Frame> pcFrame;
            try {
                pcFrame = it->second->process(depthForPc);
            }
            catch(...) {
                continue;
            }
            if(!pcFrame) {
                continue;
            }

            auto pointsFrame = pcFrame->as<ob::PointsFrame>();
            if(!pointsFrame) {
                continue;
            }

            const float scaleMm = pointsFrame->getCoordinateValueScale();
            const auto *data = reinterpret_cast<const OBPoint *>(pointsFrame->data());
            const auto count = pointsFrame->dataSize() / sizeof(OBPoint);
            if(!data || count == 0) {
                continue;
            }

            const LiveDeskPlane deskPlane = cfg_.filters.deskCrop ? fitLiveDeskPlaneSampled(data, count, scaleMm) : LiveDeskPlane{};

            const cv::Vec3b color = cfg_.differentColor ? palette[static_cast<size_t>(deviceIndex) % palette.size()] : cv::Vec3b(255, 255, 255);
            const bool forceColorCloud = extrinsicSingleTagMode;
            const bool useColorCloud = (cfg_.colorfulCloudPoints || forceColorCloud) && !!cached.color;
            cv::Mat rgbImg;
            OBCameraParam rgbDepthParam{};
            bool rgbDepthParamValid = false;
            if(useColorCloud) {
                rgbImg = visualizeObFrame(cached.color);
                if(!rgbImg.empty()) {
                    auto itParam = rgbDepthParamsByDevice_.find(deviceIndex);
                    auto itValid = rgbDepthParamsValid_.find(deviceIndex);
                    if(itParam != rgbDepthParamsByDevice_.end() && itValid != rgbDepthParamsValid_.end() && itValid->second) {
                        rgbDepthParam = itParam->second;
                        rgbDepthParamValid = true;
                    }
                }
            }
            if(forceColorCloud && (!useColorCloud || !rgbDepthParamValid || rgbImg.empty())) {
                continue;
            }

            cv::Matx33f Rcam = cv::Matx33f::eye();
            cv::Vec3f tcam(deviceIndex * spacingM, 0.0f, 0.0f);
            if(!cached.camIndex.empty()) {
                const auto *itEx = findByCamKeyVariants(depthExtrinsicsCamToWorld_, cached.camIndex);
                if(itEx && itEx->valid) {
                    Rcam = itEx->R;
                    tcam = itEx->t;
                }
            }

            const float pointScaleM = scaleMm * 0.001f;
            for(size_t i = 0; i < count; i++) {
                const OBPoint &p = data[i];
                if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= 0.0f) {
                    continue;
                }
                const float xCamM = p.x * pointScaleM;
                const float yCamM = p.y * pointScaleM;
                const float zCamM = p.z * pointScaleM;
                if(!std::isfinite(xCamM) || !std::isfinite(yCamM) || !std::isfinite(zCamM) || zCamM <= 0.0f) {
                    continue;
                }
                if(isPointOnLiveDeskPlane(deskPlane, xCamM, yCamM, zCamM)) {
                    continue;
                }

                const cv::Vec3f pCam(xCamM, yCamM, zCamM);
                const cv::Vec3f pw = Rcam * pCam + tcam;
                const float x = pw[0];
                const float y = pw[1];
                const float z = pw[2];
                if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    continue;
                }

                cv::Vec3b pointColor = color;
                bool pointColorMapped = false;
                if(useColorCloud && rgbDepthParamValid) {
                    const float zM = zCamM;
                    if(zM > 0.0f && rgbDepthParam.depthIntrinsic.fx > 0.0f && rgbDepthParam.depthIntrinsic.fy > 0.0f) {
                        const float depthMm = zM * 1000.0f;
                        OBPoint2f src{};
                        src.x = (xCamM * rgbDepthParam.depthIntrinsic.fx / zM) + rgbDepthParam.depthIntrinsic.cx;
                        src.y = (yCamM * rgbDepthParam.depthIntrinsic.fy / zM) + rgbDepthParam.depthIntrinsic.cy;
                        OBPoint2f dst{};
                        bool ok = false;
                        try {
                            ok = ob::CoordinateTransformHelper::transformation2dto2d(src,
                                                                                     depthMm,
                                                                                     rgbDepthParam.depthIntrinsic,
                                                                                     rgbDepthParam.depthDistortion,
                                                                                     rgbDepthParam.rgbIntrinsic,
                                                                                     rgbDepthParam.rgbDistortion,
                                                                                     rgbDepthParam.transform,
                                                                                     &dst);
                        }
                        catch(...) {
                            ok = false;
                        }
                        if(ok) {
                            const int uc = static_cast<int>(std::lround(dst.x));
                            const int vc = static_cast<int>(std::lround(dst.y));
                            if(uc >= 0 && vc >= 0 && uc < rgbImg.cols && vc < rgbImg.rows) {
                                pointColor = rgbImg.at<cv::Vec3b>(vc, uc);
                                pointColorMapped = true;
                            }
                        }
                    }
                }
                if(useColorCloud && rgbDepthParamValid && !pointColorMapped) {
                    continue;
                }

                const cv::Vec3f v = pw - camPos;
                const float xc = v.dot(right);
                const float yc = v.dot(up);
                const float zc = v.dot(forward);
                if(zc <= 0.05f) {
                    continue;
                }

                const int u = static_cast<int>(fx * (xc / zc) + cx);
                const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
                if(u < 0 || u >= viewState.width || vpx < 0 || vpx >= viewState.height) {
                    continue;
                }

                const size_t idx = static_cast<size_t>(vpx) * static_cast<size_t>(viewState.width) + static_cast<size_t>(u);
                if(zc < zbuf[idx]) {
                    zbuf[idx] = zc;
                    canvas.at<cv::Vec3b>(vpx, u) = pointColor;
                }
            }
        }

        if(showGtJoints_ && !gtResult.hands.empty()) {
            drawGtJointsOnCanvas(canvas, zbuf, gtResult, right, up, forward, camPos, fx, fy, cx, cy);
        }
        if(showEgoAprilTags_ && !egoTagResult.lines.empty()) {
            drawEgoAprilTagsOnCanvas(canvas, egoTagResult, right, up, forward, camPos, fx, fy, cx, cy);
        }
        if(showExtrinsicResult) {
            drawExtrinsicHealthOnCanvas(canvas, extrinsicResult, right, up, forward, camPos, fx, fy, cx, cy);
        }

        cv::putText(canvas, "LMB: rotate  RMB: pan  Wheel/Ctrl+/-: zoom", cv::Point(12, 28), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
        int statusY = 54;
        if(showGtJoints_) {
            cv::putText(canvas, buildGtStatusLine(), cv::Point(12, statusY), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            statusY += 26;
        }
        if(showEgoAprilTags_) {
            cv::putText(canvas, buildEgoAprilTagStatusLine(), cv::Point(12, statusY), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            statusY += 26;
        }
        if(showExtrinsicHealthOverlay_) {
            cv::putText(canvas, buildExtrinsicHealthStatusLine(), cv::Point(12, statusY), cv::FONT_HERSHEY_DUPLEX, 0.52,
                        showExtrinsicResult ? cv::Scalar(230, 230, 230) : cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
        }
        return canvas;
    }

    bool drawLeftPanel(cv::Mat &canvas, CvMouseState &ms) {
        cv::rectangle(canvas, layout_.camsRect, cv::Scalar(16, 16, 16), cv::FILLED);
        cv::rectangle(canvas, layout_.camsRect, cv::Scalar(60, 60, 60), 1);

        if(imageType_ != ImageType::RGB) {
            imageType_ = ImageType::RGB;
            imageScrollY_ = 0;
        }
        cv::putText(canvas, "RGB", cv::Point(layout_.camsRect.x + 10, layout_.camsRect.y + 24), cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        const int bx = layout_.camsRect.x + 8;
        int       by = layout_.camsRect.y + 34;
        const int bw = layout_.camsRect.width - 16;
        const int bh = 28;

        bool typeChanged = false;
        auto setRgbSource = [&](RgbImageSource source) {
            if(rgbImageSource_ != source) {
                rgbImageSource_ = source;
                imageType_ = ImageType::RGB;
                imageScrollY_ = 0;
                typeChanged = true;
                if(rgbImageSource_ == RgbImageSource::Pico) {
                    picoRgbStatusLine_ = "Starting PICO RGB preview";
                }
                else {
                    clearPicoRgbPreview();
                }
                stopEgoPreviewSessionIfUnused();
            }
        };

        if(uiButton(canvas, cv::Rect(bx, by, bw, bh), rgbImageSource_ == RgbImageSource::Orbbec ? "Orbbec*" : "Orbbec", ms)) {
            setRgbSource(RgbImageSource::Orbbec);
        }
        by += bh + 6;
        if(uiButton(canvas, cv::Rect(bx, by, bw, bh), rgbImageSource_ == RgbImageSource::Pico ? "PICO*" : "PICO", ms)) {
            setRgbSource(RgbImageSource::Pico);
        }
        by += bh + 12;
        cv::putText(canvas, "Overlay", cv::Point(layout_.camsRect.x + 10, by + 18), cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        by += 26;

        {
            const cv::Rect row(bx, by, bw, 30);
            if(row.y + row.height <= layout_.camsRect.y + layout_.camsRect.height) {
                if(uiCheckbox(canvas, row, showGtJoints_, "GT hand", ms)) {
                    setGtVisualizationEnabled(!showGtJoints_);
                    typeChanged = true;
                }
            }
            by += 34;
        }
        {
            const cv::Rect row(bx, by, bw, 30);
            if(row.y + row.height <= layout_.camsRect.y + layout_.camsRect.height) {
                if(uiCheckbox(canvas, row, showEgoAprilTags_, "PICO tags", ms)) {
                    setEgoAprilTagOverlayEnabled(!showEgoAprilTags_);
                    typeChanged = true;
                }
            }
            by += 34;
        }
        if(showEgoAprilTags_ && by + 20 <= layout_.camsRect.y + layout_.camsRect.height) {
            const std::string status = ellipsizeTextToWidth(buildEgoAprilTagStatusLine(),
                                                            bw - 8,
                                                            cv::FONT_HERSHEY_DUPLEX,
                                                            0.42,
                                                            1);
            if(!status.empty()) {
                cv::putText(canvas, status, cv::Point(bx + 4, by + 16), cv::FONT_HERSHEY_DUPLEX, 0.42,
                            cv::Scalar(185, 220, 255), 1, cv::LINE_AA);
            }
            by += 26;
        }

        by += 14;
        if(by + 22 <= layout_.camsRect.y + layout_.camsRect.height) {
            cv::putText(canvas, "Extrinsic", cv::Point(layout_.camsRect.x + 10, by + 20), cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        by += 28;
        {
            const cv::Rect row(bx, by, bw, 32);
            if(row.y + row.height <= layout_.camsRect.y + layout_.camsRect.height) {
                const std::string label = isExtrinsicHealthRunning() ? "Checking..." : "Sample";
                if(uiButton(canvas, row, label, ms)) {
                    requestExtrinsicHealthCheck();
                    typeChanged = true;
                }
            }
            by += 38;
        }
        {
            const std::string status = ellipsizeTextToWidth(buildExtrinsicHealthStatusLine(),
                                                            bw - 8,
                                                            cv::FONT_HERSHEY_DUPLEX,
                                                            0.42,
                                                            1);
            if(!status.empty() && by + 20 <= layout_.camsRect.y + layout_.camsRect.height) {
                cv::putText(canvas, status, cv::Point(bx + 4, by + 16), cv::FONT_HERSHEY_DUPLEX, 0.42,
                            cv::Scalar(185, 220, 255), 1, cv::LINE_AA);
                by += 24;
            }
        }
        {
            const cv::Rect row(bx, by, bw, 30);
            if(row.y + row.height <= layout_.camsRect.y + layout_.camsRect.height) {
                if(uiCheckbox(canvas, row, showExtrinsicHealthOverlay_, "Show", ms)) {
                    showExtrinsicHealthOverlay_ = !showExtrinsicHealthOverlay_;
                    if(showExtrinsicHealthOverlay_) {
                        extrinsicHealthForceColor_ = true;
                    }
                    typeChanged = true;
                }
            }
            by += 34;
        }
        if(showExtrinsicHealthOverlay_) {
            const int halfW = (bw - 8) / 2;
            const cv::Rect allBtn(bx, by, halfW, 30);
            const cv::Rect oneBtn(bx + halfW + 8, by, bw - halfW - 8, 30);
            if(oneBtn.y + oneBtn.height <= layout_.camsRect.y + layout_.camsRect.height) {
                if(uiButton(canvas, allBtn, extrinsicHealthVizMode_ == ExtrinsicHealthVizMode::AllTags ? "All*" : "All", ms)) {
                    extrinsicHealthVizMode_ = ExtrinsicHealthVizMode::AllTags;
                    typeChanged = true;
                }
                if(uiButton(canvas, oneBtn, extrinsicHealthVizMode_ == ExtrinsicHealthVizMode::SingleTag ? "One*" : "One", ms)) {
                    extrinsicHealthVizMode_ = ExtrinsicHealthVizMode::SingleTag;
                    typeChanged = true;
                }
            }
            by += 36;
            {
                const cv::Rect row(bx, by, bw, 30);
                if(row.y + row.height <= layout_.camsRect.y + layout_.camsRect.height) {
                    if(uiCheckbox(canvas, row, showExtrinsicCameraTagPoses_, "Cam poses", ms)) {
                        showExtrinsicCameraTagPoses_ = !showExtrinsicCameraTagPoses_;
                        typeChanged = true;
                    }
                }
                by += 34;
            }
            {
                const cv::Rect row(bx, by, bw, 30);
                if(row.y + row.height <= layout_.camsRect.y + layout_.camsRect.height) {
                    if(uiCheckbox(canvas, row, showExtrinsicFusedTagPoses_, "Fused poses", ms)) {
                        showExtrinsicFusedTagPoses_ = !showExtrinsicFusedTagPoses_;
                        typeChanged = true;
                    }
                }
                by += 34;
            }
            if(showExtrinsicCameraTagPoses_) {
                for(size_t i = 0; i < devices_.size() && i < extrinsicCameraTagPoseVisible_.size(); ++i) {
                    const cv::Rect row(bx + 16, by, std::max(20, bw - 16), 28);
                    if(row.y + row.height > layout_.camsRect.y + layout_.camsRect.height) {
                        break;
                    }
                    const bool visible = extrinsicCameraTagPoseVisible_[i] != 0;
                    const std::string label = "Pose " + devices_[i].cfg.index;
                    if(uiCheckbox(canvas, row, visible, label, ms)) {
                        extrinsicCameraTagPoseVisible_[i] = visible ? 0 : 1;
                        typeChanged = true;
                    }
                    by += 30;
                }
                by += 4;
            }
            if(extrinsicHealthVizMode_ == ExtrinsicHealthVizMode::SingleTag) {
                auto result = latestExtrinsicHealthResult();
                const int tagId = selectedExtrinsicTagId(result);
                const std::string tagLabel = tagId >= 0 ? ("Tag " + std::to_string(tagId)) : "No Tag";
                const int sideW = 54;
                const cv::Rect prevBtn(bx, by, sideW, 30);
                const cv::Rect labelRect(bx + sideW + 6, by, bw - 2 * sideW - 12, 30);
                const cv::Rect nextBtn(bx + bw - sideW, by, sideW, 30);
                if(nextBtn.y + nextBtn.height <= layout_.camsRect.y + layout_.camsRect.height) {
                    if(uiButton(canvas, prevBtn, "<", ms)) {
                        stepSelectedExtrinsicTag(-1);
                        typeChanged = true;
                    }
                    cv::rectangle(canvas, labelRect, cv::Scalar(28, 28, 28), cv::FILLED);
                    cv::rectangle(canvas, labelRect, cv::Scalar(100, 100, 100), 1);
                    cv::putText(canvas, tagLabel, cv::Point(labelRect.x + 8, labelRect.y + 20), cv::FONT_HERSHEY_DUPLEX, 0.5,
                                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
                    if(uiButton(canvas, nextBtn, ">", ms)) {
                        stepSelectedExtrinsicTag(1);
                        typeChanged = true;
                    }
                }
                by += 36;
            }
        }

        by += 18;
        cv::putText(canvas, "Cameras", cv::Point(layout_.camsRect.x + 10, by + 20), cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        by += 28;

        for(size_t i = 0; i < devices_.size(); i++) {
            const auto &rt = devices_[i];
            const std::string label = ellipsizeTextToWidth(rt.cfg.index.empty() ? rt.cfg.sn : rt.cfg.index,
                                                           bw - 28,
                                                           cv::FONT_HERSHEY_DUPLEX,
                                                           0.55,
                                                           1);
            const cv::Rect row(bx, by, bw, 26);
            if(row.y + row.height > layout_.camsRect.y + layout_.camsRect.height) {
                break;
            }
            const bool enabled = isCameraEnabled(static_cast<int>(i));
            if(uiCheckbox(canvas, row, enabled, label, ms)) {
                const bool next = !enabled;
                setCameraEnabled(static_cast<int>(i), next);
                restartPipeline(static_cast<int>(i), next);
            }
            by += 29;
        }

        if(!fisheyeLabels_.empty()) {
            by += 14;
            if(by + 22 < layout_.camsRect.y + layout_.camsRect.height) {
                cv::putText(canvas, "Fisheyes", cv::Point(layout_.camsRect.x + 10, by + 20), cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
            by += 28;
            for(size_t i = 0; i < fisheyeLabels_.size(); ++i) {
                const cv::Rect row(bx, by, bw, 26);
                if(row.y + row.height > layout_.camsRect.y + layout_.camsRect.height) {
                    break;
                }
                const bool visible = i < fisheyeVisible_.size() && fisheyeVisible_[i] != 0;
                const std::string label = ellipsizeTextToWidth(fisheyeLabels_[i],
                                                               bw - 28,
                                                               cv::FONT_HERSHEY_DUPLEX,
                                                               0.55,
                                                               1);
                if(uiCheckbox(canvas, row, visible, label, ms)) {
                    if(i < fisheyeVisible_.size()) {
                        fisheyeVisible_[i] = visible ? 0 : 1;
                    }
                }
                by += 29;
            }
        }
        else if(!fisheyeStatusLine_.empty() && by + 24 < layout_.camsRect.y + layout_.camsRect.height) {
            cv::putText(canvas, fisheyeStatusLine_, cv::Point(layout_.camsRect.x + 12, by + 22), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
        }
        return typeChanged;
    }

    bool drawImagePanel(cv::Mat &canvas, CvMouseState &ms) {
        cv::rectangle(canvas, layout_.imgRect, cv::Scalar(16, 16, 16), cv::FILLED);
        cv::rectangle(canvas, layout_.imgRect, cv::Scalar(60, 60, 60), 1);

        const bool hover = layout_.imgRect.contains(cv::Point(ms.x, ms.y));
        if(hover && ms.wheelDelta != 0) {
            imageScrollY_ += (ms.wheelDelta > 0) ? -60 : 60;
            ms.wheelDelta = 0;
        }

        std::string title = "Images: ";
        if(imageType_ == ImageType::Depth) {
            title += "Depth";
        }
        else if(imageType_ == ImageType::RGB) {
            title += rgbImageSource_ == RgbImageSource::Pico ? "PICO RGB" : "Orbbec RGB";
        }
        else if(imageType_ == ImageType::IRLeft) {
            title += "IR Left";
        }
        else {
            title += "IR Right";
        }
        cv::putText(canvas, title, cv::Point(layout_.imgRect.x + 12, layout_.imgRect.y + 28), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        const int contentTop = layout_.imgRect.y + 40;
        int y = contentTop - imageScrollY_;
        int totalH = 0;

        if(imageType_ == ImageType::RGB && rgbImageSource_ == RgbImageSource::Pico) {
            const int targetW = std::max(1, layout_.imgRect.width - 20);
            const cv::Mat &img = latestPicoRgbFrame_;
            if(!img.empty()) {
                const int targetH = std::max(10, static_cast<int>(static_cast<double>(img.rows) * (static_cast<double>(targetW) / static_cast<double>(img.cols))));
                cv::Mat resized;
                cv::resize(img, resized, cv::Size(targetW, targetH));
                totalH += 24 + targetH + 16;
                if(y + 24 + targetH >= contentTop && y <= layout_.imgRect.y + layout_.imgRect.height) {
                    const std::string label = latestPicoRgbVideoFrameIndex_ >= 0
                                            ? ("PICO ego RGB frame " + std::to_string(latestPicoRgbVideoFrameIndex_))
                                            : "PICO ego RGB";
                    cv::putText(canvas, label, cv::Point(layout_.imgRect.x + 12, y + 16), cv::FONT_HERSHEY_DUPLEX, 0.55,
                                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
                    const int imgY = y + 24;
                    const int imgX = layout_.imgRect.x + 10;
                    const cv::Rect roi(imgX, imgY, targetW, targetH);
                    if(roi.y >= layout_.imgRect.y && roi.y + roi.height <= layout_.imgRect.y + layout_.imgRect.height) {
                        resized.copyTo(canvas(roi));
                    }
                    cv::rectangle(canvas, roi, cv::Scalar(80, 80, 80), 1);
                    if(!picoRgbStatusLine_.empty() && roi.y + roi.height + 18 <= layout_.imgRect.y + layout_.imgRect.height) {
                        cv::putText(canvas, picoRgbStatusLine_, cv::Point(layout_.imgRect.x + 12, roi.y + roi.height + 16),
                                    cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(185, 220, 255), 1, cv::LINE_AA);
                    }
                }
            }
            else {
                const std::string status = picoRgbStatusLine_.empty() ? "PICO RGB waiting for frames" : picoRgbStatusLine_;
                totalH += 80;
                cv::putText(canvas, "PICO ego RGB", cv::Point(layout_.imgRect.x + 12, y + 22), cv::FONT_HERSHEY_DUPLEX, 0.62,
                            cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
                cv::putText(canvas, status, cv::Point(layout_.imgRect.x + 12, y + 54), cv::FONT_HERSHEY_DUPLEX, 0.5,
                            cv::Scalar(185, 220, 255), 1, cv::LINE_AA);
            }
            const int maxScroll = std::max(0, totalH - (layout_.imgRect.height - 40));
            imageScrollY_ = std::max(0, std::min(maxScroll, imageScrollY_));
            return false;
        }

        auto frames = snapshotFrames();
        if(imageType_ == ImageType::RGB && rgbImageSource_ == RgbImageSource::Orbbec) {
            std::vector<int> visibleDeviceIndices;
            visibleDeviceIndices.reserve(devices_.size());
            for(size_t i = 0; i < devices_.size(); i++) {
                if(isCameraEnabled(static_cast<int>(i))) {
                    visibleDeviceIndices.push_back(static_cast<int>(i));
                }
            }

            const int gap = 8;
            const int gridX = layout_.imgRect.x + 8;
            const int gridW = std::max(1, layout_.imgRect.width - 16);
            const int gridBottom = layout_.imgRect.y + layout_.imgRect.height - 8;
            const int gridH = std::max(1, gridBottom - contentTop);
            const int cols = 1;
            const int visibleRows = 6;
            const int tileW = std::max(1, (gridW - gap * (cols - 1)) / cols);
            const int tileH = std::max(70, (gridH - gap * (visibleRows - 1)) / visibleRows);
            const int rowsNeeded = std::max(visibleRows, static_cast<int>((visibleDeviceIndices.size() + cols - 1) / cols));
            totalH = rowsNeeded * tileH + std::max(0, rowsNeeded - 1) * gap;
            y = contentTop - imageScrollY_;

            for(size_t slot = 0; slot < visibleDeviceIndices.size(); ++slot) {
                const int deviceIndex = visibleDeviceIndices[slot];
                const int row = static_cast<int>(slot) / cols;
                const int col = static_cast<int>(slot) % cols;
                const int tileX = gridX + col * (tileW + gap);
                const int tileY = y + row * (tileH + gap);
                const cv::Rect tile(tileX, tileY, tileW, tileH);
                if(tile.y + tile.height < contentTop || tile.y > gridBottom) {
                    continue;
                }

                cv::rectangle(canvas, tile, cv::Scalar(24, 24, 24), cv::FILLED);
                cv::rectangle(canvas, tile, cv::Scalar(78, 78, 78), 1);
                const auto &rt = devices_[static_cast<size_t>(deviceIndex)];
                const std::string label = ellipsizeTextToWidth(rt.cfg.index + " " + rt.cfg.sn,
                                                               std::max(1, tile.width - 10),
                                                               cv::FONT_HERSHEY_DUPLEX,
                                                               0.42,
                                                               1);
                cv::putText(canvas, label, cv::Point(tile.x + 5, tile.y + 15), cv::FONT_HERSHEY_DUPLEX, 0.42,
                            cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

                cv::Mat img;
                auto it = frames.find(deviceIndex);
                if(it != frames.end()) {
                    img = visualizeObFrame(it->second.color);
                }
                const cv::Rect imgArea(tile.x + 3, tile.y + 20, std::max(1, tile.width - 6), std::max(1, tile.height - 23));
                if(!img.empty()) {
                    double scale = std::min(static_cast<double>(imgArea.width) / static_cast<double>(img.cols),
                                            static_cast<double>(imgArea.height) / static_cast<double>(img.rows));
                    if(!(scale > 0.0)) {
                        scale = 1.0;
                    }
                    const int drawW = std::max(1, std::min(imgArea.width, static_cast<int>(std::lround(static_cast<double>(img.cols) * scale))));
                    const int drawH = std::max(1, std::min(imgArea.height, static_cast<int>(std::lround(static_cast<double>(img.rows) * scale))));
                    const cv::Rect roi(imgArea.x + (imgArea.width - drawW) / 2,
                                       imgArea.y + (imgArea.height - drawH) / 2,
                                       drawW,
                                       drawH);
                    if(roi.x >= layout_.imgRect.x && roi.y >= layout_.imgRect.y
                       && roi.x + roi.width <= layout_.imgRect.x + layout_.imgRect.width
                       && roi.y + roi.height <= layout_.imgRect.y + layout_.imgRect.height) {
                        cv::Mat resized;
                        cv::resize(img, resized, cv::Size(drawW, drawH), 0.0, 0.0, cv::INTER_AREA);
                        resized.copyTo(canvas(roi));
                    }
                }
                else {
                    cv::putText(canvas, "waiting", cv::Point(imgArea.x + 8, imgArea.y + std::max(22, imgArea.height / 2)),
                                cv::FONT_HERSHEY_DUPLEX, 0.46, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
                }
            }

            const int maxScroll = std::max(0, totalH - gridH);
            imageScrollY_ = std::max(0, std::min(maxScroll, imageScrollY_));
            return false;
        }

        const int targetW = layout_.imgRect.width - 20;
        for(size_t i = 0; i < devices_.size(); i++) {
            if(!isCameraEnabled(static_cast<int>(i))) {
                continue;
            }
            auto it = frames.find(static_cast<int>(i));
            if(it == frames.end()) {
                continue;
            }
            const auto &b = it->second;

            std::shared_ptr<const ob::Frame> f;
            if(imageType_ == ImageType::Depth) {
                f = b.depth;
            }
            else if(imageType_ == ImageType::RGB) {
                f = b.color;
            }
            else if(imageType_ == ImageType::IRLeft) {
                f = b.irLeft ? b.irLeft : b.ir;
            }
            else {
                f = b.irRight ? b.irRight : b.ir;
            }

            cv::Mat img = visualizeObFrame(f);
            if(img.empty()) {
                continue;
            }
            const int targetH = std::max(10, static_cast<int>(static_cast<double>(img.rows) * (static_cast<double>(targetW) / static_cast<double>(img.cols))));
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(targetW, targetH));

            totalH += 24 + targetH + 16;
            if(y + 24 + targetH < contentTop || y > layout_.imgRect.y + layout_.imgRect.height) {
                y += 24 + targetH + 16;
                continue;
            }

            cv::putText(canvas, b.camIndex + "  " + b.sn, cv::Point(layout_.imgRect.x + 12, y + 16), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
            const int imgY = y + 24;
            const int imgX = layout_.imgRect.x + 10;
            const cv::Rect roi(imgX, imgY, targetW, targetH);
            if(roi.y >= layout_.imgRect.y && roi.y + roi.height <= layout_.imgRect.y + layout_.imgRect.height) {
                resized.copyTo(canvas(roi));
            }
            cv::rectangle(canvas, roi, cv::Scalar(80, 80, 80), 1);
            y += 24 + targetH + 16;
        }

        const int maxScroll = std::max(0, totalH - (layout_.imgRect.height - 40));
        imageScrollY_ = std::max(0, std::min(maxScroll, imageScrollY_));
        return false;
    }

    bool drawControls(cv::Mat &canvas, CvMouseState &ms, InteractiveViewState &viewState, InteractiveExit &out) {
        cv::rectangle(canvas, layout_.ctlRect, cv::Scalar(16, 16, 16), cv::FILLED);
        cv::rectangle(canvas, layout_.ctlRect, cv::Scalar(60, 60, 60), 1);

        const int y = layout_.ctlRect.y + 20;
        cv::Rect b1(layout_.ctlRect.x + 10, y, 220, 60);
        cv::Rect b2(layout_.ctlRect.x + 240, y, 260, 60);
        cv::Rect b3(layout_.ctlRect.x + 510, y, 220, 60);
        if(uiButton(canvas, b1, "Back to Menu", ms)) {
            out = InteractiveExit::ReturnMenu;
            return true;
        }
        if(cfg_.demo.active) {
            if(uiButton(canvas, b2, "Enter Collection", ms)) {
                out = InteractiveExit::StartCollection;
                return true;
            }
        }
        else {
            if(uiButton(canvas, b2, "Back to Config", ms)) {
                out = InteractiveExit::ReturnConfig;
                return true;
            }
        }
        if(uiButton(canvas, b3, "Reset View", ms)) {
            viewState.resetView();
        }

        cv::putText(canvas, "LMB rotate | RMB pan | Wheel/Ctrl+/- zoom", cv::Point(layout_.ctlRect.x + 760, y + 18), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        cv::putText(canvas, buildGtStatusLine(), cv::Point(layout_.ctlRect.x + 760, y + 42), cv::FONT_HERSHEY_DUPLEX, 0.52,
                    showGtJoints_ ? cv::Scalar(230, 230, 230) : cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
        cv::putText(canvas, buildEgoAprilTagStatusLine(), cv::Point(layout_.ctlRect.x + 760, y + 64), cv::FONT_HERSHEY_DUPLEX, 0.52,
                    showEgoAprilTags_ ? cv::Scalar(230, 230, 230) : cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
        cv::putText(canvas, buildExtrinsicHealthStatusLine(), cv::Point(layout_.ctlRect.x + 760, y + 86), cv::FONT_HERSHEY_DUPLEX, 0.48,
                    showExtrinsicHealthOverlay_ ? cv::Scalar(230, 230, 230) : cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
        return false;
    }

private:
    static constexpr int gtWorkerMaxImageSide_ = 384;
    static constexpr int egoTagWorkerMaxImageSide_ = 640;
    static constexpr size_t kMaxAlignedFrameQueueSize_ = 8;

    AppConfig cfg_;
    const std::atomic_bool *cancel_;
    ob::Context ctx_;
    std::vector<DeviceRuntime> devices_;

    mutable std::mutex stateMtx_;
    std::vector<uint8_t> cameraEnabled_;

    std::mutex framesMtx_;
    std::unordered_map<int, CachedFrameBundle> frames_;
    std::unordered_map<int, std::deque<CachedFrameBundle>> frameQueues_;

    uint64_t viewerIntervalUs_ = 33333;
    std::unordered_map<int, std::shared_ptr<ob::PointCloudFilter>> pointCloudFilters_;
    std::unordered_map<int, OrbbecDepthFilterChain> depthFilterChains_;
    std::unordered_map<int, OBCameraParam> rgbDepthParamsByDevice_;
    std::unordered_map<int, bool> rgbDepthParamsValid_;
    std::unordered_map<std::string, ExtrinsicCamToWorld> depthExtrinsicsCamToWorld_;
    std::unordered_map<std::string, ExtrinsicCamToWorld> rgbExtrinsicsCamToWorld_;
    LiveGtJointWorker gtWorker_;
    LiveEgoAprilTagWorker egoTagWorker_;
    EgoRecorder ownedEgoRecorder_;
    EgoRecorder &egoRecorder_;
    bool ownsEgoRecorder_ = true;
    fs::path egoPreviewEpisodeDir_;
    fs::path egoVideoPath_;
    fs::path egoCameraParamsPath_;
    std::optional<EgoFrame> latestEgoFrame_;
    std::string egoTagStatusLine_ = "PICO tags off";
    FisheyeRecorder fisheyeRecorder_;
    std::vector<uint8_t> fisheyeVisible_;
    std::vector<std::string> fisheyeLabels_;
    std::string fisheyeStatusLine_;

    Layout layout_;
    std::unordered_map<int, StreamMode> pipelineModeByDevice_;
    std::vector<int> frameCountByDevice_;
    bool clockSyncEnabled_ = false;

    ImageType imageType_ = ImageType::RGB;
    RgbImageSource rgbImageSource_ = RgbImageSource::Orbbec;
    int imageScrollY_ = 0;
    bool showGtJoints_ = false;
    bool showEgoAprilTags_ = false;
    bool showExtrinsicHealthOverlay_ = false;
    bool showExtrinsicCameraTagPoses_ = true;
    bool showExtrinsicFusedTagPoses_ = true;
    std::vector<uint8_t> extrinsicCameraTagPoseVisible_;
    bool extrinsicHealthForceColor_ = false;
    bool extrinsicHealthPending_ = false;
    ExtrinsicHealthVizMode extrinsicHealthVizMode_ = ExtrinsicHealthVizMode::AllTags;
    int selectedExtrinsicTagId_ = -1;
    mutable std::mutex extrinsicHealthMtx_;
    std::thread extrinsicHealthThread_;
    bool extrinsicHealthRunning_ = false;
    std::string extrinsicHealthStatusLine_ = "Extrinsic check idle";
    ExtrinsicHealthSampleResult latestExtrinsicHealthResult_;
    uint64_t lastEgoTagSubmitUs_ = 0;
    uint64_t lastEgoSessionAttemptUs_ = 0;
    LivePicoRgbFrameSource picoRgbFrameSource_;
    cv::Mat latestPicoRgbFrame_;
    int latestPicoRgbVideoFrameIndex_ = -1;
    std::string picoRgbStatusLine_ = "PICO RGB off";
};

InteractiveExit run_interactive_visualization(const AppConfig &cfg,
                                              const std::atomic_bool *cancel,
                                              EgoRecorder *sharedEgoRecorder) {
    InteractiveVisualizationApp app(cfg, cancel, sharedEgoRecorder);
    return app.run();
}

}  // namespace sync_app
