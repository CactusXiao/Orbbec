#include "collection.hpp"
#include "fisheyes.hpp"

#include "utils/utils.hpp"

#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>

#if defined(__linux__)
#include <execinfo.h>
#include <unistd.h>
#endif

#if defined(__has_include)
#if __has_include(<opencv2/freetype.hpp>)
#define SYNC_COLLECTION_HAS_OPENCV_FREETYPE 1
#include <opencv2/freetype.hpp>
#endif
#endif

namespace sync_app {

namespace {

static std::atomic<const char *> g_collection_stage{"init"};

static inline void collectionSetStage(const char *s) {
    g_collection_stage.store(s, std::memory_order_relaxed);
}

static void collectionInstallCrashHandlerOnce() {
    static std::atomic<bool> installed{false};
    bool expected = false;
    if(!installed.compare_exchange_strong(expected, true)) {
        return;
    }

#if defined(__linux__)
    auto handler = +[](int sig) {
        const char *stage = g_collection_stage.load(std::memory_order_relaxed);
        char buf[512];
        int n = snprintf(buf, sizeof(buf), "\n[collection] signal=%d stage=%s\n", sig, stage ? stage : "(null)");
        if(n > 0) {
            (void)write(2, buf, static_cast<size_t>(n));
        }

        void *bt[64];
        const int sz = backtrace(bt, 64);
        backtrace_symbols_fd(bt, sz, 2);
        _exit(128 + sig);
    };
    std::signal(SIGSEGV, handler);
    std::signal(SIGABRT, handler);
#endif
}

struct CvMouseState {
    int  x          = 0;
    int  y          = 0;
    bool clicked    = false;
    int  clickX     = 0;
    int  clickY     = 0;
    int  wheelDelta = 0;
};

static void mouseThunk(int event, int x, int y, int flags, void *userdata) {
    (void)flags;
    auto *s = reinterpret_cast<CvMouseState *>(userdata);
    if(!s) {
        return;
    }
    s->x = x;
    s->y = y;
    if(event == cv::EVENT_LBUTTONDOWN) {
        s->clicked = true;
        s->clickX  = x;
        s->clickY  = y;
    }
    else if(event == cv::EVENT_MOUSEWHEEL) {
        s->wheelDelta += cv::getMouseWheelDelta(flags);
    }
}

struct FrameMouse {
    int  x          = 0;
    int  y          = 0;
    bool clicked    = false;
    int  clickX     = 0;
    int  clickY     = 0;
    int  wheelDelta = 0;
};

static FrameMouse beginFrame(CvMouseState &ms) {
    FrameMouse fm;
    fm.x          = ms.x;
    fm.y          = ms.y;
    fm.clicked    = ms.clicked;
    fm.clickX     = ms.clickX;
    fm.clickY     = ms.clickY;
    fm.wheelDelta = ms.wheelDelta;
    ms.clicked    = false;
    ms.wheelDelta = 0;
    return fm;
}

static bool uiButton(cv::Mat &img, const cv::Rect &r, const std::string &label, FrameMouse &ms) {
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar bg    = hover ? cv::Scalar(60, 60, 60) : cv::Scalar(40, 40, 40);
    cv::rectangle(img, r, bg, cv::FILLED);
    cv::rectangle(img, r, cv::Scalar(120, 120, 120), 1);
    cv::putText(img, label, cv::Point(r.x + 14, r.y + r.height / 2 + 7), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static bool uiCheckbox(cv::Mat &img, const cv::Rect &r, bool checked, const std::string &label, FrameMouse &ms) {
    const cv::Rect box(r.x, r.y + 7, 20, 20);
    const bool     hover = r.contains(cv::Point(ms.x, ms.y));
    cv::rectangle(img, box, checked ? cv::Scalar(80, 200, 80) : cv::Scalar(30, 30, 30), cv::FILLED);
    cv::rectangle(img, box, cv::Scalar(200, 200, 200), 1);
    cv::putText(img, label, cv::Point(r.x + 30, r.y + 22), cv::FONT_HERSHEY_DUPLEX, 0.65, hover ? cv::Scalar(255, 255, 255) : cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static bool uiTextField(cv::Mat &img, const cv::Rect &r, const std::string &label, std::string &value, bool active, FrameMouse &ms) {
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar border = active ? cv::Scalar(80, 200, 80) : (hover ? cv::Scalar(180, 180, 180) : cv::Scalar(120, 120, 120));
    cv::rectangle(img, r, cv::Scalar(30, 30, 30), cv::FILLED);
    cv::rectangle(img, r, border, 1);
    cv::putText(img, label, cv::Point(r.x, r.y - 6), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    cv::putText(img, value, cv::Point(r.x + 8, r.y + r.height - 10), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static void handleTextInput(std::string &fieldValue, int key) {
    if(key == 8 || key == 127) {
        if(!fieldValue.empty()) {
            fieldValue.pop_back();
        }
        return;
    }
    if(key == 13 || key == 10 || key == 27) {
        return;
    }
    if(key >= 32 && key <= 126) {
        fieldValue.push_back(static_cast<char>(key));
    }
}

static std::string readPipeCommand(const char *cmd) {
    if(!cmd || !*cmd) {
        return "";
    }
    std::string out;
    FILE *fp = popen(cmd, "r");
    if(!fp) {
        return out;
    }
    char buf[512];
    while(true) {
        const size_t n = fread(buf, 1, sizeof(buf), fp);
        if(n > 0) {
            out.append(buf, n);
        }
        if(n < sizeof(buf)) {
            break;
        }
    }
    pclose(fp);
    return out;
}

static bool writePipeCommand(const char *cmd, const std::string &text) {
    if(!cmd || !*cmd) {
        return false;
    }
    FILE *fp = popen(cmd, "w");
    if(!fp) {
        return false;
    }
    bool ok = true;
    if(!text.empty()) {
        ok = fwrite(text.data(), 1, text.size(), fp) == text.size();
    }
    return pclose(fp) == 0 && ok;
}

static bool getClipboardText(std::string &out) {
#if defined(__APPLE__)
    out = readPipeCommand("pbpaste 2>/dev/null");
    return true;
#else
    static const char *kReadCommands[] = {
        "wl-paste -n 2>/dev/null",
        "xclip -selection clipboard -out 2>/dev/null",
        "xsel --clipboard --output 2>/dev/null",
        nullptr
    };
    for(int i = 0; kReadCommands[i] != nullptr; ++i) {
        out = readPipeCommand(kReadCommands[i]);
        if(!out.empty()) {
            return true;
        }
    }
    out.clear();
    return false;
#endif
}

static bool setClipboardText(const std::string &text) {
#if defined(__APPLE__)
    return writePipeCommand("pbcopy 2>/dev/null", text);
#else
    static const char *kWriteCommands[] = {
        "wl-copy 2>/dev/null",
        "xclip -selection clipboard 2>/dev/null",
        "xsel --clipboard --input 2>/dev/null",
        nullptr
    };
    for(int i = 0; kWriteCommands[i] != nullptr; ++i) {
        if(writePipeCommand(kWriteCommands[i], text)) {
            return true;
        }
    }
    return false;
#endif
}

static void handleTextInputShortcut(std::string &fieldValue, int key, bool ctrlHeld) {
    const int baseKey = key & 0xFFFF;
    if(ctrlHeld) {
        if(baseKey == 'v' || baseKey == 'V' || key == 22) {
            std::string clip;
            if(getClipboardText(clip) && !clip.empty()) {
                fieldValue += clip;
            }
            return;
        }
        if(baseKey == 'c' || baseKey == 'C' || key == 3) {
            (void)setClipboardText(fieldValue);
            return;
        }
        if(baseKey == 'x' || baseKey == 'X' || key == 24) {
            if(setClipboardText(fieldValue)) {
                fieldValue.clear();
            }
            return;
        }
    }
    handleTextInput(fieldValue, baseKey);
}

static std::vector<std::string> wrapTextToWidth(const std::string &text, int maxWidthPx, int fontFace, double fontScale, int thickness) {
    std::vector<std::string> out;
    std::string s = text;
    if(s.empty() || maxWidthPx <= 0) {
        return out;
    }

    auto fits = [&](const std::string &t) {
        int baseline = 0;
        const auto sz = cv::getTextSize(t, fontFace, fontScale, thickness, &baseline);
        return sz.width <= maxWidthPx;
    };

    if(fits(s)) {
        out.push_back(std::move(s));
        return out;
    }

    size_t pos = 0;
    while(pos < s.size()) {
        size_t best = std::string::npos;
        size_t hi = s.size();
        size_t lo = pos + 1;
        while(lo <= hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const std::string sub = s.substr(pos, mid - pos);
            if(fits(sub)) {
                best = mid;
                lo = mid + 1;
            }
            else {
                if(mid == 0) {
                    break;
                }
                hi = mid - 1;
            }
        }
        if(best == std::string::npos || best <= pos) {
            out.push_back(s.substr(pos, 1));
            pos++;
            continue;
        }

        size_t cut = best;
        size_t space = s.rfind(' ', cut - 1);
        if(space != std::string::npos && space > pos) {
            cut = space + 1;
        }

        std::string line = s.substr(pos, cut - pos);
        while(!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        if(!line.empty()) {
            out.push_back(std::move(line));
        }
        pos = cut;
        while(pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
            pos++;
        }
    }
    return out;
}

// ---- UTF-8 / 中文文本渲染支持 ----
// OpenCV内置的Hershey字体仅支持ASCII字符，中文字符会显示为"?"。
// 当编译时检测到 opencv_freetype 模块，则使用FreeType2渲染中文；否则退回英文。

#ifdef SYNC_COLLECTION_HAS_OPENCV_FREETYPE
static cv::Ptr<cv::freetype::FreeType2> &getFt2() {
    static cv::Ptr<cv::freetype::FreeType2> ft2;
    return ft2;
}
static bool &getFt2Tried() {
    static bool tried = false;
    return tried;
}

static cv::Ptr<cv::freetype::FreeType2> tryLoadFt2Font(const fs::path &path) {
    try {
        auto ft = cv::freetype::createFreeType2();
        ft->loadFontData(path.string(), 0);
        std::cerr << "[collection] FreeType2 loaded: " << path << std::endl;
        return ft;
    }
    catch(...) {
        return nullptr;
    }
}

static cv::Ptr<cv::freetype::FreeType2> initFreeType2() {
    if(getFt2Tried()) {
        return getFt2();
    }
    getFt2Tried() = true;

    // Ubuntu 上通过 `fc-list :lang=zh family file | sort -u` 确认存在的中文字体。
    // 优先选择 Noto Sans CJK，其次回退到文泉驿 / Droid / Arphic。
    static const char *kFontPaths[] = {
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        nullptr
    };
    for(int i = 0; kFontPaths[i] != nullptr; ++i) {
        const fs::path path(kFontPaths[i]);
        if(fs::exists(path)) {
            if(auto ft = tryLoadFt2Font(path)) {
                getFt2() = ft;
                return getFt2();
            }
        }
    }
    std::cerr << "[collection] FreeType2: no known Ubuntu CJK font loaded." << std::endl;
    return nullptr;
}
#endif // SYNC_COLLECTION_HAS_OPENCV_FREETYPE

// 返回UTF-8字符串中的字符边界列表（每个元素为该字符的起始字节偏移及长度）
static std::vector<std::pair<size_t, size_t>> utf8CharBounds(const std::string &s) {
    std::vector<std::pair<size_t, size_t>> result;
    size_t i = 0;
    while(i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if((c & 0x80) == 0)        len = 1;
        else if((c & 0xE0) == 0xC0) len = 2;
        else if((c & 0xF0) == 0xE0) len = 3;
        else if((c & 0xF8) == 0xF0) len = 4;
        if(i + len > s.size()) break;
        result.emplace_back(i, len);
        i += len;
    }
    return result;
}

// 将UTF-8中文文本按宽度自动换行（fontHeight为FreeType渲染高度，单位px）
static std::vector<std::string> wrapTextUtf8(const std::string &text, int maxWidthPx, int fontHeight) {
    std::vector<std::string> out;
    if(text.empty() || maxWidthPx <= 0) {
        return out;
    }

#ifdef SYNC_COLLECTION_HAS_OPENCV_FREETYPE
    auto ft2 = initFreeType2();
    if(ft2) {
        // 使用FreeType2精确测量
        auto fits = [&](const std::string &t) -> bool {
            int baseline = 0;
            try {
                const auto sz = ft2->getTextSize(t, fontHeight, -1, &baseline);
                return sz.width <= maxWidthPx;
            }
            catch(...) {
                return true;
            }
        };
        if(fits(text)) {
            out.push_back(text);
            return out;
        }
        const auto chars = utf8CharBounds(text);
        std::string cur;
        for(const auto &cb : chars) {
            const std::string added = cur + text.substr(cb.first, cb.second);
            if(!cur.empty() && !fits(added)) {
                out.push_back(cur);
                cur = text.substr(cb.first, cb.second);
            }
            else {
                cur = added;
            }
        }
        if(!cur.empty()) {
            out.push_back(cur);
        }
        return out;
    }
#endif

    // FreeType2不可用时的回退：按字符数估算（中文字符约为fontHeight宽度）
    const int estimatedCharW = static_cast<int>(fontHeight * 1.0);
    const int charsPerLine   = std::max(1, maxWidthPx / estimatedCharW);
    const auto chars         = utf8CharBounds(text);
    int        count         = 0;
    std::string cur;
    for(const auto &cb : chars) {
        const size_t charLen  = cb.second;
        // ASCII字符按半宽计算
        const bool   isAscii  = (charLen == 1);
        const int    charCost = isAscii ? 1 : 2;
        if(!cur.empty() && count + charCost > charsPerLine * 2) {
            out.push_back(cur);
            cur.clear();
            count = 0;
        }
        cur   += text.substr(cb.first, charLen);
        count += charCost;
    }
    if(!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

static std::vector<std::string> splitTextLinesUtf8(const std::string &text) {
    std::vector<std::string> lines;
    std::string current;
    current.reserve(text.size());
    for(char ch: text) {
        if(ch == '\r') {
            continue;
        }
        if(ch == '\n') {
            lines.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    lines.push_back(current);
    return lines;
}

static std::vector<std::string> wrapMultilineTextUtf8(const std::string &text, int maxWidthPx, int fontHeight) {
    std::vector<std::string> wrapped;
    for(const auto &rawLine: splitTextLinesUtf8(text)) {
        if(rawLine.empty()) {
            wrapped.emplace_back();
            continue;
        }
        auto lines = wrapTextUtf8(rawLine, maxWidthPx, fontHeight);
        if(lines.empty()) {
            wrapped.push_back(rawLine);
        }
        else {
            wrapped.insert(wrapped.end(), lines.begin(), lines.end());
        }
    }
    return wrapped;
}

static int choosePromptFontHeight(const std::string &text, int maxWidthPx, int maxHeightPx,
                                  int preferredFontHeight, int minFontHeight) {
    for(int fontHeight = preferredFontHeight; fontHeight >= minFontHeight; fontHeight -= 2) {
        const auto lines = wrapMultilineTextUtf8(text, maxWidthPx, fontHeight);
        const int lineGap = std::max(12, fontHeight / 3);
        int totalHeight = 0;
        for(const auto &line: lines) {
            totalHeight += line.empty() ? lineGap : (fontHeight + lineGap);
        }
        if(totalHeight <= maxHeightPx) {
            return fontHeight;
        }
    }
    return minFontHeight;
}

// 中文文本渲染（支持FreeType2；无FreeType时显示ASCII回退提示）
static void putTextUtf8(cv::Mat &img, const std::string &text, cv::Point org,
                        int fontHeight, cv::Scalar color) {
#ifdef SYNC_COLLECTION_HAS_OPENCV_FREETYPE
    auto ft2 = initFreeType2();
    if(ft2) {
        try {
            ft2->putText(img, text, org, fontHeight, color, -1, cv::LINE_AA, false);
            return;
        }
        catch(...) {}
    }
#endif
    // 回退：使用Hershey字体（只能显示ASCII）
    const double scale = fontHeight / 30.0;
    cv::putText(img, text, org, cv::FONT_HERSHEY_DUPLEX, scale, color, 1, cv::LINE_AA);
}
// ---- UTF-8 中文渲染支持结束 ----

// OpenCV highgui 的 waitKey 常不把 Ctrl 与数字合并到返回值里：先收到 Control 键，再收到无修饰位的 '1'。
// Control_L / Control_R 的 GDK keyval 为 0xffe3 / 0xffe4，部分构建只暴露低 8 位 0xe3 / 0xe4（十进制 227/228）。
static bool isCtrlModifierKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return key == static_cast<int>(0xFFE3) || key == static_cast<int>(0xFFE4) || lo16 == static_cast<int>(0xFFE3)
           || lo16 == static_cast<int>(0xFFE4) || lo8 == 0xE3 || lo8 == 0xE4 || key == 227 || key == 228;
}

// 当前环境下，waitKey 会先上报 Ctrl 按下（0xE3/0xE4），再上报数字键，松开 Ctrl 时会出现 0x7F。
// 因此这里维护一个显式的“Ctrl 监听中”状态，而不是依赖短时间门闩。
static bool isCtrlReleaseKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == 0x007F || lo8 == 0x7F || key == 127;
}

static bool g_ctrlShortcutListening = false;

static int parseIntOr(const std::string &s, int fallback) {
    try {
        return std::stoi(s);
    }
    catch(...) {
        return fallback;
    }
}

static double parseDoubleBound(const std::string &s, double fallback, double lo, double hi) {
    try {
        double v = std::stod(s);
        return std::max(lo, std::min(hi, v));
    }
    catch(...) {
        return std::max(lo, std::min(hi, fallback));
    }
}

static std::string formatFrameIndex(size_t i) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << i;
    return oss.str();
}

static std::string colorExtNormalized(std::string ext) {
    if(ext.empty()) {
        ext = "jpg";
    }
    if(ext[0] != '.') {
        ext = "." + ext;
    }
    return ext;
}

static uint64_t frameTimestampUs(const std::shared_ptr<ob::Frame> &frame) {
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
    if(ts == 0) {
        return 0;
    }
    return ts;
}

enum class CollectDataType { RGB, Depth, IRLeft, IRRight, CloudPoints, ColorCloudPoints };

static const char *dataTypeLabel(CollectDataType t) {
    switch(t) {
    case CollectDataType::RGB:
        return "RGB";
    case CollectDataType::Depth:
        return "Depth";
    case CollectDataType::IRLeft:
        return "IR_left";
    case CollectDataType::IRRight:
        return "IR_right";
    case CollectDataType::CloudPoints:
        return "CloudPoints";
    case CollectDataType::ColorCloudPoints:
        return "ColorCloudPoints";
    }
    return "Unknown";
}

static OBSensorType dataTypeSensor(CollectDataType t) {
    switch(t) {
    case CollectDataType::RGB:
        return OB_SENSOR_COLOR;
    case CollectDataType::Depth:
        return OB_SENSOR_DEPTH;
    case CollectDataType::IRLeft:
        return OB_SENSOR_IR_LEFT;
    case CollectDataType::IRRight:
        return OB_SENSOR_IR_RIGHT;
    case CollectDataType::CloudPoints:
        return OB_SENSOR_DEPTH;
    case CollectDataType::ColorCloudPoints:
        return OB_SENSOR_DEPTH;
    }
    return OB_SENSOR_UNKNOWN;
}

static OBFrameType dataTypeFrameType(CollectDataType t) {
    switch(t) {
    case CollectDataType::RGB:
        return OB_FRAME_COLOR;
    case CollectDataType::Depth:
        return OB_FRAME_DEPTH;
    case CollectDataType::IRLeft:
        return OB_FRAME_IR_LEFT;
    case CollectDataType::IRRight:
        return OB_FRAME_IR_RIGHT;
    case CollectDataType::CloudPoints:
        return OB_FRAME_DEPTH;
    case CollectDataType::ColorCloudPoints:
        return OB_FRAME_DEPTH;
    }
    return OB_FRAME_UNKNOWN;
}

static std::string presetLabel(int w, int h, int fps) {
    return std::to_string(w) + "x" + std::to_string(h) + "@" + std::to_string(fps);
}

struct CollectionConfigUi {
    bool enableMultiview = true;
    bool enableFisheyes  = false;
    bool enableRgb       = true;
    bool enableDepth     = true;
    bool enableIrRight   = false;
    bool enableIrLeft    = false;
    bool enableCloud     = false;
    bool enableColorCloud = false;
    bool enableImu       = false;
    std::string width    = "1280";
    std::string height   = "800";
    std::string fps      = "30";
    std::string exposureMs;
    std::string brightness;
    std::string saveRoot;
    std::string subjectId = "test";
    std::string maxDurationSec = "120";
    std::string activeField;
    std::string error;

    void enforceRules() {
        if(enableCloud || enableColorCloud) {
            enableDepth = true;
        }
        if(enableColorCloud) {
            enableRgb = true;
        }
    }

    bool hasRequiredFields() const {
        return !trimString(saveRoot).empty() && !trimString(subjectId).empty();
    }

    bool hasSelectedCaptureType() const {
        return enableMultiview || enableFisheyes;
    }

    int widthInt() const { return std::max(1, parseIntOr(width, 1280)); }
    int heightInt() const { return std::max(1, parseIntOr(height, 800)); }
    int fpsInt() const { return std::max(1, parseIntOr(fps, 30)); }
    float exposureMsFloat() const {
        if(trimString(exposureMs).empty()) {
            return 0.0f;
        }
        return static_cast<float>(parseDoubleBound(exposureMs, 0.0, 0.05, 100.0));
    }
    int brightnessInt() const { return parseIntOr(brightness, -1); }
    int maxDurationInt() const { return std::max(1, parseIntOr(maxDurationSec, 120)); }

    std::vector<CollectDataType> enabledTypesForSaving() const {
        std::vector<CollectDataType> out;
        if(enableRgb) {
            out.push_back(CollectDataType::RGB);
        }
        if(enableDepth) {
            out.push_back(CollectDataType::Depth);
        }
        if(enableIrRight) {
            out.push_back(CollectDataType::IRRight);
        }
        if(enableIrLeft) {
            out.push_back(CollectDataType::IRLeft);
        }
        if(enableCloud) {
            out.push_back(CollectDataType::CloudPoints);
        }
        if(enableColorCloud) {
            out.push_back(CollectDataType::ColorCloudPoints);
        }
        return out;
    }

    std::vector<CollectDataType> enabledTypesForStreaming() const {
        std::vector<CollectDataType> out = enabledTypesForSaving();
        if(!enableRgb) {
            out.push_back(CollectDataType::RGB);
        }
        return out;
    }

    CollectDataType referenceType() const {
        if(enableRgb) {
            return CollectDataType::RGB;
        }
        if(enableDepth) {
            return CollectDataType::Depth;
        }
        if(enableIrLeft) {
            return CollectDataType::IRLeft;
        }
        if(enableIrRight) {
            return CollectDataType::IRRight;
        }
        return CollectDataType::RGB;
    }
};

struct CollectedSeries {
    std::vector<uint64_t> ts;
    std::vector<cv::Mat>  frames;
    std::vector<float>    valueScale;
    size_t                maxFrames = 0;
};

struct ImuSample {
    uint64_t tsUs = 0;
    float    x    = 0.0f;
    float    y    = 0.0f;
    float    z    = 0.0f;
};

struct StreamParams {
    int      width  = 0;
    int      height = 0;
    int      fps    = 0;
    OBFormat format = OB_FORMAT_UNKNOWN;
    OBCameraIntrinsic  intrinsic{};
    OBCameraDistortion distortion{};
    bool valid = false;
};

struct DeviceBuffer {
    std::string camKey;
    std::unordered_map<CollectDataType, CollectedSeries> series;
    std::unordered_map<CollectDataType, StreamParams> params;

    cv::Mat   latestRgb;
    uint64_t  latestRgbTsUs = 0;

    OBCameraParam rgbDepthParam{};
    bool          rgbDepthParamValid = false;
    std::vector<ImuSample> accelSamples;
    std::vector<ImuSample> gyroSamples;
};

static size_t recordWorkerCount() {
    const unsigned int hc = std::thread::hardware_concurrency();
    if(hc == 0) {
        return 8;
    }
    const unsigned int usable = (hc > 2) ? (hc - 2) : hc;
    return static_cast<size_t>(std::max(8u, std::min(16u, usable)));
}

static size_t saveWorkerCount() {
    const unsigned int hc = std::thread::hardware_concurrency();
    if(hc == 0) {
        return 16;
    }
    return static_cast<size_t>(std::max(12u, std::min(24u, hc + 4)));
}

static int preferredProfileFormatScore(OBSensorType sensorType, OBFormat format) {
    if(sensorType == OB_SENSOR_COLOR) {
        if(format == OB_FORMAT_MJPG) {
            return 0;
        }
        if(format == OB_FORMAT_RGB) {
            return 1;
        }
        if(format == OB_FORMAT_BGR) {
            return 2;
        }
        if(format == OB_FORMAT_YUYV || format == OB_FORMAT_YUY2 || format == OB_FORMAT_UYVY) {
            return 3;
        }
        return 10;
    }
    if(sensorType == OB_SENSOR_DEPTH) {
        if(format == OB_FORMAT_Y14) {
            return 0;
        }
        if(format == OB_FORMAT_Y16 || format == OB_FORMAT_Z16) {
            return 1;
        }
        if(format == OB_FORMAT_RLE) {
            return 2;
        }
        return 10;
    }
    if(sensorType == OB_SENSOR_IR || sensorType == OB_SENSOR_IR_LEFT || sensorType == OB_SENSOR_IR_RIGHT) {
        if(format == OB_FORMAT_Y8) {
            return 0;
        }
        if(format == OB_FORMAT_Y16) {
            return 1;
        }
        if(format == OB_FORMAT_MJPG) {
            return 2;
        }
        return 10;
    }
    return 10;
}

static std::shared_ptr<ob::VideoStreamProfile> pickVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe,
                                                                OBSensorType sensorType,
                                                                int width,
                                                                int height,
                                                                int fps) {
    std::shared_ptr<ob::StreamProfileList> list;
    try {
        list = pipe->getStreamProfileList(sensorType);
    }
    catch(...) {
        return nullptr;
    }
    if(!list || list->getCount() == 0) {
        return nullptr;
    }
    auto findBest = [&](int targetW, int targetH, int targetFps) {
        std::shared_ptr<ob::VideoStreamProfile> best;
        int bestScore = std::numeric_limits<int>::max();
        for(uint32_t i = 0; i < list->getCount(); i++) {
            auto p  = list->getProfile(i);
            auto vp = p->as<ob::VideoStreamProfile>();
            if(!vp) {
                continue;
            }
            const int pw = static_cast<int>(vp->getWidth());
            const int ph = static_cast<int>(vp->getHeight());
            const int pf = static_cast<int>(vp->getFps());
            if(targetW > 0 && pw != targetW) {
                continue;
            }
            if(targetH > 0 && ph != targetH) {
                continue;
            }

            int score = preferredProfileFormatScore(sensorType, vp->getFormat());
            if(targetFps > 0) {
                score += std::abs(pf - targetFps) * 10;
                if(pf < targetFps) {
                    score += 2;
                }
            }
            if(score < bestScore) {
                bestScore = score;
                best      = vp;
            }
        }
        return best;
    };

    if(auto best = findBest(width, height, fps)) {
        return best;
    }

    if(width > 0 || height > 0) {
        std::vector<std::pair<int, int>> fallbacks;
        if(sensorType == OB_SENSOR_DEPTH || sensorType == OB_SENSOR_IR || sensorType == OB_SENSOR_IR_LEFT || sensorType == OB_SENSOR_IR_RIGHT) {
            fallbacks = { { 640, 400 }, { 1280, 800 }, { 320, 200 } };
        }
        else if(sensorType == OB_SENSOR_COLOR) {
            fallbacks = { { 1280, 720 }, { 1920, 1080 }, { 640, 480 }, { 640, 360 } };
        }
        for(const auto &fallback: fallbacks) {
            if(fallback.first == width && fallback.second == height) {
                continue;
            }
            if(auto best = findBest(fallback.first, fallback.second, fps)) {
                return best;
            }
        }
    }

    try {
        return list->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
    }
    catch(...) {
        return nullptr;
    }
}

static void saveRawMatToPng(const cv::Mat &m, const fs::path &path, int pngCompression) {
    if(m.empty()) {
        return;
    }
    cv::imwrite(path.string(), m, { cv::IMWRITE_PNG_COMPRESSION, pngCompression });
}

static bool endsWith(const std::string &s, const std::string &suffix) {
    if(s.size() < suffix.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(), [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    });
}

static void saveBgrMatToFile(const cv::Mat &bgr, const fs::path &path, const SaveOptions &options) {
    if(bgr.empty()) {
        return;
    }
    std::vector<int> params;
    const auto       p = path.string();
    if(endsWith(p, ".jpg") || endsWith(p, ".jpeg")) {
        params = { cv::IMWRITE_JPEG_QUALITY, options.jpegQuality };
    }
    else if(endsWith(p, ".png")) {
        params = { cv::IMWRITE_PNG_COMPRESSION, options.pngCompression };
    }
    cv::imwrite(p, bgr, params);
}

static bool isJpegLikePath(const fs::path &path) {
    const auto p = path.string();
    return endsWith(p, ".jpg") || endsWith(p, ".jpeg");
}

static bool isJpegLikeExt(const std::string &ext) {
    auto normalized = colorExtNormalized(ext);
    return endsWith(normalized, ".jpg") || endsWith(normalized, ".jpeg");
}

static bool writeBytesToFile(const fs::path &path, const uint8_t *data, size_t size) {
    if(!data || size == 0) {
        return false;
    }
    std::ofstream ofs(path, std::ios::binary);
    if(!ofs.is_open()) {
        return false;
    }
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(ofs);
}

static std::string shellQuote(const std::string &s) {
    std::string out = "'";
    for(char c: s) {
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

static std::string toLowerAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string h265ExtNormalized(std::string ext) {
    if(ext.empty()) {
        ext = "mp4";
    }
    if(ext[0] != '.') {
        ext = "." + ext;
    }
    return ext;
}

static std::string h265OutputFileName(const SaveOptions &options) {
    return "rgb" + h265ExtNormalized(options.h265Ext);
}

static bool h265OutputIsRawStream(const fs::path &path) {
    const std::string p = toLowerAscii(path.string());
    return endsWith(p, ".h265") || endsWith(p, ".hevc");
}

static std::string resolvedH265Codec(const SaveOptions &options) {
    const std::string codec = trimString(options.h265Codec);
    if(!codec.empty()) {
        return codec;
    }
    const std::string mode = toLowerAscii(trimString(options.h265EncoderMode));
    if(mode == "hardware" || mode == "hw") {
        return "hevc_nvenc";
    }
    return "libx265";
}

static bool h265CodecIsSoftware(const std::string &codec) {
    const std::string c = toLowerAscii(codec);
    return c == "libx265" || c == "x265";
}

static bool h265CodecIsVaapi(const std::string &codec) {
    return toLowerAscii(codec).find("vaapi") != std::string::npos;
}

static bool h265CodecIsQsv(const std::string &codec) {
    return toLowerAscii(codec).find("qsv") != std::string::npos;
}

static bool h265CodecIsNvenc(const std::string &codec) {
    const std::string c = toLowerAscii(codec);
    return c.find("nvenc") != std::string::npos;
}

static bool h265CodecIsHardware(const std::string &codec) {
    return !h265CodecIsSoftware(codec);
}

static bool copyColorFrameToBgr(const std::shared_ptr<ob::Frame> &frame, cv::Mat &outBgr) {
    outBgr.release();
    auto colorFrame = frame ? frame->as<ob::ColorFrame>() : nullptr;
    if(!colorFrame) {
        return false;
    }
    auto vf = frame->as<ob::VideoFrame>();
    if(!vf) {
        return false;
    }
    const int width  = static_cast<int>(vf->width());
    const int height = static_cast<int>(vf->height());
    if(width <= 0 || height <= 0) {
        return false;
    }
    const auto *dataPtr = reinterpret_cast<const uint8_t *>(vf->data());
    const size_t dataSize = static_cast<size_t>(vf->dataSize());
    if(!dataPtr || dataSize == 0) {
        return false;
    }
    const auto fmt = vf->format();

    if(fmt == OB_FORMAT_MJPG) {
        cv::Mat rawMat(1, static_cast<int>(dataSize), CV_8UC1, const_cast<uint8_t *>(dataPtr));
        try {
            outBgr = cv::imdecode(rawMat, cv::IMREAD_COLOR);
        }
        catch(...) {
            outBgr.release();
        }
        return !outBgr.empty();
    }

    if(fmt == OB_FORMAT_BGR) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 3) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC3, const_cast<uint8_t *>(dataPtr), stride);
        outBgr = tmp.clone();
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_RGB) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 3) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC3, const_cast<uint8_t *>(dataPtr), stride);
        cv::cvtColor(tmp, outBgr, cv::COLOR_RGB2BGR);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_YUYV || fmt == OB_FORMAT_YUY2) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 2) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC2, const_cast<uint8_t *>(dataPtr), stride);
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_YUY2);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_UYVY) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 2) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC2, const_cast<uint8_t *>(dataPtr), stride);
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_UYVY);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_NV21) {
        const size_t expect = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 / 2;
        if(dataSize < expect) {
            return false;
        }
        cv::Mat tmp(height * 3 / 2, width, CV_8UC1, const_cast<uint8_t *>(dataPtr));
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_NV21);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_I420) {
        const size_t expect = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 / 2;
        if(dataSize < expect) {
            return false;
        }
        cv::Mat tmp(height * 3 / 2, width, CV_8UC1, const_cast<uint8_t *>(dataPtr));
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_I420);
        return !outBgr.empty();
    }
    return false;
}

static bool copyVideoFrameToRawMat(const std::shared_ptr<ob::Frame> &frame, cv::Mat &out, float *outValueScale = nullptr) {
    out.release();
    if(outValueScale) {
        *outValueScale = 0.0f;
    }
    auto vf = frame ? frame->as<ob::VideoFrame>() : nullptr;
    if(!vf) {
        return false;
    }
    const int width  = static_cast<int>(vf->width());
    const int height = static_cast<int>(vf->height());
    if(width <= 0 || height <= 0) {
        return false;
    }

    const auto fmt = vf->format();
    int        cvType = -1;
    size_t     elemSize = 0;
    if(fmt == OB_FORMAT_Y8 || fmt == OB_FORMAT_GRAY) {
        cvType   = CV_8UC1;
        elemSize = 1;
    }
    else {
        cvType   = CV_16UC1;
        elemSize = sizeof(uint16_t);
    }

    const auto *dataPtr = vf->data();
    if(!dataPtr) {
        return false;
    }
    const size_t dataSize = static_cast<size_t>(vf->dataSize());
    const size_t strideBytes = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
    if(strideBytes < static_cast<size_t>(width) * elemSize) {
        return false;
    }

    cv::Mat tmp(height, width, cvType, const_cast<void *>(dataPtr), strideBytes);
    out = tmp.clone();
    if(out.empty()) {
        return false;
    }

    if(outValueScale && frame->getType() == OB_FRAME_DEPTH) {
        auto df = frame->as<ob::DepthFrame>();
        if(df) {
            try {
                *outValueScale = df->getValueScale();
            }
            catch(...) {
                *outValueScale = 0.0f;
            }
        }
    }
    return true;
}

struct DetachedVideoFrame {
    int                  width = 0;
    int                  height = 0;
    OBFormat             format = OB_FORMAT_UNKNOWN;
    std::vector<uint8_t> data;
    float                valueScale = 0.0f;
};

static bool detachVideoFrame(const std::shared_ptr<ob::Frame> &frame,
                             DetachedVideoFrame &out,
                             bool includeValueScale = false) {
    out = DetachedVideoFrame{};
    auto vf = frame ? frame->as<ob::VideoFrame>() : nullptr;
    if(!vf) {
        return false;
    }
    const int width  = static_cast<int>(vf->width());
    const int height = static_cast<int>(vf->height());
    if(width <= 0 || height <= 0) {
        return false;
    }
    const auto *dataPtr = reinterpret_cast<const uint8_t *>(vf->data());
    const size_t dataSize = static_cast<size_t>(vf->dataSize());
    if(!dataPtr || dataSize == 0) {
        return false;
    }

    out.width = width;
    out.height = height;
    out.format = vf->format();
    out.data.assign(dataPtr, dataPtr + dataSize);
    if(includeValueScale && frame->getType() == OB_FRAME_DEPTH) {
        auto df = frame->as<ob::DepthFrame>();
        if(df) {
            try {
                out.valueScale = df->getValueScale();
            }
            catch(...) {
                out.valueScale = 0.0f;
            }
        }
    }
    return true;
}

static bool copyDetachedColorToBgr(const DetachedVideoFrame &frame, cv::Mat &outBgr) {
    outBgr.release();
    if(frame.width <= 0 || frame.height <= 0 || frame.data.empty()) {
        return false;
    }
    const int width = frame.width;
    const int height = frame.height;
    const auto *dataPtr = frame.data.data();
    const size_t dataSize = frame.data.size();
    const auto fmt = frame.format;

    if(fmt == OB_FORMAT_MJPG) {
        cv::Mat rawMat(1, static_cast<int>(dataSize), CV_8UC1, const_cast<uint8_t *>(dataPtr));
        try {
            outBgr = cv::imdecode(rawMat, cv::IMREAD_COLOR);
        }
        catch(...) {
            outBgr.release();
        }
        return !outBgr.empty();
    }

    if(fmt == OB_FORMAT_BGR) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 3) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC3, const_cast<uint8_t *>(dataPtr), stride);
        outBgr = tmp.clone();
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_RGB) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 3) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC3, const_cast<uint8_t *>(dataPtr), stride);
        cv::cvtColor(tmp, outBgr, cv::COLOR_RGB2BGR);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_YUYV || fmt == OB_FORMAT_YUY2) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 2) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC2, const_cast<uint8_t *>(dataPtr), stride);
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_YUY2);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_UYVY) {
        const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
        if(stride < static_cast<size_t>(width) * 2) {
            return false;
        }
        cv::Mat tmp(height, width, CV_8UC2, const_cast<uint8_t *>(dataPtr), stride);
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_UYVY);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_NV21) {
        const size_t expect = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 / 2;
        if(dataSize < expect) {
            return false;
        }
        cv::Mat tmp(height * 3 / 2, width, CV_8UC1, const_cast<uint8_t *>(dataPtr));
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_NV21);
        return !outBgr.empty();
    }
    if(fmt == OB_FORMAT_I420) {
        const size_t expect = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 / 2;
        if(dataSize < expect) {
            return false;
        }
        cv::Mat tmp(height * 3 / 2, width, CV_8UC1, const_cast<uint8_t *>(dataPtr));
        cv::cvtColor(tmp, outBgr, cv::COLOR_YUV2BGR_I420);
        return !outBgr.empty();
    }
    return false;
}

static bool copyDetachedVideoToRawMat(const DetachedVideoFrame &frame, cv::Mat &out, float *outValueScale = nullptr) {
    out.release();
    if(outValueScale) {
        *outValueScale = frame.valueScale;
    }
    if(frame.width <= 0 || frame.height <= 0 || frame.data.empty()) {
        return false;
    }

    int cvType = -1;
    size_t elemSize = 0;
    if(frame.format == OB_FORMAT_Y8 || frame.format == OB_FORMAT_GRAY) {
        cvType = CV_8UC1;
        elemSize = 1;
    }
    else {
        cvType = CV_16UC1;
        elemSize = sizeof(uint16_t);
    }

    const size_t strideBytes = frame.height > 0 ? (frame.data.size() / static_cast<size_t>(frame.height)) : 0;
    if(strideBytes < static_cast<size_t>(frame.width) * elemSize) {
        return false;
    }

    cv::Mat tmp(frame.height, frame.width, cvType, const_cast<uint8_t *>(frame.data.data()), strideBytes);
    out = tmp.clone();
    return !out.empty();
}

static bool writePointCloudPlyFromDepth(const cv::Mat &depth16,
                                       float valueScaleMm,
                                       const StreamParams &depthParams,
                                       const fs::path &outPath,
                                       float minDepthM,
                                       float maxDepthM,
                                       int decimationFactor) {
    if(depth16.empty() || depth16.type() != CV_16UC1) {
        return false;
    }
    if(!(valueScaleMm > 0.0f)) {
        return false;
    }
    if(!depthParams.valid || depthParams.intrinsic.fx <= 0.0f || depthParams.intrinsic.fy <= 0.0f) {
        return false;
    }
    if(!(minDepthM >= 0.0f) || !(maxDepthM > 0.0f) || maxDepthM <= minDepthM) {
        return false;
    }
    const int step = std::max(1, decimationFactor);

    const int w = depth16.cols;
    const int h = depth16.rows;
    const float fx = depthParams.intrinsic.fx;
    const float fy = depthParams.intrinsic.fy;
    const float cx = depthParams.intrinsic.cx;
    const float cy = depthParams.intrinsic.cy;

    uint64_t count = 0;
    for(int y = 0; y < h; y += step) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        for(int x = 0; x < w; x += step) {
            const uint16_t d = row[x];
            if(d == 0) {
                continue;
            }
            const float z = (static_cast<float>(d) * valueScaleMm) / 1000.0f;
            if(!(z >= minDepthM && z <= maxDepthM)) {
                continue;
            }
            count++;
        }
    }
    if(count == 0) {
        return false;
    }

    std::ofstream ofs(outPath, std::ios::binary);
    if(!ofs) {
        return false;
    }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << count << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for(int y = 0; y < h; y += step) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        for(int x = 0; x < w; x += step) {
            const uint16_t d = row[x];
            if(d == 0) {
                continue;
            }
            const float z = (static_cast<float>(d) * valueScaleMm) / 1000.0f;
            if(!(z >= minDepthM && z <= maxDepthM)) {
                continue;
            }
            const float xx = (static_cast<float>(x) - cx) * z / fx;
            const float yy = (static_cast<float>(y) - cy) * z / fy;
            ofs.write(reinterpret_cast<const char *>(&xx), sizeof(float));
            ofs.write(reinterpret_cast<const char *>(&yy), sizeof(float));
            ofs.write(reinterpret_cast<const char *>(&z), sizeof(float));
        }
    }
    return static_cast<bool>(ofs);
}

static bool writeColorPointCloudPlyFromDepthAndRgb(const cv::Mat &depth16,
                                                   float valueScaleMm,
                                                   const OBCameraParam &rgbDepthParam,
                                                   const cv::Mat &rgbBgr,
                                                   const fs::path &outPath,
                                                   float minDepthM,
                                                   float maxDepthM,
                                                   int decimationFactor) {
    if(depth16.empty() || depth16.type() != CV_16UC1) {
        return false;
    }
    if(rgbBgr.empty() || rgbBgr.type() != CV_8UC3) {
        return false;
    }
    if(!(valueScaleMm > 0.0f)) {
        return false;
    }
    if(rgbDepthParam.depthIntrinsic.fx <= 0.0f || rgbDepthParam.depthIntrinsic.fy <= 0.0f) {
        return false;
    }
    if(rgbDepthParam.rgbIntrinsic.fx <= 0.0f || rgbDepthParam.rgbIntrinsic.fy <= 0.0f) {
        return false;
    }
    if(!(minDepthM >= 0.0f) || !(maxDepthM > 0.0f) || maxDepthM <= minDepthM) {
        return false;
    }

    const int step = std::max(1, decimationFactor);

    const int dw = depth16.cols;
    const int dh = depth16.rows;
    const int cw = rgbBgr.cols;
    const int ch = rgbBgr.rows;

    const float fx = rgbDepthParam.depthIntrinsic.fx;
    const float fy = rgbDepthParam.depthIntrinsic.fy;
    const float cx = rgbDepthParam.depthIntrinsic.cx;
    const float cy = rgbDepthParam.depthIntrinsic.cy;

    uint64_t count = 0;
    for(int y = 0; y < dh; y += step) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        for(int x = 0; x < dw; x += step) {
            const uint16_t d = row[x];
            if(d == 0) {
                continue;
            }
            const float zM = (static_cast<float>(d) * valueScaleMm) / 1000.0f;
            if(!(zM >= minDepthM && zM <= maxDepthM)) {
                continue;
            }
            const float depthMm = static_cast<float>(d) * valueScaleMm;
            OBPoint2f   src{ static_cast<float>(x), static_cast<float>(y) };
            OBPoint2f   dst{};
            bool        ok = false;
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
            if(!ok) {
                continue;
            }
            const int u = static_cast<int>(dst.x + 0.5f);
            const int v = static_cast<int>(dst.y + 0.5f);
            if(u < 0 || v < 0 || u >= cw || v >= ch) {
                continue;
            }
            count++;
        }
    }

    if(count == 0) {
        return false;
    }

    std::ofstream ofs(outPath, std::ios::binary);
    if(!ofs) {
        return false;
    }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << count << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    for(int y = 0; y < dh; y += step) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        for(int x = 0; x < dw; x += step) {
            const uint16_t d = row[x];
            if(d == 0) {
                continue;
            }
            const float zM = (static_cast<float>(d) * valueScaleMm) / 1000.0f;
            if(!(zM >= minDepthM && zM <= maxDepthM)) {
                continue;
            }
            const float depthMm = static_cast<float>(d) * valueScaleMm;
            OBPoint2f   src{ static_cast<float>(x), static_cast<float>(y) };
            OBPoint2f   dst{};
            bool        ok = false;
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
            if(!ok) {
                continue;
            }
            const int u = static_cast<int>(dst.x + 0.5f);
            const int v = static_cast<int>(dst.y + 0.5f);
            if(u < 0 || v < 0 || u >= cw || v >= ch) {
                continue;
            }

            const float xx = (static_cast<float>(x) - cx) * zM / fx;
            const float yy = (static_cast<float>(y) - cy) * zM / fy;

            ofs.write(reinterpret_cast<const char *>(&xx), sizeof(float));
            ofs.write(reinterpret_cast<const char *>(&yy), sizeof(float));
            ofs.write(reinterpret_cast<const char *>(&zM), sizeof(float));

            const cv::Vec3b c = rgbBgr.at<cv::Vec3b>(v, u);
            const unsigned char r = c[2];
            const unsigned char g = c[1];
            const unsigned char b = c[0];
            ofs.write(reinterpret_cast<const char *>(&r), sizeof(unsigned char));
            ofs.write(reinterpret_cast<const char *>(&g), sizeof(unsigned char));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(unsigned char));
        }
    }

    return static_cast<bool>(ofs);
}

static void jsonAddNumber(cJSON *obj, const char *key, double v) {
    cJSON_AddItemToObject(obj, key, cJSON_CreateNumber(v));
}

static void jsonAddString(cJSON *obj, const char *key, const std::string &v) {
    cJSON_AddItemToObject(obj, key, cJSON_CreateString(v.c_str()));
}

static void jsonAddIntrinsic(cJSON *obj, const OBCameraIntrinsic &in) {
    jsonAddNumber(obj, "fx", in.fx);
    jsonAddNumber(obj, "fy", in.fy);
    jsonAddNumber(obj, "cx", in.cx);
    jsonAddNumber(obj, "cy", in.cy);
    jsonAddNumber(obj, "width", in.width);
    jsonAddNumber(obj, "height", in.height);
}

static void jsonAddDistortion(cJSON *obj, const OBCameraDistortion &d) {
    jsonAddNumber(obj, "k1", d.k1);
    jsonAddNumber(obj, "k2", d.k2);
    jsonAddNumber(obj, "k3", d.k3);
    jsonAddNumber(obj, "k4", d.k4);
    jsonAddNumber(obj, "k5", d.k5);
    jsonAddNumber(obj, "k6", d.k6);
    jsonAddNumber(obj, "p1", d.p1);
    jsonAddNumber(obj, "p2", d.p2);
    jsonAddNumber(obj, "model", static_cast<int>(d.model));
}

static void jsonAddExtrinsic(cJSON *obj, const float rot[9], const float trans[3]) {
    cJSON *rotArr = cJSON_CreateArray();
    for(int r = 0; r < 3; r++) {
        cJSON *row = cJSON_CreateArray();
        for(int c = 0; c < 3; c++) {
            const int idx = r * 3 + c;
            cJSON_AddItemToArray(row, cJSON_CreateNumber(rot[idx]));
        }
        cJSON_AddItemToArray(rotArr, row);
    }
    cJSON_AddItemToObject(obj, "rotation", rotArr);

    cJSON *tArr = cJSON_CreateArray();
    for(int i = 0; i < 3; i++) {
        cJSON_AddItemToArray(tArr, cJSON_CreateNumber(trans[i]));
    }
    cJSON_AddItemToObject(obj, "translation", tArr);
}

static bool readTextFile(const fs::path &p, std::string &out);
static bool writeTextFile(const fs::path &p, const std::string &content);

static size_t findNearestFisheyeFrameSetIndex(const std::vector<FisheyeFrameSet> &samples, uint64_t targetUs) {
    if(samples.empty()) {
        return 0;
    }
    auto absDiff = [](uint64_t a, uint64_t b) {
        return a > b ? (a - b) : (b - a);
    };
    auto it = std::lower_bound(samples.begin(), samples.end(), targetUs, [](const FisheyeFrameSet &sample, uint64_t tsUs) {
        return sample.representativeTimestampUs < tsUs;
    });

    size_t bestIndex = 0;
    uint64_t bestDiff = std::numeric_limits<uint64_t>::max();
    if(it != samples.end()) {
        bestIndex = static_cast<size_t>(std::distance(samples.begin(), it));
        bestDiff = absDiff(samples[bestIndex].representativeTimestampUs, targetUs);
    }
    if(it != samples.begin()) {
        const size_t cand = static_cast<size_t>(std::distance(samples.begin(), it - 1));
        const uint64_t diff = absDiff(samples[cand].representativeTimestampUs, targetUs);
        if(diff <= bestDiff) {
            bestIndex = cand;
        }
    }
    return bestIndex;
}

static std::string fisheyeCameraDirName(size_t cameraIdx) {
    return std::to_string(cameraIdx);
}

static std::string fisheyePreviewLabel(size_t cameraIdx) {
    return "fisheye_" + std::to_string(cameraIdx);
}

static const std::vector<std::string> &preferredFisheyeCameraLabels() {
    static const std::vector<std::string> labels = { "1", "5" };
    return labels;
}

static FisheyeModuleConfig buildAutoFisheyeConfig(const SaveOptions &saveOptions,
                                                  int maxDurationSec,
                                                  const std::vector<FisheyeDeviceInfo> &devices) {
    FisheyeModuleConfig cfg;
    cfg.enabled = true;
    cfg.targetFps = 60;
    cfg.maxBufferedSets = static_cast<size_t>(std::max(2048, maxDurationSec * 80));
    cfg.save.format = endsWith(colorExtNormalized(saveOptions.colorExt), ".png") ? FisheyeImageFormat::Png : FisheyeImageFormat::Jpeg;
    cfg.save.jpegQuality = saveOptions.jpegQuality;
    cfg.save.pngCompression = saveOptions.pngCompression;
    cfg.cameras.clear();
    cfg.cameras.reserve(devices.size());
    for(size_t i = 0; i < devices.size(); ++i) {
        FisheyeCameraConfig camera;
        camera.cameraId = "cam" + std::to_string(i);
        camera.devicePath = !devices[i].stablePath.empty() ? devices[i].stablePath : devices[i].devicePath;
        camera.preferredDeviceHint.clear();
        camera.deviceIndex = -1;
        camera.width = 1280;
        camera.height = 720;
        camera.fps = 60;
        camera.preferMjpeg = devices[i].supportsMjpeg;
        cfg.cameras.push_back(std::move(camera));
    }
    return cfg;
}

static bool writeFisheyeCameraParamsJson(const fs::path &cameraDir, size_t cameraIdx, const cv::Mat &frame, int fps) {
    const int width = std::max(0, frame.cols);
    const int height = std::max(0, frame.rows);

    OBCameraIntrinsic intrinsic{};
    intrinsic.width = width;
    intrinsic.height = height;
    intrinsic.fx = 0.0f;
    intrinsic.fy = 0.0f;
    intrinsic.cx = 0.0f;
    intrinsic.cy = 0.0f;

    OBCameraDistortion distortion{};
    distortion.model = static_cast<decltype(distortion.model)>(0);

    cJSON *root = cJSON_CreateObject();
    cJSON *camObj = cJSON_CreateObject();
    jsonAddString(camObj, "sn", "fisheye_" + std::to_string(cameraIdx));

    cJSON *rgbObj = cJSON_CreateObject();
    jsonAddNumber(rgbObj, "width", width);
    jsonAddNumber(rgbObj, "height", height);
    jsonAddNumber(rgbObj, "fps", fps);
    jsonAddNumber(rgbObj, "format", static_cast<int>(OB_FORMAT_BGR));

    cJSON *intrObj = cJSON_CreateObject();
    jsonAddIntrinsic(intrObj, intrinsic);
    cJSON_AddItemToObject(rgbObj, "intrinsic", intrObj);

    cJSON *distObj = cJSON_CreateObject();
    jsonAddDistortion(distObj, distortion);
    cJSON_AddItemToObject(rgbObj, "distortion", distObj);

    cJSON_AddItemToObject(camObj, "RGB", rgbObj);
    cJSON_AddItemToObject(root, fisheyeCameraDirName(cameraIdx).c_str(), camObj);

    char *printed = cJSON_Print(root);
    bool ok = false;
    if(printed) {
        ok = writeTextFile(cameraDir / "camera_params.json", printed);
        cJSON_free(printed);
    }
    cJSON_Delete(root);
    return ok;
}

struct CamWorldPose {
    bool  valid = false;
    float R[9]{};
    float t[3]{};
};

struct FusedCloudFrameInput {
    std::string sn;
    std::string camKey;
    cv::Mat     depth;
    float       valueScaleMm = 0.0f;
    StreamParams depthParams{};
    CamWorldPose pose{};
    cv::Mat     rgbFrame;
    OBCameraParam rgbDepthParam{};
    bool        hasColor = false;
};

static bool jsonParseMat3(cJSON *arr, float out[9]) {
    if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
        return false;
    }
    for(int r = 0; r < 3; r++) {
        cJSON *row = cJSON_GetArrayItem(arr, r);
        if(!row || !cJSON_IsArray(row) || cJSON_GetArraySize(row) != 3) {
            return false;
        }
        for(int c = 0; c < 3; c++) {
            cJSON *v = cJSON_GetArrayItem(row, c);
            if(!cJSON_IsNumber(v)) {
                return false;
            }
            out[r * 3 + c] = static_cast<float>(v->valuedouble);
        }
    }
    return true;
}

static bool jsonParseVec3(cJSON *arr, float out[3]) {
    if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
        return false;
    }
    for(int i = 0; i < 3; i++) {
        cJSON *v = cJSON_GetArrayItem(arr, i);
        if(!cJSON_IsNumber(v)) {
            return false;
        }
        out[i] = static_cast<float>(v->valuedouble);
    }
    return true;
}

static void loadCamWorldPoses(const std::string &path, std::unordered_map<std::string, CamWorldPose> &out) {
    out.clear();
    if(path.empty()) {
        return;
    }
    fs::path p(path);
    std::string content;
    if(!readTextFile(p, content)) {
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
        cJSON            *rotArr = cJSON_GetObjectItemCaseSensitive(item, "rotation");
        cJSON            *tArr   = cJSON_GetObjectItemCaseSensitive(item, "translation");
        float             Rcw[9];
        float             tCw[3];
        if(!jsonParseMat3(rotArr, Rcw)) {
            continue;
        }
        if(!jsonParseVec3(tArr, tCw)) {
            continue;
        }

        float Rwc[9];
        for(int r = 0; r < 3; r++) {
            for(int c = 0; c < 3; c++) {
                const int idxSrc = r * 3 + c;
                const int idxDst = c * 3 + r;
                Rwc[idxDst]      = Rcw[idxSrc];
            }
        }
        float tWc[3]{};
        for(int r = 0; r < 3; r++) {
            const int rowIdx = r * 3;
            float     v      = 0.0f;
            for(int c = 0; c < 3; c++) {
                v += Rwc[rowIdx + c] * tCw[c];
            }
            tWc[r] = -v;
        }

        CamWorldPose pose;
        pose.valid = true;
        for(int i = 0; i < 9; i++) {
            pose.R[i] = Rwc[i];
        }
        for(int i = 0; i < 3; i++) {
            pose.t[i] = tWc[i];
        }
        out[camId] = pose;
    }

    cJSON_Delete(root);
}

static bool writeFusedPointCloudPly(const std::vector<FusedCloudFrameInput> &infos,
                                    const fs::path &outPath,
                                    float maxDepthM,
                                    int decimationFactor) {
    if(infos.empty()) {
        return false;
    }
    const int step = std::max(1, decimationFactor);

    uint64_t totalCount = 0;
    for(const auto &info : infos) {
        if(info.depth.empty() || info.depth.type() != CV_16UC1) {
            continue;
        }
        if(!(info.valueScaleMm > 0.0f)) {
            continue;
        }
        if(!info.depthParams.valid || info.depthParams.intrinsic.fx <= 0.0f || info.depthParams.intrinsic.fy <= 0.0f) {
            continue;
        }
        const int w  = info.depth.cols;
        const int h  = info.depth.rows;
        for(int y = 0; y < h; y += step) {
            const uint16_t *row = info.depth.ptr<uint16_t>(y);
            for(int x = 0; x < w; x += step) {
                const uint16_t d = row[x];
                if(d == 0) {
                    continue;
                }
                const float z = (static_cast<float>(d) * info.valueScaleMm) / 1000.0f;
                if(z >= 0.2f && z <= maxDepthM) {
                    totalCount++;
                }
            }
        }
    }
    if(totalCount == 0) {
        return false;
    }

    std::ofstream ofs(outPath, std::ios::binary);
    if(!ofs) {
        return false;
    }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << totalCount << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for(const auto &info : infos) {
        if(info.depth.empty() || info.depth.type() != CV_16UC1) {
            continue;
        }
        if(!(info.valueScaleMm > 0.0f)) {
            continue;
        }
        if(!info.depthParams.valid || info.depthParams.intrinsic.fx <= 0.0f || info.depthParams.intrinsic.fy <= 0.0f) {
            continue;
        }

        const int   w  = info.depth.cols;
        const int   h  = info.depth.rows;
        const float fx = info.depthParams.intrinsic.fx;
        const float fy = info.depthParams.intrinsic.fy;
        const float cx = info.depthParams.intrinsic.cx;
        const float cy = info.depthParams.intrinsic.cy;

        for(int y = 0; y < h; y += step) {
            const uint16_t *row = info.depth.ptr<uint16_t>(y);
            for(int x = 0; x < w; x += step) {
                const uint16_t d = row[x];
                if(d == 0) {
                    continue;
                }
                const float z = (static_cast<float>(d) * info.valueScaleMm) / 1000.0f;
                if(!(z >= 0.2f && z <= maxDepthM)) {
                    continue;
                }

                const float xCam = (static_cast<float>(x) - cx) * z / fx;
                const float yCam = (static_cast<float>(y) - cy) * z / fy;
                const float zCam = z;

                const float xW = info.pose.R[0] * xCam + info.pose.R[1] * yCam + info.pose.R[2] * zCam + info.pose.t[0];
                const float yW = info.pose.R[3] * xCam + info.pose.R[4] * yCam + info.pose.R[5] * zCam + info.pose.t[1];
                const float zW = info.pose.R[6] * xCam + info.pose.R[7] * yCam + info.pose.R[8] * zCam + info.pose.t[2];

                ofs.write(reinterpret_cast<const char *>(&xW), sizeof(float));
                ofs.write(reinterpret_cast<const char *>(&yW), sizeof(float));
                ofs.write(reinterpret_cast<const char *>(&zW), sizeof(float));
            }
        }
    }
    return static_cast<bool>(ofs);
}

static bool writeFusedColorPointCloudPly(const std::vector<FusedCloudFrameInput> &infos,
                                         const fs::path &outPath,
                                         float maxDepthM,
                                         int decimationFactor) {
    if(infos.empty()) {
        return false;
    }
    const int step = std::max(1, decimationFactor);

    uint64_t totalCount = 0;
    for(const auto &info : infos) {
        if(!info.hasColor || info.depth.empty() || info.depth.type() != CV_16UC1 || info.rgbFrame.empty()) {
            continue;
        }
        if(!(info.valueScaleMm > 0.0f)) {
            continue;
        }
        const auto &cp = info.rgbDepthParam;
        if(cp.depthIntrinsic.fx <= 0.0f || cp.depthIntrinsic.fy <= 0.0f) {
            continue;
        }

        const int w  = info.depth.cols;
        const int h  = info.depth.rows;
        const int cw = info.rgbFrame.cols;
        const int ch = info.rgbFrame.rows;
        for(int y = 0; y < h; y += step) {
            const uint16_t *row = info.depth.ptr<uint16_t>(y);
            for(int x = 0; x < w; x += step) {
                const uint16_t d = row[x];
                if(d == 0) {
                    continue;
                }
                const float zM = (static_cast<float>(d) * info.valueScaleMm) / 1000.0f;
                if(!(zM >= 0.2f && zM <= maxDepthM)) {
                    continue;
                }
                const float depthMm = static_cast<float>(d) * info.valueScaleMm;
                OBPoint2f   src{ static_cast<float>(x), static_cast<float>(y) };
                OBPoint2f   dst{};
                bool        ok = false;
                try {
                    ok = ob::CoordinateTransformHelper::transformation2dto2d(
                        src,
                        depthMm,
                        cp.depthIntrinsic,
                        cp.depthDistortion,
                        cp.rgbIntrinsic,
                        cp.rgbDistortion,
                        cp.transform,
                        &dst);
                }
                catch(...) {
                    ok = false;
                }
                if(!ok) {
                    continue;
                }
                const int u = static_cast<int>(dst.x + 0.5f);
                const int v = static_cast<int>(dst.y + 0.5f);
                if(u < 0 || v < 0 || u >= cw || v >= ch) {
                    continue;
                }
                totalCount++;
            }
        }
    }
    if(totalCount == 0) {
        return false;
    }

    std::ofstream ofs(outPath, std::ios::binary);
    if(!ofs) {
        return false;
    }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << totalCount << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    for(const auto &info : infos) {
        if(!info.hasColor || info.depth.empty() || info.depth.type() != CV_16UC1 || info.rgbFrame.empty()) {
            continue;
        }
        if(!(info.valueScaleMm > 0.0f)) {
            continue;
        }
        const auto &cp = info.rgbDepthParam;
        if(cp.depthIntrinsic.fx <= 0.0f || cp.depthIntrinsic.fy <= 0.0f) {
            continue;
        }
        const int   w  = info.depth.cols;
        const int   h  = info.depth.rows;
        const int   cw = info.rgbFrame.cols;
        const int   ch = info.rgbFrame.rows;
        const float fx = cp.depthIntrinsic.fx;
        const float fy = cp.depthIntrinsic.fy;
        const float cx = cp.depthIntrinsic.cx;
        const float cy = cp.depthIntrinsic.cy;

        for(int y = 0; y < h; y += step) {
            const uint16_t *row = info.depth.ptr<uint16_t>(y);
            for(int x = 0; x < w; x += step) {
                const uint16_t d = row[x];
                if(d == 0) {
                    continue;
                }
                const float zM = (static_cast<float>(d) * info.valueScaleMm) / 1000.0f;
                if(!(zM >= 0.2f && zM <= maxDepthM)) {
                    continue;
                }
                const float depthMm = static_cast<float>(d) * info.valueScaleMm;
                OBPoint2f   src{ static_cast<float>(x), static_cast<float>(y) };
                OBPoint2f   dst{};
                bool        ok = false;
                try {
                    ok = ob::CoordinateTransformHelper::transformation2dto2d(
                        src,
                        depthMm,
                        cp.depthIntrinsic,
                        cp.depthDistortion,
                        cp.rgbIntrinsic,
                        cp.rgbDistortion,
                        cp.transform,
                        &dst);
                }
                catch(...) {
                    ok = false;
                }
                if(!ok) {
                    continue;
                }
                const int u = static_cast<int>(dst.x + 0.5f);
                const int v = static_cast<int>(dst.y + 0.5f);
                if(u < 0 || v < 0 || u >= cw || v >= ch) {
                    continue;
                }

                const float xCam = (static_cast<float>(x) - cx) * zM / fx;
                const float yCam = (static_cast<float>(y) - cy) * zM / fy;
                const float zCam = zM;

                const float xW = info.pose.R[0] * xCam + info.pose.R[1] * yCam + info.pose.R[2] * zCam + info.pose.t[0];
                const float yW = info.pose.R[3] * xCam + info.pose.R[4] * yCam + info.pose.R[5] * zCam + info.pose.t[1];
                const float zW = info.pose.R[6] * xCam + info.pose.R[7] * yCam + info.pose.R[8] * zCam + info.pose.t[2];

                ofs.write(reinterpret_cast<const char *>(&xW), sizeof(float));
                ofs.write(reinterpret_cast<const char *>(&yW), sizeof(float));
                ofs.write(reinterpret_cast<const char *>(&zW), sizeof(float));

                const cv::Vec3b c = info.rgbFrame.at<cv::Vec3b>(v, u);
                const unsigned char r = c[2];
                const unsigned char g = c[1];
                const unsigned char b = c[0];
                ofs.write(reinterpret_cast<const char *>(&r), sizeof(unsigned char));
                ofs.write(reinterpret_cast<const char *>(&g), sizeof(unsigned char));
                ofs.write(reinterpret_cast<const char *>(&b), sizeof(unsigned char));
            }
        }
    }
    return static_cast<bool>(ofs);
}

static bool readTextFile(const fs::path &p, std::string &out) {
    std::ifstream ifs(p, std::ios::in | std::ios::binary);
    if(!ifs) {
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

static bool writeTextFile(const fs::path &p, const std::string &content) {
    std::ofstream ofs(p, std::ios::out | std::ios::binary);
    if(!ofs) {
        return false;
    }
    ofs << content;
    return true;
}

static cv::Mat alignDepthToRgb(const cv::Mat &depth16,
                                float valueScaleMm,
                                const OBCameraParam &rgbDepthParam,
                                int rgbW, int rgbH) {
    cv::Mat aligned(rgbH, rgbW, CV_16UC1, cv::Scalar(0));
    if(depth16.empty() || depth16.type() != CV_16UC1 || !(valueScaleMm > 0.0f)) {
        return aligned;
    }
    if(rgbDepthParam.depthIntrinsic.fx <= 0.0f || rgbDepthParam.rgbIntrinsic.fx <= 0.0f) {
        return aligned;
    }

    const int   dW   = depth16.cols;
    const int   dH   = depth16.rows;
    const float fx_d = rgbDepthParam.depthIntrinsic.fx;
    const float fy_d = rgbDepthParam.depthIntrinsic.fy;
    const float cx_d = rgbDepthParam.depthIntrinsic.cx;
    const float cy_d = rgbDepthParam.depthIntrinsic.cy;
    const float *R   = rgbDepthParam.transform.rot;
    const float *t   = rgbDepthParam.transform.trans;
    cv::Mat      zBuf(rgbH, rgbW, CV_32FC1, cv::Scalar(std::numeric_limits<float>::infinity()));

    for(int v = 0; v < dH; ++v) {
        const uint16_t *row = depth16.ptr<uint16_t>(v);
        for(int u = 0; u < dW; ++u) {
            const uint16_t d = row[u];
            if(d == 0) {
                continue;
            }

            const float depthMm = static_cast<float>(d) * valueScaleMm;
            const float X_mm    = (static_cast<float>(u) - cx_d) * depthMm / fx_d;
            const float Y_mm    = (static_cast<float>(v) - cy_d) * depthMm / fy_d;
            const float Z_mm    = depthMm;

            const float Xr = R[0] * X_mm + R[1] * Y_mm + R[2] * Z_mm + t[0];
            const float Yr = R[3] * X_mm + R[4] * Y_mm + R[5] * Z_mm + t[1];
            const float Zr = R[6] * X_mm + R[7] * Y_mm + R[8] * Z_mm + t[2];
            (void)Xr;
            (void)Yr;
            if(Zr <= 0.0f) {
                continue;
            }

            // Keep the RGB-aligned depth image consistent with the realtime color-cloud path:
            // map pixels with CoordinateTransformHelper and keep the nearest projected depth.
            OBPoint2f src{ static_cast<float>(u), static_cast<float>(v) };
            OBPoint2f dst{};
            bool      ok = false;
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
            if(!ok) {
                continue;
            }

            const int u_rgb = static_cast<int>(dst.x + 0.5f);
            const int v_rgb = static_cast<int>(dst.y + 0.5f);
            if(u_rgb < 0 || v_rgb < 0 || u_rgb >= rgbW || v_rgb >= rgbH) {
                continue;
            }

            const float zRatio = Zr / valueScaleMm;
            if(zRatio > 65534.0f) {
                continue;
            }

            float &bestZ = zBuf.at<float>(v_rgb, u_rgb);
            if(Zr >= bestZ) {
                continue;
            }
            bestZ = Zr;
            aligned.at<uint16_t>(v_rgb, u_rgb) = static_cast<uint16_t>(zRatio + 0.5f);
        }
    }

    return aligned;
}

class MultiDeviceStreamingRecorder {
public:
    explicit MultiDeviceStreamingRecorder(AppConfig baseCfg)
        : cfg_(std::move(baseCfg)) {
        const size_t baseQueue = cfg_.queueCapacity > 0 ? static_cast<size_t>(cfg_.queueCapacity) : 1024;
        recordQueueMax_ = std::max<size_t>(16384, baseQueue * 16);
        coordQueueMax_ = std::max<size_t>(8192, baseQueue * 8);
        writeQueueMax_ = std::max<size_t>(16384, baseQueue * 16);
    }

    bool isCapturing() const { return capturing_.load(); }
    bool isRecording() const { return recording_.load(); }
    bool hasData() const { return hasData_.load(); }

    bool isDrainComplete() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return true;
        }
        return session_.coordinatorDone && queuedWriteCount_.load() == 0 && writeInFlight_.load() == 0
               && !h265EncodingActive_.load();
    }

    std::string currentSessionLabel() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return "";
        }
        std::ostringstream oss;
        oss << session_.subjectId << "/" << session_.taskName << "/episode_" << session_.episodeN;
        return oss.str();
    }

    double currentRecordingSeconds() const {
        if(!recording_.load()) {
            return 0.0;
        }
        const auto elapsed = std::chrono::steady_clock::now() - captureStartSteady_;
        return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
    }

    double lastRecordedSeconds() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return lastRecordedSeconds_;
    }

    bool autoStopIfTimeout() {
        if(!recording_.load()) {
            return false;
        }
        if(cfg_.durationSec <= 0) {
            return false;
        }
        const auto elapsed = std::chrono::steady_clock::now() - captureStartSteady_;
        if(elapsed >= std::chrono::seconds(cfg_.durationSec)) {
            stopRecording();
            return true;
        }
        return false;
    }

    void reset() {
        joinCoordinatorThreadIfPossible();
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            closeSessionTimestampsLocked();
            session_ = SessionState{};
            coordRecordQueue_.clear();
            coordFisheyeQueue_.clear();
            coordCv_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for(auto &kv: buffers_) {
                kv.second.latestRgb.release();
                kv.second.latestRgbTsUs = 0;
                kv.second.accelSamples.clear();
                kv.second.gyroSamples.clear();
            }
        }
        {
            std::lock_guard<std::mutex> lock(streamSeqMtx_);
            streamNextSeq_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(previewTsMtx_);
            lastPreviewTs_.clear();
        }
        hasData_.store(false);
        recordInputClosing_.store(false);
        multiviewEosNotified_.store(false);
        passthroughRgbMjpg_.store(false);
    }

    void clearStatus() {
        std::lock_guard<std::mutex> lock(mtx_);
        captureInfoLine_.clear();
        slotStatsLine_.clear();
        savedAlignedMaxDiffMs_ = 0.0;
    }

    std::string lastInfoLine() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return captureInfoLine_;
    }

    double lastAlignedMaxDiffMs() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return savedAlignedMaxDiffMs_;
    }

    std::string lastSlotStatsLine() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return slotStatsLine_;
    }

    std::string currentRecordingStatsLine() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return "";
        }
        double seconds = 0.0;
        if(recording_.load()) {
            const auto elapsed = std::chrono::steady_clock::now() - captureStartSteady_;
            seconds = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
        }
        else {
            std::lock_guard<std::mutex> lockInfo(mtx_);
            seconds = lastRecordedSeconds_;
        }
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << "Time " << std::setprecision(2) << seconds << " s"
            << "   Frames " << session_.refFrameCount;
        return oss.str();
    }

    std::string captureFrameSummaryLine() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return "";
        }
        const size_t totalFrames = session_.alignedRef > 0 ? session_.alignedRef : session_.refFrameCount;
        const size_t actualFrames = session_.fullAligned > 0 ? session_.fullAligned : session_.nextFrameIndex;
        const size_t droppedFrames = totalFrames >= actualFrames ? (totalFrames - actualFrames) : session_.missingAligned;
        std::ostringstream oss;
        oss << "Actual " << actualFrames
            << " / Total " << totalFrames
            << " / Dropped " << droppedFrames;
        return oss.str();
    }

    std::string drainStatusLine() const {
        if(recording_.load()) {
            return "";
        }
        std::ostringstream oss;
        bool any = false;
        {
            std::lock_guard<std::mutex> lock(recordMtx_);
            if(!recordQueue_.empty() || recordInFlight_ > 0) {
                oss << "decode=" << recordQueue_.size() << "+" << recordInFlight_;
                any = true;
            }
        }
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            if(!coordRecordQueue_.empty() || !coordFisheyeQueue_.empty()) {
                if(any) {
                    oss << " ";
                }
                oss << "align=" << coordRecordQueue_.size() << "+" << coordFisheyeQueue_.size();
                any = true;
            }
        }
        if(queuedWriteCount_.load() > 0 || writeInFlight_.load() > 0) {
            if(any) {
                oss << " ";
            }
            oss << "write=" << queuedWriteCount_.load() << "+" << writeInFlight_.load();
            any = true;
        }
        return any ? oss.str() : "";
    }

    std::string streamProfilesLine() const {
        std::lock_guard<std::mutex> lock(mtx_);
        if(buffers_.empty()) {
            return "";
        }
        std::vector<std::string> sns;
        sns.reserve(buffers_.size());
        for(const auto &kv: buffers_) {
            sns.push_back(kv.first);
        }
        std::sort(sns.begin(), sns.end(), [&](const std::string &a, const std::string &b) {
            return buffers_.at(a).camKey < buffers_.at(b).camKey;
        });
        std::ostringstream oss;
        oss << "Profiles:";
        for(const auto &sn: sns) {
            oss << " " << buffers_.at(sn).camKey;
            for(const auto t: typesStreaming_) {
                if(t == CollectDataType::CloudPoints || t == CollectDataType::ColorCloudPoints) {
                    continue;
                }
                auto itP = buffers_.at(sn).params.find(t);
                if(itP == buffers_.at(sn).params.end()) {
                    continue;
                }
                oss << " " << dataTypeLabel(t) << "=" << itP->second.width << "x" << itP->second.height << "@" << itP->second.fps;
            }
            oss << ";";
        }
        return oss.str();
    }

    std::unordered_map<std::string, cv::Mat> latestRgbFrames() {
        auto out = latestRgbFramesImpl();
        if(fisheyeEnabled_ && fisheyeRecorder_.isRunning()) {
            std::string err;
            auto snap = fisheyeRecorder_.snapshotLatest(&err);
            if(snap) {
                for(size_t i = 0; i < snap->frames.size(); ++i) {
                    if(!snap->frames[i].bgr.empty()) {
                        out[fisheyePreviewLabel(i)] = snap->frames[i].bgr;
                    }
                }
            }
        }
        return out;
    }

    static int clampPropertyValue(int v, const OBIntPropertyRange &r) {
        const int step = (r.step > 0) ? r.step : 1;
        int out = std::max(r.min, std::min(r.max, v));
        out = r.min + ((out - r.min) / step) * step;
        out = std::max(r.min, std::min(r.max, out));
        return out;
    }

    void applyColorCaptureTuning(DeviceRuntime &rt, bool colorEnabled) {
        if(!colorEnabled || !rt.dev) {
            return;
        }

        const bool manualExposure = colorExposureMs_ > 0.0f;

        try {
            if(rt.dev->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_READ_WRITE)) {
                rt.dev->setBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, !manualExposure);
                std::cerr << "[collection] color auto exposure sn=" << rt.cfg.sn << " value=" << (!manualExposure ? "true" : "false") << std::endl;
            }
        }
        catch(...) {
        }

        try {
            if(rt.dev->isPropertySupported(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, OB_PERMISSION_READ_WRITE)) {
                rt.dev->setBoolProperty(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, true);
            }
        }
        catch(...) {
        }

        if(manualExposure) {
            try {
                const bool canRead = rt.dev->isPropertySupported(OB_PROP_COLOR_EXPOSURE_INT, OB_PERMISSION_READ);
                const bool canWrite = rt.dev->isPropertySupported(OB_PROP_COLOR_EXPOSURE_INT, OB_PERMISSION_WRITE);
                if(canRead && canWrite) {
                    const auto range = rt.dev->getIntPropertyRange(OB_PROP_COLOR_EXPOSURE_INT);
                    const int targetUsRaw = static_cast<int>(colorExposureMs_ * 1000.0f + 0.5f);
                    const int targetUs = clampPropertyValue(targetUsRaw, range);
                    rt.dev->setIntProperty(OB_PROP_COLOR_EXPOSURE_INT, targetUs);
                    const int appliedUs = rt.dev->getIntProperty(OB_PROP_COLOR_EXPOSURE_INT);
                    std::cerr << "[collection] set color exposure sn=" << rt.cfg.sn << " target_us=" << targetUs
                              << " applied_us=" << appliedUs << std::endl;
                    const int readbackDiff = (appliedUs > targetUs) ? (appliedUs - targetUs) : (targetUs - appliedUs);
                    if(readbackDiff > std::max(1, range.step)) {
                        std::cerr << "[collection] warning color exposure readback mismatch sn=" << rt.cfg.sn << std::endl;
                    }
                }
                else {
                    std::cerr << "[collection] color exposure property not writable sn=" << rt.cfg.sn << std::endl;
                }
            }
            catch(const std::exception &e) {
                std::cerr << "[collection] set color exposure failed sn=" << rt.cfg.sn << " error=" << e.what() << std::endl;
            }
            catch(...) {
                std::cerr << "[collection] set color exposure failed sn=" << rt.cfg.sn << std::endl;
            }
        }

        if(colorBrightness_ < 0) {
            return;
        }

        try {
            const bool canRead = rt.dev->isPropertySupported(OB_PROP_COLOR_BRIGHTNESS_INT, OB_PERMISSION_READ);
            const bool canWrite = rt.dev->isPropertySupported(OB_PROP_COLOR_BRIGHTNESS_INT, OB_PERMISSION_WRITE);
            if(canRead && canWrite) {
                const auto range = rt.dev->getIntPropertyRange(OB_PROP_COLOR_BRIGHTNESS_INT);
                const int target = clampPropertyValue(colorBrightness_, range);
                rt.dev->setIntProperty(OB_PROP_COLOR_BRIGHTNESS_INT, target);
                std::cerr << "[collection] set color brightness sn=" << rt.cfg.sn << " value=" << target << std::endl;
            }
        }
        catch(...) {
        }
    }

    bool start(const CollectionConfigUi &ui) {
        if(capturing_.load()) {
            return true;
        }
        collectionSetStage("start_enter");
        std::cerr << "[collection] start enter" << std::endl;
        reset();
        clearStatus();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            buffers_.clear();
        }
        cfg_.durationSec = ui.maxDurationInt();
        cfg_.collectFps  = ui.fpsInt();
        colorExposureMs_ = ui.exposureMsFloat();
        colorBrightness_ = ui.brightnessInt();
        multiviewEnabled_ = ui.enableMultiview;
        fisheyeEnabled_   = ui.enableFisheyes;
        activeFisheyeCameraCount_ = 0;

        if(!multiviewEnabled_ && !fisheyeEnabled_) {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "Select at least one capture type";
            return false;
        }

        const int w = ui.widthInt();
        const int h = ui.heightInt();
        const int f = ui.fpsInt();
        imuEnabled_ = multiviewEnabled_ && ui.enableImu;
        typesStreaming_ = multiviewEnabled_ ? ui.enabledTypesForStreaming() : std::vector<CollectDataType>{};
        typesSaving_    = multiviewEnabled_ ? ui.enabledTypesForSaving() : std::vector<CollectDataType>{};
        refType_        = ui.referenceType();
        uiFpsFallback_  = f;

        std::string fisheyeStatusLine;
        if(fisheyeEnabled_) {
            const auto fisheyeDevices = listPreferredFisheyeDevices(preferredFisheyeCameraLabels());
            if(fisheyeDevices.empty()) {
                if(multiviewEnabled_) {
                    fisheyeEnabled_ = false;
                    fisheyeStatusLine = "No fisheye detected, fallback to multiview only";
                    std::cerr << "[collection] " << fisheyeStatusLine << std::endl;
                }
                else {
                    std::lock_guard<std::mutex> lock(mtx_);
                    captureInfoLine_ = "No fisheye detected";
                    return false;
                }
            }
            else {
                const auto fisheyeCfg = buildAutoFisheyeConfig(cfg_.save, ui.maxDurationInt(), fisheyeDevices);
                std::string fisheyeError;
                if(!fisheyeRecorder_.start(fisheyeCfg, &fisheyeError)) {
                    if(multiviewEnabled_) {
                        fisheyeEnabled_ = false;
                        fisheyeStatusLine = "Fisheye unavailable, fallback to multiview only: " + fisheyeError;
                        std::cerr << "[collection] " << fisheyeStatusLine << std::endl;
                    }
                    else {
                        std::lock_guard<std::mutex> lock(mtx_);
                        captureInfoLine_ = "Fisheye start failed: " + fisheyeError;
                        return false;
                    }
                }
                else if(!fisheyeRecorder_.waitUntilReady(std::chrono::seconds(2))) {
                    fisheyeRecorder_.stop();
                    if(multiviewEnabled_) {
                        fisheyeEnabled_ = false;
                        fisheyeStatusLine = "Fisheye start timed out, fallback to multiview only";
                        std::cerr << "[collection] " << fisheyeStatusLine << std::endl;
                    }
                    else {
                        std::lock_guard<std::mutex> lock(mtx_);
                        captureInfoLine_ = "Fisheye start timed out";
                        return false;
                    }
                }
                else {
                    activeFisheyeCameraCount_ = fisheyeCfg.cameras.size();
                }
            }
        }

        if(!multiviewEnabled_) {
            stopping_.store(false);
            capturing_.store(true);
            recording_.store(false);
            softwareTriggerDevices_.clear();
            useSoftwareTrigger_ = false;
            if(!fisheyeStatusLine.empty()) {
                std::lock_guard<std::mutex> lock(mtx_);
                captureInfoLine_ = fisheyeStatusLine;
            }
            return true;
        }

        collectionSetStage("start_queryDeviceList");
        auto deviceList = ctx_.queryDeviceList();
        if(!deviceList || deviceList->deviceCount() == 0) {
            fisheyeRecorder_.stop();
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "No device connected";
            std::cerr << "[collection] no device connected" << std::endl;
            return false;
        }

        collectionSetStage("start_selectDevices");
        auto selected = selectDevicesWithPipeline(deviceList, cfg_);
        if(selected.empty()) {
            fisheyeRecorder_.stop();
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "No configured devices found";
            std::cerr << "[collection] no configured devices found" << std::endl;
            return false;
        }

        std::cerr << "[collection] selected devices=" << selected.size() << std::endl;
        collectionSetStage("start_applySyncConfig");
        applySyncConfig(selected);
        primary_.clear();
        secondary_.clear();
        all_.clear();
        splitPrimarySecondary(selected, primary_, secondary_);
        all_ = selected;

        for(const auto &rt: all_) {
            std::string camKey = rt.cfg.index.empty() ? std::to_string(rt.deviceIndex) : rt.cfg.index;
            std::lock_guard<std::mutex> lock(mtx_);
            buffers_.emplace(rt.cfg.sn, DeviceBuffer{ camKey });
        }

        {
            std::lock_guard<std::mutex> lock(callbackCountsMtx_);
            callbackCounts_.clear();
            for(const auto &rt: all_) {
                callbackCounts_[rt.cfg.sn] = 0;
            }
        }

        std::cerr << "[collection] streams: w=" << w << " h=" << h << " fps=" << f << std::endl;

        auto startOne = [&](DeviceRuntime &rt) {
            auto config = std::make_shared<ob::Config>();
            std::unordered_set<OBSensorType> enabledSensors;
            int colorW = 0;
            int colorH = 0;
            int depthW = 0;
            int depthH = 0;
            for(const auto t: typesStreaming_) {
                if(t == CollectDataType::CloudPoints || t == CollectDataType::ColorCloudPoints) {
                    continue;
                }
                const auto sensor = dataTypeSensor(t);
                if(enabledSensors.find(sensor) != enabledSensors.end()) {
                    continue;
                }
                collectionSetStage("start_pickVideoProfile");
                auto profile = pickVideoProfile(rt.pipe, sensor, w, h, f);
                if(profile) {
                    config->enableStream(profile);
                    enabledSensors.insert(sensor);
                    StreamParams sp;
                    sp.width  = static_cast<int>(profile->getWidth());
                    sp.height = static_cast<int>(profile->getHeight());
                    sp.fps    = static_cast<int>(profile->getFps());
                    sp.format = profile->getFormat();
                    try {
                        sp.intrinsic = profile->getIntrinsic();
                        sp.distortion = profile->getDistortion();
                        sp.valid = true;
                    }
                    catch(...) {
                    }
                    if(t == CollectDataType::RGB) {
                        colorW = sp.width;
                        colorH = sp.height;
                    }
                    else if(t == CollectDataType::Depth) {
                        depthW = sp.width;
                        depthH = sp.height;
                    }
                    std::lock_guard<std::mutex> lock(mtx_);
                    auto it = buffers_.find(rt.cfg.sn);
                    if(it != buffers_.end()) {
                        it->second.params[t] = sp;
                    }
                }
            }

            if(colorW > 0 && colorH > 0 && depthW > 0 && depthH > 0) {
                try {
                    auto cameraParam = rt.pipe->getCameraParamWithProfile(static_cast<uint32_t>(colorW),
                                                                          static_cast<uint32_t>(colorH),
                                                                          static_cast<uint32_t>(depthW),
                                                                          static_cast<uint32_t>(depthH));
                    std::lock_guard<std::mutex> lock(mtx_);
                    auto it = buffers_.find(rt.cfg.sn);
                    if(it != buffers_.end()) {
                        it->second.rgbDepthParam      = cameraParam;
                        it->second.rgbDepthParamValid = true;
                    }
                }
                catch(...) {
                }
            }

            const bool colorEnabled = enabledSensors.find(OB_SENSOR_COLOR) != enabledSensors.end();
            applyColorCaptureTuning(rt, colorEnabled);

            const auto deviceSn    = rt.cfg.sn;
            const auto deviceIndex = rt.deviceIndex;
            collectionSetStage("start_pipeline_start");
            std::cerr << "[collection] start pipeline sn=" << deviceSn << " streams=" << enabledSensors.size() << std::endl;
            if(enabledSensors.size() > 1) {
                config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
                try {
                    rt.pipe->enableFrameSync();
                }
                catch(...) {
                    std::cerr << "[collection] enableFrameSync failed sn=" << deviceSn << std::endl;
                }
            }
            rt.pipe->start(config, [this, deviceSn, deviceIndex](std::shared_ptr<ob::FrameSet> frameSet) { onFrameSet(deviceSn, deviceIndex, frameSet); });
        };

        for(auto &rt: secondary_) {
            startOne(rt);
        }
        for(auto &rt: primary_) {
            startOne(rt);
        }
        if(imuEnabled_) {
            for(auto &rt: all_) {
                startImuSensors(rt);
            }
        }

        if(cfg_.enableSync && all_.size() > 1) {
            try {
                collectionSetStage("start_enableDeviceClockSync");
                ctx_.enableDeviceClockSync(60000);
                std::cerr << "[collection] enableDeviceClockSync ok" << std::endl;
            }
            catch(...) {
                std::cerr << "[collection] enableDeviceClockSync failed" << std::endl;
            }
        }

        stopping_.store(false);
        capturing_.store(true);
        recording_.store(false);
        if(!fisheyeStatusLine.empty()) {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = fisheyeStatusLine;
        }

        softwareTriggerDevices_.clear();
        if(cfg_.enableSync) {
            for(const auto &rt: all_) {
                try {
                    const auto sc = rt.dev->getMultiDeviceSyncConfig();
                    if(sc.syncMode == OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING) {
                        softwareTriggerDevices_.push_back(rt.dev);
                    }
                }
                catch(...) {
                }
            }
        }
        useSoftwareTrigger_ = !softwareTriggerDevices_.empty();
        if(useSoftwareTrigger_) {
            triggerThread_ = std::thread([this]() { softwareTriggerLoop(); });
        }
        return true;
    }

    bool beginRecord(const fs::path &saveRoot, const std::string &subjectId, const std::string &taskName, int episodeN) {
        if(!capturing_.load() || recording_.load()) {
            return false;
        }
        joinCoordinatorThreadIfPossible();

        SessionState local;
        local.active = true;
        local.subjectId = subjectId;
        local.taskName = taskName;
        local.episodeN = episodeN;
        local.dest = saveRoot / subjectId / taskName / ("episode_" + std::to_string(episodeN));
        local.stepUs = sessionStepUs();
        local.maxAbsDiffUs = sessionMaxAbsDiffUs(local.stepUs, cfg_.enableSync);
        if(local.stepUs == 0) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            local.buffers.reserve(buffers_.size());
            for(const auto &kv: buffers_) {
                DeviceBuffer out;
                out.camKey = kv.second.camKey;
                out.params = kv.second.params;
                out.rgbDepthParam = kv.second.rgbDepthParam;
                out.rgbDepthParamValid = kv.second.rgbDepthParamValid;
                local.buffers.emplace(kv.first, std::move(out));
            }
        }

        local.saveCloud = std::find(typesSaving_.begin(), typesSaving_.end(), CollectDataType::CloudPoints) != typesSaving_.end();
        local.saveColorCloud = std::find(typesSaving_.begin(), typesSaving_.end(), CollectDataType::ColorCloudPoints) != typesSaving_.end();
        local.needDepthForCloud = local.saveCloud || local.saveColorCloud;
        local.needRgbForColorCloud = local.saveColorCloud;
        local.passthroughRgbMjpg = !cfg_.save.rgbH265 && isJpegLikeExt(cfg_.save.colorExt) && !local.needRgbForColorCloud;

        for(const auto t: typesSaving_) {
            if(t == CollectDataType::CloudPoints || t == CollectDataType::ColorCloudPoints) {
                continue;
            }
            local.typesPerCamSave.push_back(t);
        }
        local.typesAlign = local.typesPerCamSave;
        auto ensureAlignType = [&](CollectDataType t) {
            if(std::find(local.typesAlign.begin(), local.typesAlign.end(), t) == local.typesAlign.end()) {
                local.typesAlign.push_back(t);
            }
        };
        if(local.needDepthForCloud) {
            ensureAlignType(CollectDataType::Depth);
        }
        if(local.needRgbForColorCloud) {
            ensureAlignType(CollectDataType::RGB);
        }
        for(size_t i = 0; i < local.typesAlign.size(); ++i) {
            local.alignTypeIndex[local.typesAlign[i]] = i;
        }

        local.deviceSns.reserve(local.buffers.size());
        for(const auto &kv: local.buffers) {
            local.deviceSns.push_back(kv.first);
        }
        std::sort(local.deviceSns.begin(), local.deviceSns.end(), [&](const std::string &a, const std::string &b) {
            return local.buffers.at(a).camKey < local.buffers.at(b).camKey;
        });
        if(multiviewEnabled_ && !local.deviceSns.empty()) {
            local.refSn = local.deviceSns.front();
        }
        local.saveRgbTimesteps = std::find(local.typesPerCamSave.begin(), local.typesPerCamSave.end(), CollectDataType::RGB) != local.typesPerCamSave.end();
        local.saveDepthTimesteps = std::find(local.typesPerCamSave.begin(), local.typesPerCamSave.end(), CollectDataType::Depth) != local.typesPerCamSave.end();
        local.saveFisheye = fisheyeEnabled_ && activeFisheyeCameraCount_ > 0;
        local.fisheyeCameraCount = local.saveFisheye ? activeFisheyeCameraCount_ : 0;
        local.wroteFisheyeCameraParams.assign(local.fisheyeCameraCount, false);
        if(local.saveCloud || local.saveColorCloud) {
            loadCamWorldPoses(cfg_.initExtrinsicPath, local.camToWorld);
        }

        if(fs::exists(local.dest)) {
            bool reusableEmptyDir = false;
            try {
                reusableEmptyDir = fs::is_directory(local.dest) && (fs::directory_iterator(local.dest) == fs::directory_iterator());
            }
            catch(...) {
                reusableEmptyDir = false;
            }
            if(!reusableEmptyDir) {
                std::lock_guard<std::mutex> lock(mtx_);
                captureInfoLine_ = "Target episode dir already exists and is not empty: " + local.dest.string();
                return false;
            }
        }

        try {
            fs::create_directories(local.dest);
        }
        catch(const std::exception &ex) {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "Failed to create episode dir: " + std::string(ex.what());
            return false;
        }
        try {
            for(const auto &sn: local.deviceSns) {
                const auto &camKey = local.buffers.at(sn).camKey;
                for(const auto t: local.typesPerCamSave) {
                    fs::create_directories(local.dest / camKey / dataTypeLabel(t));
                }
                if(imuEnabled_) {
                    fs::create_directories(local.dest / camKey / "IMU");
                }
            }
            if(local.saveCloud) {
                fs::create_directories(local.dest / dataTypeLabel(CollectDataType::CloudPoints));
            }
            if(local.saveColorCloud) {
                fs::create_directories(local.dest / dataTypeLabel(CollectDataType::ColorCloudPoints));
            }
            if(local.saveFisheye) {
                for(size_t i = 0; i < local.fisheyeCameraCount; ++i) {
                    fs::create_directories(local.dest / "fisheye" / fisheyeCameraDirName(i) / "RGB");
                }
            }

            if(multiviewEnabled_) {
                writeParamsJson(local.dest, local.buffers, typesSaving_);
            }
            writeExtrinsicsJson(local.dest);
        }
        catch(const std::exception &ex) {
            try {
                fs::remove_all(local.dest);
            }
            catch(...) {
            }
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "Failed to prepare episode dir: " + std::string(ex.what());
            return false;
        }

        if(!openSessionTimestamps(local)) {
            if(local.timestampsOfs.is_open()) {
                local.timestampsOfs.close();
            }
            try {
                fs::remove_all(local.dest);
            }
            catch(...) {
            }
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "Failed to open timestamps.csv.tmp in " + local.dest.string();
            return false;
        }
        if(fisheyeEnabled_ && !fisheyeRecorder_.isRunning()) {
            if(local.timestampsOfs.is_open()) {
                local.timestampsOfs.close();
            }
            try {
                if(!local.timestampsTmpPath.empty() && fs::exists(local.timestampsTmpPath)) {
                    fs::remove(local.timestampsTmpPath);
                }
                fs::remove_all(local.dest);
            }
            catch(...) {
            }
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "Fisheye recorder is not running";
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            for(auto &kv: buffers_) {
                kv.second.accelSamples.clear();
                kv.second.gyroSamples.clear();
            }
            captureInfoLine_.clear();
            slotStatsLine_.clear();
            savedAlignedMaxDiffMs_ = 0.0;
            lastRecordedSeconds_   = 0.0;
        }
        {
            std::lock_guard<std::mutex> lock(streamSeqMtx_);
            streamNextSeq_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(lastSavedTsMtx_);
            lastSavedTs_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(previewTsMtx_);
            lastPreviewTs_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            closeSessionTimestampsLocked();
            session_ = std::move(local);
            coordRecordQueue_.clear();
            coordFisheyeQueue_.clear();
            if(!multiviewEnabled_) {
                session_.multiviewEos = true;
            }
            if(!fisheyeEnabled_) {
                session_.fisheyeEos = true;
            }
            coordCv_.notify_all();
        }
        passthroughRgbMjpg_.store(session_.passthroughRgbMjpg);
        startH265Encoders(session_);

        recordInputClosing_.store(false);
        multiviewEosNotified_.store(!multiviewEnabled_);
        hasData_.store(false);
        captureStartSteady_ = std::chrono::steady_clock::now();
        recording_.store(true);

        if(multiviewEnabled_) {
            clearRecordQueue();
            ensureRecordWorkerRunning();
        }
        ensureWriteWorkersRunning();
        startCoordinatorThread();

        if(fisheyeEnabled_) {
            if(fisheyeRecordThread_.joinable()) {
                fisheyeRecordThread_.join();
            }
            fisheyeRecordThread_ = std::thread([this]() { fisheyeRecordLoop(); });
        }
        std::cerr << "[collection] record begin" << std::endl;
        return true;
    }

    void stopRecording() {
        if(!recording_.load()) {
            return;
        }
        const auto durMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - captureStartSteady_).count();
        recording_.store(false);
        if(fisheyeRecordThread_.joinable()) {
            fisheyeRecordThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            lastRecordedSeconds_ = static_cast<double>(durMs) / 1000.0;
        }
        if(multiviewEnabled_) {
            recordInputClosing_.store(true);
            bool idle = false;
            {
                std::lock_guard<std::mutex> lock(recordMtx_);
                idle = recordQueue_.empty() && recordInFlight_ == 0;
            }
            if(idle) {
                notifyMultiviewEos();
            }
        }
        else {
            notifyMultiviewEos();
        }
        notifyFisheyeEos();
        const std::string captureInfoSnapshot = buildCaptureInfoSnapshotLocked(durMs);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = captureInfoSnapshot;
        }
        std::cerr << "[collection] record stop" << std::endl;
    }

    void stopIfRunning() {
        if(!capturing_.load()) {
            fisheyeRecorder_.stop();
            return;
        }
        stopping_.store(true);
        recording_.store(false);
        recordInputClosing_.store(true);
        if(fisheyeRecordThread_.joinable()) {
            fisheyeRecordThread_.join();
        }
        notifyFisheyeEos();
        if(multiviewEnabled_) {
            waitRecordWorkerIdle();
            notifyMultiviewEos();
        }
        else {
            notifyMultiviewEos();
        }
        joinCoordinatorThreadIfPossible();
        waitWriteQueueIdle();
        stopWriteWorkers();
        stopRecordWorker();
        if(triggerThread_.joinable()) {
            triggerThread_.join();
        }
        stopImuSensors();
        for(auto &rt: all_) {
            try {
                rt.pipe->stop();
            }
            catch(...) {
            }
        }
        fisheyeRecorder_.stop();
        capturing_.store(false);
        recording_.store(false);
        hasData_.store(false);
        passthroughRgbMjpg_.store(false);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            buffers_.clear();
            captureInfoLine_.clear();
            slotStatsLine_.clear();
            savedAlignedMaxDiffMs_ = 0.0;
        }
        {
            std::lock_guard<std::mutex> lock(previewTsMtx_);
            lastPreviewTs_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            closeSessionTimestampsLocked();
            session_ = SessionState{};
            coordRecordQueue_.clear();
            coordFisheyeQueue_.clear();
            coordCv_.notify_all();
        }
        std::cerr << "[collection] pipelines stopped" << std::endl;
    }

    bool confirmCurrentSession() {
        if(!isDrainComplete()) {
            return false;
        }
        joinCoordinatorThreadIfPossible();
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            if(!session_.active) {
                return false;
            }
            closeSessionTimestampsLocked();
            session_ = SessionState{};
            coordRecordQueue_.clear();
            coordFisheyeQueue_.clear();
            coordCv_.notify_all();
        }
        hasData_.store(false);
        recordInputClosing_.store(false);
        multiviewEosNotified_.store(false);
        passthroughRgbMjpg_.store(false);
        return true;
    }

    bool discardCurrentSession(std::string *errorMessage = nullptr) {
        if(!isDrainComplete()) {
            if(errorMessage) {
                *errorMessage = "Session is still draining";
            }
            return false;
        }
        joinCoordinatorThreadIfPossible();
        fs::path dest;
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            if(!session_.active) {
                if(errorMessage) {
                    *errorMessage = "No active session";
                }
                return false;
            }
            dest = session_.dest;
        }
        try {
            if(!dest.empty() && fs::exists(dest)) {
                fs::remove_all(dest);
            }
        }
        catch(const std::exception &ex) {
            if(errorMessage) {
                *errorMessage = ex.what();
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            closeSessionTimestampsLocked();
            session_ = SessionState{};
            coordRecordQueue_.clear();
            coordFisheyeQueue_.clear();
            coordCv_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_.clear();
            slotStatsLine_.clear();
            savedAlignedMaxDiffMs_ = 0.0;
        }
        hasData_.store(false);
        recordInputClosing_.store(false);
        multiviewEosNotified_.store(false);
        passthroughRgbMjpg_.store(false);
        return true;
    }

private:
    struct RecordTask {
        std::string       sn;
        CollectDataType   type = CollectDataType::RGB;
        uint64_t          tsUs = 0;
        uint64_t          seq = 0;
        DetachedVideoFrame detached;
    };

    struct ProcessedRecord {
        std::string sn;
        CollectDataType type = CollectDataType::RGB;
        uint64_t tsUs = 0;
        uint64_t seq = 0;
        cv::Mat frame;
        std::shared_ptr<std::vector<uint8_t>> encodedBytes;
        OBFormat encodedFormat = OB_FORMAT_UNKNOWN;
        float valueScale = 0.0f;
    };

    struct StreamPacket {
        uint64_t tsUs = 0;
        cv::Mat  frame;
        std::shared_ptr<std::vector<uint8_t>> encodedBytes;
        OBFormat encodedFormat = OB_FORMAT_UNKNOWN;
        float    valueScale = 0.0f;
    };

    struct StreamState {
        uint64_t nextSeq = 0;
        std::map<uint64_t, StreamPacket> bySeq;
        std::deque<StreamPacket> committed;
        uint64_t maxTsUs = 0;
        bool eos = false;
    };

    struct WriteTask {
        std::function<void()> fn;
    };

    struct H265FrameItem {
        cv::Mat frame;
        std::string frameIndex;
        uint64_t tsUs = 0;
    };

    class H265Encoder {
    public:
        H265Encoder(fs::path outputPath, int width, int height, int fps, int threads, SaveOptions options, size_t queueMax)
            : outputPath_(std::move(outputPath)),
              width_(width),
              height_(height),
              fps_(std::max(1, fps)),
              threads_(std::max(0, threads)),
              options_(std::move(options)),
              queueMax_(std::max<size_t>(64, queueMax)) {}

        ~H265Encoder() {
            stop();
        }

        H265Encoder(const H265Encoder &) = delete;
        H265Encoder &operator=(const H265Encoder &) = delete;

        void start() {
            worker_ = std::thread([this]() { loop(); });
        }

        void enqueue(std::string frameIndex, uint64_t tsUs, cv::Mat frame) {
            if(frame.empty() || frame.cols != width_ || frame.rows != height_ || frame.type() != CV_8UC3) {
                return;
            }
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() {
                    return stop_ || queue_.size() < queueMax_;
                });
                if(stop_) {
                    return;
                }
                queue_.push_back(H265FrameItem{ std::move(frame), std::move(frameIndex), tsUs });
                queued_.fetch_add(1);
            }
            cv_.notify_one();
        }

        void stop() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                stop_ = true;
                cv_.notify_all();
            }
            if(worker_.joinable()) {
                worker_.join();
            }
        }

        bool idle() const {
            return queued_.load() == 0 && inFlight_.load() == 0;
        }

    private:
        std::string buildCommand() const {
            const std::string codec = resolvedH265Codec(options_);
            const std::string logLevel = trimString(options_.h265LogLevel).empty() ? "info" : trimString(options_.h265LogLevel);
            std::ostringstream cmd;
            cmd << "ffmpeg -hide_banner -loglevel " << shellQuote(logLevel) << " -stats -y"
                << " -f rawvideo -pix_fmt bgr24"
                << " -s " << width_ << "x" << height_
                << " -r " << fps_
                << " -i - -an";

            if(h265CodecIsVaapi(codec)) {
                const std::string dev = trimString(options_.h265HwDevice).empty() ? "/dev/dri/renderD128" : trimString(options_.h265HwDevice);
                cmd << " -vaapi_device " << shellQuote(dev)
                    << " -vf format=nv12,hwupload";
            }
            else if(h265CodecIsQsv(codec)) {
                cmd << " -vf format=nv12";
            }

            cmd << " -c:v " << shellQuote(codec);

            const std::string preset = trimString(options_.h265Preset);
            if(!preset.empty() && (h265CodecIsSoftware(codec) || h265CodecIsNvenc(codec) || h265CodecIsQsv(codec))) {
                cmd << " -preset " << shellQuote(preset);
            }

            const int quality = std::max(0, std::min(51, options_.h265Crf));
            if(h265CodecIsSoftware(codec)) {
                cmd << " -crf " << quality;
            }
            else if(h265CodecIsNvenc(codec)) {
                cmd << " -cq " << quality << " -pix_fmt yuv420p";
            }
            else if(h265CodecIsVaapi(codec)) {
                cmd << " -qp " << quality;
            }
            else if(h265CodecIsQsv(codec)) {
                cmd << " -global_quality " << quality;
            }
            else {
                cmd << " -q:v " << quality;
            }

            if(threads_ > 0 && h265CodecIsSoftware(codec)) {
                cmd << " -threads " << threads_;
            }
            if(h265OutputIsRawStream(outputPath_)) {
                cmd << " -f hevc";
            }
            else if(endsWith(outputPath_.string(), ".mp4") || endsWith(outputPath_.string(), ".mov")) {
                cmd << " -tag:v hvc1";
            }
            cmd << " " << shellQuote(outputPath_.string());
            return cmd.str();
        }

        void logStart(const std::string &command) const {
            const std::string codec = resolvedH265Codec(options_);
            std::cerr << "[collection][h265] start encoder"
                      << " output=" << outputPath_
                      << " mode=" << (h265CodecIsHardware(codec) ? "hardware" : "software")
                      << " codec=" << codec
                      << " size=" << width_ << "x" << height_
                      << " fps=" << fps_
                      << " threads=" << threads_
                      << " preset=" << (trimString(options_.h265Preset).empty() ? "(none)" : trimString(options_.h265Preset))
                      << " quality=" << std::max(0, std::min(51, options_.h265Crf));
            if(h265CodecIsVaapi(codec) || h265CodecIsQsv(codec)) {
                std::cerr << " hwDevice=" << (trimString(options_.h265HwDevice).empty() ? "/dev/dri/renderD128" : trimString(options_.h265HwDevice));
            }
            std::cerr << " ffmpegLogLevel=" << (trimString(options_.h265LogLevel).empty() ? "info" : trimString(options_.h265LogLevel))
                      << std::endl;
            std::cerr << "[collection][h265] ffmpeg command: " << command << std::endl;
        }

        void logExitFailure(int status) const {
            const std::string codec = resolvedH265Codec(options_);
            std::cerr << "[collection][h265] ffmpeg encoder failed"
                      << " output=" << outputPath_
                      << " status=" << status
                      << " mode=" << (h265CodecIsHardware(codec) ? "hardware" : "software")
                      << " codec=" << codec << std::endl;
            if(h265CodecIsHardware(codec)) {
                std::cerr << "[collection][h265] hardware encoder debug: run `ffmpeg -hide_banner -encoders | grep -Ei 'hevc|h265|265'`, "
                          << "`ffmpeg -hide_banner -h encoder=" << codec << "`, and check GPU device/driver permissions";
                if(h265CodecIsVaapi(codec) || h265CodecIsQsv(codec)) {
                    std::cerr << " plus `ls -l " << (trimString(options_.h265HwDevice).empty() ? "/dev/dri/renderD128" : trimString(options_.h265HwDevice)) << "`";
                }
                std::cerr << std::endl;
            }
        }

        void loop() {
            try {
                fs::create_directories(outputPath_.parent_path());
            }
            catch(...) {
                return;
            }

            const fs::path timestampPath = outputPath_.string() + ".timestamps.csv";
            std::ofstream timestampOfs(timestampPath, std::ios::out | std::ios::trunc);
            if(timestampOfs.is_open()) {
                timestampOfs << "video_frame_index,frame_index,rgb_timestamp_us\n";
            }
            else {
                std::cerr << "[collection][h265] warning: failed to open timestamp sidecar: " << timestampPath << std::endl;
            }

            const std::string command = buildCommand();
            logStart(command);
            FILE *pipe = popen(command.c_str(), "w");
            if(!pipe) {
                std::cerr << "[collection][h265] failed to start ffmpeg process: " << outputPath_ << std::endl;
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    stop_ = true;
                    queue_.clear();
                    queued_.store(0);
                }
                cv_.notify_all();
                return;
            }

            while(true) {
                H265FrameItem item;
                {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_.wait(lock, [&]() {
                        return stop_ || !queue_.empty();
                    });
                    if(queue_.empty() && stop_) {
                        break;
                    }
                    if(queue_.empty()) {
                        continue;
                    }
                    item = std::move(queue_.front());
                    queue_.pop_front();
                    queued_.fetch_sub(1);
                    inFlight_.fetch_add(1);
                    cv_.notify_all();
                }

                const cv::Mat &frame = item.frame;
                if(frame.isContinuous()) {
                    const size_t bytes = static_cast<size_t>(frame.total()) * frame.elemSize();
                    (void)fwrite(frame.data, 1, bytes, pipe);
                }
                else {
                    const size_t rowBytes = static_cast<size_t>(frame.cols) * frame.elemSize();
                    for(int r = 0; r < frame.rows; ++r) {
                        (void)fwrite(frame.ptr(r), 1, rowBytes, pipe);
                    }
                }
                if(timestampOfs.is_open()) {
                    timestampOfs << encodedFrameCount_ << "," << item.frameIndex << "," << item.tsUs << "\n";
                }
                encodedFrameCount_++;
                inFlight_.fetch_sub(1);
            }

            if(timestampOfs.is_open()) {
                timestampOfs.flush();
                timestampOfs.close();
                std::cerr << "[collection][h265] timestamp sidecar written path=" << timestampPath
                          << " frames=" << encodedFrameCount_ << std::endl;
            }

            const int status = pclose(pipe);
            if(status != 0) {
                logExitFailure(status);
            }
            else {
                std::cerr << "[collection][h265] encoder finished output=" << outputPath_ << std::endl;
            }
        }

        fs::path outputPath_;
        int width_ = 0;
        int height_ = 0;
        int fps_ = 30;
        int threads_ = 0;
        SaveOptions options_;
        size_t queueMax_ = 1024;
        mutable std::mutex mtx_;
        std::condition_variable cv_;
        std::deque<H265FrameItem> queue_;
        std::thread worker_;
        bool stop_ = false;
        std::atomic<size_t> queued_{ 0 };
        std::atomic<int> inFlight_{ 0 };
        size_t encodedFrameCount_ = 0;
    };

    struct SessionState {
        bool active = false;
        fs::path dest;
        std::string subjectId;
        std::string taskName;
        int episodeN = 0;
        std::unordered_map<std::string, DeviceBuffer> buffers;
        std::vector<std::string> deviceSns;
        std::vector<CollectDataType> typesPerCamSave;
        std::vector<CollectDataType> typesAlign;
        std::unordered_map<CollectDataType, size_t> alignTypeIndex;
        std::string refSn;
        uint64_t stepUs = 0;
        uint64_t maxAbsDiffUs = 0;
        bool saveCloud = false;
        bool saveColorCloud = false;
        bool needDepthForCloud = false;
        bool needRgbForColorCloud = false;
        bool passthroughRgbMjpg = false;
        bool saveFisheye = false;
        bool saveRgbTimesteps = false;
        bool saveDepthTimesteps = false;
        size_t fisheyeCameraCount = 0;
        std::vector<bool> wroteFisheyeCameraParams;
        std::unordered_map<std::string, CamWorldPose> camToWorld;
        std::unordered_map<std::string, std::unordered_map<CollectDataType, StreamState>> streams;
        std::deque<FisheyeFrameSet> fisheyeSets;
        bool multiviewEos = false;
        bool fisheyeEos = false;
        bool coordinatorDone = false;
        bool timestampsFinalized = false;
        bool imuWritten = false;
        fs::path timestampsPath;
        fs::path timestampsTmpPath;
        std::ofstream timestampsOfs;
        bool timestampsOpen = false;
        uint64_t fisheyeOnlyNextTargetUs = 0;
        bool fisheyeOnlyTargetInit = false;
        uint64_t lastEmittedFisheyeTs = 0;
        bool hasLastEmittedFisheyeTs = false;
        size_t nextFrameIndex = 0;
        size_t refFrameCount = 0;
        uint64_t firstRefTs = 0;
        uint64_t lastRefTs = 0;
        size_t fisheyeCapturedSets = 0;
        size_t alignedRef = 0;
        size_t fullAligned = 0;
        size_t missingAligned = 0;
        size_t prevMissingIndex = std::numeric_limits<size_t>::max();
        double minMissingMs = 0.0;
        double maxDiffMs = 0.0;
        std::vector<uint64_t> alignedCenters;
    };

    struct ImuSensorHandle {
        std::string              sn;
        std::shared_ptr<ob::Sensor> sensor;
        OBSensorType             type = OB_SENSOR_UNKNOWN;
    };

    void drainReadyPacketsLocked(const std::string &sn, CollectDataType type, StreamState &state) {
        while(true) {
            auto it = state.bySeq.find(state.nextSeq);
            if(it == state.bySeq.end()) {
                break;
            }
            StreamPacket packet = std::move(it->second);
            state.bySeq.erase(it);
            state.committed.push_back(std::move(packet));
            if(!state.committed.empty()) {
                state.maxTsUs = state.committed.back().tsUs;
                if(sn == session_.refSn && type == refType_) {
                    session_.refFrameCount++;
                    if(session_.firstRefTs == 0) {
                        session_.firstRefTs = state.maxTsUs;
                    }
                    session_.lastRefTs = state.maxTsUs;
                }
            }
            state.nextSeq++;
        }
    }

    bool flushEosSequenceGapsLocked() {
        bool advanced = false;
        for(auto &kv: session_.streams) {
            for(auto &typeKv: kv.second) {
                auto &state = typeKv.second;
                if(!state.eos || state.bySeq.empty()) {
                    continue;
                }
                if(state.bySeq.find(state.nextSeq) == state.bySeq.end()) {
                    state.nextSeq = state.bySeq.begin()->first;
                    drainReadyPacketsLocked(kv.first, typeKv.first, state);
                    advanced = true;
                }
            }
        }
        return advanced;
    }

    bool hasPendingBySeqLocked() const {
        for(const auto &kv: session_.streams) {
            for(const auto &typeKv: kv.second) {
                if(!typeKv.second.bySeq.empty()) {
                    return true;
                }
            }
        }
        return false;
    }

    void releaseSessionFrameCachesLocked() {
        for(auto &kv: session_.streams) {
            for(auto &typeKv: kv.second) {
                typeKv.second.bySeq.clear();
                typeKv.second.committed.clear();
            }
        }
        session_.fisheyeSets.clear();
    }

    static size_t softWriterThreadCount(const AppConfig &cfg) {
        if(cfg.writerThreads > 0) {
            return static_cast<size_t>(std::max(12, std::min(24, cfg.writerThreads)));
        }
        return saveWorkerCount();
    }

    static bool streamPacketHasPayload(CollectDataType type, const StreamPacket &packet) {
        if(type == CollectDataType::RGB) {
            return !packet.frame.empty() || (packet.encodedBytes && !packet.encodedBytes->empty());
        }
        return !packet.frame.empty();
    }

    uint64_t sessionStepUs() const {
        int fps = cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_);
        if(fps <= 0) {
            fps = 30;
        }
        return static_cast<uint64_t>(1000000.0 / static_cast<double>(fps));
    }

    static uint64_t sessionMaxAbsDiffUs(uint64_t stepUs, bool strictSync) {
        (void)strictSync;
        return sessionCrossTypeMaxAbsDiffUs(stepUs);
    }

    static uint64_t sessionCrossTypeMaxAbsDiffUs(uint64_t stepUs) {
        const uint64_t halfWinUs = stepUs / 2;
        const uint64_t tolUs     = std::max<uint64_t>(2000, stepUs / 10);
        return halfWinUs + tolUs;
    }

    bool openSessionTimestamps(SessionState &session) {
        session.timestampsPath = session.dest / "timestamps.csv";
        session.timestampsTmpPath = session.timestampsPath;
        session.timestampsTmpPath += ".tmp";
        session.timestampsOfs.open(session.timestampsTmpPath, std::ios::out | std::ios::trunc);
        if(!session.timestampsOfs.is_open()) {
            return false;
        }
        session.timestampsOpen = true;
        std::vector<std::string> header;
        if(multiviewEnabled_) {
            header.reserve(2 + session.deviceSns.size() * 2 + session.fisheyeCameraCount);
            header.push_back("frame_index");
            header.push_back("ref_timestamp_us");
            for(const auto &sn: session.deviceSns) {
                const auto &camKey = session.buffers.at(sn).camKey;
                if(session.saveRgbTimesteps) {
                    header.push_back("cam" + camKey + "_rgb时间戳(us)");
                }
                if(session.saveDepthTimesteps) {
                    header.push_back("cam" + camKey + "_depth时间戳(us)");
                }
            }
        }
        else {
            header.push_back("frame_index");
            header.push_back("ref_timestamp_us");
        }
        for(size_t cameraIdx = 0; cameraIdx < session.fisheyeCameraCount; ++cameraIdx) {
            header.push_back("fisheye" + std::to_string(cameraIdx) + "_timestamp_us");
        }
        header.push_back("rgbd_max_diff_ms");
        header.push_back("all_modalities_max_diff_ms");
        writeCsvRow(session.timestampsOfs, header);
        return static_cast<bool>(session.timestampsOfs);
    }

    static void writeCsvRow(std::ofstream &ofs, const std::vector<std::string> &row) {
        for(size_t i = 0; i < row.size(); ++i) {
            if(i > 0) {
                ofs << ",";
            }
            ofs << row[i];
        }
        ofs << "\n";
    }

    void closeSessionTimestampsLocked() {
        if(session_.timestampsOfs.is_open()) {
            session_.timestampsOfs.flush();
            session_.timestampsOfs.close();
        }
        session_.timestampsOpen = false;
    }

    void finalizeSessionTimestampsLocked() {
        if(session_.timestampsFinalized) {
            return;
        }
        closeSessionTimestampsLocked();
        if(!session_.timestampsTmpPath.empty()) {
            try {
                fs::rename(session_.timestampsTmpPath, session_.timestampsPath);
            }
            catch(...) {
            }
        }
        session_.timestampsFinalized = true;
    }

    static std::string buildCaptureInfoFromSession(const SessionState &session, double seconds) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        double avgFps = 0.0;
        if(seconds > 0.0 && session.refFrameCount > 0) {
            avgFps = static_cast<double>(session.refFrameCount) / seconds;
        }
        oss << "Frames=" << session.refFrameCount << "  Time=" << seconds << "s  AvgFPS=" << std::setprecision(2) << avgFps;
        if(session.lastRefTs > session.firstRefTs) {
            oss << "  ts_span=" << ((session.lastRefTs - session.firstRefTs) / 1000.0) << "ms";
        }
        if(session.fisheyeCapturedSets > 0) {
            oss << "  FisheyeFrames=" << session.fisheyeCapturedSets;
        }
        return oss.str();
    }

    std::string buildCaptureInfoSnapshotLocked(int durMs) const {
        std::lock_guard<std::mutex> coordLock(coordMtx_);
        if(!session_.active) {
            return "";
        }
        return buildCaptureInfoFromSession(session_, static_cast<double>(durMs) / 1000.0);
    }

    static std::string buildSlotStatsFromSession(const SessionState &session) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << "AlignedRef=" << session.alignedRef << " Full=" << session.fullAligned << " Missing=" << session.missingAligned;
        if(session.minMissingMs > 0.0) {
            oss << " MinMissingInterval=" << std::setprecision(2) << session.minMissingMs << "ms";
        }
        if(session.maxDiffMs > 0.0) {
            oss << " MaxSameIndexDiff=" << std::setprecision(3) << session.maxDiffMs << "ms";
        }
        return oss.str();
    }

    uint64_t nextRecordSeq(const std::string &deviceSn, CollectDataType type) {
        const std::string key = deviceSn + ":" + std::string(dataTypeLabel(type));
        std::lock_guard<std::mutex> lock(streamSeqMtx_);
        uint64_t &v = streamNextSeq_[key];
        const uint64_t out = v;
        v++;
        return out;
    }

    void onImuFrame(const std::string &deviceSn, bool isGyro, const std::shared_ptr<ob::Frame> &frame) {
        if(!imuEnabled_ || !recording_.load() || !frame) {
            return;
        }
        uint64_t ts = frameTimestampUs(frame);
        if(ts == 0) {
            if(isGyro) {
                auto gyroFrame = frame->as<ob::GyroFrame>();
                if(gyroFrame) {
                    ts = gyroFrame->getTimeStampUs();
                }
            }
            else {
                auto accelFrame = frame->as<ob::AccelFrame>();
                if(accelFrame) {
                    ts = accelFrame->getTimeStampUs();
                }
            }
        }
        if(ts == 0) {
            return;
        }

        ImuSample sample;
        sample.tsUs = ts;
        if(isGyro) {
            auto gyroFrame = frame->as<ob::GyroFrame>();
            if(!gyroFrame) {
                return;
            }
            const auto v = gyroFrame->getValue();
            sample.x = v.x;
            sample.y = v.y;
            sample.z = v.z;
        }
        else {
            auto accelFrame = frame->as<ob::AccelFrame>();
            if(!accelFrame) {
                return;
            }
            const auto v = accelFrame->getValue();
            sample.x = v.x;
            sample.y = v.y;
            sample.z = v.z;
        }

        std::lock_guard<std::mutex> lock(mtx_);
        auto it = buffers_.find(deviceSn);
        if(it == buffers_.end()) {
            return;
        }
        if(isGyro) {
            it->second.gyroSamples.push_back(sample);
        }
        else {
            it->second.accelSamples.push_back(sample);
        }
    }

    void startImuSensors(DeviceRuntime &rt) {
        auto startOne = [&](OBSensorType sensorType, bool isGyro) {
            std::shared_ptr<ob::Sensor> sensor;
            try {
                sensor = rt.dev->getSensor(sensorType);
            }
            catch(...) {
                sensor.reset();
            }
            if(!sensor) {
                return;
            }

            std::shared_ptr<ob::StreamProfileList> list;
            try {
                list = sensor->getStreamProfileList();
            }
            catch(...) {
                return;
            }
            if(!list || list->getCount() == 0) {
                return;
            }

            std::shared_ptr<ob::StreamProfile> profile;
            try {
                profile = list->getProfile(OB_PROFILE_DEFAULT);
            }
            catch(...) {
                try {
                    profile = list->getProfile(0);
                }
                catch(...) {
                    profile.reset();
                }
            }
            if(!profile) {
                return;
            }

            try {
                const std::string sn = rt.cfg.sn;
                sensor->start(profile, [this, sn, isGyro](std::shared_ptr<ob::Frame> frame) { onImuFrame(sn, isGyro, frame); });
                imuSensors_.push_back(ImuSensorHandle{ sn, sensor, sensorType });
            }
            catch(...) {
            }
        };

        startOne(OB_SENSOR_ACCEL, false);
        startOne(OB_SENSOR_GYRO, true);
    }

    void stopImuSensors() {
        for(auto &h: imuSensors_) {
            if(!h.sensor) {
                continue;
            }
            try {
                h.sensor->stop();
            }
            catch(...) {
            }
        }
        imuSensors_.clear();
    }

    void ensureRecordWorkerRunning() {
        std::lock_guard<std::mutex> lock(recordMtx_);
        if(!recordWorkers_.empty()) {
            return;
        }
        recordStop_.store(false);
        recordWorkers_.reserve(recordWorkerCount());
        for(size_t i = 0; i < recordWorkerCount(); ++i) {
            recordWorkers_.emplace_back([this]() { recordWorkerLoop(); });
        }
    }

    void stopRecordWorker() {
        {
            std::lock_guard<std::mutex> lock(recordMtx_);
            recordStop_.store(true);
            recordQueue_.clear();
            recordCv_.notify_all();
            recordDrainCv_.notify_all();
        }
        for(auto &worker: recordWorkers_) {
            if(worker.joinable()) {
                worker.join();
            }
        }
        recordWorkers_.clear();
    }

    void clearRecordQueue() {
        std::lock_guard<std::mutex> lock(recordMtx_);
        recordQueue_.clear();
        recordCv_.notify_all();
        recordDrainCv_.notify_all();
    }

    void waitRecordWorkerIdle() {
        std::unique_lock<std::mutex> lock(recordMtx_);
        recordDrainCv_.wait(lock, [&]() {
            return recordQueue_.empty() && recordInFlight_ == 0;
        });
    }

    void enqueueRecordTask(RecordTask &&task) {
        {
            std::lock_guard<std::mutex> lock(recordMtx_);
            if(recordStop_.load()) {
                return;
            }
            recordQueue_.push_back(std::move(task));
        }
        recordCv_.notify_one();
    }

    void recordWorkerLoop() {
        for(;;) {
            std::vector<RecordTask> tasks;
            tasks.reserve(16);
            {
                std::unique_lock<std::mutex> lock(recordMtx_);
                recordCv_.wait(lock, [&]() {
                    return recordStop_.load() || !recordQueue_.empty();
                });
                if(recordStop_.load() && recordQueue_.empty()) {
                    break;
                }
                if(recordQueue_.empty()) {
                    continue;
                }
                while(!recordQueue_.empty() && tasks.size() < 16) {
                    tasks.push_back(std::move(recordQueue_.front()));
                    recordQueue_.pop_front();
                }
                recordInFlight_ += static_cast<int>(tasks.size());
                recordCv_.notify_all();
            }

            for(auto &task: tasks) {
                cv::Mat copied;
                float   valueScale = 0.0f;
                bool    okCopy = false;
                if(task.type == CollectDataType::RGB && passthroughRgbMjpg_.load()
                   && task.detached.format == OB_FORMAT_MJPG && !task.detached.data.empty()) {
                    enqueueProcessedRecord(ProcessedRecord{
                        std::move(task.sn),
                        task.type,
                        task.tsUs,
                        task.seq,
                        cv::Mat{},
                        std::make_shared<std::vector<uint8_t>>(std::move(task.detached.data)),
                        task.detached.format,
                        0.0f
                    });
                    continue;
                }
                if(task.type == CollectDataType::RGB) {
                    okCopy = copyDetachedColorToBgr(task.detached, copied);
                }
                else if(task.type == CollectDataType::Depth) {
                    okCopy = copyDetachedVideoToRawMat(task.detached, copied, &valueScale);
                }
                else {
                    okCopy = copyDetachedVideoToRawMat(task.detached, copied);
                }
                if(okCopy && !copied.empty()) {
                    enqueueProcessedRecord(ProcessedRecord{
                        std::move(task.sn),
                        task.type,
                        task.tsUs,
                        task.seq,
                        std::move(copied),
                        nullptr,
                        OB_FORMAT_UNKNOWN,
                        valueScale
                    });
                }
            }

            bool notifyEos = false;
            {
                std::lock_guard<std::mutex> lock(recordMtx_);
                recordInFlight_ -= static_cast<int>(tasks.size());
                if(recordQueue_.empty() && recordInFlight_ == 0) {
                    recordDrainCv_.notify_all();
                    if(recordInputClosing_.load()) {
                        notifyEos = true;
                    }
                }
            }
            if(notifyEos) {
                notifyMultiviewEos();
            }
        }
    }

    void ensureWriteWorkersRunning() {
        std::lock_guard<std::mutex> lock(writeMtx_);
        if(!writeWorkers_.empty()) {
            return;
        }
        writeStop_.store(false);
        const size_t workerN = softWriterThreadCount(cfg_);
        writeWorkers_.reserve(workerN);
        for(size_t i = 0; i < workerN; ++i) {
            writeWorkers_.emplace_back([this]() { writeWorkerLoop(); });
        }
    }

    void stopWriteWorkers() {
        {
            std::lock_guard<std::mutex> lock(writeMtx_);
            writeStop_.store(true);
            writeCv_.notify_all();
        }
        for(auto &worker: writeWorkers_) {
            if(worker.joinable()) {
                worker.join();
            }
        }
        writeWorkers_.clear();
        queuedWriteCount_.store(0);
        writeInFlight_.store(0);
    }

    void waitWriteQueueIdle() {
        while(queuedWriteCount_.load() > 0 || writeInFlight_.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    void enqueueWriteTask(WriteTask &&task) {
        {
            std::unique_lock<std::mutex> lock(writeMtx_);
            writeCv_.wait(lock, [&]() {
                return writeStop_.load() || writeQueue_.size() < writeQueueMax_;
            });
            if(writeStop_.load()) {
                return;
            }
            writeQueue_.push_back(std::move(task));
            queuedWriteCount_.fetch_add(1);
        }
        writeCv_.notify_one();
    }

    void writeWorkerLoop() {
        while(true) {
            WriteTask task;
            {
                std::unique_lock<std::mutex> lock(writeMtx_);
                writeCv_.wait(lock, [&]() {
                    return writeStop_.load() || !writeQueue_.empty();
                });
                if(writeQueue_.empty() && writeStop_.load()) {
                    return;
                }
                if(writeQueue_.empty()) {
                    continue;
                }
                task = std::move(writeQueue_.front());
                writeQueue_.pop_front();
                queuedWriteCount_.fetch_sub(1);
                writeInFlight_.fetch_add(1);
                writeCv_.notify_all();
            }
            try {
                if(task.fn) {
                    task.fn();
                }
            }
            catch(...) {
            }
            writeInFlight_.fetch_sub(1);
        }
    }

    int h265ThreadsForCamera(const std::string &sn, const std::string &camKey) const {
        auto it = cfg_.save.h265ThreadsByCamera.find(sn);
        if(it != cfg_.save.h265ThreadsByCamera.end()) {
            return it->second;
        }
        it = cfg_.save.h265ThreadsByCamera.find(camKey);
        if(it != cfg_.save.h265ThreadsByCamera.end()) {
            return it->second;
        }
        return cfg_.save.h265Threads;
    }

    void startH265Encoders(const SessionState &session) {
        stopH265Encoders();
        if(!cfg_.save.rgbH265 || !session.saveRgbTimesteps) {
            return;
        }

        std::lock_guard<std::mutex> lock(h265Mtx_);
        const size_t queueMax = std::max<size_t>(256, writeQueueMax_ / 2);
        for(const auto &sn: session.deviceSns) {
            const auto itBuf = session.buffers.find(sn);
            if(itBuf == session.buffers.end()) {
                continue;
            }
            const auto itParams = itBuf->second.params.find(CollectDataType::RGB);
            if(itParams == itBuf->second.params.end() || !itParams->second.valid
               || itParams->second.width <= 0 || itParams->second.height <= 0) {
                continue;
            }
            const std::string &camKey = itBuf->second.camKey;
            const fs::path outPath = session.dest / camKey / dataTypeLabel(CollectDataType::RGB) / h265OutputFileName(cfg_.save);
            auto encoder = std::make_unique<H265Encoder>(
                outPath,
                itParams->second.width,
                itParams->second.height,
                itParams->second.fps > 0 ? itParams->second.fps : (cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_)),
                h265ThreadsForCamera(sn, camKey),
                cfg_.save,
                queueMax);
            encoder->start();
            h265Encoders_[sn] = std::move(encoder);
        }
        h265EncodingActive_.store(!h265Encoders_.empty());
    }

    void stopH265Encoders() {
        std::unordered_map<std::string, std::unique_ptr<H265Encoder>> encoders;
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            encoders.swap(h265Encoders_);
        }
        for(auto &kv: encoders) {
            if(kv.second) {
                kv.second->stop();
            }
        }
        h265EncodingActive_.store(false);
    }

    void enqueueH265Frame(const std::string &sn, const std::string &frameIndex, uint64_t tsUs, cv::Mat frame) {
        H265Encoder *encoder = nullptr;
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            auto it = h265Encoders_.find(sn);
            if(it != h265Encoders_.end()) {
                encoder = it->second.get();
            }
        }
        if(encoder) {
            encoder->enqueue(frameIndex, tsUs, std::move(frame));
        }
    }

    void startCoordinatorThread() {
        if(coordinatorThread_.joinable()) {
            coordinatorThread_.join();
        }
        coordinatorThread_ = std::thread([this]() {
            coordinatorLoop();
            stopH265Encoders();
        });
    }

    void joinCoordinatorThreadIfPossible() {
        if(coordinatorThread_.joinable()) {
            coordinatorThread_.join();
        }
    }

    void enqueueProcessedRecord(ProcessedRecord &&item) {
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            if(!session_.active) {
                return;
            }
            coordRecordQueue_.push_back(std::move(item));
        }
        coordCv_.notify_one();
    }

    void enqueueFisheyeFrameSet(FisheyeFrameSet &&sample) {
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            if(!session_.active) {
                return;
            }
            coordFisheyeQueue_.push_back(std::move(sample));
        }
        coordCv_.notify_one();
    }

    void notifyMultiviewEos() {
        if(multiviewEosNotified_.exchange(true)) {
            return;
        }
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return;
        }
        session_.multiviewEos = true;
        for(auto &kv: session_.streams) {
            for(auto &typeKv: kv.second) {
                typeKv.second.eos = true;
            }
        }
        coordCv_.notify_one();
    }

    void notifyFisheyeEos() {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return;
        }
        session_.fisheyeEos = true;
        coordCv_.notify_one();
    }

    static bool pickNearestPacket(const std::deque<StreamPacket> &items, uint64_t centerUs, uint64_t maxAbsDiffUs, size_t &picked) {
        if(items.empty()) {
            return false;
        }
        auto absDiff = [](uint64_t a, uint64_t b) {
            return a > b ? (a - b) : (b - a);
        };
        auto it = std::lower_bound(items.begin(), items.end(), centerUs, [](const StreamPacket &item, uint64_t ts) {
            return item.tsUs < ts;
        });
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

    static bool offsetPacketIndex(size_t base, int offset, size_t count, size_t &out) {
        const long long idx = static_cast<long long>(base) + static_cast<long long>(offset);
        if(idx < 0 || idx >= static_cast<long long>(count)) {
            return false;
        }
        out = static_cast<size_t>(idx);
        return true;
    }

    bool multiviewCanFinalizeLocked(uint64_t centerUs) const {
        for(const auto &sn: session_.deviceSns) {
            for(const auto t: session_.typesAlign) {
                auto itStreamByType = session_.streams.find(sn);
                if(itStreamByType == session_.streams.end()) {
                    if(!session_.multiviewEos) {
                        return false;
                    }
                    continue;
                }
                auto itStream = itStreamByType->second.find(t);
                if(itStream == itStreamByType->second.end()) {
                    if(!session_.multiviewEos) {
                        return false;
                    }
                    continue;
                }
                if(!itStream->second.eos && itStream->second.maxTsUs < centerUs + sessionCrossTypeMaxAbsDiffUs(session_.stepUs)) {
                    return false;
                }
            }
        }
        if(session_.saveFisheye) {
            if(session_.fisheyeSets.empty()) {
                return session_.fisheyeEos;
            }
            if(!session_.fisheyeEos && session_.fisheyeSets.back().representativeTimestampUs < centerUs) {
                return false;
            }
        }
        return true;
    }

    size_t pickNearestFisheyeIndexLocked(uint64_t centerUs) const {
        if(session_.fisheyeSets.empty()) {
            return 0;
        }
        return findNearestFisheyeFrameSetIndex(std::vector<FisheyeFrameSet>(session_.fisheyeSets.begin(), session_.fisheyeSets.end()), centerUs);
    }

    void pruneCommittedFramesLocked(uint64_t centerUs) {
        for(auto &kv: session_.streams) {
            for(auto &typeKv: kv.second) {
                auto &items = typeKv.second.committed;
                while(items.size() > 1 && items[1].tsUs <= centerUs) {
                    items.pop_front();
                }
            }
        }
        while(session_.fisheyeSets.size() > 1 && session_.fisheyeSets[1].representativeTimestampUs <= centerUs) {
            session_.fisheyeSets.pop_front();
        }
    }

    void appendTimestampRowLocked(const std::vector<std::string> &row) {
        if(!session_.timestampsOpen) {
            return;
        }
        writeCsvRow(session_.timestampsOfs, row);
    }

    void buildCloudInputsLocked(const std::unordered_map<std::string, std::unordered_map<CollectDataType, size_t>> &pickedIndices,
                                std::vector<FusedCloudFrameInput> &outInfos) {
        outInfos.clear();
        outInfos.reserve(session_.deviceSns.size());
        auto itDepthTi = session_.alignTypeIndex.find(CollectDataType::Depth);
        if(itDepthTi == session_.alignTypeIndex.end()) {
            return;
        }
        for(const auto &sn: session_.deviceSns) {
            auto itBuf = session_.buffers.find(sn);
            if(itBuf == session_.buffers.end()) {
                continue;
            }
            auto itStreams = session_.streams.find(sn);
            if(itStreams == session_.streams.end()) {
                continue;
            }
            auto itDepthState = itStreams->second.find(CollectDataType::Depth);
            if(itDepthState == itStreams->second.end()) {
                continue;
            }
            auto itPickedMap = pickedIndices.find(sn);
            if(itPickedMap == pickedIndices.end()) {
                continue;
            }
            auto itPickedDepth = itPickedMap->second.find(CollectDataType::Depth);
            if(itPickedDepth == itPickedMap->second.end()) {
                continue;
            }
            const size_t pickedDepth = itPickedDepth->second;
            if(pickedDepth >= itDepthState->second.committed.size()) {
                continue;
            }
            const auto &depthPacket = itDepthState->second.committed[pickedDepth];
            if(depthPacket.frame.empty() || depthPacket.frame.type() != CV_16UC1 || !(depthPacket.valueScale > 0.0f)) {
                continue;
            }

            FusedCloudFrameInput info;
            info.sn = sn;
            info.camKey = itBuf->second.camKey;
            info.depth = depthPacket.frame;
            info.valueScaleMm = depthPacket.valueScale;
            auto itP = itBuf->second.params.find(CollectDataType::Depth);
            if(itP != itBuf->second.params.end()) {
                info.depthParams = itP->second;
            }
            auto itPose = session_.camToWorld.find(info.camKey);
            if(itPose != session_.camToWorld.end() && itPose->second.valid) {
                info.pose = itPose->second;
            }
            else {
                info.pose.valid = true;
                info.pose.R[0] = 1.0f;
                info.pose.R[4] = 1.0f;
                info.pose.R[8] = 1.0f;
            }

            if(session_.saveColorCloud && itBuf->second.rgbDepthParamValid) {
                auto itRgbState = itStreams->second.find(CollectDataType::RGB);
                if(itRgbState != itStreams->second.end()) {
                    size_t pickedRgb = 0;
                    bool   hasPickedRgb = false;
                    auto itPickedRgb = itPickedMap->second.find(CollectDataType::RGB);
                    if(itPickedRgb != itPickedMap->second.end() && itPickedRgb->second < itRgbState->second.committed.size()) {
                        pickedRgb = itPickedRgb->second;
                        hasPickedRgb = true;
                    }
                    else if(pickNearestPacket(itRgbState->second.committed, depthPacket.tsUs, session_.maxAbsDiffUs, pickedRgb)
                            && pickedRgb < itRgbState->second.committed.size()) {
                        hasPickedRgb = true;
                    }
                    if(hasPickedRgb && cfg_.colorCloudRgbFrameOffset != 0) {
                        size_t offsetRgb = pickedRgb;
                        if(offsetPacketIndex(pickedRgb, cfg_.colorCloudRgbFrameOffset, itRgbState->second.committed.size(), offsetRgb)) {
                            pickedRgb = offsetRgb;
                        }
                        else {
                            hasPickedRgb = false;
                        }
                    }
                    if(hasPickedRgb) {
                        info.rgbFrame = itRgbState->second.committed[pickedRgb].frame;
                        info.rgbDepthParam = itBuf->second.rgbDepthParam;
                        info.hasColor = !info.rgbFrame.empty();
                    }
                }
            }
            outInfos.push_back(std::move(info));
        }
    }

    bool tryFinalizeOneMultiviewSlotLocked() {
        if(session_.refSn.empty()) {
            return false;
        }
        auto itRefStreams = session_.streams.find(session_.refSn);
        if(itRefStreams == session_.streams.end()) {
            return false;
        }
        auto itRefState = itRefStreams->second.find(refType_);
        if(itRefState == itRefStreams->second.end() || itRefState->second.committed.empty()) {
            return false;
        }
        const uint64_t centerUs = itRefState->second.committed.front().tsUs;
        if(!multiviewCanFinalizeLocked(centerUs)) {
            return false;
        }

        std::unordered_map<std::string, std::unordered_map<CollectDataType, size_t>> pickedIndices;
        bool fullThis = true;
        std::unordered_map<CollectDataType, uint64_t> typeCenters;
        typeCenters[refType_] = centerUs;
        for(const auto t: session_.typesAlign) {
            if(t == refType_) {
                continue;
            }
            auto itRefType = itRefStreams->second.find(t);
            if(itRefType == itRefStreams->second.end()) {
                fullThis = false;
                continue;
            }
            size_t pickedRefType = 0;
            if(!pickNearestPacket(itRefType->second.committed, centerUs, sessionCrossTypeMaxAbsDiffUs(session_.stepUs), pickedRefType)) {
                fullThis = false;
                continue;
            }
            typeCenters[t] = itRefType->second.committed[pickedRefType].tsUs;
        }
        uint64_t tsMin = std::numeric_limits<uint64_t>::max();
        uint64_t tsMax = 0;

        for(const auto &sn: session_.deviceSns) {
            for(const auto t: session_.typesAlign) {
                auto itStreamMap = session_.streams.find(sn);
                if(itStreamMap == session_.streams.end()) {
                    fullThis = false;
                    continue;
                }
                auto itStream = itStreamMap->second.find(t);
                if(itStream == itStreamMap->second.end()) {
                    fullThis = false;
                    continue;
                }
                const auto itCenter = typeCenters.find(t);
                if(itCenter == typeCenters.end()) {
                    fullThis = false;
                    continue;
                }
                size_t picked = 0;
                if(!pickNearestPacket(itStream->second.committed, itCenter->second, session_.maxAbsDiffUs, picked)) {
                    fullThis = false;
                    continue;
                }
                pickedIndices[sn][t] = picked;
                const auto &packet = itStream->second.committed[picked];
                if(!streamPacketHasPayload(t, packet)) {
                    fullThis = false;
                    continue;
                }
                if(t == CollectDataType::Depth && (!(packet.valueScale > 0.0f) || packet.frame.type() != CV_16UC1)) {
                    fullThis = false;
                    continue;
                }
                tsMin = std::min(tsMin, packet.tsUs);
                tsMax = std::max(tsMax, packet.tsUs);
            }
            const auto itBuf = session_.buffers.find(sn);
            if(session_.saveColorCloud && (itBuf == session_.buffers.end() || !itBuf->second.rgbDepthParamValid)) {
                fullThis = false;
            }
        }

        const size_t refIndex = session_.alignedRef;
        session_.alignedRef++;
        if(!fullThis) {
            session_.missingAligned++;
            if(session_.prevMissingIndex != std::numeric_limits<size_t>::max()) {
                const size_t d = refIndex - session_.prevMissingIndex;
                if(d > 0 && session_.stepUs > 0) {
                    const double ms = static_cast<double>(d) * (1000.0 / static_cast<double>(1000000.0 / static_cast<double>(session_.stepUs)));
                    if(session_.minMissingMs == 0.0 || ms < session_.minMissingMs) {
                        session_.minMissingMs = ms;
                    }
                }
            }
            session_.prevMissingIndex = refIndex;
            itRefState->second.committed.pop_front();
            pruneCommittedFramesLocked(centerUs);
            return true;
        }

        session_.fullAligned++;
        if(tsMax >= tsMin) {
            const double diffMs = static_cast<double>(tsMax - tsMin) / 1000.0;
            if(diffMs > session_.maxDiffMs) {
                session_.maxDiffMs = diffMs;
            }
        }

        const size_t outIdx = session_.nextFrameIndex++;
        const std::string frameIndex = formatFrameIndex(outIdx);
        session_.alignedCenters.push_back(centerUs);
        hasData_.store(true);

        std::vector<std::string> row;
        row.reserve(2 + session_.deviceSns.size() * 2 + session_.fisheyeCameraCount + 2);
        row.push_back(frameIndex);
        row.push_back(std::to_string(centerUs));
        std::unordered_map<CollectDataType, uint64_t> rgbdTsMinByType;
        std::unordered_map<CollectDataType, uint64_t> rgbdTsMaxByType;
        std::unordered_map<CollectDataType, size_t> rgbdTsCountByType;
        uint64_t allTsMin = std::numeric_limits<uint64_t>::max();
        uint64_t allTsMax = 0;
        size_t   allTsCount = 0;
        auto noteRgbdTimestamp = [&](CollectDataType t, uint64_t ts) {
            auto itMin = rgbdTsMinByType.find(t);
            if(itMin == rgbdTsMinByType.end()) {
                rgbdTsMinByType[t] = ts;
                rgbdTsMaxByType[t] = ts;
                rgbdTsCountByType[t] = 1;
                return;
            }
            itMin->second = std::min(itMin->second, ts);
            rgbdTsMaxByType[t] = std::max(rgbdTsMaxByType[t], ts);
            rgbdTsCountByType[t]++;
        };

        for(const auto &sn: session_.deviceSns) {
            const auto &buf = session_.buffers.at(sn);
            auto writeTsField = [&](CollectDataType t, bool enabled) {
                if(!enabled) {
                    return;
                }
                auto itPickedMap = pickedIndices.find(sn);
                if(itPickedMap == pickedIndices.end()) {
                    row.emplace_back();
                    return;
                }
                auto itPicked = itPickedMap->second.find(t);
                if(itPicked == itPickedMap->second.end()) {
                    row.emplace_back();
                    return;
                }
                const auto &packet = session_.streams.at(sn).at(t).committed[itPicked->second];
                row.push_back(std::to_string(packet.tsUs));
                noteRgbdTimestamp(t, packet.tsUs);
                allTsMin = std::min(allTsMin, packet.tsUs);
                allTsMax = std::max(allTsMax, packet.tsUs);
                allTsCount++;
            };
            writeTsField(CollectDataType::RGB, session_.saveRgbTimesteps);
            writeTsField(CollectDataType::Depth, session_.saveDepthTimesteps);

            for(const auto t: session_.typesPerCamSave) {
                auto itPicked = pickedIndices.at(sn).find(t);
                if(itPicked == pickedIndices.at(sn).end()) {
                    continue;
                }
                const auto &packet = session_.streams.at(sn).at(t).committed[itPicked->second];
                if(!streamPacketHasPayload(t, packet)) {
                    continue;
                }
                if(t == CollectDataType::RGB) {
                    if(cfg_.save.rgbH265) {
                        cv::Mat frame = packet.frame;
                        enqueueH265Frame(sn, frameIndex, packet.tsUs, std::move(frame));
                        continue;
                    }
                    const fs::path outPath = session_.dest / buf.camKey / dataTypeLabel(t) / (frameIndex + colorExtNormalized(cfg_.save.colorExt));
                    if(packet.encodedFormat == OB_FORMAT_MJPG && packet.encodedBytes && !packet.encodedBytes->empty()
                       && isJpegLikePath(outPath)) {
                        auto encodedBytes = packet.encodedBytes;
                        enqueueWriteTask(WriteTask{ [encodedBytes, outPath]() mutable {
                            writeBytesToFile(outPath, encodedBytes->data(), encodedBytes->size());
                        } });
                    }
                    else {
                        cv::Mat frame = packet.frame;
                        const SaveOptions saveOptions = cfg_.save;
                        enqueueWriteTask(WriteTask{ [frame = std::move(frame), outPath, saveOptions]() mutable {
                            saveBgrMatToFile(frame, outPath, saveOptions);
                        } });
                    }
                }
                else if(t == CollectDataType::Depth) {
                    const fs::path outPath = session_.dest / buf.camKey / dataTypeLabel(t) / (frameIndex + ".png");
                    cv::Mat frame = packet.frame;
                    const SaveOptions saveOptions = cfg_.save;
                    enqueueWriteTask(WriteTask{ [frame = std::move(frame), outPath, saveOptions]() mutable {
                        saveRawMatToPng(frame, outPath, saveOptions.pngCompression);
                    } });
                }
                else {
                    const fs::path outPath = session_.dest / buf.camKey / dataTypeLabel(t) / (frameIndex + ".png");
                    cv::Mat frame = packet.frame;
                    const SaveOptions saveOptions = cfg_.save;
                    enqueueWriteTask(WriteTask{ [frame = std::move(frame), outPath, saveOptions]() mutable {
                        saveRawMatToPng(frame, outPath, saveOptions.pngCompression);
                    } });
                }
            }
        }

        if(session_.saveFisheye) {
            if(!session_.fisheyeSets.empty()) {
                const size_t fisheyeIdx = pickNearestFisheyeIndexLocked(centerUs);
                const auto &sample = session_.fisheyeSets[fisheyeIdx];
                for(size_t cameraIdx = 0; cameraIdx < session_.fisheyeCameraCount; ++cameraIdx) {
                    if(cameraIdx < sample.frames.size()) {
                        const auto &frame = sample.frames[cameraIdx];
                        row.push_back(std::to_string(frame.captureTimestampUs));
                        allTsMin = std::min(allTsMin, frame.captureTimestampUs);
                        allTsMax = std::max(allTsMax, frame.captureTimestampUs);
                        allTsCount++;
                        const fs::path outPath = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx) / "RGB"
                                                 / (frameIndex + colorExtNormalized(cfg_.save.colorExt));
                        cv::Mat frameImg = frame.bgr;
                        const SaveOptions saveOptions = cfg_.save;
                        enqueueWriteTask(WriteTask{ [frameImg = std::move(frameImg), outPath, saveOptions]() mutable {
                            saveBgrMatToFile(frameImg, outPath, saveOptions);
                        } });
                        if(!session_.wroteFisheyeCameraParams[cameraIdx] && !frame.bgr.empty()) {
                            const fs::path cameraDir = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx);
                            writeFisheyeCameraParamsJson(cameraDir, cameraIdx, frame.bgr, cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_));
                            session_.wroteFisheyeCameraParams[cameraIdx] = true;
                        }
                    }
                    else {
                        row.emplace_back();
                    }
                }
            }
            else {
                for(size_t cameraIdx = 0; cameraIdx < session_.fisheyeCameraCount; ++cameraIdx) {
                    row.emplace_back();
                }
            }
        }

        {
            double rgbdMaxDiffMs = 0.0;
            for(const auto &kv: rgbdTsCountByType) {
                if(kv.second <= 1) {
                    continue;
                }
                const auto itMin = rgbdTsMinByType.find(kv.first);
                const auto itMax = rgbdTsMaxByType.find(kv.first);
                if(itMin != rgbdTsMinByType.end() && itMax != rgbdTsMaxByType.end() && itMax->second >= itMin->second) {
                    rgbdMaxDiffMs = std::max(rgbdMaxDiffMs, static_cast<double>(itMax->second - itMin->second) / 1000.0);
                }
            }
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(3) << rgbdMaxDiffMs;
            row.push_back(oss.str());
        }
        {
            double allModalMaxDiffMs = 0.0;
            if(allTsCount > 1 && allTsMax >= allTsMin) {
                allModalMaxDiffMs = static_cast<double>(allTsMax - allTsMin) / 1000.0;
            }
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(3) << allModalMaxDiffMs;
            row.push_back(oss.str());
        }
        appendTimestampRowLocked(row);

        if(session_.saveCloud || session_.saveColorCloud) {
            std::vector<FusedCloudFrameInput> infos;
            buildCloudInputsLocked(pickedIndices, infos);
            if(!infos.empty()) {
                const int step = 1;
                if(session_.saveCloud) {
                    const fs::path outPath = session_.dest / dataTypeLabel(CollectDataType::CloudPoints) / (frameIndex + ".ply");
                    enqueueWriteTask(WriteTask{ [infos, outPath, this, step]() mutable {
                        writeFusedPointCloudPly(infos, outPath, cfg_.maxDepth, step);
                    } });
                }
                if(session_.saveColorCloud) {
                    const fs::path outPath = session_.dest / dataTypeLabel(CollectDataType::ColorCloudPoints) / (frameIndex + ".ply");
                    enqueueWriteTask(WriteTask{ [infos, outPath, this, step]() mutable {
                        writeFusedColorPointCloudPly(infos, outPath, cfg_.maxDepth, step);
                    } });
                }
            }
        }

        itRefState->second.committed.pop_front();
        pruneCommittedFramesLocked(centerUs);
        return true;
    }

    bool tryFinalizeFisheyeOnlySlotLocked() {
        if(!session_.saveFisheye || session_.fisheyeSets.empty()) {
            return false;
        }
        if(!session_.fisheyeOnlyTargetInit) {
            session_.fisheyeOnlyNextTargetUs = session_.fisheyeSets.front().representativeTimestampUs;
            session_.fisheyeOnlyTargetInit = true;
        }
        const uint64_t targetUs = session_.fisheyeOnlyNextTargetUs;
        if(!session_.fisheyeEos && session_.fisheyeSets.back().representativeTimestampUs < targetUs) {
            return false;
        }

        const size_t idx = pickNearestFisheyeIndexLocked(targetUs);
        const auto &sample = session_.fisheyeSets[idx];
        if(!session_.hasLastEmittedFisheyeTs || session_.lastEmittedFisheyeTs != sample.representativeTimestampUs) {
            const size_t outIdx = session_.nextFrameIndex++;
            const std::string frameIndex = formatFrameIndex(outIdx);
            std::vector<std::string> row;
            row.push_back(frameIndex);
            row.push_back(std::to_string(sample.representativeTimestampUs));
            uint64_t rowTsMin = std::numeric_limits<uint64_t>::max();
            uint64_t rowTsMax = 0;
            size_t rowTsCount = 0;
            for(size_t cameraIdx = 0; cameraIdx < session_.fisheyeCameraCount; ++cameraIdx) {
                if(cameraIdx < sample.frames.size()) {
                    const auto &frame = sample.frames[cameraIdx];
                    row.push_back(std::to_string(frame.captureTimestampUs));
                    rowTsMin = std::min(rowTsMin, frame.captureTimestampUs);
                    rowTsMax = std::max(rowTsMax, frame.captureTimestampUs);
                    rowTsCount++;
                    const fs::path outPath = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx) / "RGB"
                                             / (frameIndex + colorExtNormalized(cfg_.save.colorExt));
                    cv::Mat frameImg = frame.bgr;
                    const SaveOptions saveOptions = cfg_.save;
                    enqueueWriteTask(WriteTask{ [frameImg = std::move(frameImg), outPath, saveOptions]() mutable {
                        saveBgrMatToFile(frameImg, outPath, saveOptions);
                    } });
                    if(!session_.wroteFisheyeCameraParams[cameraIdx] && !frame.bgr.empty()) {
                        const fs::path cameraDir = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx);
                        writeFisheyeCameraParamsJson(cameraDir, cameraIdx, frame.bgr, cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_));
                        session_.wroteFisheyeCameraParams[cameraIdx] = true;
                    }
                }
                else {
                    row.emplace_back();
                }
            }
            row.push_back("");
            double allModalMaxDiffMs = 0.0;
            if(rowTsCount > 1 && rowTsMax >= rowTsMin) {
                allModalMaxDiffMs = static_cast<double>(rowTsMax - rowTsMin) / 1000.0;
            }
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(3) << allModalMaxDiffMs;
            row.push_back(oss.str());
            appendTimestampRowLocked(row);
            hasData_.store(true);
            session_.hasLastEmittedFisheyeTs = true;
            session_.lastEmittedFisheyeTs = sample.representativeTimestampUs;
        }

        session_.fisheyeOnlyNextTargetUs += session_.stepUs;
        while(session_.fisheyeSets.size() > 1 && session_.fisheyeSets[1].representativeTimestampUs <= targetUs) {
            session_.fisheyeSets.pop_front();
        }
        return true;
    }

    void writeImuCsvForSessionLocked() {
        if(!imuEnabled_ || session_.imuWritten || !multiviewEnabled_) {
            return;
        }
        std::unordered_map<std::string, std::vector<ImuSample>> accelBySn;
        std::unordered_map<std::string, std::vector<ImuSample>> gyroBySn;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for(const auto &sn: session_.deviceSns) {
                auto it = buffers_.find(sn);
                if(it == buffers_.end()) {
                    continue;
                }
                accelBySn[sn] = it->second.accelSamples;
                gyroBySn[sn] = it->second.gyroSamples;
            }
        }
        const uint64_t imuMaxAbsDiffUs = std::max<uint64_t>(10000, session_.maxAbsDiffUs * 2);
        auto pickNearestImu = [&](const std::vector<ImuSample> &samples, size_t &idx, uint64_t centerUs, size_t &outIdx) {
            if(samples.empty()) {
                return false;
            }
            while(idx < samples.size() && samples[idx].tsUs < centerUs) {
                ++idx;
            }
            size_t cand0 = (idx < samples.size()) ? idx : (samples.size() - 1);
            size_t cand1 = (cand0 > 0) ? (cand0 - 1) : cand0;
            auto absDiff = [](uint64_t a, uint64_t b) {
                return a > b ? (a - b) : (b - a);
            };
            const uint64_t d0 = absDiff(samples[cand0].tsUs, centerUs);
            const uint64_t d1 = absDiff(samples[cand1].tsUs, centerUs);
            const size_t chosen = (d1 <= d0) ? cand1 : cand0;
            if(absDiff(samples[chosen].tsUs, centerUs) > imuMaxAbsDiffUs) {
                return false;
            }
            outIdx = chosen;
            if(chosen >= idx) {
                idx = chosen + 1;
            }
            return true;
        };

        for(const auto &sn: session_.deviceSns) {
            auto itBuf = session_.buffers.find(sn);
            if(itBuf == session_.buffers.end()) {
                continue;
            }
            auto accelSamples = accelBySn[sn];
            auto gyroSamples = gyroBySn[sn];
            std::sort(accelSamples.begin(), accelSamples.end(), [](const ImuSample &a, const ImuSample &b) { return a.tsUs < b.tsUs; });
            std::sort(gyroSamples.begin(), gyroSamples.end(), [](const ImuSample &a, const ImuSample &b) { return a.tsUs < b.tsUs; });

            std::ofstream ofs(session_.dest / itBuf->second.camKey / "IMU" / "imu.csv");
            if(!ofs.is_open()) {
                continue;
            }
            ofs.setf(std::ios::fixed);
            ofs << "frame_index,ref_ts_us,accel_ts_us,accel_x,accel_y,accel_z,gyro_ts_us,gyro_x,gyro_y,gyro_z\n";

            size_t accelIdx = 0;
            size_t gyroIdx = 0;
            for(size_t outIdx = 0; outIdx < session_.alignedCenters.size(); ++outIdx) {
                const uint64_t centerUs = session_.alignedCenters[outIdx];
                size_t pickedAccel = 0;
                size_t pickedGyro = 0;
                const bool hasAccel = pickNearestImu(accelSamples, accelIdx, centerUs, pickedAccel);
                const bool hasGyro = pickNearestImu(gyroSamples, gyroIdx, centerUs, pickedGyro);

                ofs << formatFrameIndex(outIdx) << "," << centerUs << ",";
                if(hasAccel) {
                    const auto &a = accelSamples[pickedAccel];
                    ofs << a.tsUs << "," << a.x << "," << a.y << "," << a.z;
                }
                else {
                    ofs << ",,,";
                }
                ofs << ",";
                if(hasGyro) {
                    const auto &g = gyroSamples[pickedGyro];
                    ofs << g.tsUs << "," << g.x << "," << g.y << "," << g.z;
                }
                else {
                    ofs << ",,,";
                }
                ofs << "\n";
            }
        }
        session_.imuWritten = true;
    }

    bool tryCompleteSessionLocked() {
        if(session_.coordinatorDone || !session_.active) {
            return false;
        }
        if(!coordRecordQueue_.empty() || !coordFisheyeQueue_.empty()) {
            return false;
        }
        if(multiviewEnabled_) {
            auto itRefStreams = session_.streams.find(session_.refSn);
            const bool refEmpty = (itRefStreams == session_.streams.end())
                                  || (itRefStreams->second.find(refType_) == itRefStreams->second.end())
                                  || itRefStreams->second.at(refType_).committed.empty();
            if(!session_.multiviewEos || !refEmpty || hasPendingBySeqLocked()) {
                return false;
            }
        }
        else {
            if(!session_.fisheyeEos) {
                return false;
            }
            if(session_.fisheyeOnlyTargetInit && !session_.fisheyeSets.empty()
               && session_.fisheyeOnlyNextTargetUs <= session_.fisheyeSets.back().representativeTimestampUs) {
                return false;
            }
        }

        writeImuCsvForSessionLocked();
        finalizeSessionTimestampsLocked();
        releaseSessionFrameCachesLocked();
        session_.coordinatorDone = true;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = buildCaptureInfoFromSession(session_, lastRecordedSeconds_);
            slotStatsLine_ = buildSlotStatsFromSession(session_);
            savedAlignedMaxDiffMs_ = session_.maxDiffMs;
        }
        return true;
    }

    void commitProcessedRecordLocked(ProcessedRecord &&item) {
        auto &state = session_.streams[item.sn][item.type];
        state.bySeq[item.seq] = StreamPacket{ item.tsUs, std::move(item.frame), std::move(item.encodedBytes), item.encodedFormat, item.valueScale };
        drainReadyPacketsLocked(item.sn, item.type, state);
        if(state.bySeq.find(state.nextSeq) == state.bySeq.end()
           && state.bySeq.size() > coordQueueMax_) {
            const uint64_t skippedTo = state.bySeq.begin()->first;
            std::cerr << "[collection] warning: stream gap detected sn=" << item.sn
                      << " type=" << dataTypeLabel(item.type)
                      << " nextSeq=" << state.nextSeq
                      << " recoverTo=" << skippedTo
                      << " buffered=" << state.bySeq.size() << std::endl;
            state.nextSeq = skippedTo;
            drainReadyPacketsLocked(item.sn, item.type, state);
        }
    }

    void coordinatorLoop() {
        for(;;) {
            std::unique_lock<std::mutex> lock(coordMtx_);
            coordCv_.wait(lock, [&]() {
                return !coordRecordQueue_.empty() || !coordFisheyeQueue_.empty()
                       || (session_.active && (session_.multiviewEos || session_.fisheyeEos))
                       || stopping_.load();
            });
            if(!session_.active && coordRecordQueue_.empty() && coordFisheyeQueue_.empty()) {
                return;
            }

            while(!coordRecordQueue_.empty()) {
                commitProcessedRecordLocked(std::move(coordRecordQueue_.front()));
                coordRecordQueue_.pop_front();
                coordCv_.notify_all();
            }
            while(!coordFisheyeQueue_.empty()) {
                session_.fisheyeSets.push_back(std::move(coordFisheyeQueue_.front()));
                coordFisheyeQueue_.pop_front();
                session_.fisheyeCapturedSets++;
                coordCv_.notify_all();
            }

            bool progress = true;
            while(progress) {
                progress = false;
                progress = flushEosSequenceGapsLocked() || progress;
                if(multiviewEnabled_) {
                    progress = tryFinalizeOneMultiviewSlotLocked() || progress;
                }
                else if(session_.saveFisheye) {
                    progress = tryFinalizeFisheyeOnlySlotLocked() || progress;
                }
                progress = tryCompleteSessionLocked() || progress;
                if(session_.coordinatorDone) {
                    return;
                }
            }
        }
    }

    std::unordered_map<std::string, cv::Mat> latestRgbFramesImpl() {
        std::unordered_map<std::string, cv::Mat> out;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for(auto &kv: buffers_) {
                if(!kv.second.latestRgb.empty()) {
                    out.emplace(kv.first, kv.second.latestRgb);
                }
            }
        }
        return out;
    }

    void onFrameSet(const std::string &deviceSn, int deviceIndex, const std::shared_ptr<ob::FrameSet> &frameSet) {
        (void)deviceIndex;
        if(stopping_.load() || !frameSet) {
            return;
        }
        collectionSetStage("cb_enter");

        uint64_t cbCount = 0;
        {
            std::lock_guard<std::mutex> lock(callbackCountsMtx_);
            cbCount = ++callbackCounts_[deviceSn];
        }
        if((cbCount % 300) == 0) {
            std::cerr << "[collection] callback framesets=" << cbCount << " sn=" << deviceSn << " capturing=" << capturing_.load() << " recording=" << recording_.load() << std::endl;
        }

        const bool requireGlobalTs = cfg_.enableSync && all_.size() > 1;
        std::shared_ptr<ob::Frame> cachedColorFrame;
        std::shared_ptr<ob::Frame> cachedDepthFrame;
        std::unordered_map<CollectDataType, std::shared_ptr<ob::Frame>> cachedOtherFrames;

        auto getFrameTimestampUs = [&](const std::shared_ptr<ob::Frame> &frame, bool allowLocalTimestampFallback) -> uint64_t {
            if(!frame) {
                return 0;
            }
            uint64_t ts = 0;
            try {
                ts = frame->globalTimeStampUs();
            }
            catch(...) {
                ts = 0;
            }
            if(ts == 0 && (!requireGlobalTs || allowLocalTimestampFallback)) {
                try {
                    ts = frame->timeStampUs();
                }
                catch(...) {
                    ts = 0;
                }
            }
            return ts;
        };

        auto getFrameForType = [&](CollectDataType t) -> std::shared_ptr<ob::Frame> {
            if(t == CollectDataType::RGB) {
                if(!cachedColorFrame) {
                    cachedColorFrame = frameSet->colorFrame();
                }
                return cachedColorFrame;
            }
            if(t == CollectDataType::Depth) {
                if(!cachedDepthFrame) {
                    cachedDepthFrame = frameSet->depthFrame();
                }
                return cachedDepthFrame;
            }
            auto it = cachedOtherFrames.find(t);
            if(it != cachedOtherFrames.end()) {
                return it->second;
            }
            auto frame = frameSet->getFrame(dataTypeFrameType(t));
            cachedOtherFrames.emplace(t, frame);
            return frame;
        };

        {
            collectionSetStage("cb_preview_rgb");
            auto frame = getFrameForType(CollectDataType::RGB);
            if(frame) {
                const uint64_t ts = getFrameTimestampUs(frame, true);
                bool shouldRefreshPreview = (ts != 0);
                if(shouldRefreshPreview) {
                    std::lock_guard<std::mutex> lock(previewTsMtx_);
                    auto &lastTs = lastPreviewTs_[deviceSn];
                    if(lastTs != 0 && ts <= lastTs) {
                        shouldRefreshPreview = false;
                    }
                    else if(lastTs != 0 && (ts - lastTs) < 200000) {
                        shouldRefreshPreview = false;
                    }
                    else {
                        lastTs = ts;
                    }
                }
                if(shouldRefreshPreview) {
                    cv::Mat previewBgr;
                    if(copyColorFrameToBgr(frame, previewBgr) && !previewBgr.empty()) {
                        std::lock_guard<std::mutex> lock(mtx_);
                        auto it = buffers_.find(deviceSn);
                        if(it != buffers_.end()) {
                            it->second.latestRgb = std::move(previewBgr);
                            it->second.latestRgbTsUs = ts;
                        }
                    }
                }
            }
        }

        for(const auto t: typesSaving_) {
            if(t == CollectDataType::CloudPoints || t == CollectDataType::ColorCloudPoints) {
                continue;
            }
            if(!recording_.load()) {
                continue;
            }
            collectionSetStage("cb_record_stream");

            auto frame = getFrameForType(t);
            if(!frame) {
                continue;
            }

            const uint64_t ts = getFrameTimestampUs(frame, false);
            if(ts == 0) {
                continue;
            }

            int actualFps = 0;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                auto itB = buffers_.find(deviceSn);
                if(itB != buffers_.end()) {
                    auto itP = itB->second.params.find(t);
                    if(itP != itB->second.params.end()) {
                        actualFps = itP->second.fps;
                    }
                }
            }

            if(cfg_.collectFps > 0 && !useSoftwareTrigger_ && (actualFps <= 0 || actualFps > cfg_.collectFps)) {
                const uint64_t interval = static_cast<uint64_t>(1000000.0 / static_cast<double>(cfg_.collectFps));
                const std::string key   = deviceSn + ":" + std::string(dataTypeLabel(t));
                {
                    std::lock_guard<std::mutex> lock(lastSavedTsMtx_);
                    auto it = lastSavedTs_.find(key);
                    if(it != lastSavedTs_.end()) {
                        const uint64_t last = it->second;
                        if(ts > last && (ts - last) < interval) {
                            continue;
                        }
                        if(ts > last) {
                            it->second = ts;
                        }
                    }
                    else {
                        lastSavedTs_[key] = ts;
                    }
                }
            }

            DetachedVideoFrame detached;
            if(!detachVideoFrame(frame, detached, t == CollectDataType::Depth)) {
                continue;
            }

            collectionSetStage("cb_enqueue_record");
            enqueueRecordTask(RecordTask{ deviceSn, t, ts, nextRecordSeq(deviceSn, t), std::move(detached) });
        }
        collectionSetStage("cb_exit");
    }

    void softwareTriggerLoop() {
        int fps = cfg_.collectFps > 0 ? cfg_.collectFps : uiFpsFallback_;
        if(fps <= 0) {
            fps = 30;
        }
        const auto interval = std::chrono::microseconds(static_cast<int64_t>(1000000.0 / static_cast<double>(fps)));
        while(!stopping_.load()) {
            for(const auto &dev: softwareTriggerDevices_) {
                try {
                    dev->triggerCapture();
                }
                catch(...) {
                }
            }
            std::this_thread::sleep_for(interval);
        }
    }

    void fisheyeRecordLoop() {
        while(recording_.load() && !stopping_.load() && fisheyeEnabled_) {
            std::string err;
            auto sample = fisheyeRecorder_.captureNext(&err);
            if(!sample) {
                if(!recording_.load() || stopping_.load()) {
                    break;
                }
                if(!err.empty()) {
                    std::cerr << "[collection] fisheye capture error: " << err << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            enqueueFisheyeFrameSet(std::move(*sample));
        }
    }

    static void writeParamsJson(const fs::path &dest,
                                const std::unordered_map<std::string, DeviceBuffer> &buffers,
                                const std::vector<CollectDataType> &typesSaving) {
        cJSON *root = cJSON_CreateObject();
        for(const auto &kv: buffers) {
            const auto &sn  = kv.first;
            const auto &buf = kv.second;
            cJSON      *camObj = cJSON_CreateObject();
            jsonAddString(camObj, "sn", sn);

            for(const auto t: typesSaving) {
                if(t == CollectDataType::CloudPoints || t == CollectDataType::ColorCloudPoints) {
                    continue;
                }
                cJSON *stObj = cJSON_CreateObject();
                const auto itP = buf.params.find(t);
                if(itP != buf.params.end() && itP->second.valid) {
                    jsonAddNumber(stObj, "width",  itP->second.width);
                    jsonAddNumber(stObj, "height", itP->second.height);
                    jsonAddNumber(stObj, "fps",    itP->second.fps);
                    jsonAddNumber(stObj, "format", static_cast<int>(itP->second.format));
                    cJSON *intr = cJSON_CreateObject();
                    jsonAddIntrinsic(intr, itP->second.intrinsic);
                    cJSON_AddItemToObject(stObj, "intrinsic", intr);
                    cJSON *dist = cJSON_CreateObject();
                    jsonAddDistortion(dist, itP->second.distortion);
                    cJSON_AddItemToObject(stObj, "distortion", dist);
                }
                cJSON_AddItemToObject(camObj, dataTypeLabel(t), stObj);
            }

            if(buf.rgbDepthParamValid) {
                cJSON *rgbDepthObj = cJSON_CreateObject();
                cJSON *depthIntr = cJSON_CreateObject();
                jsonAddIntrinsic(depthIntr, buf.rgbDepthParam.depthIntrinsic);
                cJSON_AddItemToObject(rgbDepthObj, "depth_intrinsic", depthIntr);
                cJSON *depthDist = cJSON_CreateObject();
                jsonAddDistortion(depthDist, buf.rgbDepthParam.depthDistortion);
                cJSON_AddItemToObject(rgbDepthObj, "depth_distortion", depthDist);
                cJSON *rgbIntr = cJSON_CreateObject();
                jsonAddIntrinsic(rgbIntr, buf.rgbDepthParam.rgbIntrinsic);
                cJSON_AddItemToObject(rgbDepthObj, "rgb_intrinsic", rgbIntr);
                cJSON *rgbDist = cJSON_CreateObject();
                jsonAddDistortion(rgbDist, buf.rgbDepthParam.rgbDistortion);
                cJSON_AddItemToObject(rgbDepthObj, "rgb_distortion", rgbDist);
                cJSON *d2cObj = cJSON_CreateObject();
                jsonAddExtrinsic(d2cObj, buf.rgbDepthParam.transform.rot, buf.rgbDepthParam.transform.trans);
                cJSON_AddItemToObject(rgbDepthObj, "d2c_extrinsic", d2cObj);

                float rct[9];
                float tct[3];
                for(int r = 0; r < 3; r++) {
                    for(int c = 0; c < 3; c++) {
                        const int idxSrc = r * 3 + c;
                        const int idxDst = c * 3 + r;
                        rct[idxDst]      = buf.rgbDepthParam.transform.rot[idxSrc];
                    }
                }
                for(int i = 0; i < 3; i++) {
                    tct[i] = 0.0f;
                }
                for(int r = 0; r < 3; r++) {
                    const int rowIdx = r * 3;
                    float     v      = 0.0f;
                    for(int c = 0; c < 3; c++) {
                        v += rct[rowIdx + c] * buf.rgbDepthParam.transform.trans[c];
                    }
                    tct[r] = -v;
                }
                cJSON *c2dObj = cJSON_CreateObject();
                jsonAddExtrinsic(c2dObj, rct, tct);
                cJSON_AddItemToObject(rgbDepthObj, "c2d_extrinsic", c2dObj);

                cJSON_AddItemToObject(camObj, "rgb_to_depth", rgbDepthObj);
            }

            cJSON_AddItemToObject(root, buf.camKey.c_str(), camObj);
        }
        char *printed = cJSON_Print(root);
        if(printed) {
            writeTextFile(dest / "camera_params.json", printed);
            cJSON_free(printed);
        }
        cJSON_Delete(root);
    }

    void writeExtrinsicsJson(const fs::path &dest) const {
        if(cfg_.initExtrinsicPath.empty()) {
            writeTextFile(dest / "extrinsics.json", "{}");
            return;
        }
        fs::path p(cfg_.initExtrinsicPath);
        std::string content;
        if(!readTextFile(p, content)) {
            writeTextFile(dest / "extrinsics.json", "{}");
            return;
        }
        auto *root = cJSON_Parse(content.c_str());
        if(!root) {
            writeTextFile(dest / "extrinsics.json", "{}");
            return;
        }
        char *printed = cJSON_Print(root);
        if(printed) {
            writeTextFile(dest / "extrinsics.json", printed);
            cJSON_free(printed);
        }
        cJSON_Delete(root);
    }

    AppConfig cfg_;
    ob::Context ctx_;

    std::vector<DeviceRuntime> all_;
    std::vector<DeviceRuntime> primary_;
    std::vector<DeviceRuntime> secondary_;
    std::vector<std::shared_ptr<ob::Device>> softwareTriggerDevices_;
    std::thread triggerThread_;

    std::vector<std::thread> recordWorkers_;
    std::atomic_bool recordStop_{ false };
    std::atomic_bool recordInputClosing_{ false };
    std::atomic_bool multiviewEosNotified_{ false };
    mutable std::mutex recordMtx_;
    std::condition_variable recordCv_;
    std::condition_variable recordDrainCv_;
    std::deque<RecordTask>  recordQueue_;
    size_t                  recordQueueMax_ = 2048;
    int                     recordInFlight_ = 0;

    mutable std::mutex coordMtx_;
    std::condition_variable coordCv_;
    std::deque<ProcessedRecord> coordRecordQueue_;
    std::deque<FisheyeFrameSet> coordFisheyeQueue_;
    size_t coordQueueMax_ = 512;
    SessionState session_{};
    std::thread coordinatorThread_;

    std::mutex writeMtx_;
    std::condition_variable writeCv_;
    std::deque<WriteTask> writeQueue_;
    size_t writeQueueMax_ = 512;
    std::vector<std::thread> writeWorkers_;
    std::atomic_bool writeStop_{ false };
    std::atomic<size_t> queuedWriteCount_{ 0 };
    std::atomic<int>    writeInFlight_{ 0 };

    std::mutex h265Mtx_;
    std::unordered_map<std::string, std::unique_ptr<H265Encoder>> h265Encoders_;
    std::atomic_bool h265EncodingActive_{ false };

    std::atomic_bool capturing_{ false };
    std::atomic_bool recording_{ false };
    std::atomic_bool hasData_{ false };
    std::atomic_bool stopping_{ false };
    std::thread      fisheyeRecordThread_;

    std::chrono::steady_clock::time_point captureStartSteady_{};

    std::vector<CollectDataType> typesStreaming_;
    std::vector<CollectDataType> typesSaving_;
    CollectDataType refType_ = CollectDataType::RGB;
    int             uiFpsFallback_ = 30;
    float           colorExposureMs_ = 0.8f;
    int             colorBrightness_ = -1;
    bool            useSoftwareTrigger_ = false;
    bool            imuEnabled_ = false;
    bool            multiviewEnabled_ = true;
    bool            fisheyeEnabled_ = false;
    size_t          activeFisheyeCameraCount_ = 0;
    FisheyeRecorder fisheyeRecorder_;

    mutable std::mutex mtx_;
    std::unordered_map<std::string, DeviceBuffer> buffers_;
    std::vector<ImuSensorHandle> imuSensors_;
    std::unordered_map<std::string, uint64_t> callbackCounts_;
    std::mutex callbackCountsMtx_;
    std::unordered_map<std::string, uint64_t> lastSavedTs_;
    std::mutex lastSavedTsMtx_;
    std::unordered_map<std::string, uint64_t> lastPreviewTs_;
    std::mutex previewTsMtx_;
    std::unordered_map<std::string, uint64_t> streamNextSeq_;
    std::mutex streamSeqMtx_;
    std::atomic_bool passthroughRgbMjpg_{ false };

    std::string captureInfoLine_;
    std::string slotStatsLine_;
    double      savedAlignedMaxDiffMs_ = 0.0;
    double      lastRecordedSeconds_   = 0.0;
};

struct TaskInfo {
    std::string name;
    std::string description_cn;
    std::string description_en;
    int         repeat_times = 1;
};

struct TaskProgress {
    std::string task_name;
    int         completed = 0;
    int         total     = 1;
};

static std::vector<TaskInfo> loadTaskJson(const fs::path &path) {
    std::vector<TaskInfo> result;
    std::string raw;
    if(!readTextFile(path, raw)) {
        std::cerr << "[collection] loadTaskJson: cannot read " << path << std::endl;
        return result;
    }

    std::string cleaned;
    cleaned.reserve(raw.size());
    {
        std::istringstream iss(raw);
        std::string line;
        while(std::getline(iss, line)) {
            const auto pos = line.find("//");
            if(pos != std::string::npos) {
                line = line.substr(0, pos);
            }
            cleaned += line + "\n";
        }
    }

    auto *root = cJSON_Parse(cleaned.c_str());
    if(!root) {
        std::cerr << "[collection] loadTaskJson: parse error in " << path << std::endl;
        return result;
    }

    auto *child = root->child;
    while(child) {
        TaskInfo info;
        info.name = child->string ? child->string : "";
        if(!info.name.empty()) {
            auto *rt = cJSON_GetObjectItem(child, "repeat_times");
            if(rt && cJSON_IsNumber(rt)) {
                info.repeat_times = std::max(1, static_cast<int>(rt->valuedouble));
            }
            auto *dcn = cJSON_GetObjectItem(child, "task_description_cn");
            if(dcn && cJSON_IsString(dcn) && dcn->valuestring) {
                info.description_cn = dcn->valuestring;
            }
            auto *den = cJSON_GetObjectItem(child, "task_description_en");
            if(den && cJSON_IsString(den) && den->valuestring) {
                info.description_en = den->valuestring;
            }
            result.push_back(std::move(info));
        }
        child = child->next;
    }
    cJSON_Delete(root);
    return result;
}

static std::vector<TaskProgress> loadProgressCsv(const fs::path &path) {
    std::vector<TaskProgress> result;
    std::ifstream ifs(path);
    if(!ifs.is_open()) {
        return result;
    }
    std::string line;
    bool firstLine = true;
    while(std::getline(ifs, line)) {
        if(firstLine) {
            firstLine = false;
            if(line.find("task_name") != std::string::npos) {
                continue;
            }
        }
        if(line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string name, completedStr, totalStr;
        if(std::getline(ss, name, ',') && std::getline(ss, completedStr, ',') && std::getline(ss, totalStr)) {
            try {
                TaskProgress p;
                p.task_name = name;
                p.completed = std::stoi(completedStr);
                p.total     = std::stoi(totalStr);
                result.push_back(std::move(p));
            }
            catch(...) {
            }
        }
    }
    return result;
}

static void saveProgressCsv(const fs::path &path, const std::vector<TaskProgress> &progress) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream ofs(tmp);
        if(!ofs.is_open()) {
            std::cerr << "[collection] saveProgressCsv: cannot write " << tmp << std::endl;
            return;
        }
        ofs << "task_name,completed,total\n";
        for(const auto &p: progress) {
            ofs << p.task_name << "," << p.completed << "," << p.total << "\n";
        }
    }
    try {
        fs::rename(tmp, path);
    }
    catch(...) {
        std::cerr << "[collection] saveProgressCsv: rename failed" << std::endl;
    }
}

static std::vector<TaskProgress> buildProgressFromTasks(const std::vector<TaskInfo> &tasks,
                                                         const std::vector<TaskProgress> &existing) {
    std::vector<TaskProgress> result;
    result.reserve(tasks.size());
    for(const auto &t: tasks) {
        TaskProgress p;
        p.task_name = t.name;
        p.total     = t.repeat_times;
        p.completed = 0;
        for(const auto &e: existing) {
            if(e.task_name == t.name) {
                p.completed = std::min(e.completed, p.total);
                break;
            }
        }
        result.push_back(std::move(p));
    }
    return result;
}

static int getCurrentTaskIndex(const std::vector<TaskProgress> &progress) {
    for(int i = 0; i < static_cast<int>(progress.size()); ++i) {
        if(progress[static_cast<size_t>(i)].completed < progress[static_cast<size_t>(i)].total) {
            return i;
        }
    }
    return -1;
}

static bool uiButtonEx(cv::Mat &img, const cv::Rect &r, const std::string &label,
                        FrameMouse &ms, bool enabled) {
    const bool hover   = enabled && r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar bg      = !enabled ? cv::Scalar(25, 25, 25) : (hover ? cv::Scalar(60, 60, 60) : cv::Scalar(40, 40, 40));
    cv::Scalar fg      = enabled ? cv::Scalar(255, 255, 255) : cv::Scalar(80, 80, 80);
    cv::Scalar border  = enabled ? cv::Scalar(120, 120, 120) : cv::Scalar(50, 50, 50);
    cv::rectangle(img, r, bg, cv::FILLED);
    cv::rectangle(img, r, border, 1);
    cv::putText(img, label, cv::Point(r.x + 14, r.y + r.height / 2 + 7),
                cv::FONT_HERSHEY_DUPLEX, 0.7, fg, 1, cv::LINE_AA);
    if(enabled && ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

enum class CollectionPage { Config, Capture };

enum class CaptureState { IDLE, RECORDING, DRAINING, STOPPED_READY, DELETE_CONFIRM };

struct CollectionCaptureUi {
    std::string activeField;
    std::string msg;

    std::vector<TaskInfo>     tasks;
    std::vector<TaskProgress> progress;
    int                       currentTaskIdx = -1;
    int                       currentEpisode = 0;
    bool                      taskLoadError  = false;
    std::string               taskErrorMsg;
};

static void drawRgbGrid(cv::Mat &ui, const cv::Rect &r, const std::vector<std::pair<std::string, cv::Mat>> &frames, int w, int h, int fps) {
    cv::rectangle(ui, r, cv::Scalar(30, 30, 30), cv::FILLED);
    cv::rectangle(ui, r, cv::Scalar(120, 120, 120), 1);
    if(frames.empty()) {
        cv::putText(ui, "No RGB frames", cv::Point(r.x + 12, r.y + 30), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        return;
    }

    const int n = static_cast<int>(frames.size());
    int cols = 1;
    while(cols * cols < n) {
        cols++;
    }
    const int rows = static_cast<int>((n + cols - 1) / cols);
    const int cellW = std::max(1, r.width / cols);
    const int cellH = std::max(1, r.height / rows);

    for(int i = 0; i < n; i++) {
        const int cx = i % cols;
        const int cy = i / cols;
        cv::Rect cell(r.x + cx * cellW, r.y + cy * cellH, cellW, cellH);
        cv::Rect inner(cell.x + 4, cell.y + 28, cell.width - 8, cell.height - 32);
        cv::rectangle(ui, cell, cv::Scalar(80, 80, 80), 1);
        const auto &label = frames[i].first;
        cv::putText(ui, label, cv::Point(cell.x + 6, cell.y + 20), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(ui, presetLabel(w, h, fps), cv::Point(cell.x + 6, cell.y + cell.height - 8), cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        if(!frames[i].second.empty() && inner.width > 0 && inner.height > 0) {
            cv::Mat resized;
            cv::resize(frames[i].second, resized, inner.size(), 0, 0, cv::INTER_AREA);
            resized.copyTo(ui(inner));
        }
    }
}

}  // namespace

int run_collection(const AppConfig &cfg, const std::atomic_bool *cancel) {
    (void)cancel;

    collectionInstallCrashHandlerOnce();
    collectionSetStage("ui_enter");
    std::cerr << "[collection] run enter" << std::endl;

    const std::string winName = "Collection";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 1800, 1000);

    CvMouseState ms;
    cv::setMouseCallback(winName, mouseThunk, &ms);

    CollectionPage page = CollectionPage::Config;
    CollectionConfigUi cfgUi;
    CollectionCaptureUi capUi;
    MultiDeviceStreamingRecorder recorder(cfg);
    std::deque<std::string> uiLogs;
    int logScroll = 0;
    CaptureState captureState = CaptureState::IDLE;
    bool pendingResetAfterDrain = false;
    std::unordered_map<std::string, cv::Mat> latestFrameCache;
    if(cfg.colorExposureMs > 0.0f) {
        std::ostringstream oss;
        oss << std::setprecision(4) << cfg.colorExposureMs;
        cfgUi.exposureMs = oss.str();
    }
    if(cfg.colorBrightness >= 0) {
        cfgUi.brightness = std::to_string(cfg.colorBrightness);
    }

    auto pushUiLog = [&](std::string s) {
        s = trimString(std::move(s));
        if(s.empty()) {
            return;
        }
        uiLogs.push_back(std::move(s));
        const size_t kMax = 500;
        while(uiLogs.size() > kMax) {
            uiLogs.pop_front();
        }
    };

    auto updateReadyState = [&]() {
        captureState = CaptureState::STOPPED_READY;
        if(recorder.hasData()) {
            capUi.msg = "Data saved to disk. Confirm or reset this episode.";
            pushUiLog("Background save finished. Ready to confirm.");
        }
        else {
            capUi.msg = "No valid frames captured. Reset to discard this episode.";
            pushUiLog("Background save finished. No valid data captured.");
        }
        {
            const std::string line = recorder.lastInfoLine();
            if(!line.empty()) {
                pushUiLog(line);
            }
        }
        {
            const std::string line = recorder.lastSlotStatsLine();
            if(!line.empty()) {
                pushUiLog(line);
            }
        }
    };

    auto enterDeleteConfirm = [&]() {
        captureState = CaptureState::DELETE_CONFIRM;
        capUi.msg.clear();
    };

    bool running = true;
    while(running) {
        collectionSetStage("ui_waitKey");
        const int key = cv::waitKey(1);
        if(key > 0) {
            if(isCtrlModifierKeyEvent(key)) {
                g_ctrlShortcutListening = true;
            }
            else if(g_ctrlShortcutListening && isCtrlReleaseKeyEvent(key)) {
                g_ctrlShortcutListening = false;
            }
        }
        auto fm = beginFrame(ms);
        if(key == 27) {
            collectionSetStage("ui_exit_esc");
            recorder.stopIfRunning();
            running = false;
            break;
        }

        collectionSetStage("ui_draw");
        int winW = 1600;
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
        cv::Mat ui(winH, winW, CV_8UC3, cv::Scalar(20, 20, 20));

        if(page == CollectionPage::Config) {
            collectionSetStage("ui_page_config");
            cfgUi.enforceRules();
            cv::putText(ui, "Collection - Config", cv::Point(24, 48), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

            const int left  = 60;
            const int top   = 110;
            const int rowH  = 64;
            const int fieldW = 260;

            cv::putText(ui, "Capture Types", cv::Point(left, top - 22), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            cv::Rect type1(left, top, 220, 36);
            cv::Rect type2(left + 240, top, 220, 36);
            if(uiCheckbox(ui, type1, cfgUi.enableMultiview, "multiview", fm)) {
                cfgUi.enableMultiview = !cfgUi.enableMultiview;
            }
            if(uiCheckbox(ui, type2, cfgUi.enableFisheyes, "fisheyes", fm)) {
                cfgUi.enableFisheyes = !cfgUi.enableFisheyes;
            }
            if(!cfgUi.hasSelectedCaptureType()) {
                cv::putText(ui, "Select at least one capture type", cv::Point(left, top + rowH - 8), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(60, 60, 255), 1, cv::LINE_AA);
            }

            const int dataTop = top + 2 * rowH;
            cv::putText(ui, "Data Types", cv::Point(left, dataTop - 22), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            cv::Rect c1(left, dataTop, 220, 36);
            cv::Rect c2(left + 240, dataTop, 220, 36);
            cv::Rect c3(left + 480, dataTop, 220, 36);
            cv::Rect c4(left + 720, dataTop, 220, 36);
            cv::Rect c5(left, dataTop + rowH, 220, 36);
            cv::Rect c6(left + 240, dataTop + rowH, 220, 36);
            cv::Rect c7(left + 480, dataTop + rowH, 220, 36);

            if(uiCheckbox(ui, c1, cfgUi.enableRgb, "RGB", fm)) {
                cfgUi.enableRgb = !cfgUi.enableRgb;
            }
            if(uiCheckbox(ui, c2, cfgUi.enableDepth, "Depth", fm)) {
                cfgUi.enableDepth = !cfgUi.enableDepth;
            }
            if(uiCheckbox(ui, c3, cfgUi.enableIrRight, "IR_right", fm)) {
                cfgUi.enableIrRight = !cfgUi.enableIrRight;
            }
            if(uiCheckbox(ui, c4, cfgUi.enableIrLeft, "IR_left", fm)) {
                cfgUi.enableIrLeft = !cfgUi.enableIrLeft;
            }
            if(uiCheckbox(ui, c5, cfgUi.enableCloud, "CloudPoints", fm)) {
                cfgUi.enableCloud = !cfgUi.enableCloud;
            }
            if(uiCheckbox(ui, c6, cfgUi.enableColorCloud, "ColorCloud", fm)) {
                cfgUi.enableColorCloud = !cfgUi.enableColorCloud;
            }
            if(uiCheckbox(ui, c7, cfgUi.enableImu, "IMU", fm)) {
                cfgUi.enableImu = !cfgUi.enableImu;
            }
            if(!cfgUi.enableMultiview) {
                cv::putText(ui, "multiview disabled: Data Types settings are ignored", cv::Point(left, dataTop + 1 * rowH - 6), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
            }
            if((cfgUi.enableCloud || cfgUi.enableColorCloud) && !cfgUi.enableDepth) {
                cv::putText(ui, "CloudPoints selected: Depth auto-enabled", cv::Point(left, dataTop + 1 * rowH + 16), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(80, 200, 80), 1, cv::LINE_AA);
                cfgUi.enableDepth = true;
            }
            if(cfgUi.enableColorCloud && !cfgUi.enableRgb) {
                cv::putText(ui, "ColorCloud selected: RGB auto-enabled", cv::Point(left, dataTop + 1 * rowH + 40), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(80, 200, 80), 1, cv::LINE_AA);
                cfgUi.enableRgb = true;
            }

            const int y2 = top + 4 * rowH;
            cv::putText(ui, "Resolution/FPS", cv::Point(left, y2 - 22), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            if(uiTextField(ui, cv::Rect(left, y2, fieldW, 36), "width", cfgUi.width, cfgUi.activeField == "w", fm)) {
                cfgUi.activeField = "w";
            }
            if(uiTextField(ui, cv::Rect(left + 300, y2, fieldW, 36), "height", cfgUi.height, cfgUi.activeField == "h", fm)) {
                cfgUi.activeField = "h";
            }
            if(uiTextField(ui, cv::Rect(left + 600, y2, fieldW, 36), "fps", cfgUi.fps, cfgUi.activeField == "fps", fm)) {
                cfgUi.activeField = "fps";
            }
            cv::Rect p1(left, y2 + 44, 260, 44);
            cv::Rect p2(left + 280, y2 + 44, 260, 44);
            cv::Rect p3(left + 560, y2 + 44, 260, 44);
            if(uiButton(ui, p1, presetLabel(1280, 720, 30), fm)) {
                cfgUi.width  = "1280";
                cfgUi.height = "720";
                cfgUi.fps    = "30";
            }
            if(uiButton(ui, p2, presetLabel(1280, 800, 30), fm)) {
                cfgUi.width  = "1280";
                cfgUi.height = "800";
                cfgUi.fps    = "30";
            }
            if(uiButton(ui, p3, presetLabel(1920, 1080, 30), fm)) {
                cfgUi.width  = "1920";
                cfgUi.height = "1080";
                cfgUi.fps    = "30";
            }

            const int y3 = top + 6 * rowH;
            if(uiTextField(ui, cv::Rect(left, y3, 520, 36), "save_path (required)", cfgUi.saveRoot, cfgUi.activeField == "save", fm)) {
                cfgUi.activeField = "save";
            }
            if(uiTextField(ui, cv::Rect(left + 560, y3, 260, 36), "subject_id (required)", cfgUi.subjectId, cfgUi.activeField == "sub", fm)) {
                cfgUi.activeField = "sub";
            }

            const int y4 = top + 7 * rowH;
            if(uiTextField(ui, cv::Rect(left, y4, 260, 36), "max_duration_sec", cfgUi.maxDurationSec, cfgUi.activeField == "dur", fm)) {
                cfgUi.activeField = "dur";
            }
            if(uiTextField(ui, cv::Rect(left + 300, y4, 260, 36), "exposure_ms", cfgUi.exposureMs, cfgUi.activeField == "exp", fm)) {
                cfgUi.activeField = "exp";
            }
            if(uiTextField(ui, cv::Rect(left + 600, y4, 260, 36), "brightness", cfgUi.brightness, cfgUi.activeField == "bri", fm)) {
                cfgUi.activeField = "bri";
            }

            if(!cfgUi.error.empty()) {
                cv::putText(ui, cfgUi.error, cv::Point(left, 640), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
            }

            cv::Rect bBack(60, 680, 220, 60);
            cv::Rect bEnter(340, 680, 260, 60);
            if(uiButton(ui, bBack, "Back to Menu", fm)) {
                collectionSetStage("ui_back_menu");
                recorder.stopIfRunning();
                running = false;
            }
            if(uiButton(ui, bEnter, "Enter Capture", fm)) {
                collectionSetStage("ui_enter_capture");
                cfgUi.enforceRules();
                if(!cfgUi.hasSelectedCaptureType()) {
                    cfgUi.error = "Select at least one capture type";
                }
                else if(!cfgUi.hasRequiredFields()) {
                    cfgUi.error = "save_path and subject_id are required";
                }
                else {
                    const fs::path saveRoot = fs::path(trimString(cfgUi.saveRoot));
                    const std::string subject = trimString(cfgUi.subjectId);
                    auto tasks = loadTaskJson(saveRoot / "task.json");
                    if(tasks.empty()) {
                        cfgUi.error = "task.json not found or empty at: " + saveRoot.string();
                    }
                    else {
                        cfgUi.error.clear();
                        const fs::path progressPath = saveRoot / subject / "progress.csv";
                        fs::create_directories(saveRoot / subject);
                        const auto existing = loadProgressCsv(progressPath);
                        capUi.tasks = std::move(tasks);
                        capUi.progress = buildProgressFromTasks(capUi.tasks, existing);
                        saveProgressCsv(progressPath, capUi.progress);
                        capUi.currentTaskIdx = getCurrentTaskIndex(capUi.progress);
                        if(capUi.currentTaskIdx == -1) {
                            capUi.msg = "All tasks completed!";
                            capUi.currentEpisode = 0;
                        }
                        else {
                            capUi.currentEpisode = capUi.progress[static_cast<size_t>(capUi.currentTaskIdx)].completed + 1;
                            capUi.msg.clear();
                        }
                        captureState = CaptureState::IDLE;
                        page = CollectionPage::Capture;
                        const bool ok = recorder.start(cfgUi);
                        if(ok) {
                            pushUiLog("Enter capture");
                            {
                                const std::string s = recorder.streamProfilesLine();
                                if(!s.empty()) {
                                    pushUiLog(s);
                                }
                            }
                        }
                        else {
                            pushUiLog("Enter capture failed");
                        }
                    }
                }
            }

            if(!cfgUi.activeField.empty() && key > 0) {
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld = g_ctrlShortcutListening || ctrlFromMask;
                if(cfgUi.activeField == "w") {
                    handleTextInputShortcut(cfgUi.width, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "h") {
                    handleTextInputShortcut(cfgUi.height, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "fps") {
                    handleTextInputShortcut(cfgUi.fps, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "save") {
                    handleTextInputShortcut(cfgUi.saveRoot, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "sub") {
                    handleTextInputShortcut(cfgUi.subjectId, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "dur") {
                    handleTextInputShortcut(cfgUi.maxDurationSec, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "exp") {
                    handleTextInputShortcut(cfgUi.exposureMs, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "bri") {
                    handleTextInputShortcut(cfgUi.brightness, key, ctrlHeld);
                }
            }
        }
        else {
            collectionSetStage("ui_page_capture");

            if(captureState == CaptureState::DRAINING && recorder.isDrainComplete()) {
                updateReadyState();
                if(pendingResetAfterDrain) {
                    pushUiLog("Reset requested. Review delete confirmation.");
                    enterDeleteConfirm();
                }
            }

            // --- 布局计算 ---
            const int previewW = winW / 2;
            const int previewH = winH / 2;
            const int taskPanelX = previewW + 20;
            const int taskPanelW = winW - taskPanelX - 20;
            cv::Rect viewRect(0, 0, previewW, previewH);
            cv::Rect taskPanel(taskPanelX, 0, taskPanelW, winH - 160);
            cv::Rect statusRect(0, previewH, previewW, 148);
            cv::Rect logRect(0, previewH + 158, previewW, winH - previewH - 188);

            // --- 相机预览（左上角，1/2大小）---
            {
                static auto lastPreviewTime = std::chrono::steady_clock::now();
                const auto nowT = std::chrono::steady_clock::now();
                if(nowT - lastPreviewTime >= std::chrono::milliseconds(200)) {
                    latestFrameCache = recorder.latestRgbFrames();
                    lastPreviewTime = nowT;
                }
                std::vector<std::pair<std::string, cv::Mat>> frames;
                frames.reserve(latestFrameCache.size());
                for(auto &kv: latestFrameCache) {
                    frames.emplace_back(kv.first, kv.second);
                }
                std::sort(frames.begin(), frames.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
                drawRgbGrid(ui, viewRect, frames, cfgUi.widthInt(), cfgUi.heightInt(), cfgUi.fpsInt());
            }

            // --- 任务信息面板（右侧）---
            cv::rectangle(ui, taskPanel, cv::Scalar(25, 25, 25), cv::FILLED);
            cv::rectangle(ui, taskPanel, cv::Scalar(80, 80, 80), 1);
            if(capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size())) {
                const auto &task = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)];
                const auto &prog = capUi.progress[static_cast<size_t>(capUi.currentTaskIdx)];

                cv::putText(ui, task.name,
                            cv::Point(taskPanel.x + 20, taskPanel.y + 60),
                            cv::FONT_HERSHEY_DUPLEX, 1.4, cv::Scalar(255, 220, 50), 2, cv::LINE_AA);

                const std::string episodeStr = "Episode " + std::to_string(capUi.currentEpisode)
                                              + " / " + std::to_string(prog.total);
                cv::putText(ui, episodeStr,
                            cv::Point(taskPanel.x + 20, taskPanel.y + 100),
                            cv::FONT_HERSHEY_DUPLEX, 0.9, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

                // 优先显示中文描述，无中文时回退到英文
                const std::string &desc = task.description_cn.empty() ? task.description_en : task.description_cn;
                if(!desc.empty()) {
                    const int descLeft      = taskPanel.x + 20;
                    const int descTop       = taskPanel.y + 150;
                    const int descWidth     = std::max(1, taskPanel.width - 40);
                    const int descMaxHeight = std::max(120, taskPanel.height - 180);
                    const int descFontH     = choosePromptFontHeight(desc, descWidth, descMaxHeight, 72, 28);
                    const int lineGap       = std::max(12, descFontH / 3);
                    const auto lines        = wrapMultilineTextUtf8(desc, descWidth, descFontH);
                    const int descBottom    = taskPanel.y + taskPanel.height - 20;
                    int descY = descTop + descFontH;
                    for(const auto &line: lines) {
                        if(line.empty()) {
                            descY += lineGap;
                            continue;
                        }
                        if(descY > descBottom) {
                            break;
                        }
                        putTextUtf8(ui, line, cv::Point(descLeft, descY),
                                    descFontH, cv::Scalar(220, 220, 220));
                        descY += descFontH + lineGap;
                    }
                }
            }
            else if(capUi.currentTaskIdx == -1 && !capUi.tasks.empty()) {
                cv::putText(ui, "All tasks completed!",
                            cv::Point(taskPanel.x + 20, taskPanel.y + 60),
                            cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(50, 220, 50), 2, cv::LINE_AA);
            }

            // --- 状态显示区（预览正下方）---
            struct StateDisplay {
                const char *text;
                cv::Scalar  color;
                cv::Scalar  bgColor;
            };
            StateDisplay sd{};
            std::string stateEmphasisLine;
            std::string stateFootnoteLine;
            switch(captureState) {
            case CaptureState::IDLE:
                sd = {"READY", cv::Scalar(255, 255, 255), cv::Scalar(40, 40, 40)};
                break;
            case CaptureState::RECORDING:
                sd = {"RECORDING", cv::Scalar(50, 220, 50), cv::Scalar(20, 60, 20)};
                stateEmphasisLine = recorder.currentRecordingStatsLine();
                break;
            case CaptureState::DRAINING:
                sd = {"STOPPED", cv::Scalar(50, 200, 255), cv::Scalar(20, 40, 80)};
                stateEmphasisLine = recorder.captureFrameSummaryLine();
                stateFootnoteLine = "saving data";
                {
                    const std::string drain = recorder.drainStatusLine();
                    if(!drain.empty()) {
                        stateFootnoteLine += "  " + drain;
                    }
                }
                break;
            case CaptureState::STOPPED_READY:
                sd = {"STOPPED - CONFIRM", cv::Scalar(50, 200, 255), cv::Scalar(20, 40, 80)};
                stateEmphasisLine = recorder.captureFrameSummaryLine();
                break;
            case CaptureState::DELETE_CONFIRM:
                sd = {"DELETE CONFIRM", cv::Scalar(255, 180, 80), cv::Scalar(80, 40, 20)};
                break;
            }
            cv::rectangle(ui, statusRect, sd.bgColor, cv::FILLED);
            cv::rectangle(ui, statusRect, sd.color, 2);
            {
                int baseline = 0;
                const auto textSz = cv::getTextSize(sd.text, cv::FONT_HERSHEY_DUPLEX, 1.55, 3, &baseline);
                cv::Point textOrg(statusRect.x + (statusRect.width - textSz.width) / 2,
                                  statusRect.y + 50 + textSz.height / 2);
                cv::putText(ui, sd.text, textOrg, cv::FONT_HERSHEY_DUPLEX, 1.55, sd.color, 3, cv::LINE_AA);
            }
            if(!stateEmphasisLine.empty()) {
                int baseline = 0;
                const auto textSz = cv::getTextSize(stateEmphasisLine, cv::FONT_HERSHEY_DUPLEX, 0.9, 2, &baseline);
                cv::Point textOrg(statusRect.x + (statusRect.width - textSz.width) / 2,
                                  statusRect.y + 100);
                cv::putText(ui, stateEmphasisLine, textOrg, cv::FONT_HERSHEY_DUPLEX, 0.9, cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
            }
            if(!stateFootnoteLine.empty()) {
                int baseline = 0;
                const auto textSz = cv::getTextSize(stateFootnoteLine, cv::FONT_HERSHEY_DUPLEX, 0.55, 1, &baseline);
                cv::Point textOrg(statusRect.x + (statusRect.width - textSz.width) / 2,
                                  statusRect.y + statusRect.height - 14);
                cv::putText(ui, stateFootnoteLine, textOrg, cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            }

            // --- 日志区（左下角）---
            if(logRect.height > 40) {
                cv::rectangle(ui, logRect, cv::Scalar(18, 18, 18), cv::FILLED);
                cv::rectangle(ui, logRect, cv::Scalar(80, 80, 80), 1);
                cv::putText(ui, "Log", cv::Point(logRect.x + 8, logRect.y + 18),
                            cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

                if(fm.wheelDelta != 0 && logRect.contains(cv::Point(fm.x, fm.y))) {
                    const int step = (fm.wheelDelta > 0) ? 3 : -3;
                    logScroll = std::max(0, logScroll + step);
                }

                const int    lineH     = 18;
                const int    maxLines  = std::max(1, (logRect.height - 28) / lineH);
                const int    textLeft  = logRect.x + 8;
                const int    textTop   = logRect.y + 36;
                const int    textWidth = std::max(1, logRect.width - 16);
                const int    fontFace  = cv::FONT_HERSHEY_DUPLEX;
                const double fontScale = 0.45;
                const int    thickness = 1;

                std::vector<std::string> allLines;
                allLines.reserve(512);
                {
                    std::string head;
                    if(captureState == CaptureState::RECORDING) {
                        std::ostringstream oss;
                        oss.setf(std::ios::fixed);
                        oss << "Elapsed: " << std::setprecision(2) << recorder.currentRecordingSeconds() << " s";
                        head = oss.str();
                    }
                    else if(captureState == CaptureState::DRAINING) {
                        head = "saving data";
                        const std::string drain = recorder.drainStatusLine();
                        if(!drain.empty()) {
                            head += "  " + drain;
                        }
                    }
                    else {
                        const double lastSec = recorder.lastRecordedSeconds();
                        if(lastSec > 0.0) {
                            std::ostringstream oss;
                            oss.setf(std::ios::fixed);
                            oss << "Recorded: " << std::setprecision(2) << lastSec << " s";
                            head = oss.str();
                        }
                    }
                    if(!head.empty()) {
                        auto w = wrapTextToWidth(head, textWidth, fontFace, fontScale, thickness);
                        allLines.insert(allLines.end(), w.begin(), w.end());
                    }
                }
                for(const auto &msg: uiLogs) {
                    auto w = wrapTextToWidth(msg, textWidth, fontFace, fontScale, thickness);
                    allLines.insert(allLines.end(), w.begin(), w.end());
                }

                int maxScroll = 0;
                if(static_cast<int>(allLines.size()) > maxLines) {
                    maxScroll = static_cast<int>(allLines.size()) - maxLines;
                }
                logScroll = std::min(logScroll, maxScroll);
                const int startLine = std::max(0, static_cast<int>(allLines.size()) - maxLines - logScroll);

                int y = textTop;
                for(int i = startLine; i < static_cast<int>(allLines.size()) && (i - startLine) < maxLines; i++) {
                    cv::putText(ui, allLines[static_cast<size_t>(i)], cv::Point(textLeft, y),
                                fontFace, fontScale, cv::Scalar(210, 210, 210), thickness, cv::LINE_AA);
                    y += lineH;
                }
            }

            // --- 状态机：允许操作判断 ---
            const bool modalDelete = (captureState == CaptureState::DELETE_CONFIRM);
            const bool allowStart  = !modalDelete && (captureState == CaptureState::IDLE) && (capUi.currentTaskIdx != -1);
            const bool allowStop   = !modalDelete && (captureState == CaptureState::RECORDING);
            const bool allowSave   = !modalDelete && (captureState == CaptureState::STOPPED_READY) && recorder.hasData();
            const bool allowReset  = !modalDelete
                                     && (captureState == CaptureState::RECORDING
                                         || captureState == CaptureState::STOPPED_READY
                                         || (captureState == CaptureState::DRAINING && !pendingResetAfterDrain));
            const bool allowNav    = !modalDelete && (captureState == CaptureState::IDLE);

            // --- 右侧按钮区 ---
            const int btnX = taskPanelX + 20;
            const int btnW = taskPanelW - 40;
            cv::Rect bStart(btnX, winH - 280, btnW, 50);
            cv::Rect bStop (btnX, winH - 220, btnW, 50);
            cv::Rect bSave (btnX, winH - 160, btnW, 50);
            cv::Rect bReset(btnX, winH - 100, btnW, 50);
            cv::Rect bMenu (btnX, winH - 38,  btnW / 2 - 5, 30);
            cv::Rect bConfig(btnX + btnW / 2 + 5, winH - 38, btnW / 2 - 5, 30);

            bool doStart    = uiButtonEx(ui, bStart, "Start  [Ctrl+1]", fm, allowStart);
            bool doStop     = uiButtonEx(ui, bStop,  "Stop   [Ctrl+2]", fm, allowStop);
            bool doSave     = uiButtonEx(ui, bSave,  "Confirm [Ctrl+3]", fm, allowSave);
            bool doReset    = uiButtonEx(ui, bReset, "Reset  [Ctrl+4]", fm, allowReset);
            bool doBackMenu = uiButtonEx(ui, bMenu,  "Menu",            fm, allowNav);
            bool doBackCfg  = uiButtonEx(ui, bConfig,"Config",          fm, allowNav);
            bool doDeleteConfirm = false;
            bool doDeleteCancel  = false;

            // --- Ctrl+1/2/3/4 快捷键 ---
            // 说明：当前环境里 Ctrl 按下会单独上报 0xE3/0xE4，数字键再单独上报 ASCII；
            // 因此进入 Ctrl 监听状态后，直到收到 Ctrl 释放事件前，1~4 都视为快捷键。
            if(key > 0) {
                if(std::getenv("COLLECTION_KEY_DEBUG") != nullptr) {
                    fprintf(stderr,
                            "[KEY_DEBUG] raw=0x%08X dec=%d lo8=0x%02X lo16=0x%04X ctrlDown=%d ctrlUp=%d listening=%d maskCtrl=%d\n",
                            key, key, key & 0xFF, key & 0xFFFF, isCtrlModifierKeyEvent(key) ? 1 : 0,
                            isCtrlReleaseKeyEvent(key) ? 1 : 0, g_ctrlShortcutListening ? 1 : 0,
                            (((key & 0x20000) != 0) || ((key & 0x04000000) != 0)) ? 1 : 0);
                    fflush(stderr);
                }
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld     = g_ctrlShortcutListening || ctrlFromMask;
                const int  baseKey      = key & 0xFFFF;
                if(ctrlHeld) {
                    if(modalDelete) {
                        if(baseKey == '1') {
                            doDeleteConfirm = true;
                        }
                        else if(baseKey == '4') {
                            doDeleteCancel = true;
                        }
                    }
                    else {
                        if(baseKey == '1') {
                            doStart = allowStart;
                        }
                        else if(baseKey == '2') {
                            doStop = allowStop;
                        }
                        else if(baseKey == '3') {
                            doSave = allowSave;
                        }
                        else if(baseKey == '4') {
                            doReset = allowReset;
                        }
                    }
                }
            }

            if(modalDelete) {
                cv::Mat shade(ui.size(), ui.type(), cv::Scalar(0, 0, 0));
                cv::addWeighted(shade, 0.58, ui, 0.42, 0.0, ui);

                const int modalW = std::min(920, winW - 60);
                const int modalH = 340;
                const cv::Rect modal((winW - modalW) / 2, (winH - modalH) / 2, modalW, modalH);
                cv::rectangle(ui, modal, cv::Scalar(28, 28, 28), cv::FILLED);
                cv::rectangle(ui, modal, cv::Scalar(230, 230, 230), 3);

                const std::string sessionLabel = recorder.currentSessionLabel();
                const std::string modalTitle = "Discard Current Episode?";
                const std::string modalWarn = "This will permanently delete the current episode data.";
                const int textLeft = modal.x + 28;
                const int textWidth = modal.width - 56;
                {
                    int baseline = 0;
                    const auto titleSz = cv::getTextSize(modalTitle, cv::FONT_HERSHEY_DUPLEX, 1.25, 3, &baseline);
                    cv::Point titleOrg(modal.x + (modal.width - titleSz.width) / 2, modal.y + 56);
                    cv::putText(ui, modalTitle, titleOrg,
                                cv::FONT_HERSHEY_DUPLEX, 1.25, cv::Scalar(255, 220, 120), 3, cv::LINE_AA);
                }
                {
                    auto warnLines = wrapTextToWidth(modalWarn, textWidth, cv::FONT_HERSHEY_DUPLEX, 0.82, 2);
                    int y = modal.y + 108;
                    for(const auto &line : warnLines) {
                        cv::putText(ui, line, cv::Point(textLeft, y),
                                    cv::FONT_HERSHEY_DUPLEX, 0.82, cv::Scalar(235, 235, 235), 2, cv::LINE_AA);
                        y += 34;
                    }
                }
                cv::putText(ui, "Target:", cv::Point(textLeft, modal.y + 180),
                            cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
                {
                    const auto labelLines = wrapTextToWidth(sessionLabel.empty() ? "(unknown session)" : sessionLabel,
                                                            textWidth, cv::FONT_HERSHEY_DUPLEX, 0.95, 2);
                    int y = modal.y + 220;
                    for(const auto &line : labelLines) {
                        cv::putText(ui, line, cv::Point(textLeft, y),
                                    cv::FONT_HERSHEY_DUPLEX, 0.95, cv::Scalar(80, 200, 255), 2, cv::LINE_AA);
                        y += 36;
                    }
                }
                cv::putText(ui, "Ctrl+1 confirm delete    Ctrl+4 cancel",
                            cv::Point(textLeft, modal.y + modal.height - 82),
                            cv::FONT_HERSHEY_DUPLEX, 0.68, cv::Scalar(200, 200, 200), 2, cv::LINE_AA);

                cv::Rect bDeleteOk(modal.x + 28, modal.y + modal.height - 58, (modal.width - 68) / 2, 46);
                cv::Rect bDeleteCancel(modal.x + bDeleteOk.width + 40, modal.y + modal.height - 58, (modal.width - 68) / 2, 46);
                doDeleteConfirm = doDeleteConfirm || uiButtonEx(ui, bDeleteOk, "Confirm Delete [Ctrl+1]", fm, true);
                doDeleteCancel  = doDeleteCancel  || uiButtonEx(ui, bDeleteCancel, "Cancel [Ctrl+4]", fm, true);
            }

            // --- 处理按钮动作 ---
            if(doBackMenu) {
                collectionSetStage("ui_capture_back_menu");
                recorder.stopIfRunning();
                running = false;
            }
            if(doBackCfg) {
                collectionSetStage("ui_capture_back_config");
                recorder.stopIfRunning();
                page = CollectionPage::Config;
            }
            if(doStart) {
                collectionSetStage("ui_capture_start");
                cfgUi.enforceRules();
                const fs::path root = fs::path(trimString(cfgUi.saveRoot));
                const std::string subject = trimString(cfgUi.subjectId);
                const std::string taskName = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name;
                const int episodeN = capUi.currentEpisode;
                const bool ok = recorder.start(cfgUi) && recorder.beginRecord(root, subject, taskName, episodeN);
                if(!ok) {
                    capUi.msg = "Failed to start capture";
                    pushUiLog("Start failed");
                    {
                        const std::string line = recorder.lastInfoLine();
                        if(!line.empty()) {
                            pushUiLog(line);
                        }
                    }
                }
                else {
                    captureState = CaptureState::RECORDING;
                    pendingResetAfterDrain = false;
                    capUi.msg.clear();
                    pushUiLog("Recording: " + capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name
                              + " ep" + std::to_string(capUi.currentEpisode));
                    {
                        const std::string label = recorder.currentSessionLabel();
                        if(!label.empty()) {
                            pushUiLog("Session: " + label);
                        }
                    }
                    {
                        const std::string s = recorder.streamProfilesLine();
                        if(!s.empty()) {
                            pushUiLog(s);
                        }
                    }
                }
            }
            if(doStop) {
                collectionSetStage("ui_capture_stop");
                recorder.stopRecording();
                if(recorder.isDrainComplete()) {
                    updateReadyState();
                }
                else {
                    captureState = CaptureState::DRAINING;
                    capUi.msg = "Saving data to disk...";
                    pushUiLog("Stopped. Waiting for background save to finish.");
                }
            }
            if(doReset) {
                collectionSetStage("ui_capture_reset");
                if(captureState == CaptureState::RECORDING) {
                    recorder.stopRecording();
                    pendingResetAfterDrain = true;
                    if(recorder.isDrainComplete()) {
                        updateReadyState();
                        pushUiLog("Reset requested. Review delete confirmation.");
                        enterDeleteConfirm();
                    }
                    else {
                        captureState = CaptureState::DRAINING;
                        capUi.msg = "Saving data to disk before reset...";
                        pushUiLog("Reset requested during recording. Waiting for background save to finish.");
                    }
                }
                else if(captureState == CaptureState::DRAINING) {
                    pendingResetAfterDrain = true;
                    capUi.msg = "Saving data to disk before reset...";
                    pushUiLog("Reset requested. Waiting for background save to finish.");
                }
                else if(captureState == CaptureState::STOPPED_READY) {
                    enterDeleteConfirm();
                }
            }
            if(doSave) {
                collectionSetStage("ui_capture_save");
                const bool confirmed = recorder.confirmCurrentSession();
                if(confirmed) {
                    const int idx = capUi.currentTaskIdx;
                    if(idx >= 0 && idx < static_cast<int>(capUi.progress.size())) {
                        capUi.progress[static_cast<size_t>(idx)].completed += 1;
                        const fs::path root = fs::path(trimString(cfgUi.saveRoot));
                        const std::string subject = trimString(cfgUi.subjectId);
                        saveProgressCsv(root / subject / "progress.csv", capUi.progress);
                        const double maxDiff = recorder.lastAlignedMaxDiffMs();
                        if(maxDiff > 0.0) {
                            std::ostringstream oss;
                            oss.setf(std::ios::fixed);
                            oss << "MaxTsDiff: " << std::setprecision(3) << maxDiff << " ms";
                            pushUiLog(oss.str());
                        }
                    }

                    capUi.currentTaskIdx = getCurrentTaskIndex(capUi.progress);
                    if(capUi.currentTaskIdx == -1) {
                        capUi.msg = "All tasks completed!";
                        capUi.currentEpisode = 0;
                        pushUiLog("Confirm OK. All tasks completed.");
                    }
                    else {
                        capUi.currentEpisode = capUi.progress[static_cast<size_t>(capUi.currentTaskIdx)].completed + 1;
                        capUi.msg = "Capture confirmed";
                        pushUiLog("Confirm OK. Next: " + capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name
                                  + " ep" + std::to_string(capUi.currentEpisode));
                    }
                    recorder.clearStatus();
                    captureState = CaptureState::IDLE;
                    pendingResetAfterDrain = false;
                }
                else {
                    capUi.msg = "Confirm failed: session not ready";
                    pushUiLog("Confirm failed");
                }
            }
            if(doDeleteConfirm) {
                collectionSetStage("ui_capture_delete_confirm");
                std::string error;
                if(recorder.discardCurrentSession(&error)) {
                    recorder.clearStatus();
                    captureState = CaptureState::IDLE;
                    pendingResetAfterDrain = false;
                    capUi.msg = "Capture discarded";
                    pushUiLog("Reset OK. Current episode discarded.");
                }
                else {
                    capUi.msg = "Delete failed";
                    pushUiLog("Delete failed: " + error);
                }
            }
            if(doDeleteCancel) {
                collectionSetStage("ui_capture_delete_cancel");
                captureState = CaptureState::STOPPED_READY;
                pendingResetAfterDrain = false;
                capUi.msg = "Delete canceled";
                pushUiLog("Delete canceled");
            }

            // --- 超时自动停止 ---
            if(captureState == CaptureState::RECORDING && recorder.autoStopIfTimeout()) {
                if(recorder.isDrainComplete()) {
                    updateReadyState();
                    capUi.msg = "Auto-stopped by max duration";
                }
                else {
                    captureState = CaptureState::DRAINING;
                    capUi.msg = "Auto-stopped. Saving data to disk...";
                }
                pushUiLog("Auto stop by max duration");
            }

            // --- 消息显示 ---
            if(!capUi.msg.empty()) {
                std::string lowerMsg = capUi.msg;
                std::transform(lowerMsg.begin(), lowerMsg.end(), lowerMsg.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                const bool isError = (lowerMsg.find("fail") != std::string::npos) || (lowerMsg.find("error") != std::string::npos);
                const cv::Scalar msgColor = isError ? cv::Scalar(60, 60, 255) : cv::Scalar(80, 200, 80);
                cv::putText(ui, capUi.msg, cv::Point(4, winH - 4),
                            cv::FONT_HERSHEY_DUPLEX, 0.6, msgColor, 1, cv::LINE_AA);
            }

            // --- Info 行 ---
            {
                std::string info = recorder.lastInfoLine();
                if(info.empty() && captureState == CaptureState::DRAINING) {
                    info = recorder.drainStatusLine();
                }
                if(!info.empty()) {
                    cv::putText(ui, info, cv::Point(4, winH - 20),
                                cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
                }
            }
        }

        collectionSetStage("ui_imshow");
        cv::imshow(winName, ui);
    }

    collectionSetStage("ui_exit_done");
    cv::destroyWindow(winName);
    return 0;
}

}  // namespace sync_app
