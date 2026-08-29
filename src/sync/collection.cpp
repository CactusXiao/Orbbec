#include "collection.hpp"
#include "ego.hpp"
#include "fisheyes.hpp"
#include "tactile.hpp"
#include "task_backend_client.hpp"

#include "utils/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <future>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <unordered_set>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <execinfo.h>
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

static std::string elideTextToWidth(std::string text, int maxWidthPx, int fontFace, double fontScale, int thickness) {
    if(text.empty() || maxWidthPx <= 0) {
        return text;
    }
    auto widthOf = [&](const std::string &s) {
        int baseline = 0;
        return cv::getTextSize(s, fontFace, fontScale, thickness, &baseline).width;
    };
    if(widthOf(text) <= maxWidthPx) {
        return text;
    }
    const std::string suffix = "...";
    while(!text.empty() && widthOf(text + suffix) > maxWidthPx) {
        text.pop_back();
    }
    return text.empty() ? suffix : text + suffix;
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

static std::string bytesToLowerHex(const std::vector<uint8_t> &bytes) {
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0');
    for(uint8_t byte : bytes) {
        oss << std::setw(2) << static_cast<int>(byte);
    }
    return oss.str();
}

static std::string sanitizeCsvIdentifier(std::string s) {
    s = trimString(std::move(s));
    if(s.empty()) {
        return "touch";
    }
    for(char &ch : s) {
        if(!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    while(s.find("__") != std::string::npos) {
        s.replace(s.find("__"), 2, "_");
    }
    if(!s.empty() && s.front() == '_') {
        s.erase(s.begin());
    }
    if(!s.empty() && s.back() == '_') {
        s.pop_back();
    }
    return s.empty() ? std::string("touch") : s;
}

static std::string touchStreamIdForConfig(const TactileModuleConfig &cfg, size_t fallbackIndex) {
    if(!cfg.streamId.empty()) {
        return sanitizeCsvIdentifier(cfg.streamId);
    }
    if(!cfg.handSide.empty()) {
        return sanitizeCsvIdentifier(cfg.handSide);
    }
    if(cfg.sensorType > 0) {
        return "sensor" + std::to_string(cfg.sensorType);
    }
    return "touch" + std::to_string(fallbackIndex);
}

static std::string touchDisplayName(const std::string &streamId) {
    return "touch " + streamId + " pressure";
}

static std::vector<TactileModuleConfig> expandTouchDeviceConfigs(const TactileModuleConfig &base) {
    std::vector<TactileModuleConfig> out;
    if(base.devices.empty()) {
        TactileModuleConfig cfg = base;
        cfg.devices.clear();
        cfg.streamId = touchStreamIdForConfig(cfg, 0);
        out.push_back(std::move(cfg));
        return out;
    }

    out.reserve(base.devices.size());
    for(size_t i = 0; i < base.devices.size(); ++i) {
        const auto &dev = base.devices[i];
        TactileModuleConfig cfg = base;
        cfg.devices.clear();
        if(!dev.streamId.empty()) {
            cfg.streamId = dev.streamId;
        }
        if(!dev.handSide.empty()) {
            cfg.handSide = dev.handSide;
        }
        if(dev.sensorType > 0) {
            cfg.sensorType = dev.sensorType;
        }
        cfg.serial = dev.serial;
        cfg.streamId = touchStreamIdForConfig(cfg, i);
        out.push_back(std::move(cfg));
    }

    std::unordered_map<std::string, size_t> seen;
    for(size_t i = 0; i < out.size(); ++i) {
        auto &id = out[i].streamId;
        const size_t count = seen[id]++;
        if(count > 0) {
            id += "_" + std::to_string(count + 1);
        }
    }
    return out;
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

static std::optional<StreamType> dataTypeStreamType(CollectDataType t) {
    switch(t) {
    case CollectDataType::RGB:
        return StreamType::Color;
    case CollectDataType::Depth:
        return StreamType::Depth;
    case CollectDataType::IRLeft:
    case CollectDataType::IRRight:
        return StreamType::IR;
    case CollectDataType::CloudPoints:
    case CollectDataType::ColorCloudPoints:
        return StreamType::PointCloud;
    }
    return std::nullopt;
}

static std::string presetLabel(int w, int h, int fps) {
    return std::to_string(w) + "x" + std::to_string(h) + "@" + std::to_string(fps);
}

static constexpr int kCollectionFixedWidth = 640;
static constexpr int kCollectionFixedHeight = 400;
static constexpr int kCollectionFixedFps = 30;
static constexpr int kCollectionFixedMaxDurationSec = 0;

struct CollectionConfigUi {
    bool enableMultiview = true;
    bool enableFisheyes  = false;
    bool enableEgo       = false;
    bool enableTouch     = false;
    std::string saveRoot;
    std::string subjectId = "test";
    std::string exposureMs;
    std::string brightness;
    std::string activeField;
    std::string error;
    std::string notice;

    void enforceRules() {}

    bool hasRequiredFields() const {
        return !trimString(saveRoot).empty() && !trimString(subjectId).empty();
    }

    bool hasSelectedCaptureType() const {
        return enableMultiview || enableFisheyes || enableEgo;
    }

    int widthInt() const { return kCollectionFixedWidth; }
    int heightInt() const { return kCollectionFixedHeight; }
    int fpsInt() const { return kCollectionFixedFps; }
    float exposureMsFloat() const {
        if(trimString(exposureMs).empty()) {
            return 0.0f;
        }
        return static_cast<float>(parseDoubleBound(exposureMs, 0.0, 0.05, 100.0));
    }
    int brightnessInt() const { return trimString(brightness).empty() ? -1 : parseIntOr(brightness, -1); }
    int maxDurationInt() const { return kCollectionFixedMaxDurationSec; }

    std::vector<CollectDataType> enabledTypesForSaving() const {
        return { CollectDataType::RGB, CollectDataType::Depth };
    }

    std::vector<CollectDataType> enabledTypesForStreaming() const {
        return enabledTypesForSaving();
    }

    CollectDataType referenceType() const {
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
    std::chrono::steady_clock::time_point latestRgbSteady{};
    cv::Mat   latestDepthAlignedRgb;
    uint64_t  latestDepthTsUs = 0;
    float     latestDepthValueScaleMm = 0.0f;
    std::chrono::steady_clock::time_point latestDepthSteady{};

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

static OBFormat configuredStreamFormat(const DeviceConfig &cfg, CollectDataType type) {
    const auto streamType = dataTypeStreamType(type);
    if(!streamType.has_value()) {
        return OB_FORMAT_UNKNOWN;
    }
    for(const auto &stream: cfg.streams) {
        if(stream.type != *streamType) {
            continue;
        }
        const std::string fmt = trimString(stream.format);
        if(fmt.empty()) {
            continue;
        }
        return stringToOBFormat(fmt, *streamType);
    }
    return OB_FORMAT_UNKNOWN;
}

static std::shared_ptr<ob::VideoStreamProfile> pickVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe,
                                                                OBSensorType sensorType,
                                                                int width,
                                                                int height,
                                                                int fps,
                                                                OBFormat desiredFormat = OB_FORMAT_UNKNOWN) {
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
    auto findBest = [&](int targetW, int targetH, int targetFps, OBFormat requiredFormat) {
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
            if(requiredFormat != OB_FORMAT_UNKNOWN && vp->getFormat() != requiredFormat) {
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

    auto warnIfFormatFallback = [&](const std::shared_ptr<ob::VideoStreamProfile> &profile) {
        if(profile && desiredFormat != OB_FORMAT_UNKNOWN && profile->getFormat() != desiredFormat) {
            std::cerr << "[collection] warning: requested stream format " << static_cast<int>(desiredFormat)
                      << " unavailable, using format " << static_cast<int>(profile->getFormat()) << std::endl;
        }
    };
    auto tryResolution = [&](int targetW, int targetH) -> std::shared_ptr<ob::VideoStreamProfile> {
        if(desiredFormat != OB_FORMAT_UNKNOWN) {
            if(auto best = findBest(targetW, targetH, fps, desiredFormat)) {
                return best;
            }
        }
        if(auto best = findBest(targetW, targetH, fps, OB_FORMAT_UNKNOWN)) {
            warnIfFormatFallback(best);
            return best;
        }
        return nullptr;
    };

    if(width > 0 || height > 0) {
        if(auto best = tryResolution(width, height)) {
            return best;
        }

        std::vector<std::pair<int, int>> fallbacks;
        if(sensorType == OB_SENSOR_DEPTH || sensorType == OB_SENSOR_IR || sensorType == OB_SENSOR_IR_LEFT || sensorType == OB_SENSOR_IR_RIGHT) {
            fallbacks = { { 640, 400 }, { 1280, 800 }, { 320, 200 } };
        }
        else if(sensorType == OB_SENSOR_COLOR) {
            fallbacks = { { 640, 400 }, { 1280, 800 }, { 1280, 720 }, { 1920, 1080 }, { 640, 480 }, { 640, 360 } };
        }
        for(const auto &fallback: fallbacks) {
            if(fallback.first == width && fallback.second == height) {
                continue;
            }
            if(auto best = tryResolution(fallback.first, fallback.second)) {
                return best;
            }
        }
    }
    else if(auto best = tryResolution(width, height)) {
        return best;
    }

    if(desiredFormat != OB_FORMAT_UNKNOWN) {
        for(uint32_t i = 0; i < list->getCount(); i++) {
            auto p  = list->getProfile(i);
            auto vp = p->as<ob::VideoStreamProfile>();
            if(vp && vp->getFormat() == desiredFormat) {
                return vp;
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

static void saveMatToRawFile(const cv::Mat &m, const fs::path &path) {
    if(m.empty()) {
        return;
    }
    std::ofstream ofs(path, std::ios::binary);
    if(!ofs.is_open()) {
        return;
    }
    if(m.isContinuous()) {
        const size_t bytes = static_cast<size_t>(m.total()) * m.elemSize();
        ofs.write(reinterpret_cast<const char *>(m.data), static_cast<std::streamsize>(bytes));
        return;
    }
    const size_t rowBytes = static_cast<size_t>(m.cols) * m.elemSize();
    for(int r = 0; r < m.rows; ++r) {
        ofs.write(reinterpret_cast<const char *>(m.ptr(r)), static_cast<std::streamsize>(rowBytes));
    }
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

static int runCommandCapture(const std::string &command, std::string &output) {
    output.clear();
    FILE *pipe = popen(command.c_str(), "r");
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
    const int status = pclose(pipe);
#if defined(__unix__) || defined(__APPLE__)
    if(status == -1) {
        return -1;
    }
    if(WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
#endif
    return status == 0 ? 0 : -1;
}

static std::string sanitizePathComponent(std::string value) {
    value = trimString(std::move(value));
    if(value.empty()) {
        return "item";
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
    return out.empty() ? "item" : out;
}

static void replaceAllInPlace(std::string &s, const std::string &needle, const std::string &replacement) {
    if(needle.empty()) {
        return;
    }
    size_t pos = 0;
    while((pos = s.find(needle, pos)) != std::string::npos) {
        s.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

class VoiceAnnouncer {
    struct VoiceMessage {
        std::string key;
        std::string text;
    };

public:
    explicit VoiceAnnouncer(VoiceFeedbackConfig cfg)
        : cfg_(std::move(cfg)) {
        if(cfg_.enabled) {
            worker_ = std::thread([this]() { workerLoop(); });
        }
    }

    ~VoiceAnnouncer() {
        stop();
    }

    void say(const std::string &messageKey, const std::string &fallbackText) {
        if(!cfg_.enabled) {
            return;
        }
        const bool beepMessage = isBeepMessage(messageKey);
        std::string text = fallbackText;
        auto it = cfg_.messages.find(messageKey);
        if(it != cfg_.messages.end()) {
            text = it->second;
        }
        text = trimString(std::move(text));
        if(text.empty() && !beepMessage) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if(beepMessage) {
                const bool voicePendingOrPlaying = speakingNonBeep_
                                                   || std::any_of(queue_.begin(), queue_.end(), [](const VoiceMessage &message) {
                                                          return !isBeepMessage(message.key);
                                                      });
                const bool beepPending = std::any_of(queue_.begin(), queue_.end(), [](const VoiceMessage &message) {
                    return isBeepMessage(message.key);
                });
                if(voicePendingOrPlaying || beepPending) {
                    return;
                }
            }
            else {
                queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const VoiceMessage &message) {
                                 return isBeepMessage(message.key);
                             }),
                             queue_.end());
                if(messageKey == "record_elapsed") {
                    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const VoiceMessage &message) {
                                     return message.key == "record_elapsed";
                                 }),
                                 queue_.end());
                }
            }
            queue_.push_back(VoiceMessage{ messageKey, std::move(text) });
        }
        cv_.notify_one();
    }

    void clearPending(const std::string &messageKey) {
        if(!cfg_.enabled) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.erase(std::remove_if(queue_.begin(), queue_.end(),
                                    [&](const VoiceMessage &message) { return message.key == messageKey; }),
                     queue_.end());
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopping_ = true;
        }
        cv_.notify_all();
        if(worker_.joinable()) {
            worker_.join();
        }
    }

private:
    static bool isBeepMessage(const std::string &messageKey) {
        return messageKey == "record_tick";
    }

    static bool containsNonAscii(const std::string &text) {
        return std::any_of(text.begin(), text.end(), [](unsigned char c) {
            return c >= 0x80;
        });
    }

    static std::string commandMessage(const std::string &message) {
        return "printf '%s\\n' " + shellQuote("[collection][voice] " + message) + " >&2";
    }

    std::string buildMechanicalChineseFallback(const std::string &qText) const {
        return "if command -v spd-say >/dev/null 2>&1; then spd-say -w -l zh-cn -- " + qText + "; "
               "elif command -v ekho >/dev/null 2>&1; then ekho -v Mandarin " + qText + "; "
               "elif command -v espeak-ng >/dev/null 2>&1; then espeak-ng -v cmn -s 155 " + qText + "; "
               "elif command -v espeak >/dev/null 2>&1; then espeak -v zh -s 155 " + qText + "; "
               "else false; fi";
    }

    std::string buildNaturalChineseCommand(const std::string &qText) const {
        const std::string voice = trimString(cfg_.voice).empty() ? "zh-CN-XiaoxiaoNeural" : trimString(cfg_.voice);
        const std::string rate = trimString(cfg_.rate);
        const std::string pitch = trimString(cfg_.pitch);

        std::ostringstream cmd;
        cmd << "(tmp=$(mktemp /tmp/orbbec-voice.XXXXXX.mp3); rm -f \"$tmp\"; "
            << "cleanup(){ rm -f \"$tmp\"; }; trap cleanup EXIT; "
            << "play_voice(){ "
            << "if command -v ffplay >/dev/null 2>&1; then ffplay -nodisp -autoexit -loglevel quiet \"$1\"; "
            << "elif command -v mpv >/dev/null 2>&1; then mpv --really-quiet --no-video \"$1\"; "
            << "elif command -v mpg123 >/dev/null 2>&1; then mpg123 -q \"$1\"; "
            << "else return 1; fi; "
            << "}; "
            << "run_edge_tts(){ "
            << "if command -v edge-tts >/dev/null 2>&1; then edge-tts \"$@\"; "
            << "elif command -v python3 >/dev/null 2>&1 && python3 -c 'import edge_tts' >/dev/null 2>&1; then python3 -m edge_tts \"$@\"; "
            << "else return 127; fi; "
            << "}; "
            << "if run_edge_tts --voice " << shellQuote(voice);
        if(!rate.empty()) {
            cmd << " --rate " << shellQuote(rate);
        }
        if(!pitch.empty()) {
            cmd << " --pitch " << shellQuote(pitch);
        }
        cmd << " --text " << qText << " --write-media \"$tmp\" >/dev/null 2>&1 && play_voice \"$tmp\"; then exit 0; fi; ";
        if(cfg_.naturalOnly) {
            cmd << commandMessage("natural Chinese TTS failed. Install edge-tts and ffmpeg: python3 -m pip install --user edge-tts; sudo apt install ffmpeg")
                << "; exit 1)";
        }
        else {
            cmd << "rm -f \"$tmp\"; " << buildMechanicalChineseFallback(qText) << ")";
        }
        return cmd.str();
    }

    std::string buildBeepCommand() const {
        const int frequencyHz = std::clamp(cfg_.recordTickFrequencyHz, 200, 2000);
        const int durationMs = std::clamp(cfg_.recordTickDurationMs, 20, 500);
        const double volume = std::clamp(cfg_.recordTickVolume, 0.01, 1.0);
        const double durationSeconds = static_cast<double>(durationMs) / 1000.0;
        const int ffplayVolume = std::clamp(static_cast<int>(std::lround(volume * 100.0)), 1, 100);
        std::ostringstream tone;
        tone << std::fixed << std::setprecision(3);
#if defined(__APPLE__)
        tone << "(if command -v ffplay >/dev/null 2>&1; then "
             << "ffplay -nodisp -autoexit -loglevel quiet -volume " << ffplayVolume
             << " -f lavfi -i 'sine=frequency=" << frequencyHz << ":duration=" << durationSeconds << "'; "
             << "elif command -v play >/dev/null 2>&1; then "
             << "play -q -n synth " << durationSeconds << " sine " << frequencyHz << " vol " << volume << "; "
             << "elif command -v afplay >/dev/null 2>&1; then "
             << "afplay -v " << volume << " /System/Library/Sounds/Tink.aiff; fi)";
#else
        const std::string device = cfg_.speakerDevice.empty() ? "default" : cfg_.speakerDevice;
        const std::string qDevice = shellQuote(device);
        tone << "(if command -v ffplay >/dev/null 2>&1; then "
             << "ffplay -nodisp -autoexit -loglevel quiet -volume " << ffplayVolume
             << " -f lavfi -i 'sine=frequency=" << frequencyHz << ":duration=" << durationSeconds << "'; "
             << "elif command -v play >/dev/null 2>&1; then "
             << "play -q -n synth " << durationSeconds << " sine " << frequencyHz << " vol " << volume << "; "
             << "elif command -v aplay >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then "
             << "python3 -c 'import math,struct,sys,wave; "
             << "rate=16000; count=int(rate*" << durationSeconds << "); "
             << "w=wave.open(sys.stdout.buffer,\"wb\"); w.setparams((1,2,rate,count,\"NONE\",\"not compressed\")); "
             << "w.writeframes(b\"\".join(struct.pack(\"<h\",int(32767*" << volume
             << "*math.sin(2*math.pi*" << frequencyHz << "*i/rate))) for i in range(count))); w.close()' "
             << "| aplay -q -D " << qDevice << "; fi)";
#endif
        return tone.str();
    }

    std::string buildCommand(const std::string &text) const {
        const std::string device = cfg_.speakerDevice.empty() ? "default" : cfg_.speakerDevice;
        const std::string qText = shellQuote(text);
        const std::string qDevice = shellQuote(device);
        if(!cfg_.command.empty()) {
            std::string cmd = cfg_.command;
            const bool hasTextPlaceholder = cmd.find("{text}") != std::string::npos;
            replaceAllInPlace(cmd, "{text}", qText);
            replaceAllInPlace(cmd, "{device}", qDevice);
            if(!hasTextPlaceholder) {
                cmd += " " + qText;
            }
            return cmd;
        }

#if defined(__APPLE__)
        std::ostringstream cmd;
        cmd << "if command -v say >/dev/null 2>&1; then say ";
        if(containsNonAscii(text)) {
            cmd << "-v Tingting ";
        }
        if(!device.empty() && device != "default") {
            cmd << "-a " << qDevice << " ";
        }
        cmd << qText << "; fi";
        return cmd.str();
#else
        if(containsNonAscii(text)) {
            return buildNaturalChineseCommand(qText);
        }
        return "(if command -v spd-say >/dev/null 2>&1 && spd-say -w -- " + qText + "; then :; "
               "elif command -v espeak-ng >/dev/null 2>&1; then espeak-ng -s 155 " + qText + "; "
               "elif command -v espeak >/dev/null 2>&1 && command -v aplay >/dev/null 2>&1; then "
               "espeak --stdout " + qText + " | aplay -q -D " + qDevice + "; "
               "elif command -v espeak >/dev/null 2>&1; then espeak " + qText + "; fi)";
#endif
    }

    void workerLoop() {
        while(true) {
            std::string key;
            std::string text;
            bool nonBeepMessage = false;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() { return stopping_ || !queue_.empty(); });
                if(stopping_ && queue_.empty()) {
                    return;
                }
                key = std::move(queue_.front().key);
                nonBeepMessage = !isBeepMessage(key);
                text = std::move(queue_.front().text);
                queue_.pop_front();
                if(nonBeepMessage) {
                    speakingNonBeep_ = true;
                }
            }

            const std::string cmd = isBeepMessage(key) ? buildBeepCommand() : buildCommand(text);
            if(cmd.empty()) {
                if(nonBeepMessage) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    speakingNonBeep_ = false;
                }
                continue;
            }
            const int rc = std::system(cmd.c_str());
            if(nonBeepMessage) {
                std::lock_guard<std::mutex> lock(mtx_);
                speakingNonBeep_ = false;
            }
            if(rc != 0 && !warnedFailure_) {
                warnedFailure_ = true;
                std::cerr << "[collection][voice] playback command failed once. key=" << key << " text=" << text << std::endl;
            }
        }
    }

    VoiceFeedbackConfig       cfg_;
    std::mutex                mtx_;
    std::condition_variable   cv_;
    std::deque<VoiceMessage>  queue_;
    std::thread               worker_;
    bool                      stopping_ = false;
    bool                      speakingNonBeep_ = false;
    bool                      warnedFailure_ = false;
};

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

static bool depthOutputIsFfv1Mkv(const SaveOptions &options) {
    const std::string mode = normalizePresetKey(options.depthEncoding);
    return mode == "ffv1mkv" || mode == "ffv1" || mode == "mkv";
}

static std::string depthStorageEncodingName(const SaveOptions &options) {
    return depthOutputIsFfv1Mkv(options) ? "ffv1_mkv" : "png";
}

static std::string rgbStorageEncodingName(const SaveOptions &options) {
    return options.rgbH265 ? "h265" : "image";
}

static std::string depthFfv1OutputFileName() {
    return "depth.mkv";
}

static const char *depthRawDirName() {
    return "depth_raw";
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

static OBFormat h265RawInputFormatForColorProfile(OBFormat profileFormat) {
    return profileFormat == OB_FORMAT_RGB ? OB_FORMAT_RGB : OB_FORMAT_BGR;
}

static const char *h265RawInputPixFmt(OBFormat inputFormat) {
    return inputFormat == OB_FORMAT_RGB ? "rgb24" : "bgr24";
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

static bool copyDetachedColorRawRgbOrBgr(const DetachedVideoFrame &frame, cv::Mat &out) {
    out.release();
    if(frame.width <= 0 || frame.height <= 0 || frame.data.empty()) {
        return false;
    }
    if(frame.format != OB_FORMAT_RGB && frame.format != OB_FORMAT_BGR) {
        return false;
    }
    const int width = frame.width;
    const int height = frame.height;
    const size_t dataSize = frame.data.size();
    const size_t stride = height > 0 ? (dataSize / static_cast<size_t>(height)) : 0;
    if(stride < static_cast<size_t>(width) * 3) {
        return false;
    }
    cv::Mat tmp(height, width, CV_8UC3, const_cast<uint8_t *>(frame.data.data()), stride);
    out = tmp.clone();
    if(out.empty()) {
        return false;
    }
    return true;
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

static size_t findNearestEgoFrameIndex(const std::deque<EgoFrame> &samples, uint64_t targetUs) {
    if(samples.empty()) {
        return 0;
    }
    auto absDiff = [](uint64_t a, uint64_t b) {
        return a > b ? (a - b) : (b - a);
    };
    auto it = std::lower_bound(samples.begin(), samples.end(), targetUs, [](const EgoFrame &sample, uint64_t tsUs) {
        return sample.refTimestampUs < tsUs;
    });

    size_t bestIndex = 0;
    uint64_t bestDiff = std::numeric_limits<uint64_t>::max();
    if(it != samples.end()) {
        bestIndex = static_cast<size_t>(std::distance(samples.begin(), it));
        bestDiff = absDiff(samples[bestIndex].refTimestampUs, targetUs);
    }
    if(it != samples.begin()) {
        const size_t cand = static_cast<size_t>(std::distance(samples.begin(), it - 1));
        const uint64_t diff = absDiff(samples[cand].refTimestampUs, targetUs);
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

static bool writeFisheyeCameraParamsJson(const fs::path &cameraDir, size_t cameraIdx, const cv::Mat &frame, int fps, const SaveOptions &saveOptions) {
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
    const std::string fileName = h265OutputFileName(saveOptions);
    jsonAddString(rgbObj, "storageEncoding", "h265");
    jsonAddString(rgbObj, "storageFile", fileName);
    jsonAddString(rgbObj, "timestampFile", fileName + ".timestamps.csv");
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

static void alignDepthToRgbInto(const cv::Mat &depth16,
                                float valueScaleMm,
                                const OBCameraParam &rgbDepthParam,
                                int rgbW, int rgbH,
                                cv::Mat &aligned,
                                cv::Mat &zBuf) {
    aligned.create(rgbH, rgbW, CV_16UC1);
    aligned.setTo(cv::Scalar(0));
    if(depth16.empty() || depth16.type() != CV_16UC1 || !(valueScaleMm > 0.0f)) {
        return;
    }
    if(rgbDepthParam.depthIntrinsic.fx <= 0.0f || rgbDepthParam.rgbIntrinsic.fx <= 0.0f) {
        return;
    }

    const int   dW   = depth16.cols;
    const int   dH   = depth16.rows;
    const float fx_d = rgbDepthParam.depthIntrinsic.fx;
    const float fy_d = rgbDepthParam.depthIntrinsic.fy;
    const float cx_d = rgbDepthParam.depthIntrinsic.cx;
    const float cy_d = rgbDepthParam.depthIntrinsic.cy;
    const float *R   = rgbDepthParam.transform.rot;
    const float *t   = rgbDepthParam.transform.trans;
    zBuf.create(rgbH, rgbW, CV_32FC1);
    zBuf.setTo(cv::Scalar(std::numeric_limits<float>::infinity()));

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

            float &bestZ = zBuf.ptr<float>(v_rgb)[u_rgb];
            if(Zr >= bestZ) {
                continue;
            }
            bestZ = Zr;
            aligned.ptr<uint16_t>(v_rgb)[u_rgb] = static_cast<uint16_t>(zRatio + 0.5f);
        }
    }
}

static cv::Mat alignDepthToRgb(const cv::Mat &depth16,
                                float valueScaleMm,
                                const OBCameraParam &rgbDepthParam,
                                int rgbW, int rgbH) {
    cv::Mat aligned;
    cv::Mat zBuf;
    alignDepthToRgbInto(depth16, valueScaleMm, rgbDepthParam, rgbW, rgbH, aligned, zBuf);

    return aligned;
}

class MultiDeviceStreamingRecorder {
public:
    struct CameraStreamFault {
        std::string     sn;
        std::string     camKey;
        std::string     displayName;
        CollectDataType type = CollectDataType::RGB;
        double          silentSeconds = 0.0;
        bool            everReceived = false;
        std::string     message;
    };

    struct CameraStreamReadiness {
        size_t readyStreams = 0;
        size_t totalStreams = 0;
        bool   allReady = true;
        std::string message;
    };

    struct CameraWarmupReadiness {
        size_t warmedCameras = 0;
        size_t totalCameras = 0;
        size_t warmedStreams = 0;
        size_t totalStreams = 0;
        bool   allReady = false;
        double remainingSeconds = 0.0;
        std::string message;
    };

    explicit MultiDeviceStreamingRecorder(AppConfig baseCfg, EgoRecorder *sharedEgoRecorder = nullptr)
        : cfg_(std::move(baseCfg)),
          egoRecorder_(sharedEgoRecorder ? *sharedEgoRecorder : ownedEgoRecorder_),
          ownsEgoRecorder_(sharedEgoRecorder == nullptr) {
        const size_t baseQueue = cfg_.queueCapacity > 0 ? static_cast<size_t>(cfg_.queueCapacity) : 1024;
        recordQueueMax_ = cfg_.recordQueueCapacity > 0 ? static_cast<size_t>(cfg_.recordQueueCapacity) : std::max<size_t>(16384, baseQueue * 16);
        coordQueueMax_ = cfg_.coordQueueCapacity > 0 ? static_cast<size_t>(cfg_.coordQueueCapacity) : std::max<size_t>(8192, baseQueue * 8);
        writeQueueMax_ = cfg_.writeQueueCapacity > 0 ? static_cast<size_t>(cfg_.writeQueueCapacity) : std::max<size_t>(16384, baseQueue * 16);
    }

    bool isCapturing() const { return capturing_.load(); }
    bool isRecording() const { return recording_.load(); }
    bool hasData() const { return hasData_.load(); }

    bool hasCurrentSession() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        return session_.active;
    }

    bool isDrainComplete() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return true;
        }
        return session_.coordinatorDone && queuedWriteCount_.load() == 0 && writeInFlight_.load() == 0
               && queuedDepthAlignCount_.load() == 0 && depthAlignInFlight_.load() == 0
               && !depthAlignActive_.load() && !h265EncodingActive_.load() && !depthFfv1EncodingActive_.load();
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

    int currentSessionFrameCount() const {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return 0;
        }
        const size_t actualFrames = session_.fullAligned > 0 ? session_.fullAligned : session_.nextFrameIndex;
        if(actualFrames > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(actualFrames);
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

    void stopTouchRuntimes() {
        for(auto &runtime: touchRuntimes_) {
            if(runtime.recordThread.joinable()) {
                runtime.recordThread.join();
            }
            if(runtime.recorder) {
                runtime.recorder->stop();
            }
        }
        touchRuntimes_.clear();
    }

    void reset() {
        joinCoordinatorThreadIfPossible();
        {
            std::lock_guard<std::mutex> lock(coordMtx_);
            closeSessionTimestampsLocked();
            session_ = SessionState{};
            coordRecordQueue_.clear();
            coordFisheyeQueue_.clear();
            coordEgoQueue_.clear();
            coordTouchQueue_.clear();
            coordCv_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for(auto &kv: buffers_) {
                kv.second.latestRgb.release();
                kv.second.latestRgbTsUs = 0;
                kv.second.latestRgbSteady = {};
                kv.second.latestDepthAlignedRgb.release();
                kv.second.latestDepthTsUs = 0;
                kv.second.latestDepthValueScaleMm = 0.0f;
                kv.second.latestDepthSteady = {};
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
        activeFisheyeCameraCount_ = 0;
        activeFisheyeCameraIds_.clear();
        egoEnabled_ = false;
        touchEnabled_ = false;
        stopTouchRuntimes();
        resetStreamHealth();
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
            if(!coordRecordQueue_.empty() || !coordFisheyeQueue_.empty() || !coordEgoQueue_.empty() || !coordTouchQueue_.empty()) {
                if(any) {
                    oss << " ";
                }
                oss << "align=" << coordRecordQueue_.size()
                    << "+" << coordFisheyeQueue_.size()
                    << "+" << coordEgoQueue_.size()
                    << "+" << coordTouchQueue_.size();
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
        if(queuedDepthAlignCount_.load() > 0 || depthAlignInFlight_.load() > 0) {
            if(any) {
                oss << " ";
            }
            oss << "depthAlign=" << queuedDepthAlignCount_.load() << "+" << depthAlignInFlight_.load();
            any = true;
        }
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            size_t queued = 0;
            int inFlight = 0;
            for(const auto &kv: h265Encoders_) {
                if(kv.second) {
                    queued += kv.second->queued();
                    inFlight += kv.second->inFlight();
                }
            }
            const size_t finalizing = h265StoppingEncoders_.load();
            if(h265EncodingActive_.load() || queued > 0 || inFlight > 0 || finalizing > 0) {
                if(any) {
                    oss << " ";
                }
                oss << "h265=" << queued << "+" << inFlight;
                if(finalizing > 0) {
                    oss << "/finalize=" << finalizing;
                }
                any = true;
            }
        }
        {
            std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
            size_t queued = 0;
            int inFlight = 0;
            for(const auto &kv: depthFfv1Encoders_) {
                if(kv.second) {
                    queued += kv.second->queued();
                    inFlight += kv.second->inFlight();
                }
            }
            const size_t finalizing = depthFfv1StoppingEncoders_.load();
            if(depthFfv1EncodingActive_.load() || queued > 0 || inFlight > 0 || finalizing > 0) {
                if(any) {
                    oss << " ";
                }
                oss << "mkv=" << queued << "+" << inFlight;
                if(finalizing > 0) {
                    oss << "/finalize=" << finalizing;
                }
                any = true;
            }
        }
        if(!any) {
            std::lock_guard<std::mutex> lock(coordMtx_);
            if(session_.active && !session_.coordinatorDone) {
                oss << "finalizing";
                any = true;
            }
            else if(depthAlignActive_.load()) {
                oss << "depthAlign=stopping";
                any = true;
            }
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
                noteFisheyeFrameSet(*snap);
                for(size_t i = 0; i < snap->frames.size(); ++i) {
                    if(!snap->frames[i].bgr.empty()) {
                        out[fisheyePreviewLabel(i)] = snap->frames[i].bgr;
                    }
                }
            }
        }
        return out;
    }

    CameraStreamReadiness cameraStreamReadiness() const {
        CameraStreamReadiness readiness;
        if(!capturing_.load()) {
            readiness.allReady = false;
            readiness.message = "Starting cameras...";
            return readiness;
        }
        const_cast<MultiDeviceStreamingRecorder *>(this)->refreshEgoReadyHealth();
        const_cast<MultiDeviceStreamingRecorder *>(this)->refreshTouchReadyHealth();

        std::vector<std::string> missing;
        {
            std::lock_guard<std::mutex> lock(streamHealthMtx_);
            readiness.totalStreams = streamHealth_.size();
            for(const auto &kv: streamHealth_) {
                const auto &state = kv.second;
                if(state.everReceived) {
                    readiness.readyStreams++;
                }
                else if(missing.size() < 3) {
                    missing.push_back(streamHealthDisplayName(state));
                }
            }
        }

        readiness.allReady = readiness.totalStreams == 0 || readiness.readyStreams == readiness.totalStreams;
        std::ostringstream oss;
        if(readiness.allReady) {
            oss << "Cameras ready";
        }
        else {
            oss << "Warming up cameras: " << readiness.readyStreams << "/" << readiness.totalStreams << " streams ready";
            if(!missing.empty()) {
                oss << "  waiting for ";
                for(size_t i = 0; i < missing.size(); ++i) {
                    if(i > 0) {
                        oss << ", ";
                    }
                    oss << missing[i];
                }
            }
        }
        readiness.message = oss.str();
        return readiness;
    }

    CameraWarmupReadiness cameraWarmupReadiness(double requiredSeconds) const {
        CameraWarmupReadiness readiness;
        requiredSeconds = std::max(0.0, requiredSeconds);

        std::vector<std::pair<std::string, std::string>> cameras;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            cameras.reserve(buffers_.size());
            for(const auto &kv: buffers_) {
                cameras.emplace_back(kv.first, kv.second.camKey);
            }
        }
        readiness.totalCameras = cameras.size();

        std::vector<std::string> waiting;
        {
            std::lock_guard<std::mutex> lock(streamHealthMtx_);
            for(const auto &camera: cameras) {
                bool cameraReady = true;
                for(const auto type: { CollectDataType::RGB, CollectDataType::Depth }) {
                    readiness.totalStreams++;
                    const auto it = streamHealth_.find(streamHealthKey(camera.first, type));
                    double receivedSeconds = 0.0;
                    if(it != streamHealth_.end()
                       && it->second.firstFrameSteady.time_since_epoch().count() != 0
                       && it->second.lastFrameSteady.time_since_epoch().count() != 0) {
                        receivedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(
                            it->second.lastFrameSteady - it->second.firstFrameSteady).count();
                    }
                    if(receivedSeconds >= requiredSeconds) {
                        readiness.warmedStreams++;
                    }
                    else {
                        cameraReady = false;
                        readiness.remainingSeconds = std::max(readiness.remainingSeconds,
                                                              requiredSeconds - receivedSeconds);
                        if(waiting.size() < 3) {
                            waiting.push_back("cam" + camera.second + " " + dataTypeLabel(type));
                        }
                    }
                }
                if(cameraReady) {
                    readiness.warmedCameras++;
                }
            }
        }

        readiness.allReady = readiness.totalCameras > 0
                          && readiness.warmedCameras == readiness.totalCameras;
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        if(readiness.allReady) {
            oss << "Initial RGB/depth warm-up complete: " << readiness.totalCameras
                << " cameras received both streams for " << std::setprecision(1) << requiredSeconds << " s";
        }
        else {
            oss << "Initial RGB/depth warm-up: " << readiness.warmedCameras << "/" << readiness.totalCameras
                << " cameras (" << readiness.warmedStreams << "/" << readiness.totalStreams << " streams), "
                << std::setprecision(1) << std::max(0.0, readiness.remainingSeconds) << " s remaining";
            if(!waiting.empty()) {
                oss << "  waiting for ";
                for(size_t i = 0; i < waiting.size(); ++i) {
                    if(i > 0) {
                        oss << ", ";
                    }
                    oss << waiting[i];
                }
            }
        }
        readiness.message = oss.str();
        return readiness;
    }

    std::optional<CameraStreamFault> pollCameraStreamFault() {
        if(!capturing_.load() || !recording_.load()) {
            return std::nullopt;
        }

        std::optional<CameraStreamFault> faultToLog;
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(streamHealthMtx_);
            if(activeCameraFault_.has_value()) {
                return activeCameraFault_;
            }

            for(auto &kv: streamHealth_) {
                auto &state = kv.second;
                const auto base = state.everReceived ? state.lastFrameSteady : state.startedSteady;
                if(base.time_since_epoch().count() == 0) {
                    continue;
                }

                const double silentSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - base).count();
                if(silentSeconds < cameraStreamTimeoutSec()) {
                    continue;
                }

                state.faulted = true;
                CameraStreamFault fault;
                fault.sn = state.sn;
                fault.camKey = state.camKey;
                fault.displayName = streamHealthDisplayName(state);
                fault.type = state.type;
                fault.silentSeconds = silentSeconds;
                fault.everReceived = state.everReceived;
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss << "Camera stream timeout: " << fault.displayName
                    << " no frame for " << std::setprecision(2) << silentSeconds << "s";
                if(!state.everReceived) {
                    oss << " since stream start";
                }
                fault.message = oss.str();
                activeCameraFault_ = fault;
                faultToLog = fault;
                break;
            }
        }

        if(faultToLog.has_value()) {
            std::cerr << "[collection][camera_fault] " << faultToLog->message << std::endl;
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = faultToLog->message;
        }
        return faultToLog;
    }

    bool runExtrinsicHealthCheckBeforeStart(const fs::path &saveRoot,
                                            const std::string &subjectId,
                                            const std::string &taskName,
                                            int episodeN,
                                            std::string *statusLine = nullptr,
                                            std::vector<std::string> *detailLines = nullptr,
                                            std::string *resultStatusOut = nullptr) {
        if(resultStatusOut) {
            resultStatusOut->clear();
        }
        const auto &health = cfg_.extrinsicHealth;
        if(!health.enabled) {
            if(statusLine) {
                *statusLine = "Extrinsic check disabled";
            }
            if(resultStatusOut) {
                *resultStatusOut = "disabled";
            }
            return true;
        }
        if(!multiviewEnabled_) {
            if(statusLine) {
                *statusLine = "Extrinsic check skipped: multiview disabled";
            }
            if(resultStatusOut) {
                *resultStatusOut = "skipped";
            }
            return true;
        }
        if(!capturing_.load()) {
            if(statusLine) {
                *statusLine = "Extrinsic check failed: cameras are not running";
            }
            if(resultStatusOut) {
                *resultStatusOut = "error";
            }
            return false;
        }
        if(cfg_.initExtrinsicPath.empty()) {
            if(statusLine) {
                *statusLine = "Extrinsic check failed: init_extrinsic_path is empty";
            }
            if(resultStatusOut) {
                *resultStatusOut = "error";
            }
            return false;
        }
        if(health.scriptPath.empty() || !fs::exists(health.scriptPath)) {
            if(statusLine) {
                *statusLine = "Extrinsic check failed: script not found at " + health.scriptPath.string();
            }
            if(resultStatusOut) {
                *resultStatusOut = "error";
            }
            return false;
        }

        const auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
        const auto nowMs = now.time_since_epoch().count();
        const fs::path debugRoot = saveRoot / subjectId / ".extrinsic_health";
        const fs::path checkDir = debugRoot / (sanitizePathComponent(taskName)
                                                + "_episode_" + std::to_string(episodeN)
                                                + "_" + std::to_string(nowMs));
        try {
            fs::create_directories(checkDir);
        }
        catch(const std::exception &ex) {
            if(statusLine) {
                *statusLine = "Extrinsic check failed: cannot create debug directory: " + std::string(ex.what());
            }
            if(resultStatusOut) {
                *resultStatusOut = "error";
            }
            return false;
        }

        std::unordered_map<std::string, DeviceBuffer> paramsBuffers;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for(const auto &kv: buffers_) {
                const auto itRgb = kv.second.params.find(CollectDataType::RGB);
                const auto itDepth = kv.second.params.find(CollectDataType::Depth);
                if(itRgb == kv.second.params.end() || !itRgb->second.valid
                   || itDepth == kv.second.params.end() || !itDepth->second.valid
                   || !kv.second.rgbDepthParamValid) {
                    continue;
                }
                DeviceBuffer buf;
                buf.camKey = kv.second.camKey;
                buf.params[CollectDataType::RGB] = itRgb->second;
                buf.params[CollectDataType::Depth] = itDepth->second;
                buf.rgbDepthParam = kv.second.rgbDepthParam;
                buf.rgbDepthParamValid = kv.second.rgbDepthParamValid;
                paramsBuffers.emplace(kv.first, std::move(buf));
            }
        }
        if(paramsBuffers.size() < static_cast<size_t>(std::max(2, health.minCheckedCameras))) {
            if(!health.keepDebugSnapshots) {
                try { fs::remove_all(checkDir); } catch(...) {}
            }
            if(statusLine) {
                *statusLine = "Extrinsic check inconclusive: not enough RGB/depth camera parameters";
            }
            if(resultStatusOut) {
                *resultStatusOut = "inconclusive";
            }
            return true;
        }
        writeParamsJson(checkDir, paramsBuffers, { CollectDataType::RGB, CollectDataType::Depth }, cfg_.colorCloudRgbFrameOffset, cfg_.save);
        writeExtrinsicsJson(checkDir);
        writeExtrinsicHealthConfigJson(checkDir / "health_config.json");

        cJSON *manifest = cJSON_CreateObject();
        cJSON_AddStringToObject(manifest, "camera_params_json", "camera_params.json");
        cJSON_AddStringToObject(manifest, "extrinsics_json", "extrinsics.json");
        cJSON *samples = cJSON_CreateArray();
        const int sampleCount = std::max(1, health.sampleCount);
        for(int sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
            if(sampleIdx > 0 && health.sampleIntervalMs > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(health.sampleIntervalMs));
            }
            auto frames = latestExtrinsicHealthFrames();
            const std::string sampleName = "sample_" + formatFrameIndex(static_cast<size_t>(sampleIdx));
            const fs::path sampleDir = checkDir / sampleName;
            try {
                fs::create_directories(sampleDir);
            }
            catch(...) {
            }

            cJSON *sampleObj = cJSON_CreateObject();
            cJSON_AddNumberToObject(sampleObj, "index", sampleIdx);
            cJSON *cameras = cJSON_CreateArray();
            for(const auto &frame: frames) {
                const double maxAgeMs = std::max(frame.ageMs, frame.depthAgeMs);
                if(frame.bgr.empty() || frame.depthAlignedRgb16.empty()
                   || maxAgeMs > static_cast<double>(health.maxSnapshotAgeMs)) {
                    continue;
                }
                const std::string fileName = frame.camKey + ".jpg";
                const std::string depthFileName = frame.camKey + ".depth.png";
                const fs::path outPath = sampleDir / fileName;
                const fs::path depthOutPath = sampleDir / depthFileName;
                std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, std::max(1, std::min(100, health.jpegQuality)) };
                if(!cv::imwrite(outPath.string(), frame.bgr, params)) {
                    continue;
                }
                if(!cv::imwrite(depthOutPath.string(), frame.depthAlignedRgb16)) {
                    continue;
                }
                cJSON *camObj = cJSON_CreateObject();
                cJSON_AddStringToObject(camObj, "id", frame.camKey.c_str());
                cJSON_AddStringToObject(camObj, "sn", frame.sn.c_str());
                cJSON_AddStringToObject(camObj, "image", (sampleName + "/" + fileName).c_str());
                cJSON_AddStringToObject(camObj, "depth", (sampleName + "/" + depthFileName).c_str());
                cJSON_AddBoolToObject(camObj, "depth_aligned_to_rgb", true);
                cJSON_AddNumberToObject(camObj, "timestamp_us", static_cast<double>(frame.tsUs));
                cJSON_AddNumberToObject(camObj, "depth_timestamp_us", static_cast<double>(frame.depthTsUs));
                cJSON_AddNumberToObject(camObj, "age_ms", frame.ageMs);
                cJSON_AddNumberToObject(camObj, "depth_age_ms", frame.depthAgeMs);
                cJSON_AddNumberToObject(camObj, "depth_value_scale_mm", frame.depthValueScaleMm);
                cJSON_AddItemToArray(cameras, camObj);
            }
            cJSON_AddItemToObject(sampleObj, "cameras", cameras);
            cJSON_AddItemToArray(samples, sampleObj);
        }
        cJSON_AddItemToObject(manifest, "samples", samples);
        char *manifestText = cJSON_Print(manifest);
        if(manifestText) {
            writeTextFile(checkDir / "manifest.json", manifestText);
            cJSON_free(manifestText);
        }
        cJSON_Delete(manifest);

        const fs::path resultJson = checkDir / "result.json";
        const std::string command = shellQuote(health.pythonExecutable)
                                  + " " + shellQuote(health.scriptPath.string())
                                  + " --snapshot-dir " + shellQuote(checkDir.string())
                                  + " --config-json " + shellQuote((checkDir / "health_config.json").string())
                                  + " --result-json " + shellQuote(resultJson.string())
                                  + " 2>&1";
        std::string output;
        const int exitCode = runCommandCapture(command, output);
        std::string resultStatus;
        std::string resultSummary;
        std::vector<std::string> resultDetails;
        readExtrinsicHealthResult(resultJson, resultStatus, resultSummary, &resultDetails);
        if(resultStatusOut) {
            *resultStatusOut = resultStatus.empty() ? "error" : resultStatus;
        }
        if(resultSummary.empty()) {
            resultSummary = trimString(output);
        }
        if(resultSummary.empty()) {
            resultSummary = "exit_code=" + std::to_string(exitCode);
        }

        // A health-check result blocks collection only when it explicitly fails.
        // Operational errors still block because no valid health result was produced.
        const bool ok = resultStatus == "pass"
                     || resultStatus == "warn"
                     || resultStatus == "inconclusive";

        if(statusLine) {
            *statusLine = "Extrinsic check " + resultSummary;
            if(health.keepDebugSnapshots || !ok) {
                *statusLine += " debug=" + checkDir.string();
            }
        }
        if(detailLines) {
            detailLines->insert(detailLines->end(), resultDetails.begin(), resultDetails.end());
        }
        std::cerr << "[collection][extrinsic_check] status=" << (resultStatus.empty() ? "(missing)" : resultStatus)
                  << " ok=" << (ok ? 1 : 0)
                  << " exit=" << exitCode
                  << " dir=" << checkDir
                  << " summary=" << resultSummary << std::endl;

        if(ok && !health.keepDebugSnapshots) {
            try {
                fs::remove_all(checkDir);
            }
            catch(...) {
            }
        }
        return ok;
    }

    void clearCameraStreamFault() {
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        activeCameraFault_.reset();
        for(auto &kv: streamHealth_) {
            kv.second.faulted = false;
        }
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
        const bool egoWasRunning = egoRecorder_.isRunning();
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
        egoEnabled_       = ui.enableEgo;
        touchEnabled_     = ui.enableTouch;
        activeFisheyeCameraCount_ = 0;
        activeFisheyeCameraIds_.clear();

        if(!multiviewEnabled_ && !fisheyeEnabled_ && !egoEnabled_) {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = touchEnabled_
                ? "Touch is an auxiliary modality; select multiview, fisheyes, or ego too"
                : "Select at least one capture type";
            return false;
        }

        const int w = ui.widthInt();
        const int h = ui.heightInt();
        const int f = ui.fpsInt();
        imuEnabled_ = false;
        typesStreaming_ = multiviewEnabled_ ? ui.enabledTypesForStreaming() : std::vector<CollectDataType>{};
        typesSaving_    = multiviewEnabled_ ? ui.enabledTypesForSaving() : std::vector<CollectDataType>{};
        refType_        = ui.referenceType();
        uiFpsFallback_  = f;

        std::string fisheyeStatusLine;
        if(fisheyeEnabled_) {
            const auto fisheyeDevices = listPreferredFisheyeDevices(preferredFisheyeCameraLabels());
            if(fisheyeDevices.empty()) {
                if(multiviewEnabled_ || egoEnabled_) {
                    fisheyeEnabled_ = false;
                    fisheyeStatusLine = "No fisheye detected, continuing with other selected sources";
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
                    if(multiviewEnabled_ || egoEnabled_) {
                        fisheyeEnabled_ = false;
                        activeFisheyeCameraCount_ = 0;
                        activeFisheyeCameraIds_.clear();
                        fisheyeStatusLine = "Fisheye unavailable, continuing with other selected sources: " + fisheyeError;
                        std::cerr << "[collection] " << fisheyeStatusLine << std::endl;
                    }
                    else {
                        std::lock_guard<std::mutex> lock(mtx_);
                        captureInfoLine_ = "Fisheye start failed: " + fisheyeError;
                        return false;
                    }
                }
                else {
                    activeFisheyeCameraCount_ = fisheyeCfg.cameras.size();
                    activeFisheyeCameraIds_.clear();
                    activeFisheyeCameraIds_.reserve(fisheyeCfg.cameras.size());
                    for(size_t i = 0; i < fisheyeCfg.cameras.size(); ++i) {
                        const auto &camera = fisheyeCfg.cameras[i];
                        activeFisheyeCameraIds_.push_back(camera.cameraId.empty() ? ("cam" + std::to_string(i)) : camera.cameraId);
                    }
                }
            }
        }

        std::string egoStatusLine;
        if(egoEnabled_) {
            EgoModuleConfig egoCfg = cfg_.ego;
            egoCfg.enabled = true;
            if(egoCfg.maxBufferedFrames == 0) {
                egoCfg.maxBufferedFrames = static_cast<size_t>(std::max(2048, ui.maxDurationInt() * std::max(1, ui.fpsInt()) * 2));
            }
            std::string egoError;
            if(ownsEgoRecorder_) {
                if(!egoRecorder_.start(egoCfg, &egoError)) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    captureInfoLine_ = "Ego server start failed: " + egoError;
                    return false;
                }
                egoStatusLine = "Ego server listening on " + egoCfg.host + ":" + std::to_string(egoCfg.port);
                std::cerr << "[collection] " << egoStatusLine << std::endl;
            }
            else {
                if(!egoRecorder_.isRunning()) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    captureInfoLine_ = "Ego server is not ready from menu";
                    return false;
                }
                egoStatusLine = egoRecorder_.isConnected()
                                  ? "Ego client connected"
                                  : ("Ego server listening on " + egoCfg.host + ":" + std::to_string(egoCfg.port) + ", waiting for PICO");
                std::cerr << "[collection] using shared " << egoStatusLine << std::endl;
            }
        }

        std::string touchStatusLine;
        if(touchEnabled_) {
            const auto touchCfgs = expandTouchDeviceConfigs(cfg_.touch);
            if(touchCfgs.empty()) {
                touchEnabled_ = false;
                if(cfg_.touch.required) {
                    fisheyeRecorder_.stop();
                    if(!egoWasRunning && ownsEgoRecorder_) {
                        egoRecorder_.stop();
                    }
                    std::lock_guard<std::mutex> lock(mtx_);
                    captureInfoLine_ = "Touch start failed: no touch devices configured";
                    return false;
                }
                touchStatusLine = "Touch unavailable, continuing without touch: no touch devices configured";
            }
            else {
                for(const auto &baseTouchCfg: touchCfgs) {
                    TactileModuleConfig touchCfg = baseTouchCfg;
                    touchCfg.enabled = true;
                    if(touchCfg.maxBufferedSamples == 0) {
                        touchCfg.maxBufferedSamples = static_cast<size_t>(std::max(2048, ui.maxDurationInt() * std::max(1, touchCfg.targetFps) * 2));
                    }

                    TouchRuntime runtime;
                    runtime.streamId = touchCfg.streamId;
                    runtime.handSide = touchCfg.handSide;
                    runtime.sensorType = touchCfg.sensorType;
                    runtime.config = touchCfg;
                    runtime.recorder = std::make_unique<TactileRecorder>();

                    std::string touchError;
                    if(!runtime.recorder->start(touchCfg, &touchError)) {
                        if(cfg_.touch.required) {
                            stopTouchRuntimes();
                            touchEnabled_ = false;
                            fisheyeRecorder_.stop();
                            if(!egoWasRunning && ownsEgoRecorder_) {
                                egoRecorder_.stop();
                            }
                            std::lock_guard<std::mutex> lock(mtx_);
                            captureInfoLine_ = "Touch " + runtime.streamId + " start failed: " + touchError;
                            return false;
                        }
                        std::cerr << "[collection] Touch " << runtime.streamId
                                  << " unavailable, continuing without it: " << touchError << std::endl;
                        continue;
                    }
                    const std::string port = touchCfg.serial.portPath.empty() ? "(auto)" : touchCfg.serial.portPath;
                    if(!touchStatusLine.empty()) {
                        touchStatusLine += "  ";
                    }
                    touchStatusLine += "Touch " + runtime.streamId + " listening on " + port
                                     + " @" + std::to_string(touchCfg.serial.baudRate);
                    touchRuntimes_.push_back(std::move(runtime));
                }
                if(touchRuntimes_.empty()) {
                    touchEnabled_ = false;
                    if(cfg_.touch.required) {
                        fisheyeRecorder_.stop();
                        if(!egoWasRunning && ownsEgoRecorder_) {
                            egoRecorder_.stop();
                        }
                        std::lock_guard<std::mutex> lock(mtx_);
                        captureInfoLine_ = "Touch start failed: no touch recorder is running";
                        return false;
                    }
                    touchStatusLine = "Touch unavailable, continuing without touch";
                }
            }
            if(!touchStatusLine.empty()) {
                std::cerr << "[collection] " << touchStatusLine << std::endl;
            }
        }

        if(!multiviewEnabled_) {
            stopping_.store(false);
            capturing_.store(true);
            recording_.store(false);
            initStreamHealthForActiveStreams();
            softwareTriggerDevices_.clear();
            useSoftwareTrigger_ = false;
            if(!fisheyeStatusLine.empty() || !egoStatusLine.empty() || !touchStatusLine.empty()) {
                std::lock_guard<std::mutex> lock(mtx_);
                captureInfoLine_ = !fisheyeStatusLine.empty() ? fisheyeStatusLine
                                   : (!egoStatusLine.empty() ? egoStatusLine : touchStatusLine);
            }
            return true;
        }

        collectionSetStage("start_queryDeviceList");
        auto deviceList = ctx_.queryDeviceList();
        if(!deviceList || deviceList->deviceCount() == 0) {
            fisheyeRecorder_.stop();
            stopTouchRuntimes();
            if(!egoWasRunning && ownsEgoRecorder_) {
                egoRecorder_.stop();
            }
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "No device connected";
            std::cerr << "[collection] no device connected" << std::endl;
            return false;
        }

        collectionSetStage("start_selectDevices");
        auto selected = selectDevicesWithPipeline(deviceList, cfg_);
        if(selected.empty()) {
            fisheyeRecorder_.stop();
            stopTouchRuntimes();
            if(!egoWasRunning && ownsEgoRecorder_) {
                egoRecorder_.stop();
            }
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
                const OBFormat desiredFormat = (t == CollectDataType::RGB)
                    ? OB_FORMAT_UNKNOWN
                    : configuredStreamFormat(rt.cfg, t);
                auto profile = pickVideoProfile(rt.pipe, sensor, w, h, f, desiredFormat);
                if(profile) {
                    config->enableStream(profile);
                    enabledSensors.insert(sensor);
                    StreamParams sp;
                    sp.width  = static_cast<int>(profile->getWidth());
                    sp.height = static_cast<int>(profile->getHeight());
                    sp.fps    = static_cast<int>(profile->getFps());
                    sp.format = profile->getFormat();
                    std::cerr << "[collection] selected stream sn=" << rt.cfg.sn
                              << " type=" << dataTypeLabel(t)
                              << " size=" << sp.width << "x" << sp.height
                              << " fps=" << sp.fps
                              << " format=" << static_cast<int>(sp.format);
                    if(desiredFormat != OB_FORMAT_UNKNOWN) {
                        std::cerr << " requestedFormat=" << static_cast<int>(desiredFormat);
                    }
                    std::cerr << std::endl;
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
        initStreamHealthForActiveStreams();
        if(!fisheyeStatusLine.empty() || !egoStatusLine.empty() || !touchStatusLine.empty()) {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = !fisheyeStatusLine.empty() ? fisheyeStatusLine
                               : (!egoStatusLine.empty() ? egoStatusLine : touchStatusLine);
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
        local.saveEgo = egoEnabled_;
        local.saveTouch = touchEnabled_ && !touchRuntimes_.empty();
        if(local.saveTouch) {
            local.touchStreams.reserve(touchRuntimes_.size());
            for(const auto &runtime: touchRuntimes_) {
                TouchSessionStream stream;
                stream.streamId = runtime.streamId;
                stream.handSide = runtime.handSide;
                stream.sensorType = runtime.sensorType;
                local.touchSamples.emplace(stream.streamId, std::deque<TactileSample>{});
                local.touchStreams.push_back(std::move(stream));
            }
        }
        local.egoSoftAlignEnabled = cfg_.ego.softAlignToOrbbecFirstFrame
                                     && local.saveEgo && multiviewEnabled_ && !local.refSn.empty();
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
                if(cfg_.save.saveRaw && local.saveDepthTimesteps) {
                    fs::create_directories(local.dest / camKey / depthRawDirName());
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
            if(local.saveEgo) {
                fs::create_directories(local.dest / "ego" / "RGB");
            }
            if(local.saveTouch) {
                const std::string touchDirName = cfg_.touch.save.directoryName.empty() ? std::string("touch") : cfg_.touch.save.directoryName;
                fs::create_directories(local.dest / touchDirName);
                writeTouchManifestJson(local.dest / touchDirName);
            }

            if(multiviewEnabled_) {
                writeParamsJson(local.dest, local.buffers, typesSaving_, cfg_.colorCloudRgbFrameOffset, cfg_.save);
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
            if(local.egoAlignedTimestampsOfs.is_open()) {
                local.egoAlignedTimestampsOfs.close();
            }
            for(auto &stream: local.touchStreams) {
                if(stream.rawOfs.is_open()) {
                    stream.rawOfs.close();
                }
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
        std::string egoSessionName;
        if(local.saveEgo) {
            if(!egoRecorder_.isRunning() || !egoRecorder_.isConnected()) {
                if(local.timestampsOfs.is_open()) {
                    local.timestampsOfs.close();
                }
                if(local.egoAlignedTimestampsOfs.is_open()) {
                    local.egoAlignedTimestampsOfs.close();
                }
                for(auto &stream: local.touchStreams) {
                    if(stream.rawOfs.is_open()) {
                        stream.rawOfs.close();
                    }
                }
                try {
                    if(!local.timestampsTmpPath.empty() && fs::exists(local.timestampsTmpPath)) {
                        fs::remove(local.timestampsTmpPath);
                    }
                    if(!local.egoAlignedTimestampsTmpPath.empty() && fs::exists(local.egoAlignedTimestampsTmpPath)) {
                        fs::remove(local.egoAlignedTimestampsTmpPath);
                    }
                    for(const auto &stream: local.touchStreams) {
                        if(!stream.rawTmpPath.empty() && fs::exists(stream.rawTmpPath)) {
                            fs::remove(stream.rawTmpPath);
                        }
                    }
                    fs::remove_all(local.dest);
                }
                catch(...) {
                }
                std::lock_guard<std::mutex> lock(mtx_);
                captureInfoLine_ = "Ego client is not connected";
                return false;
            }
            egoSessionName = subjectId + "_" + taskName + "_episode_" + std::to_string(episodeN);
        }
        if(fisheyeEnabled_ && !fisheyeRecorder_.isRunning()) {
            if(local.timestampsOfs.is_open()) {
                local.timestampsOfs.close();
            }
            if(local.egoAlignedTimestampsOfs.is_open()) {
                local.egoAlignedTimestampsOfs.close();
            }
            for(auto &stream: local.touchStreams) {
                if(stream.rawOfs.is_open()) {
                    stream.rawOfs.close();
                }
            }
            try {
                if(!local.timestampsTmpPath.empty() && fs::exists(local.timestampsTmpPath)) {
                    fs::remove(local.timestampsTmpPath);
                }
                if(!local.egoAlignedTimestampsTmpPath.empty() && fs::exists(local.egoAlignedTimestampsTmpPath)) {
                    fs::remove(local.egoAlignedTimestampsTmpPath);
                }
                for(const auto &stream: local.touchStreams) {
                    if(!stream.rawTmpPath.empty() && fs::exists(stream.rawTmpPath)) {
                        fs::remove(stream.rawTmpPath);
                    }
                }
                fs::remove_all(local.dest);
            }
            catch(...) {
            }
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = "Fisheye recorder is not running";
            return false;
        }
        auto failedTouchRuntime = std::find_if(touchRuntimes_.begin(), touchRuntimes_.end(), [](const TouchRuntime &runtime) {
            return !runtime.recorder || !runtime.recorder->isRunning();
        });
        if(touchEnabled_ && (touchRuntimes_.empty() || failedTouchRuntime != touchRuntimes_.end())) {
            if(local.timestampsOfs.is_open()) {
                local.timestampsOfs.close();
            }
            if(local.egoAlignedTimestampsOfs.is_open()) {
                local.egoAlignedTimestampsOfs.close();
            }
            for(auto &stream: local.touchStreams) {
                if(stream.rawOfs.is_open()) {
                    stream.rawOfs.close();
                }
            }
            try {
                if(!local.timestampsTmpPath.empty() && fs::exists(local.timestampsTmpPath)) {
                    fs::remove(local.timestampsTmpPath);
                }
                if(!local.egoAlignedTimestampsTmpPath.empty() && fs::exists(local.egoAlignedTimestampsTmpPath)) {
                    fs::remove(local.egoAlignedTimestampsTmpPath);
                }
                for(const auto &stream: local.touchStreams) {
                    if(!stream.rawTmpPath.empty() && fs::exists(stream.rawTmpPath)) {
                        fs::remove(stream.rawTmpPath);
                    }
                }
                fs::remove_all(local.dest);
            }
            catch(...) {
            }
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = failedTouchRuntime == touchRuntimes_.end()
                ? "Touch recorder is not running"
                : ("Touch " + failedTouchRuntime->streamId + " recorder is not running");
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
            coordEgoQueue_.clear();
            coordTouchQueue_.clear();
            if(!multiviewEnabled_) {
                session_.multiviewEos = true;
            }
            if(!fisheyeEnabled_) {
                session_.fisheyeEos = true;
            }
            if(!egoEnabled_) {
                session_.egoEos = true;
            }
            if(!touchEnabled_) {
                session_.touchEos = true;
            }
            coordCv_.notify_all();
        }
        passthroughRgbMjpg_.store(session_.passthroughRgbMjpg);
        startH265Encoders(session_);
        startDepthFfv1Encoders(session_);
        startDepthAlignWorkers(session_);

        recordInputClosing_.store(false);
        multiviewEosNotified_.store(!multiviewEnabled_);
        hasData_.store(false);

        if(multiviewEnabled_) {
            clearRecordQueue();
            ensureRecordWorkerRunning();
        }
        ensureWriteWorkersRunning();
        startCoordinatorThread();

        if(session_.saveEgo) {
            std::string egoError;
            if(!egoRecorder_.beginSession(session_.dest, egoSessionName, &egoError)) {
                stopDepthAlignWorkers();
                stopH265Encoders();
                stopDepthFfv1Encoders();
                {
                    std::lock_guard<std::mutex> lock(coordMtx_);
                    closeSessionTimestampsLocked();
                    session_ = SessionState{};
                    coordRecordQueue_.clear();
                    coordFisheyeQueue_.clear();
                    coordEgoQueue_.clear();
                    coordTouchQueue_.clear();
                    coordCv_.notify_all();
                }
                joinCoordinatorThreadIfPossible();
                try {
                    if(!egoSessionName.empty()) {
                        fs::remove_all(saveRoot / subjectId / taskName / ("episode_" + std::to_string(episodeN)));
                    }
                }
                catch(...) {
                }
                recordInputClosing_.store(false);
                multiviewEosNotified_.store(false);
                hasData_.store(false);
                std::lock_guard<std::mutex> lock(mtx_);
                captureInfoLine_ = "Ego session start failed: " + egoError;
                return false;
            }
        }

        captureStartSteady_ = std::chrono::steady_clock::now();
        markStreamHealthCaptureStarted(captureStartSteady_);
        recording_.store(true);

        if(fisheyeEnabled_) {
            if(fisheyeRecordThread_.joinable()) {
                fisheyeRecordThread_.join();
            }
            fisheyeRecordThread_ = std::thread([this]() { fisheyeRecordLoop(); });
        }
        if(egoEnabled_) {
            if(egoRecordThread_.joinable()) {
                egoRecordThread_.join();
            }
            egoRecordThread_ = std::thread([this]() { egoRecordLoop(); });
        }
        if(touchEnabled_) {
            for(auto &runtime: touchRuntimes_) {
                if(runtime.recordThread.joinable()) {
                    runtime.recordThread.join();
                }
                if(runtime.recorder) {
                    runtime.recorder->resetCaptureCursorToLatest();
                }
                runtime.recordThread = std::thread([this, streamId = runtime.streamId]() {
                    touchRecordLoop(streamId);
                });
            }
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
        if(egoEnabled_) {
            std::string egoError;
            if(!egoRecorder_.stopSessionAndWait(std::chrono::milliseconds(std::max(100, cfg_.ego.stopTimeoutMs)), &egoError)
               && !egoError.empty()) {
                std::cerr << "[collection] ego stop warning: " << egoError << std::endl;
            }
        }
        if(egoRecordThread_.joinable()) {
            egoRecordThread_.join();
        }
        for(auto &runtime: touchRuntimes_) {
            if(runtime.recordThread.joinable()) {
                runtime.recordThread.join();
            }
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
        notifyEgoEos();
        notifyTouchEos();
        const std::string captureInfoSnapshot = buildCaptureInfoSnapshotLocked(durMs);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            captureInfoLine_ = captureInfoSnapshot;
        }
        std::cerr << "[collection] record stop" << std::endl;
    }

    void stopIfRunning(bool closeEgoTcp = true) {
        if(!capturing_.load()) {
            fisheyeRecorder_.stop();
            stopTouchRuntimes();
            if(closeEgoTcp) {
                if(ownsEgoRecorder_) {
                    egoRecorder_.stop();
                }
            }
            activeFisheyeCameraCount_ = 0;
            activeFisheyeCameraIds_.clear();
            egoEnabled_ = false;
            touchEnabled_ = false;
            return;
        }
        stopping_.store(true);
        recordCv_.notify_all();
        coordCv_.notify_all();
        depthAlignCv_.notify_all();
        recording_.store(false);
        recordInputClosing_.store(true);
        if(fisheyeRecordThread_.joinable()) {
            fisheyeRecordThread_.join();
        }
        if(egoEnabled_) {
            std::string egoError;
            if(!egoRecorder_.stopSessionAndWait(std::chrono::milliseconds(std::max(100, cfg_.ego.stopTimeoutMs)), &egoError)
               && !egoError.empty()) {
                std::cerr << "[collection] ego stop warning: " << egoError << std::endl;
            }
        }
        if(egoRecordThread_.joinable()) {
            egoRecordThread_.join();
        }
        for(auto &runtime: touchRuntimes_) {
            if(runtime.recordThread.joinable()) {
                runtime.recordThread.join();
            }
        }
        notifyFisheyeEos();
        notifyEgoEos();
        notifyTouchEos();
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
        stopTouchRuntimes();
        if(closeEgoTcp) {
            if(ownsEgoRecorder_) {
                egoRecorder_.stop();
            }
        }
        capturing_.store(false);
        recording_.store(false);
        hasData_.store(false);
        passthroughRgbMjpg_.store(false);
        activeFisheyeCameraCount_ = 0;
        activeFisheyeCameraIds_.clear();
        egoEnabled_ = false;
        touchEnabled_ = false;
        resetStreamHealth();
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
            coordEgoQueue_.clear();
            coordTouchQueue_.clear();
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
            coordEgoQueue_.clear();
            coordTouchQueue_.clear();
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
            coordEgoQueue_.clear();
            coordTouchQueue_.clear();
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
    struct StreamHealthState {
        std::string healthKey;
        std::string sn;
        std::string camKey;
        std::string displayName;
        CollectDataType type = CollectDataType::RGB;
        std::chrono::steady_clock::time_point startedSteady{};
        std::chrono::steady_clock::time_point firstFrameSteady{};
        std::chrono::steady_clock::time_point lastFrameSteady{};
        uint64_t lastFrameTimestampUs = 0;
        bool requireTimestampAdvance = false;
        bool everReceived = false;
        bool faulted = false;
    };

    struct RecordTask {
        std::string       sn;
        CollectDataType   type = CollectDataType::RGB;
        uint64_t          tsUs = 0;
        uint64_t          seq = 0;
        DetachedVideoFrame detached;
    };

    struct ExtrinsicHealthFrame {
        std::string sn;
        std::string camKey;
        cv::Mat bgr;
        cv::Mat depthAlignedRgb16;
        uint64_t tsUs = 0;
        uint64_t depthTsUs = 0;
        float depthValueScaleMm = 0.0f;
        double ageMs = 0.0;
        double depthAgeMs = 0.0;
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

    std::vector<ExtrinsicHealthFrame> latestExtrinsicHealthFrames() {
        std::vector<ExtrinsicHealthFrame> out;
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mtx_);
        out.reserve(buffers_.size());
        for(const auto &kv: buffers_) {
            const auto &buf = kv.second;
            if(buf.latestRgb.empty() || buf.latestDepthAlignedRgb.empty()
               || buf.latestRgbSteady.time_since_epoch().count() == 0
               || buf.latestDepthSteady.time_since_epoch().count() == 0
               || !(buf.latestDepthValueScaleMm > 0.0f)) {
                continue;
            }
            if(buf.params.find(CollectDataType::RGB) == buf.params.end()
               || buf.params.find(CollectDataType::Depth) == buf.params.end()
               || !buf.rgbDepthParamValid) {
                continue;
            }
            ExtrinsicHealthFrame frame;
            frame.sn = kv.first;
            frame.camKey = buf.camKey;
            frame.bgr = buf.latestRgb.clone();
            frame.depthAlignedRgb16 = buf.latestDepthAlignedRgb.clone();
            frame.tsUs = buf.latestRgbTsUs;
            frame.depthTsUs = buf.latestDepthTsUs;
            frame.depthValueScaleMm = buf.latestDepthValueScaleMm;
            frame.ageMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - buf.latestRgbSteady).count();
            frame.depthAgeMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - buf.latestDepthSteady).count();
            out.push_back(std::move(frame));
        }
        std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
            return a.camKey < b.camKey;
        });
        return out;
    }

    void writeExtrinsicHealthConfigJson(const fs::path &path) const {
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
        cJSON_AddNumberToObject(root, "minPassingSnapshots", std::min(std::max(1, h.minPassingSnapshots), std::max(1, h.sampleCount)));
        cJSON_AddNumberToObject(root, "minFailingSnapshots", std::min(std::max(1, h.minFailingSnapshots), std::max(1, h.sampleCount)));
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
        if(printed) {
            writeTextFile(path, printed);
            cJSON_free(printed);
        }
        cJSON_Delete(root);
    }

    struct ExtrinsicHealthJudgedResidual {
        double transM = 0.0;
        double rotDeg = 0.0;
        double reprojPx = 0.0;
        bool hasTrans = false;
        bool hasRot = false;
        bool hasReproj = false;
    };

    static bool readJsonNumber(cJSON *obj, const char *name, double &out) {
        cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, name);
        if(!item || !cJSON_IsNumber(item)) {
            return false;
        }
        out = item->valuedouble;
        return true;
    }

    static std::string formatMetric(double value, int precision) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(precision) << value;
        return oss.str();
    }

    static int readJsonIntDefault(cJSON *obj, const char *name, int fallback = 0) {
        double value = static_cast<double>(fallback);
        if(readJsonNumber(obj, name, value)) {
            return static_cast<int>(std::llround(value));
        }
        return fallback;
    }

    static std::string readStringArrayJoined(cJSON *root, const char *name) {
        cJSON *array = cJSON_GetObjectItemCaseSensitive(root, name);
        if(!array || !cJSON_IsArray(array)) {
            return "";
        }
        std::ostringstream oss;
        bool first = true;
        cJSON *item = nullptr;
        cJSON_ArrayForEach(item, array) {
            if(!item || !cJSON_IsString(item) || !item->valuestring) {
                continue;
            }
            if(!first) {
                oss << ",";
            }
            first = false;
            oss << item->valuestring;
        }
        return oss.str();
    }

    static void appendExtrinsicHealthCameraStatusLine(cJSON *root, std::vector<std::string> &detailLines) {
        cJSON *counts = cJSON_GetObjectItemCaseSensitive(root, "camera_counts");
        if(!counts || !cJSON_IsObject(counts)) {
            return;
        }
        const int total = readJsonIntDefault(counts, "total");
        const int pass = readJsonIntDefault(counts, "pass");
        const int warn = readJsonIntDefault(counts, "warn");
        const int fail = readJsonIntDefault(counts, "fail");
        const int inconclusive = readJsonIntDefault(counts, "inconclusive");
        std::ostringstream oss;
        oss << "Extrinsic camera status:"
            << " total=" << total
            << " pass=" << pass << "[" << readStringArrayJoined(root, "pass_cameras") << "]"
            << " warn=" << warn << "[" << readStringArrayJoined(root, "warn_cameras") << "]"
            << " fail=" << fail << "[" << readStringArrayJoined(root, "fail_cameras") << "]"
            << " inconclusive=" << inconclusive << "[" << readStringArrayJoined(root, "inconclusive_cameras") << "]";
        detailLines.push_back(oss.str());
    }

    static void collectExtrinsicHealthDetailLines(cJSON *root, std::vector<std::string> &detailLines) {
        appendExtrinsicHealthCameraStatusLine(root, detailLines);
        std::map<std::string, ExtrinsicHealthJudgedResidual> byCamera;
        cJSON *samplesObj = cJSON_GetObjectItemCaseSensitive(root, "samples");
        if(!samplesObj || !cJSON_IsArray(samplesObj)) {
            return;
        }

        cJSON *sampleObj = nullptr;
        cJSON_ArrayForEach(sampleObj, samplesObj) {
            if(!sampleObj || !cJSON_IsObject(sampleObj)) {
                continue;
            }
            cJSON *camerasObj = cJSON_GetObjectItemCaseSensitive(sampleObj, "cameras");
            if(!camerasObj || !cJSON_IsObject(camerasObj)) {
                continue;
            }
            cJSON *cameraObj = nullptr;
            cJSON_ArrayForEach(cameraObj, camerasObj) {
                if(!cameraObj || !cJSON_IsObject(cameraObj) || !cameraObj->string) {
                    continue;
                }
                auto &acc = byCamera[cameraObj->string];
                double value = 0.0;
                if(readJsonNumber(cameraObj, "median_trans_m", value)) {
                    acc.transM = acc.hasTrans ? std::max(acc.transM, value) : value;
                    acc.hasTrans = true;
                }
                if(readJsonNumber(cameraObj, "median_rot_deg", value)) {
                    acc.rotDeg = acc.hasRot ? std::max(acc.rotDeg, value) : value;
                    acc.hasRot = true;
                }
                if(readJsonNumber(cameraObj, "median_reproj_px", value)) {
                    acc.reprojPx = acc.hasReproj ? std::max(acc.reprojPx, value) : value;
                    acc.hasReproj = true;
                }
            }
        }

        if(byCamera.empty()) {
            return;
        }
        detailLines.push_back("Extrinsic judged median residuals by camera:");
        for(const auto &kv: byCamera) {
            const auto &r = kv.second;
            std::ostringstream oss;
            oss << "cam" << kv.first
                << " median reproj=" << (r.hasReproj ? formatMetric(r.reprojPx, 3) : "n/a") << "px"
                << " trans=" << (r.hasTrans ? formatMetric(r.transM, 4) : "n/a") << "m"
                << " rot=" << (r.hasRot ? formatMetric(r.rotDeg, 3) : "n/a") << "deg";
            detailLines.push_back(oss.str());
        }
    }

    static bool readExtrinsicHealthResult(const fs::path &path,
                                          std::string &status,
                                          std::string &summary,
                                          std::vector<std::string> *detailLines = nullptr) {
        status.clear();
        summary.clear();
        if(detailLines) {
            detailLines->clear();
        }
        std::string content;
        if(!readTextFile(path, content)) {
            return false;
        }
        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            return false;
        }
        cJSON *statusObj = cJSON_GetObjectItemCaseSensitive(root, "status");
        if(statusObj && cJSON_IsString(statusObj) && statusObj->valuestring) {
            status = statusObj->valuestring;
        }
        cJSON *summaryObj = cJSON_GetObjectItemCaseSensitive(root, "summary_line");
        if(summaryObj && cJSON_IsString(summaryObj) && summaryObj->valuestring) {
            summary = summaryObj->valuestring;
        }
        if(detailLines) {
            collectExtrinsicHealthDetailLines(root, *detailLines);
        }
        cJSON_Delete(root);
        return !status.empty();
    }

    struct H265FrameItem {
        cv::Mat frame;
        std::string frameIndex;
        uint64_t tsUs = 0;
    };

    struct DepthFfv1FrameItem {
        cv::Mat frame;
        std::string frameIndex;
        uint64_t tsUs = 0;
    };

    struct DepthAlignTask {
        std::string sn;
        std::string camKey;
        std::string frameIndex;
        uint64_t tsUs = 0;
        cv::Mat frame;
        fs::path outPath;
        SaveOptions saveOptions;
        OBCameraParam rgbDepthParam{};
        bool rgbDepthParamValid = false;
        int rgbW = 0;
        int rgbH = 0;
        float valueScale = 0.0f;
        bool useFfv1 = false;
    };

    class H265Encoder {
    public:
        H265Encoder(fs::path outputPath, int width, int height, int fps, int threads, OBFormat inputFormat, SaveOptions options, size_t queueMax)
            : outputPath_(std::move(outputPath)),
              width_(width),
              height_(height),
              fps_(std::max(1, fps)),
              threads_(std::max(0, threads)),
              inputFormat_(h265RawInputFormatForColorProfile(inputFormat)),
              options_(std::move(options)),
              queueMax_(std::max<size_t>(1, queueMax)) {}

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
            requestStop();
            join();
        }

        void requestStop() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                stop_ = true;
                cv_.notify_all();
            }
        }

        void join() {
            if(worker_.joinable()) {
                worker_.join();
            }
        }

        bool idle() const {
            return queued_.load() == 0 && inFlight_.load() == 0;
        }

        size_t queued() const {
            return queued_.load();
        }

        int inFlight() const {
            return inFlight_.load();
        }

    private:
        std::string buildCommand() const {
            const std::string codec = resolvedH265Codec(options_);
            const std::string logLevel = trimString(options_.h265LogLevel).empty() ? "info" : trimString(options_.h265LogLevel);
            std::ostringstream cmd;
            cmd << "ffmpeg -hide_banner -loglevel " << shellQuote(logLevel) << " -stats -y"
                << " -f rawvideo -pix_fmt " << h265RawInputPixFmt(inputFormat_)
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
                      << " inputPixFmt=" << h265RawInputPixFmt(inputFormat_)
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
        OBFormat inputFormat_ = OB_FORMAT_BGR;
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

    class DepthFfv1Encoder {
    public:
        DepthFfv1Encoder(fs::path outputPath, int width, int height, int fps, size_t queueMax)
            : outputPath_(std::move(outputPath)),
              width_(width),
              height_(height),
              fps_(std::max(1, fps)),
              queueMax_(std::max<size_t>(1, queueMax)) {}

        ~DepthFfv1Encoder() {
            stop();
        }

        DepthFfv1Encoder(const DepthFfv1Encoder &) = delete;
        DepthFfv1Encoder &operator=(const DepthFfv1Encoder &) = delete;

        void start() {
            worker_ = std::thread([this]() { loop(); });
        }

        bool enqueue(std::string frameIndex, uint64_t tsUs, cv::Mat frame) {
            if(frame.empty() || frame.cols != width_ || frame.rows != height_ || frame.type() != CV_16UC1) {
                return false;
            }
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() {
                    return stop_ || queue_.size() < queueMax_;
                });
                if(stop_) {
                    return false;
                }
                queue_.push_back(DepthFfv1FrameItem{ std::move(frame), std::move(frameIndex), tsUs });
                queued_.fetch_add(1);
            }
            cv_.notify_one();
            return true;
        }

        void stop() {
            requestStop();
            join();
        }

        void requestStop() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                stop_ = true;
                cv_.notify_all();
            }
        }

        void join() {
            if(worker_.joinable()) {
                worker_.join();
            }
        }

        size_t queued() const {
            return queued_.load();
        }

        int inFlight() const {
            return inFlight_.load();
        }

    private:
        std::string buildCommand() const {
            std::ostringstream cmd;
            cmd << "ffmpeg -hide_banner -loglevel info -stats -y"
                << " -f rawvideo -pix_fmt gray16le"
                << " -s " << width_ << "x" << height_
                << " -r " << fps_
                << " -i - -an"
                << " -c:v ffv1 -level 3 -g 1 -slices 16 -slicecrc 1"
                << " -pix_fmt gray16le"
                << " " << shellQuote(outputPath_.string());
            return cmd.str();
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
                timestampOfs << "video_frame_index,frame_index,depth_timestamp_us\n";
            }
            else {
                std::cerr << "[collection][depth_ffv1] warning: failed to open timestamp sidecar: " << timestampPath << std::endl;
            }

            const std::string command = buildCommand();
            std::cerr << "[collection][depth_ffv1] start encoder"
                      << " output=" << outputPath_
                      << " size=" << width_ << "x" << height_
                      << " fps=" << fps_
                      << " codec=ffv1 pix_fmt=gray16le" << std::endl;
            std::cerr << "[collection][depth_ffv1] ffmpeg command: " << command << std::endl;
            FILE *pipe = popen(command.c_str(), "w");
            if(!pipe) {
                std::cerr << "[collection][depth_ffv1] failed to start ffmpeg process: " << outputPath_ << std::endl;
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
                DepthFfv1FrameItem item;
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
                std::cerr << "[collection][depth_ffv1] timestamp sidecar written path=" << timestampPath
                          << " frames=" << encodedFrameCount_ << std::endl;
            }

            const int status = pclose(pipe);
            if(status != 0) {
                std::cerr << "[collection][depth_ffv1] ffmpeg encoder failed"
                          << " output=" << outputPath_
                          << " status=" << status << std::endl;
            }
            else {
                std::cerr << "[collection][depth_ffv1] encoder finished output=" << outputPath_ << std::endl;
            }
        }

        fs::path outputPath_;
        int width_ = 0;
        int height_ = 0;
        int fps_ = 30;
        size_t queueMax_ = 1024;
        mutable std::mutex mtx_;
        std::condition_variable cv_;
        std::deque<DepthFfv1FrameItem> queue_;
        std::thread worker_;
        bool stop_ = false;
        std::atomic<size_t> queued_{ 0 };
        std::atomic<int> inFlight_{ 0 };
        size_t encodedFrameCount_ = 0;
    };

    struct TouchSessionStream {
        std::string streamId;
        std::string handSide;
        int         sensorType = 0;
        fs::path    rawPath;
        fs::path    rawTmpPath;
        std::ofstream rawOfs;
        bool        rawOpen = false;
        size_t      capturedSamples = 0;
    };

    struct TouchQueuedSample {
        std::string streamId;
        TactileSample sample;
    };

    struct TouchRuntime {
        std::string streamId;
        std::string handSide;
        int         sensorType = 0;
        TactileModuleConfig config;
        std::unique_ptr<TactileRecorder> recorder;
        std::thread recordThread;
    };

    struct PickedTouchSample {
        std::string streamId;
        size_t      sampleIndex = 0;
        bool        hasSample = false;
        uint64_t    absDiffUs = 0;
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
        bool saveEgo = false;
        bool saveTouch = false;
        bool saveRgbTimesteps = false;
        bool saveDepthTimesteps = false;
        size_t fisheyeCameraCount = 0;
        std::vector<bool> wroteFisheyeCameraParams;
        std::unordered_map<std::string, CamWorldPose> camToWorld;
        std::unordered_map<std::string, std::unordered_map<CollectDataType, StreamState>> streams;
        std::deque<FisheyeFrameSet> fisheyeSets;
        std::deque<EgoFrame> egoFrames;
        std::vector<TouchSessionStream> touchStreams;
        std::unordered_map<std::string, std::deque<TactileSample>> touchSamples;
        bool multiviewEos = false;
        bool fisheyeEos = false;
        bool egoEos = false;
        bool touchEos = false;
        bool coordinatorDone = false;
        bool timestampsFinalized = false;
        bool imuWritten = false;
        fs::path timestampsPath;
        fs::path timestampsTmpPath;
        std::ofstream timestampsOfs;
        bool timestampsOpen = false;
        fs::path egoAlignedTimestampsPath;
        fs::path egoAlignedTimestampsTmpPath;
        std::ofstream egoAlignedTimestampsOfs;
        bool egoAlignedTimestampsOpen = false;
        fs::path egoAllTimestampsPath;
        fs::path egoAllTimestampsTmpPath;
        uint64_t fisheyeOnlyNextTargetUs = 0;
        bool fisheyeOnlyTargetInit = false;
        uint64_t lastEmittedFisheyeTs = 0;
        bool hasLastEmittedFisheyeTs = false;
        uint64_t egoOnlyNextTargetUs = 0;
        bool egoOnlyTargetInit = false;
        uint64_t lastEmittedEgoTs = 0;
        bool hasLastEmittedEgoTs = false;
        size_t nextFrameIndex = 0;
        size_t refFrameCount = 0;
        uint64_t firstRefTs = 0;
        uint64_t lastRefTs = 0;
        size_t fisheyeCapturedSets = 0;
        size_t egoCapturedFrames = 0;
        std::unordered_set<int> egoAlignedSourceFrameIndices;
        std::unordered_set<uint64_t> egoAlignedRefTimestamps;
        size_t egoMissingAlignedRows = 0;
        std::deque<EgoFrame> egoPendingSoftAlignFrames;
        std::vector<EgoFrame> egoAllFrames;
        bool egoSoftAlignEnabled = false;
        bool egoSoftAlignReady = false;
        uint64_t egoSoftAlignGlobalFirstRefUs = 0;
        uint64_t egoSoftAlignPicoFirstRawUs = 0;
        int64_t egoSoftAlignOffsetUs = 0;
        bool egoSoftAlignHasLastRaw = false;
        uint64_t egoSoftAlignLastRawUs = 0;
        size_t egoTimestampNonMonotonic = 0;
        size_t egoTimestampLargeGap = 0;
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

    static uint64_t egoRawRefTimestamp(const EgoFrame &frame) {
        return frame.rawRefTimestampUs != 0 ? frame.rawRefTimestampUs : frame.refTimestampUs;
    }

    static uint64_t egoBaseRefTimestamp(const EgoFrame &frame) {
        return frame.refTimestampUs != 0 ? frame.refTimestampUs : egoRawRefTimestamp(frame);
    }

    bool ensureEgoSoftAlignReadyLocked() {
        if(!session_.egoSoftAlignEnabled) {
            return true;
        }
        if(session_.egoSoftAlignReady) {
            return true;
        }
        if(session_.firstRefTs == 0 || session_.egoPendingSoftAlignFrames.empty()) {
            return false;
        }
        const uint64_t firstBase = egoBaseRefTimestamp(session_.egoPendingSoftAlignFrames.front());
        if(firstBase == 0) {
            return false;
        }
        session_.egoSoftAlignGlobalFirstRefUs = session_.firstRefTs;
        session_.egoSoftAlignPicoFirstRawUs = firstBase;
        session_.egoSoftAlignOffsetUs = signedDiffUs(session_.egoSoftAlignGlobalFirstRefUs, session_.egoSoftAlignPicoFirstRawUs);
        session_.egoSoftAlignReady = true;
        std::cerr << "[collection][ego] soft timestamp alignment ready"
                  << " orbbec_first_us=" << session_.egoSoftAlignGlobalFirstRefUs
                  << " ego_first_base_us=" << session_.egoSoftAlignPicoFirstRawUs
                  << " offset_us=" << session_.egoSoftAlignOffsetUs << std::endl;
        return true;
    }

    void validateAndApplyEgoSoftAlignmentLocked(EgoFrame &frame) {
        const uint64_t raw = egoRawRefTimestamp(frame);
        const uint64_t base = egoBaseRefTimestamp(frame);
        frame.rawRefTimestampUs = raw;
        frame.rawDeltaUs = 0;
        frame.timestampValidation = "ok";
        if(raw == 0 || base == 0) {
            frame.timestampValidation = "missing_raw_timestamp";
            return;
        }

        if(session_.egoSoftAlignHasLastRaw) {
            if(raw <= session_.egoSoftAlignLastRawUs) {
                session_.egoTimestampNonMonotonic++;
                frame.timestampValidation = "non_monotonic";
            }
            else {
                frame.rawDeltaUs = raw - session_.egoSoftAlignLastRawUs;
                const uint64_t largeGapUs = std::max<uint64_t>(session_.stepUs * 3, session_.stepUs + session_.maxAbsDiffUs * 2);
                if(largeGapUs > 0 && frame.rawDeltaUs > largeGapUs) {
                    session_.egoTimestampLargeGap++;
                    frame.timestampValidation = "large_gap";
                }
            }
        }
        session_.egoSoftAlignLastRawUs = raw;
        session_.egoSoftAlignHasLastRaw = true;

        if(session_.egoSoftAlignEnabled) {
            frame.softAlignOffsetUs = session_.egoSoftAlignOffsetUs;
            frame.refTimestampUs = addSignedUs(base, session_.egoSoftAlignOffsetUs);
        }
        else {
            frame.softAlignOffsetUs = 0;
            frame.refTimestampUs = base;
            if(frame.timestampValidation == "ok") {
                frame.timestampValidation = frame.timeCalibrationStatus == "ok" ? "time_calibrated" : "raw_no_soft_align";
            }
        }
    }

    void commitEgoFrameLocked(EgoFrame &&frame) {
        validateAndApplyEgoSoftAlignmentLocked(frame);
        session_.egoAllFrames.push_back(frame);
        auto insertPos = std::upper_bound(session_.egoFrames.begin(),
                                          session_.egoFrames.end(),
                                          frame.refTimestampUs,
                                          [](uint64_t tsUs, const EgoFrame &sample) {
                                              return tsUs < sample.refTimestampUs;
                                          });
        session_.egoFrames.insert(insertPos, std::move(frame));
        session_.egoCapturedFrames++;
    }

    void flushPendingEgoSoftAlignLocked() {
        if(session_.egoPendingSoftAlignFrames.empty()) {
            return;
        }
        if(!ensureEgoSoftAlignReadyLocked()) {
            return;
        }
        while(!session_.egoPendingSoftAlignFrames.empty()) {
            commitEgoFrameLocked(std::move(session_.egoPendingSoftAlignFrames.front()));
            session_.egoPendingSoftAlignFrames.pop_front();
        }
    }

    void flushPendingEgoWithoutGlobalAnchorLocked() {
        if(session_.egoPendingSoftAlignFrames.empty()) {
            return;
        }
        session_.egoSoftAlignEnabled = false;
        while(!session_.egoPendingSoftAlignFrames.empty()) {
            commitEgoFrameLocked(std::move(session_.egoPendingSoftAlignFrames.front()));
            session_.egoPendingSoftAlignFrames.pop_front();
        }
    }

    void acceptEgoFrameLocked(EgoFrame &&frame) {
        if(session_.egoSoftAlignEnabled && !session_.egoSoftAlignReady) {
            session_.egoPendingSoftAlignFrames.push_back(std::move(frame));
            flushPendingEgoSoftAlignLocked();
            return;
        }
        commitEgoFrameLocked(std::move(frame));
    }

    TouchSessionStream *findTouchStreamLocked(const std::string &streamId) {
        for(auto &stream: session_.touchStreams) {
            if(stream.streamId == streamId) {
                return &stream;
            }
        }
        return nullptr;
    }

    const TouchSessionStream *findTouchStreamLocked(const std::string &streamId) const {
        for(const auto &stream: session_.touchStreams) {
            if(stream.streamId == streamId) {
                return &stream;
            }
        }
        return nullptr;
    }

    void writeTouchRawRowLocked(const std::string &streamId, const TactileSample &sample) {
        auto *stream = findTouchStreamLocked(streamId);
        if(!stream || !stream->rawOpen || !stream->rawOfs.is_open()) {
            return;
        }
        const auto &frame = sample.frame;
        std::vector<std::string> row;
        row.reserve(15 + kJqShroomPressureChannelCount);
        row.push_back(std::to_string(sample.sequence));
        row.push_back(std::to_string(sample.representativeTimestampUs));
        {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6) << sample.representativeTimestampSec;
            row.push_back(oss.str());
        }
        row.push_back(frame.side);
        row.push_back(std::to_string(frame.sensorType));
        row.push_back(std::to_string(frame.packet1TimestampUs));
        row.push_back(std::to_string(frame.packet2TimestampUs));
        row.push_back(std::to_string(frame.packetGapUs));
        row.push_back(bytesToLowerHex(frame.imuRaw));
        row.push_back(std::to_string(frame.imuW));
        row.push_back(std::to_string(frame.imuX));
        row.push_back(std::to_string(frame.imuY));
        row.push_back(std::to_string(frame.imuZ));
        row.push_back(frame.imuValid ? "1" : "0");
        row.push_back(frame.qualityFlag);
        for(size_t i = 0; i < kJqShroomPressureChannelCount; ++i) {
            if(i < frame.rawAdc.size()) {
                row.push_back(std::to_string(frame.rawAdc[i]));
            }
            else {
                row.emplace_back();
            }
        }
        writeCsvRow(stream->rawOfs, row);
    }

    void commitTouchSampleLocked(TouchQueuedSample &&queued) {
        writeTouchRawRowLocked(queued.streamId, queued.sample);
        auto &samples = session_.touchSamples[queued.streamId];
        auto insertPos = std::upper_bound(samples.begin(),
                                          samples.end(),
                                          queued.sample.representativeTimestampUs,
                                          [](uint64_t tsUs, const TactileSample &item) {
                                              return tsUs < item.representativeTimestampUs;
                                          });
        samples.insert(insertPos, std::move(queued.sample));
        if(auto *stream = findTouchStreamLocked(queued.streamId)) {
            stream->capturedSamples++;
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
        session_.egoFrames.clear();
        session_.touchSamples.clear();
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

    double cameraStreamTimeoutSec() const {
        return cfg_.cameraStreamTimeoutSec > 0.0 ? cfg_.cameraStreamTimeoutSec : 2.0;
    }

    uint64_t egoMissingSlotDropWaitUsLocked() const {
        const auto timeoutUs = static_cast<uint64_t>(std::max(0.1, cameraStreamTimeoutSec()) * 1000000.0 + 0.5);
        return std::max<uint64_t>(timeoutUs, std::max<uint64_t>(session_.stepUs * 3, session_.maxAbsDiffUs * 2));
    }

    bool egoSlotDropWaitElapsedLocked(uint64_t centerUs, const StreamState &refState) const {
        if(refState.eos) {
            return true;
        }
        // Bound the head-of-line wait when Pico/ego stops producing frames.
        const uint64_t staleRefTailUs = saturatingAddUs(centerUs, egoMissingSlotDropWaitUsLocked());
        return refState.maxTsUs >= staleRefTailUs;
    }

    bool egoCanResolveOrDropSlotLocked(uint64_t centerUs, const StreamState &refState) const {
        if(!session_.saveEgo || session_.egoEos) {
            return true;
        }
        const uint64_t requiredEgoTailUs = saturatingAddUs(centerUs, session_.maxAbsDiffUs);
        if(!session_.egoFrames.empty() && session_.egoFrames.back().refTimestampUs >= requiredEgoTailUs) {
            return true;
        }
        return egoSlotDropWaitElapsedLocked(centerUs, refState);
    }

    static bool isCameraHealthStream(CollectDataType t) {
        return t == CollectDataType::RGB || t == CollectDataType::Depth;
    }

    static std::string streamHealthKey(const std::string &sn, CollectDataType type) {
        return sn + ":" + dataTypeLabel(type);
    }

    static std::string fisheyeStreamHealthKey(const std::string &cameraId) {
        return "fisheye:" + cameraId + ":RGB";
    }

    static std::string fisheyeStreamDisplayName(const std::string &cameraId) {
        return "fisheye " + cameraId + " RGB";
    }

    static std::string egoStreamHealthKey() {
        return "ego:RGB";
    }

    static std::string egoStreamDisplayName() {
        return "ego RGB";
    }

    static std::string touchStreamHealthKey(const std::string &streamId) {
        return "touch:" + streamId + ":pressure";
    }

    static std::string streamHealthDisplayName(const StreamHealthState &state) {
        if(!state.displayName.empty()) {
            return state.displayName;
        }
        return "cam" + state.camKey + " " + dataTypeLabel(state.type);
    }

    void resetStreamHealth() {
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        streamHealth_.clear();
        activeCameraFault_.reset();
    }

    void initStreamHealthForActiveStreams() {
        std::vector<StreamHealthState> states;
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            states.reserve(buffers_.size() * 2 + activeFisheyeCameraCount_);
            for(const auto &kv: buffers_) {
                for(const auto t: { CollectDataType::RGB, CollectDataType::Depth }) {
                    if(kv.second.params.find(t) == kv.second.params.end()) {
                        continue;
                    }
                    StreamHealthState state;
                    state.healthKey = streamHealthKey(kv.first, t);
                    state.sn = kv.first;
                    state.camKey = kv.second.camKey;
                    state.displayName = "cam" + state.camKey + " " + dataTypeLabel(t);
                    state.type = t;
                    state.startedSteady = now;
                    state.lastFrameSteady = now;
                    states.push_back(std::move(state));
                }
            }
        }
        if(fisheyeEnabled_ && activeFisheyeCameraCount_ > 0) {
            for(size_t i = 0; i < activeFisheyeCameraCount_; ++i) {
                const std::string cameraId = (i < activeFisheyeCameraIds_.size() && !activeFisheyeCameraIds_[i].empty())
                    ? activeFisheyeCameraIds_[i]
                    : ("cam" + std::to_string(i));
                StreamHealthState state;
                state.healthKey = fisheyeStreamHealthKey(cameraId);
                state.sn = cameraId;
                state.camKey = cameraId;
                state.displayName = fisheyeStreamDisplayName(cameraId);
                state.type = CollectDataType::RGB;
                state.startedSteady = now;
                state.lastFrameSteady = now;
                state.requireTimestampAdvance = true;
                states.push_back(std::move(state));
            }
        }
        if(egoEnabled_) {
            StreamHealthState state;
            state.healthKey = egoStreamHealthKey();
            state.sn = "ego";
            state.camKey = "ego";
            state.displayName = egoStreamDisplayName();
            state.type = CollectDataType::RGB;
            state.startedSteady = now;
            state.lastFrameSteady = now;
            state.requireTimestampAdvance = true;
            state.everReceived = egoRecorder_.isConnected();
            states.push_back(std::move(state));
        }
        if(touchEnabled_) {
            for(const auto &runtime: touchRuntimes_) {
                StreamHealthState state;
                state.healthKey = touchStreamHealthKey(runtime.streamId);
                state.sn = "touch_" + runtime.streamId;
                state.camKey = runtime.streamId;
                state.displayName = touchDisplayName(runtime.streamId);
                state.type = CollectDataType::RGB;
                state.startedSteady = now;
                state.lastFrameSteady = now;
                state.requireTimestampAdvance = true;
                state.everReceived = false;
                states.push_back(std::move(state));
            }
        }

        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        streamHealth_.clear();
        activeCameraFault_.reset();
        for(auto &state: states) {
            const std::string key = state.healthKey;
            streamHealth_.emplace(key, std::move(state));
        }
    }

    void noteCameraStreamFrame(const std::string &deviceSn, CollectDataType type) {
        if(!isCameraHealthStream(type)) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        auto it = streamHealth_.find(streamHealthKey(deviceSn, type));
        if(it == streamHealth_.end()) {
            return;
        }
        if(it->second.firstFrameSteady.time_since_epoch().count() == 0) {
            it->second.firstFrameSteady = now;
        }
        it->second.lastFrameSteady = now;
        it->second.everReceived = true;
    }

    void noteFisheyeFrameSet(const FisheyeFrameSet &frameSet) {
        if(frameSet.frames.empty()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        for(const auto &frame: frameSet.frames) {
            if(frame.cameraId.empty() || frame.captureTimestampUs == 0) {
                continue;
            }
            auto it = streamHealth_.find(fisheyeStreamHealthKey(frame.cameraId));
            if(it == streamHealth_.end()) {
                continue;
            }
            auto &state = it->second;
            if(state.requireTimestampAdvance && frame.captureTimestampUs <= state.lastFrameTimestampUs) {
                continue;
            }
            state.lastFrameTimestampUs = frame.captureTimestampUs;
            state.lastFrameSteady = now;
            state.everReceived = true;
        }
    }

    void noteEgoFrame(const EgoFrame &frame) {
        if(frame.refTimestampUs == 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        auto it = streamHealth_.find(egoStreamHealthKey());
        if(it == streamHealth_.end()) {
            return;
        }
        auto &state = it->second;
        if(state.requireTimestampAdvance && frame.refTimestampUs <= state.lastFrameTimestampUs) {
            return;
        }
        state.lastFrameTimestampUs = frame.refTimestampUs;
        state.lastFrameSteady = now;
        state.everReceived = true;
    }

    void noteTouchSample(const std::string &streamId, const TactileSample &sample) {
        if(sample.representativeTimestampUs == 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        auto it = streamHealth_.find(touchStreamHealthKey(streamId));
        if(it == streamHealth_.end()) {
            return;
        }
        auto &state = it->second;
        if(state.requireTimestampAdvance && sample.representativeTimestampUs <= state.lastFrameTimestampUs) {
            return;
        }
        state.lastFrameTimestampUs = sample.representativeTimestampUs;
        state.lastFrameSteady = now;
        state.everReceived = true;
    }

    void refreshEgoReadyHealth() {
        if(!egoEnabled_ || recording_.load()) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const bool connected = egoRecorder_.isConnected();
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        auto it = streamHealth_.find(egoStreamHealthKey());
        if(it == streamHealth_.end()) {
            return;
        }
        it->second.lastFrameSteady = now;
        it->second.everReceived = connected;
        if(!connected) {
            it->second.lastFrameTimestampUs = 0;
        }
    }

    void refreshTouchReadyHealth() {
        if(!touchEnabled_ || recording_.load()) {
            return;
        }
        for(auto &runtime: touchRuntimes_) {
            if(!runtime.recorder || !runtime.recorder->isRunning()) {
                continue;
            }
            std::string err;
            auto sample = runtime.recorder->snapshotLatest(&err);
            if(sample) {
                noteTouchSample(runtime.streamId, *sample);
            }
        }
    }

    void markStreamHealthCaptureStarted(std::chrono::steady_clock::time_point startTime) {
        std::lock_guard<std::mutex> lock(streamHealthMtx_);
        activeCameraFault_.reset();
        for(auto &kv: streamHealth_) {
            kv.second.startedSteady = startTime;
            kv.second.lastFrameSteady = startTime;
            kv.second.lastFrameTimestampUs = 0;
            kv.second.faulted = false;
        }
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

    static uint64_t saturatingAddUs(uint64_t value, uint64_t delta) {
        if(std::numeric_limits<uint64_t>::max() - value < delta) {
            return std::numeric_limits<uint64_t>::max();
        }
        return value + delta;
    }

    static int64_t signedDiffUs(uint64_t a, uint64_t b) {
        return static_cast<int64_t>(a) - static_cast<int64_t>(b);
    }

    static uint64_t addSignedUs(uint64_t value, int64_t delta) {
        if(delta >= 0) {
            return saturatingAddUs(value, static_cast<uint64_t>(delta));
        }
        const uint64_t sub = static_cast<uint64_t>(-(delta + 1)) + 1ULL;
        return value > sub ? value - sub : 0;
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
        if(session.saveEgo) {
            header.push_back("ego_frame_index");
            header.push_back("ego_timestamp_us");
        }
        if(session.saveTouch) {
            for(const auto &stream: session.touchStreams) {
                header.push_back("touch_" + stream.streamId + "_frame_index");
                header.push_back("touch_" + stream.streamId + "_timestamp_us");
                header.push_back("touch_" + stream.streamId + "_abs_diff_us");
            }
        }
        header.push_back("rgbd_max_diff_ms");
        header.push_back("all_modalities_max_diff_ms");
        writeCsvRow(session.timestampsOfs, header);
        if(session.saveEgo) {
            session.egoAlignedTimestampsPath = session.dest / "ego" / "RGB" / "rgb.h265.timestamps.csv";
            session.egoAlignedTimestampsTmpPath = session.egoAlignedTimestampsPath;
            session.egoAlignedTimestampsTmpPath += ".tmp";
            session.egoAlignedTimestampsOfs.open(session.egoAlignedTimestampsTmpPath, std::ios::out | std::ios::trunc);
            if(!session.egoAlignedTimestampsOfs.is_open()) {
                return false;
            }
            session.egoAlignedTimestampsOpen = true;
            writeCsvRow(session.egoAlignedTimestampsOfs,
                        { "frame_index",
                          "ego_frame_index",
                          "ego_source_frame_index",
                          "ego_ref_timestamp_us",
                          "ego_raw_ref_timestamp_us",
                          "ego_time_calibration_offset_us",
                          "ego_time_calibration_status",
                          "ego_soft_align_offset_us",
                          "ego_timestamp_validation",
                          "ego_rgb_timestamp_us",
                          "pico_frame_timestamp_ns",
                          "pico_xr_head_timestamp_us",
                          "pico_gaze_timestamp_us" });
            session.egoAllTimestampsPath = session.dest / "ego" / "RGB" / "rgb.h265.all_timestamps.csv";
            session.egoAllTimestampsTmpPath = session.egoAllTimestampsPath;
            session.egoAllTimestampsTmpPath += ".tmp";
        }
        if(session.saveTouch) {
            const std::string touchDirName = cfg_.touch.save.directoryName.empty() ? std::string("touch") : cfg_.touch.save.directoryName;
            for(auto &stream: session.touchStreams) {
                const std::string rawName = stream.streamId + "_raw.csv";
                stream.rawPath = session.dest / touchDirName / rawName;
                stream.rawTmpPath = stream.rawPath;
                stream.rawTmpPath += ".tmp";
                stream.rawOfs.open(stream.rawTmpPath, std::ios::out | std::ios::trunc);
                if(!stream.rawOfs.is_open()) {
                    return false;
                }
                stream.rawOpen = true;
                stream.rawOfs << "sample_index,touch_timestamp_us,touch_timestamp_s,side,sensor_type,"
                              << "packet1_ts_us,packet2_ts_us,packet_gap_us,"
                              << "imu_raw_hex,imu_w,imu_x,imu_y,imu_z,imu_valid,quality_flag";
                for(size_t i = 0; i < kJqShroomPressureChannelCount; ++i) {
                    stream.rawOfs << ",pressure_" << std::setw(3) << std::setfill('0') << i;
                }
                stream.rawOfs << std::setfill(' ') << "\n";
            }
        }
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
        if(session_.egoAlignedTimestampsOfs.is_open()) {
            session_.egoAlignedTimestampsOfs.flush();
            session_.egoAlignedTimestampsOfs.close();
        }
        session_.egoAlignedTimestampsOpen = false;
        for(auto &stream: session_.touchStreams) {
            if(stream.rawOfs.is_open()) {
                stream.rawOfs.flush();
                stream.rawOfs.close();
            }
            stream.rawOpen = false;
        }
    }

    void writeEgoAllTimestampsLocked() {
        if(!session_.saveEgo || session_.egoAllTimestampsTmpPath.empty()) {
            return;
        }
        std::ofstream ofs(session_.egoAllTimestampsTmpPath, std::ios::out | std::ios::trunc);
        if(!ofs.is_open()) {
            std::cerr << "[collection][ego] warning: failed to write all ego timestamps path="
                      << session_.egoAllTimestampsTmpPath << std::endl;
            return;
        }
        writeCsvRow(ofs,
                    { "ego_frame_index",
                      "ego_source_frame_index",
                      "ego_ref_timestamp_us",
                      "ego_raw_ref_timestamp_us",
                      "ego_time_calibration_offset_us",
                      "ego_time_calibration_status",
                      "ego_soft_align_offset_us",
                      "ego_timestamp_validation",
                      "ego_raw_delta_us",
                      "ego_rgb_timestamp_us",
                      "pico_frame_timestamp_ns",
                      "pico_xr_head_timestamp_us",
                      "pico_gaze_timestamp_us" });
        for(const auto &frame: session_.egoAllFrames) {
            writeCsvRow(ofs,
                        { std::to_string(frame.videoFrameIndex),
                          std::to_string(frame.sourceFrameIndex),
                          std::to_string(frame.refTimestampUs),
                          std::to_string(frame.rawRefTimestampUs),
                          std::to_string(frame.timeCalibrationOffsetUs),
                          frame.timeCalibrationStatus,
                          std::to_string(frame.softAlignOffsetUs),
                          frame.timestampValidation,
                          std::to_string(frame.rawDeltaUs),
                          std::to_string(frame.rgbTimestampUs),
                          std::to_string(frame.picoFrameTimestampNs),
                          std::to_string(frame.xrHeadTimestampUs),
                          std::to_string(frame.gazeTimestampUs) });
        }
        ofs.flush();
        ofs.close();
        try {
            fs::rename(session_.egoAllTimestampsTmpPath, session_.egoAllTimestampsPath);
        }
        catch(...) {
        }
    }

    void finalizeSessionTimestampsLocked() {
        if(session_.timestampsFinalized) {
            return;
        }
        writeEgoAllTimestampsLocked();
        closeSessionTimestampsLocked();
        if(!session_.timestampsTmpPath.empty()) {
            try {
                fs::rename(session_.timestampsTmpPath, session_.timestampsPath);
            }
            catch(...) {
            }
        }
        if(!session_.egoAlignedTimestampsTmpPath.empty()) {
            try {
                fs::rename(session_.egoAlignedTimestampsTmpPath, session_.egoAlignedTimestampsPath);
            }
            catch(...) {
            }
        }
        for(const auto &stream: session_.touchStreams) {
            if(!stream.rawTmpPath.empty()) {
                try {
                    fs::rename(stream.rawTmpPath, stream.rawPath);
                }
                catch(...) {
                }
            }
        }
        session_.timestampsFinalized = true;
    }

    void writeTouchManifestJson(const fs::path &touchDir) const {
        cJSON *root = cJSON_CreateObject();
        cJSON_AddStringToObject(root, "schema", "orbbec.touch.jq_shroom.v1");
        cJSON_AddStringToObject(root, "protocol", "jq_shroom_record_tactile_py");
        cJSON_AddNumberToObject(root, "pressure_channels", static_cast<double>(kJqShroomPressureChannelCount));
        cJSON_AddNumberToObject(root, "target_fps", cfg_.touch.targetFps);
        cJSON_AddStringToObject(root, "timestamp_domain", "collection_ref_timestamp_us");
        cJSON *devices = cJSON_CreateArray();
        for(const auto &runtime: touchRuntimes_) {
            const auto &touchCfg = runtime.config;
            cJSON *dev = cJSON_CreateObject();
            cJSON_AddStringToObject(dev, "id", runtime.streamId.c_str());
            cJSON_AddStringToObject(dev, "side", touchCfg.handSide.c_str());
            cJSON_AddNumberToObject(dev, "sensor_type", touchCfg.sensorType);
            cJSON_AddNumberToObject(dev, "baud_rate", touchCfg.serial.baudRate);
            cJSON_AddStringToObject(dev, "port_path", touchCfg.serial.portPath.c_str());
            cJSON_AddStringToObject(dev, "raw_csv", (runtime.streamId + "_raw.csv").c_str());
            cJSON_AddItemToArray(devices, dev);
        }
        cJSON_AddItemToObject(root, "devices", devices);
        char *printed = cJSON_Print(root);
        cJSON_Delete(root);
        if(!printed) {
            return;
        }
        writeTextFile(touchDir / "touch_manifest.json", printed);
        cJSON_free(printed);
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
        size_t touchCapturedSamples = 0;
        for(const auto &stream: session.touchStreams) {
            touchCapturedSamples += stream.capturedSamples;
        }
        if(touchCapturedSamples > 0) {
            oss << "  TouchSamples=" << touchCapturedSamples;
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
            std::unique_lock<std::mutex> lock(recordMtx_);
            recordCv_.wait(lock, [&]() {
                return recordStop_.load() || stopping_.load() || recordQueue_.size() < recordQueueMax_;
            });
            if(recordStop_.load() || stopping_.load()) {
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
                    if(cfg_.save.rgbH265) {
                        okCopy = copyDetachedColorRawRgbOrBgr(task.detached, copied);
                    }
                    if(!okCopy) {
                        okCopy = copyDetachedColorToBgr(task.detached, copied);
                    }
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

    size_t configuredDepthAlignWorkerCount(const SessionState &session) const {
        if(!session.saveDepthTimesteps) {
            return 0;
        }
        if(cfg_.depthAlignWorkers > 0) {
            return static_cast<size_t>(cfg_.depthAlignWorkers);
        }
        return softWriterThreadCount(cfg_);
    }

    size_t configuredDepthAlignQueueCapacity(const SessionState &session) const {
        if(cfg_.depthAlignQueueCapacity > 0) {
            return static_cast<size_t>(cfg_.depthAlignQueueCapacity);
        }
        const size_t cams = std::max<size_t>(1, session.deviceSns.size());
        int fps = cfg_.collectFps > 0 ? cfg_.collectFps : uiFpsFallback_;
        if(fps <= 0) {
            fps = 30;
        }
        return std::max<size_t>(16, cams * static_cast<size_t>(fps));
    }

    void startDepthAlignWorkers(const SessionState &session) {
        stopDepthAlignWorkers();
        const size_t workerN = configuredDepthAlignWorkerCount(session);
        if(workerN == 0) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(depthAlignMtx_);
            depthAlignStop_.store(false);
            depthAlignQueue_.clear();
            depthAlignQueueMax_ = configuredDepthAlignQueueCapacity(session);
            queuedDepthAlignCount_.store(0);
            depthAlignInFlight_.store(0);
        }
        depthAlignWorkers_.reserve(workerN);
        for(size_t i = 0; i < workerN; ++i) {
            depthAlignWorkers_.emplace_back([this]() { depthAlignWorkerLoop(); });
        }
        depthAlignActive_.store(true);
        std::cerr << "[collection][depth_align] start workers=" << workerN
                  << " queueCapacity=" << depthAlignQueueMax_ << std::endl;
    }

    void waitDepthAlignQueueIdle() {
        std::unique_lock<std::mutex> lock(depthAlignMtx_);
        depthAlignDrainCv_.wait(lock, [&]() {
            return depthAlignQueue_.empty() && depthAlignInFlight_.load() == 0;
        });
    }

    void stopDepthAlignWorkers() {
        {
            std::lock_guard<std::mutex> lock(depthAlignMtx_);
            depthAlignStop_.store(true);
            depthAlignCv_.notify_all();
            depthAlignDrainCv_.notify_all();
        }
        for(auto &worker: depthAlignWorkers_) {
            if(worker.joinable()) {
                worker.join();
            }
        }
        depthAlignWorkers_.clear();
        {
            std::lock_guard<std::mutex> lock(depthAlignMtx_);
            depthAlignQueue_.clear();
            queuedDepthAlignCount_.store(0);
            depthAlignInFlight_.store(0);
        }
        depthAlignActive_.store(false);
    }

    void enqueueDepthAlignTask(DepthAlignTask &&task) {
        if(task.frame.empty()) {
            return;
        }
        {
            std::unique_lock<std::mutex> lock(depthAlignMtx_);
            depthAlignCv_.wait(lock, [&]() {
                return depthAlignStop_.load() || depthAlignQueue_.size() < depthAlignQueueMax_;
            });
            if(depthAlignStop_.load()) {
                return;
            }
            depthAlignQueue_.push_back(std::move(task));
            queuedDepthAlignCount_.fetch_add(1);
        }
        depthAlignCv_.notify_one();
    }

    void processDepthAlignTask(DepthAlignTask &task, cv::Mat &aligned, cv::Mat &zBuf) {
        cv::Mat output = task.frame;
        if(task.rgbDepthParamValid && task.rgbW > 0 && task.rgbH > 0 && task.valueScale > 0.0f) {
            alignDepthToRgbInto(task.frame, task.valueScale, task.rgbDepthParam, task.rgbW, task.rgbH, aligned, zBuf);
            output = aligned;
        }

        if(task.useFfv1) {
            cv::Mat frameForEncoder = output.clone();
            if(enqueueDepthFfv1Frame(task.sn, task.frameIndex, task.tsUs, std::move(frameForEncoder))) {
                return;
            }
            std::cerr << "[collection][depth_ffv1] warning: encoder unavailable, fallback PNG frame="
                      << task.frameIndex << " cam=" << task.camKey << std::endl;
        }

        saveRawMatToPng(output, task.outPath, task.saveOptions.pngCompression);
    }

    void depthAlignWorkerLoop() {
        cv::Mat aligned;
        cv::Mat zBuf;
        while(true) {
            DepthAlignTask task;
            {
                std::unique_lock<std::mutex> lock(depthAlignMtx_);
                depthAlignCv_.wait(lock, [&]() {
                    return depthAlignStop_.load() || !depthAlignQueue_.empty();
                });
                if(depthAlignQueue_.empty() && depthAlignStop_.load()) {
                    break;
                }
                if(depthAlignQueue_.empty()) {
                    continue;
                }
                task = std::move(depthAlignQueue_.front());
                depthAlignQueue_.pop_front();
                queuedDepthAlignCount_.fetch_sub(1);
                depthAlignInFlight_.fetch_add(1);
                depthAlignCv_.notify_all();
            }
            try {
                processDepthAlignTask(task, aligned, zBuf);
            }
            catch(...) {
            }
            depthAlignInFlight_.fetch_sub(1);
            {
                std::lock_guard<std::mutex> lock(depthAlignMtx_);
                if(depthAlignQueue_.empty() && depthAlignInFlight_.load() == 0) {
                    depthAlignDrainCv_.notify_all();
                }
            }
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
        const size_t queueMax = static_cast<size_t>(std::max(1, cfg_.save.h265QueueCapacity));
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
                h265RawInputFormatForColorProfile(itParams->second.format),
                cfg_.save,
                queueMax);
            encoder->start();
            h265Encoders_[sn] = std::move(encoder);
        }
        h265EncodingActive_.store(!h265Encoders_.empty());
    }

    void stopH265Encoders() {
        std::vector<H265Encoder *> encoders;
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            encoders.reserve(h265Encoders_.size());
            for(auto &kv: h265Encoders_) {
                if(kv.second) {
                    encoders.push_back(kv.second.get());
                }
            }
        }
        for(auto *encoder: encoders) {
            if(encoder) {
                encoder->requestStop();
            }
        }
        h265StoppingEncoders_.store(encoders.size());
        for(auto *encoder: encoders) {
            if(encoder) {
                encoder->join();
                h265StoppingEncoders_.fetch_sub(1);
            }
        }
        h265StoppingEncoders_.store(0);
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            h265Encoders_.clear();
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

    void enqueueFisheyeH265Frame(size_t cameraIdx, const std::string &frameIndex, uint64_t tsUs, cv::Mat frame) {
        if(frame.empty()) {
            return;
        }
        if(frame.type() == CV_8UC4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        }
        else if(frame.type() == CV_8UC1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }
        if(frame.empty() || frame.type() != CV_8UC3 || frame.cols <= 0 || frame.rows <= 0) {
            return;
        }

        const std::string encoderKey = "fisheye:" + std::to_string(cameraIdx);
        H265Encoder *encoder = nullptr;
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            auto it = h265Encoders_.find(encoderKey);
            if(it != h265Encoders_.end()) {
                encoder = it->second.get();
            }
            else {
                const std::string cameraKey = "fisheye_" + std::to_string(cameraIdx);
                const fs::path outPath = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx) / "RGB" / h265OutputFileName(cfg_.save);
                const int fps = cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_);
                const size_t queueMax = static_cast<size_t>(std::max(1, cfg_.save.h265QueueCapacity));
                auto newEncoder = std::make_unique<H265Encoder>(
                    outPath,
                    frame.cols,
                    frame.rows,
                    fps,
                    h265ThreadsForCamera(cameraKey, cameraKey),
                    OB_FORMAT_BGR,
                    cfg_.save,
                    queueMax);
                newEncoder->start();
                encoder = newEncoder.get();
                h265Encoders_[encoderKey] = std::move(newEncoder);
                h265EncodingActive_.store(true);
            }
        }
        if(encoder) {
            encoder->enqueue(frameIndex, tsUs, std::move(frame));
        }
    }

    void startDepthFfv1Encoders(const SessionState &session) {
        stopDepthFfv1Encoders();
        if(!depthOutputIsFfv1Mkv(cfg_.save) || !session.saveDepthTimesteps) {
            return;
        }

        std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
        const size_t queueMax = static_cast<size_t>(std::max(1, cfg_.save.depthFfv1QueueCapacity));
        for(const auto &sn: session.deviceSns) {
            const auto itBuf = session.buffers.find(sn);
            if(itBuf == session.buffers.end()) {
                continue;
            }
            const auto itParams = itBuf->second.params.find(CollectDataType::Depth);
            if(itParams == itBuf->second.params.end() || !itParams->second.valid
               || itParams->second.width <= 0 || itParams->second.height <= 0) {
                continue;
            }
            int outputW = itParams->second.width;
            int outputH = itParams->second.height;
            const auto itRgbParams = itBuf->second.params.find(CollectDataType::RGB);
            if(itRgbParams != itBuf->second.params.end() && itRgbParams->second.valid
               && itRgbParams->second.width > 0 && itRgbParams->second.height > 0) {
                outputW = itRgbParams->second.width;
                outputH = itRgbParams->second.height;
            }
            const std::string &camKey = itBuf->second.camKey;
            const fs::path outPath = session.dest / camKey / dataTypeLabel(CollectDataType::Depth) / depthFfv1OutputFileName();
            auto encoder = std::make_unique<DepthFfv1Encoder>(
                outPath,
                outputW,
                outputH,
                itParams->second.fps > 0 ? itParams->second.fps : (cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_)),
                queueMax);
            encoder->start();
            depthFfv1Encoders_[sn] = std::move(encoder);
        }
        depthFfv1EncodingActive_.store(!depthFfv1Encoders_.empty());
    }

    void stopDepthFfv1Encoders() {
        std::vector<DepthFfv1Encoder *> encoders;
        {
            std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
            encoders.reserve(depthFfv1Encoders_.size());
            for(auto &kv: depthFfv1Encoders_) {
                if(kv.second) {
                    encoders.push_back(kv.second.get());
                }
            }
        }
        for(auto *encoder: encoders) {
            if(encoder) {
                encoder->requestStop();
            }
        }
        depthFfv1StoppingEncoders_.store(encoders.size());
        for(auto *encoder: encoders) {
            if(encoder) {
                encoder->join();
                depthFfv1StoppingEncoders_.fetch_sub(1);
            }
        }
        depthFfv1StoppingEncoders_.store(0);
        {
            std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
            depthFfv1Encoders_.clear();
        }
        depthFfv1EncodingActive_.store(false);
    }

    void stopVideoEncoders() {
        std::vector<H265Encoder *> h265Encoders;
        std::vector<DepthFfv1Encoder *> depthFfv1Encoders;
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            h265Encoders.reserve(h265Encoders_.size());
            for(auto &kv: h265Encoders_) {
                if(kv.second) {
                    h265Encoders.push_back(kv.second.get());
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
            depthFfv1Encoders.reserve(depthFfv1Encoders_.size());
            for(auto &kv: depthFfv1Encoders_) {
                if(kv.second) {
                    depthFfv1Encoders.push_back(kv.second.get());
                }
            }
        }
        const auto stopStart = std::chrono::steady_clock::now();
        if(!h265Encoders.empty() || !depthFfv1Encoders.empty()) {
            std::cerr << "[collection] stopping video encoders h265=" << h265Encoders.size()
                      << " mkv=" << depthFfv1Encoders.size() << std::endl;
        }

        for(auto *encoder: h265Encoders) {
            if(encoder) {
                encoder->requestStop();
            }
        }
        for(auto *encoder: depthFfv1Encoders) {
            if(encoder) {
                encoder->requestStop();
            }
        }

        h265StoppingEncoders_.store(h265Encoders.size());
        depthFfv1StoppingEncoders_.store(depthFfv1Encoders.size());
        for(auto *encoder: h265Encoders) {
            if(encoder) {
                encoder->join();
                h265StoppingEncoders_.fetch_sub(1);
            }
        }
        for(auto *encoder: depthFfv1Encoders) {
            if(encoder) {
                encoder->join();
                depthFfv1StoppingEncoders_.fetch_sub(1);
            }
        }
        h265StoppingEncoders_.store(0);
        depthFfv1StoppingEncoders_.store(0);
        {
            std::lock_guard<std::mutex> lock(h265Mtx_);
            h265Encoders_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
            depthFfv1Encoders_.clear();
        }
        h265EncodingActive_.store(false);
        depthFfv1EncodingActive_.store(false);
        if(!h265Encoders.empty() || !depthFfv1Encoders.empty()) {
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - stopStart).count();
            std::cerr << "[collection] video encoders stopped in "
                      << (static_cast<double>(elapsedMs) / 1000.0) << "s" << std::endl;
        }
    }

    bool enqueueDepthFfv1Frame(const std::string &sn, const std::string &frameIndex, uint64_t tsUs, cv::Mat frame) {
        DepthFfv1Encoder *encoder = nullptr;
        {
            std::lock_guard<std::mutex> lock(depthFfv1Mtx_);
            auto it = depthFfv1Encoders_.find(sn);
            if(it != depthFfv1Encoders_.end()) {
                encoder = it->second.get();
            }
        }
        if(encoder) {
            return encoder->enqueue(frameIndex, tsUs, std::move(frame));
        }
        return false;
    }

    void startCoordinatorThread() {
        if(coordinatorThread_.joinable()) {
            coordinatorThread_.join();
        }
        coordinatorThread_ = std::thread([this]() {
            coordinatorLoop();
            waitDepthAlignQueueIdle();
            stopDepthAlignWorkers();
            stopVideoEncoders();
        });
    }

    void joinCoordinatorThreadIfPossible() {
        if(coordinatorThread_.joinable()) {
            coordinatorThread_.join();
        }
    }

    size_t coordQueuedCountLocked() const {
        return coordRecordQueue_.size() + coordFisheyeQueue_.size() + coordEgoQueue_.size() + coordTouchQueue_.size();
    }

    void enqueueProcessedRecord(ProcessedRecord &&item) {
        {
            std::unique_lock<std::mutex> lock(coordMtx_);
            coordCv_.wait(lock, [&]() {
                return !session_.active || stopping_.load() || coordQueuedCountLocked() < coordQueueMax_;
            });
            if(!session_.active || stopping_.load()) {
                return;
            }
            coordRecordQueue_.push_back(std::move(item));
        }
        coordCv_.notify_one();
    }

    void enqueueFisheyeFrameSet(FisheyeFrameSet &&sample) {
        {
            std::unique_lock<std::mutex> lock(coordMtx_);
            coordCv_.wait(lock, [&]() {
                return !session_.active || stopping_.load() || coordQueuedCountLocked() < coordQueueMax_;
            });
            if(!session_.active || stopping_.load()) {
                return;
            }
            coordFisheyeQueue_.push_back(std::move(sample));
        }
        coordCv_.notify_one();
    }

    void enqueueEgoFrame(EgoFrame &&sample) {
        {
            std::unique_lock<std::mutex> lock(coordMtx_);
            coordCv_.wait(lock, [&]() {
                return !session_.active || stopping_.load() || coordQueuedCountLocked() < coordQueueMax_;
            });
            if(!session_.active || stopping_.load()) {
                return;
            }
            coordEgoQueue_.push_back(std::move(sample));
        }
        coordCv_.notify_one();
    }

    void enqueueTouchSample(const std::string &streamId, TactileSample &&sample) {
        {
            std::unique_lock<std::mutex> lock(coordMtx_);
            coordCv_.wait(lock, [&]() {
                return !session_.active || stopping_.load() || coordQueuedCountLocked() < coordQueueMax_;
            });
            if(!session_.active || stopping_.load()) {
                return;
            }
            TouchQueuedSample queued;
            queued.streamId = streamId;
            queued.sample = std::move(sample);
            coordTouchQueue_.push_back(std::move(queued));
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

    void notifyEgoEos() {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return;
        }
        session_.egoEos = true;
        coordCv_.notify_one();
    }

    void notifyTouchEos() {
        std::lock_guard<std::mutex> lock(coordMtx_);
        if(!session_.active) {
            return;
        }
        session_.touchEos = true;
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
                const uint64_t requiredTailUs = saturatingAddUs(centerUs, sessionCrossTypeMaxAbsDiffUs(session_.stepUs));
                if(!itStream->second.eos && itStream->second.maxTsUs < requiredTailUs) {
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
        if(session_.saveTouch) {
            const uint64_t requiredTouchTailUs = saturatingAddUs(centerUs, session_.maxAbsDiffUs);
            for(const auto &stream: session_.touchStreams) {
                auto itSamples = session_.touchSamples.find(stream.streamId);
                if(itSamples == session_.touchSamples.end() || itSamples->second.empty()) {
                    if(!session_.touchEos) {
                        return false;
                    }
                    continue;
                }
                if(!session_.touchEos && itSamples->second.back().representativeTimestampUs < requiredTouchTailUs) {
                    return false;
                }
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

    size_t pickNearestEgoIndexLocked(uint64_t centerUs) const {
        return findNearestEgoFrameIndex(session_.egoFrames, centerUs);
    }

    size_t pickNearestTouchIndexLocked(const std::string &streamId, uint64_t centerUs) const {
        auto itSamples = session_.touchSamples.find(streamId);
        if(itSamples == session_.touchSamples.end() || itSamples->second.empty()) {
            return 0;
        }
        const auto &samples = itSamples->second;
        auto absDiff = [](uint64_t a, uint64_t b) {
            return a > b ? (a - b) : (b - a);
        };
        auto it = std::lower_bound(samples.begin(),
                                   samples.end(),
                                   centerUs,
                                   [](const TactileSample &sample, uint64_t ts) {
                                       return sample.representativeTimestampUs < ts;
                                   });
        size_t cand0 = (it == samples.end())
            ? (samples.size() - 1)
            : static_cast<size_t>(std::distance(samples.begin(), it));
        size_t cand1 = cand0 > 0 ? cand0 - 1 : cand0;
        const uint64_t d0 = absDiff(samples[cand0].representativeTimestampUs, centerUs);
        const uint64_t d1 = absDiff(samples[cand1].representativeTimestampUs, centerUs);
        return d1 <= d0 ? cand1 : cand0;
    }

    bool pickNearestTouchSampleLocked(const std::string &streamId, uint64_t centerUs, size_t &picked, uint64_t &absDiffUs) const {
        picked = 0;
        absDiffUs = 0;
        auto itSamples = session_.touchSamples.find(streamId);
        if(itSamples == session_.touchSamples.end() || itSamples->second.empty()) {
            return false;
        }
        const auto &samples = itSamples->second;
        picked = pickNearestTouchIndexLocked(streamId, centerUs);
        const uint64_t ts = samples[picked].representativeTimestampUs;
        absDiffUs = ts > centerUs ? (ts - centerUs) : (centerUs - ts);
        return absDiffUs <= session_.maxAbsDiffUs;
    }

    bool touchTailReadyLocked(uint64_t targetUs) const {
        if(!session_.saveTouch) {
            return true;
        }
        const uint64_t requiredTailUs = saturatingAddUs(targetUs, session_.maxAbsDiffUs);
        for(const auto &stream: session_.touchStreams) {
            auto itSamples = session_.touchSamples.find(stream.streamId);
            if(itSamples == session_.touchSamples.end() || itSamples->second.empty()) {
                if(!session_.touchEos) {
                    return false;
                }
                continue;
            }
            if(!session_.touchEos && itSamples->second.back().representativeTimestampUs < requiredTailUs) {
                return false;
            }
        }
        return true;
    }

    bool pickNearestMappedEgoFrameLocked(uint64_t centerUs,
                                         size_t &picked,
                                         int &videoFrameIndex,
                                         bool &hasPendingVideoCandidate) const {
        picked = 0;
        videoFrameIndex = -1;
        hasPendingVideoCandidate = false;
        if(session_.egoFrames.empty()) {
            return false;
        }
        auto absDiff = [](uint64_t a, uint64_t b) {
            return a > b ? (a - b) : (b - a);
        };
        uint64_t bestDiff = std::numeric_limits<uint64_t>::max();
        for(size_t i = 0; i < session_.egoFrames.size(); ++i) {
            const auto &frame = session_.egoFrames[i];
            const uint64_t diff = absDiff(frame.refTimestampUs, centerUs);
            if(diff > session_.maxAbsDiffUs) {
                continue;
            }
            const int candidateVideoFrame = egoVideoFrameIndexForSourceFrameLocked(frame);
            if(candidateVideoFrame < 0) {
                hasPendingVideoCandidate = true;
                continue;
            }
            if(diff < bestDiff || (diff == bestDiff && candidateVideoFrame < videoFrameIndex)) {
                picked = i;
                videoFrameIndex = candidateVideoFrame;
                bestDiff = diff;
            }
        }
        return videoFrameIndex >= 0;
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
        while(session_.egoFrames.size() > 1 && session_.egoFrames[1].refTimestampUs <= centerUs) {
            session_.egoFrames.pop_front();
        }
        for(auto &kv: session_.touchSamples) {
            auto &samples = kv.second;
            while(samples.size() > 1 && samples[1].representativeTimestampUs <= centerUs) {
                samples.pop_front();
            }
        }
    }

    void appendTimestampRowLocked(const std::vector<std::string> &row) {
        if(!session_.timestampsOpen) {
            return;
        }
        writeCsvRow(session_.timestampsOfs, row);
    }

    int egoVideoFrameIndexForSourceFrameLocked(const EgoFrame &frame) const {
        if(frame.videoFrameIndex >= 0) {
            return frame.videoFrameIndex;
        }
        return frame.sourceFrameIndex >= 0 ? egoRecorder_.videoFrameIndexForSourceFrame(frame.sourceFrameIndex) : -1;
    }

    void noteEgoAlignedFrameLocked(const EgoFrame &frame) {
        if(frame.sourceFrameIndex >= 0) {
            session_.egoAlignedSourceFrameIndices.insert(frame.sourceFrameIndex);
        }
        else if(frame.refTimestampUs > 0) {
            session_.egoAlignedRefTimestamps.insert(frame.refTimestampUs);
        }
    }

    void appendEgoAlignedTimestampRowLocked(const std::string &frameIndex, const EgoFrame *frame, int egoVideoFrameIndex) {
        if(!frame || egoVideoFrameIndex < 0) {
            session_.egoMissingAlignedRows++;
            if(!session_.egoAlignedTimestampsOpen) {
                return;
            }
            writeCsvRow(session_.egoAlignedTimestampsOfs,
                        { frameIndex, "", "", "", "", "", "", "", "", "", "", "", "" });
            return;
        }
        noteEgoAlignedFrameLocked(*frame);
        if(!session_.egoAlignedTimestampsOpen) {
            return;
        }
        writeCsvRow(session_.egoAlignedTimestampsOfs,
                    { frameIndex,
                      std::to_string(egoVideoFrameIndex),
                      std::to_string(frame->sourceFrameIndex),
                      std::to_string(frame->refTimestampUs),
                      std::to_string(frame->rawRefTimestampUs),
                      std::to_string(frame->timeCalibrationOffsetUs),
                      frame->timeCalibrationStatus,
                      std::to_string(frame->softAlignOffsetUs),
                      frame->timestampValidation,
                      std::to_string(frame->rgbTimestampUs),
                      std::to_string(frame->picoFrameTimestampNs),
                      std::to_string(frame->xrHeadTimestampUs),
                      std::to_string(frame->gazeTimestampUs) });
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

    bool tryFinalizeOneMultiviewSlotLocked(std::vector<DepthAlignTask> &depthAlignTasks) {
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
        const bool egoDropWaitElapsed = session_.saveEgo && egoSlotDropWaitElapsedLocked(centerUs, itRefState->second);
        if(!egoCanResolveOrDropSlotLocked(centerUs, itRefState->second)) {
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
        size_t pickedEgoIdx = 0;
        bool hasPickedEgo = false;
        int pickedEgoVideoFrameIndex = -1;
        if(session_.saveEgo) {
            bool hasPendingEgoVideoCandidate = false;
            if(pickNearestMappedEgoFrameLocked(centerUs, pickedEgoIdx, pickedEgoVideoFrameIndex, hasPendingEgoVideoCandidate)) {
                const auto &egoFrame = session_.egoFrames[pickedEgoIdx];
                hasPickedEgo = true;
                tsMin = std::min(tsMin, egoFrame.refTimestampUs);
                tsMax = std::max(tsMax, egoFrame.refTimestampUs);
            }
            else if(hasPendingEgoVideoCandidate && !session_.egoEos && !egoDropWaitElapsed) {
                return false;
            }
            else {
                fullThis = false;
            }
        }
        std::vector<PickedTouchSample> pickedTouches;
        if(session_.saveTouch) {
            pickedTouches.reserve(session_.touchStreams.size());
            for(const auto &stream: session_.touchStreams) {
                PickedTouchSample pickedTouch;
                pickedTouch.streamId = stream.streamId;
                if(pickNearestTouchSampleLocked(stream.streamId, centerUs, pickedTouch.sampleIndex, pickedTouch.absDiffUs)) {
                    const auto &touchSample = session_.touchSamples.at(stream.streamId)[pickedTouch.sampleIndex];
                    pickedTouch.hasSample = true;
                    tsMin = std::min(tsMin, touchSample.representativeTimestampUs);
                    tsMax = std::max(tsMax, touchSample.representativeTimestampUs);
                }
                else {
                    fullThis = false;
                }
                pickedTouches.push_back(std::move(pickedTouch));
            }
        }

        const size_t refIndex = session_.alignedRef;
        session_.alignedRef++;
        auto dropCurrentSlot = [&]() {
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
        };
        if(!fullThis) {
            return dropCurrentSlot();
        }

        const size_t outIdx = session_.nextFrameIndex;
        const std::string frameIndex = formatFrameIndex(outIdx);

        session_.fullAligned++;
        if(tsMax >= tsMin) {
            const double diffMs = static_cast<double>(tsMax - tsMin) / 1000.0;
            if(diffMs > session_.maxDiffMs) {
                session_.maxDiffMs = diffMs;
            }
        }

        session_.nextFrameIndex++;
        session_.alignedCenters.push_back(centerUs);
        hasData_.store(true);

        std::vector<std::string> row;
        row.reserve(2 + session_.deviceSns.size() * 2 + session_.fisheyeCameraCount
                    + (session_.saveEgo ? 2 : 0) + (session_.saveTouch ? session_.touchStreams.size() * 3 : 0) + 2);
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
                    const auto itRgbParam = buf.params.find(CollectDataType::RGB);
                    const int rgbW = (itRgbParam != buf.params.end() && itRgbParam->second.valid) ? itRgbParam->second.width : 0;
                    const int rgbH = (itRgbParam != buf.params.end() && itRgbParam->second.valid) ? itRgbParam->second.height : 0;
                    const auto rgbDepthParam = buf.rgbDepthParam;
                    const bool rgbDepthParamValid = buf.rgbDepthParamValid;
                    const float valueScale = packet.valueScale;
                    if(cfg_.save.saveRaw) {
                        const fs::path rawOutPath = session_.dest / buf.camKey / depthRawDirName() / (frameIndex + ".raw");
                        cv::Mat rawFrame = packet.frame;
                        enqueueWriteTask(WriteTask{ [rawFrame = std::move(rawFrame), rawOutPath]() mutable {
                            saveMatToRawFile(rawFrame, rawOutPath);
                        } });
                    }
                    depthAlignTasks.push_back(DepthAlignTask{
                        sn,
                        buf.camKey,
                        frameIndex,
                        packet.tsUs,
                        packet.frame,
                        session_.dest / buf.camKey / dataTypeLabel(t) / (frameIndex + ".png"),
                        cfg_.save,
                        rgbDepthParam,
                        rgbDepthParamValid,
                        rgbW,
                        rgbH,
                        valueScale,
                        depthOutputIsFfv1Mkv(cfg_.save)
                    });
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
                        cv::Mat frameImg = frame.bgr;
                        enqueueFisheyeH265Frame(cameraIdx, frameIndex, frame.captureTimestampUs, std::move(frameImg));
                        if(!session_.wroteFisheyeCameraParams[cameraIdx] && !frame.bgr.empty()) {
                            const fs::path cameraDir = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx);
                            writeFisheyeCameraParamsJson(cameraDir, cameraIdx, frame.bgr, cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_), cfg_.save);
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

        if(session_.saveEgo) {
            if(hasPickedEgo && pickedEgoIdx < session_.egoFrames.size()) {
                const auto &egoFrame = session_.egoFrames[pickedEgoIdx];
                row.push_back(std::to_string(pickedEgoVideoFrameIndex));
                row.push_back(std::to_string(egoFrame.refTimestampUs));
                allTsMin = std::min(allTsMin, egoFrame.refTimestampUs);
                allTsMax = std::max(allTsMax, egoFrame.refTimestampUs);
                allTsCount++;
                appendEgoAlignedTimestampRowLocked(frameIndex, &egoFrame, pickedEgoVideoFrameIndex);
            }
            else {
                row.emplace_back();
                row.emplace_back();
                appendEgoAlignedTimestampRowLocked(frameIndex, nullptr, -1);
            }
        }

        if(session_.saveTouch) {
            for(const auto &pickedTouch: pickedTouches) {
                auto itSamples = session_.touchSamples.find(pickedTouch.streamId);
                if(pickedTouch.hasSample && itSamples != session_.touchSamples.end()
                   && pickedTouch.sampleIndex < itSamples->second.size()) {
                    const auto &touchSample = itSamples->second[pickedTouch.sampleIndex];
                    row.push_back(std::to_string(touchSample.sequence));
                    row.push_back(std::to_string(touchSample.representativeTimestampUs));
                    row.push_back(std::to_string(pickedTouch.absDiffUs));
                    allTsMin = std::min(allTsMin, touchSample.representativeTimestampUs);
                    allTsMax = std::max(allTsMax, touchSample.representativeTimestampUs);
                    allTsCount++;
                }
                else {
                    row.emplace_back();
                    row.emplace_back();
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

    bool tryFinalizeEgoOnlySlotLocked() {
        if(!session_.saveEgo || session_.egoFrames.empty()) {
            return false;
        }
        if(!session_.egoOnlyTargetInit) {
            session_.egoOnlyNextTargetUs = session_.egoFrames.front().refTimestampUs;
            session_.egoOnlyTargetInit = true;
        }
        const uint64_t targetUs = session_.egoOnlyNextTargetUs;
        if(!session_.egoEos && session_.egoFrames.back().refTimestampUs < targetUs) {
            return false;
        }
        if(session_.saveFisheye) {
            if(session_.fisheyeSets.empty()) {
                if(!session_.fisheyeEos) {
                    return false;
                }
            }
            else if(!session_.fisheyeEos && session_.fisheyeSets.back().representativeTimestampUs < targetUs) {
                return false;
            }
        }
        if(!touchTailReadyLocked(targetUs)) {
            return false;
        }

        const size_t egoIdx = pickNearestEgoIndexLocked(targetUs);
        if(egoIdx >= session_.egoFrames.size()) {
            session_.egoOnlyNextTargetUs += session_.stepUs;
            return true;
        }
        const auto &egoFrame = session_.egoFrames[egoIdx];
        auto advanceEgoOnlyTarget = [&]() {
            session_.egoOnlyNextTargetUs += session_.stepUs;
            if(egoIdx < session_.egoFrames.size()) {
                session_.egoFrames.erase(session_.egoFrames.begin() + egoIdx);
            }
            while(session_.egoFrames.size() > 1 && session_.egoFrames[1].refTimestampUs <= targetUs) {
                session_.egoFrames.pop_front();
            }
            return true;
        };
        if(session_.hasLastEmittedEgoTs && session_.lastEmittedEgoTs == egoFrame.refTimestampUs) {
            return advanceEgoOnlyTarget();
        }
        const int egoVideoFrameIndex = egoVideoFrameIndexForSourceFrameLocked(egoFrame);
        if(egoVideoFrameIndex < 0) {
            if(!session_.egoEos && egoIdx + 1 >= session_.egoFrames.size()) {
                return false;
            }
            return advanceEgoOnlyTarget();
        }
        std::vector<PickedTouchSample> pickedTouches;
        if(session_.saveTouch) {
            pickedTouches.reserve(session_.touchStreams.size());
            for(const auto &stream: session_.touchStreams) {
                PickedTouchSample pickedTouch;
                pickedTouch.streamId = stream.streamId;
                if(!pickNearestTouchSampleLocked(stream.streamId, targetUs, pickedTouch.sampleIndex, pickedTouch.absDiffUs)) {
                    return advanceEgoOnlyTarget();
                }
                pickedTouch.hasSample = true;
                pickedTouches.push_back(std::move(pickedTouch));
            }
        }

        const size_t outIdx = session_.nextFrameIndex;
        const std::string frameIndex = formatFrameIndex(outIdx);
        appendEgoAlignedTimestampRowLocked(frameIndex, &egoFrame, egoVideoFrameIndex);
        session_.nextFrameIndex++;
        session_.refFrameCount++;
        if(session_.firstRefTs == 0) {
            session_.firstRefTs = egoFrame.refTimestampUs;
        }
        session_.lastRefTs = egoFrame.refTimestampUs;
        std::vector<std::string> row;
        row.reserve(2 + session_.fisheyeCameraCount + 4 + (session_.saveTouch ? session_.touchStreams.size() * 3 : 0));
        row.push_back(frameIndex);
        row.push_back(std::to_string(egoFrame.refTimestampUs));

        uint64_t rowTsMin = egoFrame.refTimestampUs;
        uint64_t rowTsMax = egoFrame.refTimestampUs;
        size_t rowTsCount = 1;

        if(session_.saveFisheye) {
            if(!session_.fisheyeSets.empty()) {
                const size_t fisheyeIdx = pickNearestFisheyeIndexLocked(targetUs);
                const auto &sample = session_.fisheyeSets[fisheyeIdx];
                for(size_t cameraIdx = 0; cameraIdx < session_.fisheyeCameraCount; ++cameraIdx) {
                    if(cameraIdx < sample.frames.size()) {
                        const auto &frame = sample.frames[cameraIdx];
                        row.push_back(std::to_string(frame.captureTimestampUs));
                        rowTsMin = std::min(rowTsMin, frame.captureTimestampUs);
                        rowTsMax = std::max(rowTsMax, frame.captureTimestampUs);
                        rowTsCount++;
                        cv::Mat frameImg = frame.bgr;
                        enqueueFisheyeH265Frame(cameraIdx, frameIndex, frame.captureTimestampUs, std::move(frameImg));
                        if(!session_.wroteFisheyeCameraParams[cameraIdx] && !frame.bgr.empty()) {
                            const fs::path cameraDir = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx);
                            writeFisheyeCameraParamsJson(cameraDir, cameraIdx, frame.bgr, cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_), cfg_.save);
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

        row.push_back(std::to_string(egoVideoFrameIndex));
        row.push_back(std::to_string(egoFrame.refTimestampUs));
        if(session_.saveTouch) {
            for(const auto &pickedTouch: pickedTouches) {
                auto itSamples = session_.touchSamples.find(pickedTouch.streamId);
                if(pickedTouch.hasSample && itSamples != session_.touchSamples.end()
                   && pickedTouch.sampleIndex < itSamples->second.size()) {
                    const auto &touchSample = itSamples->second[pickedTouch.sampleIndex];
                    row.push_back(std::to_string(touchSample.sequence));
                    row.push_back(std::to_string(touchSample.representativeTimestampUs));
                    row.push_back(std::to_string(pickedTouch.absDiffUs));
                    rowTsMin = std::min(rowTsMin, touchSample.representativeTimestampUs);
                    rowTsMax = std::max(rowTsMax, touchSample.representativeTimestampUs);
                    rowTsCount++;
                }
                else {
                    row.emplace_back();
                    row.emplace_back();
                    row.emplace_back();
                }
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
        session_.hasLastEmittedEgoTs = true;
        session_.lastEmittedEgoTs = egoFrame.refTimestampUs;
        session_.alignedCenters.push_back(egoFrame.refTimestampUs);

        session_.egoOnlyNextTargetUs += session_.stepUs;
        while(session_.egoFrames.size() > 1 && session_.egoFrames[1].refTimestampUs <= targetUs) {
            session_.egoFrames.pop_front();
        }
        if(session_.saveFisheye) {
            while(session_.fisheyeSets.size() > 1 && session_.fisheyeSets[1].representativeTimestampUs <= targetUs) {
                session_.fisheyeSets.pop_front();
            }
        }
        for(auto &kv: session_.touchSamples) {
            auto &samples = kv.second;
            while(samples.size() > 1 && samples[1].representativeTimestampUs <= targetUs) {
                samples.pop_front();
            }
        }
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
        if(!touchTailReadyLocked(targetUs)) {
            return false;
        }

        const size_t idx = pickNearestFisheyeIndexLocked(targetUs);
        const auto &sample = session_.fisheyeSets[idx];
        auto advanceFisheyeOnlyTarget = [&]() {
            session_.fisheyeOnlyNextTargetUs += session_.stepUs;
            while(session_.fisheyeSets.size() > 1 && session_.fisheyeSets[1].representativeTimestampUs <= targetUs) {
                session_.fisheyeSets.pop_front();
            }
            for(auto &kv: session_.touchSamples) {
                auto &samples = kv.second;
                while(samples.size() > 1 && samples[1].representativeTimestampUs <= targetUs) {
                    samples.pop_front();
                }
            }
            return true;
        };
        std::vector<PickedTouchSample> pickedTouches;
        if(session_.saveTouch) {
            pickedTouches.reserve(session_.touchStreams.size());
            for(const auto &stream: session_.touchStreams) {
                PickedTouchSample pickedTouch;
                pickedTouch.streamId = stream.streamId;
                if(!pickNearestTouchSampleLocked(stream.streamId, targetUs, pickedTouch.sampleIndex, pickedTouch.absDiffUs)) {
                    return advanceFisheyeOnlyTarget();
                }
                pickedTouch.hasSample = true;
                pickedTouches.push_back(std::move(pickedTouch));
            }
        }
        if(!session_.hasLastEmittedFisheyeTs || session_.lastEmittedFisheyeTs != sample.representativeTimestampUs) {
            const size_t outIdx = session_.nextFrameIndex++;
            const std::string frameIndex = formatFrameIndex(outIdx);
            std::vector<std::string> row;
            row.reserve(2 + session_.fisheyeCameraCount + (session_.saveTouch ? session_.touchStreams.size() * 3 : 0) + 2);
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
                    cv::Mat frameImg = frame.bgr;
                    enqueueFisheyeH265Frame(cameraIdx, frameIndex, frame.captureTimestampUs, std::move(frameImg));
                    if(!session_.wroteFisheyeCameraParams[cameraIdx] && !frame.bgr.empty()) {
                        const fs::path cameraDir = session_.dest / "fisheye" / fisheyeCameraDirName(cameraIdx);
                        writeFisheyeCameraParamsJson(cameraDir, cameraIdx, frame.bgr, cfg_.collectFps > 0 ? cfg_.collectFps : std::max(1, uiFpsFallback_), cfg_.save);
                        session_.wroteFisheyeCameraParams[cameraIdx] = true;
                    }
                }
                else {
                    row.emplace_back();
                }
            }
            if(session_.saveTouch) {
                for(const auto &pickedTouch: pickedTouches) {
                    auto itSamples = session_.touchSamples.find(pickedTouch.streamId);
                    if(pickedTouch.hasSample && itSamples != session_.touchSamples.end()
                       && pickedTouch.sampleIndex < itSamples->second.size()) {
                        const auto &touchSample = itSamples->second[pickedTouch.sampleIndex];
                        row.push_back(std::to_string(touchSample.sequence));
                        row.push_back(std::to_string(touchSample.representativeTimestampUs));
                        row.push_back(std::to_string(pickedTouch.absDiffUs));
                        rowTsMin = std::min(rowTsMin, touchSample.representativeTimestampUs);
                        rowTsMax = std::max(rowTsMax, touchSample.representativeTimestampUs);
                        rowTsCount++;
                    }
                    else {
                        row.emplace_back();
                        row.emplace_back();
                        row.emplace_back();
                    }
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

        return advanceFisheyeOnlyTarget();
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
        if(!coordRecordQueue_.empty() || !coordFisheyeQueue_.empty() || !coordEgoQueue_.empty() || !coordTouchQueue_.empty()) {
            return false;
        }
        if(!session_.egoPendingSoftAlignFrames.empty()) {
            flushPendingEgoSoftAlignLocked();
            if(!session_.egoPendingSoftAlignFrames.empty() && session_.egoEos && session_.multiviewEos && session_.firstRefTs == 0) {
                flushPendingEgoWithoutGlobalAnchorLocked();
            }
            if(!session_.egoPendingSoftAlignFrames.empty()) {
                return false;
            }
        }
        if(multiviewEnabled_) {
            auto itRefStreams = session_.streams.find(session_.refSn);
            const bool refEmpty = (itRefStreams == session_.streams.end())
                                  || (itRefStreams->second.find(refType_) == itRefStreams->second.end())
                                  || itRefStreams->second.at(refType_).committed.empty();
            if(!session_.multiviewEos || !refEmpty || hasPendingBySeqLocked()) {
                return false;
            }
            if(session_.saveFisheye && !session_.fisheyeEos) {
                return false;
            }
            if(session_.saveEgo && !session_.egoEos) {
                return false;
            }
            if(session_.saveTouch && !session_.touchEos) {
                return false;
            }
        }
        else {
            if(session_.saveFisheye && !session_.fisheyeEos) {
                return false;
            }
            if(session_.saveEgo && !session_.egoEos) {
                return false;
            }
            if(session_.saveTouch && !session_.touchEos) {
                return false;
            }
            if(session_.fisheyeOnlyTargetInit && !session_.fisheyeSets.empty()
               && session_.fisheyeOnlyNextTargetUs <= session_.fisheyeSets.back().representativeTimestampUs) {
                return false;
            }
            if(session_.egoOnlyTargetInit && !session_.egoFrames.empty()
               && session_.egoOnlyNextTargetUs <= session_.egoFrames.back().refTimestampUs) {
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
                       || !coordEgoQueue_.empty()
                       || !coordTouchQueue_.empty()
                       || (session_.active && (session_.multiviewEos || session_.fisheyeEos))
                       || (session_.active && session_.egoEos)
                       || (session_.active && session_.touchEos)
                       || stopping_.load();
            });
            if(!session_.active && coordRecordQueue_.empty() && coordFisheyeQueue_.empty()
               && coordEgoQueue_.empty() && coordTouchQueue_.empty()) {
                return;
            }

            while(!coordRecordQueue_.empty()) {
                commitProcessedRecordLocked(std::move(coordRecordQueue_.front()));
                coordRecordQueue_.pop_front();
                coordCv_.notify_all();
            }
            flushPendingEgoSoftAlignLocked();
            while(!coordFisheyeQueue_.empty()) {
                session_.fisheyeSets.push_back(std::move(coordFisheyeQueue_.front()));
                coordFisheyeQueue_.pop_front();
                session_.fisheyeCapturedSets++;
                coordCv_.notify_all();
            }
            while(!coordEgoQueue_.empty()) {
                acceptEgoFrameLocked(std::move(coordEgoQueue_.front()));
                coordEgoQueue_.pop_front();
                coordCv_.notify_all();
            }
            while(!coordTouchQueue_.empty()) {
                commitTouchSampleLocked(std::move(coordTouchQueue_.front()));
                coordTouchQueue_.pop_front();
                coordCv_.notify_all();
            }
            flushPendingEgoSoftAlignLocked();

            bool progress = true;
            while(progress) {
                progress = false;
                std::vector<DepthAlignTask> deferredDepthAlignTasks;
                progress = flushEosSequenceGapsLocked() || progress;
                if(multiviewEnabled_) {
                    progress = tryFinalizeOneMultiviewSlotLocked(deferredDepthAlignTasks) || progress;
                }
                else if(session_.saveEgo) {
                    progress = tryFinalizeEgoOnlySlotLocked() || progress;
                }
                else if(session_.saveFisheye) {
                    progress = tryFinalizeFisheyeOnlySlotLocked() || progress;
                }
                progress = tryCompleteSessionLocked() || progress;
                const bool coordinatorDone = session_.coordinatorDone;
                if(!deferredDepthAlignTasks.empty()) {
                    lock.unlock();
                    for(auto &task: deferredDepthAlignTasks) {
                        enqueueDepthAlignTask(std::move(task));
                    }
                    deferredDepthAlignTasks.clear();
                    lock.lock();
                }
                if(coordinatorDone || session_.coordinatorDone) {
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
                    const std::string label = kv.second.camKey.empty() ? kv.first : kv.second.camKey;
                    out.emplace(label, kv.second.latestRgb);
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

        for(const auto t: { CollectDataType::RGB, CollectDataType::Depth }) {
            if(getFrameForType(t)) {
                noteCameraStreamFrame(deviceSn, t);
            }
        }

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
                        OBCameraParam rgbDepthParam{};
                        bool rgbDepthParamValid = false;
                        {
                            std::lock_guard<std::mutex> lock(mtx_);
                            auto it = buffers_.find(deviceSn);
                            if(it != buffers_.end() && it->second.rgbDepthParamValid) {
                                rgbDepthParam = it->second.rgbDepthParam;
                                rgbDepthParamValid = true;
                            }
                        }

                        cv::Mat alignedDepth;
                        uint64_t depthTs = 0;
                        float depthValueScaleMm = 0.0f;
                        if(rgbDepthParamValid) {
                            auto depthFrame = getFrameForType(CollectDataType::Depth);
                            cv::Mat depthRaw;
                            if(depthFrame && copyVideoFrameToRawMat(depthFrame, depthRaw, &depthValueScaleMm)
                               && depthRaw.type() == CV_16UC1 && depthValueScaleMm > 0.0f) {
                                cv::Mat zBuf;
                                alignDepthToRgbInto(depthRaw, depthValueScaleMm, rgbDepthParam,
                                                    previewBgr.cols, previewBgr.rows, alignedDepth, zBuf);
                                if(alignedDepth.empty() || alignedDepth.type() != CV_16UC1 || cv::countNonZero(alignedDepth) == 0) {
                                    alignedDepth.release();
                                    depthValueScaleMm = 0.0f;
                                }
                                else {
                                    depthTs = getFrameTimestampUs(depthFrame, true);
                                }
                            }
                        }

                        std::lock_guard<std::mutex> lock(mtx_);
                        auto it = buffers_.find(deviceSn);
                        if(it != buffers_.end()) {
                            it->second.latestRgb = std::move(previewBgr);
                            it->second.latestRgbTsUs = ts;
                            it->second.latestRgbSteady = std::chrono::steady_clock::now();
                            if(!alignedDepth.empty() && depthValueScaleMm > 0.0f) {
                                it->second.latestDepthAlignedRgb = std::move(alignedDepth);
                                it->second.latestDepthTsUs = depthTs;
                                it->second.latestDepthValueScaleMm = depthValueScaleMm;
                                it->second.latestDepthSteady = it->second.latestRgbSteady;
                            }
                            else {
                                it->second.latestDepthAlignedRgb.release();
                                it->second.latestDepthTsUs = 0;
                                it->second.latestDepthValueScaleMm = 0.0f;
                                it->second.latestDepthSteady = {};
                            }
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
            noteFisheyeFrameSet(*sample);
            enqueueFisheyeFrameSet(std::move(*sample));
        }
    }

    void egoRecordLoop() {
        while(!stopping_.load() && egoEnabled_) {
            EgoFrame frame;
            if(egoRecorder_.popFrame(frame, std::chrono::milliseconds(20))) {
                noteEgoFrame(frame);
                enqueueEgoFrame(std::move(frame));
                continue;
            }
            if(!recording_.load() && !egoRecorder_.isSessionActive() && !egoRecorder_.hasPendingFrames()) {
                break;
            }
        }
    }

    TouchRuntime *findTouchRuntime(const std::string &streamId) {
        for(auto &runtime: touchRuntimes_) {
            if(runtime.streamId == streamId) {
                return &runtime;
            }
        }
        return nullptr;
    }

    void touchRecordLoop(const std::string &streamId) {
        TouchRuntime *runtime = findTouchRuntime(streamId);
        if(!runtime || !runtime->recorder) {
            return;
        }
        uint64_t lastTouchTsUs = 0;
        while(recording_.load() && !stopping_.load() && touchEnabled_) {
            std::string err;
            auto sample = runtime->recorder->captureNext(&err);
            if(!sample) {
                if(!recording_.load() || stopping_.load()) {
                    break;
                }
                if(!err.empty()) {
                    std::cerr << "[collection] touch " << streamId << " capture error: " << err << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if(sample->representativeTimestampUs == 0 || sample->representativeTimestampUs == lastTouchTsUs) {
                continue;
            }
            lastTouchTsUs = sample->representativeTimestampUs;
            noteTouchSample(streamId, *sample);
            enqueueTouchSample(streamId, std::move(*sample));
        }
    }

    static void writeParamsJson(const fs::path &dest,
                                const std::unordered_map<std::string, DeviceBuffer> &buffers,
                                const std::vector<CollectDataType> &typesSaving,
                                int colorCloudRgbFrameOffset,
                                const SaveOptions &saveOptions) {
        cJSON *root = cJSON_CreateObject();
        cJSON *viewerObj = cJSON_CreateObject();
        cJSON_AddNumberToObject(viewerObj, "colorCloudRgbFrameOffset", colorCloudRgbFrameOffset);
        cJSON_AddItemToObject(root, "viewer", viewerObj);
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
                if(t == CollectDataType::RGB) {
                    jsonAddString(stObj, "storageEncoding", rgbStorageEncodingName(saveOptions));
                    if(saveOptions.rgbH265) {
                        const std::string fileName = h265OutputFileName(saveOptions);
                        jsonAddString(stObj, "storageFile", fileName);
                        jsonAddString(stObj, "timestampFile", fileName + ".timestamps.csv");
                    }
                    else {
                        jsonAddString(stObj, "filePattern", "%05d" + colorExtNormalized(saveOptions.colorExt));
                    }
                }
                else if(t == CollectDataType::Depth) {
                    jsonAddString(stObj, "storageEncoding", depthStorageEncodingName(saveOptions));
                    if(depthOutputIsFfv1Mkv(saveOptions)) {
                        const std::string fileName = depthFfv1OutputFileName();
                        jsonAddString(stObj, "storageFile", fileName);
                        jsonAddString(stObj, "timestampFile", fileName + ".timestamps.csv");
                    }
                    else {
                        jsonAddString(stObj, "filePattern", "%05d.png");
                    }
                }
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
    std::deque<EgoFrame> coordEgoQueue_;
    std::deque<TouchQueuedSample> coordTouchQueue_;
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

    std::mutex depthAlignMtx_;
    std::condition_variable depthAlignCv_;
    std::condition_variable depthAlignDrainCv_;
    std::deque<DepthAlignTask> depthAlignQueue_;
    size_t depthAlignQueueMax_ = 90;
    std::vector<std::thread> depthAlignWorkers_;
    std::atomic_bool depthAlignStop_{ false };
    std::atomic_bool depthAlignActive_{ false };
    std::atomic<size_t> queuedDepthAlignCount_{ 0 };
    std::atomic<int> depthAlignInFlight_{ 0 };

    mutable std::mutex h265Mtx_;
    std::unordered_map<std::string, std::unique_ptr<H265Encoder>> h265Encoders_;
    std::atomic_bool h265EncodingActive_{ false };
    std::atomic<size_t> h265StoppingEncoders_{ 0 };

    mutable std::mutex depthFfv1Mtx_;
    std::unordered_map<std::string, std::unique_ptr<DepthFfv1Encoder>> depthFfv1Encoders_;
    std::atomic_bool depthFfv1EncodingActive_{ false };
    std::atomic<size_t> depthFfv1StoppingEncoders_{ 0 };

    std::atomic_bool capturing_{ false };
    std::atomic_bool recording_{ false };
    std::atomic_bool hasData_{ false };
    std::atomic_bool stopping_{ false };
    std::thread      fisheyeRecordThread_;
    std::thread      egoRecordThread_;

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
    std::vector<std::string> activeFisheyeCameraIds_;
    FisheyeRecorder fisheyeRecorder_;
    bool            egoEnabled_ = false;
    EgoRecorder     ownedEgoRecorder_;
    EgoRecorder    &egoRecorder_;
    bool            ownsEgoRecorder_ = true;
    bool            touchEnabled_ = false;
    std::vector<TouchRuntime> touchRuntimes_;

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
    mutable std::mutex streamHealthMtx_;
    std::unordered_map<std::string, StreamHealthState> streamHealth_;
    std::optional<CameraStreamFault> activeCameraFault_;

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
    int         completed = 0;
    int         total = 1;
    std::string claimedBySubject;
    bool        claimedByOther = false;
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

enum class CaptureState { IDLE, RECORDING, DRAINING, STOPPED_READY, BACKEND_SYNC_PENDING, DELETE_CONFIRM };

enum class PendingExitAction { None, ExitCollection, ReturnConfig, ReturnTaskSelect };

struct EpisodeReservationUi {
    bool        active = false;
    std::string reservationId;
    std::string taskName;
    std::string storageName;
    int         episodeNumber = 0;
    fs::path    collectionPath;
    std::string idempotencyKey;
    double      durationSeconds = 0.0;
    int         frameCount = 0;
    bool        localFinalized = false;
    bool        countedComplete = false;

    void clear() {
        active = false;
        reservationId.clear();
        taskName.clear();
        storageName.clear();
        episodeNumber = 0;
        collectionPath.clear();
        idempotencyKey.clear();
        durationSeconds = 0.0;
        frameCount = 0;
        localFinalized = false;
        countedComplete = false;
    }
};

struct TrackedUploadUi {
    std::string episodeId;
    std::string taskName;
    int         episodeNumber = 0;
    TaskUploadStatus status;
    bool        terminalLogged = false;
    bool        pollErrorLogged = false;
    std::chrono::steady_clock::time_point nextPoll{};
};

static bool uploadStatusTerminal(const std::string &status) {
    return status == "succeeded" || status == "failed" || status == "canceled";
}

static std::string formatUploadBytes(uint64_t bytes) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while(value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    if(unit == 0) {
        oss << static_cast<uint64_t>(value) << " " << units[unit];
    }
    else {
        oss.setf(std::ios::fixed);
        oss << std::setprecision(value >= 100.0 ? 0 : 1) << value << " " << units[unit];
    }
    return oss.str();
}

static std::string uploadStatusLine(const TrackedUploadUi &tracked) {
    const auto &st = tracked.status;
    const std::string task = tracked.taskName.empty() ? std::string("episode") : tracked.taskName;
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "NAS " << task << " ep" << tracked.episodeNumber << ": ";
    if(!st.available) {
        oss << "queued";
        return oss.str();
    }
    oss << (st.jobStatus.empty() ? std::string("unknown") : st.jobStatus);
    if(!st.phase.empty()) {
        oss << " " << st.phase;
    }
    oss << " " << std::setprecision(1) << st.percent << "%";
    if(st.totalBytes > 0) {
        oss << " (" << formatUploadBytes(st.copiedBytes) << "/" << formatUploadBytes(st.totalBytes) << ")";
    }
    if(!st.error.empty()) {
        oss << " error: " << st.error;
    }
    return oss.str();
}

struct CaptureNasUploadResult {
    bool        ok = false;
    std::string episodeUri;
    fs::path    destPath;
    std::string error;
    uint64_t    totalBytes = 0;
    int         filesTotal = 0;
};

struct CaptureNasFinalizeResult {
    CaptureNasUploadResult upload;
    bool                    backendConfirmed = false;
    std::vector<TaskBackendTask> refreshedTasks;
    std::string             backendError;
    bool                    localCleanupAttempted = false;
    bool                    localCleanupSucceeded = false;
    std::string             localCleanupError;
};

struct CaptureNasFinalizeJob {
    std::string reservationId;
    std::string taskName;
    int         episodeNumber = 0;
    fs::path    collectionPath;
    std::future<CaptureNasFinalizeResult> future;
};

static std::string cleanNasPathPart(const std::string &value, const std::string &fallback) {
    std::string out;
    for(const char ch: trimString(value)) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if(std::isalnum(uch) || ch == '_' || ch == '-' || ch == '.') {
            out.push_back(ch);
        }
        else if(!out.empty() && out.back() != '_') {
            out.push_back('_');
        }
    }
    while(!out.empty() && (out.back() == '_' || out.back() == '.' || out.back() == '-')) {
        out.pop_back();
    }
    return out.empty() ? fallback : out;
}

static std::string cleanNasUriPrefix(std::string value) {
    value = trimString(std::move(value));
    while(value.size() > 1 && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

static std::string joinNasUriPath(const std::string &prefix,
                                  const std::string &subject,
                                  const std::string &task,
                                  const std::string &episode) {
    return cleanNasUriPrefix(prefix) + "/" + subject + "/" + task + "/" + episode;
}

static bool copyEpisodeTreeToNasTemp(const fs::path &source,
                                     const fs::path &tmp,
                                     uint64_t &totalBytes,
                                     int &filesTotal,
                                     std::string *errorMessage) {
    std::error_code ec;
    if(!fs::exists(source, ec) || !fs::is_directory(source, ec)) {
        if(errorMessage) {
            *errorMessage = "collection path is not a directory: " + source.string();
        }
        return false;
    }
    fs::remove_all(tmp, ec);
    ec.clear();
    fs::create_directories(tmp, ec);
    if(ec) {
        if(errorMessage) {
            *errorMessage = "failed to create NAS temp dir " + tmp.string() + ": " + ec.message();
        }
        return false;
    }

    totalBytes = 0;
    filesTotal = 0;
    for(fs::recursive_directory_iterator it(source, ec), end; it != end; it.increment(ec)) {
        if(ec) {
            if(errorMessage) {
                *errorMessage = "failed to scan collection path " + source.string() + ": " + ec.message();
            }
            return false;
        }
        const fs::path rel = fs::relative(it->path(), source, ec);
        if(ec || rel.empty()) {
            if(errorMessage) {
                *errorMessage = "failed to calculate relative path under " + source.string();
            }
            return false;
        }
        const fs::path out = tmp / rel;
        if(it->is_directory(ec)) {
            fs::create_directories(out, ec);
            if(ec) {
                if(errorMessage) {
                    *errorMessage = "failed to create NAS directory " + out.string() + ": " + ec.message();
                }
                return false;
            }
            continue;
        }
        if(!it->is_regular_file(ec)) {
            continue;
        }
        fs::create_directories(out.parent_path(), ec);
        if(ec) {
            if(errorMessage) {
                *errorMessage = "failed to create NAS parent " + out.parent_path().string() + ": " + ec.message();
            }
            return false;
        }
        const auto fileSize = fs::file_size(it->path(), ec);
        if(ec) {
            if(errorMessage) {
                *errorMessage = "failed to stat " + it->path().string() + ": " + ec.message();
            }
            return false;
        }
        fs::copy_file(it->path(), out, fs::copy_options::overwrite_existing, ec);
        if(ec) {
            if(errorMessage) {
                *errorMessage = "failed to copy " + it->path().string() + " to " + out.string() + ": " + ec.message();
            }
            return false;
        }
        totalBytes += static_cast<uint64_t>(fileSize);
        filesTotal += 1;
    }
    return true;
}

static bool writeCaptureNasManifest(const fs::path &path,
                                    const std::string &reservationId,
                                    const std::string &subjectId,
                                    const std::string &taskName,
                                    int episodeNumber,
                                    const std::string &storageName,
                                    const fs::path &source,
                                    const std::string &episodeUri,
                                    int filesTotal,
                                    uint64_t totalBytes,
                                    std::string *errorMessage) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "episode_id", reservationId.c_str());
    cJSON_AddStringToObject(root, "episode_uuid", reservationId.c_str());
    cJSON_AddStringToObject(root, "subject_id", subjectId.c_str());
    cJSON_AddStringToObject(root, "task_name", taskName.c_str());
    cJSON_AddNumberToObject(root, "episode_index", episodeNumber);
    cJSON_AddStringToObject(root, "storage_name", storageName.c_str());
    cJSON_AddStringToObject(root, "source_path", source.string().c_str());
    cJSON_AddStringToObject(root, "nas_uri", episodeUri.c_str());
    cJSON_AddNumberToObject(root, "files_total", filesTotal);
    cJSON_AddNumberToObject(root, "total_bytes", static_cast<double>(totalBytes));
    cJSON_AddStringToObject(root, "completed_by", "capture_side_uploader");
    cJSON_AddStringToObject(root, "storage_backend", "nas");
    char *printed = cJSON_Print(root);
    cJSON_Delete(root);
    if(!printed) {
        if(errorMessage) {
            *errorMessage = "failed to serialize NAS manifest";
        }
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if(!out) {
        if(errorMessage) {
            *errorMessage = "failed to open NAS manifest " + path.string();
        }
        cJSON_free(printed);
        return false;
    }
    out << printed << "\n";
    cJSON_free(printed);
    return true;
}

static bool canReplaceNasEpisode(const fs::path &dest,
                                 const std::string &reservationId,
                                 std::string *errorMessage) {
    std::error_code ec;
    if(!fs::exists(dest, ec)) {
        return true;
    }
    const fs::path manifestPath = dest / ".orbbec_upload_manifest.json";
    std::ifstream in(manifestPath, std::ios::binary);
    if(!in) {
        if(errorMessage) {
            *errorMessage = "NAS destination already exists without an upload manifest: " + dest.string();
        }
        return false;
    }
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    cJSON *root = cJSON_Parse(content.c_str());
    if(!root) {
        if(errorMessage) {
            *errorMessage = "NAS destination has an invalid upload manifest: " + manifestPath.string();
        }
        return false;
    }
    cJSON *idItem = cJSON_GetObjectItemCaseSensitive(root, "episode_uuid");
    if(!cJSON_IsString(idItem)) {
        idItem = cJSON_GetObjectItemCaseSensitive(root, "episode_id");
    }
    const std::string existingId = cJSON_IsString(idItem) && idItem->valuestring
        ? std::string(idItem->valuestring)
        : std::string();
    cJSON_Delete(root);
    if(existingId != reservationId) {
        if(errorMessage) {
            *errorMessage = "NAS destination belongs to another episode UUID: " + dest.string();
        }
        return false;
    }
    return true;
}

static CaptureNasUploadResult uploadEpisodeToNas(const NasConfig &nas,
                                                 const fs::path &collectionPath,
                                                 const std::string &subjectId,
                                                 const std::string &taskName,
                                                 const std::string &reservationId,
                                                 int episodeNumber,
                                                 const std::string &reservedStorageName) {
    CaptureNasUploadResult result;
    const std::string subject = cleanNasPathPart(subjectId, "subject");
    const std::string task = cleanNasPathPart(taskName, "task");
    const std::string episode = reservedStorageName.empty()
        ? "episode" + std::to_string(episodeNumber)
        : cleanNasPathPart(reservedStorageName, "episode");
    const std::string uriPrefix = cleanNasUriPrefix(nas.uriPrefix);
    if(uriPrefix.empty()) {
        result.error = "NAS uriPrefix is empty";
        return result;
    }
    const fs::path root = nas.mountPath;
    if(root.empty()) {
        result.error = "NAS mountPath is empty";
        return result;
    }
    result.episodeUri = joinNasUriPath(uriPrefix, subject, task, episode);
    result.destPath = root / subject / task / episode;
    const fs::path tmp = root / ".upload_tmp"
                       / (cleanNasPathPart(reservationId, episode) + ".capture_upload");

    std::string error;
    if(!copyEpisodeTreeToNasTemp(collectionPath, tmp, result.totalBytes, result.filesTotal, &error)) {
        result.error = error;
        return result;
    }
    if(!writeCaptureNasManifest(tmp / ".orbbec_upload_manifest.json",
                                reservationId,
                                subjectId,
                                taskName,
                                episodeNumber,
                                episode,
                                collectionPath,
                                result.episodeUri,
                                result.filesTotal,
                                result.totalBytes,
                                &error)) {
        result.error = error;
        return result;
    }

    std::error_code ec;
    fs::create_directories(result.destPath.parent_path(), ec);
    if(ec) {
        result.error = "failed to create NAS destination parent " + result.destPath.parent_path().string() + ": " + ec.message();
        return result;
    }
    if(!canReplaceNasEpisode(result.destPath, reservationId, &error)) {
        result.error = error;
        return result;
    }
    fs::remove_all(result.destPath, ec);
    ec.clear();
    fs::rename(tmp, result.destPath, ec);
    if(ec) {
        result.error = "failed to publish NAS episode " + result.destPath.string() + ": " + ec.message();
        return result;
    }
    result.ok = true;
    return result;
}

static bool pathIsSameOrBelow(const fs::path &path, const fs::path &root) {
    auto pathIt = path.begin();
    const auto pathEnd = path.end();
    for(auto rootIt = root.begin(), rootEnd = root.end(); rootIt != rootEnd; ++rootIt, ++pathIt) {
        if(pathIt == pathEnd || *pathIt != *rootIt) {
            return false;
        }
    }
    return true;
}

static bool deleteLocalEpisodeAfterNasUpload(const fs::path &collectionPath,
                                             const fs::path &nasMountPath,
                                             int episodeNumber,
                                             std::string *errorMessage) {
    if(collectionPath.empty()) {
        if(errorMessage) {
            *errorMessage = "local episode path is empty";
        }
        return false;
    }
    const std::string expectedName = "episode_" + std::to_string(episodeNumber);
    if(collectionPath.filename() != expectedName) {
        if(errorMessage) {
            *errorMessage = "refusing to delete unexpected local episode path: " + collectionPath.string();
        }
        return false;
    }

    std::error_code ec;
    const fs::file_status status = fs::symlink_status(collectionPath, ec);
    if(ec) {
        if(errorMessage) {
            *errorMessage = "failed to inspect local episode " + collectionPath.string() + ": " + ec.message();
        }
        return false;
    }
    if(!fs::exists(status)) {
        return true;
    }
    if(fs::is_symlink(status) || !fs::is_directory(status)) {
        if(errorMessage) {
            *errorMessage = "refusing to delete a non-directory or symlink: " + collectionPath.string();
        }
        return false;
    }

    const fs::path resolvedLocal = fs::weakly_canonical(collectionPath, ec);
    if(ec) {
        if(errorMessage) {
            *errorMessage = "failed to resolve local episode " + collectionPath.string() + ": " + ec.message();
        }
        return false;
    }
    if(!nasMountPath.empty()) {
        const fs::path resolvedNas = fs::weakly_canonical(nasMountPath, ec);
        if(ec) {
            if(errorMessage) {
                *errorMessage = "failed to resolve NAS mount " + nasMountPath.string() + ": " + ec.message();
            }
            return false;
        }
        if(pathIsSameOrBelow(resolvedLocal, resolvedNas)) {
            if(errorMessage) {
                *errorMessage = "refusing to delete an episode inside the NAS mount: " + resolvedLocal.string();
            }
            return false;
        }
    }

    const auto removed = fs::remove_all(resolvedLocal, ec);
    if(ec) {
        if(errorMessage) {
            *errorMessage = "failed to delete local episode " + resolvedLocal.string() + ": " + ec.message();
        }
        return false;
    }
    const bool stillExists = fs::exists(resolvedLocal, ec);
    if(ec) {
        if(errorMessage) {
            *errorMessage = "failed to verify local cleanup " + resolvedLocal.string() + ": " + ec.message();
        }
        return false;
    }
    if(removed == 0 || stillExists) {
        if(errorMessage) {
            *errorMessage = "local episode still exists after cleanup: " + resolvedLocal.string();
        }
        return false;
    }
    return true;
}

static TaskInfo taskInfoFromBackend(const TaskBackendTask &src) {
    TaskInfo out;
    out.name = src.taskName;
    out.description_cn = src.descriptionCn;
    out.description_en = src.descriptionEn;
    out.repeat_times = std::max(1, src.total);
    out.total = std::max(1, src.total);
    out.completed = std::max(0, std::min(src.completed, out.total));
    out.claimedBySubject = src.claimedBySubject;
    out.claimedByOther = src.claimedByOther;
    return out;
}

static int findTaskIndexByName(const std::vector<TaskInfo> &tasks, const std::string &taskName) {
    for(int i = 0; i < static_cast<int>(tasks.size()); ++i) {
        if(tasks[static_cast<size_t>(i)].name == taskName) {
            return i;
        }
    }
    return -1;
}

static bool isTaskComplete(const TaskInfo &task) {
    return task.completed >= task.total;
}

static bool isTaskSelectable(const TaskInfo &task) {
    return !task.claimedByOther && !isTaskComplete(task);
}

static std::string makeCollectionClientId() {
    char host[128] = {0};
#if defined(__unix__) || defined(__APPLE__)
    if(::gethostname(host, sizeof(host) - 1) != 0) {
        std::snprintf(host, sizeof(host), "host");
    }
    const long pid = static_cast<long>(::getpid());
#else
    std::snprintf(host, sizeof(host), "host");
    const long pid = 0;
#endif
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream oss;
    oss << host << "-" << pid << "-" << now;
    return oss.str();
}

static std::string makeIdempotencyKey(const std::string &clientId,
                                      const std::string &reservationId,
                                      int episodeNumber) {
    std::ostringstream oss;
    oss << clientId << ":" << reservationId << ":episode_" << episodeNumber << ":confirm";
    return oss.str();
}

static std::string captureStateName(CaptureState state,
                                    const EpisodeReservationUi &reservation,
                                    bool cameraFaultActive) {
    if(cameraFaultActive) {
        return "camera-error";
    }
    if(reservation.active && reservation.localFinalized && state == CaptureState::BACKEND_SYNC_PENDING) {
        return "backend-sync-pending";
    }
    switch(state) {
    case CaptureState::IDLE:
        return "idle";
    case CaptureState::RECORDING:
        return "recording";
    case CaptureState::DRAINING:
        return "draining";
    case CaptureState::STOPPED_READY:
        return "stopped-not-confirmed";
    case CaptureState::BACKEND_SYNC_PENDING:
        return "backend-sync-pending";
    case CaptureState::DELETE_CONFIRM:
        return "delete-confirm";
    }
    return "unknown";
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

enum class CollectionPage { Config, TaskSelect, Capture };

struct CollectionCaptureUi {
    std::string activeField;
    std::string msg;

    std::vector<TaskInfo>     tasks;
    int                       currentTaskIdx = -1;
    int                       currentEpisode = 0;
    int                       taskListPage = 0;
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

struct CameraFaultModalActions {
    bool exitCollection = false;
    bool restartCapture = false;
};

static CameraFaultModalActions drawCameraFaultModal(cv::Mat &ui,
                                                    FrameMouse &fm,
                                                    const MultiDeviceStreamingRecorder::CameraStreamFault &fault,
                                                    bool restartEnabled,
                                                    const std::string &detailLine) {
    CameraFaultModalActions actions;
    cv::Mat shade(ui.size(), ui.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(shade, 0.64, ui, 0.36, 0.0, ui);

    const int winW = ui.cols;
    const int winH = ui.rows;
    const int modalW = std::min(980, winW - 60);
    const int modalH = 390;
    const cv::Rect modal((winW - modalW) / 2, (winH - modalH) / 2, modalW, modalH);
    cv::rectangle(ui, modal, cv::Scalar(24, 24, 30), cv::FILLED);
    cv::rectangle(ui, modal, cv::Scalar(80, 80, 255), 3);

    const std::string title = "Camera Stream Timeout";
    int baseline = 0;
    const auto titleSz = cv::getTextSize(title, cv::FONT_HERSHEY_DUPLEX, 1.22, 3, &baseline);
    cv::putText(ui, title, cv::Point(modal.x + (modal.width - titleSz.width) / 2, modal.y + 58),
                cv::FONT_HERSHEY_DUPLEX, 1.22, cv::Scalar(90, 90, 255), 3, cv::LINE_AA);

    const int textLeft = modal.x + 34;
    const int textWidth = modal.width - 68;
    std::vector<std::string> lines;
    auto appendWrapped = [&](const std::string &s, double scale, int thickness) {
        auto wrapped = wrapTextToWidth(s, textWidth, cv::FONT_HERSHEY_DUPLEX, scale, thickness);
        lines.insert(lines.end(), wrapped.begin(), wrapped.end());
    };
    appendWrapped(fault.message, 0.78, 2);
    appendWrapped("Collection has been paused to prevent saving a broken capture.", 0.68, 1);
    if(!detailLine.empty()) {
        appendWrapped(detailLine, 0.62, 1);
    }

    int y = modal.y + 112;
    for(size_t i = 0; i < lines.size() && i < 6; ++i) {
        const cv::Scalar color = (i == 0) ? cv::Scalar(235, 235, 255) : cv::Scalar(215, 215, 215);
        const double scale = (i == 0) ? 0.78 : 0.62;
        const int thickness = (i == 0) ? 2 : 1;
        cv::putText(ui, lines[i], cv::Point(textLeft, y),
                    cv::FONT_HERSHEY_DUPLEX, scale, color, thickness, cv::LINE_AA);
        y += (i == 0) ? 34 : 28;
    }

    cv::putText(ui, "Ctrl+1 exit collection    Ctrl+2 delete episode, restart cameras, and return to READY",
                cv::Point(textLeft, modal.y + modal.height - 94),
                cv::FONT_HERSHEY_DUPLEX, 0.58, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

    cv::Rect bExit(modal.x + 34, modal.y + modal.height - 64, (modal.width - 88) / 2, 48);
    cv::Rect bRestart(modal.x + bExit.width + 54, modal.y + modal.height - 64, (modal.width - 88) / 2, 48);
    actions.exitCollection = uiButtonEx(ui, bExit, "Exit Collection [Ctrl+1]", fm, true);
    actions.restartCapture = uiButtonEx(ui, bRestart, "Delete + Restart [Ctrl+2]", fm, restartEnabled);
    return actions;
}

struct ExitConfirmModalActions {
    bool confirm = false;
    bool cancel = false;
};

static ExitConfirmModalActions drawExitConfirmModal(cv::Mat &ui,
                                                    FrameMouse &fm,
                                                    PendingExitAction action,
                                                    const std::string &taskName,
                                                    int completedThisRun,
                                                    const std::string &episodeState,
                                                    const EpisodeReservationUi &reservation) {
    ExitConfirmModalActions actions;
    cv::Mat shade(ui.size(), ui.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(shade, 0.62, ui, 0.38, 0.0, ui);

    const int winW = ui.cols;
    const int winH = ui.rows;
    const int modalW = std::min(900, winW - 60);
    const int modalH = 390;
    const cv::Rect modal((winW - modalW) / 2, (winH - modalH) / 2, modalW, modalH);
    cv::rectangle(ui, modal, cv::Scalar(24, 24, 30), cv::FILLED);
    cv::rectangle(ui, modal, cv::Scalar(255, 220, 120), 3);

    const std::string title = (action == PendingExitAction::ReturnConfig)
                                  ? "Return to Config?"
                                  : (action == PendingExitAction::ReturnTaskSelect ? "Return to Tasks?" : "Exit Collection?");
    int baseline = 0;
    const auto titleSz = cv::getTextSize(title, cv::FONT_HERSHEY_DUPLEX, 1.18, 3, &baseline);
    cv::putText(ui, title, cv::Point(modal.x + (modal.width - titleSz.width) / 2, modal.y + 58),
                cv::FONT_HERSHEY_DUPLEX, 1.18, cv::Scalar(255, 220, 120), 3, cv::LINE_AA);

    std::vector<std::string> lines;
    lines.push_back("Current task: " + (taskName.empty() ? std::string("(none)") : taskName));
    lines.push_back("Completed since entering collection: " + std::to_string(completedThisRun));
    lines.push_back("Current episode state: " + episodeState);
    if(reservation.active) {
        lines.push_back("Reserved episode: " + reservation.taskName + " episode_" + std::to_string(reservation.episodeNumber));
    }
    if(action == PendingExitAction::ReturnConfig) {
        lines.push_back("Returning to Config stops cameras but keeps collection open.");
    }
    else if(action == PendingExitAction::ReturnTaskSelect) {
        lines.push_back("Returning to Tasks stops cameras and keeps collection open.");
    }
    else {
        lines.push_back("Confirming will stop cameras and leave collection.");
    }

    const int textLeft = modal.x + 34;
    const int textWidth = modal.width - 68;
    int y = modal.y + 110;
    for(const auto &line: lines) {
        auto wrapped = wrapTextToWidth(line, textWidth, cv::FONT_HERSHEY_DUPLEX, 0.68, 1);
        for(const auto &w: wrapped) {
            if(y > modal.y + modal.height - 115) {
                break;
            }
            cv::putText(ui, w, cv::Point(textLeft, y),
                        cv::FONT_HERSHEY_DUPLEX, 0.68, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
            y += 30;
        }
    }

    cv::putText(ui, "Ctrl+1 confirm    Ctrl+4 cancel",
                cv::Point(textLeft, modal.y + modal.height - 84),
                cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

    cv::Rect bConfirm(modal.x + 34, modal.y + modal.height - 62, (modal.width - 88) / 2, 46);
    cv::Rect bCancel(modal.x + bConfirm.width + 54, modal.y + modal.height - 62, (modal.width - 88) / 2, 46);
    actions.confirm = uiButtonEx(ui, bConfirm,
                                 action == PendingExitAction::ReturnConfig
                                     ? "Return Config [Ctrl+1]"
                                     : (action == PendingExitAction::ReturnTaskSelect ? "Return Tasks [Ctrl+1]" : "Exit [Ctrl+1]"),
                                 fm, true);
    actions.cancel = uiButtonEx(ui, bCancel, "Cancel [Ctrl+4]", fm, true);
    return actions;
}

}  // namespace

int run_collection(const AppConfig &cfg,
                   const std::atomic_bool *cancel,
                   EgoRecorder *sharedEgoRecorder,
                   const std::string &operatorId) {
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
    const std::string collectionOperatorId = trimString(operatorId);
    const bool subjectBoundToLogin = !collectionOperatorId.empty();
    if(subjectBoundToLogin) {
        cfgUi.subjectId = collectionOperatorId;
    }
    CollectionCaptureUi capUi;
    MultiDeviceStreamingRecorder recorder(cfg, sharedEgoRecorder);
    TaskBackendClient backendClient(cfg.taskBackend.baseUrl, cfg.taskBackend.timeoutMs);
    const std::string collectionClientId = makeCollectionClientId();
    EpisodeReservationUi currentReservation;
    VoiceAnnouncer voice(cfg.voiceFeedback);
    std::deque<std::string> uiLogs;
    std::deque<TrackedUploadUi> trackedUploads;
    std::deque<CaptureNasFinalizeJob> captureNasFinalizeJobs;
    int logScroll = 0;
    CaptureState captureState = CaptureState::IDLE;
    bool pendingResetAfterDrain = false;
    int completedThisCollection = 0;
    bool exitConfirmActive = false;
    PendingExitAction pendingExitAction = PendingExitAction::None;
    bool pendingExitDeleteFaultEpisode = false;
    std::optional<MultiDeviceStreamingRecorder::CameraStreamFault> activeCameraFault;
    bool cameraFaultDrainCompleteLogged = false;
    bool cameraReadyAnnounced = false;
    bool extrinsicReadyChecked = false;
    bool extrinsicReadyAllowsStart = false;
    std::string extrinsicReadyStatus;
    bool manualExtrinsicRecheckPending = false;
    bool manualExtrinsicRecheckScreenShown = false;
    // This flag intentionally survives reset/confirm retries and is recreated only
    // when run_collection is entered again.
    bool initialCameraWarmupCompleted = false;
    constexpr double kInitialCameraWarmupSeconds = 5.0;
    std::chrono::steady_clock::time_point drainStartedAt{};
    std::chrono::steady_clock::time_point nextDrainStatusLog{};
    std::unordered_map<std::string, cv::Mat> latestFrameCache;
    cfgUi.enableEgo = cfg.ego.enabled;
    cfgUi.enableTouch = cfg.touch.enabled;
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

    auto announce = [&](const std::string &messageKey, const std::string &fallbackText) {
        voice.say(messageKey, fallbackText);
    };

    bool recordTickActive = false;
    std::chrono::steady_clock::time_point nextRecordTick{};
    std::chrono::steady_clock::time_point nextRecordElapsedAnnouncement{};
    const auto recordTickInterval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::clamp(cfg.voiceFeedback.recordTickIntervalSeconds, 0.5, 3600.0)));
    auto formatRecordElapsedAnnouncement = [](int64_t elapsedSeconds) {
        elapsedSeconds = std::max<int64_t>(0, elapsedSeconds);
        const int64_t minutes = elapsedSeconds / 60;
        const int64_t seconds = elapsedSeconds % 60;
        return "已采集" + std::to_string(minutes) + "分" + std::to_string(seconds) + "秒";
    };
    auto startRecordingTick = [&]() {
        const auto now = std::chrono::steady_clock::now();
        recordTickActive = true;
        nextRecordTick = now + recordTickInterval;
        nextRecordElapsedAnnouncement = now + std::chrono::seconds(15);
    };
    auto stopRecordingTick = [&]() {
        recordTickActive = false;
        nextRecordTick = {};
        nextRecordElapsedAnnouncement = {};
        voice.clearPending("record_tick");
        voice.clearPending("record_elapsed");
    };
    auto updateRecordingTick = [&]() {
        if(captureState != CaptureState::RECORDING || !recorder.isRecording()) {
            if(recordTickActive) {
                stopRecordingTick();
            }
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if(!recordTickActive) {
            recordTickActive = true;
            nextRecordTick = now + recordTickInterval;
            nextRecordElapsedAnnouncement = now + std::chrono::seconds(15);
            return;
        }
        if(now >= nextRecordElapsedAnnouncement) {
            const auto elapsedSeconds = static_cast<int64_t>(std::floor(std::max(0.0, recorder.currentRecordingSeconds())));
            announce("record_elapsed", formatRecordElapsedAnnouncement(elapsedSeconds));
            do {
                nextRecordElapsedAnnouncement += std::chrono::seconds(15);
            } while(now >= nextRecordElapsedAnnouncement + std::chrono::milliseconds(200));
        }
        if(now >= nextRecordTick) {
            announce("record_tick", "di");
            do {
                nextRecordTick += recordTickInterval;
            } while(now >= nextRecordTick + std::chrono::milliseconds(200));
        }
    };

    auto resetCameraReadyAnnouncement = [&]() {
        cameraReadyAnnounced = false;
        extrinsicReadyChecked = false;
        extrinsicReadyAllowsStart = false;
        extrinsicReadyStatus.clear();
        manualExtrinsicRecheckPending = false;
        manualExtrinsicRecheckScreenShown = false;
    };

    auto beginDrainStatusTracking = [&]() {
        drainStartedAt = std::chrono::steady_clock::now();
        nextDrainStatusLog = drainStartedAt + std::chrono::seconds(3);
    };

    auto resetDrainStatusTracking = [&]() {
        drainStartedAt = {};
        nextDrainStatusLog = {};
    };

    auto updateDrainStatusTracking = [&]() {
        if(captureState != CaptureState::DRAINING) {
            return;
        }
        if(recorder.isDrainComplete()) {
            resetDrainStatusTracking();
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if(drainStartedAt.time_since_epoch().count() == 0) {
            beginDrainStatusTracking();
        }
        if(nextDrainStatusLog.time_since_epoch().count() != 0 && now >= nextDrainStatusLog) {
            const std::string status = recorder.drainStatusLine();
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - drainStartedAt).count();
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << "Background save still running after " << std::setprecision(1)
                << (static_cast<double>(elapsedMs) / 1000.0) << "s";
            if(!status.empty()) {
                oss << ": " << status;
            }
            pushUiLog(oss.str());
            std::cerr << "[collection] " << oss.str() << std::endl;
            do {
                nextDrainStatusLog += std::chrono::seconds(3);
            } while(now >= nextDrainStatusLog + std::chrono::milliseconds(200));
        }
    };

    auto selectedTaskName = [&]() -> std::string {
        if(currentReservation.active && !currentReservation.taskName.empty()) {
            return currentReservation.taskName;
        }
        if(capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size())) {
            return capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name;
        }
        return "";
    };

    auto applyBackendTasks = [&](const std::vector<TaskBackendTask> &backendTasks,
                                 const std::string &preferredTaskName,
                                 bool preserveCurrentSelection) {
        std::string keep = preferredTaskName;
        if(keep.empty() && preserveCurrentSelection
           && capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size())) {
            keep = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name;
        }

        capUi.tasks.clear();
        capUi.tasks.reserve(backendTasks.size());
        for(const auto &task: backendTasks) {
            capUi.tasks.push_back(taskInfoFromBackend(task));
        }
        // Backend progress does not include capture-side uploads that have not
        // finished yet. Keep those accepted episodes counted in the UI so the
        // operator cannot over-capture a task while uploads run in background.
        for(auto &job: captureNasFinalizeJobs) {
            if(job.future.valid()
               && job.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                continue;
            }
            const int taskIdx = findTaskIndexByName(capUi.tasks, job.taskName);
            if(taskIdx >= 0) {
                auto &task = capUi.tasks[static_cast<size_t>(taskIdx)];
                task.completed = std::min(task.total, task.completed + 1);
            }
        }

        capUi.currentTaskIdx = keep.empty() ? -1 : findTaskIndexByName(capUi.tasks, keep);
        if(capUi.currentTaskIdx >= 0 && !isTaskSelectable(capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)])) {
            capUi.currentTaskIdx = -1;
        }
        if(capUi.currentTaskIdx >= 0) {
            const auto &task = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)];
            capUi.currentEpisode = isTaskComplete(task) ? task.total : task.completed + 1;
        }
        else {
            capUi.currentEpisode = 0;
        }
        const int maxPageByTask = std::max(0, static_cast<int>(capUi.tasks.size()) - 1);
        capUi.taskListPage = std::max(0, std::min(capUi.taskListPage, maxPageByTask));
        capUi.taskLoadError = false;
        capUi.taskErrorMsg.clear();
    };

    auto refreshTasksFromBackend = [&](const std::string &preferredTaskName,
                                       bool preserveCurrentSelection,
                                       std::string *errorMessage = nullptr) {
        std::vector<TaskBackendTask> tasks;
        std::string error;
        if(!backendClient.getTasks(trimString(cfgUi.subjectId), tasks, &error)) {
            capUi.taskLoadError = true;
            capUi.taskErrorMsg = error;
            if(errorMessage) {
                *errorMessage = error;
            }
            return false;
        }
        applyBackendTasks(tasks, preferredTaskName, preserveCurrentSelection);
        return true;
    };

    auto upsertTrackedUpload = [&](const std::string &episodeId,
                                   const std::string &taskName,
                                   int episodeNumber) {
        if(episodeId.empty()) {
            return;
        }
        for(auto &tracked: trackedUploads) {
            if(tracked.episodeId == episodeId) {
                tracked.taskName = taskName;
                tracked.episodeNumber = episodeNumber;
                tracked.nextPoll = std::chrono::steady_clock::now();
                return;
            }
        }
        TrackedUploadUi tracked;
        tracked.episodeId = episodeId;
        tracked.taskName = taskName;
        tracked.episodeNumber = episodeNumber;
        tracked.nextPoll = std::chrono::steady_clock::now();
        trackedUploads.push_front(std::move(tracked));
        while(trackedUploads.size() > 8) {
            trackedUploads.pop_back();
        }
    };

    auto pollTrackedUploads = [&](bool force) {
        const auto now = std::chrono::steady_clock::now();
        for(auto &tracked: trackedUploads) {
            if(!force && uploadStatusTerminal(tracked.status.jobStatus)) {
                continue;
            }
            if(!force && tracked.nextPoll.time_since_epoch().count() != 0 && now < tracked.nextPoll) {
                continue;
            }
            tracked.nextPoll = now + std::chrono::seconds(1);
            TaskUploadStatus status;
            std::string error;
            if(!backendClient.getUploadStatus(tracked.episodeId, status, &error)) {
                tracked.status.available = false;
                tracked.status.jobStatus = "poll-error";
                tracked.status.phase = "poll";
                tracked.status.error = error;
                if(!tracked.pollErrorLogged) {
                    pushUiLog("NAS upload status unavailable: " + error);
                    tracked.pollErrorLogged = true;
                }
                continue;
            }
            tracked.status = std::move(status);
            tracked.pollErrorLogged = false;
            if(uploadStatusTerminal(tracked.status.jobStatus) && !tracked.terminalLogged) {
                pushUiLog(uploadStatusLine(tracked));
                if(tracked.status.jobStatus == "succeeded" && !tracked.status.nasUri.empty()) {
                    pushUiLog("NAS URI: " + tracked.status.nasUri);
                }
                tracked.terminalLogged = true;
            }
        }
    };

    auto latestUploadLine = [&]() -> std::string {
        if(!captureNasFinalizeJobs.empty()) {
            const auto &job = captureNasFinalizeJobs.front();
            std::string line = "NAS " + job.taskName + " ep" + std::to_string(job.episodeNumber)
                             + ": capture-side upload running in background";
            if(captureNasFinalizeJobs.size() > 1) {
                line += " (" + std::to_string(captureNasFinalizeJobs.size()) + " active)";
            }
            return line;
        }
        if(trackedUploads.empty()) {
            return "";
        }
        return uploadStatusLine(trackedUploads.front());
    };

    auto requestExit = [&](PendingExitAction action, bool deleteFaultEpisode) {
        exitConfirmActive = true;
        pendingExitAction = action;
        pendingExitDeleteFaultEpisode = deleteFaultEpisode;
        capUi.msg.clear();
    };

    auto releaseCurrentReservation = [&](const std::string &reason) {
        if(!currentReservation.active) {
            return true;
        }
        if(currentReservation.localFinalized) {
            capUi.msg = "Cannot release: local episode is finalized and waiting for backend confirm";
            pushUiLog(capUi.msg);
            return false;
        }
        std::string error;
        const std::string subject = trimString(cfgUi.subjectId);
        if(!backendClient.releaseEpisode(currentReservation.reservationId,
                                         subject,
                                         currentReservation.taskName,
                                         collectionOperatorId,
                                         &error)) {
            capUi.msg = "Backend release failed: " + error;
            pushUiLog(capUi.msg);
            return false;
        }
        pushUiLog("Backend reservation released (" + reason + "): "
                  + currentReservation.taskName + " ep" + std::to_string(currentReservation.episodeNumber));
        currentReservation.clear();
        return true;
    };

    auto enqueueCaptureNasFinalizeForCurrentReservation = [&]() {
        if(!currentReservation.active) {
            capUi.msg = "NAS upload failed: no active reservation";
            pushUiLog(capUi.msg);
            return false;
        }
        const std::string subject = trimString(cfgUi.subjectId);
        if(subject.empty()) {
            capUi.msg = "NAS upload failed: subject is empty";
            pushUiLog(capUi.msg);
            return false;
        }
        if(currentReservation.collectionPath.empty()) {
            capUi.msg = "NAS upload failed: collection path is empty";
            pushUiLog(capUi.msg);
            return false;
        }
        const NasConfig nas = cfg.taskBackend.nas;
        const fs::path collectionPath = currentReservation.collectionPath;
        const std::string taskName = currentReservation.taskName;
        const std::string reservationId = currentReservation.reservationId;
        const int episodeNumber = currentReservation.episodeNumber;
        const std::string storageName = currentReservation.storageName;
        const std::string idempotencyKey = currentReservation.idempotencyKey;
        const double durationSeconds = currentReservation.durationSeconds;
        const int frameCount = currentReservation.frameCount;
        const std::string backendUrl = cfg.taskBackend.baseUrl;
        const int backendTimeoutMs = cfg.taskBackend.timeoutMs;
        const std::string operatorIdForJob = collectionOperatorId;
        const bool deleteLocalAfterUpload = nas.deleteLocalAfterUpload;

        CaptureNasFinalizeJob job;
        job.reservationId = reservationId;
        job.taskName = taskName;
        job.episodeNumber = episodeNumber;
        job.collectionPath = collectionPath;
        job.future = std::async(std::launch::async,
                                [nas, collectionPath, subject, taskName, reservationId, episodeNumber,
                                 storageName, idempotencyKey, durationSeconds, frameCount, backendUrl,
                                 backendTimeoutMs, operatorIdForJob, deleteLocalAfterUpload]() {
            CaptureNasFinalizeResult result;
            result.upload = uploadEpisodeToNas(nas, collectionPath, subject, taskName,
                                               reservationId, episodeNumber, storageName);
            if(!result.upload.ok) {
                return result;
            }

            TaskBackendClient workerBackend(backendUrl, backendTimeoutMs);
            result.backendConfirmed = workerBackend.confirmEpisode(
                reservationId, subject, taskName, episodeNumber, collectionPath.string(),
                result.upload.episodeUri, durationSeconds, frameCount, idempotencyKey,
                operatorIdForJob, result.refreshedTasks, &result.backendError);
            if(!result.backendConfirmed) {
                return result;
            }

            if(deleteLocalAfterUpload) {
                result.localCleanupAttempted = true;
                result.localCleanupSucceeded = deleteLocalEpisodeAfterNasUpload(
                    collectionPath, nas.mountPath, episodeNumber, &result.localCleanupError);
            }
            return result;
        });
        captureNasFinalizeJobs.push_back(std::move(job));

        if(!currentReservation.countedComplete) {
            completedThisCollection += 1;
            currentReservation.countedComplete = true;
            const int taskIdx = findTaskIndexByName(capUi.tasks, taskName);
            if(taskIdx >= 0) {
                auto &task = capUi.tasks[static_cast<size_t>(taskIdx)];
                task.completed = std::min(task.total, task.completed + 1);
            }
        }
        capUi.msg = "Capture confirmed; NAS upload is running in background";
        pushUiLog("Capture-side NAS upload queued: " + collectionPath.string());
        currentReservation.clear();
        return true;
    };

    auto confirmReservationWithBackend = [&]() {
        if(!currentReservation.active) {
            capUi.msg = "Backend confirm failed: no active reservation";
            pushUiLog(capUi.msg);
            return false;
        }
        std::vector<TaskBackendTask> refreshedTasks;
        std::string error;
        const std::string subject = trimString(cfgUi.subjectId);
        if(!backendClient.confirmEpisode(currentReservation.reservationId,
                                         subject,
                                         currentReservation.taskName,
                                         currentReservation.episodeNumber,
                                         currentReservation.collectionPath.string(),
                                         "",
                                         currentReservation.durationSeconds,
                                         currentReservation.frameCount,
                                         currentReservation.idempotencyKey,
                                         collectionOperatorId,
                                         refreshedTasks,
                                         &error)) {
            capUi.msg = "Backend confirm failed: " + error;
            pushUiLog(capUi.msg);
            return false;
        }

        applyBackendTasks(refreshedTasks, currentReservation.taskName, true);
        if(!currentReservation.countedComplete) {
            completedThisCollection += 1;
            currentReservation.countedComplete = true;
        }
        pushUiLog("Backend confirm OK: " + currentReservation.taskName
                  + " ep" + std::to_string(currentReservation.episodeNumber));
        upsertTrackedUpload(currentReservation.reservationId,
                             currentReservation.taskName,
                             currentReservation.episodeNumber);
        pollTrackedUploads(true);
        {
            const std::string line = latestUploadLine();
            if(!line.empty()) {
                pushUiLog(line);
            }
        }
        currentReservation.clear();
        return true;
    };

    auto pollCaptureNasFinalizeJobs = [&]() {
        for(auto it = captureNasFinalizeJobs.begin(); it != captureNasFinalizeJobs.end();) {
            if(it->future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                ++it;
                continue;
            }

            CaptureNasFinalizeResult result;
            try {
                result = it->future.get();
            }
            catch(const std::exception &e) {
                result.upload.error = e.what();
            }
            catch(...) {
                result.upload.error = "unknown capture-side NAS finalization error";
            }

            const std::string taskName = it->taskName;
            const int episodeNumber = it->episodeNumber;
            if(!result.upload.ok || !result.backendConfirmed) {
                const std::string error = !result.upload.ok ? result.upload.error : result.backendError;
                pushUiLog("ERROR: background NAS finalize failed for " + taskName
                          + " ep" + std::to_string(episodeNumber) + ": " + error);
                const int taskIdx = findTaskIndexByName(capUi.tasks, taskName);
                if(taskIdx >= 0) {
                    auto &task = capUi.tasks[static_cast<size_t>(taskIdx)];
                    task.completed = std::max(0, task.completed - 1);
                }
                completedThisCollection = std::max(0, completedThisCollection - 1);
                announce("confirm_failed", "nas upload failed");
                it = captureNasFinalizeJobs.erase(it);
                continue;
            }

            std::ostringstream oss;
            oss << "Capture-side NAS upload OK: " << result.upload.episodeUri
                << " files=" << result.upload.filesTotal
                << " bytes=" << result.upload.totalBytes;
            pushUiLog(oss.str());
            pushUiLog("Background backend confirm OK: " + taskName
                      + " ep" + std::to_string(episodeNumber));
            upsertTrackedUpload(it->reservationId, taskName, episodeNumber);
            if(result.localCleanupAttempted) {
                if(result.localCleanupSucceeded) {
                    pushUiLog("Local episode deleted after NAS upload: " + it->collectionPath.string());
                }
                else {
                    pushUiLog("WARNING: NAS upload is safe, but local episode cleanup failed: "
                              + result.localCleanupError);
                }
            }
            it = captureNasFinalizeJobs.erase(it);
        }
    };

    auto updateReadyState = [&]() {
        captureState = CaptureState::STOPPED_READY;
        resetDrainStatusTracking();
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
        if(captureState != CaptureState::DELETE_CONFIRM) {
            announce("reset_select", "reset select");
        }
        captureState = CaptureState::DELETE_CONFIRM;
        capUi.msg.clear();
    };

    bool running = true;
    auto performConfirmedExit = [&]() {
        const PendingExitAction action = pendingExitAction;
        const bool deleteFaultEpisode = pendingExitDeleteFaultEpisode;
        exitConfirmActive = false;
        pendingExitAction = PendingExitAction::None;
        pendingExitDeleteFaultEpisode = false;

        bool okToLeave = true;
        if(currentReservation.active && currentReservation.localFinalized) {
            pushUiLog("Exit requested while backend sync is pending. Retrying backend confirm first.");
            if(!confirmReservationWithBackend()) {
                captureState = CaptureState::BACKEND_SYNC_PENDING;
                okToLeave = false;
                announce("confirm_failed", "confirm failed");
            }
        }
        else {
            if(recorder.isRecording()) {
                stopRecordingTick();
                recorder.stopRecording();
            }
            if(recorder.hasCurrentSession()) {
                capUi.msg = deleteFaultEpisode ? "Deleting faulted episode before exit..." : "Deleting unconfirmed episode before exit...";
                pushUiLog(capUi.msg);
                collectionSetStage("ui_exit_wait_drain");
                while(!recorder.isDrainComplete()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                std::string error;
                if(!recorder.discardCurrentSession(&error)) {
                    capUi.msg = "Delete failed";
                    pushUiLog("Delete failed before exit: " + error);
                    okToLeave = false;
                }
            }
            if(okToLeave && currentReservation.active && !currentReservation.localFinalized) {
                okToLeave = releaseCurrentReservation("exit");
            }
        }

        if(!okToLeave) {
            return;
        }

        if(action == PendingExitAction::ReturnConfig || action == PendingExitAction::ReturnTaskSelect) {
            collectionSetStage(action == PendingExitAction::ReturnConfig ? "ui_confirm_return_config" : "ui_confirm_return_tasks");
            announce(action == PendingExitAction::ReturnConfig ? "config" : "tasks",
                     action == PendingExitAction::ReturnConfig ? "config" : "tasks");
            recorder.stopIfRunning(false);
            latestFrameCache.clear();
            activeCameraFault.reset();
            recorder.clearCameraStreamFault();
            cameraFaultDrainCompleteLogged = false;
            pendingResetAfterDrain = false;
            resetCameraReadyAnnouncement();
            captureState = CaptureState::IDLE;
            if(action == PendingExitAction::ReturnTaskSelect) {
                std::string error;
                if(!refreshTasksFromBackend(selectedTaskName(), true, &error)) {
                    capUi.msg = "Task refresh failed: " + error;
                    pushUiLog(capUi.msg);
                }
                else {
                    capUi.msg.clear();
                }
                page = CollectionPage::TaskSelect;
            }
            else {
                page = CollectionPage::Config;
                capUi.msg.clear();
            }
        }
        else {
            collectionSetStage("ui_confirm_exit_collection");
            recorder.stopIfRunning();
            running = false;
        }
    };

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
        pollTrackedUploads(false);
        pollCaptureNasFinalizeJobs();
        if(key == 27) {
            collectionSetStage("ui_exit_esc");
            if(exitConfirmActive) {
                exitConfirmActive = false;
                pendingExitAction = PendingExitAction::None;
                pendingExitDeleteFaultEpisode = false;
            }
            else {
                requestExit(PendingExitAction::ExitCollection, activeCameraFault.has_value());
            }
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

            cv::putText(ui, "Capture Types", cv::Point(left, top - 22), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            cv::Rect type1(left, top, 210, 36);
            cv::Rect type2(left + 230, top, 210, 36);
            cv::Rect type3(left + 460, top, 210, 36);
            cv::Rect type4(left + 690, top, 210, 36);
            if(uiCheckbox(ui, type1, cfgUi.enableMultiview, "multiview", fm)) {
                cfgUi.enableMultiview = !cfgUi.enableMultiview;
            }
            if(uiCheckbox(ui, type2, cfgUi.enableFisheyes, "fisheyes", fm)) {
                cfgUi.enableFisheyes = !cfgUi.enableFisheyes;
            }
            if(uiCheckbox(ui, type3, cfgUi.enableEgo, "ego", fm)) {
                cfgUi.enableEgo = !cfgUi.enableEgo;
            }
            if(uiCheckbox(ui, type4, cfgUi.enableTouch, "touch", fm)) {
                cfgUi.enableTouch = !cfgUi.enableTouch;
            }
            if(!cfgUi.hasSelectedCaptureType()) {
                cv::putText(ui, "Select at least one visual capture type", cv::Point(left, top + rowH - 8), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(60, 60, 255), 1, cv::LINE_AA);
            }

            const int fieldsTop = top + 2 * rowH;
            if(uiTextField(ui, cv::Rect(left, fieldsTop, 520, 36), "save_path (required)", cfgUi.saveRoot, cfgUi.activeField == "save", fm)) {
                cfgUi.activeField = "save";
            }
            const cv::Rect subjectRect(left + 560, fieldsTop, 260, 36);
            if(subjectBoundToLogin) {
                cv::rectangle(ui, subjectRect, cv::Scalar(30, 30, 30), cv::FILLED);
                cv::rectangle(ui, subjectRect, cv::Scalar(100, 140, 160), 1);
                cv::putText(ui, "subject_id (login account)", cv::Point(subjectRect.x, subjectRect.y - 6),
                            cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
                cv::putText(ui, cfgUi.subjectId, cv::Point(subjectRect.x + 8, subjectRect.y + subjectRect.height - 10),
                            cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(220, 245, 255), 1, cv::LINE_AA);
            }
            else if(uiTextField(ui, subjectRect, "subject_id (required)", cfgUi.subjectId, cfgUi.activeField == "sub", fm)) {
                cfgUi.activeField = "sub";
            }

            const int tuneTop = fieldsTop + rowH;
            if(uiTextField(ui, cv::Rect(left, tuneTop, 260, 36), "exposure_ms", cfgUi.exposureMs, cfgUi.activeField == "exp", fm)) {
                cfgUi.activeField = "exp";
            }
            if(uiTextField(ui, cv::Rect(left + 300, tuneTop, 260, 36), "brightness", cfgUi.brightness, cfgUi.activeField == "bri", fm)) {
                cfgUi.activeField = "bri";
            }

            if(!cfgUi.error.empty()) {
                cv::putText(ui, cfgUi.error, cv::Point(left, tuneTop + rowH + 46), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
            }
            else if(!cfgUi.notice.empty()) {
                cv::putText(ui, cfgUi.notice, cv::Point(left, tuneTop + rowH + 46), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(80, 200, 80), 2, cv::LINE_AA);
            }

            cv::Rect bBack(60, fieldsTop + 3 * rowH, 220, 60);
            cv::Rect bEnter(340, fieldsTop + 3 * rowH, 260, 60);
            if(!exitConfirmActive && uiButton(ui, bBack, "Back to Menu", fm)) {
                collectionSetStage("ui_back_menu");
                announce("menu", "menu");
                requestExit(PendingExitAction::ExitCollection, false);
            }
            if(!exitConfirmActive && uiButton(ui, bEnter, "Load Tasks", fm)) {
                collectionSetStage("ui_enter_capture");
                cfgUi.enforceRules();
                cfgUi.notice.clear();
                if(!cfgUi.hasSelectedCaptureType()) {
                    cfgUi.error = "Select at least one capture type";
                    announce("enter_failed", "enter failed");
                }
                else if(!cfgUi.hasRequiredFields()) {
                    cfgUi.error = "save_path and login account are required";
                    announce("enter_failed", "enter failed");
                }
                else {
                    const fs::path saveRoot = fs::path(trimString(cfgUi.saveRoot));
                    const std::string subject = trimString(cfgUi.subjectId);
                    (void)saveRoot;
                    (void)subject;
                    if(!cfg.taskBackend.enabled) {
                        cfgUi.error = "Task backend is disabled in config";
                        announce("enter_failed", "enter failed");
                    }
                    else {
                        std::string backendError;
                        if(!refreshTasksFromBackend("", false, &backendError) || capUi.tasks.empty()) {
                            cfgUi.error = capUi.tasks.empty() && backendError.empty()
                                              ? "Task backend returned no tasks"
                                              : "Task backend unavailable: " + backendError;
                            announce("enter_failed", "enter failed");
                            pushUiLog(cfgUi.error);
                        }
                        else {
                            cfgUi.error.clear();
                            capUi.currentTaskIdx = -1;
                            capUi.currentEpisode = 0;
                            capUi.msg = "Select one task";
                            currentReservation.clear();
                            captureState = CaptureState::IDLE;
                            pendingResetAfterDrain = false;
                            page = CollectionPage::TaskSelect;
                            resetCameraReadyAnnouncement();
                            announce("enter", "enter");
                            pushUiLog("Task list loaded from " + backendClient.baseUrl());
                        }
                    }
                }
            }

            if(!exitConfirmActive && !cfgUi.activeField.empty() && key > 0) {
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld = g_ctrlShortcutListening || ctrlFromMask;
                if(cfgUi.activeField == "save") {
                    handleTextInputShortcut(cfgUi.saveRoot, key, ctrlHeld);
                }
                else if(!subjectBoundToLogin && cfgUi.activeField == "sub") {
                    handleTextInputShortcut(cfgUi.subjectId, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "exp") {
                    handleTextInputShortcut(cfgUi.exposureMs, key, ctrlHeld);
                }
                else if(cfgUi.activeField == "bri") {
                    handleTextInputShortcut(cfgUi.brightness, key, ctrlHeld);
                }
            }
        }
        else if(page == CollectionPage::TaskSelect) {
            collectionSetStage("ui_page_task_select");
            cv::putText(ui, "Collection - Select Task", cv::Point(24, 48),
                        cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

            const std::string header = "Subject: " + trimString(cfgUi.subjectId)
                                     + "    Backend: " + backendClient.baseUrl();
            cv::putText(ui, header, cv::Point(28, 82),
                        cv::FONT_HERSHEY_DUPLEX, 0.58, cv::Scalar(190, 190, 190), 1, cv::LINE_AA);

            const int margin = 28;
            const int top = 110;
            const int bottomButtonsH = 78;
            const cv::Rect listPanel(margin, top,
                                     std::max(360, winW / 2 - 48),
                                     std::max(240, winH - top - bottomButtonsH - 18));
            const cv::Rect detailPanel(listPanel.x + listPanel.width + 24, top,
                                       std::max(320, winW - (listPanel.x + listPanel.width + 24) - margin),
                                       listPanel.height);
            cv::rectangle(ui, listPanel, cv::Scalar(24, 24, 26), cv::FILLED);
            cv::rectangle(ui, listPanel, cv::Scalar(90, 90, 90), 1);
            cv::rectangle(ui, detailPanel, cv::Scalar(28, 28, 32), cv::FILLED);
            cv::rectangle(ui, detailPanel, cv::Scalar(90, 90, 90), 1);

            cv::putText(ui, "Tasks", cv::Point(listPanel.x + 14, listPanel.y + 28),
                        cv::FONT_HERSHEY_DUPLEX, 0.68, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            const int rowH = 42;
            const int rowTop = listPanel.y + 58;
            const int maxRows = std::max(1, (listPanel.height - 70) / rowH);
            const int taskCount = static_cast<int>(capUi.tasks.size());
            const int pageCount = std::max(1, (taskCount + maxRows - 1) / maxRows);
            capUi.taskListPage = std::max(0, std::min(capUi.taskListPage, pageCount - 1));
            if(fm.wheelDelta != 0 && listPanel.contains(cv::Point(fm.x, fm.y))) {
                const int step = (fm.wheelDelta > 0) ? -1 : 1;
                capUi.taskListPage = std::max(0, std::min(pageCount - 1, capUi.taskListPage + step));
            }

            const cv::Rect bPrev(listPanel.x + listPanel.width - 176, listPanel.y + 8, 62, 30);
            const cv::Rect bNext(listPanel.x + listPanel.width - 76, listPanel.y + 8, 62, 30);
            const std::string pageLabel = "Page " + std::to_string(capUi.taskListPage + 1)
                                        + "/" + std::to_string(pageCount);
            int pageBaseline = 0;
            const auto pageSz = cv::getTextSize(pageLabel, cv::FONT_HERSHEY_DUPLEX, 0.52, 1, &pageBaseline);
            cv::putText(ui, pageLabel,
                        cv::Point(std::max(listPanel.x + 86, bPrev.x - pageSz.width - 10), listPanel.y + 28),
                        cv::FONT_HERSHEY_DUPLEX, 0.52, cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
            if(uiButtonEx(ui, bPrev, "Prev", fm, !exitConfirmActive && capUi.taskListPage > 0)) {
                capUi.taskListPage = std::max(0, capUi.taskListPage - 1);
            }
            if(uiButtonEx(ui, bNext, "Next", fm, !exitConfirmActive && capUi.taskListPage + 1 < pageCount)) {
                capUi.taskListPage = std::min(pageCount - 1, capUi.taskListPage + 1);
            }

            const int pageStart = capUi.taskListPage * maxRows;
            for(int row = 0; row < maxRows; ++row) {
                const int taskIdx = pageStart + row;
                if(taskIdx >= taskCount) {
                    break;
                }
                const auto &task = capUi.tasks[static_cast<size_t>(taskIdx)];
                const cv::Rect rowRect(listPanel.x + 10, rowTop + row * rowH,
                                       listPanel.width - 20, rowH - 6);
                const bool selected = taskIdx == capUi.currentTaskIdx;
                const bool complete = isTaskComplete(task);
                const bool claimedByOther = task.claimedByOther;
                const bool hover = rowRect.contains(cv::Point(fm.x, fm.y));
                cv::Scalar bg = selected ? cv::Scalar(62, 58, 34)
                              : (hover ? cv::Scalar(44, 44, 48) : cv::Scalar(32, 32, 35));
                if(claimedByOther && !selected) {
                    bg = hover ? cv::Scalar(45, 35, 35) : cv::Scalar(34, 28, 28);
                }
                if(complete && !selected) {
                    bg = hover ? cv::Scalar(38, 50, 38) : cv::Scalar(28, 38, 28);
                }
                cv::rectangle(ui, rowRect, bg, cv::FILLED);
                cv::rectangle(ui, rowRect, selected ? cv::Scalar(255, 220, 80) : cv::Scalar(68, 68, 72), 1);

                const std::string progress = claimedByOther ? "claimed" : (std::to_string(task.completed) + "/" + std::to_string(task.total));
                int baseline = 0;
                const auto progSz = cv::getTextSize(progress, cv::FONT_HERSHEY_DUPLEX, 0.58, 1, &baseline);
                cv::putText(ui, progress,
                            cv::Point(rowRect.x + rowRect.width - progSz.width - 10, rowRect.y + 24),
                            cv::FONT_HERSHEY_DUPLEX, 0.58,
                            claimedByOther ? cv::Scalar(150, 150, 170) : (complete ? cv::Scalar(120, 220, 120) : cv::Scalar(220, 220, 220)),
                            1, cv::LINE_AA);

                std::string label = task.name;
                const int labelMaxW = std::max(40, rowRect.width - progSz.width - 34);
                while(!label.empty()) {
                    const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.55, 1, &baseline);
                    if(sz.width <= labelMaxW) {
                        break;
                    }
                    label.pop_back();
                }
                cv::putText(ui, label, cv::Point(rowRect.x + 10, rowRect.y + 24),
                            cv::FONT_HERSHEY_DUPLEX, 0.55,
                            claimedByOther ? cv::Scalar(155, 155, 165) : (selected ? cv::Scalar(255, 235, 130) : cv::Scalar(235, 235, 235)),
                            1, cv::LINE_AA);

                if(!exitConfirmActive && fm.clicked && rowRect.contains(cv::Point(fm.clickX, fm.clickY))) {
                    fm.clicked = false;
                    if(claimedByOther) {
                        capUi.msg = task.claimedBySubject.empty()
                                        ? "Task already claimed by another subject"
                                        : ("Task already claimed by subject " + task.claimedBySubject);
                        pushUiLog(capUi.msg + ": " + task.name);
                        continue;
                    }
                    capUi.currentTaskIdx = taskIdx;
                    capUi.currentEpisode = complete ? task.total : task.completed + 1;
                    capUi.msg = complete ? "Selected task is complete" : "Task selected";
                    resetCameraReadyAnnouncement();
                    pushUiLog("Selected task: " + task.name + " progress "
                              + std::to_string(task.completed) + "/" + std::to_string(task.total));
                }
            }

            cv::putText(ui, "Task Detail", cv::Point(detailPanel.x + 16, detailPanel.y + 30),
                        cv::FONT_HERSHEY_DUPLEX, 0.68, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            const bool taskSelected = capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size());
            const bool selectedTaskComplete = taskSelected && isTaskComplete(capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)]);
            const bool selectedTaskClaimedByOther = taskSelected && capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].claimedByOther;
            const bool selectedTaskSelectable = taskSelected && isTaskSelectable(capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)]);
            if(taskSelected) {
                const auto &task = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)];
                auto nameLines = wrapTextToWidth(task.name, detailPanel.width - 32,
                                                 cv::FONT_HERSHEY_DUPLEX, 0.95, 2);
                int y = detailPanel.y + 72;
                for(size_t i = 0; i < nameLines.size() && i < 3; ++i) {
                    cv::putText(ui, nameLines[i], cv::Point(detailPanel.x + 16, y),
                                cv::FONT_HERSHEY_DUPLEX, 0.95,
                                selectedTaskComplete ? cv::Scalar(120, 220, 120) : cv::Scalar(255, 220, 50),
                                2, cv::LINE_AA);
                    y += 36;
                }
                std::string ownerLabel = task.claimedBySubject.empty() ? std::string("another subject") : task.claimedBySubject;
                if(ownerLabel.size() > 24) {
                    ownerLabel = ownerLabel.substr(0, 21) + "...";
                }
                const std::string progressLine = "Progress " + std::to_string(task.completed)
                                               + " / " + std::to_string(task.total)
                                               + (selectedTaskClaimedByOther
                                                      ? ("  claimed by " + ownerLabel)
                                                      : (selectedTaskComplete ? "  complete" : "  ready to capture"));
                cv::putText(ui, progressLine, cv::Point(detailPanel.x + 16, y + 8),
                            cv::FONT_HERSHEY_DUPLEX, 0.68, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

                const std::string &desc = task.description_cn.empty() ? task.description_en : task.description_cn;
                if(!desc.empty()) {
                    const int descLeft = detailPanel.x + 16;
                    const int descTop = y + 46;
                    const int descWidth = std::max(1, detailPanel.width - 32);
                    const int descMaxHeight = std::max(80, detailPanel.y + detailPanel.height - descTop - 18);
                    const int descFontH = choosePromptFontHeight(desc, descWidth, descMaxHeight, 42, 22);
                    const int lineGap = std::max(10, descFontH / 3);
                    const auto lines = wrapMultilineTextUtf8(desc, descWidth, descFontH);
                    int descY = descTop + descFontH;
                    const int descBottom = detailPanel.y + detailPanel.height - 14;
                    for(const auto &line: lines) {
                        if(descY > descBottom) {
                            break;
                        }
                        if(line.empty()) {
                            descY += lineGap;
                            continue;
                        }
                        putTextUtf8(ui, line, cv::Point(descLeft, descY),
                                    descFontH, cv::Scalar(225, 225, 225));
                        descY += descFontH + lineGap;
                    }
                }
            }
            else {
                cv::putText(ui, capUi.tasks.empty() ? "No tasks loaded" : "Select a task from the list",
                            cv::Point(detailPanel.x + 16, detailPanel.y + 78),
                            cv::FONT_HERSHEY_DUPLEX, 0.78, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
            }

            const int btnY = winH - 62;
            const int btnH = 42;
            cv::Rect bConfig(margin, btnY, 170, btnH);
            cv::Rect bMenu(margin + 184, btnY, 150, btnH);
            cv::Rect bRefresh(winW - margin - 380, btnY, 160, btnH);
            cv::Rect bContinue(winW - margin - 204, btnY, 204, btnH);
            const bool allowContinue = !exitConfirmActive && selectedTaskSelectable;
            bool doConfig = uiButtonEx(ui, bConfig, "Back Config", fm, !exitConfirmActive);
            bool doMenu = uiButtonEx(ui, bMenu, "Menu", fm, !exitConfirmActive);
            bool doRefresh = uiButtonEx(ui, bRefresh, "Refresh", fm, !exitConfirmActive);
            bool doContinue = uiButtonEx(ui, bContinue, "Enter Capture", fm, allowContinue);

            if(key > 0 && !exitConfirmActive) {
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld = g_ctrlShortcutListening || ctrlFromMask;
                const int baseKey = key & 0xFFFF;
                if(ctrlHeld) {
                    if(baseKey == '1') {
                        doContinue = allowContinue;
                    }
                    else if(baseKey == '2') {
                        doRefresh = true;
                    }
                    else if(baseKey == '3') {
                        doConfig = true;
                    }
                    else if(baseKey == '4') {
                        doMenu = true;
                    }
                }
            }

            if(doConfig) {
                page = CollectionPage::Config;
                capUi.msg.clear();
            }
            if(doMenu) {
                announce("menu", "menu");
                requestExit(PendingExitAction::ExitCollection, false);
            }
            if(doRefresh) {
                std::string error;
                const std::string keep = selectedTaskName();
                if(refreshTasksFromBackend(keep, true, &error)) {
                    capUi.msg = "Tasks refreshed";
                    pushUiLog("Tasks refreshed from backend.");
                }
                else {
                    capUi.msg = "Task refresh failed: " + error;
                    pushUiLog(capUi.msg);
                    announce("enter_failed", "enter failed");
                }
            }
            if(doContinue) {
                collectionSetStage("ui_task_select_enter_capture");
                std::string keep = selectedTaskName();
                if(keep.empty()) {
                    capUi.msg = "Select one task";
                }
                else {
                    currentReservation.clear();
                    activeCameraFault.reset();
                    recorder.clearCameraStreamFault();
                    cameraFaultDrainCompleteLogged = false;
                    captureState = CaptureState::IDLE;
                    pendingResetAfterDrain = false;
                    resetCameraReadyAnnouncement();
                    const bool ok = recorder.start(cfgUi);
                    if(ok) {
                        page = CollectionPage::Capture;
                        announce("enter", "enter");
                        pushUiLog("Enter capture: " + keep);
                        if(cfg.extrinsicHealth.enabled && cfgUi.enableMultiview && !initialCameraWarmupCompleted) {
                            pushUiLog("Initial RGB/depth warm-up started: every camera must receive both streams for a full 5 seconds before the first extrinsic check.");
                        }
                        const std::string s = recorder.streamProfilesLine();
                        if(!s.empty()) {
                            pushUiLog(s);
                        }
                    }
                    else {
                        capUi.msg = "Camera start failed";
                        announce("enter_failed", "enter failed");
                        pushUiLog("Enter capture camera start failed");
                        const std::string line = recorder.lastInfoLine();
                        if(!line.empty()) {
                            pushUiLog(line);
                        }
                    }
                }
            }

            if(!capUi.msg.empty()) {
                std::string lowerMsg = capUi.msg;
                std::transform(lowerMsg.begin(), lowerMsg.end(), lowerMsg.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                const bool isError = (lowerMsg.find("fail") != std::string::npos) || (lowerMsg.find("error") != std::string::npos);
                cv::putText(ui, capUi.msg, cv::Point(28, winH - 82),
                            cv::FONT_HERSHEY_DUPLEX, 0.62,
                            isError ? cv::Scalar(60, 60, 255) : cv::Scalar(80, 200, 80),
                            1, cv::LINE_AA);
            }
        }
        else {
            collectionSetStage("ui_page_capture");

            if(!activeCameraFault.has_value()) {
                auto fault = recorder.pollCameraStreamFault();
                if(fault.has_value()) {
                    activeCameraFault = fault;
                    cameraFaultDrainCompleteLogged = false;
                    pendingResetAfterDrain = false;
                    resetDrainStatusTracking();
                    resetCameraReadyAnnouncement();
                    capUi.msg = "Camera stream timeout. Choose an action.";
                    pushUiLog(fault->message);
                    announce("camera_fault", "camera error");
                    if(captureState == CaptureState::RECORDING) {
                        stopRecordingTick();
                        recorder.stopRecording();
                        captureState = recorder.isDrainComplete() ? CaptureState::STOPPED_READY : CaptureState::DRAINING;
                        if(captureState == CaptureState::DRAINING) {
                            beginDrainStatusTracking();
                        }
                        pushUiLog("Camera fault fuse tripped. Recording stopped.");
                    }
                }
            }

            const bool cameraFaultActive = activeCameraFault.has_value();
            const bool cameraFaultRestartBlocked = cameraFaultActive && recorder.hasCurrentSession() && !recorder.isDrainComplete();
            const auto cameraReadiness = recorder.cameraStreamReadiness();
            const bool taskSelected = capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size());
            const bool selectedTaskComplete = taskSelected && isTaskComplete(capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)]);
            const bool selectedTaskClaimedByOther = taskSelected && capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].claimedByOther;
            const bool selectedTaskSelectable = taskSelected && isTaskSelectable(capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)]);
            const bool initialCameraWarmupRequired = cfg.extrinsicHealth.enabled && cfgUi.enableMultiview;
            const auto initialCameraWarmup = recorder.cameraWarmupReadiness(kInitialCameraWarmupSeconds);
            if(initialCameraWarmupRequired && !initialCameraWarmupCompleted && initialCameraWarmup.allReady) {
                initialCameraWarmupCompleted = true;
                pushUiLog(initialCameraWarmup.message);
            }
            bool deferExtrinsicCheckForUi = false;
            if(manualExtrinsicRecheckPending && !manualExtrinsicRecheckScreenShown) {
                // Render one CHECKING frame before the synchronous check occupies
                // the UI thread, so the operator sees the log and disabled Start.
                manualExtrinsicRecheckScreenShown = true;
                deferExtrinsicCheckForUi = true;
            }
            if(!cameraFaultActive && captureState == CaptureState::IDLE && cameraReadiness.allReady
               && selectedTaskSelectable && !extrinsicReadyChecked
               && !deferExtrinsicCheckForUi
               && (!initialCameraWarmupRequired || initialCameraWarmupCompleted)) {
                collectionSetStage("ui_extrinsic_ready_check");
                const bool isManualRecheck = manualExtrinsicRecheckPending;
                pushUiLog(isManualRecheck ? "Resampling camera extrinsics..."
                                          : "Checking camera extrinsics before READY...");

                const fs::path root = fs::path(trimString(cfgUi.saveRoot));
                const std::string subject = trimString(cfgUi.subjectId);
                const std::string taskName = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name;
                const int episodeN = capUi.currentEpisode;
                std::string checkLine;
                std::vector<std::string> checkDetails;
                extrinsicReadyAllowsStart = recorder.runExtrinsicHealthCheckBeforeStart(
                    root, subject, taskName, episodeN, &checkLine, &checkDetails, &extrinsicReadyStatus);
                extrinsicReadyChecked = true;
                if(isManualRecheck) {
                    manualExtrinsicRecheckPending = false;
                    manualExtrinsicRecheckScreenShown = false;
                    // Ignore clicks made while the synchronous sampler was running.
                    fm.clicked = false;
                    ms.clicked = false;
                    if(extrinsicReadyStatus == "pass") {
                        capUi.msg = "Extrinsic recheck passed";
                    }
                    else if(extrinsicReadyStatus == "warn") {
                        capUi.msg = "Extrinsic recheck warning; collection allowed";
                    }
                    else if(extrinsicReadyStatus == "inconclusive") {
                        capUi.msg = "Extrinsic recheck inconclusive; collection allowed";
                    }
                    else if(extrinsicReadyStatus == "fail") {
                        capUi.msg = "Extrinsic recheck failed; collection blocked";
                    }
                    else {
                        capUi.msg = "Extrinsic recheck error; collection blocked";
                    }
                }
                if(!checkLine.empty()) {
                    pushUiLog(checkLine);
                }
                for(const auto &line: checkDetails) {
                    pushUiLog(line);
                }
            }
            const bool readyForStart = cameraReadiness.allReady
                                    && selectedTaskSelectable
                                    && (extrinsicReadyChecked && extrinsicReadyAllowsStart);
            if(!cameraFaultActive && captureState == CaptureState::IDLE && cameraReadiness.allReady) {
                if(readyForStart && !cameraReadyAnnounced) {
                    announce("ready", "ready");
                    cameraReadyAnnounced = true;
                }
            }
            else if(!cameraFaultActive && captureState == CaptureState::IDLE && !cameraReadiness.allReady) {
                resetCameraReadyAnnouncement();
            }

            if(!cameraFaultActive && captureState == CaptureState::DRAINING && recorder.isDrainComplete()) {
                updateReadyState();
                if(pendingResetAfterDrain) {
                    pushUiLog("Reset requested. Review delete confirmation.");
                    enterDeleteConfirm();
                }
            }
            else if(cameraFaultActive && captureState == CaptureState::DRAINING && recorder.isDrainComplete()
                    && !cameraFaultDrainCompleteLogged) {
                pushUiLog("Faulted episode is ready to delete before restart.");
                cameraFaultDrainCompleteLogged = true;
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
            cv::putText(ui, "Current Task", cv::Point(taskPanel.x + 20, taskPanel.y + 32),
                        cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
            if(capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size())) {
                const auto &task = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)];
                auto nameLines = wrapTextToWidth(task.name, taskPanel.width - 40,
                                                 cv::FONT_HERSHEY_DUPLEX, 1.08, 2);
                int y = taskPanel.y + 76;
                for(size_t i = 0; i < nameLines.size() && i < 3; ++i) {
                    cv::putText(ui, nameLines[i], cv::Point(taskPanel.x + 20, y),
                                cv::FONT_HERSHEY_DUPLEX, 1.08,
                                selectedTaskClaimedByOther ? cv::Scalar(150, 150, 170) : (isTaskComplete(task) ? cv::Scalar(120, 220, 120) : cv::Scalar(255, 220, 50)),
                                2, cv::LINE_AA);
                    y += 40;
                }
                std::string ownerLabel = task.claimedBySubject.empty() ? std::string("another subject") : task.claimedBySubject;
                if(ownerLabel.size() > 24) {
                    ownerLabel = ownerLabel.substr(0, 21) + "...";
                }
                const std::string progressLine = "Progress " + std::to_string(task.completed)
                                               + " / " + std::to_string(task.total)
                                               + (selectedTaskClaimedByOther
                                                      ? ("  claimed by " + ownerLabel)
                                                      : (isTaskComplete(task) ? "  complete" : "  next episode after reserve"));
                cv::putText(ui, progressLine, cv::Point(taskPanel.x + 20, y + 8),
                            cv::FONT_HERSHEY_DUPLEX, 0.72, cv::Scalar(205, 205, 205), 1, cv::LINE_AA);

                const std::string &desc = task.description_cn.empty() ? task.description_en : task.description_cn;
                if(!desc.empty()) {
                    const int descLeft = taskPanel.x + 20;
                    const int descTop = y + 34;
                    const int descWidth = std::max(1, taskPanel.width - 40);
                    const int descMaxHeight = std::max(80, taskPanel.y + taskPanel.height - descTop - 20);
                    const int descFontH = choosePromptFontHeight(desc, descWidth, descMaxHeight, 48, 24);
                    const int lineGap = std::max(10, descFontH / 3);
                    const auto lines = wrapMultilineTextUtf8(desc, descWidth, descFontH);
                    int descY = descTop + descFontH;
                    const int descBottom = taskPanel.y + taskPanel.height - 18;
                    for(const auto &line: lines) {
                        if(descY > descBottom) {
                            break;
                        }
                        if(line.empty()) {
                            descY += lineGap;
                            continue;
                        }
                        putTextUtf8(ui, line, cv::Point(descLeft, descY),
                                    descFontH, cv::Scalar(220, 220, 220));
                        descY += descFontH + lineGap;
                    }
                }
            }
            else {
                const std::string prompt = capUi.tasks.empty() ? "No tasks from backend" : "Return to Tasks and select one task";
                cv::putText(ui, prompt, cv::Point(taskPanel.x + 20, taskPanel.y + 86),
                            cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
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
                if(!taskSelected) {
                    sd = {"SELECT TASK", cv::Scalar(255, 220, 80), cv::Scalar(70, 55, 20)};
                    stateEmphasisLine = "Choose one task from the backend task list";
                    stateFootnoteLine = cameraReadiness.message;
                }
                else if(selectedTaskClaimedByOther) {
                    sd = {"TASK CLAIMED", cv::Scalar(170, 170, 190), cv::Scalar(48, 35, 35)};
                    stateEmphasisLine = "Selected task belongs to another subject";
                    stateFootnoteLine = "Return to Tasks and select an available task";
                }
                else if(selectedTaskComplete) {
                    sd = {"TASK COMPLETE", cv::Scalar(120, 220, 120), cv::Scalar(30, 60, 30)};
                    stateEmphasisLine = "Selected task already reached total episodes";
                    stateFootnoteLine = "Select another task to continue";
                }
                else if(readyForStart) {
                    if(extrinsicReadyStatus == "inconclusive") {
                        sd = {"READY - INCONCLUSIVE", cv::Scalar(255, 220, 80), cv::Scalar(70, 55, 20)};
                        stateEmphasisLine = "Extrinsic check inconclusive; collection is allowed";
                        stateFootnoteLine = "Start now or use Resample Extrinsic Check";
                    }
                    else if(extrinsicReadyStatus == "warn") {
                        sd = {"READY - CHECK WARNING", cv::Scalar(255, 220, 80), cv::Scalar(70, 55, 20)};
                        stateEmphasisLine = "Extrinsic check warning; collection is allowed";
                        stateFootnoteLine = "Start now or use Resample Extrinsic Check";
                    }
                    else {
                        sd = {"READY", cv::Scalar(255, 255, 255), cv::Scalar(40, 40, 40)};
                        stateEmphasisLine = cameraReadiness.message;
                    }
                }
                else if(cameraReadiness.allReady) {
                    if(initialCameraWarmupRequired && !initialCameraWarmupCompleted) {
                        sd = {"WARMING UP", cv::Scalar(255, 220, 80), cv::Scalar(70, 55, 20)};
                        stateEmphasisLine = initialCameraWarmup.message;
                        stateFootnoteLine = "The first extrinsic check waits for 5 seconds of RGB and depth from every camera";
                    }
                    else if(extrinsicReadyChecked && !extrinsicReadyAllowsStart) {
                        if(extrinsicReadyStatus == "fail") {
                            sd = {"CHECK FAILED", cv::Scalar(80, 80, 255), cv::Scalar(55, 25, 25)};
                            stateEmphasisLine = "Extrinsic check failed; collection is blocked";
                            stateFootnoteLine = "Fix the setup, then use Resample Extrinsic Check";
                        }
                        else {
                            sd = {"CHECK ERROR", cv::Scalar(80, 80, 255), cv::Scalar(55, 25, 25)};
                            stateEmphasisLine = "Extrinsic check could not run";
                            stateFootnoteLine = "Fix the check error, then use Resample Extrinsic Check";
                        }
                    }
                    else {
                        sd = {"CHECKING", cv::Scalar(255, 220, 80), cv::Scalar(70, 55, 20)};
                        stateEmphasisLine = "Checking camera extrinsics...";
                        stateFootnoteLine = "Start is enabled after the extrinsic check passes";
                    }
                }
                else {
                    sd = {"WARMING UP", cv::Scalar(255, 220, 80), cv::Scalar(70, 55, 20)};
                    stateEmphasisLine = cameraReadiness.message;
                    stateFootnoteLine = "Start is enabled after streams are ready and extrinsic check passes";
                }
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
            case CaptureState::BACKEND_SYNC_PENDING:
                sd = {"BACKEND SYNC PENDING", cv::Scalar(80, 80, 255), cv::Scalar(55, 25, 25)};
                stateEmphasisLine = "Retry Confirm after the finalize request error is fixed";
                if(currentReservation.active) {
                    stateFootnoteLine = currentReservation.taskName + " episode_" + std::to_string(currentReservation.episodeNumber);
                }
                break;
            case CaptureState::DELETE_CONFIRM:
                sd = {"DELETE CONFIRM", cv::Scalar(255, 180, 80), cv::Scalar(80, 40, 20)};
                break;
            }
            if(cameraFaultActive && activeCameraFault.has_value()) {
                sd = {"CAMERA ERROR", cv::Scalar(80, 80, 255), cv::Scalar(55, 25, 25)};
                stateEmphasisLine = activeCameraFault->displayName.empty()
                    ? ("cam" + activeCameraFault->camKey + " " + dataTypeLabel(activeCameraFault->type) + " stalled")
                    : (activeCameraFault->displayName + " stalled");
                stateFootnoteLine = cameraFaultRestartBlocked ? "waiting for current episode to finish saving"
                                                              : "choose exit or delete + restart";
            }
            {
                const std::string uploadLine = latestUploadLine();
                if(!uploadLine.empty()
                   && captureState != CaptureState::RECORDING
                   && captureState != CaptureState::DRAINING
                   && captureState != CaptureState::DELETE_CONFIRM
                   && !cameraFaultActive) {
                    stateFootnoteLine = uploadLine;
                }
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
                stateFootnoteLine = elideTextToWidth(stateFootnoteLine, std::max(1, statusRect.width - 20),
                                                     cv::FONT_HERSHEY_DUPLEX, 0.55, 1);
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
                        const std::string uploadLine = latestUploadLine();
                        if(!uploadLine.empty()) {
                            head = uploadLine;
                        }
                    }
                    if(head.empty()) {
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
            const bool modalExit = exitConfirmActive;
            const bool modalFault = !modalExit && cameraFaultActive;
            const bool modalDelete = !modalExit && !modalFault && (captureState == CaptureState::DELETE_CONFIRM);
            const bool allowStart  = !modalFault && !modalDelete && (captureState == CaptureState::IDLE)
                                     && !modalExit && !currentReservation.active
                                     && !manualExtrinsicRecheckPending && readyForStart;
            const bool allowStop   = !modalFault && !modalDelete && !modalExit && (captureState == CaptureState::RECORDING);
            const bool allowSave   = !modalFault && !modalDelete && !modalExit
                                     && ((captureState == CaptureState::STOPPED_READY && recorder.hasData())
                                         || (captureState == CaptureState::BACKEND_SYNC_PENDING && currentReservation.active && currentReservation.localFinalized));
            const bool allowReset  = !modalFault && !modalDelete
                                     && !modalExit
                                     && (captureState == CaptureState::RECORDING
                                         || captureState == CaptureState::STOPPED_READY
                                         || (captureState == CaptureState::DRAINING && !pendingResetAfterDrain));
            const bool allowNav    = !modalFault && !modalDelete && !modalExit
                                     && (captureState == CaptureState::IDLE || captureState == CaptureState::BACKEND_SYNC_PENDING);
            const bool allowExtrinsicRecheck = !modalFault && !modalDelete && !modalExit
                                             && captureState == CaptureState::IDLE
                                             && !currentReservation.active
                                             && cameraReadiness.allReady
                                             && selectedTaskSelectable
                                             && extrinsicReadyChecked
                                             && cfg.extrinsicHealth.enabled
                                             && cfgUi.enableMultiview;

            // --- 右侧按钮区 ---
            const int btnX = taskPanelX + 20;
            const int btnW = taskPanelW - 40;
            cv::Rect bExtrinsicRecheck(btnX, winH - 330, btnW, 38);
            cv::Rect bStart(btnX, winH - 280, btnW, 50);
            cv::Rect bStop (btnX, winH - 220, btnW, 50);
            cv::Rect bSave (btnX, winH - 160, btnW, 50);
            cv::Rect bReset(btnX, winH - 100, btnW, 50);
            cv::Rect bMenu (btnX, winH - 38,  btnW / 2 - 5, 30);
            cv::Rect bTasks(btnX + btnW / 2 + 5, winH - 38, btnW / 2 - 5, 30);

            std::string startLabel = "Start  [Ctrl+1]";
            if(!allowStart && captureState == CaptureState::IDLE && taskSelected) {
                if(selectedTaskClaimedByOther) {
                    startLabel = "Start (claimed)";
                }
                else if(selectedTaskComplete) {
                    startLabel = "Start (task complete)";
                }
                else if(!cameraReadiness.allReady) {
                    startLabel = "Start (warming up)";
                }
                else if(initialCameraWarmupRequired && !initialCameraWarmupCompleted) {
                    startLabel = "Start (RGB/depth warm-up)";
                }
                else if(extrinsicReadyChecked && !extrinsicReadyAllowsStart) {
                    startLabel = extrinsicReadyStatus == "fail" ? "Start (check failed)" : "Start (check error)";
                }
                else {
                    startLabel = "Start (checking)";
                }
            }
            bool doExtrinsicRecheck = uiButtonEx(ui, bExtrinsicRecheck, "Resample Extrinsic Check", fm, allowExtrinsicRecheck);
            bool doStart    = uiButtonEx(ui, bStart, startLabel, fm, allowStart);
            bool doStop     = uiButtonEx(ui, bStop,  "Stop   [Ctrl+2]", fm, allowStop);
            std::string saveLabel = "Confirm [Ctrl+3]";
            if(captureState == CaptureState::BACKEND_SYNC_PENDING) {
                saveLabel = "Retry Confirm [Ctrl+3]";
            }
            bool doSave     = uiButtonEx(ui, bSave, saveLabel, fm, allowSave);
            bool doReset    = uiButtonEx(ui, bReset, "Reset  [Ctrl+4]", fm, allowReset);
            bool doBackMenu = uiButtonEx(ui, bMenu,  "Menu",            fm, allowNav);
            bool doBackTasks = uiButtonEx(ui, bTasks,"Tasks",           fm, allowNav);
            bool doDeleteConfirm = false;
            bool doDeleteCancel  = false;
            bool doFaultExit = false;
            bool doFaultRestart = false;
            bool doExitConfirm = false;
            bool doExitCancel = false;

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
                    if(modalExit) {
                        if(baseKey == '1') {
                            doExitConfirm = true;
                        }
                        else if(baseKey == '4') {
                            doExitCancel = true;
                        }
                    }
                    else if(modalFault) {
                        if(baseKey == '1') {
                            doFaultExit = true;
                        }
                        else if(baseKey == '2') {
                            doFaultRestart = !cameraFaultRestartBlocked;
                        }
                    }
                    else if(modalDelete) {
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

            if(modalFault && activeCameraFault.has_value()) {
                const std::string detailLine = cameraFaultRestartBlocked
                    ? "Waiting for current episode data to finish saving. Restart will be enabled after that."
                    : "Restart will delete the current episode, restart all cameras, and return to READY.";
                const auto actions = drawCameraFaultModal(ui, fm, *activeCameraFault,
                                                          !cameraFaultRestartBlocked,
                                                          detailLine);
                doFaultExit = doFaultExit || actions.exitCollection;
                doFaultRestart = doFaultRestart || actions.restartCapture;
            }

            if(modalExit) {
                const auto actions = drawExitConfirmModal(ui, fm,
                                                          pendingExitAction,
                                                          selectedTaskName(),
                                                          completedThisCollection,
                                                          captureStateName(captureState, currentReservation, cameraFaultActive),
                                                          currentReservation);
                doExitConfirm = doExitConfirm || actions.confirm;
                doExitCancel = doExitCancel || actions.cancel;
            }

            // --- 处理按钮动作 ---
            if(doExitCancel) {
                exitConfirmActive = false;
                pendingExitAction = PendingExitAction::None;
                pendingExitDeleteFaultEpisode = false;
                capUi.msg = "Exit canceled";
                pushUiLog("Exit canceled");
            }
            if(doExitConfirm) {
                performConfirmedExit();
            }
            if(doFaultExit) {
                collectionSetStage("ui_camera_fault_exit");
                announce("fault_exit", "exit");
                requestExit(PendingExitAction::ExitCollection, true);
            }
            if(doFaultRestart && running) {
                collectionSetStage("ui_camera_fault_restart");
                if(recorder.hasCurrentSession() && !recorder.isDrainComplete()) {
                    capUi.msg = "Waiting for current episode to finish saving...";
                    pushUiLog("Restart delayed: current episode is still saving.");
                    announce("fault_restart_wait", "waiting");
                }
                else {
                    bool okToRestart = true;
                    std::string error;
                    const bool hadSession = recorder.hasCurrentSession();
                    if(hadSession && !recorder.discardCurrentSession(&error)) {
                        okToRestart = false;
                        capUi.msg = "Delete failed";
                        pushUiLog("Delete failed before camera restart: " + error);
                        announce("delete_failed", "delete failed");
                    }
                    if(okToRestart && currentReservation.active && !currentReservation.localFinalized) {
                        okToRestart = releaseCurrentReservation("camera restart");
                    }

                    if(okToRestart) {
                        if(hadSession) {
                            pushUiLog("Faulted episode deleted.");
                        }
                        recorder.stopIfRunning(false);
                        latestFrameCache.clear();

                        std::string startError;
                        bool restarted = recorder.start(cfgUi);
                        if(!restarted) {
                            startError = recorder.lastInfoLine();
                            if(startError.empty()) {
                                startError = "Failed to restart cameras";
                            }
                        }

                        if(restarted) {
                            activeCameraFault.reset();
                            recorder.clearCameraStreamFault();
                            cameraFaultDrainCompleteLogged = false;
                            pendingResetAfterDrain = false;
                            captureState = CaptureState::IDLE;
                            resetCameraReadyAnnouncement();
                            capUi.msg = "Cameras restarted. Press Start to retry.";
                            pushUiLog("Cameras restarted. Ready to start current episode again.");
                            announce("fault_restart", "restart");
                        }
                        else {
                            recorder.stopIfRunning(false);
                            capUi.msg = "Restart failed";
                            pushUiLog("Restart failed: " + startError);
                            announce("fault_restart_failed", "restart failed");
                        }
                    }
                }
            }
            if(doExtrinsicRecheck) {
                collectionSetStage("ui_extrinsic_recheck_requested");
                resetCameraReadyAnnouncement();
                manualExtrinsicRecheckPending = true;
                manualExtrinsicRecheckScreenShown = false;
                capUi.msg = "Resampling extrinsic check...";
                pushUiLog("Manual extrinsic resample requested.");
            }
            if(doBackMenu) {
                collectionSetStage("ui_capture_back_menu");
                announce("menu", "menu");
                requestExit(PendingExitAction::ExitCollection, false);
            }
            if(doBackTasks) {
                collectionSetStage("ui_capture_back_tasks");
                requestExit(PendingExitAction::ReturnTaskSelect, false);
            }
            if(doStart) {
                collectionSetStage("ui_capture_start");
                cfgUi.enforceRules();
                resetDrainStatusTracking();
                const fs::path root = fs::path(trimString(cfgUi.saveRoot));
                const std::string subject = trimString(cfgUi.subjectId);
                const std::string taskName = capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)].name;
                TaskEpisodeReservation reservation;
                std::string backendError;
                bool ok = backendClient.reserveEpisode(collectionClientId, subject, taskName, collectionOperatorId, reservation, &backendError);
                if(!ok) {
                    capUi.msg = "Backend reserve failed: " + backendError;
                    pushUiLog(capUi.msg);
                    announce("start_failed", "start failed");
                }
                if(ok) {
                    currentReservation.clear();
                    currentReservation.active = true;
                    currentReservation.reservationId = reservation.reservationId;
                    currentReservation.taskName = reservation.taskName;
                    currentReservation.storageName = reservation.storageName;
                    currentReservation.episodeNumber = reservation.episodeNumber;
                    currentReservation.collectionPath = root / subject / reservation.taskName
                                                        / ("episode_" + std::to_string(reservation.episodeNumber));
                    currentReservation.idempotencyKey = makeIdempotencyKey(collectionClientId,
                                                                           reservation.reservationId,
                                                                           reservation.episodeNumber);
                    capUi.currentEpisode = reservation.episodeNumber;
                    pushUiLog("Backend reserved: " + reservation.taskName
                              + " ep" + std::to_string(reservation.episodeNumber));

                    ok = recorder.start(cfgUi);
                    if(ok) {
                        ok = recorder.beginRecord(root, subject, reservation.taskName, reservation.episodeNumber);
                    }
                }
                if(!ok) {
                    if(capUi.msg.empty() || capUi.msg.find("Backend reserve failed") == std::string::npos) {
                        capUi.msg = "Failed to start capture";
                    }
                    pushUiLog("Start failed");
                    announce("start_failed", "start failed");
                    {
                        const std::string line = recorder.lastInfoLine();
                        if(!line.empty()) {
                            pushUiLog(line);
                        }
                    }
                    if(currentReservation.active && !currentReservation.localFinalized) {
                        (void)releaseCurrentReservation("start failed");
                    }
                }
                else {
                    captureState = CaptureState::RECORDING;
                    pendingResetAfterDrain = false;
                    resetDrainStatusTracking();
                    capUi.msg.clear();
                    startRecordingTick();
                    announce("start", "start");
                    pushUiLog("Recording: " + currentReservation.taskName
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
                stopRecordingTick();
                recorder.stopRecording();
                announce("stop", "stop");
                if(recorder.isDrainComplete()) {
                    updateReadyState();
                }
                else {
                    captureState = CaptureState::DRAINING;
                    beginDrainStatusTracking();
                    capUi.msg = "Saving data to disk...";
                    pushUiLog("Stopped. Waiting for background save to finish.");
                }
            }
            if(doReset) {
                collectionSetStage("ui_capture_reset");
                announce("reset", "reset");
                if(captureState == CaptureState::RECORDING) {
                    stopRecordingTick();
                    recorder.stopRecording();
                    pendingResetAfterDrain = true;
                    if(recorder.isDrainComplete()) {
                        updateReadyState();
                        pushUiLog("Reset requested. Review delete confirmation.");
                        enterDeleteConfirm();
                    }
                    else {
                        captureState = CaptureState::DRAINING;
                        beginDrainStatusTracking();
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
                bool localConfirmed = currentReservation.localFinalized;
                const double sessionDurationSeconds = recorder.lastRecordedSeconds();
                const int sessionFrameCount = recorder.currentSessionFrameCount();
                if(!localConfirmed) {
                    localConfirmed = recorder.confirmCurrentSession();
                    if(localConfirmed) {
                        currentReservation.localFinalized = true;
                        currentReservation.durationSeconds = sessionDurationSeconds;
                        currentReservation.frameCount = sessionFrameCount;
                    }
                }
                if(!localConfirmed) {
                    capUi.msg = "Confirm failed: session not ready";
                    pushUiLog("Confirm failed");
                    announce("confirm_failed", "confirm failed");
                }
                else {
                    const double maxDiff = recorder.lastAlignedMaxDiffMs();
                    const bool finalizeAccepted = cfg.taskBackend.nas.enabled
                        ? enqueueCaptureNasFinalizeForCurrentReservation()
                        : confirmReservationWithBackend();
                    if(finalizeAccepted) {
                        if(maxDiff > 0.0) {
                            std::ostringstream oss;
                            oss.setf(std::ios::fixed);
                            oss << "MaxTsDiff: " << std::setprecision(3) << maxDiff << " ms";
                            pushUiLog(oss.str());
                        }
                        if(capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size())
                           && isTaskComplete(capUi.tasks[static_cast<size_t>(capUi.currentTaskIdx)])) {
                            capUi.msg = "Task complete";
                            pushUiLog("Confirm OK. Selected task complete.");
                            announce("task_complete", "selected task episodes complete");
                        }
                        else if(capUi.currentTaskIdx >= 0 && capUi.currentTaskIdx < static_cast<int>(capUi.tasks.size())) {
                            capUi.msg = cfg.taskBackend.nas.enabled
                                ? "Capture confirmed; NAS upload running in background"
                                : "Capture confirmed";
                            pushUiLog(cfg.taskBackend.nas.enabled
                                ? "Confirm accepted. Same task can continue while capture-side NAS upload runs."
                                : "Confirm OK. Same task next backend episode will be reserved on Start.");
                            announce("confirm", "confirm");
                        }
                        else {
                            capUi.msg = "Capture confirmed";
                            pushUiLog("Confirm OK. Select a task.");
                            announce("confirm", "confirm");
                        }
                        recorder.clearStatus();
                        captureState = CaptureState::IDLE;
                        pendingResetAfterDrain = false;
                        resetDrainStatusTracking();
                        resetCameraReadyAnnouncement();
                    }
                    else {
                        captureState = CaptureState::BACKEND_SYNC_PENDING;
                        pendingResetAfterDrain = false;
                        announce("confirm_failed", "confirm failed");
                    }
                }
            }
            if(doDeleteConfirm) {
                collectionSetStage("ui_capture_delete_confirm");
                std::string error;
                if(recorder.discardCurrentSession(&error)) {
                    recorder.clearStatus();
                    const bool releaseOk = releaseCurrentReservation("reset");
                    captureState = CaptureState::IDLE;
                    pendingResetAfterDrain = false;
                    resetDrainStatusTracking();
                    resetCameraReadyAnnouncement();
                    if(releaseOk) {
                        capUi.msg = "Capture discarded";
                        pushUiLog("Reset OK. Current episode discarded.");
                        announce("reset_confirm", "reset confirmed");
                    }
                    else {
                        announce("delete_failed", "delete failed");
                    }
                }
                else {
                    capUi.msg = "Delete failed";
                    pushUiLog("Delete failed: " + error);
                    announce("delete_failed", "delete failed");
                }
            }
            if(doDeleteCancel) {
                collectionSetStage("ui_capture_delete_cancel");
                captureState = CaptureState::STOPPED_READY;
                pendingResetAfterDrain = false;
                resetDrainStatusTracking();
                capUi.msg = "Delete canceled";
                pushUiLog("Delete canceled");
                announce("reset_cancel", "reset canceled");
            }

            // --- 超时自动停止 ---
            if(captureState == CaptureState::RECORDING && recorder.autoStopIfTimeout()) {
                stopRecordingTick();
                if(recorder.isDrainComplete()) {
                    updateReadyState();
                    capUi.msg = "Auto-stopped by max duration";
                }
                else {
                    captureState = CaptureState::DRAINING;
                    beginDrainStatusTracking();
                    capUi.msg = "Auto-stopped. Saving data to disk...";
                }
                pushUiLog("Auto stop by max duration");
            }

            updateRecordingTick();
            updateDrainStatusTracking();

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
                std::string info;
                if(captureState == CaptureState::DRAINING) {
                    info = recorder.drainStatusLine();
                }
                if(info.empty()) {
                    info = recorder.lastInfoLine();
                }
                if(!info.empty()) {
                    cv::putText(ui, info, cv::Point(4, winH - 20),
                                cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
                }
            }
        }

        if((page == CollectionPage::Config || page == CollectionPage::TaskSelect) && exitConfirmActive) {
            bool doExitConfirm = false;
            bool doExitCancel = false;
            if(key > 0) {
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld = g_ctrlShortcutListening || ctrlFromMask;
                const int baseKey = key & 0xFFFF;
                if(ctrlHeld) {
                    if(baseKey == '1') {
                        doExitConfirm = true;
                    }
                    else if(baseKey == '4') {
                        doExitCancel = true;
                    }
                }
            }
            const auto actions = drawExitConfirmModal(ui, fm,
                                                      pendingExitAction,
                                                      selectedTaskName(),
                                                      completedThisCollection,
                                                      captureStateName(captureState, currentReservation, activeCameraFault.has_value()),
                                                      currentReservation);
            doExitConfirm = doExitConfirm || actions.confirm;
            doExitCancel = doExitCancel || actions.cancel;
            if(doExitCancel) {
                exitConfirmActive = false;
                pendingExitAction = PendingExitAction::None;
                pendingExitDeleteFaultEpisode = false;
            }
            if(doExitConfirm) {
                performConfirmedExit();
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
