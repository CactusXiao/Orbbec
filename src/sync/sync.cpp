// Copyright (c) Orbbec Inc. All Rights Reserved.
// Licensed under the MIT License.

#include "shared_utils.hpp"

#include "interactive_visualization.hpp"
#include "viewer.hpp"
#include "collection.hpp"
#include "calibration.hpp"

#include <cmath>
#include <utility>

namespace sync_app {

struct CvMouseState {
    int x = 0;
    int y = 0;
    bool clicked = false;
    int clickX = 0;
    int clickY = 0;
    int wheelDelta = 0;
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
        s->clickX = x;
        s->clickY = y;
    }
    else if(event == cv::EVENT_MOUSEWHEEL) {
        s->wheelDelta += cv::getMouseWheelDelta(flags);
    }
}

struct FrameMouse {
    int x = 0;
    int y = 0;
    bool clicked = false;
    int clickX = 0;
    int clickY = 0;
    int wheelDelta = 0;
};

static FrameMouse beginFrame(CvMouseState &ms) {
    FrameMouse fm;
    fm.x = ms.x;
    fm.y = ms.y;
    fm.clicked = ms.clicked;
    fm.clickX = ms.clickX;
    fm.clickY = ms.clickY;
    fm.wheelDelta = ms.wheelDelta;
    ms.clicked = false;
    ms.wheelDelta = 0;
    return fm;
}

static bool uiButton(cv::Mat &img, const cv::Rect &r, const std::string &label, FrameMouse &ms) {
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar bg = hover ? cv::Scalar(60, 60, 60) : cv::Scalar(40, 40, 40);
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
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
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

static std::string toStringInt(int v) {
    return std::to_string(v);
}

static std::string toStringFloat(float v) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(3) << v;
    return oss.str();
}

static int parseIntOr(const std::string &s, int fallback) {
    try {
        return std::stoi(s);
    }
    catch(...) {
        return fallback;
    }
}

static float parseFloatOr(const std::string &s, float fallback) {
    try {
        return std::stof(s);
    }
    catch(...) {
        return fallback;
    }
}

static float parseExposureMsOrAuto(const std::string &s, float fallback) {
    const std::string trimmed = trimString(s);
    if(trimmed.empty()) {
        return 0.0f;
    }
    try {
        const float v = std::stof(trimmed);
        if(!std::isfinite(v)) {
            return fallback;
        }
        if(v <= 0.0f) {
            return 0.0f;
        }
        return std::max(0.05f, std::min(100.0f, v));
    }
    catch(...) {
        return fallback;
    }
}

static void applyPointCloudResolution(AppConfig &cfg, int w, int h, int fps) {
    for(auto &d: cfg.devices) {
        for(auto &s: d.streams) {
            if(s.type == StreamType::PointCloud) {
                if(w > 0) {
                    s.width = w;
                }
                if(h > 0) {
                    s.height = h;
                }
                if(fps > 0) {
                    s.fps = fps;
                }
            }
        }
    }
}

enum class AppPage {
    Menu,
    InteractionConfig,
    Placeholder
};

enum class PlaceholderMode {
    Viewer,
    Collection,
    Calibration
};

struct InteractionConfigUi {
    AppConfig defaults;
    std::string viewerFps;
    std::string maxDepth;
    bool differentColor = false;
    bool colorfulCloudPoints = false;
    std::string exposureMs;
    std::string pcWidth;
    std::string pcHeight;
    std::string pcFps;
    std::string pcDecimation;
    std::string confThreshold;
    std::string decimationFilterScale;
    std::string noiseRemovalMaxSize;
    std::string noiseRemovalMinDiff;
    std::string spatialAlpha;
    std::string spatialDispDiff;
    std::string spatialMagnitude;
    std::string spatialRadius;
    std::string smoothThreshold;
    std::string temporalDiffScale;
    std::string temporalWeight;
    std::string holeFillingMode;
    bool deskCrop = false;
    std::string preset;

    bool showAdvanced = false;

    std::string activeField;
    int scrollY = 0;

    void loadFromCfg(const AppConfig &cfg) {
        viewerFps = toStringInt(cfg.viewerFps);
        maxDepth = toStringFloat(cfg.maxDepth);
        differentColor = cfg.differentColor;
        colorfulCloudPoints = cfg.colorfulCloudPoints;
        exposureMs = cfg.colorExposureMs > 0.0f ? toStringFloat(cfg.colorExposureMs) : std::string();
        deskCrop = cfg.filters.deskCrop;
        pcDecimation = toStringInt(cfg.filters.pointCloudDecimationFactor);
        confThreshold = toStringFloat(static_cast<float>(cfg.filters.confThreshold));
        decimationFilterScale = toStringInt(cfg.filters.decimationFilterScale);
        noiseRemovalMaxSize = toStringInt(cfg.filters.noiseRemovalMaxSize);
        noiseRemovalMinDiff = toStringInt(cfg.filters.noiseRemovalMinDiff);
        spatialAlpha = toStringFloat(static_cast<float>(cfg.filters.spatialAlpha));
        spatialDispDiff = toStringInt(cfg.filters.spatialDispDiff);
        spatialMagnitude = toStringInt(cfg.filters.spatialMagnitude);
        spatialRadius = toStringInt(cfg.filters.spatialRadius);
        smoothThreshold = toStringFloat(static_cast<float>(cfg.filters.smoothThresholdM));
        temporalDiffScale = toStringFloat(static_cast<float>(cfg.filters.temporalDiffScale));
        temporalWeight = toStringFloat(static_cast<float>(cfg.filters.temporalWeight));
        holeFillingMode = toStringInt(cfg.filters.holeFillingMode);
        preset = cfg.filters.preset.empty() ? std::string("Default") : cfg.filters.preset;
        showAdvanced = false;

        int w = 0, h = 0, f = 0;
        for(const auto &d: cfg.devices) {
            for(const auto &s: d.streams) {
                if(s.type == StreamType::PointCloud) {
                    w = s.width;
                    h = s.height;
                    f = s.fps;
                    break;
                }
            }
            if(w > 0 || h > 0 || f > 0) {
                break;
            }
        }
        pcWidth = toStringInt(w);
        pcHeight = toStringInt(h);
        pcFps = toStringInt(f);
    }

    void setDefaults(const AppConfig &cfg) {
        defaults = cfg;
        loadFromCfg(cfg);
    }

    AppConfig buildCfg() const {
        AppConfig cfg = defaults;
        cfg.viewerFps = std::max(1, parseIntOr(viewerFps, cfg.viewerFps));
        cfg.maxDepth = std::max(0.1f, parseFloatOr(maxDepth, cfg.maxDepth));
        cfg.differentColor = differentColor;
        cfg.colorfulCloudPoints = colorfulCloudPoints;
        cfg.colorExposureMs = parseExposureMsOrAuto(exposureMs, cfg.colorExposureMs);
        cfg.filters.deskCrop = deskCrop;
        cfg.filters.pointCloudDecimationFactor = std::max(0, parseIntOr(pcDecimation, cfg.filters.pointCloudDecimationFactor));
        cfg.filters.confThreshold = std::max(0.0, std::min(1.0, static_cast<double>(parseFloatOr(confThreshold, static_cast<float>(cfg.filters.confThreshold)))));
        cfg.filters.decimationFilterScale = std::max(0, parseIntOr(decimationFilterScale, cfg.filters.decimationFilterScale));
        cfg.filters.noiseRemovalMaxSize = std::max(0, parseIntOr(noiseRemovalMaxSize, cfg.filters.noiseRemovalMaxSize));
        cfg.filters.noiseRemovalMinDiff = std::max(0, parseIntOr(noiseRemovalMinDiff, cfg.filters.noiseRemovalMinDiff));
        cfg.filters.spatialAlpha = std::max(0.0, static_cast<double>(parseFloatOr(spatialAlpha, static_cast<float>(cfg.filters.spatialAlpha))));
        cfg.filters.spatialDispDiff = std::max(0, parseIntOr(spatialDispDiff, cfg.filters.spatialDispDiff));
        cfg.filters.spatialMagnitude = std::max(0, parseIntOr(spatialMagnitude, cfg.filters.spatialMagnitude));
        cfg.filters.spatialRadius = std::max(0, parseIntOr(spatialRadius, cfg.filters.spatialRadius));
        cfg.filters.smoothThresholdM = std::max(0.0, static_cast<double>(parseFloatOr(smoothThreshold, static_cast<float>(cfg.filters.smoothThresholdM))));
        cfg.filters.temporalDiffScale = std::max(0.0, static_cast<double>(parseFloatOr(temporalDiffScale, static_cast<float>(cfg.filters.temporalDiffScale))));
        cfg.filters.temporalWeight = std::max(0.0, static_cast<double>(parseFloatOr(temporalWeight, static_cast<float>(cfg.filters.temporalWeight))));
        cfg.filters.holeFillingMode = std::max(0, parseIntOr(holeFillingMode, cfg.filters.holeFillingMode));
        cfg.filters.preset = trimString(preset);
        applyPointCloudResolution(cfg, std::max(0, parseIntOr(pcWidth, 0)), std::max(0, parseIntOr(pcHeight, 0)), std::max(0, parseIntOr(pcFps, 0)));
        cfg.mode = "interaction";
        return cfg;
    }
};

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

class MenuPicoConnection {
public:
    explicit MenuPicoConnection(EgoModuleConfig cfg)
        : cfg_(std::move(cfg)) {
        if(!cfg_.enabled) {
            return;
        }
        if(cfg_.maxBufferedFrames == 0) {
            cfg_.maxBufferedFrames = 8192;
        }
        worker_ = std::thread([this]() {
            std::string error;
            const bool ok = recorder_.start(cfg_, &error);
            if(!ok) {
                std::lock_guard<std::mutex> lock(errorMtx_);
                startupError_ = error.empty() ? "unknown error" : error;
            }
            startupOk_.store(ok);
            startupDone_.store(true);
        });
    }

    ~MenuPicoConnection() {
        waitForStartup();
        recorder_.stop();
    }

    MenuPicoConnection(const MenuPicoConnection &) = delete;
    MenuPicoConnection &operator=(const MenuPicoConnection &) = delete;

    EgoRecorder *recorder() {
        return &recorder_;
    }

    void waitForStartup() {
        if(worker_.joinable()) {
            worker_.join();
        }
    }

    std::string statusLine() const {
        std::string line;
        if(!cfg_.enabled) {
            line = "PICO: disabled in config";
        }
        else if(!startupDone_.load()) {
            line = "PICO: starting server on " + endpoint();
        }
        else if(!startupOk_.load()) {
            std::lock_guard<std::mutex> lock(errorMtx_);
            line = "PICO: start failed - " + startupError_;
        }
        else if(!recorder_.isRunning()) {
            line = "PICO: server stopped";
        }
        else if(recorder_.isConnected()) {
            std::string hello = recorder_.lastHelloSummary();
            if(hello.size() > 54) {
                hello = hello.substr(0, 51) + "...";
            }
            line = hello.empty() ? "PICO: connected" : ("PICO: connected " + hello);
        }
        else {
            line = "PICO: waiting on " + endpoint();
        }
        return compactLine(line);
    }

    cv::Scalar statusColor() const {
        if(!cfg_.enabled) {
            return cv::Scalar(160, 160, 160);
        }
        if(!startupDone_.load()) {
            return cv::Scalar(80, 190, 255);
        }
        if(!startupOk_.load() || !recorder_.isRunning()) {
            return cv::Scalar(70, 70, 255);
        }
        if(recorder_.isConnected()) {
            return cv::Scalar(90, 220, 120);
        }
        return cv::Scalar(80, 200, 255);
    }

private:
    std::string endpoint() const {
        return cfg_.host + ":" + std::to_string(cfg_.port);
    }

    static std::string compactLine(const std::string &line) {
        constexpr size_t kMaxChars = 108;
        if(line.size() <= kMaxChars) {
            return line;
        }
        return line.substr(0, kMaxChars - 3) + "...";
    }

    EgoModuleConfig cfg_;
    EgoRecorder recorder_;
    std::thread worker_;
    std::atomic_bool startupDone_{ false };
    std::atomic_bool startupOk_{ false };
    mutable std::mutex errorMtx_;
    std::string startupError_;
};

}  // namespace sync_app

int main(int argc, char **argv) {
    using namespace sync_app;
    try {
        fs::path configPath;
        if(argc > 1) {
            configPath = fs::path(argv[1]);
        }
        else if(fs::exists("config.json")) {
            configPath = fs::path("config.json");
        }
        else {
            configPath = fs::path("src/sync/config.json");
        }
        const fs::path configPathAbs = fs::absolute(configPath);
        std::cerr << "Using config: " << configPathAbs.string() << std::endl;
        AppConfig baseCfg = loadConfig(configPathAbs);
        MenuPicoConnection menuPico(baseCfg.ego);

        InteractionConfigUi cfgUi;
        cfgUi.setDefaults(baseCfg);

        AppPage page = AppPage::Menu;
        PlaceholderMode placeholderMode = PlaceholderMode::Viewer;

        const std::string winName = "Sync";
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 900, 640);

        CvMouseState ms;
        cv::setMouseCallback(winName, mouseThunk, &ms);

        std::atomic_bool modeCancel{ false };
        std::thread modeThread;
        std::string menuNotice;
        std::string menuError;

        auto stopPlaceholderMode = [&]() {
            modeCancel.store(true);
            if(modeThread.joinable()) {
                modeThread.join();
            }
            modeCancel.store(false);
        };

        bool running = true;
        while(running) {
            const int key = cv::waitKey(1);
            auto fm = beginFrame(ms);
            if(key == 27) {
                stopPlaceholderMode();
                running = false;
                break;
            }

            cv::Mat ui(640, 900, CV_8UC3, cv::Scalar(20, 20, 20));

            if(page == AppPage::Menu) {
                cv::putText(ui, "Sync Menu", cv::Point(24, 48), cv::FONT_HERSHEY_DUPLEX, 1.1, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                cv::putText(ui, menuPico.statusLine(), cv::Point(24, 78), cv::FONT_HERSHEY_DUPLEX, 0.56, menuPico.statusColor(), 1, cv::LINE_AA);
                cv::Rect b1(60, 105, 780, 76);
                cv::Rect b2(60, 193, 780, 76);
                cv::Rect b3(60, 281, 780, 76);
                cv::Rect b4(60, 369, 780, 76);
                cv::Rect b5(60, 457, 780, 76);
                if(uiButton(ui, b1, "Interaction", fm)) {
                    menuNotice.clear();
                    menuError.clear();
                    page = AppPage::InteractionConfig;
                }
                if(uiButton(ui, b2, "Viewer", fm)) {
                    menuNotice.clear();
                    menuError.clear();
                    stopPlaceholderMode();
                    page = AppPage::Placeholder;
                    placeholderMode = PlaceholderMode::Viewer;
                    modeThread = std::thread([&]() {
                        AppConfig cfg = baseCfg;
                        cfg.mode = "viewer";
                        run_viewer(cfg, &modeCancel);
                    });
                }
                if(uiButton(ui, b3, "Collection", fm)) {
                    menuNotice.clear();
                    menuError.clear();
                    stopPlaceholderMode();
                    menuPico.waitForStartup();
                    AppConfig cfg = baseCfg;
                    cfg.mode      = "collection";
                    cv::destroyWindow(winName);
                    run_collection(cfg, nullptr, menuPico.recorder());
                    page = AppPage::Menu;
                    cv::namedWindow(winName, cv::WINDOW_NORMAL);
                    cv::resizeWindow(winName, 900, 640);
                    cv::setMouseCallback(winName, mouseThunk, &ms);
                }
                if(uiButton(ui, b4, "Calibration", fm)) {
                    menuNotice.clear();
                    menuError.clear();
                    stopPlaceholderMode();
                    page = AppPage::Menu;
                    AppConfig cfg = baseCfg;
                    cfg.mode = "calibration";
                    cv::destroyWindow(winName);
                    run_calibration(cfg, nullptr);
                    page = AppPage::Menu;
                    cv::namedWindow(winName, cv::WINDOW_NORMAL);
                    cv::resizeWindow(winName, 900, 640);
                    cv::setMouseCallback(winName, mouseThunk, &ms);
                    continue;
                }
                if(uiButton(ui, b5, "Label", fm)) {
                    stopPlaceholderMode();
                    std::string detail;
                    if(launchManualLabelFrontend(baseCfg.taskBackend.baseUrl, "", &detail)) {
                        menuError.clear();
                        menuNotice = "Label frontend opened. Log: " + detail;
                    }
                    else {
                        menuNotice.clear();
                        menuError = "Label frontend failed: " + detail;
                    }
                }
                if(!menuError.empty()) {
                    const std::string line = ellipsizeTextToWidth(menuError, 780, cv::FONT_HERSHEY_DUPLEX, 0.62, 2);
                    cv::putText(ui, line, cv::Point(60, 575), cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(70, 70, 255), 2, cv::LINE_AA);
                }
                else if(!menuNotice.empty()) {
                    const std::string line = ellipsizeTextToWidth(menuNotice, 780, cv::FONT_HERSHEY_DUPLEX, 0.62, 1);
                    cv::putText(ui, line, cv::Point(60, 575), cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(90, 210, 90), 1, cv::LINE_AA);
                }
            }
            else if(page == AppPage::InteractionConfig) {
                cv::putText(ui, "Interaction - Config", cv::Point(24, 48), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

                const int left = 60;
                const int top = 100;
                const int rowH = 56;
                const int fieldW = 320;

                const cv::Rect scrollArea(20, 70, ui.cols - 40, 480);
                if(fm.wheelDelta != 0 && scrollArea.contains(cv::Point(fm.x, fm.y))) {
                    cfgUi.scrollY += (fm.wheelDelta > 0) ? -rowH : rowH;
                }

                if(key == 'j' || key == 'J') {
                    cfgUi.scrollY += rowH;
                }
                else if(key == 'k' || key == 'K') {
                    cfgUi.scrollY -= rowH;
                }

                const int rowsCount = cfgUi.showAdvanced ? 19 : 8;
                const int contentH = rowsCount * rowH + 120;
                const int maxScroll = std::max(0, contentH - scrollArea.height);
                cfgUi.scrollY = std::max(0, std::min(maxScroll, cfgUi.scrollY));

                const int y0 = top - cfgUi.scrollY;
                auto rowRect = [&](int row) { return cv::Rect(left, y0 + row * rowH, fieldW, 36); };
                auto rowRectR = [&](int row) { return cv::Rect(420, y0 + row * rowH, 400, 36); };

                if(uiTextField(ui, rowRect(0), "viewer_fps", cfgUi.viewerFps, cfgUi.activeField == "viewer_fps", fm)) {
                    cfgUi.activeField = "viewer_fps";
                }
                if(uiTextField(ui, rowRect(1), "max_depth", cfgUi.maxDepth, cfgUi.activeField == "max_depth", fm)) {
                    cfgUi.activeField = "max_depth";
                }
                if(uiTextField(ui, rowRect(2), "pc_width", cfgUi.pcWidth, cfgUi.activeField == "pc_w", fm)) {
                    cfgUi.activeField = "pc_w";
                }
                if(uiTextField(ui, rowRect(3), "pc_height", cfgUi.pcHeight, cfgUi.activeField == "pc_h", fm)) {
                    cfgUi.activeField = "pc_h";
                }
                if(uiTextField(ui, rowRect(4), "pc_fps", cfgUi.pcFps, cfgUi.activeField == "pc_fps", fm)) {
                    cfgUi.activeField = "pc_fps";
                }
                if(uiTextField(ui, rowRect(5), "point_cloud_decimation_factor", cfgUi.pcDecimation, cfgUi.activeField == "pc_dec", fm)) {
                    cfgUi.activeField = "pc_dec";
                }
                if(uiTextField(ui, rowRect(6), "conf_threshold (0~1)", cfgUi.confThreshold, cfgUi.activeField == "conf", fm)) {
                    cfgUi.activeField = "conf";
                }
                if(uiTextField(ui, rowRect(7), "exposure_ms (blank/0 auto)", cfgUi.exposureMs, cfgUi.activeField == "exp", fm)) {
                    cfgUi.activeField = "exp";
                }

                if(cfgUi.showAdvanced) {
                    if(uiTextField(ui, rowRect(8), "decimation_filter_scale", cfgUi.decimationFilterScale, cfgUi.activeField == "dec_scale", fm)) {
                        cfgUi.activeField = "dec_scale";
                    }
                    if(uiTextField(ui, rowRect(9), "noise_removal_filter_max_size", cfgUi.noiseRemovalMaxSize, cfgUi.activeField == "nr_max", fm)) {
                        cfgUi.activeField = "nr_max";
                    }
                    if(uiTextField(ui, rowRect(10), "noise_removal_filter_min_diff", cfgUi.noiseRemovalMinDiff, cfgUi.activeField == "nr_min", fm)) {
                        cfgUi.activeField = "nr_min";
                    }

                    if(uiTextField(ui, rowRect(11), "spatial_filter_alpha", cfgUi.spatialAlpha, cfgUi.activeField == "sp_a", fm)) {
                        cfgUi.activeField = "sp_a";
                    }
                    if(uiTextField(ui, rowRect(12), "spatial_filter_disp_diff", cfgUi.spatialDispDiff, cfgUi.activeField == "sp_d", fm)) {
                        cfgUi.activeField = "sp_d";
                    }
                    if(uiTextField(ui, rowRect(13), "spatial_filter_magnitude", cfgUi.spatialMagnitude, cfgUi.activeField == "sp_m", fm)) {
                        cfgUi.activeField = "sp_m";
                    }
                    if(uiTextField(ui, rowRect(14), "spatial_filter_radius", cfgUi.spatialRadius, cfgUi.activeField == "sp_r", fm)) {
                        cfgUi.activeField = "sp_r";
                    }

                    if(uiTextField(ui, rowRect(15), "smooth_threshold", cfgUi.smoothThreshold, cfgUi.activeField == "smooth", fm)) {
                        cfgUi.activeField = "smooth";
                    }
                    if(uiTextField(ui, rowRect(16), "temporal_filter_diff_scale", cfgUi.temporalDiffScale, cfgUi.activeField == "t_d", fm)) {
                        cfgUi.activeField = "t_d";
                    }
                    if(uiTextField(ui, rowRect(17), "temporal_filter_weight", cfgUi.temporalWeight, cfgUi.activeField == "t_w", fm)) {
                        cfgUi.activeField = "t_w";
                    }
                    if(uiTextField(ui, rowRect(18), "hole_filling_filter_mode", cfgUi.holeFillingMode, cfgUi.activeField == "hf", fm)) {
                        cfgUi.activeField = "hf";
                    }
                }

                if(uiCheckbox(ui, rowRectR(0), cfgUi.differentColor, "different_color", fm)) {
                    cfgUi.differentColor = !cfgUi.differentColor;
                }
                if(uiCheckbox(ui, rowRectR(1), cfgUi.deskCrop, "desk_crop", fm)) {
                    cfgUi.deskCrop = !cfgUi.deskCrop;
                }
                if(uiCheckbox(ui, rowRectR(2), cfgUi.colorfulCloudPoints, "colorful_cloud_points", fm)) {
                    cfgUi.colorfulCloudPoints = !cfgUi.colorfulCloudPoints;
                }

                if(uiCheckbox(ui, rowRectR(3), cfgUi.showAdvanced, "show_advanced_filters", fm)) {
                    cfgUi.showAdvanced = !cfgUi.showAdvanced;
                    cfgUi.scrollY = 0;
                }

                cv::Rect rPreset(420, y0 + 3 * rowH, 400, 60);
                cv::rectangle(ui, rPreset, cv::Scalar(30, 30, 30), cv::FILLED);
                cv::rectangle(ui, rPreset, cv::Scalar(120, 120, 120), 1);
                cv::putText(ui, "preset (click to cycle)", cv::Point(rPreset.x, rPreset.y - 6), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
                cv::putText(ui, cfgUi.preset, cv::Point(rPreset.x + 10, rPreset.y + 38), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                if(fm.clicked && rPreset.contains(cv::Point(fm.clickX, fm.clickY))) {
                    fm.clicked = false;
                    static const std::vector<std::string> presets = { "Default", "Hand", "High Accuracy", "High Density", "Medium Density", "Custom" };
                    auto it = std::find(presets.begin(), presets.end(), cfgUi.preset);
                    if(it == presets.end() || (it + 1) == presets.end()) {
                        cfgUi.preset = presets.front();
                    }
                    else {
                        cfgUi.preset = *(it + 1);
                    }
                }

                cv::Rect bBack(60, 560, 220, 60);
                cv::Rect bReset(300, 560, 220, 60);
                cv::Rect bStart(540, 560, 300, 60);
                if(uiButton(ui, bBack, "Back to Menu", fm)) {
                    cfgUi.activeField.clear();
                    page = AppPage::Menu;
                }
                if(uiButton(ui, bReset, "Reset Defaults", fm)) {
                    cfgUi.loadFromCfg(cfgUi.defaults);
                    cfgUi.activeField.clear();
                }
                if(uiButton(ui, bStart, "Start Visualization", fm)) {
                    AppConfig cfg = cfgUi.buildCfg();
                    menuPico.waitForStartup();
                    cv::destroyWindow(winName);
                    const auto ex = run_interactive_visualization(cfg, nullptr, menuPico.recorder());
                    if(ex == InteractiveExit::ReturnMenu) {
                        page = AppPage::Menu;
                    }
                    else if(ex == InteractiveExit::ReturnConfig) {
                        cfgUi.defaults = cfg;
                        cfgUi.loadFromCfg(cfg);
                        page = AppPage::InteractionConfig;
                    }
                    else {
                        page = AppPage::Menu;
                    }
                    cv::namedWindow(winName, cv::WINDOW_NORMAL);
                    cv::resizeWindow(winName, 900, 640);
                    cv::setMouseCallback(winName, mouseThunk, &ms);
                }

                if(!cfgUi.activeField.empty() && key > 0) {
                    if(cfgUi.activeField == "viewer_fps") {
                        handleTextInput(cfgUi.viewerFps, key);
                    }
                    else if(cfgUi.activeField == "max_depth") {
                        handleTextInput(cfgUi.maxDepth, key);
                    }
                    else if(cfgUi.activeField == "pc_w") {
                        handleTextInput(cfgUi.pcWidth, key);
                    }
                    else if(cfgUi.activeField == "pc_h") {
                        handleTextInput(cfgUi.pcHeight, key);
                    }
                    else if(cfgUi.activeField == "pc_fps") {
                        handleTextInput(cfgUi.pcFps, key);
                    }
                    else if(cfgUi.activeField == "pc_dec") {
                        handleTextInput(cfgUi.pcDecimation, key);
                    }
                    else if(cfgUi.activeField == "conf") {
                        handleTextInput(cfgUi.confThreshold, key);
                    }
                    else if(cfgUi.activeField == "exp") {
                        handleTextInput(cfgUi.exposureMs, key);
                    }
                    else if(cfgUi.activeField == "dec_scale") {
                        handleTextInput(cfgUi.decimationFilterScale, key);
                    }
                    else if(cfgUi.activeField == "nr_max") {
                        handleTextInput(cfgUi.noiseRemovalMaxSize, key);
                    }
                    else if(cfgUi.activeField == "nr_min") {
                        handleTextInput(cfgUi.noiseRemovalMinDiff, key);
                    }
                    else if(cfgUi.activeField == "sp_a") {
                        handleTextInput(cfgUi.spatialAlpha, key);
                    }
                    else if(cfgUi.activeField == "sp_d") {
                        handleTextInput(cfgUi.spatialDispDiff, key);
                    }
                    else if(cfgUi.activeField == "sp_m") {
                        handleTextInput(cfgUi.spatialMagnitude, key);
                    }
                    else if(cfgUi.activeField == "sp_r") {
                        handleTextInput(cfgUi.spatialRadius, key);
                    }
                    else if(cfgUi.activeField == "smooth") {
                        handleTextInput(cfgUi.smoothThreshold, key);
                    }
                    else if(cfgUi.activeField == "t_d") {
                        handleTextInput(cfgUi.temporalDiffScale, key);
                    }
                    else if(cfgUi.activeField == "t_w") {
                        handleTextInput(cfgUi.temporalWeight, key);
                    }
                    else if(cfgUi.activeField == "hf") {
                        handleTextInput(cfgUi.holeFillingMode, key);
                    }
                }
            }
            else {
                cv::putText(ui, "Other UI logic TODO", cv::Point(24, 60), cv::FONT_HERSHEY_DUPLEX, 1.1, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                cv::putText(ui, "Mode is running with existing CLI behavior", cv::Point(24, 110), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
                cv::Rect bBack(60, 560, 220, 60);
                if(uiButton(ui, bBack, "Back to Menu", fm)) {
                    stopPlaceholderMode();
                    page = AppPage::Menu;
                }
                (void)placeholderMode;
            }

            cv::imshow(winName, ui);
        }

        stopPlaceholderMode();
        return 0;
    }
    catch(const ob::Error &e) {
        std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs() << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
        return 1;
    }
    catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
