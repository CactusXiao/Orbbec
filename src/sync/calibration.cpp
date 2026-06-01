#include "calibration.hpp"

#include "utils/utils.hpp"

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include <Eigen/SVD>

#include <condition_variable>

#if defined(__unix__) || defined(__APPLE__)
#include <poll.h>
#include <unistd.h>
#endif

namespace sync_app {

static std::string readFileAllLocal(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path.string());
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr removeDominantPlaneRansacCompat(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                                                                           int maxIterations,
                                                                           double distanceThreshold,
                                                                           double minInlierRatio,
                                                                           int minInliersAbs) {
    if(!in || in->empty()) {
        return pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    const size_t minInliers = static_cast<size_t>(std::max(minInliersAbs, static_cast<int>(static_cast<double>(in->size()) * minInlierRatio)));
    return removeDominantPlaneRansac(in, maxIterations, distanceThreshold, minInliers);
}

class VisualCalibrator {
public:
    explicit VisualCalibrator(AppConfig cfg, const std::atomic_bool *cancel)
        : cfg_(std::move(cfg)), cancel_(cancel) {}

    int run() {
        auto deviceList = ctx_.queryDeviceList();
        if(!deviceList || deviceList->deviceCount() == 0) {
            return 1;
        }
        devices_ = selectDevices(deviceList);
        if(devices_.empty()) {
            return 1;
        }
        if(cfg_.enableSync) {
            applySyncConfig(devices_);
            ctx_.enableDeviceClockSync(60000);
        }

        cbRows_ = std::to_string(std::max(3, cfg_.calibration.chessboard.rows));
        cbCols_ = std::to_string(std::max(3, cfg_.calibration.chessboard.cols));
        icpIter_ = "300";
        icpStop_ = "0.005";
        icpStopRot_ = "0.005";
        {
            std::ostringstream oss;
            oss << std::setprecision(2) << std::fixed << (cfg_.maxDepth > 0.0f ? cfg_.maxDepth : 6.0f);
            icpMaxDepth_ = oss.str();
        }
        pushLog("calibration ui ready");

        const std::string win = "Calibration";
        cv::namedWindow(win, cv::WINDOW_NORMAL);
        cv::resizeWindow(win, 1600, 900);
        cv::setMouseCallback(win, mouseThunk, &mouse_);
        startSampleWorker();

        bool running = true;
        while(running) {
            if(cancel_ && cancel_->load()) {
                break;
            }
            const int key = cv::waitKey(1);
            if(key == 27) {
                running = false;
            }
            FrameMouse fm = beginFrame();
            cv::Mat canvas(900, 1600, CV_8UC3, cv::Scalar(16, 16, 16));
            drawFrame(canvas, fm, key, running);
            cv::imshow(win, canvas);
        }
        stopSampleWorker();
        stopActivePair();
        cv::destroyWindow(win);
        return 0;
    }

private:
    struct DeviceRuntimeLite {
        DeviceConfig cfg;
        std::shared_ptr<ob::Device> dev;
    };

    struct PairPipes {
        std::shared_ptr<ob::Pipeline> p1;
        std::shared_ptr<ob::Pipeline> p2;
        cv::Mat K1;
        cv::Mat D1;
        cv::Mat K2;
        cv::Mat D2;
        int colorW = 0;
        int colorH = 0;
        int colorFps1 = 0;
        int colorFps2 = 0;
    };

    struct Sample {
        uint64_t ts1 = 0;
        uint64_t ts2 = 0;
        cv::Mat img1;
        cv::Mat img2;
    };

    struct SampleJob {
        std::string pairKey;
        Sample      sample;
    };

    struct PairData {
        std::vector<Sample> samples;
        int validCount = 0;
        double rmsPx = -1.0;
        bool calibrated = false;
        cv::Mat latestDet1;
        cv::Mat latestDet2;
        cv::Mat K1;
        cv::Mat D1;
        cv::Mat K2;
        cv::Mat D2;
        int colorW = 0;
        int colorH = 0;
    };

    struct PairPreviewBuffer {
        std::string camKey;
        std::shared_ptr<ob::Frame> latestRgbFrame;
        uint64_t latestRgbFrameTsUs = 0;
        cv::Mat latestRgb;
        uint64_t latestRgbTsUs = 0;
    };

    struct EdgeExtrinsic {
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d t{ 0.0, 0.0, 0.0 };
        bool valid = false;
    };

    struct IcpDevice {
        std::string index;
        std::shared_ptr<ob::Device> dev;
        std::shared_ptr<ob::Pipeline> pipe;
        std::shared_ptr<ob::PointCloudFilter> pcFilter;
        OrbbecDepthFilterChain depthFilters;
    };

    struct IcpDepthSlot {
        std::shared_ptr<ob::Frame> latestDepthFrame;
        uint64_t latestDepthTsUs = 0;
    };

    struct MouseState {
        int x = 0;
        int y = 0;
        int wheel = 0;
        bool clicked = false;
        int clickX = 0;
        int clickY = 0;
    };

    struct FrameMouse {
        int x = 0;
        int y = 0;
        int wheel = 0;
        bool clicked = false;
        int clickX = 0;
        int clickY = 0;
    };

    enum class UIMode {
        Chessboard,
        ICP
    };

    static std::string pairKey(const std::string &a, const std::string &b) {
        return a + "->" + b;
    }

    static std::string presetLabel(int w, int h, int fps) {
        return std::to_string(w) + "x" + std::to_string(h) + "@" + std::to_string(fps);
    }

    static void blitKeepAspect(cv::Mat &dst, const cv::Rect &r, const cv::Mat &src) {
        if(src.empty() || r.width <= 0 || r.height <= 0) {
            return;
        }
        const double sx = static_cast<double>(r.width) / static_cast<double>(src.cols);
        const double sy = static_cast<double>(r.height) / static_cast<double>(src.rows);
        const double s = std::min(sx, sy);
        const int w = std::max(1, static_cast<int>(std::round(static_cast<double>(src.cols) * s)));
        const int h = std::max(1, static_cast<int>(std::round(static_cast<double>(src.rows) * s)));
        const int x = r.x + (r.width - w) / 2;
        const int y = r.y + (r.height - h) / 2;
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        resized.copyTo(dst(cv::Rect(x, y, w, h)));
    }

    static void drawRgbGrid(cv::Mat &ui, const cv::Rect &r, const std::vector<std::pair<std::string, cv::Mat>> &frames, int w, int h, int fps) {
        cv::rectangle(ui, r, cv::Scalar(30, 30, 30), cv::FILLED);
        cv::rectangle(ui, r, cv::Scalar(120, 120, 120), 1);
        if(frames.empty()) {
            cv::putText(ui, "No RGB frames", cv::Point(r.x + 12, r.y + 30), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            return;
        }
        const int n = static_cast<int>(frames.size());
        int cols = 1;
        int rows = 1;
        if(n == 2) {
            cols = 1;
            rows = 2;
        }
        else {
            cols = 1;
            while(cols * cols < n) {
                cols++;
            }
            rows = static_cast<int>((n + cols - 1) / cols);
        }
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
                cv::rectangle(ui, inner, cv::Scalar(10, 10, 10), cv::FILLED);
                blitKeepAspect(ui, inner, frames[i].second);
            }
        }
    }

    static void mouseThunk(int event, int x, int y, int flags, void *userdata) {
        auto *s = reinterpret_cast<MouseState *>(userdata);
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
            s->wheel += cv::getMouseWheelDelta(flags);
        }
    }

    FrameMouse beginFrame() {
        FrameMouse fm;
        fm.x = mouse_.x;
        fm.y = mouse_.y;
        fm.wheel = mouse_.wheel;
        fm.clicked = mouse_.clicked;
        fm.clickX = mouse_.clickX;
        fm.clickY = mouse_.clickY;
        mouse_.wheel = 0;
        mouse_.clicked = false;
        return fm;
    }

    static bool uiButton(cv::Mat &img, const cv::Rect &r, const std::string &label, FrameMouse &fm) {
        const bool hover = r.contains(cv::Point(fm.x, fm.y));
        cv::rectangle(img, r, hover ? cv::Scalar(65, 65, 65) : cv::Scalar(45, 45, 45), cv::FILLED);
        cv::rectangle(img, r, cv::Scalar(120, 120, 120), 1);
        cv::putText(img, label, cv::Point(r.x + 10, r.y + r.height / 2 + 7), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        if(fm.clicked && r.contains(cv::Point(fm.clickX, fm.clickY))) {
            fm.clicked = false;
            return true;
        }
        return false;
    }

    static bool parseExtrinsicObject(cJSON *obj, EdgeExtrinsic &out) {
        if(!obj || !cJSON_IsObject(obj)) {
            return false;
        }
        auto *rotArr = cJSON_GetObjectItemCaseSensitive(obj, "rotation");
        auto *tArr = cJSON_GetObjectItemCaseSensitive(obj, "translation");
        cv::Matx33d R;
        cv::Vec3d t;
        if(!parseMat3d(rotArr, R) || !parseVec3d(tArr, t)) {
            return false;
        }
        out.valid = true;
        out.R = R;
        out.t = t;
        return true;
    }

    static EdgeExtrinsic edgeFromD2CTransform(const OBD2CTransform &tf) {
        EdgeExtrinsic ex;
        ex.valid = true;
        ex.R = cv::Matx33d(tf.rot[0], tf.rot[1], tf.rot[2],
                           tf.rot[3], tf.rot[4], tf.rot[5],
                           tf.rot[6], tf.rot[7], tf.rot[8]);
        ex.t = cv::Vec3d(tf.trans[0], tf.trans[1], tf.trans[2]);
        return ex;
    }

    static cJSON *makeExtrinsicJson(const EdgeExtrinsic &ex) {
        cJSON *obj = cJSON_CreateObject();
        cJSON *rot = cJSON_CreateArray();
        for(int y = 0; y < 3; y++) {
            cJSON *row = cJSON_CreateArray();
            for(int x = 0; x < 3; x++) {
                cJSON_AddItemToArray(row, cJSON_CreateNumber(ex.R(y, x)));
            }
            cJSON_AddItemToArray(rot, row);
        }
        cJSON *t = cJSON_CreateArray();
        cJSON_AddItemToArray(t, cJSON_CreateNumber(ex.t[0]));
        cJSON_AddItemToArray(t, cJSON_CreateNumber(ex.t[1]));
        cJSON_AddItemToArray(t, cJSON_CreateNumber(ex.t[2]));
        cJSON_AddItemToObject(obj, "rotation", rot);
        cJSON_AddItemToObject(obj, "translation", t);
        return obj;
    }

    static cJSON *toJsonMat3(const cv::Matx33d &R) {
        cJSON *rot = cJSON_CreateArray();
        for(int y = 0; y < 3; y++) {
            cJSON *row = cJSON_CreateArray();
            for(int x = 0; x < 3; x++) {
                cJSON_AddItemToArray(row, cJSON_CreateNumber(R(y, x)));
            }
            cJSON_AddItemToArray(rot, row);
        }
        return rot;
    }

    static cJSON *toJsonVec3(const cv::Vec3d &t) {
        cJSON *arr = cJSON_CreateArray();
        cJSON_AddItemToArray(arr, cJSON_CreateNumber(t[0]));
        cJSON_AddItemToArray(arr, cJSON_CreateNumber(t[1]));
        cJSON_AddItemToArray(arr, cJSON_CreateNumber(t[2]));
        return arr;
    }

    static bool uiField(cv::Mat &img,
                        const cv::Rect &r,
                        const std::string &label,
                        std::string &value,
                        const std::string &activeId,
                        const std::string &id,
                        FrameMouse &fm) {
        const bool hover = r.contains(cv::Point(fm.x, fm.y));
        const bool active = activeId == id;
        cv::rectangle(img, r, cv::Scalar(30, 30, 30), cv::FILLED);
        cv::rectangle(img, r, active ? cv::Scalar(80, 220, 120) : (hover ? cv::Scalar(180, 180, 180) : cv::Scalar(110, 110, 110)), 1);
        cv::putText(img, label, cv::Point(r.x, r.y - 6), cv::FONT_HERSHEY_DUPLEX, 0.46, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        cv::putText(img, value, cv::Point(r.x + 7, r.y + r.height - 9), cv::FONT_HERSHEY_DUPLEX, 0.54, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        if(fm.clicked && r.contains(cv::Point(fm.clickX, fm.clickY))) {
            fm.clicked = false;
            return true;
        }
        return false;
    }

    static void handleText(std::string &value, int key, int maxLen = 24) {
        if(key == 8 || key == 127) {
            if(!value.empty()) {
                value.pop_back();
            }
            return;
        }
        if(key == 13 || key == 10 || key == 27) {
            return;
        }
        if(key >= 32 && key <= 126) {
            if(static_cast<int>(value.size()) < maxLen) {
                value.push_back(static_cast<char>(key));
            }
        }
    }

    static int parseIntBound(const std::string &s, int fallback, int lo, int hi) {
        try {
            int v = std::stoi(trimString(s));
            return std::max(lo, std::min(hi, v));
        }
        catch(...) {
            return std::max(lo, std::min(hi, fallback));
        }
    }

    static double parseDoubleBound(const std::string &s, double fallback, double lo, double hi) {
        try {
            double v = std::stod(trimString(s));
            return std::max(lo, std::min(hi, v));
        }
        catch(...) {
            return std::max(lo, std::min(hi, fallback));
        }
    }

    void pushLog(const std::string &s) {
        std::cout << "[LOG] " << s << std::endl;
        std::lock_guard<std::mutex> lock(logMtx_);
        logs_.push_back(s);
        if(static_cast<int>(logs_.size()) > 500) {
            logs_.erase(logs_.begin(), logs_.begin() + (static_cast<int>(logs_.size()) - 500));
        }
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
            ts = frame->timeStampUs();
        }
        return ts;
    }

    std::vector<DeviceRuntimeLite> selectDevices(const std::shared_ptr<ob::DeviceList> &deviceList) {
        std::unordered_map<std::string, std::shared_ptr<ob::Device>> bySn;
        for(uint32_t i = 0; i < deviceList->deviceCount(); i++) {
            auto dev = deviceList->getDevice(i);
            bySn.emplace(std::string(dev->getDeviceInfo()->serialNumber()), dev);
        }
        std::vector<DeviceRuntimeLite> out;
        for(const auto &dc: cfg_.devices) {
            auto it = bySn.find(dc.sn);
            if(it == bySn.end()) {
                continue;
            }
            DeviceRuntimeLite rt;
            rt.cfg = dc;
            rt.dev = it->second;
            out.push_back(std::move(rt));
        }
        return out;
    }

    void applySyncConfig(std::vector<DeviceRuntimeLite> &devices) {
        for(size_t i = 0; i < devices.size(); i++) {
            auto &rt = devices[i];
            auto cur = rt.dev->getMultiDeviceSyncConfig();
            auto cfg = rt.cfg.hasSyncConfig ? rt.cfg.syncConfig : cur;
            if(!rt.cfg.hasSyncConfig) {
                const bool isPrimary = (i == 0);
                cfg.syncMode = isPrimary ? OB_MULTI_DEVICE_SYNC_MODE_PRIMARY : OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED;
                cfg.triggerOutEnable = isPrimary;
                cfg.triggerOutDelayUs = 0;
            }
            if(cfg.syncMode != OB_MULTI_DEVICE_SYNC_MODE_PRIMARY) {
                cfg.triggerOutEnable = false;
            }
            if(cfg.framesPerTrigger <= 0) {
                cfg.framesPerTrigger = 1;
            }
            cur.syncMode = cfg.syncMode;
            cur.depthDelayUs = cfg.depthDelayUs;
            cur.colorDelayUs = cfg.colorDelayUs;
            cur.trigger2ImageDelayUs = cfg.trigger2ImageDelayUs;
            cur.triggerOutEnable = cfg.triggerOutEnable;
            cur.triggerOutDelayUs = cfg.triggerOutDelayUs;
            cur.framesPerTrigger = cfg.framesPerTrigger;
            rt.dev->setMultiDeviceSyncConfig(cur);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const DeviceRuntimeLite *findByIndex(const std::string &idx) const {
        for(const auto &d: devices_) {
            if(d.cfg.index == idx) {
                return &d;
            }
        }
        return nullptr;
    }

    bool hasIndex(const std::string &idx) const {
        return findByIndex(idx) != nullptr;
    }

    static cv::Mat toCameraMatrix(const OBCameraIntrinsic &in) {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = static_cast<double>(in.fx);
        K.at<double>(1, 1) = static_cast<double>(in.fy);
        K.at<double>(0, 2) = static_cast<double>(in.cx);
        K.at<double>(1, 2) = static_cast<double>(in.cy);
        return K;
    }

    static cv::Mat toDistCoeffs(const OBCameraDistortion &d) {
        cv::Mat coeffs = cv::Mat::zeros(1, 8, CV_64F);
        coeffs.at<double>(0, 0) = static_cast<double>(d.k1);
        coeffs.at<double>(0, 1) = static_cast<double>(d.k2);
        coeffs.at<double>(0, 2) = static_cast<double>(d.p1);
        coeffs.at<double>(0, 3) = static_cast<double>(d.p2);
        coeffs.at<double>(0, 4) = static_cast<double>(d.k3);
        coeffs.at<double>(0, 5) = static_cast<double>(d.k4);
        coeffs.at<double>(0, 6) = static_cast<double>(d.k5);
        coeffs.at<double>(0, 7) = static_cast<double>(d.k6);
        return coeffs;
    }

    static int preferredProfileFormatScore(OBSensorType sensorType, OBFormat format) {
        if(sensorType == OB_SENSOR_COLOR) {
            if(format == OB_FORMAT_RGB) {
                return 0;
            }
            if(format == OB_FORMAT_BGR) {
                return 1;
            }
            if(format == OB_FORMAT_MJPG) {
                return 2;
            }
            if(format == OB_FORMAT_YUYV || format == OB_FORMAT_YUY2 || format == OB_FORMAT_UYVY) {
                return 3;
            }
            return 10;
        }
        if(sensorType == OB_SENSOR_DEPTH) {
            if(format == OB_FORMAT_Y16 || format == OB_FORMAT_Z16) {
                return 0;
            }
            if(format == OB_FORMAT_Y14) {
                return 1;
            }
            if(format == OB_FORMAT_RLE) {
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
                auto p = list->getProfile(i);
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
                    best = vp;
                }
            }
            return best;
        };

        if(auto best = findBest(width, height, fps)) {
            return best;
        }

        if(width > 0 || height > 0) {
            std::vector<std::pair<int, int>> fallbacks;
            if(sensorType == OB_SENSOR_DEPTH) {
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

    bool startPair(const DeviceRuntimeLite &a, const DeviceRuntimeLite &b) {
        stopActivePair();
        activePair_.p1 = std::make_shared<ob::Pipeline>(a.dev);
        activePair_.p2 = std::make_shared<ob::Pipeline>(b.dev);
        const int targetFps = std::max(1, cfg_.viewerFps > 0 ? cfg_.viewerFps : 30);
        const int targetW = 1280;
        const int targetH = 720;
        auto c1 = pickVideoProfile(activePair_.p1, OB_SENSOR_COLOR, targetW, targetH, targetFps);
        auto c2 = pickVideoProfile(activePair_.p2, OB_SENSOR_COLOR, targetW, targetH, targetFps);
        if(!c1 || !c2) {
            stopActivePair();
            return false;
        }
        auto d1 = pickVideoProfile(activePair_.p1, OB_SENSOR_DEPTH, 0, 0, 0);
        auto d2 = pickVideoProfile(activePair_.p2, OB_SENSOR_DEPTH, 0, 0, 0);
        if(!d1 || !d2) {
            stopActivePair();
            return false;
        }
        auto cfg1 = std::make_shared<ob::Config>();
        cfg1->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
        cfg1->enableStream(c1);
        auto cfg2 = std::make_shared<ob::Config>();
        cfg2->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
        cfg2->enableStream(c2);
        {
            std::lock_guard<std::mutex> lock(previewMtx_);
            previewBuffers_.clear();
            previewBuffers_.emplace(a.cfg.index, PairPreviewBuffer{ a.cfg.index });
            previewBuffers_.emplace(b.cfg.index, PairPreviewBuffer{ b.cfg.index });
            activeCam1_ = a.cfg.index;
            activeCam2_ = b.cfg.index;
        }
        const auto mode1 = a.dev->getMultiDeviceSyncConfig().syncMode;
        const auto mode2 = b.dev->getMultiDeviceSyncConfig().syncMode;
        const bool p1IsPrimary = (mode1 == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY);
        const bool p2IsSecondary = (mode2 == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY || mode2 == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED);
        if(p1IsPrimary && p2IsSecondary) {
            activePair_.p2->start(cfg2, [this, idx = b.cfg.index](std::shared_ptr<ob::FrameSet> fs) { onPairFrameSet(idx, fs); });
            activePair_.p1->start(cfg1, [this, idx = a.cfg.index](std::shared_ptr<ob::FrameSet> fs) { onPairFrameSet(idx, fs); });
        }
        else {
            activePair_.p1->start(cfg1, [this, idx = a.cfg.index](std::shared_ptr<ob::FrameSet> fs) { onPairFrameSet(idx, fs); });
            activePair_.p2->start(cfg2, [this, idx = b.cfg.index](std::shared_ptr<ob::FrameSet> fs) { onPairFrameSet(idx, fs); });
        }
        const auto cp1 = activePair_.p1->getCameraParamWithProfile(c1->getWidth(), c1->getHeight(), d1->getWidth(), d1->getHeight());
        const auto cp2 = activePair_.p2->getCameraParamWithProfile(c2->getWidth(), c2->getHeight(), d2->getWidth(), d2->getHeight());
        EdgeExtrinsic d2c1 = edgeFromD2CTransform(cp1.transform);
        EdgeExtrinsic d2c2 = edgeFromD2CTransform(cp2.transform);
        depthToRgbByCam_[a.cfg.index] = d2c1;
        depthToRgbByCam_[b.cfg.index] = d2c2;
        rgbToDepthByCam_[a.cfg.index] = fromEigenWorldToCam(invertRigid(toEigenWorldToCam(d2c1)));
        rgbToDepthByCam_[b.cfg.index] = fromEigenWorldToCam(invertRigid(toEigenWorldToCam(d2c2)));
        activePair_.K1 = toCameraMatrix(cp1.rgbIntrinsic);
        activePair_.D1 = toDistCoeffs(cp1.rgbDistortion);
        activePair_.K2 = toCameraMatrix(cp2.rgbIntrinsic);
        activePair_.D2 = toDistCoeffs(cp2.rgbDistortion);
        activePair_.colorW = static_cast<int>(c1->getWidth());
        activePair_.colorH = static_cast<int>(c1->getHeight());
        activePair_.colorFps1 = static_cast<int>(c1->getFps());
        activePair_.colorFps2 = static_cast<int>(c2->getFps());
        return true;
    }

    void stopActivePair() {
        try {
            if(activePair_.p1) {
                activePair_.p1->stop();
            }
        }
        catch(...) {
        }
        try {
            if(activePair_.p2) {
                activePair_.p2->stop();
            }
        }
        catch(...) {
        }
        activePair_ = PairPipes{};
        pairStreaming_ = false;
        {
            std::lock_guard<std::mutex> lock(previewMtx_);
            previewBuffers_.clear();
            activeCam1_.clear();
            activeCam2_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(sampleMtx_);
            sampleQueue_.clear();
        }
    }

    void onPairFrameSet(const std::string &camKey, const std::shared_ptr<ob::FrameSet> &fs) {
        if(!fs) {
            return;
        }
        auto frame = fs->colorFrame();
        if(!frame) {
            return;
        }
        uint64_t ts = 0;
        const bool requireGlobalTs = cfg_.enableSync && devices_.size() > 1;
        try {
            ts = frame->globalTimeStampUs();
        }
        catch(...) {
            ts = 0;
        }
        if(ts == 0 && !requireGlobalTs) {
            try {
                ts = frame->timeStampUs();
            }
            catch(...) {
                ts = 0;
            }
        }
        std::lock_guard<std::mutex> lock(previewMtx_);
        auto it = previewBuffers_.find(camKey);
        if(it == previewBuffers_.end()) {
            return;
        }
        it->second.latestRgbFrame = frame;
        it->second.latestRgbFrameTsUs = ts;
    }

    std::unordered_map<std::string, cv::Mat> latestRgbFramesFromPairImpl() {
        struct Pending {
            std::shared_ptr<ob::Frame> frame;
            uint64_t                   tsUs = 0;
        };
        std::unordered_map<std::string, cv::Mat> out;
        std::unordered_map<std::string, Pending> pending;
        {
            std::lock_guard<std::mutex> lock(previewMtx_);
            pending.reserve(previewBuffers_.size());
            for(auto &kv: previewBuffers_) {
                auto &buf = kv.second;
                if(!buf.latestRgb.empty() && buf.latestRgbTsUs == buf.latestRgbFrameTsUs) {
                    out.emplace(kv.first, buf.latestRgb);
                    continue;
                }
                if(buf.latestRgbFrame) {
                    pending.emplace(kv.first, Pending{ buf.latestRgbFrame, buf.latestRgbFrameTsUs });
                }
                else if(!buf.latestRgb.empty()) {
                    out.emplace(kv.first, buf.latestRgb);
                }
            }
        }
        for(auto &kv: pending) {
            cv::Mat img;
            try {
                img = visualizeObFrame(kv.second.frame);
            }
            catch(...) {
                img.release();
            }
            if(img.empty()) {
                continue;
            }
            std::lock_guard<std::mutex> lock(previewMtx_);
            auto it = previewBuffers_.find(kv.first);
            if(it == previewBuffers_.end()) {
                continue;
            }
            if(it->second.latestRgbFrameTsUs == kv.second.tsUs) {
                it->second.latestRgb = img;
                it->second.latestRgbTsUs = kv.second.tsUs;
                it->second.latestRgbFrame.reset();
            }
            if(!it->second.latestRgb.empty()) {
                out.emplace(kv.first, it->second.latestRgb);
            }
        }
        return out;
    }

    void updateLiveFramesFromPreview() {
        const auto latest = latestRgbFramesFromPairImpl();
        auto it1 = latest.find(activeCam1_);
        if(it1 != latest.end()) {
            live1_ = it1->second;
        }
        auto it2 = latest.find(activeCam2_);
        if(it2 != latest.end()) {
            live2_ = it2->second;
        }
    }

    std::vector<std::pair<std::string, cv::Mat>> latestRgbFramesForGrid() {
        std::vector<std::pair<std::string, cv::Mat>> frames;
        std::lock_guard<std::mutex> lock(previewMtx_);
        if(!activeCam1_.empty()) {
            auto it = previewBuffers_.find(activeCam1_);
            if(it != previewBuffers_.end() && !it->second.latestRgb.empty()) {
                frames.emplace_back(it->first, it->second.latestRgb);
            }
        }
        if(!activeCam2_.empty()) {
            auto it = previewBuffers_.find(activeCam2_);
            if(it != previewBuffers_.end() && !it->second.latestRgb.empty()) {
                frames.emplace_back(it->first, it->second.latestRgb);
            }
        }
        return frames;
    }

    bool captureSyncedColorFromPreview(Sample &out, uint64_t maxDiffUs) {
        std::lock_guard<std::mutex> lock(previewMtx_);
        auto it1 = previewBuffers_.find(activeCam1_);
        auto it2 = previewBuffers_.find(activeCam2_);
        if(it1 == previewBuffers_.end() || it2 == previewBuffers_.end()) {
            return false;
        }
        if(it1->second.latestRgb.empty() || it2->second.latestRgb.empty()) {
            return false;
        }
        const uint64_t a = it1->second.latestRgbTsUs;
        const uint64_t b = it2->second.latestRgbTsUs;
        if(a == 0 || b == 0) {
            return false;
        }
        const uint64_t diff = (a > b) ? (a - b) : (b - a);
        if(diff > maxDiffUs) {
            return false;
        }
        out.ts1 = a;
        out.ts2 = b;
        out.img1 = it1->second.latestRgb.clone();
        out.img2 = it2->second.latestRgb.clone();
        return true;
    }

    bool detectChessboard(const Sample &s, cv::Mat &draw1, cv::Mat &draw2) {
        if(s.img1.empty() || s.img2.empty()) {
            return false;
        }
        const int rows = parseIntBound(cbRows_, cfg_.calibration.chessboard.rows, 3, 64);
        const int cols = parseIntBound(cbCols_, cfg_.calibration.chessboard.cols, 3, 64);
        const cv::Size pat(cols, rows);
        cv::Mat g1, g2;
        cv::cvtColor(s.img1, g1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(s.img2, g2, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> c1;
        std::vector<cv::Point2f> c2;
        bool ok1 = cv::findChessboardCorners(g1, pat, c1, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool ok2 = cv::findChessboardCorners(g2, pat, c2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(!ok1 || !ok2) {
            return false;
        }
        cv::cornerSubPix(g1, c1, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
        cv::cornerSubPix(g2, c2, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
        draw1 = s.img1.clone();
        draw2 = s.img2.clone();
        cv::drawChessboardCorners(draw1, pat, c1, true);
        cv::drawChessboardCorners(draw2, pat, c2, true);
        return true;
    }

    bool computePairExtrinsic(PairData &pd, cv::Matx33d &R12, cv::Vec3d &t12, double &rms) {
        const int rows = parseIntBound(cbRows_, cfg_.calibration.chessboard.rows, 3, 64);
        const int cols = parseIntBound(cbCols_, cfg_.calibration.chessboard.cols, 3, 64);
        cfg_.calibration.chessboard.rows = rows;
        cfg_.calibration.chessboard.cols = cols;
        const cv::Size pat(cols, rows);
        std::vector<std::vector<cv::Point3f>> objPts;
        std::vector<std::vector<cv::Point2f>> img1Pts;
        std::vector<std::vector<cv::Point2f>> img2Pts;
        std::vector<cv::Point3f> obj;
        for(int y = 0; y < rows; y++) {
            for(int x = 0; x < cols; x++) {
                obj.emplace_back(static_cast<float>(x) * cfg_.calibration.chessboard.squareSize, static_cast<float>(y) * cfg_.calibration.chessboard.squareSize, 0.0f);
            }
        }
        for(const auto &s: pd.samples) {
            cv::Mat g1, g2;
            cv::cvtColor(s.img1, g1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(s.img2, g2, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> c1;
            std::vector<cv::Point2f> c2;
            bool ok1 = cv::findChessboardCorners(g1, pat, c1, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            bool ok2 = cv::findChessboardCorners(g2, pat, c2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if(!ok1 || !ok2) {
                continue;
            }
            cv::cornerSubPix(g1, c1, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(g2, c2, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            objPts.push_back(obj);
            img1Pts.push_back(std::move(c1));
            img2Pts.push_back(std::move(c2));
        }
        if(objPts.size() < 3 || pd.K1.empty() || pd.K2.empty()) {
            return false;
        }
        cv::Mat K1 = pd.K1.clone();
        cv::Mat D1 = pd.D1.clone();
        cv::Mat K2 = pd.K2.clone();
        cv::Mat D2 = pd.D2.clone();
        cv::Mat R, T, E, F;
        int flags = cv::CALIB_FIX_INTRINSIC;
        if((D1.total() == 8 && (D1.rows == 1 || D1.cols == 1)) && (D2.total() == 8 && (D2.rows == 1 || D2.cols == 1))) {
            flags |= cv::CALIB_RATIONAL_MODEL;
        }
        rms = cv::stereoCalibrate(objPts, img1Pts, img2Pts, K1, D1, K2, D2, cv::Size(pd.colorW, pd.colorH), R, T, E, F, flags,
                                  cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-6));
        cv::Mat Rd, Td;
        R.convertTo(Rd, CV_64F);
        T.convertTo(Td, CV_64F);
        R12 = cv::Matx33d(Rd.at<double>(0, 0), Rd.at<double>(0, 1), Rd.at<double>(0, 2),
                          Rd.at<double>(1, 0), Rd.at<double>(1, 1), Rd.at<double>(1, 2),
                          Rd.at<double>(2, 0), Rd.at<double>(2, 1), Rd.at<double>(2, 2));
        t12 = cv::Vec3d(Td.at<double>(0, 0), Td.at<double>(1, 0), Td.at<double>(2, 0));
        return true;
    }

    static bool parseVec3d(cJSON *arr, cv::Vec3d &out) {
        if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
            return false;
        }
        auto *a0 = cJSON_GetArrayItem(arr, 0);
        auto *a1 = cJSON_GetArrayItem(arr, 1);
        auto *a2 = cJSON_GetArrayItem(arr, 2);
        if(!a0 || !a1 || !a2 || !cJSON_IsNumber(a0) || !cJSON_IsNumber(a1) || !cJSON_IsNumber(a2)) {
            return false;
        }
        out = cv::Vec3d(a0->valuedouble, a1->valuedouble, a2->valuedouble);
        return true;
    }

    static bool parseMat3d(cJSON *arr, cv::Matx33d &out) {
        if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
            return false;
        }
        double v[9];
        for(int y = 0; y < 3; y++) {
            auto *row = cJSON_GetArrayItem(arr, y);
            if(!row || !cJSON_IsArray(row) || cJSON_GetArraySize(row) != 3) {
                return false;
            }
            for(int x = 0; x < 3; x++) {
                auto *it = cJSON_GetArrayItem(row, x);
                if(!it || !cJSON_IsNumber(it)) {
                    return false;
                }
                v[y * 3 + x] = it->valuedouble;
            }
        }
        out = cv::Matx33d(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
        return true;
    }

    bool loadExtrinsics(std::unordered_map<std::string, EdgeExtrinsic> &worldToCam) {
        if(cfg_.initExtrinsicPath.empty()) {
            return false;
        }
        std::string content;
        try {
            content = readFileAllLocal(fs::absolute(cfg_.initExtrinsicPath));
        }
        catch(...) {
            return false;
        }
        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            return false;
        }
        worldToCam.clear();
        rgbToDepthByCam_.clear();
        depthToRgbByCam_.clear();
        for(cJSON *item = root->child; item != nullptr; item = item->next) {
            if(!item->string || !cJSON_IsObject(item)) {
                continue;
            }
            auto *rotArr = cJSON_GetObjectItemCaseSensitive(item, "rotation");
            auto *tArr = cJSON_GetObjectItemCaseSensitive(item, "translation");
            cv::Vec3d tRgb;
            cv::Matx33d RRgb;
            if(!parseVec3d(tArr, tRgb) || !parseMat3d(rotArr, RRgb)) {
                continue;
            }
            EdgeExtrinsic exRgb;
            exRgb.valid = true;
            exRgb.R = RRgb;
            exRgb.t = tRgb;

            EdgeExtrinsic exRgbToDepth;
            bool haveRgbToDepth = false;
            auto *rgbDepthObj = cJSON_GetObjectItemCaseSensitive(item, "rgb_to_depth");
            if(rgbDepthObj && cJSON_IsObject(rgbDepthObj)) {
                auto *c2dObj = cJSON_GetObjectItemCaseSensitive(rgbDepthObj, "c2d_extrinsic");
                if(parseExtrinsicObject(c2dObj, exRgbToDepth)) {
                    haveRgbToDepth = true;
                    rgbToDepthByCam_[item->string] = exRgbToDepth;
                    depthToRgbByCam_[item->string] = fromEigenWorldToCam(invertRigid(toEigenWorldToCam(exRgbToDepth)));
                }
                else {
                    EdgeExtrinsic exDepthToRgb;
                    auto *d2cObj = cJSON_GetObjectItemCaseSensitive(rgbDepthObj, "d2c_extrinsic");
                    if(parseExtrinsicObject(d2cObj, exDepthToRgb)) {
                        haveRgbToDepth = true;
                        depthToRgbByCam_[item->string] = exDepthToRgb;
                        rgbToDepthByCam_[item->string] = fromEigenWorldToCam(invertRigid(toEigenWorldToCam(exDepthToRgb)));
                    }
                }
            }

            if(haveRgbToDepth) {
                const Eigen::Matrix4f Twr = toEigenWorldToCam(exRgb);
                const Eigen::Matrix4f Trd = toEigenWorldToCam(rgbToDepthByCam_[item->string]);
                worldToCam[item->string] = fromEigenWorldToCam(makeRigid(Trd * Twr));
            }
            else {
                worldToCam[item->string] = exRgb;
            }
        }
        cJSON_Delete(root);
        return true;
    }

    bool writeExtrinsics(const std::unordered_map<std::string, EdgeExtrinsic> &worldToCam) {
        if(cfg_.initExtrinsicPath.empty()) {
            return false;
        }
        cJSON *root = cJSON_CreateObject();
        for(const auto &dc: cfg_.devices) {
            auto it = worldToCam.find(dc.index);
            if(it == worldToCam.end() || !it->second.valid) {
                continue;
            }
            const auto &ex = it->second;
            cJSON *camObj = cJSON_CreateObject();
            cJSON_AddItemToObject(camObj, "rotation", toJsonMat3(ex.R));
            cJSON_AddItemToObject(camObj, "translation", toJsonVec3(ex.t));

            auto itD2c = depthToRgbByCam_.find(dc.index);
            auto itC2d = rgbToDepthByCam_.find(dc.index);
            if(itD2c != depthToRgbByCam_.end() || itC2d != rgbToDepthByCam_.end()) {
                EdgeExtrinsic exD2c;
                EdgeExtrinsic exC2d;
                if(itD2c != depthToRgbByCam_.end() && itD2c->second.valid) {
                    exD2c = itD2c->second;
                    exC2d = fromEigenWorldToCam(invertRigid(toEigenWorldToCam(exD2c)));
                }
                else if(itC2d != rgbToDepthByCam_.end() && itC2d->second.valid) {
                    exC2d = itC2d->second;
                    exD2c = fromEigenWorldToCam(invertRigid(toEigenWorldToCam(exC2d)));
                }
                cJSON *rgbDepthObj = cJSON_CreateObject();
                cJSON_AddItemToObject(rgbDepthObj, "d2c_extrinsic", makeExtrinsicJson(exD2c));
                cJSON_AddItemToObject(rgbDepthObj, "c2d_extrinsic", makeExtrinsicJson(exC2d));
                cJSON_AddItemToObject(camObj, "rgb_to_depth", rgbDepthObj);
            }
            cJSON_AddItemToObject(root, dc.index.c_str(), camObj);
        }
        char *printed = cJSON_Print(root);
        cJSON_Delete(root);
        if(!printed) {
            return false;
        }
        fs::path p = fs::absolute(cfg_.initExtrinsicPath);
        if(p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if(!f.is_open()) {
            cJSON_free(printed);
            return false;
        }
        f << printed;
        f.flush();
        bool ok = f.good();
        f.close();
        cJSON_free(printed);
        return ok;
    }

    bool calcOverallAndSave() {
        const std::string root = hasIndex("00") ? "00" : devices_.front().cfg.index;
        std::unordered_map<std::string, EdgeExtrinsic> worldToCam;
        std::unordered_set<std::string> vis;
        std::deque<std::string> q;
        EdgeExtrinsic rootEx;
        rootEx.valid = true;
        worldToCam[root] = rootEx;
        vis.insert(root);
        q.push_back(root);
        while(!q.empty()) {
            std::string u = q.front();
            q.pop_front();
            auto exU = worldToCam[u];
            for(const auto &kv: edges_) {
                const std::string &k = kv.first;
                const EdgeExtrinsic &e = kv.second;
                auto pos = k.find("->");
                if(pos == std::string::npos || !e.valid) {
                    continue;
                }
                const std::string from = k.substr(0, pos);
                const std::string to = k.substr(pos + 2);
                if(from != u || vis.count(to)) {
                    continue;
                }
                EdgeExtrinsic exV;
                exV.valid = true;
                exV.R = e.R * exU.R;
                exV.t = e.R * exU.t + e.t;
                worldToCam[to] = exV;
                vis.insert(to);
                q.push_back(to);
            }
        }
        for(const auto &d: devices_) {
            if(worldToCam.find(d.cfg.index) == worldToCam.end()) {
                return false;
            }
        }
        return writeExtrinsics(worldToCam);
    }

    static Eigen::Matrix4f toEigenWorldToCam(const EdgeExtrinsic &ex) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(0, 0) = static_cast<float>(ex.R(0, 0));
        T(0, 1) = static_cast<float>(ex.R(0, 1));
        T(0, 2) = static_cast<float>(ex.R(0, 2));
        T(1, 0) = static_cast<float>(ex.R(1, 0));
        T(1, 1) = static_cast<float>(ex.R(1, 1));
        T(1, 2) = static_cast<float>(ex.R(1, 2));
        T(2, 0) = static_cast<float>(ex.R(2, 0));
        T(2, 1) = static_cast<float>(ex.R(2, 1));
        T(2, 2) = static_cast<float>(ex.R(2, 2));
        T(0, 3) = static_cast<float>(ex.t[0]);
        T(1, 3) = static_cast<float>(ex.t[1]);
        T(2, 3) = static_cast<float>(ex.t[2]);
        return T;
    }

    static Eigen::Matrix3f projectRotationToSO3(const Eigen::Matrix3f &Rin) {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(Rin, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        Eigen::Matrix3f R = U * V.transpose();
        if(R.determinant() < 0.0f) {
            U.col(2) *= -1.0f;
            R = U * V.transpose();
        }
        return R;
    }

    static Eigen::Matrix4f makeRigid(const Eigen::Matrix4f &Tin) {
        Eigen::Matrix4f T = Tin;
        T.block<3, 3>(0, 0) = projectRotationToSO3(T.block<3, 3>(0, 0));
        T(3, 0) = 0.0f;
        T(3, 1) = 0.0f;
        T(3, 2) = 0.0f;
        T(3, 3) = 1.0f;
        return T;
    }

    static EdgeExtrinsic fromEigenWorldToCam(const Eigen::Matrix4f &T) {
        const Eigen::Matrix4f Tr = makeRigid(T);
        EdgeExtrinsic ex;
        ex.valid = true;
        ex.R = cv::Matx33d(static_cast<double>(Tr(0, 0)), static_cast<double>(Tr(0, 1)), static_cast<double>(Tr(0, 2)),
                           static_cast<double>(Tr(1, 0)), static_cast<double>(Tr(1, 1)), static_cast<double>(Tr(1, 2)),
                           static_cast<double>(Tr(2, 0)), static_cast<double>(Tr(2, 1)), static_cast<double>(Tr(2, 2)));
        ex.t = cv::Vec3d(static_cast<double>(Tr(0, 3)), static_cast<double>(Tr(1, 3)), static_cast<double>(Tr(2, 3)));
        return ex;
    }

    static Eigen::Matrix4f invertRigid(const Eigen::Matrix4f &T) {
        const Eigen::Matrix4f Tr = makeRigid(T);
        const Eigen::Matrix3f R = Tr.block<3, 3>(0, 0);
        const Eigen::Vector3f t = Tr.block<3, 1>(0, 3);
        Eigen::Matrix4f inv = Eigen::Matrix4f::Identity();
        inv.block<3, 3>(0, 0) = R.transpose();
        inv.block<3, 1>(0, 3) = -R.transpose() * t;
        return inv;
    }

    static pcl::PointCloud<pcl::PointXYZ>::Ptr pointsFrameToCloudCam(const std::shared_ptr<ob::PointsFrame> &pf) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if(!pf) {
            return cloud;
        }
        const float scaleMm = pf->getCoordinateValueScale();
        const auto *data = reinterpret_cast<const OBPoint *>(pf->data());
        const auto count = pf->dataSize() / sizeof(OBPoint);
        for(size_t i = 0; i < count; i++) {
            const auto &p = data[i];
            if(!std::isfinite(p.z) || p.z <= 0.0f) {
                continue;
            }
            cloud->points.emplace_back(p.x * scaleMm * 0.001f, p.y * scaleMm * 0.001f, p.z * scaleMm * 0.001f);
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    bool startIcpPipelines(std::vector<IcpDevice> &devs) {
        {
            std::lock_guard<std::mutex> lock(icpDepthMtx_);
            icpDepthSlots_.clear();
        }
        for(auto &d: devs) {
            d.pipe = std::make_shared<ob::Pipeline>(d.dev);
            d.pcFilter = std::make_shared<ob::PointCloudFilter>();
            d.pcFilter->setCreatePointFormat(OB_FORMAT_POINT);
            d.pcFilter->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
            d.pcFilter->setDecimationFactor(cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 1);
            auto depthProfile = pickVideoProfile(d.pipe, OB_SENSOR_DEPTH, 0, 0, 0);
            if(!depthProfile) {
                return false;
            }
            auto cfg = std::make_shared<ob::Config>();
            cfg->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
            cfg->enableStream(depthProfile);
            d.pipe->start(cfg, [this, idx = d.index](std::shared_ptr<ob::FrameSet> fs) { onIcpFrameSet(idx, fs); });
        }
        return true;
    }

    void stopIcpPipelines(std::vector<IcpDevice> &devs) {
        for(auto &d: devs) {
            try {
                if(d.pipe) {
                    d.pipe->stop();
                }
            }
            catch(...) {
            }
            d.pipe.reset();
            d.pcFilter.reset();
        }
        std::lock_guard<std::mutex> lock(icpDepthMtx_);
        icpDepthSlots_.clear();
    }

    void onIcpFrameSet(const std::string &idx, const std::shared_ptr<ob::FrameSet> &fs) {
        if(!fs) {
            return;
        }
        auto depth = fs->depthFrame();
        if(!depth) {
            return;
        }
        uint64_t ts = 0;
        const bool requireGlobalTs = cfg_.enableSync && devices_.size() > 1;
        try {
            ts = depth->globalTimeStampUs();
        }
        catch(...) {
            ts = 0;
        }
        if(ts == 0 && !requireGlobalTs) {
            try {
                ts = depth->timeStampUs();
            }
            catch(...) {
                ts = 0;
            }
        }
        std::lock_guard<std::mutex> lock(icpDepthMtx_);
        auto &slot = icpDepthSlots_[idx];
        slot.latestDepthFrame = depth;
        slot.latestDepthTsUs = ts;
    }

    bool fetchIcpDepthFrame(const std::string &idx, std::shared_ptr<ob::Frame> &depth, uint64_t &ts) {
        std::lock_guard<std::mutex> lock(icpDepthMtx_);
        auto it = icpDepthSlots_.find(idx);
        if(it == icpDepthSlots_.end() || !it->second.latestDepthFrame) {
            return false;
        }
        depth = it->second.latestDepthFrame;
        ts = it->second.latestDepthTsUs;
        return true;
    }

    bool runIcpOnce(int maxOuterIters, double stopRotDeg, double stopTransM) {
        std::unordered_map<std::string, EdgeExtrinsic> work = icpWork_;
        std::vector<IcpDevice> devs;
        for(const auto &d: devices_) {
            IcpDevice id;
            id.index = d.cfg.index;
            id.dev = d.dev;
            devs.push_back(std::move(id));
        }
        if(!startIcpPipelines(devs)) {
            stopIcpPipelines(devs);
            return false;
        }
        std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudsW;
        for(const auto &d: devs) {
            cloudsW[d.index] = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        }
        const float minZ = 0.2f;
        const float maxZ = cfg_.maxDepth > 0.0f ? cfg_.maxDepth : 6.0f;
        std::unordered_map<std::string, uint64_t> lastUsedTs;
        for(int i = 0; i < 10; i++) {
            if(cancel_ && cancel_->load()) {
                break;
            }
            for(auto &d: devs) {
                std::shared_ptr<ob::Frame> depth;
                uint64_t ts = 0;
                if(!fetchIcpDepthFrame(d.index, depth, ts)) {
                    continue;
                }
                if(lastUsedTs[d.index] == ts) {
                    continue;
                }
                lastUsedTs[d.index] = ts;
                std::shared_ptr<ob::Frame> depthF = depth;
                depthF = refineDepthFrameForPointCloud(depthF, d.depthFilters, minZ, maxZ, cfg_.filters);
                if(!depthF) {
                    continue;
                }
                std::shared_ptr<ob::Frame> pcFrame;
                try {
                    pcFrame = d.pcFilter->process(depthF);
                }
                catch(...) {
                    continue;
                }
                auto pf = pcFrame ? pcFrame->as<ob::PointsFrame>() : nullptr;
                if(!pf) {
                    continue;
                }
                auto itEx = work.find(d.index);
                if(itEx == work.end() || !itEx->second.valid) {
                    continue;
                }
                auto camCloud = pointsFrameToCloudCam(pf);
                auto worldCloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                Eigen::Matrix4f Tcw = toEigenWorldToCam(itEx->second);
                pcl::transformPointCloud(*camCloud, *worldCloud, invertRigid(Tcw));
                *cloudsW[d.index] += *worldCloud;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        for(auto &kv: cloudsW) {
            if(cfg_.filters.deskCrop) {
                kv.second = removeDominantPlaneRansacCompat(kv.second, 100, 0.015, 0.25, 1500);
            }
            kv.second = denoiseCloudSor(kv.second, 20, 1.0);
        }

        auto mergeExcluding = [&](const std::string &idx) {
            auto m = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for(const auto &d: devices_) {
                if(d.cfg.index == idx) {
                    continue;
                }
                auto it = cloudsW.find(d.cfg.index);
                if(it == cloudsW.end()) {
                    continue;
                }
                *m += *it->second;
            }
            return m;
        };
        const float maxCorr = 0.08f;
        const std::string rootIdx = devices_.empty() ? std::string() : devices_.front().cfg.index;
        const auto calcRotDegFromR = [&](const Eigen::Matrix3f &Rin) -> double {
            const Eigen::Matrix3f R = projectRotationToSO3(Rin);
            double tr = static_cast<double>(R.trace());
            double c = (tr - 1.0) * 0.5;
            c = std::min(1.0, std::max(-1.0, c));
            const double angle = std::acos(c);
            return angle * (180.0 / 3.14159265358979323846);
        };
        for(int it = 0; it < maxOuterIters; it++) {
            bool allSmall = true;
            double maxRotIter = 0.0;
            double maxTransIter = 0.0;

            for(const auto &d: devices_) {
                auto srcIt = cloudsW.find(d.cfg.index);
                if(srcIt == cloudsW.end() || srcIt->second->size() < 100) {
                    continue;
                }
                auto tgt = mergeExcluding(d.cfg.index);
                if(!tgt || tgt->size() < 200) {
                    continue;
                }
                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                icp.setInputSource(srcIt->second);
                icp.setInputTarget(tgt);
                icp.setMaximumIterations(1);
                icp.setMaxCorrespondenceDistance(maxCorr);
                icp.setTransformationEpsilon(1e-10);
                icp.setEuclideanFitnessEpsilon(1e-9);
                icp.setUseReciprocalCorrespondences(true);
                icp.setRANSACOutlierRejectionThreshold(maxCorr * 0.5f);
                auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                icp.align(*aligned);
                if(!icp.hasConverged()) {
                    continue;
                }
                Eigen::Matrix4f delta = makeRigid(icp.getFinalTransformation());
                srcIt->second = aligned;
                auto exIt = work.find(d.cfg.index);
                if(exIt != work.end() && exIt->second.valid) {
                    Eigen::Matrix4f TcwOld = toEigenWorldToCam(exIt->second);
                    Eigen::Matrix4f TwcNew = delta * invertRigid(TcwOld);
                    Eigen::Matrix4f TcwNew = invertRigid(TwcNew);
                    exIt->second = fromEigenWorldToCam(TcwNew);
                }
                const double rotDeg = calcRotDegFromR(delta.block<3, 3>(0, 0));
                const double trans = static_cast<double>(delta.block<3, 1>(0, 3).norm());
                
                if(rotDeg > maxRotIter) maxRotIter = rotDeg;
                if(trans > maxTransIter) maxTransIter = trans;

                if(rotDeg > stopRotDeg || trans > stopTransM) {
                    allSmall = false;
                }
            }
            icpLastIter_ = it + 1;
            
            {
                std::ostringstream oss;
                oss << "ICP iter " << icpLastIter_ << ": max_rot=" << std::fixed << std::setprecision(4) << maxRotIter 
                    << " deg, max_trans=" << std::setprecision(4) << maxTransIter << " m";
                pushLog(oss.str());
            }

            if(allSmall) {
                break;
            }
        }
        pushLog("icp outer iterations used: " + std::to_string(icpLastIter_) + "/" + std::to_string(maxOuterIters));
        if(!rootIdx.empty()) {
            auto itRoot = work.find(rootIdx);
            if(itRoot != work.end() && itRoot->second.valid) {
                const Eigen::Matrix4f TcwRoot = toEigenWorldToCam(itRoot->second);
                const Eigen::Matrix4f Trw = invertRigid(TcwRoot);
                for(auto &kv: work) {
                    if(!kv.second.valid) {
                        continue;
                    }
                    const Eigen::Matrix4f Tcw = toEigenWorldToCam(kv.second);
                    const Eigen::Matrix4f Tcr = makeRigid(Tcw * Trw);
                    kv.second = fromEigenWorldToCam(Tcr);
                }
                auto itRootSet = work.find(rootIdx);
                if(itRootSet != work.end()) {
                    itRootSet->second.valid = true;
                    itRootSet->second.R = cv::Matx33d::eye();
                    itRootSet->second.t = cv::Vec3d(0.0, 0.0, 0.0);
                }
            }
        }
        icpWork_ = std::move(work);
        buildIcpPreview();
        stopIcpPipelines(devs);
        return true;
    }

    void buildIcpPreview() {
        icpPreview_ = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(20, 20, 20));
        const float minZ = 0.2f;
        const float maxZ = cfg_.maxDepth > 0.0f ? cfg_.maxDepth : 6.0f;
        for(size_t i = 0; i < devices_.size(); i++) {
            const auto &d = devices_[i];
            std::shared_ptr<ob::Frame> depth;
            uint64_t ts = 0;
            if(!fetchIcpDepthFrame(d.cfg.index, depth, ts)) {
                continue;
            }
            std::shared_ptr<ob::Frame> depthF = depth;
            OrbbecDepthFilterChain chain;
            depthF = refineDepthFrameForPointCloud(depthF, chain, minZ, maxZ, cfg_.filters);
            if(!depthF) {
                continue;
            }
            auto pcFilter = std::make_shared<ob::PointCloudFilter>();
            pcFilter->setCreatePointFormat(OB_FORMAT_POINT);
            pcFilter->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
            pcFilter->setDecimationFactor(cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 1);
            std::shared_ptr<ob::Frame> pcFrame;
            try {
                pcFrame = pcFilter->process(depthF);
            }
            catch(...) {
                continue;
            }
            auto pf = pcFrame ? pcFrame->as<ob::PointsFrame>() : nullptr;
            if(!pf) {
                continue;
            }
            auto cloud = pointsFrameToCloudCam(pf);
            auto exIt = icpWork_.find(d.cfg.index);
            if(exIt == icpWork_.end() || !exIt->second.valid) {
                continue;
            }
            Eigen::Matrix4f Tcw = toEigenWorldToCam(exIt->second);
            Eigen::Matrix4f Twc = invertRigid(Tcw);
            for(size_t k = 0; k < cloud->points.size(); k += 8) {
                const auto &p = cloud->points[k];
                Eigen::Vector4f pc(p.x, p.y, p.z, 1.0f);
                Eigen::Vector4f pw = Twc * pc;
                const int u = static_cast<int>(pw.x() * 220.0f + 640.0f);
                const int v = static_cast<int>(pw.z() * 220.0f + 80.0f);
                if(u < 0 || v < 0 || u >= icpPreview_.cols || v >= icpPreview_.rows) {
                    continue;
                }
                cv::Vec3b col;
                if(i % 3 == 0) {
                    col = cv::Vec3b(40, 120, 255);
                }
                else if(i % 3 == 1) {
                    col = cv::Vec3b(60, 240, 80);
                }
                else {
                    col = cv::Vec3b(255, 160, 40);
                }
                icpPreview_.at<cv::Vec3b>(v, u) = col;
            }
        }
        cv::putText(icpPreview_, "Top View Preview", cv::Point(30, 40), cv::FONT_HERSHEY_DUPLEX, 0.9, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    }

    void drawFrame(cv::Mat &ui, FrameMouse &fm, int key, bool &running) {
        const int W = ui.cols;
        const int H = ui.rows;
        const int leftW = W * 20 / 100;
        const int midW = W * 40 / 100;
        const int rightW = W - leftW - midW;
        const cv::Rect left(0, 0, leftW, H);
        const cv::Rect mid(leftW, 0, midW, H);
        const cv::Rect right(leftW + midW, 0, rightW, H);
        cv::rectangle(ui, left, cv::Scalar(24, 24, 24), cv::FILLED);
        cv::rectangle(ui, mid, cv::Scalar(10, 10, 10), cv::FILLED);
        cv::rectangle(ui, right, cv::Scalar(12, 12, 12), cv::FILLED);

        cv::Rect btnCb(12, 12, (leftW - 36) / 2, 36);
        cv::Rect btnIcp(btnCb.x + btnCb.width + 12, 12, (leftW - 36) / 2, 36);
        if(uiButton(ui, btnCb, "chessboard", fm)) {
            mode_ = UIMode::Chessboard;
            activeField_.clear();
        }
        if(uiButton(ui, btnIcp, "ICP", fm)) {
            stopActivePair();
            mode_ = UIMode::ICP;
            activeField_.clear();
        }

        if(mode_ == UIMode::Chessboard) {
            drawChessboardPanel(ui, fm, key, running, left, mid, right);
        }
        else {
            drawIcpPanel(ui, fm, key, running, left, mid, right);
        }
    }

    void drawLogBox(cv::Mat &ui, const cv::Rect &r, FrameMouse &fm) {
        cv::rectangle(ui, r, cv::Scalar(18, 18, 18), cv::FILLED);
        cv::rectangle(ui, r, cv::Scalar(100, 100, 100), 1);
        cv::putText(ui, "log", cv::Point(r.x + 6, r.y + 20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        if(r.contains(cv::Point(fm.x, fm.y)) && fm.wheel != 0) {
            logScroll_ += (fm.wheel > 0) ? -3 : 3;
        }
        const int maxLines = std::max(1, (r.height - 28) / 18);
        std::vector<std::string> logs;
        {
            std::lock_guard<std::mutex> lock(logMtx_);
            logs = logs_;
        }
        const int total = static_cast<int>(logs.size());
        const int maxScroll = std::max(0, total - maxLines);
        logScroll_ = std::max(0, std::min(maxScroll, logScroll_));
        int start = std::max(0, total - maxLines - logScroll_);
        int y = r.y + 40;
        for(int i = start; i < total && y < r.y + r.height - 6; i++) {
            cv::putText(ui, logs[i], cv::Point(r.x + 6, y), cv::FONT_HERSHEY_DUPLEX, 0.42, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
            y += 18;
        }
    }

    void drawChessboardPanel(cv::Mat &ui, FrameMouse &fm, int key, bool &running, const cv::Rect &left, const cv::Rect &mid, const cv::Rect &right) {
        cv::Rect f1(left.x + 12, left.y + 72, (left.width - 36) / 2, 34);
        cv::Rect f2(f1.x + f1.width + 12, f1.y, (left.width - 36) / 2, 34);
        cv::Rect f3(left.x + 12, f1.y + 56, (left.width - 36) / 2, 34);
        cv::Rect f4(f3.x + f3.width + 12, f3.y, (left.width - 36) / 2, 34);
        if(uiField(ui, f1, "first camera", camFirst_, activeField_, "camFirst", fm)) {
            activeField_ = "camFirst";
        }
        if(uiField(ui, f2, "second camera", camSecond_, activeField_, "camSecond", fm)) {
            activeField_ = "camSecond";
        }
        if(uiField(ui, f3, "board cols", cbCols_, activeField_, "cbCols", fm)) {
            activeField_ = "cbCols";
        }
        if(uiField(ui, f4, "board rows", cbRows_, activeField_, "cbRows", fm)) {
            activeField_ = "cbRows";
        }
        if(!activeField_.empty() && key > 0) {
            if(activeField_ == "camFirst") handleText(camFirst_, key, 8);
            else if(activeField_ == "camSecond") handleText(camSecond_, key, 8);
            else if(activeField_ == "cbCols") handleText(cbCols_, key, 4);
            else if(activeField_ == "cbRows") handleText(cbRows_, key, 4);
        }

        const std::string pk = pairKey(trimString(camFirst_), trimString(camSecond_));
        int valid = 0;
        double rms = -1.0;
        int calibratedCount = 0;
        cv::Mat det1, det2;
        {
            std::lock_guard<std::mutex> lock(pairMtx_);
            auto it = pairStore_.find(pk);
            if(it != pairStore_.end()) {
                valid = it->second.validCount;
                rms = it->second.rmsPx;
                det1 = it->second.latestDet1;
                det2 = it->second.latestDet2;
            }
            for(const auto &kv: pairStore_) {
                if(kv.second.calibrated) {
                    calibratedCount++;
                }
            }
        }
        cv::putText(ui, "valid image pairs: " + std::to_string(valid), cv::Point(left.x + 12, f4.y + 56), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(4);
            if(rms > 0.0) {
                oss << rms;
            }
            else {
                oss << "-";
            }
            cv::putText(ui, "stereo rms (px): " + oss.str(), cv::Point(left.x + 12, f4.y + 82), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        }
        cv::putText(ui, "calibrated camera pairs: " + std::to_string(calibratedCount), cv::Point(left.x + 12, f4.y + 108), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

        cv::Rect logBox(left.x + 10, f4.y + 132, left.width - 20, 330);
        drawLogBox(ui, logBox, fm);

        const int by = left.y + left.height - 190;
        cv::Rect b1(left.x + 10, by, left.width - 20, 30);
        cv::Rect b2(left.x + 10, by + 34, left.width - 20, 30);
        cv::Rect b3(left.x + 10, by + 68, left.width - 20, 30);
        cv::Rect b4(left.x + 10, by + 102, left.width - 20, 30);
        cv::Rect b5(left.x + 10, by + 136, left.width - 20, 30);
        if(uiButton(ui, b1, "start", fm)) onStartChessboard();
        if(uiButton(ui, b2, "pause", fm)) onPauseChessboard();
        if(uiButton(ui, b3, "calculate camera pair", fm)) onCalcPair();
        if(uiButton(ui, b4, "calculate overall extrinsic", fm)) onCalcOverall();
        if(uiButton(ui, b5, "back to menu", fm)) {
            clearAllState();
            running = false;
        }

        updateStreaming();
        drawRgbGrid(ui, mid, latestRgbFramesForGrid(), activePair_.colorW, activePair_.colorH, std::max(activePair_.colorFps1, activePair_.colorFps2));
        std::vector<std::pair<std::string, cv::Mat>> detections;
        if(!det1.empty()) {
            detections.emplace_back(activeCam1_.empty() ? "cam1" : activeCam1_, det1);
        }
        if(!det2.empty()) {
            detections.emplace_back(activeCam2_.empty() ? "cam2" : activeCam2_, det2);
        }
        drawRgbGrid(ui, right, detections, activePair_.colorW, activePair_.colorH, std::max(activePair_.colorFps1, activePair_.colorFps2));
    }

    void onStartChessboard() {
        const std::string a = trimString(camFirst_);
        const std::string b = trimString(camSecond_);
        const int rows = parseIntBound(cbRows_, cfg_.calibration.chessboard.rows, 3, 64);
        const int cols = parseIntBound(cbCols_, cfg_.calibration.chessboard.cols, 3, 64);
        if(a.empty() || b.empty()) {
            pushLog("start failed: camera index empty");
            return;
        }
        if(a == b) {
            pushLog("start failed: two cameras are same");
            return;
        }
        if(!hasIndex(a) || !hasIndex(b)) {
            pushLog("start failed: camera not found");
            return;
        }
        cfg_.calibration.chessboard.rows = rows;
        cfg_.calibration.chessboard.cols = cols;
        const auto *d1 = findByIndex(a);
        const auto *d2 = findByIndex(b);
        if(!d1 || !d2) {
            pushLog("start failed: internal index mismatch");
            return;
        }
        if(!startPair(*d1, *d2)) {
            pushLog("start failed: pipeline start error");
            return;
        }
        activePairKey_ = pairKey(a, b);
        {
            std::lock_guard<std::mutex> lock(pairMtx_);
            auto &pd = pairStore_[activePairKey_];
            pd.K1 = activePair_.K1.clone();
            pd.D1 = activePair_.D1.clone();
            pd.K2 = activePair_.K2.clone();
            pd.D2 = activePair_.D2.clone();
            pd.colorW = activePair_.colorW;
            pd.colorH = activePair_.colorH;
        }
        pairStreaming_ = true;
        lastSampleTp_ = std::chrono::steady_clock::now();
        pushLog("start streaming pair " + activePairKey_);
        pushLog("rgb profile cam " + a + ": " + std::to_string(activePair_.colorW) + "x" + std::to_string(activePair_.colorH) + "@" + std::to_string(activePair_.colorFps1));
        pushLog("rgb profile cam " + b + ": " + std::to_string(activePair_.colorW) + "x" + std::to_string(activePair_.colorH) + "@" + std::to_string(activePair_.colorFps2));
    }

    void onPauseChessboard() {
        stopActivePair();
        pushLog("stream paused");
    }

    void onCalcPair() {
        const std::string a = trimString(camFirst_);
        const std::string b = trimString(camSecond_);
        const std::string pk = pairKey(a, b);
        PairData snap;
        {
            std::lock_guard<std::mutex> lock(pairMtx_);
            auto it = pairStore_.find(pk);
            if(it == pairStore_.end() || it->second.samples.size() < 3) {
                pushLog("calculate pair failed: need >=3 valid samples");
                return;
            }
            snap = it->second;
        }
        cv::Matx33d R12;
        cv::Vec3d t12;
        double rms = -1.0;
        if(!computePairExtrinsic(snap, R12, t12, rms)) {
            pushLog("calculate pair failed");
            return;
        }
        EdgeExtrinsic e12;
        e12.valid = true;
        e12.R = R12;
        e12.t = t12;
        edges_[a + "->" + b] = e12;
        EdgeExtrinsic e21;
        e21.valid = true;
        e21.R = R12.t();
        e21.t = -(e21.R * t12);
        edges_[b + "->" + a] = e21;
        {
            std::lock_guard<std::mutex> lock(pairMtx_);
            auto it = pairStore_.find(pk);
            if(it != pairStore_.end()) {
                it->second.calibrated = true;
                it->second.rmsPx = rms;
            }
        }
        pushLog("pair calibrated " + pk);
    }

    void onCalcOverall() {
        if(edges_.empty()) {
            pushLog("calculate overall failed: no calibrated pair");
            return;
        }
        if(calcOverallAndSave()) {
            pushLog("overall extrinsic saved");
        }
        else {
            pushLog("calculate overall failed: graph disconnected or file write failed");
        }
    }

    void updateStreaming() {
        if(!pairStreaming_ || !activePair_.p1 || !activePair_.p2) {
            return;
        }
        updateLiveFramesFromPreview();

        const auto now = std::chrono::steady_clock::now();
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSampleTp_).count();
        if(dt < 2000) {
            return;
        }
        Sample s;
        if(!captureSyncedColorFromPreview(s, 20000)) {
            return;
        }
        lastSampleTp_ = now;
        enqueueSampleJob(SampleJob{ activePairKey_, std::move(s) });
    }

    void startSampleWorker() {
        if(sampleThread_.joinable()) {
            return;
        }
        sampleStop_.store(false);
        sampleThread_ = std::thread([this]() { sampleWorkerLoop(); });
    }

    void stopSampleWorker() {
        sampleStop_.store(true);
        sampleCv_.notify_all();
        if(sampleThread_.joinable()) {
            sampleThread_.join();
        }
        std::lock_guard<std::mutex> lock(sampleMtx_);
        sampleQueue_.clear();
    }

    void enqueueSampleJob(SampleJob job) {
        if(sampleStop_.load()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(sampleMtx_);
            sampleQueue_.push_back(std::move(job));
            while(sampleQueue_.size() > 16) {
                sampleQueue_.pop_front();
            }
        }
        sampleCv_.notify_one();
    }

    void sampleWorkerLoop() {
        while(true) {
            SampleJob job;
            {
                std::unique_lock<std::mutex> lock(sampleMtx_);
                sampleCv_.wait(lock, [&]() { return sampleStop_.load() || !sampleQueue_.empty(); });
                if(sampleStop_.load()) {
                    return;
                }
                job = std::move(sampleQueue_.front());
                sampleQueue_.pop_front();
            }
            cv::Mat d1, d2;
            const bool ok = detectChessboard(job.sample, d1, d2);
            if(ok) {
                int count = 0;
                {
                    std::lock_guard<std::mutex> lock(pairMtx_);
                    auto &pd = pairStore_[job.pairKey];
                    pd.samples.push_back(Sample{ job.sample.ts1, job.sample.ts2, std::move(job.sample.img1), std::move(job.sample.img2) });
                    pd.validCount = static_cast<int>(pd.samples.size());
                    pd.latestDet1 = d1;
                    pd.latestDet2 = d2;
                    count = pd.validCount;
                }
                pushLog("sample accepted: " + job.pairKey + " count=" + std::to_string(count));
            }
            else {
                pushLog("sample rejected: chessboard not found");
            }
        }
    }

    void drawIcpPanel(cv::Mat &ui, FrameMouse &fm, int key, bool &running, const cv::Rect &left, const cv::Rect &mid, const cv::Rect &right) {
        cv::Rect f1(left.x + 12, left.y + 72, left.width - 24, 34);
        cv::Rect f2(left.x + 12, f1.y + 50, left.width - 24, 34);
        cv::Rect f3(left.x + 12, f2.y + 50, left.width - 24, 34);
        cv::Rect f4(left.x + 12, f3.y + 50, left.width - 24, 34);

        if(uiField(ui, f1, "max iterations", icpIter_, activeField_, "icpIter", fm)) activeField_ = "icpIter";
        if(uiField(ui, f2, "stop threshold (m)", icpStop_, activeField_, "icpStop", fm)) activeField_ = "icpStop";
        if(uiField(ui, f3, "stop threshold (deg)", icpStopRot_, activeField_, "icpStopRot", fm)) activeField_ = "icpStopRot";
        if(uiField(ui, f4, "max depth (m)", icpMaxDepth_, activeField_, "icpMaxDepth", fm)) activeField_ = "icpMaxDepth";

        if(!activeField_.empty() && key > 0) {
            if(activeField_ == "icpIter") handleText(icpIter_, key, 6);
            else if(activeField_ == "icpStop") handleText(icpStop_, key, 12);
            else if(activeField_ == "icpStopRot") handleText(icpStopRot_, key, 12);
            else if(activeField_ == "icpMaxDepth") handleText(icpMaxDepth_, key, 6);
        }
        cv::putText(ui, "last icp iterations: " + std::to_string(icpLastIter_), cv::Point(left.x + 12, f4.y + 50), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

        cv::Rect logBox(left.x + 10, f4.y + 72, left.width - 20, 320);
        drawLogBox(ui, logBox, fm);

        cv::Rect b1(left.x + 10, left.y + left.height - 120, left.width - 20, 30);
        cv::Rect b2(left.x + 10, left.y + left.height - 84, left.width - 20, 30);
        cv::Rect b3(left.x + 10, left.y + left.height - 48, left.width - 20, 30);
        if(uiButton(ui, b1, "start", fm)) onStartIcp();
        if(uiButton(ui, b2, "save", fm)) onSaveIcp();
        if(uiButton(ui, b3, "back to menu", fm)) {
            clearAllState();
            running = false;
        }

        cv::Rect big(mid.x, mid.y, mid.width + right.width, mid.height);
        cv::rectangle(ui, big, cv::Scalar(8, 8, 8), cv::FILLED);
        if(!icpPreview_.empty()) {
            cv::Mat rs;
            cv::resize(icpPreview_, rs, big.size(), 0.0, 0.0, cv::INTER_NEAREST);
            rs.copyTo(ui(big));
        }
        cv::putText(ui, "ICP Preview", cv::Point(big.x + 10, big.y + 28), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }

    void onStartIcp() {
        if(icpWork_.empty()) {
            if(!loadExtrinsics(icpWork_)) {
                pushLog("icp start failed: load extrinsic file failed");
                return;
            }
            for(const auto &d: devices_) {
                if(icpWork_.find(d.cfg.index) == icpWork_.end()) {
                    pushLog("icp start failed: missing camera in extrinsic file " + d.cfg.index);
                    return;
                }
            }
        }
        int maxIter = parseIntBound(icpIter_, 300, 1, 2000);
        double stopM = parseDoubleBound(icpStop_, 0.005, 0.00001, 1.0);
        double stopRotDeg = parseDoubleBound(icpStopRot_, 0.005, 0.00001, 45.0);
        double maxD = parseDoubleBound(icpMaxDepth_, 6.0, 0.1, 20.0);
        cfg_.maxDepth = static_cast<float>(maxD);

        if(runIcpOnce(maxIter, stopRotDeg, stopM)) {
            pushLog("icp refinement done");
        }
        else {
            pushLog("icp refinement failed");
        }
    }

    void onSaveIcp() {
        if(icpWork_.empty()) {
            pushLog("save failed: no icp result");
            return;
        }
        std::unordered_map<std::string, EdgeExtrinsic> rgbWorldToCam;
        for(const auto &dc: devices_) {
            auto it = icpWork_.find(dc.cfg.index);
            if(it == icpWork_.end() || !it->second.valid) {
                continue;
            }
            auto itD2c = depthToRgbByCam_.find(dc.cfg.index);
            if(itD2c != depthToRgbByCam_.end() && itD2c->second.valid) {
                const Eigen::Matrix4f Twd = toEigenWorldToCam(it->second);
                const Eigen::Matrix4f Tdr = toEigenWorldToCam(itD2c->second);
                rgbWorldToCam[dc.cfg.index] = fromEigenWorldToCam(makeRigid(Tdr * Twd));
            }
            else {
                rgbWorldToCam[dc.cfg.index] = it->second;
            }
        }
        if(writeExtrinsics(rgbWorldToCam)) {
            pushLog("icp result saved");
        }
        else {
            pushLog("save failed");
        }
    }

    void clearAllState() {
        stopActivePair();
        {
            std::lock_guard<std::mutex> lock(pairMtx_);
            pairStore_.clear();
        }
        edges_.clear();
        rgbToDepthByCam_.clear();
        depthToRgbByCam_.clear();
        live1_.release();
        live2_.release();
        icpWork_.clear();
        icpPreview_.release();
        {
            std::lock_guard<std::mutex> lock(icpDepthMtx_);
            icpDepthSlots_.clear();
        }
        pushLog("state cleared");
    }

    AppConfig cfg_;
    const std::atomic_bool *cancel_ = nullptr;
    ob::Context ctx_;
    std::vector<DeviceRuntimeLite> devices_;

    MouseState mouse_;
    std::vector<std::string> logs_;
    std::mutex logMtx_;
    int logScroll_ = 0;

    UIMode mode_ = UIMode::Chessboard;
    std::string activeField_;
    std::string camFirst_;
    std::string camSecond_;
    std::string cbCols_;
    std::string cbRows_;
    std::string icpIter_;
    std::string icpStop_;
    std::string icpStopRot_;
    std::string icpMaxDepth_;

    PairPipes activePair_;
    std::unordered_map<std::string, PairPreviewBuffer> previewBuffers_;
    std::string activeCam1_;
    std::string activeCam2_;
    std::mutex previewMtx_;
    bool pairStreaming_ = false;
    std::string activePairKey_;
    std::chrono::steady_clock::time_point lastSampleTp_{};
    cv::Mat live1_;
    cv::Mat live2_;

    std::unordered_map<std::string, PairData> pairStore_;
    std::mutex pairMtx_;
    std::unordered_map<std::string, EdgeExtrinsic> edges_;
    std::unordered_map<std::string, EdgeExtrinsic> rgbToDepthByCam_;
    std::unordered_map<std::string, EdgeExtrinsic> depthToRgbByCam_;

    std::atomic_bool sampleStop_{ false };
    std::thread sampleThread_;
    std::mutex sampleMtx_;
    std::condition_variable sampleCv_;
    std::deque<SampleJob> sampleQueue_;

    std::unordered_map<std::string, EdgeExtrinsic> icpWork_;
    std::unordered_map<std::string, IcpDepthSlot> icpDepthSlots_;
    std::mutex icpDepthMtx_;
    cv::Mat icpPreview_;
    int icpLastIter_ = 0;
};

int run_calibration(const AppConfig &cfg, const std::atomic_bool *cancel) {
    VisualCalibrator calibrator(cfg, cancel);
    return calibrator.run();
}

}  // namespace sync_app
