#include "calibration.hpp"

#include "utils/utils.hpp"

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include <Eigen/SVD>

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

class MultiDeviceCalibrator {
public:
    explicit MultiDeviceCalibrator(AppConfig cfg, const std::atomic_bool *cancel)
        : cfg_(std::move(cfg)), cancel_(cancel) {}

    int run() {
        auto deviceList = ctx_.queryDeviceList();
        if(deviceList->deviceCount() == 0) {
            std::cerr << "No devices found" << std::endl;
            return 1;
        }

        auto selected = selectDevices(deviceList);
        if(selected.empty()) {
            std::cerr << "No configured devices found in current device list" << std::endl;
            return 1;
        }

        if(cfg_.enableSync) {
            applySyncConfig(selected);
            ctx_.enableDeviceClockSync(60000);
        }

        const bool ansi = ob_smpl::supportAnsiEscape();
        bool quit = false;

        while(!quit) {
            if(cancel_ && cancel_->load()) {
                break;
            }
            std::cout << "==========calibration mode(press \"q\" for quitting)==========" << std::endl;
            std::cout << "choose calibration method:" << std::endl;
            std::cout << "    1. chessboard" << std::endl;
            std::cout << "    2. block" << std::endl;
            std::cout << "    3. icp" << std::endl;

            const auto input = readLine();
            if(input.empty() && cancel_ && cancel_->load()) {
                break;
            }
            if(isQuit(input)) {
                quit = true;
                break;
            }
            if(input == "1") {
                runChessboardCalibration(selected, ansi, quit);
            }
            else if(input == "2") {
                runBlockCalibration(ansi, quit);
            }
            else if(input == "3") {
                runIcpRefinement(selected, ansi);
            }
        }

        return 0;
    }

private:
    struct DeviceRuntime {
        DeviceConfig              cfg;
        std::shared_ptr<ob::Device> dev;
    };

    struct ColorSample {
        uint64_t ts1 = 0;
        uint64_t ts2 = 0;
        cv::Mat  img1;
        cv::Mat  img2;
    };

    struct EdgeExtrinsic {
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d t{0.0, 0.0, 0.0};
        bool valid = false;
    };

    std::string readLine() const {
        std::string line;
        while(true) {
            if(cancel_ && cancel_->load()) {
                return std::string();
            }
#if defined(__unix__) || defined(__APPLE__)
            pollfd pfd{};
            pfd.fd      = STDIN_FILENO;
            pfd.events  = POLLIN;
            pfd.revents = 0;
            const int rc = poll(&pfd, 1, 50);
            if(rc > 0 && (pfd.revents & POLLIN)) {
                if(!std::getline(std::cin, line)) {
                    return std::string();
                }
                break;
            }
#else
            if(!std::getline(std::cin, line)) {
                return std::string();
            }
            break;
#endif
        }
        size_t b = 0;
        while(b < line.size() && std::isspace(static_cast<unsigned char>(line[b]))) {
            b++;
        }
        size_t e = line.size();
        while(e > b && std::isspace(static_cast<unsigned char>(line[e - 1]))) {
            e--;
        }
        line = line.substr(b, e - b);
        std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return line;
    }

    static bool isQuit(const std::string &s) {
        return s == "q" || s == "quit" || s == "exit";
    }

    static std::string colorize(const std::string &s, const char *code, bool ansi) {
        if(!ansi) {
            return s;
        }
        return std::string("\033[") + code + "m" + s + "\033[0m";
    }

    static std::string normalizePairKey(std::string a, std::string b) {
        if(a > b) {
            std::swap(a, b);
        }
        return a + "|" + b;
    }

    static std::string edgeKey(const std::string &from, const std::string &to) {
        return from + "->" + to;
    }

    std::vector<DeviceRuntime> selectDevices(const std::shared_ptr<ob::DeviceList> &deviceList) {
        std::unordered_map<std::string, std::shared_ptr<ob::Device>> bySn;
        for(uint32_t i = 0; i < deviceList->deviceCount(); i++) {
            auto dev = deviceList->getDevice(i);
            bySn.emplace(std::string(dev->getDeviceInfo()->serialNumber()), dev);
        }

        std::vector<DeviceRuntime> out;
        out.reserve(cfg_.devices.size());
        for(const auto &dc: cfg_.devices) {
            auto it = bySn.find(dc.sn);
            if(it == bySn.end()) {
                std::cerr << "Configured device not found: " << dc.sn << std::endl;
                continue;
            }
            DeviceRuntime rt;
            rt.cfg = dc;
            rt.dev = it->second;
            out.push_back(std::move(rt));
        }
        return out;
    }

    void applySyncConfig(std::vector<DeviceRuntime> &devices) {
        for(size_t i = 0; i < devices.size(); i++) {
            auto &rt = devices[i];
            auto cur = rt.dev->getMultiDeviceSyncConfig();
            auto cfg = rt.cfg.hasSyncConfig ? rt.cfg.syncConfig : cur;
            if(!rt.cfg.hasSyncConfig) {
                cfg.syncMode = (i == 0) ? OB_MULTI_DEVICE_SYNC_MODE_PRIMARY : OB_MULTI_DEVICE_SYNC_MODE_SECONDARY;
            }

            cur.syncMode             = cfg.syncMode;
            cur.depthDelayUs         = cfg.depthDelayUs;
            cur.colorDelayUs         = cfg.colorDelayUs;
            cur.trigger2ImageDelayUs = cfg.trigger2ImageDelayUs;
            cur.triggerOutEnable     = cfg.triggerOutEnable;
            cur.triggerOutDelayUs    = cfg.triggerOutDelayUs;
            cur.framesPerTrigger     = cfg.framesPerTrigger;

            rt.dev->setMultiDeviceSyncConfig(cur);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    bool hasCameraIndex(const std::vector<DeviceRuntime> &devices, const std::string &idx) {
        for(const auto &rt: devices) {
            if(rt.cfg.index == idx) {
                return true;
            }
        }
        return false;
    }

    const DeviceRuntime *findByIndex(const std::vector<DeviceRuntime> &devices, const std::string &idx) {
        for(const auto &rt: devices) {
            if(rt.cfg.index == idx) {
                return &rt;
            }
        }
        return nullptr;
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

    static cv::Mat colorFrameToBgr(const std::shared_ptr<ob::Frame> &frame) {
        auto colorFrame = frame ? frame->as<ob::ColorFrame>() : nullptr;
        if(!colorFrame) {
            return cv::Mat();
        }

        thread_local auto converter = std::make_shared<ob::FormatConvertFilter>();
        if(colorFrame->format() != OB_FORMAT_BGR) {
            if(colorFrame->format() != OB_FORMAT_RGB) {
                if(colorFrame->format() == OB_FORMAT_MJPG) {
                    converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
                }
                else if(colorFrame->format() == OB_FORMAT_UYVY) {
                    converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
                }
                else if(colorFrame->format() == OB_FORMAT_YUYV) {
                    converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
                }
                else {
                    return cv::Mat();
                }
                colorFrame = converter->process(colorFrame)->as<ob::ColorFrame>();
                if(!colorFrame) {
                    return cv::Mat();
                }
            }
            converter->setFormatConvertType(FORMAT_RGB_TO_BGR);
            colorFrame = converter->process(colorFrame)->as<ob::ColorFrame>();
            if(!colorFrame) {
                return cv::Mat();
            }
        }

        const int width  = static_cast<int>(colorFrame->width());
        const int height = static_cast<int>(colorFrame->height());
        cv::Mat bgr(height, width, CV_8UC3, const_cast<void *>(colorFrame->data()));
        return bgr.clone();
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

    struct PairPipelines {
        std::shared_ptr<ob::Pipeline> p1;
        std::shared_ptr<ob::Pipeline> p2;
        cv::Mat K1, D1, K2, D2;
        int colorW = 0;
        int colorH = 0;
    };

    static std::shared_ptr<ob::VideoStreamProfile> defaultVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe, OBSensorType sensorType) {
        auto list = pipe->getStreamProfileList(sensorType);
        if(!list || list->getCount() == 0) {
            return nullptr;
        }
        return list->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
    }

    bool startPairPipelines(const DeviceRuntime &a, const DeviceRuntime &b, PairPipelines &out) {
        out.p1 = std::make_shared<ob::Pipeline>(a.dev);
        out.p2 = std::make_shared<ob::Pipeline>(b.dev);

        auto c1 = defaultVideoProfile(out.p1, OB_SENSOR_COLOR);
        auto d1 = defaultVideoProfile(out.p1, OB_SENSOR_DEPTH);
        auto c2 = defaultVideoProfile(out.p2, OB_SENSOR_COLOR);
        auto d2 = defaultVideoProfile(out.p2, OB_SENSOR_DEPTH);
        if(!c1 || !d1 || !c2 || !d2) {
            return false;
        }

        out.colorW = static_cast<int>(c1->getWidth());
        out.colorH = static_cast<int>(c1->getHeight());

        auto cfg1 = std::make_shared<ob::Config>();
        cfg1->enableStream(c1);
        cfg1->enableStream(d1);
        auto cfg2 = std::make_shared<ob::Config>();
        cfg2->enableStream(c2);
        cfg2->enableStream(d2);

        const auto mode1 = a.dev->getMultiDeviceSyncConfig().syncMode;
        const auto mode2 = b.dev->getMultiDeviceSyncConfig().syncMode;
        const bool p1IsPrimary   = (mode1 == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY);
        const bool p2IsSecondary = (mode2 == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY || mode2 == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED);

        if(p1IsPrimary && p2IsSecondary) {
            out.p2->start(cfg2);
            out.p1->start(cfg1);
        }
        else {
            out.p1->start(cfg1);
            out.p2->start(cfg2);
        }

        const auto cp1 = out.p1->getCameraParamWithProfile(c1->getWidth(), c1->getHeight(), d1->getWidth(), d1->getHeight());
        const auto cp2 = out.p2->getCameraParamWithProfile(c2->getWidth(), c2->getHeight(), d2->getWidth(), d2->getHeight());
        out.K1 = toCameraMatrix(cp1.rgbIntrinsic);
        out.D1 = toDistCoeffs(cp1.rgbDistortion);
        out.K2 = toCameraMatrix(cp2.rgbIntrinsic);
        out.D2 = toDistCoeffs(cp2.rgbDistortion);

        return true;
    }

    static void stopPairPipelines(PairPipelines &pp) {
        try {
            if(pp.p1) {
                pp.p1->stop();
            }
        }
        catch(...) {
        }
        try {
            if(pp.p2) {
                pp.p2->stop();
            }
        }
        catch(...) {
        }
        pp.p1.reset();
        pp.p2.reset();
    }

    static bool tryGetColorFrame(const std::shared_ptr<ob::Pipeline> &p, uint32_t timeoutMs, cv::Mat &img, uint64_t &tsUs) {
        auto fs = p->waitForFrameset(timeoutMs);
        if(!fs) {
            return false;
        }
        auto frame = fs->colorFrame();
        if(!frame) {
            return false;
        }
        tsUs = frameTimestampUs(frame);
        img  = colorFrameToBgr(frame);
        return !img.empty() && tsUs != 0;
    }

    bool captureSyncedPair(const PairPipelines &pp, uint64_t maxWaitMs, uint64_t maxDiffUs, ColorSample &out) {
        std::optional<cv::Mat>  img1;
        std::optional<cv::Mat>  img2;
        std::optional<uint64_t> ts1;
        std::optional<uint64_t> ts2;

        const auto start = std::chrono::steady_clock::now();
        while(true) {
            if(cancel_ && cancel_->load()) {
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count());
            if(elapsedMs > maxWaitMs) {
                return false;
            }

            if(!img1.has_value()) {
                cv::Mat im;
                uint64_t ts = 0;
                if(tryGetColorFrame(pp.p1, 50, im, ts)) {
                    img1 = std::move(im);
                    ts1  = ts;
                }
            }
            if(!img2.has_value()) {
                cv::Mat im;
                uint64_t ts = 0;
                if(tryGetColorFrame(pp.p2, 50, im, ts)) {
                    img2 = std::move(im);
                    ts2  = ts;
                }
            }

            if(img1.has_value() && img2.has_value() && ts1.has_value() && ts2.has_value()) {
                const uint64_t a = *ts1;
                const uint64_t b = *ts2;
                const uint64_t diff = (a > b) ? (a - b) : (b - a);
                if(diff <= maxDiffUs) {
                    out.ts1  = a;
                    out.ts2  = b;
                    out.img1 = img1->clone();
                    out.img2 = img2->clone();
                    return true;
                }
                if(a < b) {
                    img1.reset();
                    ts1.reset();
                }
                else {
                    img2.reset();
                    ts2.reset();
                }
            }
        }
    }

    bool computeRelativeExtrinsic(const PairPipelines &pp, const std::vector<ColorSample> &samples, cv::Matx33d &R12, cv::Vec3d &t12) {
        const auto &cb = cfg_.calibration.chessboard;
        const cv::Size patternSize(cb.cols, cb.rows);

        std::vector<std::vector<cv::Point3f>> objPts;
        std::vector<std::vector<cv::Point2f>> imgPts1;
        std::vector<std::vector<cv::Point2f>> imgPts2;

        std::vector<cv::Point3f> obj;
        obj.reserve(static_cast<size_t>(cb.cols * cb.rows));
        for(int y = 0; y < cb.rows; y++) {
            for(int x = 0; x < cb.cols; x++) {
                obj.emplace_back(static_cast<float>(x) * cb.squareSize, static_cast<float>(y) * cb.squareSize, 0.0f);
            }
        }

        for(const auto &s: samples) {
            if(s.img1.empty() || s.img2.empty()) {
                continue;
            }
            cv::Mat g1, g2;
            cv::cvtColor(s.img1, g1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(s.img2, g2, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners1;
            std::vector<cv::Point2f> corners2;
            const bool ok1 = cv::findChessboardCorners(g1, patternSize, corners1, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            const bool ok2 = cv::findChessboardCorners(g2, patternSize, corners2, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if(!ok1 || !ok2) {
                continue;
            }

            cv::cornerSubPix(g1, corners1, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(g2, corners2, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

            objPts.push_back(obj);
            imgPts1.push_back(std::move(corners1));
            imgPts2.push_back(std::move(corners2));
        }

        if(objPts.size() < 3) {
            return false;
        }

        cv::Mat K1 = pp.K1.clone();
        cv::Mat D1 = pp.D1.clone();
        cv::Mat K2 = pp.K2.clone();
        cv::Mat D2 = pp.D2.clone();

        int flags = cv::CALIB_FIX_INTRINSIC;
        if((D1.total() == 8 && (D1.rows == 1 || D1.cols == 1)) && (D2.total() == 8 && (D2.rows == 1 || D2.cols == 1))) {
            flags |= cv::CALIB_RATIONAL_MODEL;
        }

        cv::Mat R, T, E, F;
        const cv::Size imageSize(pp.colorW, pp.colorH);
        cv::stereoCalibrate(objPts, imgPts1, imgPts2, K1, D1, K2, D2, imageSize, R, T, E, F, flags,
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

    static bool isPrimary(const std::shared_ptr<ob::Device> &dev) {
        if(!dev) {
            return false;
        }
        try {
            return dev->getMultiDeviceSyncConfig().syncMode == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY;
        }
        catch(...) {
            return false;
        }
    }

    struct IcpDevice {
        std::string index;
        std::shared_ptr<ob::Device> dev;
        std::shared_ptr<ob::Pipeline> pipe;
        std::shared_ptr<ob::PointCloudFilter> pcFilter;
        OrbbecDepthFilterChain depthFilters;
    };

    bool startIcpPipelines(std::vector<IcpDevice> &devs) {
        for(auto &d: devs) {
            d.pipe     = std::make_shared<ob::Pipeline>(d.dev);
            d.pcFilter = std::make_shared<ob::PointCloudFilter>();
            d.pcFilter->setCreatePointFormat(OB_FORMAT_POINT);
            d.pcFilter->setCoordinateSystem(OB_RIGHT_HAND_COORDINATE_SYSTEM);
            d.pcFilter->setDecimationFactor(cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 1);
        }

        std::vector<IcpDevice *> secondary;
        std::vector<IcpDevice *> primary;
        for(auto &d: devs) {
            if(isPrimary(d.dev)) {
                primary.push_back(&d);
            }
            else {
                secondary.push_back(&d);
            }
        }

        auto startOne = [](IcpDevice &d) -> bool {
            auto depthProfile = defaultVideoProfile(d.pipe, OB_SENSOR_DEPTH);
            if(!depthProfile) {
                return false;
            }
            auto cfg = std::make_shared<ob::Config>();
            cfg->enableStream(depthProfile);
            d.pipe->start(cfg);
            return true;
        };

        for(auto *d: secondary) {
            if(!startOne(*d)) {
                return false;
            }
        }
        for(auto *d: primary) {
            if(!startOne(*d)) {
                return false;
            }
        }
        return true;
    }

    static void stopIcpPipelines(std::vector<IcpDevice> &devs) {
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
    }

    static bool tryGetDepthFrame(const std::shared_ptr<ob::Pipeline> &p, uint32_t timeoutMs, std::shared_ptr<ob::Frame> &depth, uint64_t &tsUs) {
        auto fs = p->waitForFrameset(timeoutMs);
        if(!fs) {
            return false;
        }
        auto frame = fs->depthFrame();
        if(!frame) {
            return false;
        }
        tsUs = frameTimestampUs(frame);
        if(tsUs == 0) {
            return false;
        }
        depth = frame;
        return true;
    }

    bool captureSyncedDepthSet(std::vector<IcpDevice> &devs,
                               uint64_t maxWaitMs,
                               uint64_t maxDiffUs,
                               std::unordered_map<std::string, std::pair<uint64_t, std::shared_ptr<ob::Frame>>> &out) {
        struct Slot {
            std::shared_ptr<ob::Frame> depth;
            uint64_t tsUs = 0;
            bool ready = false;
        };
        std::vector<Slot> slots(devs.size());

        const auto start = std::chrono::steady_clock::now();
        while(true) {
            if(cancel_ && cancel_->load()) {
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count());
            if(elapsedMs > maxWaitMs) {
                return false;
            }

            for(size_t i = 0; i < devs.size(); i++) {
                if(slots[i].ready) {
                    continue;
                }
                std::shared_ptr<ob::Frame> depth;
                uint64_t tsUs = 0;
                if(tryGetDepthFrame(devs[i].pipe, 50, depth, tsUs)) {
                    slots[i].depth = std::move(depth);
                    slots[i].tsUs  = tsUs;
                    slots[i].ready = true;
                }
            }

            bool allReady = true;
            for(const auto &s: slots) {
                if(!s.ready) {
                    allReady = false;
                    break;
                }
            }
            if(!allReady) {
                continue;
            }

            uint64_t minTs = std::numeric_limits<uint64_t>::max();
            uint64_t maxTs = 0;
            size_t minIdx = 0;
            for(size_t i = 0; i < slots.size(); i++) {
                const auto ts = slots[i].tsUs;
                if(ts < minTs) {
                    minTs = ts;
                    minIdx = i;
                }
                if(ts > maxTs) {
                    maxTs = ts;
                }
            }

            if(maxTs - minTs <= maxDiffUs) {
                out.clear();
                for(size_t i = 0; i < devs.size(); i++) {
                    out.emplace(devs[i].index, std::make_pair(slots[i].tsUs, slots[i].depth));
                }
                return true;
            }

            slots[minIdx] = Slot{};
        }
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

    static pcl::PointCloud<pcl::PointXYZ>::Ptr pointsFrameToCloudCam(const std::shared_ptr<ob::PointsFrame> &pointsFrame, float minZ, float maxZ) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        (void)minZ;
        (void)maxZ;
        if(!pointsFrame) {
            return cloud;
        }

        const float scaleMm = pointsFrame->getCoordinateValueScale();
        const auto *data = reinterpret_cast<const OBPoint *>(pointsFrame->data());
        const auto count = pointsFrame->dataSize() / sizeof(OBPoint);
        if(!data || count == 0) {
            return cloud;
        }

        cloud->points.reserve(static_cast<size_t>(count));
        for(size_t i = 0; i < count; i++) {
            const OBPoint &p = data[i];
            if(!std::isfinite(p.z) || p.z <= 0.0f) {
                continue;
            }
            const float z = p.z * scaleMm * 0.001f;
            if(!std::isfinite(z) || z <= 0.0f) {
                continue;
            }
            const float x = p.x * scaleMm * 0.001f;
            const float y = p.y * scaleMm * 0.001f;
            if(!std::isfinite(x) || !std::isfinite(y)) {
                continue;
            }
            cloud->points.emplace_back(x, y, z);
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    bool refineExtrinsicsWithPclIcp(std::vector<DeviceRuntime> &devices, std::unordered_map<std::string, EdgeExtrinsic> &worldToCam, const std::string &rootIdx, int &outItersUsed) {
        outItersUsed = 0;
        const int frames = 10;
        const uint64_t maxWaitMs = 4000;
        const uint64_t maxDiffUs = 20000;
        const float minZ = 0.2f;
        const float maxZ = (cfg_.maxDepth > 0.0f) ? cfg_.maxDepth : 6.0f;
        const float maxCorrespondenceDist = 0.08f;
        const int maxOuterIters = 300;
        const int icpInnerIters = 1;
        const double stopRotDeg = 0.005;
        const double stopTransM = 0.0005;

        const auto itRoot = worldToCam.find(rootIdx);
        if(itRoot == worldToCam.end() || !itRoot->second.valid) {
            return false;
        }

        for(auto &kv: worldToCam) {
            if(!kv.second.valid) {
                continue;
            }
            kv.second = fromEigenWorldToCam(toEigenWorldToCam(kv.second));
        }

        std::vector<IcpDevice> devs;
        devs.reserve(devices.size());
        for(auto &rt: devices) {
            IcpDevice d;
            d.index = rt.cfg.index;
            d.dev = rt.dev;
            devs.push_back(std::move(d));
        }

        if(!startIcpPipelines(devs)) {
            stopIcpPipelines(devs);
            return false;
        }

        std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudsW;
        cloudsW.reserve(devs.size());
        for(const auto &d: devs) {
            cloudsW.emplace(d.index, pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>());
        }

        for(int f = 0; f < frames; f++) {
            if(cancel_ && cancel_->load()) {
                break;
            }
            std::unordered_map<std::string, std::pair<uint64_t, std::shared_ptr<ob::Frame>>> depthFrames;
            if(!captureSyncedDepthSet(devs, maxWaitMs, maxDiffUs, depthFrames)) {
                break;
            }

            for(auto &d: devs) {
                const auto it = depthFrames.find(d.index);
                if(it == depthFrames.end()) {
                    continue;
                }
                auto depth = it->second.second;
                if(!depth) {
                    continue;
                }

                std::shared_ptr<ob::Frame> depthForPc = depth;
                depthForPc = refineDepthFrameForPointCloud(depthForPc, d.depthFilters, minZ, maxZ, cfg_.filters);
                if(!depthForPc) {
                    continue;
                }

                std::shared_ptr<ob::Frame> pcFrame;
                try {
                    pcFrame = d.pcFilter->process(depthForPc);
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

                const auto itEx = worldToCam.find(d.index);
                if(itEx == worldToCam.end() || !itEx->second.valid) {
                    continue;
                }
                const Eigen::Matrix4f Tcw = toEigenWorldToCam(itEx->second);
                const Eigen::Matrix4f Twc = invertRigid(Tcw);

                auto cloudCam = pointsFrameToCloudCam(pointsFrame, minZ, maxZ);
                auto cloudW = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                pcl::transformPointCloud(*cloudCam, *cloudW, Twc);

                auto &acc = cloudsW[d.index];
                *acc += *cloudW;
            }
        }

        stopIcpPipelines(devs);

        for(auto &kv: cloudsW) {
            if(cfg_.filters.deskCrop) {
                kv.second = removeDominantPlaneRansacCompat(kv.second, 100, 0.015, 0.25, 1500);
            }
            kv.second = denoiseCloudSor(kv.second, 20, 1.0);
        }

        const auto mergeCloudsExcluding = [&](const std::string &excludeIdx) -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
            auto merged = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for(const auto &dc: cfg_.devices) {
                if(dc.index == excludeIdx) {
                    continue;
                }
                const auto it = cloudsW.find(dc.index);
                if(it == cloudsW.end() || !it->second || it->second->empty()) {
                    continue;
                }
                *merged += *it->second;
            }
            merged->width = static_cast<uint32_t>(merged->points.size());
            merged->height = 1;
            merged->is_dense = false;
            return merged;
        };
        const auto calcRotDegFromR = [](const Eigen::Matrix3f &Rin) -> double {
            const Eigen::Matrix3f R = projectRotationToSO3(Rin);
            double tr = static_cast<double>(R.trace());
            double c = (tr - 1.0) * 0.5;
            c = std::min(1.0, std::max(-1.0, c));
            const double angle = std::acos(c);
            return angle * (180.0 / 3.14159265358979323846);
        };

        const auto tryParseIndex = [](const std::string &s) -> std::optional<int> {
            if(s.empty()) {
                return std::nullopt;
            }
            for(const char c: s) {
                if(!std::isdigit(static_cast<unsigned char>(c))) {
                    return std::nullopt;
                }
            }
            try {
                return std::stoi(s);
            }
            catch(...) {
                return std::nullopt;
            }
        };

        std::vector<std::string> ordered;
        ordered.reserve(cfg_.devices.size());
        for(const auto &dc: cfg_.devices) {
            ordered.push_back(dc.index);
        }
        std::sort(ordered.begin(), ordered.end(), [&](const std::string &a, const std::string &b) {
            const auto ia = tryParseIndex(a);
            const auto ib = tryParseIndex(b);
            if(ia.has_value() && ib.has_value()) {
                if(*ia != *ib) {
                    return *ia < *ib;
                }
                return a < b;
            }
            return a < b;
        });
        ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
        const auto itRootPos = std::find(ordered.begin(), ordered.end(), rootIdx);
        if(itRootPos != ordered.end() && itRootPos != ordered.begin()) {
            const std::string r = *itRootPos;
            ordered.erase(itRootPos);
            ordered.insert(ordered.begin(), r);
        }

        if(ordered.empty() || ordered.front() != rootIdx) {
            return true;
        }

        int itersUsed = 0;
        for(int iter = 0; iter < maxOuterIters; iter++) {
            if(cancel_ && cancel_->load()) {
                break;
            }
            bool allSmall = true;
            for(size_t i = 0; i < ordered.size(); i++) {
                const std::string &curIdx = ordered[i];
                const auto itSrc = cloudsW.find(curIdx);
                if(itSrc == cloudsW.end() || !itSrc->second || itSrc->second->size() < 100) {
                    continue;
                }

                auto target = mergeCloudsExcluding(curIdx);
                if(!target || target->size() < 200) {
                    continue;
                }

                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                icp.setInputSource(itSrc->second);
                icp.setInputTarget(target);
                icp.setMaximumIterations(icpInnerIters);
                icp.setMaxCorrespondenceDistance(maxCorrespondenceDist);
                icp.setTransformationEpsilon(1e-10);
                icp.setEuclideanFitnessEpsilon(1e-9);
                icp.setUseReciprocalCorrespondences(true);
                icp.setRANSACOutlierRejectionThreshold(maxCorrespondenceDist * 0.5f);

                auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                icp.align(*aligned);
                if(!icp.hasConverged()) {
                    continue;
                }

                const Eigen::Matrix4f delta = makeRigid(icp.getFinalTransformation());
                cloudsW[curIdx] = aligned;

                const auto itEx = worldToCam.find(curIdx);
                if(itEx != worldToCam.end() && itEx->second.valid) {
                    const Eigen::Matrix4f Tcw_old = toEigenWorldToCam(itEx->second);
                    const Eigen::Matrix4f Twc_old = invertRigid(Tcw_old);
                    const Eigen::Matrix4f Twc_new = delta * Twc_old;
                    const Eigen::Matrix4f Tcw_new = invertRigid(Twc_new);
                    worldToCam[curIdx] = fromEigenWorldToCam(Tcw_new);
                }

                const double rotDeg = calcRotDegFromR(delta.block<3, 3>(0, 0));
                const double transM = static_cast<double>(delta.block<3, 1>(0, 3).norm());
                if(rotDeg > stopRotDeg || transM > stopTransM) {
                    allSmall = false;
                }
            }
            itersUsed = iter + 1;
            if(allSmall) {
                break;
            }
        }
        outItersUsed = itersUsed;

        const auto itRootFinal = worldToCam.find(rootIdx);
        if(itRootFinal != worldToCam.end() && itRootFinal->second.valid) {
            const Eigen::Matrix4f Tcw_root = toEigenWorldToCam(itRootFinal->second);
            const Eigen::Matrix4f Trw = invertRigid(Tcw_root);
            for(const auto &dc: cfg_.devices) {
                auto it = worldToCam.find(dc.index);
                if(it == worldToCam.end() || !it->second.valid) {
                    continue;
                }
                const Eigen::Matrix4f Tcw = toEigenWorldToCam(it->second);
                const Eigen::Matrix4f Tcr = makeRigid(Tcw * Trw);
                it->second = fromEigenWorldToCam(Tcr);
            }

            auto itRootSet = worldToCam.find(rootIdx);
            if(itRootSet != worldToCam.end()) {
                itRootSet->second.valid = true;
                itRootSet->second.R = cv::Matx33d::eye();
                itRootSet->second.t = cv::Vec3d(0.0, 0.0, 0.0);
            }
        }

        return true;
    }

    static bool parseVec3d(cJSON *arr, cv::Vec3d &out) {
        if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
            return false;
        }
        const auto *a0 = cJSON_GetArrayItem(arr, 0);
        const auto *a1 = cJSON_GetArrayItem(arr, 1);
        const auto *a2 = cJSON_GetArrayItem(arr, 2);
        if(!a0 || !a1 || !a2 || !cJSON_IsNumber(a0) || !cJSON_IsNumber(a1) || !cJSON_IsNumber(a2)) {
            return false;
        }
        out = cv::Vec3d(a0->valuedouble, a1->valuedouble, a2->valuedouble);
        return true;
    }

    static bool parseMat3d(cJSON *arr, cv::Matx33d &out) {
        if(!arr || !cJSON_IsArray(arr)) {
            return false;
        }
        const int n = cJSON_GetArraySize(arr);
        if(n == 9) {
            double v[9];
            for(int i = 0; i < 9; i++) {
                auto *it = cJSON_GetArrayItem(arr, i);
                if(!it || !cJSON_IsNumber(it)) {
                    return false;
                }
                v[i] = it->valuedouble;
            }
            out = cv::Matx33d(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
            return true;
        }
        if(n == 3) {
            auto *r0 = cJSON_GetArrayItem(arr, 0);
            auto *r1 = cJSON_GetArrayItem(arr, 1);
            auto *r2 = cJSON_GetArrayItem(arr, 2);
            if(r0 && r1 && r2 && cJSON_IsArray(r0) && cJSON_IsArray(r1) && cJSON_IsArray(r2) && cJSON_GetArraySize(r0) == 3 && cJSON_GetArraySize(r1) == 3
               && cJSON_GetArraySize(r2) == 3) {
                double v[9];
                cJSON *rows[3] = { r0, r1, r2 };
                for(int y = 0; y < 3; y++) {
                    for(int x = 0; x < 3; x++) {
                        auto *it = cJSON_GetArrayItem(rows[y], x);
                        if(!it || !cJSON_IsNumber(it)) {
                            return false;
                        }
                        v[y * 3 + x] = it->valuedouble;
                    }
                }
                out = cv::Matx33d(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
                return true;
            }
        }
        return false;
    }

    bool loadWorldToCamFromFile(const fs::path &path, std::unordered_map<std::string, EdgeExtrinsic> &worldToCam) {
        std::string content;
        try {
            content = readFileAllLocal(path);
        }
        catch(const std::exception &e) {
            std::cerr << "Failed to read init_extrinsic_path: " << path.string() << ", error=" << e.what() << std::endl;
            return false;
        }

        cJSON *root = cJSON_Parse(content.c_str());
        if(!root || !cJSON_IsObject(root)) {
            if(root) {
                cJSON_Delete(root);
            }
            std::cerr << "Invalid JSON in init_extrinsic_path: " << path.string() << std::endl;
            return false;
        }

        worldToCam.clear();
        for(cJSON *item = root->child; item != nullptr; item = item->next) {
            if(!item->string || !cJSON_IsObject(item)) {
                continue;
            }
            const std::string camId = item->string;
            auto *rotArr = cJSON_GetObjectItemCaseSensitive(item, "rotation");
            auto *tArr = cJSON_GetObjectItemCaseSensitive(item, "translation");

            cv::Vec3d tCw;
            cv::Matx33d Rcw = cv::Matx33d::eye();
            if(!parseVec3d(tArr, tCw) || !parseMat3d(rotArr, Rcw)) {
                continue;
            }
            EdgeExtrinsic ex;
            ex.valid = true;
            ex.R = Rcw;
            ex.t = tCw;
            worldToCam.emplace(camId, ex);
        }

        cJSON_Delete(root);
        return true;
    }

    bool writeWorldToCamToFile(const std::unordered_map<std::string, EdgeExtrinsic> &worldToCam) {
        if(cfg_.initExtrinsicPath.empty()) {
            std::cerr << "init_extrinsic_path is empty" << std::endl;
            return false;
        }

        cJSON *root = cJSON_CreateObject();
        for(const auto &dc: cfg_.devices) {
            const auto it = worldToCam.find(dc.index);
            if(it == worldToCam.end() || !it->second.valid) {
                continue;
            }
            const auto &ex = it->second;

            cJSON *camObj = cJSON_CreateObject();
            cJSON *rot    = cJSON_CreateArray();
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

            cJSON_AddItemToObject(camObj, "rotation", rot);
            cJSON_AddItemToObject(camObj, "translation", t);
            cJSON_AddItemToObject(root, dc.index.c_str(), camObj);
        }

        char *printed = cJSON_Print(root);
        cJSON_Delete(root);
        if(!printed) {
            return false;
        }

        fs::path p(cfg_.initExtrinsicPath);
        if(p.is_relative()) {
            p = fs::absolute(p);
        }
        const fs::path absP = fs::absolute(p);

        if(absP.has_parent_path()) {
            try {
                fs::create_directories(absP.parent_path());
            }
            catch(const std::exception &e) {
                std::cerr << "Failed to create directories: " << absP.parent_path().string() << ", error=" << e.what() << std::endl;
                cJSON_free(printed);
                return false;
            }
        }

        std::ofstream f(absP, std::ios::binary | std::ios::trunc);
        if(!f.is_open()) {
            std::cerr << "Failed to open for writing: " << absP.string() << std::endl;
            cJSON_free(printed);
            return false;
        }
        f << printed;
        f.flush();
        const bool ok = f.good();
        f.close();
        cJSON_free(printed);
        return ok;
    }

    bool writeCoarseExtrinsics(const std::string &rootIdx, const std::unordered_map<std::string, EdgeExtrinsic> &edges) {
        std::unordered_map<std::string, EdgeExtrinsic> worldToCam;
        std::unordered_set<std::string> visited;
        std::deque<std::string> q;

        EdgeExtrinsic rootEx;
        rootEx.valid = true;
        worldToCam[rootIdx] = rootEx;
        visited.insert(rootIdx);
        q.push_back(rootIdx);

        while(!q.empty()) {
            if(cancel_ && cancel_->load()) {
                return false;
            }
            const auto u = q.front();
            q.pop_front();
            const auto exU = worldToCam[u];

            for(const auto &kv: edges) {
                const auto &k = kv.first;
                const auto &e = kv.second;
                if(!e.valid) {
                    continue;
                }
                const auto pos = k.find("->");
                if(pos == std::string::npos) {
                    continue;
                }
                const auto from = k.substr(0, pos);
                const auto to = k.substr(pos + 2);
                if(from != u) {
                    continue;
                }
                if(visited.find(to) != visited.end()) {
                    continue;
                }

                EdgeExtrinsic exV;
                exV.valid = true;
                exV.R = e.R * exU.R;
                exV.t = e.R * exU.t + e.t;
                worldToCam[to] = exV;
                visited.insert(to);
                q.push_back(to);
            }
        }

        for(const auto &dc: cfg_.devices) {
            if(worldToCam.find(dc.index) == worldToCam.end()) {
                std::cerr << "Missing calibration path from " << rootIdx << " to " << dc.index << std::endl;
                return false;
            }
        }

        return writeWorldToCamToFile(worldToCam);
    }

    void runIcpRefinement(std::vector<DeviceRuntime> &devices, bool ansi) {
        if(cfg_.initExtrinsicPath.empty()) {
            std::cout << colorize("init_extrinsic_path is empty", "31", ansi) << std::endl;
            return;
        }

        fs::path p(cfg_.initExtrinsicPath);
        if(p.is_relative()) {
            p = fs::absolute(p);
        }
        const fs::path absP = fs::absolute(p);

        std::unordered_map<std::string, EdgeExtrinsic> worldToCam;
        if(!loadWorldToCamFromFile(absP, worldToCam)) {
            std::cout << colorize("failed to read coarse extrinsics from file", "31", ansi) << std::endl;
            return;
        }

        for(const auto &dc: cfg_.devices) {
            const auto it = worldToCam.find(dc.index);
            if(it == worldToCam.end() || !it->second.valid) {
                std::cout << colorize("missing camera extrinsic in file: " + dc.index, "31", ansi) << std::endl;
                return;
            }
        }

        const std::string rootIdx = hasCameraIndex(devices, "00") ? std::string("00") : devices.front().cfg.index;
        int itersUsed = 0;
        if(!refineExtrinsicsWithPclIcp(devices, worldToCam, rootIdx, itersUsed)) {
            std::cout << colorize("ICP refinement failed", "31", ansi) << std::endl;
            return;
        }
        std::cout << colorize("ICP iterations: " + std::to_string(itersUsed), "36", ansi) << std::endl;

        if(writeWorldToCamToFile(worldToCam)) {
            std::cout << colorize("ICP refined extrinsics saved to: " + absP.string(), "32", ansi) << std::endl;
        }
        else {
            std::cout << colorize("failed to save ICP refined extrinsics", "31", ansi) << std::endl;
        }
    }

    void runBlockCalibration(bool ansi, bool &quit) {
        std::cout << colorize("block calibration is not implemented yet", "33", ansi) << std::endl;
        const auto input = readLine();
        if(isQuit(input)) {
            quit = true;
        }
    }

    void runChessboardCalibration(std::vector<DeviceRuntime> &devices, bool ansi, bool &quit) {
        std::unordered_set<std::string> calibratedPairKeys;
        std::vector<std::pair<std::string, std::string>> calibratedPairs;
        std::unordered_map<std::string, EdgeExtrinsic> edges;

        const std::string rootIdx = hasCameraIndex(devices, "00") ? std::string("00") : devices.front().cfg.index;

        while(!quit) {
            if(cancel_ && cancel_->load()) {
                quit = true;
                break;
            }
            std::cout << "==========chessboard calibration(press \"q\" for quitting)==========" << std::endl;
            if(!calibratedPairs.empty()) {
                std::cout << "camera calibrated:" << std::endl;
                for(const auto &p: calibratedPairs) {
                    std::cout << "    " << colorize("(" + p.first + "," + p.second + ")", "32", ansi) << std::endl;
                }
            }
            std::cout << colorize("press \"c\" to save coarse calibration result", "36", ansi) << std::endl;
            std::cout << "    index of first camera(e.g. \"00\"): ";
            std::cout.flush();
            const auto first = readLine();
            if(first.empty() && cancel_ && cancel_->load()) {
                quit = true;
                break;
            }
            if(isQuit(first)) {
                quit = true;
                break;
            }
            if(first == "c") {
                if(edges.empty()) {
                    std::cout << colorize("no calibrated pairs yet", "33", ansi) << std::endl;
                    continue;
                }
                if(writeCoarseExtrinsics(rootIdx, edges)) {
                    const fs::path absP = fs::absolute(fs::path(cfg_.initExtrinsicPath));
                    std::cout << colorize("coarse calibration saved to: " + absP.string(), "32", ansi) << std::endl;
                }
                else {
                    std::cout << colorize("failed to calculate/save coarse calibration (missing paths?)", "31", ansi) << std::endl;
                }
                continue;
            }
            if(!hasCameraIndex(devices, first)) {
                std::cout << colorize("invalid camera index: " + first, "31", ansi) << std::endl;
                continue;
            }
            std::cout << "    index of second camera(e.g. \"01\"): ";
            std::cout.flush();
            const auto second = readLine();
            if(second.empty() && cancel_ && cancel_->load()) {
                quit = true;
                break;
            }
            if(isQuit(second)) {
                quit = true;
                break;
            }
            if(!hasCameraIndex(devices, second)) {
                std::cout << colorize("invalid camera index: " + second, "31", ansi) << std::endl;
                continue;
            }
            if(first == second) {
                std::cout << colorize("two camera indices are the same", "31", ansi) << std::endl;
                continue;
            }

            const auto np = normalizePairKey(first, second);
            if(calibratedPairKeys.find(np) != calibratedPairKeys.end()) {
                std::cout << colorize("this camera pair has been calibrated, please choose another pair", "33", ansi) << std::endl;
                continue;
            }

            const auto *rt1 = findByIndex(devices, first);
            const auto *rt2 = findByIndex(devices, second);
            if(!rt1 || !rt2) {
                std::cout << colorize("failed to locate devices for indices", "31", ansi) << std::endl;
                continue;
            }

            if(!runPairSampling(*rt1, *rt2, ansi, quit, edges)) {
                if(quit) {
                    break;
                }
                std::cout << colorize("pair calibration failed", "31", ansi) << std::endl;
                continue;
            }

            calibratedPairKeys.insert(np);
            calibratedPairs.push_back({ first, second });
        }
    }

    bool runPairSampling(const DeviceRuntime &rt1,
                         const DeviceRuntime &rt2,
                         bool ansi,
                         bool &quit,
                         std::unordered_map<std::string, EdgeExtrinsic> &edges) {
        PairPipelines pp;
        if(!startPairPipelines(rt1, rt2, pp)) {
            stopPairPipelines(pp);
            return false;
        }

        std::vector<ColorSample> samples;
        while(!quit) {
            if(cancel_ && cancel_->load()) {
                quit = true;
                break;
            }
            std::cout << "==========chessboard calibration(press \"q\" for quitting)==========" << std::endl;
            std::cout << "calibrating " << rt1.cfg.index << " and " << rt2.cfg.index << std::endl;
            std::cout << "    press \"1\" to sample images" << std::endl;
            std::cout << "    press \"0\" to calculate relative extrinsic" << std::endl;
            const char key = ob_smpl::waitForKeyPressed(0);
            if(key == 'q' || key == 'Q') {
                quit = true;
                break;
            }
            if(key == '1') {
                ColorSample s;
                if(captureSyncedPair(pp, 2000, 20000, s)) {
                    samples.push_back(ColorSample{ s.ts1, s.ts2, std::move(s.img1), std::move(s.img2) });
                    std::cout << "captured image:" << std::endl;
                    std::cout << "    " << rt1.cfg.index << ": " << s.ts1 << ",  " << rt2.cfg.index << ": " << s.ts2 << std::endl;
                }
                else {
                    std::cout << colorize("capture timeout, please try again", "33", ansi) << std::endl;
                }
            }
            else if(key == '0') {
                cv::Matx33d R12;
                cv::Vec3d t12;
                if(!computeRelativeExtrinsic(pp, samples, R12, t12)) {
                    std::cout << colorize("not enough valid chessboard samples", "33", ansi) << std::endl;
                    continue;
                }

                EdgeExtrinsic e12;
                e12.valid = true;
                e12.R = R12;
                e12.t = t12;
                edges[edgeKey(rt1.cfg.index, rt2.cfg.index)] = e12;

                EdgeExtrinsic e21;
                e21.valid = true;
                e21.R = R12.t();
                e21.t = -(e21.R * t12);
                edges[edgeKey(rt2.cfg.index, rt1.cfg.index)] = e21;

                break;
            }
        }

        stopPairPipelines(pp);
        return !quit;
    }

private:
    AppConfig cfg_;
    const std::atomic_bool *cancel_;
    ob::Context ctx_;
};

int run_calibration(const AppConfig &cfg, const std::atomic_bool *cancel) {
    MultiDeviceCalibrator calibrator(cfg, cancel);
    return calibrator.run();
}

}  // namespace sync_app
