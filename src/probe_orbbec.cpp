#include <libobsensor/ObSensor.hpp>

#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string hex4(int value) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << value << std::dec;
    return oss.str();
}

std::string safeText(const char *s) {
    return (s && *s) ? std::string(s) : std::string("(empty)");
}

std::string syncModeName(OBMultiDeviceSyncMode mode) {
    switch(mode) {
    case OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN:
        return "OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN";
    case OB_MULTI_DEVICE_SYNC_MODE_STANDALONE:
        return "OB_MULTI_DEVICE_SYNC_MODE_STANDALONE";
    case OB_MULTI_DEVICE_SYNC_MODE_PRIMARY:
        return "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY";
    case OB_MULTI_DEVICE_SYNC_MODE_SECONDARY:
        return "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY";
    case OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED:
        return "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED";
    case OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING:
        return "OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING";
    case OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING:
        return "OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING";
    default:
        return "UNKNOWN_SYNC_MODE";
    }
}

void printSyncModeIfSupported(uint16_t bitmap, OBMultiDeviceSyncMode mode) {
    if((bitmap & static_cast<uint16_t>(mode)) != 0) {
        std::cout << "      - " << syncModeName(mode) << '\n';
    }
}

void printSupportedSyncModes(const std::shared_ptr<ob::Device> &device) {
    try {
        const uint16_t bitmap = device->getSupportedMultiDeviceSyncModeBitmap();
        std::cout << "    supported_sync_modes_bitmap: 0x" << std::hex << std::uppercase << bitmap << std::dec << '\n';
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN);
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_STANDALONE);
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_PRIMARY);
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_SECONDARY);
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED);
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING);
        printSyncModeIfSupported(bitmap, OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING);
    }
    catch(const std::exception &e) {
        std::cout << "    supported_sync_modes: unavailable (" << e.what() << ")\n";
    }
}

void printCurrentSyncConfig(const std::shared_ptr<ob::Device> &device) {
    try {
        const auto cfg = device->getMultiDeviceSyncConfig();
        std::cout << "    current_sync_config:\n";
        std::cout << "      syncMode: " << syncModeName(cfg.syncMode) << '\n';
        std::cout << "      depthDelayUs: " << cfg.depthDelayUs << '\n';
        std::cout << "      colorDelayUs: " << cfg.colorDelayUs << '\n';
        std::cout << "      trigger2ImageDelayUs: " << cfg.trigger2ImageDelayUs << '\n';
        std::cout << "      triggerOutEnable: " << (cfg.triggerOutEnable ? "true" : "false") << '\n';
        std::cout << "      triggerOutDelayUs: " << cfg.triggerOutDelayUs << '\n';
        std::cout << "      framesPerTrigger: " << cfg.framesPerTrigger << '\n';
    }
    catch(const std::exception &e) {
        std::cout << "    current_sync_config: unavailable (" << e.what() << ")\n";
    }
}

void applySyncConfig(const std::shared_ptr<ob::Device> &device, OBMultiDeviceSyncMode mode, bool triggerOutEnable) {
    auto cfg = device->getMultiDeviceSyncConfig();
    cfg.syncMode = mode;
    cfg.depthDelayUs = 0;
    cfg.colorDelayUs = 0;
    cfg.trigger2ImageDelayUs = 0;
    cfg.triggerOutEnable = triggerOutEnable;
    cfg.triggerOutDelayUs = 0;
    cfg.framesPerTrigger = 1;
    device->setMultiDeviceSyncConfig(cfg);
}

void printAppliedSyncConfig(const std::shared_ptr<ob::Device> &device, const char *label) {
    const auto info = device->getDeviceInfo();
    const auto cfg = device->getMultiDeviceSyncConfig();
    std::cout << "  " << label
              << " sn=" << safeText(info->getSerialNumber())
              << " mode=" << syncModeName(cfg.syncMode)
              << " triggerOutEnable=" << (cfg.triggerOutEnable ? "true" : "false")
              << " framesPerTrigger=" << cfg.framesPerTrigger
              << '\n';
}

void printPresetList(const std::shared_ptr<ob::Device> &device) {
    try {
        auto list = device->getAvailablePresetList();
        if(!list || list->getCount() == 0) {
            std::cout << "    depth_presets: none\n";
            return;
        }
        std::cout << "    depth_presets:\n";
        for(uint32_t i = 0; i < list->getCount(); ++i) {
            std::cout << "      - " << safeText(list->getName(i)) << '\n';
        }
    }
    catch(const std::exception &e) {
        std::cout << "    depth_presets: unavailable (" << e.what() << ")\n";
    }
}

void printVideoProfile(const std::shared_ptr<ob::StreamProfile> &profile, uint32_t index) {
    try {
        auto video = profile->as<ob::VideoStreamProfile>();
        std::cout << "      [" << index << "] "
                  << video->getWidth() << "x" << video->getHeight()
                  << " @" << video->getFps()
                  << " format=" << ob::TypeHelper::convertOBFormatTypeToString(video->getFormat());
        try {
            const auto in = video->getIntrinsic();
            std::cout << " intrinsics="
                      << "fx:" << in.fx << ",fy:" << in.fy
                      << ",cx:" << in.cx << ",cy:" << in.cy;
        }
        catch(...) {
        }
        std::cout << '\n';
    }
    catch(const std::exception &e) {
        std::cout << "      [" << index << "] video profile unavailable (" << e.what() << ")\n";
    }
}

void printImuProfile(const std::shared_ptr<ob::StreamProfile> &profile, OBSensorType sensorType, uint32_t index) {
    try {
        if(sensorType == OB_SENSOR_ACCEL) {
            auto accel = profile->as<ob::AccelStreamProfile>();
            std::cout << "      [" << index << "] sample_rate="
                      << ob::TypeHelper::convertOBIMUSampleRateTypeToString(accel->getSampleRate()) << '\n';
        }
        else if(sensorType == OB_SENSOR_GYRO) {
            auto gyro = profile->as<ob::GyroStreamProfile>();
            std::cout << "      [" << index << "] sample_rate="
                      << ob::TypeHelper::convertOBIMUSampleRateTypeToString(gyro->getSampleRate()) << '\n';
        }
    }
    catch(const std::exception &e) {
        std::cout << "      [" << index << "] imu profile unavailable (" << e.what() << ")\n";
    }
}

void printSensorProfiles(const std::shared_ptr<ob::Sensor> &sensor) {
    const auto sensorType = sensor->getType();
    std::cout << "    sensor: " << ob::TypeHelper::convertOBSensorTypeToString(sensorType) << '\n';
    try {
        auto profiles = sensor->getStreamProfileList();
        if(!profiles || profiles->getCount() == 0) {
            std::cout << "      profiles: none\n";
            return;
        }
        for(uint32_t i = 0; i < profiles->getCount(); ++i) {
            auto profile = profiles->getProfile(i);
            if(ob::TypeHelper::isVideoSensorType(sensorType)) {
                printVideoProfile(profile, i);
            }
            else if(sensorType == OB_SENSOR_ACCEL || sensorType == OB_SENSOR_GYRO) {
                printImuProfile(profile, sensorType, i);
            }
            else {
                std::cout << "      [" << i << "] format="
                          << ob::TypeHelper::convertOBFormatTypeToString(profile->getFormat()) << '\n';
            }
        }
    }
    catch(const std::exception &e) {
        std::cout << "      profiles: unavailable (" << e.what() << ")\n";
    }
}

void printCameraParamProbe(const std::shared_ptr<ob::Pipeline> &pipe) {
    struct Candidate {
        uint32_t colorW;
        uint32_t colorH;
        uint32_t depthW;
        uint32_t depthH;
    };

    const std::vector<Candidate> candidates = {
        { 1920, 1080, 640, 400 },
        { 1280, 720, 640, 400 },
        { 1280, 800, 1280, 800 },
        { 848, 480, 848, 480 },
        { 640, 480, 640, 480 },
        { 640, 400, 640, 400 },
    };

    std::cout << "    rgb_depth_calibration_probe:\n";
    for(const auto &c: candidates) {
        try {
            auto p = pipe->getCameraParamWithProfile(c.colorW, c.colorH, c.depthW, c.depthH);
            std::cout << "      ok color=" << c.colorW << "x" << c.colorH
                      << " depth=" << c.depthW << "x" << c.depthH
                      << " rgb_fx=" << p.rgbIntrinsic.fx
                      << " depth_fx=" << p.depthIntrinsic.fx << '\n';
        }
        catch(const std::exception &e) {
            std::cout << "      no color=" << c.colorW << "x" << c.colorH
                      << " depth=" << c.depthW << "x" << c.depthH
                      << " (" << e.what() << ")\n";
        }
    }
}

std::shared_ptr<ob::VideoStreamProfile> findVideoProfile(const std::shared_ptr<ob::Pipeline> &pipe,
                                                         OBSensorType sensorType,
                                                         uint32_t width,
                                                         uint32_t height,
                                                         uint32_t fps,
                                                         OBFormat preferredFormat = OB_FORMAT_UNKNOWN) {
    auto list = pipe->getStreamProfileList(sensorType);
    if(!list) {
        return nullptr;
    }
    std::shared_ptr<ob::VideoStreamProfile> fallback;
    for(uint32_t i = 0; i < list->getCount(); ++i) {
        auto profile = list->getProfile(i)->as<ob::VideoStreamProfile>();
        if(!profile || profile->getWidth() != width || profile->getHeight() != height || profile->getFps() != fps) {
            continue;
        }
        if(preferredFormat == OB_FORMAT_UNKNOWN || profile->getFormat() == preferredFormat) {
            return profile;
        }
        if(!fallback) {
            fallback = profile;
        }
    }
    return fallback;
}

void printProfileSummary(const char *label, const std::shared_ptr<ob::VideoStreamProfile> &profile) {
    if(!profile) {
        return;
    }
    std::cout << "      " << label << "="
              << profile->getWidth() << "x" << profile->getHeight()
              << "@" << profile->getFps()
              << "/" << ob::TypeHelper::convertOBFormatTypeToString(profile->getFormat()) << '\n';
}

struct StreamStartTest {
    const char *name;
    bool        color;
    uint32_t    colorW;
    uint32_t    colorH;
    OBFormat    colorFormat;
    bool        depth;
    uint32_t    depthW;
    uint32_t    depthH;
    OBFormat    depthFormat;
    uint32_t    fps;
    bool        frameSync;
};

bool runStreamStartTest(const std::shared_ptr<ob::Device> &device, const StreamStartTest &test) {
    std::cout << "    stream_test: " << test.name << '\n';

    std::shared_ptr<ob::Pipeline> pipe;
    try {
        pipe = std::make_shared<ob::Pipeline>(device);
        auto config = std::make_shared<ob::Config>();

        std::shared_ptr<ob::VideoStreamProfile> colorProfile;
        std::shared_ptr<ob::VideoStreamProfile> depthProfile;
        if(test.color) {
            colorProfile = findVideoProfile(pipe, OB_SENSOR_COLOR, test.colorW, test.colorH, test.fps, test.colorFormat);
            if(!colorProfile) {
                std::cout << "      result: profile_missing color="
                          << test.colorW << "x" << test.colorH << "@" << test.fps << '\n';
                return false;
            }
            config->enableStream(colorProfile);
            printProfileSummary("color", colorProfile);
        }
        if(test.depth) {
            depthProfile = findVideoProfile(pipe, OB_SENSOR_DEPTH, test.depthW, test.depthH, test.fps, test.depthFormat);
            if(!depthProfile) {
                std::cout << "      result: profile_missing depth="
                          << test.depthW << "x" << test.depthH << "@" << test.fps << '\n';
                return false;
            }
            config->enableStream(depthProfile);
            printProfileSummary("depth", depthProfile);
        }

        if(test.color && test.depth) {
            config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
            if(test.frameSync) {
                try {
                    pipe->enableFrameSync();
                    std::cout << "      frame_sync: enabled\n";
                }
                catch(const std::exception &e) {
                    std::cout << "      frame_sync: enable_failed (" << e.what() << ")\n";
                }
            }
        }
        else {
            config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
        }

        pipe->start(config);
        bool gotColor = !test.color;
        bool gotDepth = !test.depth;
        for(int i = 0; i < 10 && (!gotColor || !gotDepth); ++i) {
            auto fs = pipe->waitForFrameset(1000);
            if(!fs) {
                continue;
            }
            if(test.color && fs->colorFrame()) {
                gotColor = true;
            }
            if(test.depth && fs->depthFrame()) {
                gotDepth = true;
            }
        }
        pipe->stop();

        const bool ok = gotColor && gotDepth;
        std::cout << "      result: " << (ok ? "ok" : "started_but_missing_frame")
                  << " color_frame=" << (gotColor ? "yes" : "no")
                  << " depth_frame=" << (gotDepth ? "yes" : "no") << '\n';
        return ok;
    }
    catch(const std::exception &e) {
        try {
            if(pipe) {
                pipe->stop();
            }
        }
        catch(...) {
        }
        std::cout << "      result: failed (" << e.what() << ")\n";
        return false;
    }
}

void runStreamStartTests(const std::shared_ptr<ob::Device> &device) {
    const std::vector<StreamStartTest> tests = {
        { "collection_default_mjpg_y16", true, 1280, 720, OB_FORMAT_MJPG, true, 640, 400, OB_FORMAT_Y16, 30, true },
        { "collection_default_mjpg_y14", true, 1280, 720, OB_FORMAT_MJPG, true, 640, 400, OB_FORMAT_Y14, 30, true },
        { "collection_default_rgb_y16", true, 1280, 720, OB_FORMAT_RGB, true, 640, 400, OB_FORMAT_Y16, 30, true },
        { "collection_default_rgb_y14", true, 1280, 720, OB_FORMAT_RGB, true, 640, 400, OB_FORMAT_Y14, 30, true },
        { "current_config_high_rgb_depth", true, 1920, 1080, OB_FORMAT_RGB, true, 640, 400, OB_FORMAT_Y16, 30, true },
        { "calibration_preview_color", true, 1280, 720, OB_FORMAT_RGB, false, 0, 0, OB_FORMAT_UNKNOWN, 30, false },
        { "point_cloud_source_depth", false, 0, 0, OB_FORMAT_UNKNOWN, true, 640, 400, OB_FORMAT_Y16, 30, false },
        { "gemini2_max_depth", false, 0, 0, OB_FORMAT_UNKNOWN, true, 1280, 800, OB_FORMAT_Y16, 30, false },
    };

    std::cout << "    stream_start_tests:\n";
    for(const auto &test: tests) {
        runStreamStartTest(device, test);
    }
}

struct SyncSample {
    uint64_t globalTs = 0;
    uint64_t localTs = 0;
    bool     hasColor = false;
    bool     hasDepth = false;
};

uint64_t frameTimestamp(const std::shared_ptr<ob::Frame> &frame, bool global) {
    if(!frame) {
        return 0;
    }
    try {
        return global ? frame->globalTimeStampUs() : frame->timeStampUs();
    }
    catch(...) {
        return 0;
    }
}

bool waitForDepthSample(const std::shared_ptr<ob::Pipeline> &pipe, SyncSample &out) {
    auto fs = pipe->waitForFrameset(1000);
    if(!fs) {
        return false;
    }
    auto depth = fs->depthFrame();
    auto color = fs->colorFrame();
    out.hasDepth = static_cast<bool>(depth);
    out.hasColor = static_cast<bool>(color);
    out.globalTs = frameTimestamp(depth, true);
    out.localTs = frameTimestamp(depth, false);
    return out.hasDepth;
}

void runMultiDeviceSyncTest(const std::shared_ptr<ob::DeviceList> &devices) {
    const uint32_t count = devices ? devices->getCount() : 0;
    std::cout << "\nMulti-device sync test\n";
    if(count < 2) {
        std::cout << "  result: need at least 2 connected Orbbec devices\n";
        return;
    }

    auto primaryDevice = devices->getDevice(0);
    auto secondaryDevice = devices->getDevice(1);
    try {
        applySyncConfig(primaryDevice, OB_MULTI_DEVICE_SYNC_MODE_PRIMARY, true);
        applySyncConfig(secondaryDevice, OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED, false);
        printAppliedSyncConfig(primaryDevice, "primary");
        printAppliedSyncConfig(secondaryDevice, "secondary");
    }
    catch(const std::exception &e) {
        std::cout << "  result: failed_to_apply_sync_config (" << e.what() << ")\n";
        return;
    }

    std::shared_ptr<ob::Pipeline> primaryPipe;
    std::shared_ptr<ob::Pipeline> secondaryPipe;
    try {
        primaryPipe = std::make_shared<ob::Pipeline>(primaryDevice);
        secondaryPipe = std::make_shared<ob::Pipeline>(secondaryDevice);

        auto makeConfig = [](const std::shared_ptr<ob::Pipeline> &pipe) {
            auto cfg = std::make_shared<ob::Config>();
            auto color = findVideoProfile(pipe, OB_SENSOR_COLOR, 1280, 720, 30, OB_FORMAT_MJPG);
            auto depth = findVideoProfile(pipe, OB_SENSOR_DEPTH, 640, 400, 30, OB_FORMAT_Y14);
            if(!color || !depth) {
                throw std::runtime_error("missing 1280x720@30 color or 640x400@30 depth profile");
            }
            cfg->enableStream(color);
            cfg->enableStream(depth);
            printProfileSummary("sync_color", color);
            printProfileSummary("sync_depth", depth);
            cfg->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
            return cfg;
        };

        primaryPipe->enableFrameSync();
        secondaryPipe->enableFrameSync();

        secondaryPipe->start(makeConfig(secondaryPipe));
        primaryPipe->start(makeConfig(primaryPipe));

        std::vector<int64_t> globalDiffs;
        std::vector<int64_t> localDiffs;
        int primaryFrames = 0;
        int secondaryFrames = 0;
        int primaryGlobal = 0;
        int secondaryGlobal = 0;
        int paired = 0;
        int bothColor = 0;

        for(int i = 0; i < 60; ++i) {
            SyncSample a;
            SyncSample b;
            if(waitForDepthSample(primaryPipe, a)) {
                primaryFrames++;
                if(a.globalTs != 0) {
                    primaryGlobal++;
                }
            }
            if(waitForDepthSample(secondaryPipe, b)) {
                secondaryFrames++;
                if(b.globalTs != 0) {
                    secondaryGlobal++;
                }
            }
            if(a.hasDepth && b.hasDepth) {
                paired++;
                if(a.hasColor && b.hasColor) {
                    bothColor++;
                }
                if(a.globalTs != 0 && b.globalTs != 0) {
                    globalDiffs.push_back(static_cast<int64_t>(a.globalTs) - static_cast<int64_t>(b.globalTs));
                }
                if(a.localTs != 0 && b.localTs != 0) {
                    localDiffs.push_back(static_cast<int64_t>(a.localTs) - static_cast<int64_t>(b.localTs));
                }
            }
        }

        primaryPipe->stop();
        secondaryPipe->stop();

        auto printStats = [](const char *label, const std::vector<int64_t> &values) {
            if(values.empty()) {
                std::cout << "  " << label << ": none\n";
                return;
            }
            int64_t minAbs = std::numeric_limits<int64_t>::max();
            int64_t maxAbs = 0;
            long double sumAbs = 0.0;
            for(const auto v: values) {
                const int64_t av = v < 0 ? -v : v;
                minAbs = std::min(minAbs, av);
                maxAbs = std::max(maxAbs, av);
                sumAbs += static_cast<long double>(av);
            }
            std::cout << "  " << label
                      << ": pairs=" << values.size()
                      << " min_abs_us=" << minAbs
                      << " avg_abs_us=" << static_cast<double>(sumAbs / values.size())
                      << " max_abs_us=" << maxAbs
                      << '\n';
        };

        std::cout << "  frames primary=" << primaryFrames
                  << " secondary=" << secondaryFrames
                  << " paired=" << paired
                  << " both_color=" << bothColor << '\n';
        std::cout << "  global_ts_nonzero primary=" << primaryGlobal
                  << " secondary=" << secondaryGlobal << '\n';
        printStats("global_depth_ts_diff", globalDiffs);
        printStats("local_depth_ts_diff", localDiffs);
    }
    catch(const std::exception &e) {
        try {
            if(primaryPipe) {
                primaryPipe->stop();
            }
        }
        catch(...) {
        }
        try {
            if(secondaryPipe) {
                secondaryPipe->stop();
            }
        }
        catch(...) {
        }
        std::cout << "  result: failed (" << e.what() << ")\n";
    }
}

void printDevice(const std::shared_ptr<ob::Device> &device, uint32_t index, bool runStreamTests) {
    auto info = device->getDeviceInfo();
    std::cout << "\n[" << index << "] device\n";
    std::cout << "    name: " << safeText(info->getName()) << '\n';
    std::cout << "    serial_number: " << safeText(info->getSerialNumber()) << '\n';
    std::cout << "    uid: " << safeText(info->getUid()) << '\n';
    std::cout << "    vid: " << hex4(info->getVid()) << '\n';
    std::cout << "    pid: " << hex4(info->getPid()) << '\n';
    std::cout << "    connection: " << safeText(info->getConnectionType()) << '\n';
    std::cout << "    firmware: " << safeText(info->getFirmwareVersion()) << '\n';
    std::cout << "    hardware: " << safeText(info->getHardwareVersion()) << '\n';
    std::cout << "    min_supported_sdk: " << safeText(info->getSupportedMinSdkVersion()) << '\n';
    std::cout << "    asic: " << safeText(info->getAsicName()) << '\n';

    printSupportedSyncModes(device);
    printCurrentSyncConfig(device);
    printPresetList(device);

    try {
        auto sensorList = device->getSensorList();
        if(!sensorList || sensorList->getCount() == 0) {
            std::cout << "    sensors: none\n";
        }
        else {
            std::cout << "    sensors:\n";
            for(uint32_t i = 0; i < sensorList->getCount(); ++i) {
                printSensorProfiles(sensorList->getSensor(i));
            }
        }
    }
    catch(const std::exception &e) {
        std::cout << "    sensors: unavailable (" << e.what() << ")\n";
    }

    try {
        auto pipe = std::make_shared<ob::Pipeline>(device);
        printCameraParamProbe(pipe);
    }
    catch(const std::exception &e) {
        std::cout << "    rgb_depth_calibration_probe: unavailable (" << e.what() << ")\n";
    }

    if(runStreamTests) {
        runStreamStartTests(device);
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        bool runStreamTests = false;
        bool runSyncTest = false;
        for(int i = 1; i < argc; ++i) {
            const std::string arg = argv[i] ? argv[i] : "";
            if(arg == "--stream-test") {
                runStreamTests = true;
            }
            else if(arg == "--sync-test") {
                runSyncTest = true;
            }
            else if(arg == "-h" || arg == "--help") {
                std::cout << "Usage: orbbec_probe [--stream-test] [--sync-test]\n"
                          << "  default: print device metadata, sync capability, calibration probes, and profiles\n"
                          << "  --stream-test: also start common RGB/depth stream combinations and wait for frames\n"
                          << "  --sync-test: configure device 0 as primary and device 1 as secondary-synced, then compare timestamps\n";
                return 0;
            }
            else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return 2;
            }
        }

        std::cout << "Orbbec SDK version: " << ob::Version::getMajor() << '.'
                  << ob::Version::getMinor() << '.'
                  << ob::Version::getPatch();
        const char *stage = ob::Version::getStageVersion();
        if(stage && *stage) {
            std::cout << " (" << stage << ")";
        }
        std::cout << "\n";

        ob::Context context;
        auto devices = context.queryDeviceList();
        const uint32_t count = devices ? devices->getCount() : 0;
        std::cout << "Connected Orbbec devices: " << count << "\n";
        if(count == 0) {
            return 2;
        }

        for(uint32_t i = 0; i < count; ++i) {
            printDevice(devices->getDevice(i), i, runStreamTests);
        }
        if(runSyncTest) {
            runMultiDeviceSyncTest(devices);
        }
        return 0;
    }
    catch(const ob::Error &e) {
        std::cerr << "Orbbec error\n"
                  << "  function: " << e.getName() << "\n"
                  << "  args: " << e.getArgs() << "\n"
                  << "  message: " << e.getMessage() << "\n"
                  << "  type: " << e.getExceptionType() << "\n";
        return 1;
    }
    catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
