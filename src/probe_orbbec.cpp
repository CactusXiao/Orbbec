#include <libobsensor/ObSensor.hpp>

#include <iomanip>
#include <iostream>
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
                                                         uint32_t fps) {
    auto list = pipe->getStreamProfileList(sensorType);
    if(!list) {
        return nullptr;
    }
    for(uint32_t i = 0; i < list->getCount(); ++i) {
        auto profile = list->getProfile(i)->as<ob::VideoStreamProfile>();
        if(profile && profile->getWidth() == width && profile->getHeight() == height && profile->getFps() == fps) {
            return profile;
        }
    }
    return nullptr;
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
    bool        depth;
    uint32_t    depthW;
    uint32_t    depthH;
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
            colorProfile = findVideoProfile(pipe, OB_SENSOR_COLOR, test.colorW, test.colorH, test.fps);
            if(!colorProfile) {
                std::cout << "      result: profile_missing color="
                          << test.colorW << "x" << test.colorH << "@" << test.fps << '\n';
                return false;
            }
            config->enableStream(colorProfile);
            printProfileSummary("color", colorProfile);
        }
        if(test.depth) {
            depthProfile = findVideoProfile(pipe, OB_SENSOR_DEPTH, test.depthW, test.depthH, test.fps);
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
        { "collection_ui_default_rgb_depth", true, 848, 480, true, 848, 480, 30, true },
        { "current_config_rgb_depth", true, 1280, 720, true, 640, 400, 30, true },
        { "current_config_high_rgb_depth", true, 1920, 1080, true, 640, 400, 30, true },
        { "calibration_preview_color", true, 1280, 800, false, 0, 0, 30, false },
        { "point_cloud_source_depth", false, 0, 0, true, 640, 400, 30, false },
        { "gemini2_max_depth", false, 0, 0, true, 1280, 800, 30, false },
    };

    std::cout << "    stream_start_tests:\n";
    for(const auto &test: tests) {
        runStreamStartTest(device, test);
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
        for(int i = 1; i < argc; ++i) {
            const std::string arg = argv[i] ? argv[i] : "";
            if(arg == "--stream-test") {
                runStreamTests = true;
            }
            else if(arg == "-h" || arg == "--help") {
                std::cout << "Usage: orbbec_probe [--stream-test]\n"
                          << "  default: print device metadata, sync capability, calibration probes, and profiles\n"
                          << "  --stream-test: also start common RGB/depth stream combinations and wait for frames\n";
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
