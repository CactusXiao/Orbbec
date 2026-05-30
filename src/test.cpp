#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <libobsensor/ObSensor.hpp>

namespace fs = std::filesystem;

// 存储包：包含深度图数据及其路径
struct SavePackage {
    cv::Mat image;
    std::string fullPath;
};

std::queue<SavePackage> g_saveQueue;
std::mutex g_queueMutex;
std::condition_variable g_cv;
std::atomic<bool> g_keepRunning(true);

// 消费者线程：统一处理磁盘写入
void saveWorker() {
    while (g_keepRunning || !g_saveQueue.empty()) {
        SavePackage pkg;
        {
            std::unique_lock<std::mutex> lock(g_queueMutex);
            g_cv.wait(lock, [] { return !g_saveQueue.empty() || !g_keepRunning; });
            if (g_saveQueue.empty()) continue;
            pkg = std::move(g_saveQueue.front());
            g_saveQueue.pop();
        }
        if (!pkg.image.empty()) {
            // 保存为无损 PNG 以保留 16bit 深度精度
            cv::imwrite(pkg.fullPath, pkg.image);
        }
    }
}

class CameraHandler {
public:
    CameraHandler(std::shared_ptr<ob::Device> dev, fs::path baseDir) : device(dev) {
        sn = device->getDeviceInfo()->serialNumber();
        saveDir = baseDir / sn;
        if (!fs::exists(saveDir)) fs::create_directory(saveDir);

        pipe = std::make_unique<ob::Pipeline>(device);
        auto config = std::make_shared<ob::Config>();

        // 1. 配置深度流
        try {
            auto depthProfiles = pipe->getStreamProfileList(OB_SENSOR_DEPTH);
            auto profile = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
            config->enableStream(profile);
            std::cout << "[Camera " << sn << "] Depth stream enabled." << std::endl;
        } catch (const ob::Error &e) {
            std::cerr << "Enable depth failed for " << sn << ": " << e.getMessage() << std::endl;
            return;
        }

        // 设置间隔：100ms = 10fps
        const int intervalMs = 100;

        pipe->start(config, [this, intervalMs](std::shared_ptr<ob::FrameSet> frameset) {
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastSaveTime).count();

            if (elapsed >= intervalMs) {
                lastSaveTime = currentTime;
                auto depthFrame = frameset->depthFrame();
                if (depthFrame) {
                    int idx = frameIndex.fetch_add(1);
                    
                    // 2. 深度数据处理：16UC1 (16位无符号单通道)
                    cv::Mat depthMat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
                    
                    SavePackage pkg;
                    pkg.image = depthMat.clone(); // 必须 clone 拷贝内存
                    pkg.fullPath = (saveDir / ("depth_" + std::to_string(idx) + ".png")).string();

                    {
                        std::lock_guard<std::mutex> lock(g_queueMutex);
                        if(g_saveQueue.size() < 300) { // 三相机共用队列，上限稍微调大一点
                            g_saveQueue.push(std::move(pkg));
                        }
                    }
                    g_cv.notify_one();
                }
            }
        });
    }

    void stop() {
        if (pipe) pipe->stop();
    }

private:
    std::shared_ptr<ob::Device> device;
    std::unique_ptr<ob::Pipeline> pipe;
    fs::path saveDir;
    std::string sn;
    std::atomic<int> frameIndex{0};
    std::chrono::steady_clock::time_point lastSaveTime = std::chrono::steady_clock::now();
};

int main() {
    try {
        fs::path baseDir = fs::current_path() / "multi_depth_data";
        if (!fs::exists(baseDir)) fs::create_directory(baseDir);

        ob::Context ctx;
        auto devList = ctx.queryDeviceList();
        int actualDevCount = devList->deviceCount(); // 自动获取实际连接的相机数

        if (actualDevCount == 0) {
            std::cerr << "No devices found!" << std::endl;
            return -1;
        }

        std::cout << "Found " << actualDevCount << " devices. Initializing..." << std::endl;

        std::thread worker(saveWorker);

        std::vector<std::unique_ptr<CameraHandler>> handlers;
        for (int i = 0; i < actualDevCount; i++) {
            auto dev = devList->getDevice(i);
            handlers.push_back(std::make_unique<CameraHandler>(dev, baseDir));
        }

        std::cout << "Recording... Capturing 10 frames per second. Press Ctrl+C to stop (or wait 10s)." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));

        std::cout << "Shutting down..." << std::endl;
        for (auto& h : handlers) {
            h->stop();
        }

        g_keepRunning = false;
        g_cv.notify_one();
        if (worker.joinable()) worker.join();

        std::cout << "Successfully saved depth frames to " << baseDir << std::endl;

    } catch (const ob::Error &e) {
        std::cerr << "SDK Global Error: " << e.getMessage() << std::endl;
    }

    return 0;
}