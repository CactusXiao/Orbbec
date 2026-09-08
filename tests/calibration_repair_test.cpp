#include "calibration_repair.hpp"
#include <cassert>
#include <iostream>

using namespace sync_app::calibration_repair;

template<class F> void rejects(F operation) {
    bool threw = false;
    try { operation(); } catch(const std::exception &) { threw = true; }
    assert(threw);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    assert(cameraIds() == (std::vector<std::string>{"00", "01", "02", "03", "04", "05", "06"}));
    const std::string source = R"({
      "00":{"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation":[100,200,300],"note":"keep"},
      "01":{"rotation":[[0,-1,0],[1,0,0],[0,0,1]],"translation":[10,20,30],
            "rgb_to_depth":{"c2d_extrinsic":{"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation":[999,888,777]}}},
      "02":{"serial":"untouched","precise":0.12345678901234567},
      "03":{"other":true},"04":{"other":4},"05":{"other":5},
      "06":{"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation":[-1,-2,-3],"rgb_to_depth":{"keep":"factory"}},
      "metadata":{"units":"mm"}
    })";
    const Pose edge{cv::Matx33d(1,0,0,0,0,-1,0,1,0), cv::Vec3d(1,2,3)};
    for(const auto &target : {"00", "06"}) {
        const auto output = patch(source, target, "01", edge);
        auto before = parse(source);
        auto after = parse(output);
        const Pose result = readPose(cJSON_GetObjectItemCaseSensitive(after.get(), target));
        // Expected composition uses the reference RGB pose, not its depth pose.
        assert(cv::norm(cv::Mat(result.R - cv::Matx33d(0,-1,0,0,0,-1,1,0,0))) < 1e-12);
        assert(cv::norm(result.t - cv::Vec3d(11,-28,23)) < 1e-12);
        for(cJSON *item = before->child; item; item = item->next) {
            if(std::string(item->string) == target) continue;
            assert(cJSON_Compare(item, cJSON_GetObjectItemCaseSensitive(after.get(), item->string), true));
        }
        if(std::string(target) == "06") {
            assert(cJSON_Compare(cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(before.get(), target), "rgb_to_depth"),
                                 cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(after.get(), target), "rgb_to_depth"), true));
        }
    }
    rejects([&] { patch(source, "01", "01", edge); });
    rejects([&] { patch(source, "07", "01", edge); });
    rejects([&] { patch(source, "00", "02", edge); });
    rejects([&] { patch("{}", "00", "01", edge); });
    rejects([&] { patch("broken", "00", "01", edge); });
    rejects([&] { patch(source, "00", "01", Pose{cv::Matx33d::zeros(), {0,0,0}}); });
    auto missing = parse(source);
    cJSON_DeleteItemFromObjectCaseSensitive(missing.get(), "06");
    char *printed = cJSON_Print(missing.get());
    auto factory = parse(R"({"d2c_extrinsic":{"test":true}})");
    auto added = parse(patch(printed, "06", "01", edge, factory.get()));
    cJSON_free(printed);
    assert(cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(added.get(), "06"), "rgb_to_depth"));
    const std::filesystem::path path = std::filesystem::path(argv[1]) / "extrinsic.json";
    { std::ofstream file(path); file << source; }
    const auto output = patch(source, "06", "01", edge);
    save(path, source, output);
    assert(readFile(path) == output);
    assert(readFile(path.string() + ".before_single_camera.bak") == source);
    rejects([&] { save(path, source, "{}"); });
    assert(readFile(path) == output);
    assert(!std::filesystem::exists(path.string() + ".single_camera.tmp"));
    std::cout << "PASS: seven cameras, RGB composition, repair 00/06, preserve others, validation, backup and stale-file protection\n";
}
