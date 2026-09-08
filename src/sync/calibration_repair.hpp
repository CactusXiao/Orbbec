#pragma once

#include "utils/cJSON.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sync_app::calibration_repair {

inline std::vector<std::string> cameraIds() {
    return {"00", "01", "02", "03", "04", "05", "06"};
}

struct Pose {
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0, 0, 0};
};
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

inline Json parse(const std::string &content) {
    Json root(cJSON_Parse(content.c_str()), cJSON_Delete);
    if(!root || !cJSON_IsObject(root.get())) {
        throw std::runtime_error("Invalid extrinsic JSON object");
    }
    return root;
}

inline Pose readPose(cJSON *camera) {
    auto *rotation = cJSON_GetObjectItemCaseSensitive(camera, "rotation");
    auto *translation = cJSON_GetObjectItemCaseSensitive(camera, "translation");
    if(!cJSON_IsArray(rotation) || cJSON_GetArraySize(rotation) != 3 ||
       !cJSON_IsArray(translation) || cJSON_GetArraySize(translation) != 3) {
        throw std::runtime_error("Missing or invalid RGB extrinsic");
    }
    Pose pose;
    for(int y = 0; y < 3; ++y) {
        auto *row = cJSON_GetArrayItem(rotation, y);
        auto *t = cJSON_GetArrayItem(translation, y);
        if(!cJSON_IsArray(row) || cJSON_GetArraySize(row) != 3 || !cJSON_IsNumber(t) || !std::isfinite(t->valuedouble)) {
            throw std::runtime_error("Invalid extrinsic values");
        }
        pose.t[y] = t->valuedouble;
        for(int x = 0; x < 3; ++x) {
            auto *value = cJSON_GetArrayItem(row, x);
            if(!cJSON_IsNumber(value) || !std::isfinite(value->valuedouble)) {
                throw std::runtime_error("Invalid rotation values");
            }
            pose.R(y, x) = value->valuedouble;
        }
    }
    if(cv::norm(cv::Mat(pose.R * pose.R.t() - cv::Matx33d::eye())) > 0.01 ||
       std::abs(cv::determinant(pose.R) - 1.0) > 0.01) {
        throw std::runtime_error("Extrinsic rotation is not rigid");
    }
    return pose;
}

inline void validateSelection(const std::string &target, const std::string &reference) {
    const auto ids = cameraIds();
    const auto valid = [&](const std::string &id) {
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    };
    if(target == reference || !valid(target) || !valid(reference)) {
        throw std::runtime_error("Select distinct target/reference cameras in 00-06");
    }
}

inline Pose referencePose(const std::string &content, const std::string &target, const std::string &reference) {
    validateSelection(target, reference);
    auto root = parse(content);
    return readPose(cJSON_GetObjectItemCaseSensitive(root.get(), reference.c_str()));
}

// Both transforms use RGB coordinates and the same length unit as stereoCalibrate.
inline std::string patch(const std::string &content, const std::string &target,
                         const std::string &reference, const Pose &referenceToTarget, cJSON *rgbToDepth = nullptr) {
    const Pose worldToReference = referencePose(content, target, reference);
    auto root = parse(content);
    auto *camera = cJSON_GetObjectItemCaseSensitive(root.get(), target.c_str());
    if(!camera) {
        camera = cJSON_CreateObject();
        cJSON_AddItemToObject(root.get(), target.c_str(), camera);
    }
    if(!cJSON_IsObject(camera)) {
        throw std::runtime_error("Target camera entry must be an object");
    }
    const Pose result{referenceToTarget.R * worldToReference.R,
                      referenceToTarget.R * worldToReference.t + referenceToTarget.t};
    auto *rotation = cJSON_CreateArray();
    auto *translation = cJSON_CreateArray();
    for(int y = 0; y < 3; ++y) {
        auto *row = cJSON_CreateArray();
        for(int x = 0; x < 3; ++x) cJSON_AddItemToArray(row, cJSON_CreateNumber(result.R(y, x)));
        cJSON_AddItemToArray(rotation, row);
        cJSON_AddItemToArray(translation, cJSON_CreateNumber(result.t[y]));
    }
    cJSON_DeleteItemFromObjectCaseSensitive(camera, "rotation");
    cJSON_DeleteItemFromObjectCaseSensitive(camera, "translation");
    cJSON_AddItemToObject(camera, "rotation", rotation);
    cJSON_AddItemToObject(camera, "translation", translation);
    if(rgbToDepth && !cJSON_GetObjectItemCaseSensitive(camera, "rgb_to_depth")) {
        cJSON_AddItemToObject(camera, "rgb_to_depth", cJSON_Duplicate(rgbToDepth, true));
    }
    readPose(camera); // Reject non-finite or non-rigid calibration results before saving.
    char *printed = cJSON_Print(root.get());
    if(!printed) throw std::runtime_error("Cannot serialize repaired extrinsics");
    std::string output(printed);
    cJSON_free(printed);
    return output;
}

inline std::string readFile(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file) throw std::runtime_error("Cannot read extrinsic file: " + path.string());
    std::ostringstream content;
    content << file.rdbuf();
    if(file.bad()) throw std::runtime_error("Failed reading extrinsic file");
    return content.str();
}

inline void save(const std::filesystem::path &path, const std::string &snapshot, const std::string &updated) {
    if(readFile(path) != snapshot) {
        throw std::runtime_error("Extrinsic file changed during calibration; restart to use the latest file");
    }
    const auto backup = path.string() + ".before_single_camera.bak";
    const auto temporary = path.string() + ".single_camera.tmp";
    std::filesystem::copy_file(path, backup, std::filesystem::copy_options::overwrite_existing);
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file << updated;
        file.close();
        if(!file) throw std::runtime_error("Failed writing repaired extrinsics");
        if(readFile(path) != snapshot) throw std::runtime_error("Extrinsic file changed before save; restart calibration");
        std::filesystem::rename(temporary, path);
    }
    catch(...) {
        std::filesystem::remove(temporary);
        throw;
    }
}

} // namespace sync_app::calibration_repair
