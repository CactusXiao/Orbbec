#include "utils/cJSON.h"

#include <libobsensor/ObSensor.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string readStdin() {
    std::ostringstream oss;
    oss << std::cin.rdbuf();
    return oss.str();
}

bool jsonNumber(cJSON *obj, const char *key, float &out) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(!item || !cJSON_IsNumber(item)) {
        return false;
    }
    out = static_cast<float>(item->valuedouble);
    return true;
}

bool jsonImageSize(cJSON *obj, const char *key, int16_t &out) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(!item || !cJSON_IsNumber(item)) {
        return false;
    }
    const int value = item->valueint;
    if(value <= 0 || value > 32767) {
        return false;
    }
    out = static_cast<int16_t>(value);
    return true;
}

bool jsonArrayNumber(cJSON *array, int index, float &out) {
    cJSON *item = cJSON_GetArrayItem(array, index);
    if(!item || !cJSON_IsNumber(item)) {
        return false;
    }
    out = static_cast<float>(item->valuedouble);
    return true;
}

float optionalNumber(cJSON *obj, const char *key) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return (item && cJSON_IsNumber(item)) ? static_cast<float>(item->valuedouble) : 0.0f;
}

bool parseIntrinsic(cJSON *obj, OBCameraIntrinsic &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    return jsonNumber(obj, "fx", out.fx)
        && jsonNumber(obj, "fy", out.fy)
        && jsonNumber(obj, "cx", out.cx)
        && jsonNumber(obj, "cy", out.cy)
        && jsonImageSize(obj, "width", out.width)
        && jsonImageSize(obj, "height", out.height);
}

bool parseDistortion(cJSON *obj, OBCameraDistortion &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    out.k1 = optionalNumber(obj, "k1");
    out.k2 = optionalNumber(obj, "k2");
    out.k3 = optionalNumber(obj, "k3");
    out.k4 = optionalNumber(obj, "k4");
    out.k5 = optionalNumber(obj, "k5");
    out.k6 = optionalNumber(obj, "k6");
    out.p1 = optionalNumber(obj, "p1");
    out.p2 = optionalNumber(obj, "p2");
    cJSON *model = cJSON_GetObjectItemCaseSensitive(obj, "model");
    if(model && cJSON_IsNumber(model)) {
        out.model = static_cast<decltype(out.model)>(model->valueint);
    }
    return true;
}

OBExtrinsic identityExtrinsic() {
    OBExtrinsic ex{};
    ex.rot[0] = 1.0f;
    ex.rot[4] = 1.0f;
    ex.rot[8] = 1.0f;
    return ex;
}

cJSON *makeError(const std::string &message) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "ok", false);
    cJSON_AddStringToObject(root, "error", message.c_str());
    return root;
}

}  // namespace

int main() {
    const std::string input = readStdin();
    cJSON *root = cJSON_Parse(input.c_str());
    if(!root || !cJSON_IsObject(root)) {
        cJSON *err = makeError("invalid input json");
        char *text = cJSON_PrintUnformatted(err);
        if(text) {
            std::cout << text << std::endl;
            cJSON_free(text);
        }
        cJSON_Delete(err);
        if(root) {
            cJSON_Delete(root);
        }
        return 1;
    }

    cJSON *camerasObj = cJSON_GetObjectItemCaseSensitive(root, "cameras");
    cJSON *requests = cJSON_GetObjectItemCaseSensitive(root, "requests");
    if(!camerasObj || !cJSON_IsObject(camerasObj) || !requests || !cJSON_IsArray(requests)) {
        cJSON *err = makeError("missing cameras or requests");
        char *text = cJSON_PrintUnformatted(err);
        if(text) {
            std::cout << text << std::endl;
            cJSON_free(text);
        }
        cJSON_Delete(err);
        cJSON_Delete(root);
        return 1;
    }

    cJSON *out = cJSON_CreateObject();
    cJSON_AddBoolToObject(out, "ok", true);
    cJSON *outRequests = cJSON_CreateArray();
    const OBExtrinsic identity = identityExtrinsic();

    const int requestCount = cJSON_GetArraySize(requests);
    for(int i = 0; i < requestCount; ++i) {
        cJSON *request = cJSON_GetArrayItem(requests, i);
        cJSON *outRequest = cJSON_CreateObject();
        cJSON_AddBoolToObject(outRequest, "ok", false);

        cJSON *camIdObj = request ? cJSON_GetObjectItemCaseSensitive(request, "camera_id") : nullptr;
        cJSON *points = request ? cJSON_GetObjectItemCaseSensitive(request, "points") : nullptr;
        if(!camIdObj || !cJSON_IsString(camIdObj) || !camIdObj->valuestring || !points || !cJSON_IsArray(points)) {
            cJSON_AddStringToObject(outRequest, "error", "invalid request");
            cJSON_AddItemToArray(outRequests, outRequest);
            continue;
        }

        cJSON *camObj = cJSON_GetObjectItemCaseSensitive(camerasObj, camIdObj->valuestring);
        cJSON *intrObj = camObj ? cJSON_GetObjectItemCaseSensitive(camObj, "intrinsic") : nullptr;
        cJSON *distObj = camObj ? cJSON_GetObjectItemCaseSensitive(camObj, "distortion") : nullptr;
        OBCameraIntrinsic intrinsic{};
        OBCameraDistortion distortion{};
        if(!parseIntrinsic(intrObj, intrinsic) || !parseDistortion(distObj, distortion)) {
            cJSON_AddStringToObject(outRequest, "error", "invalid camera parameters");
            cJSON_AddItemToArray(outRequests, outRequest);
            continue;
        }

        cJSON *outPoints = cJSON_CreateArray();
        bool allOk = true;
        const int pointCount = cJSON_GetArraySize(points);
        for(int j = 0; j < pointCount; ++j) {
            cJSON *point = cJSON_GetArrayItem(points, j);
            cJSON *outPoint = cJSON_CreateObject();
            if(!point || !cJSON_IsArray(point) || cJSON_GetArraySize(point) != 3) {
                cJSON_AddBoolToObject(outPoint, "ok", false);
                cJSON_AddStringToObject(outPoint, "error", "invalid point");
                cJSON_AddItemToArray(outPoints, outPoint);
                allOk = false;
                continue;
            }

            OBPoint3f p3{};
            if(!jsonArrayNumber(point, 0, p3.x) || !jsonArrayNumber(point, 1, p3.y) || !jsonArrayNumber(point, 2, p3.z)) {
                cJSON_AddBoolToObject(outPoint, "ok", false);
                cJSON_AddStringToObject(outPoint, "error", "invalid point value");
                cJSON_AddItemToArray(outPoints, outPoint);
                allOk = false;
                continue;
            }
            OBPoint2f p2{};
            bool ok = false;
            try {
                ok = ob::CoordinateTransformHelper::transformation3dto2d(p3, intrinsic, distortion, identity, &p2);
            }
            catch(const std::exception &ex) {
                cJSON_AddStringToObject(outPoint, "error", ex.what());
            }
            catch(...) {
                cJSON_AddStringToObject(outPoint, "error", "projection threw");
            }

            cJSON_AddBoolToObject(outPoint, "ok", ok);
            if(ok) {
                cJSON_AddNumberToObject(outPoint, "x", p2.x);
                cJSON_AddNumberToObject(outPoint, "y", p2.y);
            }
            else {
                allOk = false;
            }
            cJSON_AddItemToArray(outPoints, outPoint);
        }

        cJSON_ReplaceItemInObject(outRequest, "ok", cJSON_CreateBool(allOk));
        cJSON_AddItemToObject(outRequest, "points", outPoints);
        cJSON_AddItemToArray(outRequests, outRequest);
    }

    cJSON_AddItemToObject(out, "requests", outRequests);
    char *text = cJSON_PrintUnformatted(out);
    if(text) {
        std::cout << text << std::endl;
        cJSON_free(text);
    }
    cJSON_Delete(out);
    cJSON_Delete(root);
    return 0;
}
