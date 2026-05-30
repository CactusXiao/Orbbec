#include "viewer.hpp"

#include <cstdio>
#include <cstdlib>
#include <list>
#include <regex>

#if defined(__has_include)
#if __has_include(<opencv2/highgui/clipboard.hpp>)
#include <opencv2/highgui/clipboard.hpp>
#define SYNC_VIEWER_HAS_OPENCV_CLIPBOARD 1
#endif
#endif

namespace sync_app {

struct CvMouseState {
    int x = 0;
    int y = 0;
    bool clicked = false;
    int clickX = 0;
    int clickY = 0;
    int wheelDelta = 0;
};

struct PointCloudMouseContext;

static void mouseCallbackPointCloud(int event, int x, int y, int flags, void *userdata);

struct MainMouseContext {
    CvMouseState *ui = nullptr;
    PointCloudMouseContext *pc = nullptr;
};

static bool isCtrlModifierKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return key == static_cast<int>(0xFFE3) || key == static_cast<int>(0xFFE4) || lo16 == static_cast<int>(0xFFE3)
           || lo16 == static_cast<int>(0xFFE4) || lo8 == 0xE3 || lo8 == 0xE4 || key == 227 || key == 228;
}

static bool isCtrlReleaseKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == 0x007F || lo8 == 0x7F || key == 127;
}

static bool isCtrlZoomInKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == '+' || lo16 == '=' || lo16 == static_cast<int>(0xFFAB) || lo8 == '+' || lo8 == '=' || lo8 == 0xAB;
}

static bool isCtrlZoomOutKeyEvent(int key) {
    const int lo16 = key & 0xFFFF;
    const int lo8  = key & 0xFF;
    return lo16 == '-' || lo16 == '_' || lo16 == static_cast<int>(0xFFAD) || lo8 == '-' || lo8 == '_' || lo8 == 0xAD;
}

static bool g_viewerCtrlShortcutListening = false;

static void mouseCallbackMain(int event, int x, int y, int flags, void *userdata) {
    auto *ctx = reinterpret_cast<MainMouseContext *>(userdata);
    if(!ctx) {
        return;
    }
    if(ctx->ui) {
        ctx->ui->x = x;
        ctx->ui->y = y;
        if(event == cv::EVENT_LBUTTONDOWN) {
            ctx->ui->clicked = true;
            ctx->ui->clickX = x;
            ctx->ui->clickY = y;
        }
        else if(event == cv::EVENT_MOUSEWHEEL || event == cv::EVENT_MOUSEHWHEEL) {
            ctx->ui->wheelDelta += cv::getMouseWheelDelta(flags);
        }
    }
    if(ctx->pc) {
        mouseCallbackPointCloud(event, x, y, flags, ctx->pc);
    }
}

static bool uiButton(cv::Mat &img, const cv::Rect &r, const std::string &label, CvMouseState &ms) {
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar bg = hover ? cv::Scalar(60, 60, 60) : cv::Scalar(40, 40, 40);
    cv::rectangle(img, r, bg, cv::FILLED);
    cv::rectangle(img, r, cv::Scalar(120, 120, 120), 1);
    cv::putText(img, label, cv::Point(r.x + 12, r.y + r.height / 2 + 6), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static bool uiButtonDisabled(cv::Mat &img, const cv::Rect &r, const std::string &label) {
    cv::rectangle(img, r, cv::Scalar(30, 30, 30), cv::FILLED);
    cv::rectangle(img, r, cv::Scalar(80, 80, 80), 1);
    cv::putText(img, label, cv::Point(r.x + 12, r.y + r.height / 2 + 6), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(140, 140, 140), 1, cv::LINE_AA);
    return false;
}

static bool uiCheckbox(cv::Mat &img, const cv::Rect &r, bool checked, const std::string &label, bool enabled, CvMouseState &ms) {
    const cv::Rect box(r.x, r.y + 7, 20, 20);
    cv::Scalar border = enabled ? cv::Scalar(200, 200, 200) : cv::Scalar(90, 90, 90);
    cv::rectangle(img, box, border, 1);
    if(checked) {
        cv::rectangle(img, box, cv::Scalar(80, 200, 80), cv::FILLED);
        cv::rectangle(img, box, border, 1);
    }
    cv::Scalar text = enabled ? cv::Scalar(230, 230, 230) : cv::Scalar(120, 120, 120);
    cv::putText(img, label, cv::Point(r.x + 30, r.y + 22), cv::FONT_HERSHEY_DUPLEX, 0.6, text, 1, cv::LINE_AA);
    if(enabled && ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static bool uiTextField(cv::Mat &img, const cv::Rect &r, const std::string &label, std::string &value, bool active, CvMouseState &ms) {
    const bool hover = r.contains(cv::Point(ms.x, ms.y));
    cv::Scalar border = active ? cv::Scalar(80, 200, 80) : (hover ? cv::Scalar(180, 180, 180) : cv::Scalar(120, 120, 120));
    cv::rectangle(img, r, cv::Scalar(30, 30, 30), cv::FILLED);
    cv::rectangle(img, r, border, 1);
    cv::putText(img, label, cv::Point(r.x, r.y - 6), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    std::string shown = value;
    if(shown.size() > 84) {
        shown = "..." + shown.substr(shown.size() - 84);
    }
    cv::putText(img, shown, cv::Point(r.x + 8, r.y + r.height - 10), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(ms.clicked && r.contains(cv::Point(ms.clickX, ms.clickY))) {
        ms.clicked = false;
        return true;
    }
    return false;
}

static std::string popenReadAll(const char *cmd) {
    if(!cmd || !*cmd) {
        return std::string();
    }
    FILE *fp = popen(cmd, "r");
    if(!fp) {
        return std::string();
    }
    std::string out;
    char buf[4096];
    while(true) {
        const size_t n = fread(buf, 1, sizeof(buf), fp);
        if(n > 0) {
            out.append(buf, buf + n);
        }
        if(n < sizeof(buf)) {
            break;
        }
    }
    pclose(fp);
    return out;
}

static std::string trimWhitespace(std::string s) {
    size_t b = 0;
    while(b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        b++;
    }
    size_t e = s.size();
    while(e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        e--;
    }
    return s.substr(b, e - b);
}

static std::string pickDirectoryDialogBestEffort() {
#if defined(__APPLE__)
    {
        std::string s = popenReadAll("osascript -e 'POSIX path of (choose folder with prompt \"Select data root\")'");
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            if(s.back() == '/') {
                s.pop_back();
            }
            return s;
        }
    }
#endif

#if defined(__linux__)
    {
        std::string s = popenReadAll("zenity --file-selection --directory --title=\"Select data root\" 2>/dev/null");
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
    {
        std::string s = popenReadAll("kdialog --getexistingdirectory 2>/dev/null");
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
    {
        std::string s = popenReadAll("yad --file-selection --directory --title=\"Select data root\" 2>/dev/null");
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
#endif

#if defined(_WIN32)
    {
        std::string s = popenReadAll(
            "powershell -NoProfile -Command \"Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; if($d.ShowDialog() -eq 'OK'){ $d.SelectedPath }\""
        );
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
#endif

    return std::string();
}

static std::string promptTextDialogBestEffort(const std::string &title, const std::string &text, const std::string &defaultValue) {
#if defined(__linux__)
    {
        std::string cmd = "zenity --entry --title=\"" + title + "\" --text=\"" + text + "\" --entry-text=\"" + defaultValue + "\" 2>/dev/null";
        std::string s = popenReadAll(cmd.c_str());
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
    {
        std::string cmd = "kdialog --inputbox \"" + text + "\" \"" + defaultValue + "\" 2>/dev/null";
        std::string s = popenReadAll(cmd.c_str());
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
    {
        std::string cmd = "yad --entry --title=\"" + title + "\" --text=\"" + text + "\" --entry-text=\"" + defaultValue + "\" 2>/dev/null";
        std::string s = popenReadAll(cmd.c_str());
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
#endif

#if defined(__APPLE__)
    {
        std::string cmd = "osascript -e 'text returned of (display dialog \"" + text + "\" default answer \"" + defaultValue + "\" with title \"" + title + "\")'";
        std::string s = popenReadAll(cmd.c_str());
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
#endif

#if defined(_WIN32)
    {
        std::string cmd = "powershell -NoProfile -Command \"Add-Type -AssemblyName Microsoft.VisualBasic; [Microsoft.VisualBasic.Interaction]::InputBox('" + text + "','" + title + "','" + defaultValue + "')\"";
        std::string s = popenReadAll(cmd.c_str());
        s = trimWhitespace(std::move(s));
        if(!s.empty()) {
            return s;
        }
    }
#endif

    return std::string();
}

static std::string getClipboardTextBestEffort() {
#if defined(SYNC_VIEWER_HAS_OPENCV_CLIPBOARD)
    try {
        return cv::clipboard::getText();
    }
    catch(...) {
    }
#endif

#if defined(__APPLE__)
    {
        std::string s = popenReadAll("pbpaste");
        if(!s.empty()) {
            return s;
        }
    }
#endif

#if defined(__linux__)
    {
        std::string s = popenReadAll("wl-paste -n 2>/dev/null");
        if(!s.empty()) {
            return s;
        }
    }
    {
        std::string s = popenReadAll("xclip -o -selection clipboard 2>/dev/null");
        if(!s.empty()) {
            return s;
        }
    }
    {
        std::string s = popenReadAll("xsel --clipboard --output 2>/dev/null");
        if(!s.empty()) {
            return s;
        }
    }
#endif

    return std::string();
}

static bool setClipboardTextBestEffort(const std::string &text) {
#if defined(SYNC_VIEWER_HAS_OPENCV_CLIPBOARD)
    try {
        cv::clipboard::setText(text);
        return true;
    }
    catch(...) {
    }
#endif

#if defined(__APPLE__)
    {
        FILE *fp = popen("pbcopy", "w");
        if(fp) {
            fwrite(text.data(), 1, text.size(), fp);
            pclose(fp);
            return true;
        }
    }
#endif

#if defined(__linux__)
    {
        FILE *fp = popen("wl-copy 2>/dev/null", "w");
        if(fp) {
            fwrite(text.data(), 1, text.size(), fp);
            pclose(fp);
            return true;
        }
    }
    {
        FILE *fp = popen("xclip -selection clipboard -i 2>/dev/null", "w");
        if(fp) {
            fwrite(text.data(), 1, text.size(), fp);
            pclose(fp);
            return true;
        }
    }
    {
        FILE *fp = popen("xsel --clipboard --input 2>/dev/null", "w");
        if(fp) {
            fwrite(text.data(), 1, text.size(), fp);
            pclose(fp);
            return true;
        }
    }
#endif

    return false;
}

static void appendSanitizedPaste(std::string &fieldValue, const std::string &paste) {
    if(paste.empty()) {
        return;
    }
    for(char c: paste) {
        if(c == '\r' || c == '\n') {
            continue;
        }
        if(static_cast<unsigned char>(c) < 32) {
            continue;
        }
        fieldValue.push_back(c);
    }
}

static void handleTextInput(std::string &fieldValue, int key) {
    if(key <= 0) {
        return;
    }
    const int k = (key & 0xff);
    if(k == 8 || k == 127) {
        if(!fieldValue.empty()) {
            fieldValue.pop_back();
        }
        return;
    }
    if(k == 13 || k == 10 || k == 27) {
        return;
    }
    if(k == 22) {
        appendSanitizedPaste(fieldValue, getClipboardTextBestEffort());
        return;
    }
    if(k == 3) {
        (void)setClipboardTextBestEffort(fieldValue);
        return;
    }
    if(k >= 32 && k <= 126) {
        fieldValue.push_back(static_cast<char>(k));
    }
}

static std::string readFileAllLocal(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) {
        return std::string();
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

static bool parseVec3(cJSON *arr, cv::Vec3f &out) {
    if(!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) != 3) {
        return false;
    }
    const auto *a0 = cJSON_GetArrayItem(arr, 0);
    const auto *a1 = cJSON_GetArrayItem(arr, 1);
    const auto *a2 = cJSON_GetArrayItem(arr, 2);
    if(!a0 || !a1 || !a2 || !cJSON_IsNumber(a0) || !cJSON_IsNumber(a1) || !cJSON_IsNumber(a2)) {
        return false;
    }
    out = cv::Vec3f(static_cast<float>(a0->valuedouble), static_cast<float>(a1->valuedouble), static_cast<float>(a2->valuedouble));
    return true;
}

static bool parseMat3(cJSON *arr, cv::Matx33f &out) {
    if(!arr || !cJSON_IsArray(arr)) {
        return false;
    }
    const int n = cJSON_GetArraySize(arr);
    if(n == 9) {
        float v[9];
        for(int i = 0; i < 9; i++) {
            auto *it = cJSON_GetArrayItem(arr, i);
            if(!it || !cJSON_IsNumber(it)) {
                return false;
            }
            v[i] = static_cast<float>(it->valuedouble);
        }
        out = cv::Matx33f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
        return true;
    }
    if(n == 3) {
        auto *r0 = cJSON_GetArrayItem(arr, 0);
        auto *r1 = cJSON_GetArrayItem(arr, 1);
        auto *r2 = cJSON_GetArrayItem(arr, 2);
        if(r0 && r1 && r2 && cJSON_IsArray(r0) && cJSON_IsArray(r1) && cJSON_IsArray(r2) && cJSON_GetArraySize(r0) == 3 && cJSON_GetArraySize(r1) == 3 && cJSON_GetArraySize(r2) == 3) {
            float v[9];
            cJSON *rows[3] = { r0, r1, r2 };
            for(int y = 0; y < 3; y++) {
                for(int x = 0; x < 3; x++) {
                    auto *it = cJSON_GetArrayItem(rows[y], x);
                    if(!it || !cJSON_IsNumber(it)) {
                        return false;
                    }
                    v[y * 3 + x] = static_cast<float>(it->valuedouble);
                }
            }
            out = cv::Matx33f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
            return true;
        }
    }
    return false;
}

struct ExtrinsicCamToWorld {
    bool valid = false;
    cv::Matx33f R = cv::Matx33f::eye();
    cv::Vec3f t = cv::Vec3f(0, 0, 0);
};

static bool parseMat4(cJSON *arr, cv::Matx44f &out) {
    if(!arr || !cJSON_IsArray(arr)) {
        return false;
    }
    const int n = cJSON_GetArraySize(arr);
    if(n == 16) {
        float v[16];
        for(int i = 0; i < 16; ++i) {
            auto *it = cJSON_GetArrayItem(arr, i);
            if(!it || !cJSON_IsNumber(it)) {
                return false;
            }
            v[i] = static_cast<float>(it->valuedouble);
        }
        out = cv::Matx44f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
        return true;
    }
    if(n == 4) {
        float v[16];
        for(int y = 0; y < 4; ++y) {
            auto *row = cJSON_GetArrayItem(arr, y);
            if(!row || !cJSON_IsArray(row) || cJSON_GetArraySize(row) != 4) {
                return false;
            }
            for(int x = 0; x < 4; ++x) {
                auto *it = cJSON_GetArrayItem(row, x);
                if(!it || !cJSON_IsNumber(it)) {
                    return false;
                }
                v[y * 4 + x] = static_cast<float>(it->valuedouble);
            }
        }
        out = cv::Matx44f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
        return true;
    }
    return false;
}

static std::unordered_map<std::string, ExtrinsicCamToWorld> loadExtrinsicsCamToWorld(const fs::path &path) {
    std::unordered_map<std::string, ExtrinsicCamToWorld> out;
    const std::string content = readFileAllLocal(path);
    if(content.empty()) {
        return out;
    }
    cJSON *root = cJSON_Parse(content.c_str());
    if(!root || !cJSON_IsObject(root)) {
        if(root) {
            cJSON_Delete(root);
        }
        return out;
    }
    for(cJSON *item = root->child; item != nullptr; item = item->next) {
        if(!item->string || !cJSON_IsObject(item)) {
            continue;
        }
        const std::string camId = item->string;
        cJSON *rotArr = cJSON_GetObjectItemCaseSensitive(item, "rotation");
        cJSON *tArr = cJSON_GetObjectItemCaseSensitive(item, "translation");
        cv::Matx33f Rcw = cv::Matx33f::eye();
        cv::Vec3f tCw;
        if(!parseMat3(rotArr, Rcw)) {
            continue;
        }
        if(!parseVec3(tArr, tCw)) {
            continue;
        }
        const cv::Matx33f Rwc = Rcw.t();
        const cv::Vec3f tWc = -(Rwc * tCw);
        ExtrinsicCamToWorld ex;
        ex.valid = true;
        ex.R = Rwc;
        ex.t = tWc;
        out[camId] = ex;
    }
    cJSON_Delete(root);
    return out;
}

static std::unordered_map<int, ExtrinsicCamToWorld> loadEgoExtrinsicsCamToWorldJson(const fs::path &path, const ExtrinsicCamToWorld &worldFromReference) {
    std::unordered_map<int, ExtrinsicCamToWorld> out;
    if(!worldFromReference.valid) {
        return out;
    }
    auto isNumericFrameKey = [](const char *text) {
        if(!text || !*text) {
            return false;
        }
        for(const char *p = text; *p; ++p) {
            if(*p < '0' || *p > '9') {
                return false;
            }
        }
        return true;
    };
    const std::string content = readFileAllLocal(path);
    if(content.empty()) {
        return out;
    }
    cJSON *root = cJSON_Parse(content.c_str());
    if(!root || !cJSON_IsObject(root)) {
        if(root) {
            cJSON_Delete(root);
        }
        return out;
    }
    for(cJSON *item = root->child; item != nullptr; item = item->next) {
        if(!isNumericFrameKey(item->string)) {
            continue;
        }
        cv::Matx44f egoFromReference;
        if(!parseMat4(item, egoFromReference)) {
            continue;
        }
        const cv::Matx33f RegoFromReference(egoFromReference(0, 0),
                                            egoFromReference(0, 1),
                                            egoFromReference(0, 2),
                                            egoFromReference(1, 0),
                                            egoFromReference(1, 1),
                                            egoFromReference(1, 2),
                                            egoFromReference(2, 0),
                                            egoFromReference(2, 1),
                                            egoFromReference(2, 2));
        const cv::Vec3f tegoFromReference(egoFromReference(0, 3), egoFromReference(1, 3), egoFromReference(2, 3));
        const cv::Matx33f RreferenceFromEgo = RegoFromReference.t();
        const cv::Vec3f treferenceFromEgo = -(RreferenceFromEgo * tegoFromReference);

        ExtrinsicCamToWorld worldFromEgo;
        worldFromEgo.valid = true;
        worldFromEgo.R = worldFromReference.R * RreferenceFromEgo;
        worldFromEgo.t = worldFromReference.R * treferenceFromEgo + worldFromReference.t;
        out[std::stoi(item->string)] = worldFromEgo;
    }
    cJSON_Delete(root);
    return out;
}

struct CameraRgbToDepthParams {
    bool valid = false;
    OBCameraIntrinsic depthIntrinsic{};
    OBCameraDistortion depthDistortion{};
    OBCameraIntrinsic rgbIntrinsic{};
    OBCameraDistortion rgbDistortion{};
    OBExtrinsic d2cExtrinsic{};
    OBExtrinsic c2dExtrinsic{};
};

struct CameraStreamParams {
    bool valid = false;
    int width = 0;
    int height = 0;
    int fps = 0;
    int format = 0;
    OBCameraIntrinsic intrinsic{};
    OBCameraDistortion distortion{};
};

static bool parseIntrinsic(cJSON *obj, OBCameraIntrinsic &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    auto get = [&](const char *k, float &v) {
        auto *it = cJSON_GetObjectItemCaseSensitive(obj, k);
        if(!it || !cJSON_IsNumber(it)) {
            return false;
        }
        v = static_cast<float>(it->valuedouble);
        return true;
    };
    auto getInt = [&](const char *k, int &v) {
        auto *it = cJSON_GetObjectItemCaseSensitive(obj, k);
        if(!it || !cJSON_IsNumber(it)) {
            return false;
        }
        v = it->valueint;
        return true;
    };
    float fx = 0, fy = 0, cx = 0, cy = 0;
    if(!get("fx", fx) || !get("fy", fy) || !get("cx", cx) || !get("cy", cy)) {
        return false;
    }
    out.fx = fx;
    out.fy = fy;
    out.cx = cx;
    out.cy = cy;

    int w = 0;
    int h = 0;
    if(getInt("width", w)) {
        out.width = w;
    }
    if(getInt("height", h)) {
        out.height = h;
    }
    return true;
}

static bool parseDistortion(cJSON *obj, OBCameraDistortion &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    auto get = [&](const char *k, float &v, bool required) {
        auto *it = cJSON_GetObjectItemCaseSensitive(obj, k);
        if(!it || !cJSON_IsNumber(it)) {
            return !required;
        }
        v = static_cast<float>(it->valuedouble);
        return true;
    };
    float k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0;
    float p1 = 0, p2 = 0;
    if(!get("k1", k1, false) || !get("k2", k2, false) || !get("k3", k3, false) || !get("k4", k4, false) || !get("k5", k5, false) || !get("k6", k6, false) || !get("p1", p1, false) || !get("p2", p2, false)) {
        return false;
    }
    out.k1 = k1;
    out.k2 = k2;
    out.k3 = k3;
    out.k4 = k4;
    out.k5 = k5;
    out.k6 = k6;
    out.p1 = p1;
    out.p2 = p2;
    auto *modelIt = cJSON_GetObjectItemCaseSensitive(obj, "model");
    if(modelIt && cJSON_IsNumber(modelIt)) {
        out.model = static_cast<decltype(out.model)>(modelIt->valueint);
    }
    return true;
}

static bool parseExtrinsic(cJSON *obj, OBExtrinsic &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    cJSON *rot = cJSON_GetObjectItemCaseSensitive(obj, "rotation");
    cJSON *t = cJSON_GetObjectItemCaseSensitive(obj, "translation");
    cv::Matx33f R;
    cv::Vec3f tv;
    if(!parseMat3(rot, R) || !parseVec3(t, tv)) {
        return false;
    }
    out.rot[0] = R(0, 0);
    out.rot[1] = R(0, 1);
    out.rot[2] = R(0, 2);
    out.rot[3] = R(1, 0);
    out.rot[4] = R(1, 1);
    out.rot[5] = R(1, 2);
    out.rot[6] = R(2, 0);
    out.rot[7] = R(2, 1);
    out.rot[8] = R(2, 2);
    out.trans[0] = tv[0];
    out.trans[1] = tv[1];
    out.trans[2] = tv[2];
    return true;
}

struct CameraParamsBundle {
    std::unordered_map<std::string, CameraStreamParams> rgb;
    std::unordered_map<std::string, CameraStreamParams> depth;
    std::unordered_map<std::string, CameraStreamParams> irLeft;
    std::unordered_map<std::string, CameraStreamParams> irRight;
    std::unordered_map<std::string, CameraRgbToDepthParams> rgbToDepth;
};

static bool parseStreamParams(cJSON *obj, CameraStreamParams &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    auto getInt = [&](const char *k, int &v, bool required) {
        auto *it = cJSON_GetObjectItemCaseSensitive(obj, k);
        if(!it || !cJSON_IsNumber(it)) {
            return !required;
        }
        v = it->valueint;
        return true;
    };
    cJSON *intr = cJSON_GetObjectItemCaseSensitive(obj, "intrinsic");
    cJSON *dist = cJSON_GetObjectItemCaseSensitive(obj, "distortion");
    if(!getInt("width", out.width, false) || !getInt("height", out.height, false) || !getInt("fps", out.fps, false) || !getInt("format", out.format, false)) {
        return false;
    }
    parseIntrinsic(intr, out.intrinsic);
    parseDistortion(dist, out.distortion);
    if(out.width > 0 && out.height > 0) {
        if(out.intrinsic.width <= 0) {
            out.intrinsic.width = out.width;
        }
        if(out.intrinsic.height <= 0) {
            out.intrinsic.height = out.height;
        }
    }
    out.valid = out.intrinsic.fx > 0.0f && out.intrinsic.fy > 0.0f;
    return true;
}

static bool parseRgbToDepth(cJSON *obj, CameraRgbToDepthParams &out) {
    if(!obj || !cJSON_IsObject(obj)) {
        return false;
    }
    cJSON *depthIntr = cJSON_GetObjectItemCaseSensitive(obj, "depth_intrinsic");
    cJSON *depthDist = cJSON_GetObjectItemCaseSensitive(obj, "depth_distortion");
    cJSON *rgbIntr = cJSON_GetObjectItemCaseSensitive(obj, "rgb_intrinsic");
    cJSON *rgbDist = cJSON_GetObjectItemCaseSensitive(obj, "rgb_distortion");
    cJSON *d2c = cJSON_GetObjectItemCaseSensitive(obj, "d2c_extrinsic");
    cJSON *c2d = cJSON_GetObjectItemCaseSensitive(obj, "c2d_extrinsic");
    if(!parseIntrinsic(depthIntr, out.depthIntrinsic) || !parseIntrinsic(rgbIntr, out.rgbIntrinsic)) {
        return false;
    }
    parseDistortion(depthDist, out.depthDistortion);
    parseDistortion(rgbDist, out.rgbDistortion);
    if(!parseExtrinsic(d2c, out.d2cExtrinsic)) {
        return false;
    }
    parseExtrinsic(c2d, out.c2dExtrinsic);
    out.valid = true;
    return true;
}

static CameraParamsBundle loadCameraParams(const fs::path &path) {
    CameraParamsBundle bundle;
    const std::string content = readFileAllLocal(path);
    if(content.empty()) {
        return bundle;
    }
    cJSON *root = cJSON_Parse(content.c_str());
    if(!root || !cJSON_IsObject(root)) {
        if(root) {
            cJSON_Delete(root);
        }
        return bundle;
    }
    for(cJSON *camItem = root->child; camItem != nullptr; camItem = camItem->next) {
        if(!camItem->string || !cJSON_IsObject(camItem)) {
            continue;
        }
        const std::string camKey = camItem->string;
        CameraStreamParams sp;
        if(parseStreamParams(cJSON_GetObjectItemCaseSensitive(camItem, "RGB"), sp)) {
            bundle.rgb[camKey] = sp;
        }
        if(parseStreamParams(cJSON_GetObjectItemCaseSensitive(camItem, "Depth"), sp)) {
            bundle.depth[camKey] = sp;
        }
        if(parseStreamParams(cJSON_GetObjectItemCaseSensitive(camItem, "IR_left"), sp)) {
            bundle.irLeft[camKey] = sp;
        }
        if(parseStreamParams(cJSON_GetObjectItemCaseSensitive(camItem, "IR_right"), sp)) {
            bundle.irRight[camKey] = sp;
        }
        CameraRgbToDepthParams rp;
        if(parseRgbToDepth(cJSON_GetObjectItemCaseSensitive(camItem, "rgb_to_depth"), rp)) {
            bundle.rgbToDepth[camKey] = rp;
        }
    }
    cJSON_Delete(root);
    return bundle;
}

static bool isDigits(const std::string &s) {
    if(s.empty()) {
        return false;
    }
    for(char c: s) {
        if(!(c >= '0' && c <= '9')) {
            return false;
        }
    }
    return true;
}

static std::string stripLeadingZeros(const std::string &s) {
    if(s.empty()) {
        return s;
    }
    size_t i = 0;
    while(i + 1 < s.size() && s[i] == '0') {
        i++;
    }
    return s.substr(i);
}

static std::string padLeftZeros(const std::string &s, size_t width) {
    if(s.size() >= width) {
        return s;
    }
    return std::string(width - s.size(), '0') + s;
}

static bool cameraKeyEquivalent(const std::string &a, const std::string &b) {
    if(a == b) {
        return true;
    }
    if(!isDigits(a) || !isDigits(b)) {
        return false;
    }
    return stripLeadingZeros(a) == stripLeadingZeros(b);
}

static std::string sanitizeCameraIdInput(const std::string &value) {
    std::string out;
    out.reserve(std::min<size_t>(4, value.size()));
    for(char c : value) {
        if(c >= '0' && c <= '9') {
            out.push_back(c);
            if(out.size() >= 4) {
                break;
            }
        }
    }
    return out;
}

template <class MapT>
static const typename MapT::mapped_type *findByCamKeyVariants(const MapT &m, const std::string &camKey) {
    auto it = m.find(camKey);
    if(it != m.end()) {
        return &it->second;
    }
    const std::string stripped = stripLeadingZeros(camKey);
    it = m.find(stripped);
    if(it != m.end()) {
        return &it->second;
    }
    if(isDigits(camKey)) {
        const std::string pad2 = padLeftZeros(stripped, 2);
        it = m.find(pad2);
        if(it != m.end()) {
            return &it->second;
        }
        const std::string pad3 = padLeftZeros(stripped, 3);
        it = m.find(pad3);
        if(it != m.end()) {
            return &it->second;
        }
        const std::string pad4 = padLeftZeros(stripped, 4);
        it = m.find(pad4);
        if(it != m.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

static int computeTotalFramesFromDir(const fs::path &dir) {
    if(!fs::exists(dir) || !fs::is_directory(dir)) {
        return 0;
    }
    static const std::regex re("^(\\d{5})\\.[^.]+$");
    int maxIdx = -1;
    for(const auto &e: fs::directory_iterator(dir)) {
        if(!e.is_regular_file()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        std::smatch m;
        if(!std::regex_match(name, m, re)) {
            continue;
        }
        try {
            const int idx = std::stoi(m[1].str());
            if(idx > maxIdx) {
                maxIdx = idx;
            }
        }
        catch(...) {
        }
    }
    return (maxIdx >= 0) ? (maxIdx + 1) : 0;
}

static int countCsvDataRows(const fs::path &path) {
    std::ifstream ifs(path);
    if(!ifs.is_open()) {
        return 0;
    }
    std::string line;
    int rows = 0;
    bool first = true;
    while(std::getline(ifs, line)) {
        if(first) {
            first = false;
            continue;
        }
        if(trimWhitespace(line).empty()) {
            continue;
        }
        rows++;
    }
    return rows;
}

static fs::path findFrameFile(const fs::path &dir, int frameIdx, const std::vector<std::string> &extensions) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << frameIdx;
    const std::string base = oss.str();
    for(const auto &ext: extensions) {
        fs::path p = dir / (base + ext);
        if(fs::exists(p) && fs::is_regular_file(p)) {
            return p;
        }
    }
    if(!fs::exists(dir) || !fs::is_directory(dir)) {
        return fs::path();
    }
    for(const auto &e: fs::directory_iterator(dir)) {
        if(!e.is_regular_file()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        if(name.rfind(base + ".", 0) == 0) {
            return e.path();
        }
    }
    return fs::path();
}

static cv::Mat toBgrForDisplay(const cv::Mat &m) {
    if(m.empty()) {
        return cv::Mat();
    }
    if(m.type() == CV_8UC3) {
        return m;
    }
    if(m.type() == CV_8UC1) {
        cv::Mat bgr;
        cv::cvtColor(m, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if(m.type() == CV_16UC1) {
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(m, &minv, &maxv);
        if(maxv <= minv) {
            maxv = minv + 1.0;
        }
        cv::Mat tmp8;
        m.convertTo(tmp8, CV_8UC1, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
        cv::Mat bgr;
        cv::cvtColor(tmp8, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    cv::Mat out;
    try {
        m.convertTo(out, CV_8UC3);
    }
    catch(...) {
        out.release();
    }
    return out;
}

static cv::Mat depth16ToYellowBlue(const cv::Mat &depth16, float maxDepthM, float valueScaleMm) {
    cv::Mat out(depth16.rows, depth16.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    if(depth16.empty() || depth16.type() != CV_16UC1 || depth16.rows <= 0 || depth16.cols <= 0) {
        return out;
    }
    const float maxMm = std::max(100.0f, maxDepthM * 1000.0f);
    const float sMm = (valueScaleMm > 0.0f) ? valueScaleMm : 1.0f;
    for(int y = 0; y < depth16.rows; y++) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        cv::Vec3b *dst = out.ptr<cv::Vec3b>(y);
        for(int x = 0; x < depth16.cols; x++) {
            const uint16_t d = row[x];
            if(d == 0) {
                dst[x] = cv::Vec3b(0, 0, 0);
                continue;
            }
            const float mm = std::min(maxMm, static_cast<float>(d) * sMm);
            const float t = std::max(0.0f, std::min(1.0f, mm / maxMm));
            const float inv = 1.0f - t;
            const float b = 255.0f * inv;
            const float g = 255.0f * t;
            const float r = 255.0f * t;
            dst[x] = cv::Vec3b(static_cast<uint8_t>(b), static_cast<uint8_t>(g), static_cast<uint8_t>(r));
        }
    }
    return out;
}

struct ViewerViewState {
    cv::Rect pcRect{0, 0, 1, 1};
    cv::Point cursor{0, 0};
    bool rotating = false;
    bool panning = false;
    cv::Point lastMouse{0, 0};
    float yawRad = 0.0f;
    float pitchRad = 0.0f;
    float distance = 1.5f;
    cv::Vec3f target{0.0f, 0.0f, 1.0f};
    void resetView() {
        rotating = false;
        panning = false;
        yawRad = 0.0f;
        pitchRad = 0.0f;
        distance = 1.5f;
        target = cv::Vec3f(0.0f, 0.0f, 1.0f);
        cursor = cv::Point(0, 0);
        lastMouse = cv::Point(0, 0);
    }
};

static cv::Vec3f normalizeVec3(const cv::Vec3f &v) {
    const float n = std::sqrt(v.dot(v));
    if(n <= 1e-8f) {
        return cv::Vec3f(0, 0, 0);
    }
    return v * (1.0f / n);
}

static cv::Vec3f crossVec3(const cv::Vec3f &a, const cv::Vec3f &b) {
    return cv::Vec3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

static void computeCameraBasis(const ViewerViewState &s, cv::Vec3f &right, cv::Vec3f &up, cv::Vec3f &forward, cv::Vec3f &camPos) {
    const float cy = std::cos(s.yawRad);
    const float sy = std::sin(s.yawRad);
    const float cp = std::cos(s.pitchRad);
    const float sp = std::sin(s.pitchRad);
    forward = normalizeVec3(cv::Vec3f(sy * cp, -sp, cy * cp));
    const cv::Vec3f worldUp(0.0f, -1.0f, 0.0f);
    right = normalizeVec3(crossVec3(worldUp, forward));
    up = crossVec3(forward, right);
    camPos = s.target - forward * s.distance;
}

struct PointCloudMouseContext {
    ViewerViewState *view = nullptr;
    const bool *allow = nullptr;
};

static void mouseCallbackPointCloud(int event, int x, int y, int flags, void *userdata) {
    auto *ctx = reinterpret_cast<PointCloudMouseContext *>(userdata);
    if(!ctx || !ctx->view) {
        return;
    }
    if(ctx->allow && !(*ctx->allow)) {
        if(event == cv::EVENT_LBUTTONUP) {
            ctx->view->rotating = false;
        }
        if(event == cv::EVENT_RBUTTONUP) {
            ctx->view->panning = false;
        }
        return;
    }
    auto *s = ctx->view;
    s->cursor = cv::Point(x, y);
    if(event == cv::EVENT_LBUTTONUP) {
        s->rotating = false;
    }
    if(event == cv::EVENT_RBUTTONUP) {
        s->panning = false;
    }
    const bool inside = s->pcRect.contains(cv::Point(x, y));
    if(!inside) {
        if(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
            return;
        }
        if(event == cv::EVENT_MOUSEMOVE && !(s->rotating || s->panning)) {
            return;
        }
    }
    if(event == cv::EVENT_LBUTTONDOWN) {
        s->rotating = true;
        s->panning = false;
        s->lastMouse = cv::Point(x, y);
        return;
    }
    if(event == cv::EVENT_RBUTTONDOWN) {
        s->panning = true;
        s->rotating = false;
        s->lastMouse = cv::Point(x, y);
        return;
    }
    if(event == cv::EVENT_MOUSEMOVE) {
        const int dx = x - s->lastMouse.x;
        const int dy = y - s->lastMouse.y;
        s->lastMouse = cv::Point(x, y);
        if(s->rotating) {
            s->yawRad += static_cast<float>(dx) * 0.005f;
            s->pitchRad += static_cast<float>(dy) * 0.005f;
            s->pitchRad = std::min(1.55f, std::max(-1.55f, s->pitchRad));
            return;
        }
        if(s->panning) {
            cv::Vec3f right, up, forward, camPos;
            computeCameraBasis(*s, right, up, forward, camPos);
            const float scale = s->distance * 0.001f;
            s->target -= right * (static_cast<float>(dx) * scale);
            s->target += up * (static_cast<float>(dy) * scale);
            return;
        }
    }
}

enum class ViewerDataType {
    RGB,
    Depth,
    IR,
    PointCloud,
    ColorCloud
};

enum class ViewerSourceKind {
    Multiview,
    Fisheye,
};

struct ViewerSource {
    std::string      sourceId;
    std::string      displayName;
    ViewerSourceKind kind = ViewerSourceKind::Multiview;
    std::string      camKey;
    fs::path         rgbDir;
    fs::path         depthDir;
    fs::path         irDir;
    bool             hasRgb = false;
    bool             hasDepth = false;
    bool             hasIr = false;
    bool             visible = true;
};

static std::string dataTypeToLabel(ViewerDataType t) {
    switch(t) {
    case ViewerDataType::RGB:
        return "RGB";
    case ViewerDataType::Depth:
        return "Depth";
    case ViewerDataType::IR:
        return "IR";
    case ViewerDataType::PointCloud:
        return "PointCloud";
    case ViewerDataType::ColorCloud:
        return "ColorCloud";
    }
    return "Unknown";
}

struct SubjectEntry {
    std::string name;
    struct TaskEntry {
        std::string name;
        std::vector<std::string> episodes;
        bool expanded = false;
    };
    std::vector<TaskEntry> tasks;
    bool expanded = false;
};

static bool isEpisodeDirName(const std::string &name) {
    return name.rfind("episode_", 0) == 0;
}

static std::vector<SubjectEntry::TaskEntry> listTasks(const fs::path &subjectDir) {
    static const std::unordered_set<std::string> ignore = { "RGB", "Depth", "IR", "IR_left", "IR_right", "PointCloud", "CloudPoints", "ColorCloud", "ColorCloudPoints", "record.csv",
                                                           "labels.json", "camera_params.json", "extrinsics.json" };
    std::vector<SubjectEntry::TaskEntry> out;
    if(!fs::exists(subjectDir) || !fs::is_directory(subjectDir)) {
        return out;
    }
    for(const auto &e: fs::directory_iterator(subjectDir)) {
        if(!e.is_directory()) {
            continue;
        }
        const std::string name = e.path().filename().string();
        if(ignore.find(name) != ignore.end()) {
            continue;
        }
        SubjectEntry::TaskEntry task;
        task.name = name;
        for(const auto &child: fs::directory_iterator(e.path())) {
            if(!child.is_directory()) {
                continue;
            }
            const std::string childName = child.path().filename().string();
            if(isEpisodeDirName(childName)) {
                task.episodes.push_back(childName);
            }
        }
        std::sort(task.episodes.begin(), task.episodes.end());
        out.push_back(std::move(task));
    }
    std::sort(out.begin(), out.end(), [](const SubjectEntry::TaskEntry &a, const SubjectEntry::TaskEntry &b) { return a.name < b.name; });
    return out;
}

static std::vector<SubjectEntry> scanSubjects(const fs::path &root) {
    std::vector<SubjectEntry> out;
    if(!fs::exists(root) || !fs::is_directory(root)) {
        return out;
    }
    for(const auto &e: fs::directory_iterator(root)) {
        if(!e.is_directory()) {
            continue;
        }
        SubjectEntry s;
        s.name = e.path().filename().string();
        s.tasks = listTasks(e.path());
        out.push_back(std::move(s));
    }
    std::sort(out.begin(), out.end(), [](const SubjectEntry &a, const SubjectEntry &b) { return a.name < b.name; });
    return out;
}

static std::unordered_map<int, std::unordered_map<std::string, std::vector<cv::Point2f>>> loadLabelsForTask(const fs::path &labelsJson, const std::string &taskName) {
    std::unordered_map<int, std::unordered_map<std::string, std::vector<cv::Point2f>>> out;
    const std::string content = readFileAllLocal(labelsJson);
    if(content.empty()) {
        return out;
    }
    cJSON *root = cJSON_Parse(content.c_str());
    if(!root || !cJSON_IsObject(root)) {
        if(root) {
            cJSON_Delete(root);
        }
        return out;
    }
    cJSON *taskObj = cJSON_GetObjectItemCaseSensitive(root, taskName.c_str());
    if(!taskObj || !cJSON_IsObject(taskObj)) {
        cJSON_Delete(root);
        return out;
    }
    for(cJSON *camItem = taskObj->child; camItem != nullptr; camItem = camItem->next) {
        if(!camItem->string || !cJSON_IsObject(camItem)) {
            continue;
        }
        const std::string camId = camItem->string;
        for(cJSON *frameItem = camItem->child; frameItem != nullptr; frameItem = frameItem->next) {
            if(!frameItem->string) {
                continue;
            }
            int frameKey = -1;
            try {
                frameKey = std::stoi(std::string(frameItem->string));
            }
            catch(...) {
                continue;
            }
            if(!cJSON_IsArray(frameItem)) {
                continue;
            }
            std::vector<cv::Point2f> pts;
            const int n = cJSON_GetArraySize(frameItem);
            pts.reserve(static_cast<size_t>(std::max(0, n)));
            for(int i = 0; i < n; i++) {
                cJSON *p = cJSON_GetArrayItem(frameItem, i);
                if(!p || !cJSON_IsArray(p) || cJSON_GetArraySize(p) != 2) {
                    continue;
                }
                cJSON *px = cJSON_GetArrayItem(p, 0);
                cJSON *py = cJSON_GetArrayItem(p, 1);
                if(!px || !py || !cJSON_IsNumber(px) || !cJSON_IsNumber(py)) {
                    continue;
                }
                pts.emplace_back(static_cast<float>(px->valueint), static_cast<float>(py->valueint));
            }
            out[frameKey][camId] = std::move(pts);
        }
    }
    cJSON_Delete(root);
    return out;
}

static void drawFitImage(cv::Mat &dst, const cv::Rect &cell, const cv::Mat &bgr, const std::string &title) {
    cv::rectangle(dst, cell, cv::Scalar(25, 25, 25), cv::FILLED);
    cv::rectangle(dst, cell, cv::Scalar(70, 70, 70), 1);
    cv::putText(dst, title, cv::Point(cell.x + 10, cell.y + 24), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    if(bgr.empty()) {
        cv::putText(dst, "(no data)", cv::Point(cell.x + 10, cell.y + 54), cv::FONT_HERSHEY_DUPLEX, 0.55, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
        return;
    }
    const int headerH = 32;
    cv::Rect view(cell.x + 6, cell.y + headerH + 4, cell.width - 12, cell.height - headerH - 10);
    if(view.width <= 1 || view.height <= 1) {
        return;
    }
    const float sx = static_cast<float>(view.width) / static_cast<float>(bgr.cols);
    const float sy = static_cast<float>(view.height) / static_cast<float>(bgr.rows);
    const float s = std::min(sx, sy);
    const int w = std::max(1, static_cast<int>(std::round(static_cast<float>(bgr.cols) * s)));
    const int h = std::max(1, static_cast<int>(std::round(static_cast<float>(bgr.rows) * s)));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    cv::Rect roi(view.x + (view.width - w) / 2, view.y + (view.height - h) / 2, w, h);
    resized.copyTo(dst(roi));
}

static void drawGridImages(cv::Mat &dst, const cv::Rect &r, const std::vector<std::pair<std::string, cv::Mat>> &frames) {
    cv::rectangle(dst, r, cv::Scalar(18, 18, 18), cv::FILLED);
    cv::rectangle(dst, r, cv::Scalar(80, 80, 80), 1);
    if(frames.empty()) {
        cv::putText(dst, "No camera data", cv::Point(r.x + 12, r.y + 30), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
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
        cv::Rect cell(r.x + cx * cellW, r.y + cy * cellH, (cx == cols - 1) ? (r.x + r.width - (r.x + cx * cellW)) : cellW,
                      (cy == rows - 1) ? (r.y + r.height - (r.y + cy * cellH)) : cellH);
        drawFitImage(dst, cell, frames[i].second, frames[i].first);
    }
}

struct AlignMapCache {
    int frameIdx = -1;
    int rgbW = 0;
    int rgbH = 0;
    std::vector<int32_t> colorToDepth;
    std::vector<uint16_t> colorDepthMm;
    int depthW = 0;
    int depthH = 0;
    std::vector<int32_t> depthToColor;
};

static bool isAlignedDepthToRgb(const cv::Mat &depth16, const cv::Mat &rgb) {
    return !depth16.empty() && !rgb.empty() && depth16.cols == rgb.cols && depth16.rows == rgb.rows;
}

static bool isAlignedDepthToRgb(const cv::Mat &depth16, int rgbW, int rgbH) {
    return !depth16.empty() && rgbW > 0 && rgbH > 0 && depth16.cols == rgbW && depth16.rows == rgbH;
}

static int32_t packXY(int x, int y) {
    if(x < 0 || y < 0 || x > 65535 || y > 65535) {
        return -1;
    }
    return static_cast<int32_t>((y << 16) | (x & 0xffff));
}

static bool unpackXY(int32_t v, int &x, int &y) {
    if(v < 0) {
        return false;
    }
    x = v & 0xffff;
    y = (v >> 16) & 0xffff;
    return true;
}

class MatLruCache {
public:
    explicit MatLruCache(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity)) {}

    void clear() {
        map_.clear();
        lru_.clear();
    }

    bool tryGet(const std::string &key, cv::Mat &out) {
        auto it = map_.find(key);
        if(it == map_.end()) {
            return false;
        }
        lru_.splice(lru_.begin(), lru_, it->second);
        out = it->second->second;
        return !out.empty();
    }

    template <class Loader>
    cv::Mat getOrLoad(const std::string &key, Loader loader) {
        cv::Mat v;
        if(tryGet(key, v)) {
            return v;
        }
        v = loader();
        put(key, v);
        return v;
    }

    template <class Loader>
    void prefetch(const std::string &key, Loader loader) {
        auto it = map_.find(key);
        if(it != map_.end()) {
            return;
        }
        cv::Mat v = loader();
        if(v.empty()) {
            return;
        }
        put(key, v);
    }

private:
    void put(const std::string &key, const cv::Mat &v) {
        if(v.empty()) {
            return;
        }
        auto it = map_.find(key);
        if(it != map_.end()) {
            it->second->second = v;
            lru_.splice(lru_.begin(), lru_, it->second);
            return;
        }
        lru_.emplace_front(key, v);
        map_[key] = lru_.begin();
        while(map_.size() > capacity_) {
            auto last = lru_.end();
            --last;
            map_.erase(last->first);
            lru_.pop_back();
        }
    }

    size_t capacity_;
    std::list<std::pair<std::string, cv::Mat>> lru_;
    std::unordered_map<std::string, std::list<std::pair<std::string, cv::Mat>>::iterator> map_;
};

static void buildDepthToColorMap(AlignMapCache &cache,
                                 int frameIdx,
                                 const cv::Mat &depth16,
                                 int rgbW,
                                 int rgbH,
                                 const CameraRgbToDepthParams &p,
                                 float maxDepthM,
                                 float valueScaleMm) {
    cache.frameIdx = frameIdx;
    cache.rgbW = rgbW;
    cache.rgbH = rgbH;
    cache.colorToDepth.assign(static_cast<size_t>(rgbW) * static_cast<size_t>(rgbH), -1);
    cache.colorDepthMm.assign(static_cast<size_t>(rgbW) * static_cast<size_t>(rgbH), 0);
    cache.depthW = depth16.empty() ? 0 : depth16.cols;
    cache.depthH = depth16.empty() ? 0 : depth16.rows;
    cache.depthToColor.assign(static_cast<size_t>(std::max(0, cache.depthW)) * static_cast<size_t>(std::max(0, cache.depthH)), -1);
    if(depth16.empty() || depth16.type() != CV_16UC1 || rgbW <= 0 || rgbH <= 0 || !p.valid) {
        return;
    }
    const float maxMm = std::max(100.0f, maxDepthM * 1000.0f);
    const float sMm = (valueScaleMm > 0.0f) ? valueScaleMm : 1.0f;
    const int step = 1;
    for(int y = 0; y < depth16.rows; y += step) {
        const uint16_t *row = depth16.ptr<uint16_t>(y);
        for(int x = 0; x < depth16.cols; x += step) {
            const uint16_t d = row[x];
            if(d == 0) {
                continue;
            }
            const float depthMm = static_cast<float>(d) * sMm;
            if(!(depthMm > 0.0f && depthMm <= maxMm)) {
                continue;
            }
            OBPoint2f src;
            src.x = static_cast<float>(x);
            src.y = static_cast<float>(y);
            OBPoint2f dst;
            const bool ok = ob::CoordinateTransformHelper::transformation2dto2d(src,
                                                                                depthMm,
                                                                                p.depthIntrinsic,
                                                                                p.depthDistortion,
                                                                                p.rgbIntrinsic,
                                                                                p.rgbDistortion,
                                                                                p.d2cExtrinsic,
                                                                                &dst);
            if(!ok) {
                continue;
            }
            const int u = static_cast<int>(std::lround(dst.x));
            const int v = static_cast<int>(std::lround(dst.y));
            if(u < 0 || u >= rgbW || v < 0 || v >= rgbH) {
                continue;
            }

            cache.depthToColor[static_cast<size_t>(y) * static_cast<size_t>(cache.depthW) + static_cast<size_t>(x)] = packXY(u, v);

            const size_t idx = static_cast<size_t>(v) * static_cast<size_t>(rgbW) + static_cast<size_t>(u);
            const int32_t packed = packXY(x, y);
            const uint16_t prevD = cache.colorDepthMm[idx];
            if(cache.colorToDepth[idx] < 0 || prevD == 0 || d < prevD) {
                cache.colorToDepth[idx] = packed;
                cache.colorDepthMm[idx] = d;
            }
        }
    }
}

static bool lookupDepthForRgb(const AlignMapCache &cache, int rgbX, int rgbY, int &outDx, int &outDy) {
    if(cache.rgbW <= 0 || cache.rgbH <= 0) {
        return false;
    }
    auto in = [&](int x, int y) { return x >= 0 && x < cache.rgbW && y >= 0 && y < cache.rgbH; };
    if(in(rgbX, rgbY)) {
        const int32_t v = cache.colorToDepth[static_cast<size_t>(rgbY) * static_cast<size_t>(cache.rgbW) + static_cast<size_t>(rgbX)];
        if(unpackXY(v, outDx, outDy)) {
            return true;
        }
    }
    const int maxR = 24;
    for(int r = 1; r <= maxR; r++) {
        for(int dy = -r; dy <= r; dy++) {
            const int y = rgbY + dy;
            const int x1 = rgbX - r;
            const int x2 = rgbX + r;
            if(in(x1, y)) {
                const int32_t v = cache.colorToDepth[static_cast<size_t>(y) * static_cast<size_t>(cache.rgbW) + static_cast<size_t>(x1)];
                if(unpackXY(v, outDx, outDy)) {
                    return true;
                }
            }
            if(in(x2, y)) {
                const int32_t v = cache.colorToDepth[static_cast<size_t>(y) * static_cast<size_t>(cache.rgbW) + static_cast<size_t>(x2)];
                if(unpackXY(v, outDx, outDy)) {
                    return true;
                }
            }
        }
        for(int dx = -r + 1; dx <= r - 1; dx++) {
            const int x = rgbX + dx;
            const int y1 = rgbY - r;
            const int y2 = rgbY + r;
            if(in(x, y1)) {
                const int32_t v = cache.colorToDepth[static_cast<size_t>(y1) * static_cast<size_t>(cache.rgbW) + static_cast<size_t>(x)];
                if(unpackXY(v, outDx, outDy)) {
                    return true;
                }
            }
            if(in(x, y2)) {
                const int32_t v = cache.colorToDepth[static_cast<size_t>(y2) * static_cast<size_t>(cache.rgbW) + static_cast<size_t>(x)];
                if(unpackXY(v, outDx, outDy)) {
                    return true;
                }
            }
        }
    }
    return false;
}

struct CloudPoint {
    cv::Vec3f p;
    cv::Vec3b c;
    bool hasColor = false;
};

using CloudPointVec = std::vector<CloudPoint>;
using CloudPointVecPtr = std::shared_ptr<CloudPointVec>;
using JointWorldVec = std::vector<cv::Vec3f>;
using JointWorldVecPtr = std::shared_ptr<JointWorldVec>;

class CloudLruCache {
public:
    explicit CloudLruCache(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity)) {}

    void clear() {
        map_.clear();
        lru_.clear();
    }

    bool tryGet(const std::string &key, CloudPointVecPtr &out) {
        auto it = map_.find(key);
        if(it == map_.end()) {
            return false;
        }
        lru_.splice(lru_.begin(), lru_, it->second);
        out = it->second->second;
        return static_cast<bool>(out);
    }

    template <class Loader>
    CloudPointVecPtr getOrLoad(const std::string &key, Loader loader) {
        CloudPointVecPtr v;
        if(tryGet(key, v)) {
            return v;
        }
        v = loader();
        put(key, v);
        return v;
    }

    template <class Loader>
    void prefetch(const std::string &key, Loader loader) {
        auto it = map_.find(key);
        if(it != map_.end()) {
            return;
        }
        CloudPointVecPtr v = loader();
        if(!v) {
            return;
        }
        put(key, v);
    }

private:
    void put(const std::string &key, const CloudPointVecPtr &v) {
        if(!v) {
            return;
        }
        auto it = map_.find(key);
        if(it != map_.end()) {
            it->second->second = v;
            lru_.splice(lru_.begin(), lru_, it->second);
            return;
        }
        lru_.emplace_front(key, v);
        map_[key] = lru_.begin();
        while(map_.size() > capacity_) {
            auto last = lru_.end();
            --last;
            map_.erase(last->first);
            lru_.pop_back();
        }
    }

    size_t capacity_;
    std::list<std::pair<std::string, CloudPointVecPtr>> lru_;
    std::unordered_map<std::string, std::list<std::pair<std::string, CloudPointVecPtr>>::iterator> map_;
};

class JointLruCache {
public:
    explicit JointLruCache(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity)) {}

    void clear() {
        map_.clear();
        lru_.clear();
    }

    bool tryGet(const std::string &key, JointWorldVecPtr &out) {
        auto it = map_.find(key);
        if(it == map_.end()) {
            return false;
        }
        lru_.splice(lru_.begin(), lru_, it->second);
        out = it->second->second;
        return static_cast<bool>(out);
    }

    template <class Loader>
    JointWorldVecPtr getOrLoad(const std::string &key, Loader loader) {
        JointWorldVecPtr v;
        if(tryGet(key, v)) {
            return v;
        }
        v = loader();
        put(key, v);
        return v;
    }

    template <class Loader>
    void prefetch(const std::string &key, Loader loader) {
        auto it = map_.find(key);
        if(it != map_.end()) {
            return;
        }
        JointWorldVecPtr v = loader();
        if(!v) {
            return;
        }
        put(key, v);
    }

private:
    void put(const std::string &key, const JointWorldVecPtr &v) {
        if(!v) {
            return;
        }
        auto it = map_.find(key);
        if(it != map_.end()) {
            it->second->second = v;
            lru_.splice(lru_.begin(), lru_, it->second);
            return;
        }
        lru_.emplace_front(key, v);
        map_[key] = lru_.begin();
        while(map_.size() > capacity_) {
            auto last = lru_.end();
            --last;
            map_.erase(last->first);
            lru_.pop_back();
        }
    }

    size_t capacity_;
    std::list<std::pair<std::string, JointWorldVecPtr>> lru_;
    std::unordered_map<std::string, std::list<std::pair<std::string, JointWorldVecPtr>>::iterator> map_;
};

static CloudPointVecPtr loadColorCloudPly(const fs::path &path) {
    if(path.empty()) {
        return {};
    }
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs.is_open()) {
        return {};
    }

    struct PropertySpec {
        enum class Type { Float32, UChar };
        Type        type;
        std::string name;
    };

    bool                      binaryLittleEndian = false;
    bool                      headerDone         = false;
    bool                      inVertexElement    = false;
    uint64_t                  vertexCount        = 0;
    std::vector<PropertySpec> props;
    std::string               line;
    while(std::getline(ifs, line)) {
        if(!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if(line == "format binary_little_endian 1.0") {
            binaryLittleEndian = true;
            continue;
        }
        if(line.rfind("element ", 0) == 0) {
            inVertexElement = false;
            std::istringstream iss(line);
            std::string kw;
            std::string elemName;
            iss >> kw >> elemName;
            if(elemName == "vertex") {
                iss >> vertexCount;
                inVertexElement = true;
            }
            continue;
        }
        if(inVertexElement && line.rfind("property ", 0) == 0) {
            std::istringstream iss(line);
            std::string kw;
            std::string typeName;
            std::string propName;
            iss >> kw >> typeName >> propName;
            if(typeName == "float") {
                props.push_back(PropertySpec{ PropertySpec::Type::Float32, propName });
            }
            else if(typeName == "uchar") {
                props.push_back(PropertySpec{ PropertySpec::Type::UChar, propName });
            }
            else {
                return {};
            }
            continue;
        }
        if(line == "end_header") {
            headerDone = true;
            break;
        }
    }
    if(!headerDone || !binaryLittleEndian || vertexCount == 0 || props.empty()) {
        return {};
    }

    int idxX = -1;
    int idxY = -1;
    int idxZ = -1;
    int idxR = -1;
    int idxG = -1;
    int idxB = -1;
    for(size_t i = 0; i < props.size(); ++i) {
        if(props[i].name == "x" && props[i].type == PropertySpec::Type::Float32) {
            idxX = static_cast<int>(i);
        }
        else if(props[i].name == "y" && props[i].type == PropertySpec::Type::Float32) {
            idxY = static_cast<int>(i);
        }
        else if(props[i].name == "z" && props[i].type == PropertySpec::Type::Float32) {
            idxZ = static_cast<int>(i);
        }
        else if(props[i].name == "red" && props[i].type == PropertySpec::Type::UChar) {
            idxR = static_cast<int>(i);
        }
        else if(props[i].name == "green" && props[i].type == PropertySpec::Type::UChar) {
            idxG = static_cast<int>(i);
        }
        else if(props[i].name == "blue" && props[i].type == PropertySpec::Type::UChar) {
            idxB = static_cast<int>(i);
        }
        else {
            return {};
        }
    }
    if(idxX < 0 || idxY < 0 || idxZ < 0) {
        return {};
    }

    const bool hasColor = idxR >= 0 && idxG >= 0 && idxB >= 0;
    auto       cloud    = std::make_shared<CloudPointVec>();
    cloud->reserve(static_cast<size_t>(vertexCount));
    for(uint64_t i = 0; i < vertexCount; ++i) {
        float         x = 0.0f;
        float         y = 0.0f;
        float         z = 0.0f;
        unsigned char r = 255;
        unsigned char g = 255;
        unsigned char b = 255;
        for(size_t pi = 0; pi < props.size(); ++pi) {
            if(props[pi].type == PropertySpec::Type::Float32) {
                float v = 0.0f;
                if(!ifs.read(reinterpret_cast<char *>(&v), sizeof(float))) {
                    return {};
                }
                if(static_cast<int>(pi) == idxX) {
                    x = v;
                }
                else if(static_cast<int>(pi) == idxY) {
                    y = v;
                }
                else if(static_cast<int>(pi) == idxZ) {
                    z = v;
                }
            }
            else {
                unsigned char v = 0;
                if(!ifs.read(reinterpret_cast<char *>(&v), sizeof(unsigned char))) {
                    return {};
                }
                if(static_cast<int>(pi) == idxR) {
                    r = v;
                }
                else if(static_cast<int>(pi) == idxG) {
                    g = v;
                }
                else if(static_cast<int>(pi) == idxB) {
                    b = v;
                }
            }
        }
        if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            continue;
        }
        CloudPoint cp;
        cp.p        = cv::Vec3f(x, y, z);
        cp.c        = cv::Vec3b(b, g, r);
        cp.hasColor = hasColor;
        cloud->push_back(cp);
    }
    return cloud;
}

static cv::Mat renderPointCloudCanvas(const std::vector<CloudPoint> &pts,
                                      const std::vector<cv::Vec3f> &joints,
                                      const ViewerViewState &viewState,
                                      const cv::Size &size,
                                      bool differentColor) {
    cv::Mat canvas(size.height, size.width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<float> zbuf(static_cast<size_t>(size.width) * static_cast<size_t>(size.height), std::numeric_limits<float>::infinity());
    cv::Vec3f right, up, forward, camPos;
    computeCameraBasis(viewState, right, up, forward, camPos);
    const float fx = 900.0f;
    const float fy = 900.0f;
    const float cx = static_cast<float>(size.width) * 0.5f;
    const float cy = static_cast<float>(size.height) * 0.5f;
    const cv::Vec3b defaultColor = differentColor ? cv::Vec3b(220, 220, 255) : cv::Vec3b(255, 255, 255);
    for(const auto &pt: pts) {
        const cv::Vec3f pw = pt.p;
        const cv::Vec3f v = pw - camPos;
        const float xc = v.dot(right);
        const float yc = v.dot(up);
        const float zc = v.dot(forward);
        if(zc <= 0.05f) {
            continue;
        }
        const int u = static_cast<int>(fx * (xc / zc) + cx);
        const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
        if(u < 0 || u >= size.width || vpx < 0 || vpx >= size.height) {
            continue;
        }
        const size_t idx = static_cast<size_t>(vpx) * static_cast<size_t>(size.width) + static_cast<size_t>(u);
        if(zc < zbuf[idx]) {
            zbuf[idx] = zc;
            canvas.at<cv::Vec3b>(vpx, u) = pt.hasColor ? pt.c : defaultColor;
        }
    }
    for(const auto &jp: joints) {
        const cv::Vec3f v = jp - camPos;
        const float xc = v.dot(right);
        const float yc = v.dot(up);
        const float zc = v.dot(forward);
        if(zc <= 0.05f) {
            continue;
        }
        const int u = static_cast<int>(fx * (xc / zc) + cx);
        const int vpx = static_cast<int>(fy * (-yc / zc) + cy);
        if(u < 0 || u >= size.width || vpx < 0 || vpx >= size.height) {
            continue;
        }
        const int radius = 2;
        cv::circle(canvas, cv::Point(u, vpx), radius, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, cv::Point(u, vpx), radius + 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    return canvas;
}

static cv::Vec3f medoidPoint(const std::vector<cv::Vec3f> &pts) {
    if(pts.empty()) {
        return cv::Vec3f(0, 0, 0);
    }
    if(pts.size() == 1) {
        return pts[0];
    }
    size_t bestIdx = 0;
    float bestScore = std::numeric_limits<float>::infinity();
    for(size_t i = 0; i < pts.size(); i++) {
        float s = 0.0f;
        for(size_t j = 0; j < pts.size(); j++) {
            const cv::Vec3f d = pts[i] - pts[j];
            s += std::sqrt(d.dot(d));
        }
        if(s < bestScore) {
            bestScore = s;
            bestIdx = i;
        }
    }
    return pts[bestIdx];
}

class DatasetViewer {
public:
    explicit DatasetViewer(AppConfig cfg, const std::atomic_bool *cancel)
        : cfg_(std::move(cfg)), cancel_(cancel) {}

    ~DatasetViewer() {
        stopPreloadWorker();
    }

    int run() {
        std::string last = "";
        for(int i = 0; i < 8; i++) {
            const std::string s = promptTextDialogBestEffort("Viewer", "Input data root", last);
            if(s.empty()) {
                return 0;
            }
            last = s;
            const fs::path root = fs::path(trimString(s));
            if(root.empty() || !fs::exists(root) || !fs::is_directory(root)) {
                continue;
            }
            dataRoot_ = root;
            subjects_ = scanSubjects(dataRoot_);
            break;
        }
        if(dataRoot_.empty() || !fs::exists(dataRoot_) || !fs::is_directory(dataRoot_)) {
            return 1;
        }

        winName_ = "Viewer";
        cv::namedWindow(winName_, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName_, 1600, 900);
        mainMouseCtx_.ui = &ms_;
        mainMouseCtx_.pc = &pcMouseCtx_;
        cv::setMouseCallback(winName_, mouseCallbackMain, &mainMouseCtx_);
        pcMouseCtx_.view = &pcView_;
        pcMouseCtx_.allow = &pcAllowMouse_;

        bool running = true;
        while(running) {
            if(cancel_ && cancel_->load()) {
                break;
            }
            const int key = cv::waitKeyEx(10);
            if(key > 0) {
                if(isCtrlModifierKeyEvent(key)) {
                    g_viewerCtrlShortcutListening = true;
                }
                else if(g_viewerCtrlShortcutListening && isCtrlReleaseKeyEvent(key)) {
                    g_viewerCtrlShortcutListening = false;
                }
            }
            if(key == 27) {
                break;
            }
            if(key > 0) {
                const bool ctrlFromMask = ((key & 0x20000) != 0) || ((key & 0x04000000) != 0);
                const bool ctrlHeld = g_viewerCtrlShortcutListening || ctrlFromMask;
                if(ctrlHeld && (dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud)) {
                    bool zoomed = false;
                    if(isCtrlZoomInKeyEvent(key)) {
                        pcView_.distance = std::min(20.0f, std::max(0.2f, pcView_.distance * 0.9f));
                        zoomed = true;
                    }
                    else if(isCtrlZoomOutKeyEvent(key)) {
                        pcView_.distance = std::min(20.0f, std::max(0.2f, pcView_.distance * 1.1f));
                        zoomed = true;
                    }
                    if(zoomed) {
                        invalidatePointCloudPreload();
                    }
                }
            }
            const cv::Size canvasSize = currentWindowSize();
            cv::Mat ui(canvasSize.height, canvasSize.width, CV_8UC3, cv::Scalar(20, 20, 20));

            drawViewerPage(ui, key);

            cv::imshow(winName_, ui);

            if(shouldExit_) {
                running = false;
            }
        }
        cv::destroyWindow(winName_);
        return 0;
    }

private:
    cv::Size preloadCanvasSize() const {
        if(mainRect_.width > 64 && mainRect_.height > 64) {
            return cv::Size(mainRect_.width, mainRect_.height);
        }
        return cv::Size(1280, 720);
    }

    fs::path selectedTaskDir() const {
        if(selectedSubject_.empty() || selectedTask_.empty()) {
            return fs::path();
        }
        return dataRoot_ / selectedSubject_ / selectedTask_;
    }

    fs::path selectedDataDir() const {
        fs::path taskDir = selectedTaskDir();
        if(taskDir.empty()) {
            return fs::path();
        }
        if(selectedEpisode_.empty()) {
            return taskDir;
        }
        return taskDir / selectedEpisode_;
    }

    bool selectedTaskHasEpisodes() const {
        const fs::path taskDir = selectedTaskDir();
        if(taskDir.empty() || !fs::exists(taskDir) || !fs::is_directory(taskDir)) {
            return false;
        }
        for(const auto &entry: fs::directory_iterator(taskDir)) {
            if(entry.is_directory() && isEpisodeDirName(entry.path().filename().string())) {
                return true;
            }
        }
        return false;
    }

    bool sourceSupportsType(const ViewerSource &source, ViewerDataType type) const {
        switch(type) {
        case ViewerDataType::RGB:
            return source.hasRgb;
        case ViewerDataType::Depth:
            return source.kind == ViewerSourceKind::Multiview && source.hasDepth;
        case ViewerDataType::IR:
            return source.kind == ViewerSourceKind::Multiview && source.hasIr;
        case ViewerDataType::PointCloud:
            return source.kind == ViewerSourceKind::Multiview && source.hasDepth;
        case ViewerDataType::ColorCloud:
            return source.kind == ViewerSourceKind::Multiview && source.hasDepth && source.hasRgb;
        }
        return false;
    }

    bool sourceToggleEnabled(const ViewerSource &source) const {
        if(source.kind == ViewerSourceKind::Fisheye) {
            return source.hasRgb && (dataType_ == ViewerDataType::RGB || dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud);
        }
        return sourceSupportsType(source, dataType_);
    }

    std::vector<const ViewerSource *> activeSourcesForType(ViewerDataType type) const {
        std::vector<const ViewerSource *> out;
        out.reserve(sources_.size());
        for(const auto &source : sources_) {
            if(!source.visible || !sourceSupportsType(source, type)) {
                continue;
            }
            out.push_back(&source);
        }
        return out;
    }

    std::vector<const ViewerSource *> activeMultiviewSourcesForType(ViewerDataType type) const {
        std::vector<const ViewerSource *> out;
        out.reserve(cameras_.size());
        for(const auto &source : sources_) {
            if(source.kind != ViewerSourceKind::Multiview || !source.visible || !sourceSupportsType(source, type)) {
                continue;
            }
            out.push_back(&source);
        }
        return out;
    }

    std::string activeMultiviewSignature(ViewerDataType type) const {
        std::vector<std::string> ids;
        for(const auto &source : sources_) {
            if(source.kind != ViewerSourceKind::Multiview || !source.visible || !sourceSupportsType(source, type)) {
                continue;
            }
            ids.push_back(source.sourceId);
        }
        std::sort(ids.begin(), ids.end());
        std::ostringstream oss;
        for(size_t i = 0; i < ids.size(); ++i) {
            if(i > 0) {
                oss << ",";
            }
            oss << ids[i];
        }
        return oss.str();
    }

    std::string resolvedHeadPoseCamera() const {
        const std::string input = sanitizeCameraIdInput(trimWhitespace(headPoseCamInput_));
        if(input.empty()) {
            return "";
        }
        for(const auto &cam : cameras_) {
            if(cameraKeyEquivalent(cam, input)) {
                return cam;
            }
        }
        return "";
    }

    std::string headCamPoseSpec() const {
        if(!headCamPoseAvailable_) {
            return "off";
        }
        const std::string resolved = resolvedHeadPoseCamera();
        if(!resolved.empty()) {
            const fs::path jsonPath = headPoseJsonPathForCamera(resolved);
            return jsonPath.empty() ? ("missing:" + resolved) : ("cam:" + resolved);
        }
        if(headPoseCamInput_.empty()) {
            return "idle";
        }
        return "invalid:" + sanitizeCameraIdInput(headPoseCamInput_);
    }

    bool shouldUseHeadPoseForCamera(const std::string &cam) const {
        if(!headCamPoseAvailable_) {
            return false;
        }
        const std::string resolved = resolvedHeadPoseCamera();
        return !resolved.empty() && cameraKeyEquivalent(resolved, cam) && !headPoseJsonPathForCamera(resolved).empty();
    }

    void refreshHeadCamPoseAvailability(const fs::path &dataDir) {
        headCamPoseAvailable_ = false;
        headCamPoseLoadedJsonPath_.clear();
        headPoseCamFieldActive_ = false;
        headPoseCamFieldRect_ = cv::Rect();
        {
            std::lock_guard<std::mutex> lock(headCamPoseMtx_);
            headCamPoseFrameCache_.clear();
        }
        if(!fs::exists(dataDir) || !fs::is_directory(dataDir)) {
            return;
        }
        for(const auto &entry : fs::directory_iterator(dataDir)) {
            if(!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if(name.rfind("ego_extrinsics_cam", 0) == 0 && entry.path().extension() == ".json") {
                headCamPoseAvailable_ = true;
                break;
            }
        }
    }

    fs::path headPoseJsonPathForCamera(const std::string &cam) const {
        const fs::path dataDir = selectedDataDir();
        if(!fs::exists(dataDir) || !fs::is_directory(dataDir)) {
            return {};
        }
        std::vector<std::string> candidates;
        const std::string stripped = stripLeadingZeros(cam);
        auto pushCandidate = [&](const std::string &value) {
            if(value.empty()) {
                return;
            }
            if(std::find(candidates.begin(), candidates.end(), value) == candidates.end()) {
                candidates.push_back(value);
            }
        };
        pushCandidate(cam);
        pushCandidate(stripped);
        if(isDigits(cam)) {
            pushCandidate(padLeftZeros(stripped, 2));
            pushCandidate(padLeftZeros(stripped, 3));
            pushCandidate(padLeftZeros(stripped, 4));
        }
        for(const auto &candidate : candidates) {
            const fs::path jsonPath = dataDir / ("ego_extrinsics_cam" + candidate + ".json");
            if(fs::exists(jsonPath) && fs::is_regular_file(jsonPath)) {
                return jsonPath;
            }
        }
        return {};
    }

    std::string referenceCameraForHeadPose(const std::string &egoCam) const {
        for(const auto &cam : cameras_) {
            if(!cameraKeyEquivalent(cam, egoCam)) {
                return cam;
            }
        }
        return "";
    }

    std::optional<ExtrinsicCamToWorld> loadHeadPoseExtrinsicForFrame(const std::string &cam, int frameIdx) const {
        if(!headCamPoseAvailable_ || frameIdx < 0) {
            return std::nullopt;
        }
        const fs::path jsonPath = headPoseJsonPathForCamera(cam);
        if(jsonPath.empty()) {
            return std::nullopt;
        }
        const std::string referenceCam = referenceCameraForHeadPose(cam);
        if(referenceCam.empty()) {
            return std::nullopt;
        }
        const ExtrinsicCamToWorld worldFromReference = staticExtrinsicForCamera(referenceCam);
        if(!worldFromReference.valid) {
            return std::nullopt;
        }
        {
            std::lock_guard<std::mutex> lock(headCamPoseMtx_);
            if(headCamPoseLoadedJsonPath_ != jsonPath) {
                headCamPoseLoadedJsonPath_ = jsonPath;
                headCamPoseFrameCache_.clear();
                const auto parsed = loadEgoExtrinsicsCamToWorldJson(jsonPath, worldFromReference);
                for(const auto &[parsedFrameIdx, parsedPose] : parsed) {
                    headCamPoseFrameCache_[parsedFrameIdx] = parsedPose;
                }
            }
            auto it = headCamPoseFrameCache_.find(frameIdx);
            if(it != headCamPoseFrameCache_.end()) {
                return it->second;
            }
        }
        {
            std::lock_guard<std::mutex> lock(headCamPoseMtx_);
            headCamPoseFrameCache_[frameIdx] = std::nullopt;
        }
        return std::nullopt;
    }

    ExtrinsicCamToWorld staticExtrinsicForCamera(const std::string &cam) const {
        ExtrinsicCamToWorld ex;
        if(const auto *ep = findByCamKeyVariants(extrinsics_, cam); ep && ep->valid) {
            ex = *ep;
        }
        else {
            ex.valid = true;
            ex.R = cv::Matx33f::eye();
            ex.t = cv::Vec3f(0, 0, 0);
        }
        return ex;
    }

    ExtrinsicCamToWorld effectiveExtrinsicForFrame(const std::string &cam, int frameIdx) const {
        if(shouldUseHeadPoseForCamera(cam)) {
            const auto dynamicPose = loadHeadPoseExtrinsicForFrame(cam, frameIdx);
            if(dynamicPose && dynamicPose->valid) {
                return *dynamicPose;
            }
        }
        return staticExtrinsicForCamera(cam);
    }

    bool canUseSavedColorCloud() const {
        if(!headCamPoseAvailable_) {
            return true;
        }
        const std::string resolved = resolvedHeadPoseCamera();
        if(resolved.empty()) {
            return true;
        }
        for(const auto &source : sources_) {
            if(source.kind != ViewerSourceKind::Multiview || !source.visible) {
                continue;
            }
            if(!sourceSupportsType(source, ViewerDataType::ColorCloud)) {
                continue;
            }
            if(cameraKeyEquivalent(source.camKey, resolved)) {
                return false;
            }
        }
        return true;
    }

    void handleHeadPoseFieldInput(int key) {
        if(!headCamPoseAvailable_ || !headPoseCamFieldActive_ || key <= 0) {
            return;
        }
        const int k = key & 0xff;
        if(k == 13 || k == 10) {
            headPoseCamFieldActive_ = false;
            return;
        }
        std::string next = headPoseCamInput_;
        handleTextInput(next, key);
        next = sanitizeCameraIdInput(next);
        if(next == headPoseCamInput_) {
            return;
        }
        invalidatePointCloudPreload();
        headPoseCamInput_ = next;
        std::lock_guard<std::mutex> lock(headCamPoseMtx_);
        headCamPoseFrameCache_.clear();
    }

    void stopPreloadWorker() {
        preloadStop_.store(true);
        if(preloadThread_.joinable()) {
            preloadThread_.join();
        }
        preloadStop_.store(false);
    }

    void clearPreloaded() {
        std::lock_guard<std::mutex> lock(preloadMtx_);
        preDepthCanvas_.clear();
        preDepthCanvasL_.clear();
        prePointCloudCanvas_.clear();
        prePointCloudCanvasL_.clear();
        preColorCloudCanvas_.clear();
        preColorCloudCanvasL_.clear();
        preloadReadyDepth_ = false;
        preloadReadyPointCloud_ = false;
        preloadReadyColorCloud_ = false;
    }

    void invalidatePointCloudPreload() {
        stopPreloadWorker();
        clearPreloaded();
        preloadSpec_.clear();
        preloadStartFrame_ = 0;
        preloadSpan_ = 0;
    }

    cv::Mat loadRgbFrameNoCache(const std::string &cam, int frameIdx) const {
        const fs::path dir = selectedDataDir() / cam / "RGB";
        const fs::path p = findFrameFile(dir, frameIdx, { ".jpg", ".jpeg", ".png" });
        if(p.empty()) {
            return cv::Mat();
        }
        return cv::imread(p.string(), cv::IMREAD_COLOR);
    }

    cv::Mat loadDepthFrameNoCache(const std::string &cam, int frameIdx) const {
        const fs::path dir = selectedDataDir() / cam / "Depth";
        const fs::path p = findFrameFile(dir, frameIdx, { ".png" });
        if(p.empty()) {
            return cv::Mat();
        }
        cv::Mat m = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
        if(!m.empty() && m.type() != CV_16UC1) {
            if(m.type() == CV_8UC1) {
                cv::Mat tmp;
                m.convertTo(tmp, CV_16UC1);
                m = tmp;
            }
            else {
                m.release();
            }
        }
        return m;
    }

    std::vector<cv::Point2f> labelsForFrameCamAt(int frameIdx, const std::string &cam) const {
        auto itF = labelsByFrame_.find(frameIdx);
        if(itF == labelsByFrame_.end()) {
            return {};
        }
        const auto *p = findByCamKeyVariants(itF->second, cam);
        if(!p) {
            return {};
        }
        return *p;
    }

    void renderDepthGridFramePair(int frameIdx, cv::Mat &outNo, cv::Mat *outWith) const {
        std::vector<std::pair<std::string, cv::Mat>> frames;
        std::vector<std::pair<std::string, cv::Mat>> framesL;
        frames.reserve(cameras_.size());
        if(outWith) {
            framesL.reserve(cameras_.size());
        }
        for(const auto &cam: cameras_) {
            const cv::Mat depth16 = loadDepthFrameNoCache(cam, frameIdx);
            cv::Mat img = depth16ToYellowBlue(depth16, cfg_.maxDepth, 1.0f);
            frames.emplace_back(cam, img);
            if(outWith) {
                cv::Mat imgL = img.clone();
                const cv::Mat rgb = loadRgbFrameNoCache(cam, frameIdx);
                if(!rgb.empty()) {
                    AlignMapCache cache;
                    const auto *rp = findByCamKeyVariants(taskCamParams_.rgbToDepth, cam);
                    const bool useLegacyMap = !isAlignedDepthToRgb(depth16, rgb) && rp && rp->valid;
                    if(useLegacyMap) {
                        buildDepthToColorMap(cache, frameIdx, depth16, rgb.cols, rgb.rows, *rp, cfg_.maxDepth, 1.0f);
                    }
                    const auto pts = labelsForFrameCamAt(frameIdx, cam);
                    for(const auto &p: pts) {
                        if(!(p.x >= 0.0f && p.y >= 0.0f)) {
                            continue;
                        }
                        int dx = static_cast<int>(std::lround(p.x));
                        int dy = static_cast<int>(std::lround(p.y));
                        const bool ok = isAlignedDepthToRgb(depth16, rgb) ? (dx >= 0 && dx < depth16.cols && dy >= 0 && dy < depth16.rows)
                                                                          : lookupDepthForRgb(cache, dx, dy, dx, dy);
                        if(ok && dx >= 0 && dx < depth16.cols && dy >= 0 && dy < depth16.rows) {
                            cv::circle(imgL, cv::Point(dx, dy), 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
                        }
                    }
                }
                framesL.emplace_back(cam, imgL);
            }
        }
        const cv::Size sz = preloadCanvasSize();
        outNo = cv::Mat(sz.height, sz.width, CV_8UC3, cv::Scalar(20, 20, 20));
        drawGridImages(outNo, cv::Rect(0, 0, sz.width, sz.height), frames);
        if(outWith) {
            *outWith = cv::Mat(sz.height, sz.width, CV_8UC3, cv::Scalar(20, 20, 20));
            drawGridImages(*outWith, cv::Rect(0, 0, sz.width, sz.height), framesL);
        }
    }

    std::vector<cv::Vec3f> clusterCandidatesMedoids(const std::vector<cv::Vec3f> &candidates, float mergeR) const {
        if(candidates.empty()) {
            return {};
        }
        std::vector<std::vector<cv::Vec3f>> clusters;
        clusters.reserve(candidates.size());
        for(const auto &p: candidates) {
            bool assigned = false;
            for(auto &cl: clusters) {
                if(cl.empty()) {
                    continue;
                }
                const cv::Vec3f d = p - cl.front();
                if(std::sqrt(d.dot(d)) <= mergeR) {
                    cl.push_back(p);
                    assigned = true;
                    break;
                }
            }
            if(!assigned) {
                clusters.push_back({ p });
            }
        }
        std::vector<cv::Vec3f> out;
        out.reserve(clusters.size());
        for(const auto &cl: clusters) {
            if(cl.empty()) {
                continue;
            }
            out.push_back(medoidPoint(cl));
        }
        return out;
    }

    void renderPointCloudFramePair(int frameIdx, bool colorCloud, cv::Mat &outNo, cv::Mat *outWith) const {
        std::vector<CloudPoint> cloud;
        const bool allowSavedColorCloud = colorCloud && canUseSavedColorCloud();
        const auto savedColorCloud = allowSavedColorCloud ? loadSavedColorCloudFrame(frameIdx) : CloudPointVecPtr{};
        const bool useSavedColorCloud = savedColorCloud && !savedColorCloud->empty();
        if(useSavedColorCloud) {
            cloud = *savedColorCloud;
        }
        const int step = std::max(1, cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 2);
        std::vector<cv::Vec3f> jointCandidates;
        const auto activeSources = activeMultiviewSourcesForType(colorCloud ? ViewerDataType::ColorCloud : ViewerDataType::PointCloud);
        for(const auto *source : activeSources) {
            if(!source) {
                continue;
            }
            if(useSavedColorCloud && !outWith) {
                break;
            }
            const std::string &cam = source->camKey;
            const cv::Mat depth16 = loadDepthFrameNoCache(cam, frameIdx);
            if(depth16.empty() || depth16.type() != CV_16UC1) {
                continue;
            }
            const ExtrinsicCamToWorld ex = effectiveExtrinsicForFrame(cam, frameIdx);

            cv::Mat rgb;
            AlignMapCache cache;
            AlignMapCache *align = nullptr;
            const OBCameraIntrinsic *pointIntr = nullptr;
            if(colorCloud || outWith) {
                rgb = loadRgbFrameNoCache(cam, frameIdx);
                pointIntr = pointCloudIntrinsicForDepth(cam, depth16, rgb);
                const auto *rp = findByCamKeyVariants(taskCamParams_.rgbToDepth, cam);
                if(!rgb.empty() && !isAlignedDepthToRgb(depth16, rgb) && rp && rp->valid) {
                    buildDepthToColorMap(cache, frameIdx, depth16, rgb.cols, rgb.rows, *rp, cfg_.maxDepth, 1.0f);
                    align = &cache;
                }
            }
            if(!pointIntr) {
                pointIntr = pointCloudIntrinsicForDepth(cam, depth16, rgb);
            }
            if(!pointIntr || pointIntr->fx <= 0.0f || pointIntr->fy <= 0.0f) {
                continue;
            }

            const float fx = pointIntr->fx;
            const float fy = pointIntr->fy;
            const float cx = pointIntr->cx;
            const float cy = pointIntr->cy;
            const float sMm = 1.0f;
            if(useSavedColorCloud) {
                if(outWith && !rgb.empty()) {
                    const auto pts2d = labelsForFrameCamAt(frameIdx, cam);
                    for(const auto &p2: pts2d) {
                        if(!(p2.x >= 0.0f && p2.y >= 0.0f)) {
                            continue;
                        }
                        int dx = static_cast<int>(std::lround(p2.x));
                        int dy = static_cast<int>(std::lround(p2.y));
                        const bool ok = isAlignedDepthToRgb(depth16, rgb) ? (dx >= 0 && dx < depth16.cols && dy >= 0 && dy < depth16.rows)
                                                                          : (align && lookupDepthForRgb(*align, dx, dy, dx, dy));
                        if(!ok) {
                            continue;
                        }
                        if(dx < 0 || dx >= depth16.cols || dy < 0 || dy >= depth16.rows) {
                            continue;
                        }
                        const uint16_t d = depth16.at<uint16_t>(dy, dx);
                        if(d == 0) {
                            continue;
                        }
                        const float z = (static_cast<float>(d) * sMm) * 0.001f;
                        if(!(z >= 0.2f && z <= cfg_.maxDepth)) {
                            continue;
                        }
                        const float xCam = (static_cast<float>(dx) - cx) * z / fx;
                        const float yCam = (static_cast<float>(dy) - cy) * z / fy;
                        const cv::Vec3f pw = ex.R * cv::Vec3f(xCam, yCam, z) + ex.t;
                        if(std::isfinite(pw[0]) && std::isfinite(pw[1]) && std::isfinite(pw[2])) {
                            jointCandidates.push_back(pw);
                        }
                    }
                }
                continue;
            }

            for(int y = 0; y < depth16.rows; y += step) {
                const uint16_t *row = depth16.ptr<uint16_t>(y);
                for(int x = 0; x < depth16.cols; x += step) {
                    const uint16_t d = row[x];
                    if(d == 0) {
                        continue;
                    }
                    const float depthMm = static_cast<float>(d) * sMm;
                    const float z = depthMm * 0.001f;
                    if(!(z >= 0.2f && z <= cfg_.maxDepth)) {
                        continue;
                    }
                    const float xCam = (static_cast<float>(x) - cx) * z / fx;
                    const float yCam = (static_cast<float>(y) - cy) * z / fy;
                    const cv::Vec3f pCam(xCam, yCam, z);
                    const cv::Vec3f pw = ex.R * pCam + ex.t;
                    CloudPoint cp;
                    cp.p = pw;
                    if(colorCloud && !rgb.empty()) {
                        if(isAlignedDepthToRgb(depth16, rgb)) {
                            cp.c = rgb.at<cv::Vec3b>(y, x);
                            cp.hasColor = true;
                        }
                        else if(align && align->depthW == depth16.cols && align->depthH == depth16.rows) {
                            const int32_t packedUv = align->depthToColor[static_cast<size_t>(y) * static_cast<size_t>(align->depthW) + static_cast<size_t>(x)];
                            int u = -1, v = -1;
                            if(unpackXY(packedUv, u, v) && u >= 0 && u < rgb.cols && v >= 0 && v < rgb.rows) {
                                cp.c = rgb.at<cv::Vec3b>(v, u);
                                cp.hasColor = true;
                            }
                        }
                    }
                    cloud.push_back(cp);
                }
            }

            if(outWith && !rgb.empty()) {
                const auto pts2d = labelsForFrameCamAt(frameIdx, cam);
                for(const auto &p2: pts2d) {
                    if(!(p2.x >= 0.0f && p2.y >= 0.0f)) {
                        continue;
                    }
                    int dx = static_cast<int>(std::lround(p2.x));
                    int dy = static_cast<int>(std::lround(p2.y));
                    const bool ok = isAlignedDepthToRgb(depth16, rgb) ? (dx >= 0 && dx < depth16.cols && dy >= 0 && dy < depth16.rows)
                                                                      : (align && lookupDepthForRgb(*align, dx, dy, dx, dy));
                    if(!ok) {
                        continue;
                    }
                    if(dx < 0 || dx >= depth16.cols || dy < 0 || dy >= depth16.rows) {
                        continue;
                    }
                    const uint16_t d = depth16.at<uint16_t>(dy, dx);
                    if(d == 0) {
                        continue;
                    }
                    const float z = (static_cast<float>(d) * sMm) * 0.001f;
                    if(!(z >= 0.2f && z <= cfg_.maxDepth)) {
                        continue;
                    }
                    const float xCam = (static_cast<float>(dx) - cx) * z / fx;
                    const float yCam = (static_cast<float>(dy) - cy) * z / fy;
                    const cv::Vec3f pw = ex.R * cv::Vec3f(xCam, yCam, z) + ex.t;
                    if(std::isfinite(pw[0]) && std::isfinite(pw[1]) && std::isfinite(pw[2])) {
                        jointCandidates.push_back(pw);
                    }
                }
            }
        }

        const cv::Size sz = preloadCanvasSize();
        outNo = renderPointCloudCanvas(cloud, {}, preloadView_, sz, cfg_.differentColor);
        if(outWith) {
            const auto joints = clusterCandidatesMedoids(jointCandidates, 0.008f);
            *outWith = renderPointCloudCanvas(cloud, joints, preloadView_, sz, cfg_.differentColor);
        }
    }

    int wrapFrameIndex(int frameIdx) const {
        if(totalFrames_ <= 0) {
            return 0;
        }
        int v = frameIdx;
        while(v < 0) {
            v += totalFrames_;
        }
        while(v >= totalFrames_) {
            v -= totalFrames_;
        }
        return v;
    }

    std::string pointCloudPreloadSpec() const {
        if(dataType_ != ViewerDataType::PointCloud && dataType_ != ViewerDataType::ColorCloud) {
            return "";
        }
        auto q = [](float v) {
            return static_cast<long long>(std::llround(static_cast<double>(v) * 1000.0));
        };
        const cv::Size preloadSz = preloadCanvasSize();
        std::ostringstream oss;
        oss << (dataType_ == ViewerDataType::ColorCloud ? "cc" : "pc");
        oss << "|labels=" << (showLabels_ && labelsAvailable_ ? 1 : 0);
        oss << "|src=" << activeMultiviewSignature(dataType_ == ViewerDataType::ColorCloud ? ViewerDataType::ColorCloud : ViewerDataType::PointCloud);
        oss << "|w=" << preloadSz.width;
        oss << "|h=" << preloadSz.height;
        oss << "|yaw=" << q(pcView_.yawRad);
        oss << "|pitch=" << q(pcView_.pitchRad);
        oss << "|dist=" << q(pcView_.distance);
        oss << "|tx=" << q(pcView_.target[0]);
        oss << "|ty=" << q(pcView_.target[1]);
        oss << "|tz=" << q(pcView_.target[2]);
        oss << "|head=" << headCamPoseSpec();
        return oss.str();
    }

    bool isFrameCoveredByPreloadSpan(int frameIdx) const {
        if(totalFrames_ <= 0 || preloadSpan_ <= 0) {
            return false;
        }
        const int start = wrapFrameIndex(preloadStartFrame_);
        const int frame = wrapFrameIndex(frameIdx);
        const int forward = (frame - start + totalFrames_) % totalFrames_;
        return forward >= 0 && forward < preloadSpan_;
    }

    bool hasPreloadedCanvasForFrame(int frameIdx) const {
        if(frameIdx < 0) {
            return false;
        }
        const size_t idx = static_cast<size_t>(frameIdx);
        std::lock_guard<std::mutex> lock(preloadMtx_);
        if(dataType_ == ViewerDataType::PointCloud) {
            const auto &vec =
                (showLabels_ && labelsAvailable_ && idx < prePointCloudCanvasL_.size() && !prePointCloudCanvasL_[idx].empty()) ? prePointCloudCanvasL_ : prePointCloudCanvas_;
            return idx < vec.size() && !vec[idx].empty();
        }
        if(dataType_ == ViewerDataType::ColorCloud) {
            const auto &vec =
                (showLabels_ && labelsAvailable_ && idx < preColorCloudCanvasL_.size() && !preColorCloudCanvasL_[idx].empty()) ? preColorCloudCanvasL_ : preColorCloudCanvas_;
            return idx < vec.size() && !vec[idx].empty();
        }
        return false;
    }

    void startPreloadWorker() {
        stopPreloadWorker();
        clearPreloaded();
        if(totalFrames_ <= 0 || cameras_.empty() || selectedSubject_.empty() || selectedTask_.empty()) {
            return;
        }
        const bool preloadPointCloud = dataType_ == ViewerDataType::PointCloud;
        const bool preloadColorCloud = dataType_ == ViewerDataType::ColorCloud;
        if(!preloadPointCloud && !preloadColorCloud) {
            return;
        }
        const bool preloadLabels = showLabels_ && labelsAvailable_;
        preloadView_ = pcView_;
        preloadSpec_ = pointCloudPreloadSpec();
        preloadStartFrame_ = wrapFrameIndex(currentFrame_);
        preloadSpan_ = std::min(totalFrames_, playing_ ? 180 : 60);
        if(preloadPointCloud) {
            prePointCloudCanvas_.assign(static_cast<size_t>(totalFrames_), cv::Mat());
            if(preloadLabels) {
                prePointCloudCanvasL_.assign(static_cast<size_t>(totalFrames_), cv::Mat());
            }
        }
        if(preloadColorCloud) {
            preColorCloudCanvas_.assign(static_cast<size_t>(totalFrames_), cv::Mat());
            if(preloadLabels) {
                preColorCloudCanvasL_.assign(static_cast<size_t>(totalFrames_), cv::Mat());
            }
        }
        if(prePointCloudCanvas_.empty() && preColorCloudCanvas_.empty()) {
            return;
        }
        const int preloadStart = preloadStartFrame_;
        const int preloadCount = preloadSpan_;
        preloadThread_ = std::thread([this, preloadPointCloud, preloadColorCloud, preloadLabels, preloadStart, preloadCount]() {
            for(int offset = 0; offset < preloadCount; ++offset) {
                if(preloadStop_.load()) {
                    break;
                }
                const int f = wrapFrameIndex(preloadStart + offset);
                if(preloadPointCloud && f >= 0 && static_cast<size_t>(f) < prePointCloudCanvas_.size()) {
                    cv::Mat noL;
                    cv::Mat withL;
                    cv::Mat *pWith = preloadLabels ? &withL : nullptr;
                    renderPointCloudFramePair(f, false, noL, pWith);
                    {
                        std::lock_guard<std::mutex> lock(preloadMtx_);
                        prePointCloudCanvas_[static_cast<size_t>(f)] = std::move(noL);
                        if(pWith && static_cast<size_t>(f) < prePointCloudCanvasL_.size()) {
                            prePointCloudCanvasL_[static_cast<size_t>(f)] = std::move(withL);
                        }
                    }
                }
                if(preloadStop_.load()) {
                    break;
                }
                if(preloadColorCloud && f >= 0 && static_cast<size_t>(f) < preColorCloudCanvas_.size()) {
                    cv::Mat noL;
                    cv::Mat withL;
                    cv::Mat *pWith = preloadLabels ? &withL : nullptr;
                    renderPointCloudFramePair(f, true, noL, pWith);
                    {
                        std::lock_guard<std::mutex> lock(preloadMtx_);
                        preColorCloudCanvas_[static_cast<size_t>(f)] = std::move(noL);
                        if(pWith && static_cast<size_t>(f) < preColorCloudCanvasL_.size()) {
                            preColorCloudCanvasL_[static_cast<size_t>(f)] = std::move(withL);
                        }
                    }
                }
            }
            preloadReadyPointCloud_ = preloadPointCloud && !prePointCloudCanvas_.empty();
            preloadReadyColorCloud_ = preloadColorCloud && !preColorCloudCanvas_.empty();
        });
    }

    void ensurePointCloudPreload() {
        if(dataType_ != ViewerDataType::PointCloud && dataType_ != ViewerDataType::ColorCloud) {
            return;
        }
        if(cameras_.empty() || totalFrames_ <= 0) {
            return;
        }
        if(pcView_.rotating || pcView_.panning) {
            return;
        }
        const std::string spec = pointCloudPreloadSpec();
        const bool specChanged = spec != preloadSpec_;
        const bool spanMiss = !isFrameCoveredByPreloadSpan(currentFrame_);
        const bool canvasMissing = !hasPreloadedCanvasForFrame(currentFrame_);
        const auto now = std::chrono::steady_clock::now();
        if(!(specChanged || spanMiss || canvasMissing)) {
            return;
        }
        if(!specChanged && !spanMiss && canvasMissing && preloadThread_.joinable()) {
            return;
        }
        if(preloadLastRestart_.time_since_epoch().count() != 0 && now - preloadLastRestart_ < std::chrono::milliseconds(150)) {
            return;
        }
        preloadLastRestart_ = now;
        startPreloadWorker();
    }

    bool blitPreloadedToMain(cv::Mat &ui, const cv::Mat &src) const {
        if(src.empty() || mainRect_.width <= 1 || mainRect_.height <= 1) {
            return false;
        }
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(mainRect_.width, mainRect_.height), 0, 0, cv::INTER_AREA);
        resized.copyTo(ui(mainRect_));
        return true;
    }

    bool drawPreloadedIfAvailable(cv::Mat &ui) const {
        if(currentFrame_ < 0) {
            return false;
        }
        if(dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud) {
            if(pcView_.rotating || pcView_.panning) {
                return false;
            }
            if(pointCloudPreloadSpec() != preloadSpec_) {
                return false;
            }
        }
        const size_t idx = static_cast<size_t>(currentFrame_);
        if(dataType_ == ViewerDataType::PointCloud) {
            std::lock_guard<std::mutex> lock(preloadMtx_);
            const auto &vec = (showLabels_ && labelsAvailable_ && idx < prePointCloudCanvasL_.size() && !prePointCloudCanvasL_[idx].empty()) ? prePointCloudCanvasL_ : prePointCloudCanvas_;
            if(idx < vec.size() && !vec[idx].empty()) {
                return blitPreloadedToMain(ui, vec[idx]);
            }
        }
        if(dataType_ == ViewerDataType::ColorCloud) {
            std::lock_guard<std::mutex> lock(preloadMtx_);
            const auto &vec = (showLabels_ && labelsAvailable_ && idx < preColorCloudCanvasL_.size() && !preColorCloudCanvasL_[idx].empty()) ? preColorCloudCanvasL_ : preColorCloudCanvas_;
            if(idx < vec.size() && !vec[idx].empty()) {
                return blitPreloadedToMain(ui, vec[idx]);
            }
        }
        return false;
    }

    std::vector<const ViewerSource *> activeFisheyeOverlaySources() const {
        std::vector<const ViewerSource *> out;
        out.reserve(sources_.size());
        for(const auto &source : sources_) {
            if(source.kind != ViewerSourceKind::Fisheye || !source.visible || !source.hasRgb) {
                continue;
            }
            out.push_back(&source);
        }
        return out;
    }

    void drawFisheyeOverlay(cv::Mat &ui) {
        if(dataType_ != ViewerDataType::PointCloud && dataType_ != ViewerDataType::ColorCloud) {
            return;
        }
        const auto overlaySources = activeFisheyeOverlaySources();
        if(overlaySources.empty() || mainRect_.width <= 0 || mainRect_.height <= 0) {
            return;
        }

        const int maxTileW = std::min(220, std::max(120, mainRect_.width / 4));
        const int gap = 8;
        const int cols = (overlaySources.size() > 2) ? 2 : 1;
        int x = mainRect_.x + 10;
        int y = mainRect_.y + 10;
        int col = 0;
        int rowMaxH = 0;
        for(size_t order = 0; order < overlaySources.size(); ++order) {
            const auto *source = overlaySources[order];
            if(!source) {
                continue;
            }
            const cv::Mat frame = loadRgbFrame(*source, currentFrame_);
            if(frame.empty()) {
                continue;
            }
            const int tileW = maxTileW;
            const int tileH = std::max(72, static_cast<int>(static_cast<double>(frame.rows) * (static_cast<double>(tileW) / static_cast<double>(frame.cols))));
            const cv::Rect tile(x, y, std::min(tileW, mainRect_.x + mainRect_.width - x - 10), std::min(tileH + 28, mainRect_.y + mainRect_.height - y - 10));
            if(tile.width <= 40 || tile.height <= 40) {
                break;
            }
            cv::rectangle(ui, tile, cv::Scalar(15, 15, 15), cv::FILLED);
            cv::rectangle(ui, tile, cv::Scalar(120, 120, 120), 1);
            cv::putText(ui, source->displayName, cv::Point(tile.x + 8, tile.y + 18), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
            cv::Rect imageR(tile.x + 4, tile.y + 24, tile.width - 8, tile.height - 28);
            if(imageR.width > 8 && imageR.height > 8) {
                cv::Mat resized;
                cv::resize(frame, resized, imageR.size(), 0, 0, cv::INTER_AREA);
                resized.copyTo(ui(imageR));
            }
            rowMaxH = std::max(rowMaxH, tile.height);
            col++;
            if(col >= cols) {
                col = 0;
                x = mainRect_.x + 10;
                y += rowMaxH + gap;
                rowMaxH = 0;
            }
            else {
                x += tile.width + gap;
            }
        }
    }

    void drawPointCloudOverlays(cv::Mat &ui) {
        drawFisheyeOverlay(ui);
        cv::putText(ui, "LMB rotate  RMB pan  Ctrl+/- zoom", cv::Point(mainRect_.x + 12, mainRect_.y + 24), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
    }
    cv::Size currentWindowSize() const {
        cv::Rect r;
        try {
            r = cv::getWindowImageRect(winName_);
        }
        catch(...) {
            r = cv::Rect(0, 0, 1600, 900);
        }
        int w = std::max(640, r.width);
        int h = std::max(480, r.height);
        return cv::Size(w, h);
    }


    void layout(const cv::Mat &ui) {
        const int w = ui.cols;
        const int h = ui.rows;
        leftPanel_ = cv::Rect(0, 0, std::max(240, w / 6), h);
        const int topBarHeight = headCamPoseAvailable_ ? 154 : 110;
        topBar_ = cv::Rect(leftPanel_.width, 0, w - leftPanel_.width, topBarHeight);
        bottomBar_ = cv::Rect(leftPanel_.width, h - 70, w - leftPanel_.width, 70);
        mainRect_ = cv::Rect(leftPanel_.width, topBar_.height, w - leftPanel_.width, h - topBar_.height - bottomBar_.height);
    }

    void drawViewerPage(cv::Mat &ui, int key) {
        layout(ui);
        if(ms_.clicked && headPoseCamFieldActive_ && headPoseCamFieldRect_.area() > 0
           && !headPoseCamFieldRect_.contains(cv::Point(ms_.clickX, ms_.clickY))) {
            headPoseCamFieldActive_ = false;
        }
        drawLeftPanel(ui);
        drawTopBar(ui);
        handleHeadPoseFieldInput(key);
        drawBottomBar(ui);
        drawMain(ui);
        drawDropdownOverlay(ui);
    }

    void drawLeftPanel(cv::Mat &ui) {
        cv::rectangle(ui, leftPanel_, cv::Scalar(16, 16, 16), cv::FILLED);
        cv::rectangle(ui, leftPanel_, cv::Scalar(80, 80, 80), 1);
        cv::putText(ui, "Subjects", cv::Point(14, 30), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        const int contentTop = 50;
        const int rowH = 34;
        const int x0 = 10;
        const int w = leftPanel_.width - 20;
        const cv::Rect scrollArea(leftPanel_.x, contentTop, leftPanel_.width, leftPanel_.height - contentTop);
        if(ms_.wheelDelta != 0 && scrollArea.contains(cv::Point(ms_.x, ms_.y))) {
            leftScrollY_ += (ms_.wheelDelta > 0) ? -rowH : rowH;
            ms_.wheelDelta = 0;
        }

        int totalRows = 0;
        for(const auto &s: subjects_) {
            totalRows += 1;
            if(s.expanded) {
                for(const auto &t: s.tasks) {
                    totalRows += 1;
                    if(t.expanded) {
                        totalRows += static_cast<int>(t.episodes.size());
                    }
                }
            }
        }
        const int contentH = totalRows * rowH + 20;
        const int maxScroll = std::max(0, contentH - scrollArea.height);
        leftScrollY_ = std::max(0, std::min(maxScroll, leftScrollY_));

        int y = contentTop - leftScrollY_;
        for(auto &s: subjects_) {
            cv::Rect r(x0, y, w, rowH - 2);
            const bool selected = (selectedSubject_ == s.name && selectedTask_.empty() && selectedEpisode_.empty());
            const bool hover = r.contains(cv::Point(ms_.x, ms_.y));
            cv::Scalar bg = selected ? cv::Scalar(60, 80, 120) : (hover ? cv::Scalar(40, 40, 40) : cv::Scalar(22, 22, 22));
            cv::rectangle(ui, r, bg, cv::FILLED);
            cv::rectangle(ui, r, cv::Scalar(60, 60, 60), 1);
            const std::string prefix = s.expanded ? "v " : "> ";
            cv::putText(ui, prefix + s.name, cv::Point(r.x + 8, r.y + 23), cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
            if(ms_.clicked && r.contains(cv::Point(ms_.clickX, ms_.clickY))) {
                ms_.clicked = false;
                s.expanded = !s.expanded;
                selectedSubject_ = s.name;
                selectedTask_.clear();
                selectedEpisode_.clear();
                clearTaskState();
            }
            y += rowH;
            if(s.expanded) {
                for(auto &t: s.tasks) {
                    cv::Rect rt(x0 + 18, y, w - 18, rowH - 2);
                    const bool sel = (selectedSubject_ == s.name && selectedTask_ == t.name && selectedEpisode_.empty());
                    const bool hov = rt.contains(cv::Point(ms_.x, ms_.y));
                    cv::Scalar bg2 = sel ? cv::Scalar(80, 110, 160) : (hov ? cv::Scalar(36, 36, 36) : cv::Scalar(18, 18, 18));
                    cv::rectangle(ui, rt, bg2, cv::FILLED);
                    cv::rectangle(ui, rt, cv::Scalar(55, 55, 55), 1);
                    const std::string taskPrefix = t.episodes.empty() ? "- " : (t.expanded ? "v " : "> ");
                    cv::putText(ui, taskPrefix + t.name, cv::Point(rt.x + 8, rt.y + 23), cv::FONT_HERSHEY_DUPLEX, 0.60, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
                    if(ms_.clicked && rt.contains(cv::Point(ms_.clickX, ms_.clickY))) {
                        ms_.clicked = false;
                        selectedSubject_ = s.name;
                        selectedTask_ = t.name;
                        if(t.episodes.empty()) {
                            selectedEpisode_.clear();
                            loadSelectedTask();
                        }
                        else {
                            t.expanded = !t.expanded;
                            selectedEpisode_.clear();
                            clearTaskState();
                        }
                    }
                    y += rowH;
                    if(t.expanded) {
                        for(const auto &episode: t.episodes) {
                            cv::Rect re(x0 + 36, y, w - 36, rowH - 2);
                            const bool episodeSel = (selectedSubject_ == s.name && selectedTask_ == t.name && selectedEpisode_ == episode);
                            const bool episodeHov = re.contains(cv::Point(ms_.x, ms_.y));
                            cv::Scalar bg3 = episodeSel ? cv::Scalar(96, 132, 188) : (episodeHov ? cv::Scalar(32, 32, 32) : cv::Scalar(16, 16, 16));
                            cv::rectangle(ui, re, bg3, cv::FILLED);
                            cv::rectangle(ui, re, cv::Scalar(50, 50, 50), 1);
                            cv::putText(ui, episode, cv::Point(re.x + 8, re.y + 23), cv::FONT_HERSHEY_DUPLEX, 0.56, cv::Scalar(225, 225, 225), 1, cv::LINE_AA);
                            if(ms_.clicked && re.contains(cv::Point(ms_.clickX, ms_.clickY))) {
                                ms_.clicked = false;
                                selectedSubject_ = s.name;
                                selectedTask_ = t.name;
                                selectedEpisode_ = episode;
                                loadSelectedTask();
                            }
                            y += rowH;
                        }
                    }
                }
            }
        }
    }

    void drawTopBar(cv::Mat &ui) {
        cv::rectangle(ui, topBar_, cv::Scalar(18, 18, 18), cv::FILLED);
        cv::rectangle(ui, topBar_, cv::Scalar(80, 80, 80), 1);
        const int x0 = topBar_.x + 14;
        const int y0 = topBar_.y + 18;
        cv::putText(ui, "Data Type", cv::Point(x0, y0 + 18), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        cv::Rect drop(x0 + 110, y0 - 4, 240, 38);
        cv::rectangle(ui, drop, cv::Scalar(30, 30, 30), cv::FILLED);
        cv::rectangle(ui, drop, cv::Scalar(120, 120, 120), 1);
        cv::putText(ui, dataTypeToLabel(dataType_), cv::Point(drop.x + 10, drop.y + 25), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(ui, dropdownOpen_ ? "^" : "v", cv::Point(drop.x + drop.width - 18, drop.y + 25), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        if(ms_.clicked && drop.contains(cv::Point(ms_.clickX, ms_.clickY))) {
            ms_.clicked = false;
            dropdownOpen_ = !dropdownOpen_;
        }
        int rightX = drop.x + drop.width + 24;
        const bool labelsEnabled = labelsAvailable_ && dataType_ != ViewerDataType::IR;
        cv::Rect chk(rightX, y0 - 4, 180, 38);
        if(uiCheckbox(ui, chk, showLabels_, "show labels", labelsEnabled, ms_)) {
            showLabels_ = !showLabels_;
            if(dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud) {
                invalidatePointCloudPreload();
            }
        }

        cv::Rect backBtn(topBar_.x + topBar_.width - 160, y0 - 4, 140, 38);
        if(uiButton(ui, backBtn, "Back to Menu", ms_)) {
            shouldExit_ = true;
        }

        int sourceLabelY = y0 + 60;
        headPoseCamFieldRect_ = cv::Rect();
        if(headCamPoseAvailable_) {
            const int fieldY = y0 + 42;
            const cv::Rect field(x0 + 2, fieldY, 124, 36);
            headPoseCamFieldRect_ = field;
            if(uiTextField(ui, field, "First-person cam", headPoseCamInput_, headPoseCamFieldActive_, ms_)) {
                headPoseCamFieldActive_ = true;
            }
            const std::string resolved = resolvedHeadPoseCamera();
            const std::string hint =
                resolved.empty() ? (headPoseCamInput_.empty() ? "input 00/08" : "camera not found") : ("active on " + resolved);
            const cv::Scalar hintColor = resolved.empty() ? cv::Scalar(150, 150, 150) : cv::Scalar(80, 200, 80);
            cv::putText(ui, hint, cv::Point(field.x + field.width + 14, field.y + 24), cv::FONT_HERSHEY_DUPLEX, 0.55, hintColor, 1, cv::LINE_AA);
            sourceLabelY += 44;
        }

        cv::putText(ui, "Sources", cv::Point(x0, sourceLabelY), cv::FONT_HERSHEY_DUPLEX, 0.62, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        int sx = x0 + 90;
        int sy = sourceLabelY - 18;
        const int itemW = 150;
        const int itemH = 34;
        const int maxX = backBtn.x - 10;
        for(auto &source : sources_) {
            if(sx + itemW > maxX) {
                sx = x0 + 90;
                sy += itemH + 6;
            }
            const cv::Rect item(sx, sy, itemW, itemH);
            const bool enabled = sourceToggleEnabled(source);
            if(uiCheckbox(ui, item, source.visible, source.displayName, enabled, ms_)) {
                source.visible = !source.visible;
                if(source.kind == ViewerSourceKind::Multiview && (dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud)) {
                    invalidatePointCloudPreload();
                }
            }
            sx += itemW + 8;
        }

        dropdownAnchor_ = drop;
    }

    void drawDropdownOverlay(cv::Mat &ui) {
        if(!dropdownOpen_) {
            return;
        }
        const cv::Rect drop = dropdownAnchor_;
        if(drop.width <= 0 || drop.height <= 0) {
            return;
        }
        const int itemH = 34;
        const int listH = std::min(5, static_cast<int>(availableTypes_.size())) * itemH + 8;
        cv::Rect list(drop.x, drop.y + drop.height + 6, drop.width, std::max(44, listH));
        cv::rectangle(ui, list, cv::Scalar(24, 24, 24), cv::FILLED);
        cv::rectangle(ui, list, cv::Scalar(120, 120, 120), 1);
        int y = list.y + 6;
        for(const auto &t: availableTypes_) {
            cv::Rect item(list.x + 6, y, list.width - 12, itemH - 6);
            const bool sel = (t == dataType_);
            const bool hov = item.contains(cv::Point(ms_.x, ms_.y));
            cv::Scalar bg = sel ? cv::Scalar(60, 80, 120) : (hov ? cv::Scalar(40, 40, 40) : cv::Scalar(24, 24, 24));
            cv::rectangle(ui, item, bg, cv::FILLED);
            cv::rectangle(ui, item, cv::Scalar(70, 70, 70), 1);
            cv::putText(ui, dataTypeToLabel(t), cv::Point(item.x + 10, item.y + 20), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
            if(ms_.clicked && item.contains(cv::Point(ms_.clickX, ms_.clickY))) {
                ms_.clicked = false;
                dropdownOpen_ = false;
                const ViewerDataType prevType = dataType_;
                dataType_ = t;
                if(prevType != dataType_
                   && (prevType == ViewerDataType::PointCloud || prevType == ViewerDataType::ColorCloud || dataType_ == ViewerDataType::PointCloud
                       || dataType_ == ViewerDataType::ColorCloud)) {
                    invalidatePointCloudPreload();
                }
            }
            y += itemH;
            if(y > list.y + list.height - itemH) {
                break;
            }
        }
        if(ms_.clicked && !list.contains(cv::Point(ms_.clickX, ms_.clickY)) && !drop.contains(cv::Point(ms_.clickX, ms_.clickY))) {
            ms_.clicked = false;
            dropdownOpen_ = false;
        }
    }

    void drawBottomBar(cv::Mat &ui) {
        cv::rectangle(ui, bottomBar_, cv::Scalar(18, 18, 18), cv::FILLED);
        cv::rectangle(ui, bottomBar_, cv::Scalar(80, 80, 80), 1);
        const int y = bottomBar_.y + 14;
        int x = bottomBar_.x + 14;
        const int bw = 68;
        const int bh = 44;
        cv::Rect rPlay(x, y, 120, bh);
        if(uiButton(ui, rPlay, playing_ ? "Pause" : "Play", ms_)) {
            playing_ = !playing_;
        }
        x += 140;
        auto stepButton = [&](const std::string &label, int deltaFrames) {
            cv::Rect r(x, y, bw, bh);
            if(playing_) {
                uiButtonDisabled(ui, r, label);
            }
            else {
                if(uiButton(ui, r, label, ms_)) {
                    seekFrames(deltaFrames);
                }
            }
            x += bw + 10;
        };
        const int fps = std::max(1, cfg_.viewerFps > 0 ? cfg_.viewerFps : 30);
        stepButton("<<", -5 * fps);
        stepButton("<", -1);
        stepButton(">", +1);
        stepButton(">>", +5 * fps);
        std::ostringstream oss;
        oss << currentFrame_ << "/" << std::max(0, totalFrames_ - 1) << "  (" << totalFrames_ << " frames)";
        const std::string progress = oss.str();
        const int tx = bottomBar_.x + bottomBar_.width - 14;
        const int baseline = y + 30;
        int bl = 0;
        const cv::Size sz = cv::getTextSize(progress, cv::FONT_HERSHEY_DUPLEX, 0.65, 1, &bl);
        cv::putText(ui, progress, cv::Point(tx - sz.width, baseline), cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }

    void drawMain(cv::Mat &ui) {
        cv::rectangle(ui, mainRect_, cv::Scalar(12, 12, 12), cv::FILLED);
        cv::rectangle(ui, mainRect_, cv::Scalar(80, 80, 80), 1);
        if(selectedSubject_.empty() || selectedTask_.empty()) {
            cv::putText(ui, "Select a subject and task on the left", cv::Point(mainRect_.x + 20, mainRect_.y + 40), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(220, 220, 220), 1,
                        cv::LINE_AA);
            return;
        }
        if(selectedEpisode_.empty() && selectedTaskHasEpisodes()) {
            cv::putText(ui, "Select an episode on the left", cv::Point(mainRect_.x + 20, mainRect_.y + 40), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(220, 220, 220), 1,
                        cv::LINE_AA);
            return;
        }
        if(selectedDataDir().empty() || !fs::exists(selectedDataDir()) || !fs::is_directory(selectedDataDir())) {
            cv::putText(ui, "Select a task or episode on the left", cv::Point(mainRect_.x + 20, mainRect_.y + 40), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(220, 220, 220), 1,
                        cv::LINE_AA);
            return;
        }
        if(totalFrames_ <= 0) {
            cv::putText(ui, "No frames found", cv::Point(mainRect_.x + 20, mainRect_.y + 40), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            return;
        }

        if(playing_) {
            const auto now = std::chrono::steady_clock::now();
            const int fps = std::max(1, cfg_.viewerFps > 0 ? cfg_.viewerFps : 30);
            const auto step = std::chrono::microseconds(static_cast<int64_t>(1000000.0 / static_cast<double>(fps)));
            if(lastStep_.time_since_epoch().count() == 0) {
                lastStep_ = now;
            }
            if(now - lastStep_ >= step) {
                lastStep_ = now;
                seekFrames(1);
            }
        }
        else {
            lastStep_ = std::chrono::steady_clock::time_point{};
        }

        prefetchAroundCurrent();

        if(dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud) {
            pcAllowMouse_ = true;
            pcView_.pcRect = mainRect_;
            ensurePointCloudPreload();
            if(!drawPreloadedIfAvailable(ui)) {
                drawPointCloud(ui);
            }
            drawPointCloudOverlays(ui);
            return;
        }
        pcAllowMouse_ = false;

        std::vector<std::pair<std::string, cv::Mat>> frames;
        const auto activeSources = (dataType_ == ViewerDataType::RGB) ? activeSourcesForType(dataType_) : activeMultiviewSourcesForType(dataType_);
        frames.reserve(activeSources.size());
        for(const auto *source : activeSources) {
            if(!source) {
                continue;
            }
            cv::Mat img;
            if(dataType_ == ViewerDataType::RGB) {
                img = loadRgbFrame(*source, currentFrame_);
                if(source->kind == ViewerSourceKind::Multiview && showLabels_ && labelsAvailable_) {
                    drawLabelsOnRgb(img, source->camKey);
                }
            }
            else if(dataType_ == ViewerDataType::Depth) {
                const cv::Mat depth = loadDepthFrame(source->camKey, currentFrame_);
                img = depth16ToYellowBlue(depth, cfg_.maxDepth, 1.0f);
                if(showLabels_ && labelsAvailable_) {
                    drawLabelsOnDepth(img, depth, source->camKey);
                }
            }
            else {
                img = loadIrFrame(source->camKey, currentFrame_);
            }
            frames.emplace_back(source->displayName, img);
        }
        drawGridImages(ui, mainRect_, frames);
    }

    void clearTaskState() {
        invalidatePointCloudPreload();
        sources_.clear();
        cameras_.clear();
        availableTypes_.clear();
        totalFrames_ = 0;
        currentFrame_ = 0;
        playing_ = false;
        dropdownOpen_ = false;
        labelsAvailable_ = false;
        showLabels_ = false;
        taskCamParams_ = CameraParamsBundle{};
        extrinsics_ = {};
        labelsByFrame_.clear();
        rgbCache_.clear();
        depthCache_.clear();
        irCache_.clear();
        colorCloudCache_.clear();
        pointCloudFrameCache_.clear();
        fusedColorCloudFrameCache_.clear();
        jointWorldCache_.clear();
        alignCache_.clear();
        headCamPoseAvailable_ = false;
        headCamPoseLoadedJsonPath_.clear();
        headPoseCamFieldActive_ = false;
        headPoseCamFieldRect_ = cv::Rect();
        {
            std::lock_guard<std::mutex> lock(headCamPoseMtx_);
            headCamPoseFrameCache_.clear();
        }
        pcView_.resetView();
    }

    static std::string makeCacheKey(int typeId, const std::string &cam, int frameIdx) {
        std::string k;
        k.reserve(cam.size() + 32);
        k.append(cam);
        k.push_back('|');
        k.append(std::to_string(typeId));
        k.push_back('|');
        k.append(std::to_string(frameIdx));
        return k;
    }

    std::string pointCloudFrameKey(int frameIdx, bool colorCloud) const {
        const std::string signature =
            activeMultiviewSignature(colorCloud ? ViewerDataType::ColorCloud : ViewerDataType::PointCloud) + "|head=" + headCamPoseSpec();
        return makeCacheKey(colorCloud ? 30 : 29, signature, frameIdx);
    }

    std::string jointWorldFrameKey(int frameIdx) const {
        return makeCacheKey(31, activeMultiviewSignature(ViewerDataType::PointCloud) + "|head=" + headCamPoseSpec(), frameIdx);
    }

    void loadSelectedTask() {
        clearTaskState();
        const fs::path dataDir = selectedDataDir();
        if(!fs::exists(dataDir) || !fs::is_directory(dataDir)) {
            statusLine_ = "Task or episode directory missing";
            return;
        }
        for(const auto &e: fs::directory_iterator(dataDir)) {
            if(e.is_directory()) {
                const std::string name = e.path().filename().string();
                if(isDigits(name)) {
                    cameras_.push_back(name);
                    ViewerSource source;
                    source.sourceId = "mv:" + name;
                    source.displayName = name;
                    source.kind = ViewerSourceKind::Multiview;
                    source.camKey = name;
                    source.rgbDir = e.path() / "RGB";
                    source.depthDir = e.path() / "Depth";
                    source.irDir = e.path() / "IR";
                    if(!fs::exists(source.irDir)) {
                        source.irDir = e.path() / "IR_left";
                    }
                    if(!fs::exists(source.irDir)) {
                        source.irDir = e.path() / "IR_right";
                    }
                    source.hasRgb = computeTotalFramesFromDir(source.rgbDir) > 0;
                    source.hasDepth = computeTotalFramesFromDir(source.depthDir) > 0;
                    source.hasIr = computeTotalFramesFromDir(source.irDir) > 0;
                    sources_.push_back(std::move(source));
                }
            }
        }
        std::sort(cameras_.begin(), cameras_.end());

        const fs::path fisheyeDir = dataDir / "fisheye";
        if(fs::exists(fisheyeDir) && fs::is_directory(fisheyeDir)) {
            std::vector<fs::path> fisheyeCameraDirs;
            for(const auto &e : fs::directory_iterator(fisheyeDir)) {
                if(e.is_directory()) {
                    fisheyeCameraDirs.push_back(e.path());
                }
            }
            std::sort(fisheyeCameraDirs.begin(), fisheyeCameraDirs.end(), [](const fs::path &a, const fs::path &b) {
                return a.filename().string() < b.filename().string();
            });
            for(const auto &cameraDir : fisheyeCameraDirs) {
                ViewerSource source;
                source.sourceId = "fe:" + cameraDir.filename().string();
                source.displayName = cameraDir.filename().string();
                source.kind = ViewerSourceKind::Fisheye;
                source.camKey = cameraDir.filename().string();
                source.rgbDir = cameraDir / "RGB";
                source.hasRgb = computeTotalFramesFromDir(source.rgbDir) > 0;
                if(source.hasRgb) {
                    sources_.push_back(std::move(source));
                }
            }
        }
        std::sort(sources_.begin(), sources_.end(), [](const ViewerSource &a, const ViewerSource &b) {
            if(a.kind != b.kind) {
                return static_cast<int>(a.kind) < static_cast<int>(b.kind);
            }
            return a.displayName < b.displayName;
        });

        if(sources_.empty()) {
            statusLine_ = "No camera folders";
            return;
        }

        taskCamParams_ = loadCameraParams(dataDir / "camera_params.json");
        extrinsics_ = loadExtrinsicsCamToWorld(dataDir / "extrinsics.json");
        refreshHeadCamPoseAvailability(dataDir);

        {
            cv::Vec3f center(0.0f, 0.0f, 0.8f);
            int centerCount = 0;
            for(const auto &cam : cameras_) {
                auto itEx = extrinsics_.find(cam);
                if(itEx == extrinsics_.end() || !itEx->second.valid) {
                    continue;
                }
                const cv::Vec3f camPos = itEx->second.t;
                const cv::Vec3f forwardW = normalizeVec3(itEx->second.R * cv::Vec3f(0, 0, 1));
                if(forwardW.dot(forwardW) <= 1e-6f) {
                    continue;
                }
                center += camPos + forwardW * 1.0f;
                centerCount++;
            }
            if(centerCount > 0) {
                center *= (1.0f / static_cast<float>(centerCount + 1));
            }
            pcView_.yawRad = 0.72f;
            pcView_.pitchRad = 0.48f;
            pcView_.distance = 2.6f;
            pcView_.target = center;
        }

        labelsByFrame_.clear();
        labelsAvailable_ = false;
        const fs::path labelsJson = dataRoot_ / selectedSubject_ / "labels.json";
        if(fs::exists(labelsJson) && fs::is_regular_file(labelsJson)) {
            labelsByFrame_ = loadLabelsForTask(labelsJson, selectedTask_);
            labelsAvailable_ = true;
        }
        showLabels_ = false;

        bool hasAnyRgb = false;
        bool hasMultiviewRgb = false;
        bool hasDepth = false;
        bool hasIr = false;
        int totalRgb = 0;
        int totalDepth = 0;
        int totalIr = 0;
        for(const auto &cam: cameras_) {
            const fs::path camDir = dataDir / cam;
            const int nRgb = computeTotalFramesFromDir(camDir / "RGB");
            const int nDepth = computeTotalFramesFromDir(camDir / "Depth");
            int nIr = computeTotalFramesFromDir(camDir / "IR");
            if(nIr == 0) {
                nIr = computeTotalFramesFromDir(camDir / "IR_left");
            }
            if(nIr == 0) {
                nIr = computeTotalFramesFromDir(camDir / "IR_right");
            }
            if(nRgb > 0) {
                hasAnyRgb = true;
                hasMultiviewRgb = true;
                totalRgb = (totalRgb == 0) ? nRgb : std::min(totalRgb, nRgb);
            }
            if(nDepth > 0) {
                hasDepth = true;
                totalDepth = (totalDepth == 0) ? nDepth : std::min(totalDepth, nDepth);
            }
            if(nIr > 0) {
                hasIr = true;
                totalIr = (totalIr == 0) ? nIr : std::min(totalIr, nIr);
            }
        }
        int fisheyeRgbTotal = 0;
        for(const auto &source : sources_) {
            if(source.kind != ViewerSourceKind::Fisheye || !source.hasRgb) {
                continue;
            }
            hasAnyRgb = true;
            const int nRgb = computeTotalFramesFromDir(source.rgbDir);
            if(nRgb > 0) {
                fisheyeRgbTotal = (fisheyeRgbTotal == 0) ? nRgb : std::min(fisheyeRgbTotal, nRgb);
            }
        }
        totalFrames_ = 0;
        const int csvFrames = countCsvDataRows(dataDir / "timestamps.csv");
        if(csvFrames > 0) {
            totalFrames_ = csvFrames;
        }
        else if(totalRgb > 0) {
            totalFrames_ = totalRgb;
        }
        else if(fisheyeRgbTotal > 0) {
            totalFrames_ = fisheyeRgbTotal;
        }
        if(totalFrames_ > 0 && hasDepth && totalDepth > 0) {
            totalFrames_ = std::min(totalFrames_, totalDepth);
        }
        else if(hasDepth) {
            totalFrames_ = (totalFrames_ == 0) ? totalDepth : std::min(totalFrames_, totalDepth);
        }
        if(totalFrames_ == 0 && hasIr) {
            totalFrames_ = totalIr;
        }
        totalFrames_ = std::max(0, totalFrames_);
        currentFrame_ = 0;

        availableTypes_.clear();
        if(hasAnyRgb) {
            availableTypes_.push_back(ViewerDataType::RGB);
        }
        if(hasDepth) {
            availableTypes_.push_back(ViewerDataType::Depth);
        }
        if(hasIr) {
            availableTypes_.push_back(ViewerDataType::IR);
        }
        if(hasDepth) {
            availableTypes_.push_back(ViewerDataType::PointCloud);
            if(hasMultiviewRgb) {
                availableTypes_.push_back(ViewerDataType::ColorCloud);
            }
        }
        if(availableTypes_.empty()) {
            availableTypes_.push_back(ViewerDataType::RGB);
        }
        dataType_ = availableTypes_.front();
        dropdownOpen_ = false;
        statusLine_.clear();
    }

    void seekFrames(int delta) {
        if(totalFrames_ <= 0) {
            currentFrame_ = 0;
            return;
        }
        int next = currentFrame_ + delta;
        while(next < 0) {
            next += totalFrames_;
        }
        while(next >= totalFrames_) {
            next -= totalFrames_;
        }
        currentFrame_ = next;
    }

    void prefetchAroundCurrent() {
        if(totalFrames_ <= 0 || selectedSubject_.empty() || selectedTask_.empty() || selectedDataDir().empty()) {
            return;
        }
        const int ahead = playing_ ? 20 : 6;
        auto wrap = [&](int f) {
            int v = f;
            while(v < 0) {
                v += totalFrames_;
            }
            while(v >= totalFrames_) {
                v -= totalFrames_;
            }
            return v;
        };

        auto preRgb = [&](const std::string &cam, int f) {
            const fs::path dir = selectedDataDir() / cam / "RGB";
            const std::string key = makeCacheKey(0, cam, f);
            rgbCache_.prefetch(key, [&]() {
                const fs::path p = findFrameFile(dir, f, { ".jpg", ".jpeg", ".png" });
                if(p.empty()) {
                    return cv::Mat();
                }
                return cv::imread(p.string(), cv::IMREAD_COLOR);
            });
        };
        auto preRgbSource = [&](const ViewerSource &source, int f) {
            if(source.kind == ViewerSourceKind::Multiview) {
                preRgb(source.camKey, f);
                return;
            }
            const std::string key = makeCacheKey(0, source.sourceId, f);
            rgbCache_.prefetch(key, [&]() {
                const fs::path p = findFrameFile(source.rgbDir, f, { ".jpg", ".jpeg", ".png" });
                if(p.empty()) {
                    return cv::Mat();
                }
                return cv::imread(p.string(), cv::IMREAD_COLOR);
            });
        };
        auto preDepth = [&](const std::string &cam, int f) {
            const fs::path dir = selectedDataDir() / cam / "Depth";
            const std::string key = makeCacheKey(1, cam, f);
            depthCache_.prefetch(key, [&]() {
                const fs::path p = findFrameFile(dir, f, { ".png" });
                if(p.empty()) {
                    return cv::Mat();
                }
                cv::Mat m = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
                if(!m.empty() && m.type() != CV_16UC1) {
                    if(m.type() == CV_8UC1) {
                        cv::Mat tmp;
                        m.convertTo(tmp, CV_16UC1);
                        m = tmp;
                    }
                    else {
                        m.release();
                    }
                }
                return m;
            });
        };
        auto preIr = [&](const std::string &cam, int f) {
            fs::path dir = selectedDataDir() / cam / "IR";
            if(!fs::exists(dir)) {
                dir = selectedDataDir() / cam / "IR_left";
            }
            if(!fs::exists(dir)) {
                dir = selectedDataDir() / cam / "IR_right";
            }
            const std::string key = makeCacheKey(2, cam, f);
            irCache_.prefetch(key, [&]() {
                const fs::path p = findFrameFile(dir, f, { ".png" });
                if(p.empty()) {
                    return cv::Mat();
                }
                cv::Mat raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
                if(raw.empty()) {
                    return cv::Mat();
                }
                cv::Mat bgr;
                if(raw.type() == CV_8UC1) {
                    cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR);
                }
                else if(raw.type() == CV_16UC1) {
                    double minv = 0.0, maxv = 0.0;
                    cv::minMaxLoc(raw, &minv, &maxv);
                    if(maxv <= minv) {
                        maxv = minv + 1.0;
                    }
                    cv::Mat tmp8;
                    raw.convertTo(tmp8, CV_8UC1, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
                    cv::cvtColor(tmp8, bgr, cv::COLOR_GRAY2BGR);
                }
                else {
                    bgr = toBgrForDisplay(raw);
                }
                return bgr;
            });
        };

        const auto activeRgbSources = activeSourcesForType(ViewerDataType::RGB);
        const auto activeDepthSources = activeMultiviewSourcesForType(ViewerDataType::Depth);
        const auto activeIrSources = activeMultiviewSourcesForType(ViewerDataType::IR);
        for(int i = 0; i <= ahead; i++) {
            const int f = wrap(currentFrame_ + i);
            if(dataType_ == ViewerDataType::RGB) {
                for(const auto *source : activeRgbSources) {
                    if(source) {
                        preRgbSource(*source, f);
                    }
                }
            }
            else if(dataType_ == ViewerDataType::Depth) {
                for(const auto *source : activeDepthSources) {
                    if(!source) {
                        continue;
                    }
                    preDepth(source->camKey, f);
                    if(showLabels_) {
                        preRgb(source->camKey, f);
                    }
                }
            }
            else if(dataType_ == ViewerDataType::IR) {
                for(const auto *source : activeIrSources) {
                    if(source) {
                        preIr(source->camKey, f);
                    }
                }
            }
            else if(dataType_ == ViewerDataType::PointCloud) {
                pointCloudFrameCache_.prefetch(pointCloudFrameKey(f, false), [&]() {
                    return buildPointCloudFrameData(f, false);
                });
                if(showLabels_) {
                    jointWorldCache_.prefetch(jointWorldFrameKey(f), [&]() {
                        return buildJointWorldPointsFrame(f);
                    });
                }
            }
            else if(dataType_ == ViewerDataType::ColorCloud) {
                fusedColorCloudFrameCache_.prefetch(pointCloudFrameKey(f, true), [&]() {
                    return buildPointCloudFrameData(f, true);
                });
                if(showLabels_) {
                    jointWorldCache_.prefetch(jointWorldFrameKey(f), [&]() {
                        return buildJointWorldPointsFrame(f);
                    });
                }
            }
        }
        /*
         * Keep the image caches warm around the current frame without tying playback to a
         * pre-rendered canvas. Point cloud modes use geometry caches above so the current
         * interactive view can be applied at draw time.
         */
        for(const auto &cam: cameras_) {
            for(int i = 0; i <= ahead; i++) {
                const int f = wrap(currentFrame_ + i);
                if(dataType_ == ViewerDataType::PointCloud || dataType_ == ViewerDataType::ColorCloud) {
                    preDepth(cam, f);
                    preRgb(cam, f);
                }
            }
        }
    }

    cv::Mat loadRgbFrame(const std::string &cam, int frameIdx) {
        const fs::path dir = selectedDataDir() / cam / "RGB";
        const std::string key = makeCacheKey(0, cam, frameIdx);
        cv::Mat img = rgbCache_.getOrLoad(key, [&]() {
            const fs::path p = findFrameFile(dir, frameIdx, { ".jpg", ".jpeg", ".png" });
            if(p.empty()) {
                return cv::Mat();
            }
            return cv::imread(p.string(), cv::IMREAD_COLOR);
        });
        return img.clone();
    }

    cv::Mat loadRgbFrame(const ViewerSource &source, int frameIdx) {
        if(source.kind == ViewerSourceKind::Multiview) {
            return loadRgbFrame(source.camKey, frameIdx);
        }
        const std::string key = makeCacheKey(0, source.sourceId, frameIdx);
        cv::Mat img = rgbCache_.getOrLoad(key, [&]() {
            const fs::path p = findFrameFile(source.rgbDir, frameIdx, { ".jpg", ".jpeg", ".png" });
            if(p.empty()) {
                return cv::Mat();
            }
            return cv::imread(p.string(), cv::IMREAD_COLOR);
        });
        return img.clone();
    }

    cv::Mat loadDepthFrame(const std::string &cam, int frameIdx) {
        const fs::path dir = selectedDataDir() / cam / "Depth";
        const std::string key = makeCacheKey(1, cam, frameIdx);
        cv::Mat img = depthCache_.getOrLoad(key, [&]() {
            const fs::path p = findFrameFile(dir, frameIdx, { ".png" });
            if(p.empty()) {
                return cv::Mat();
            }
            cv::Mat m = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
            if(!m.empty() && m.type() != CV_16UC1) {
                if(m.type() == CV_8UC1) {
                    cv::Mat tmp;
                    m.convertTo(tmp, CV_16UC1);
                    m = tmp;
                }
                else {
                    m.release();
                }
            }
            return m;
        });
        return img.clone();
    }

    cv::Mat loadIrFrame(const std::string &cam, int frameIdx) {
        fs::path dir = selectedDataDir() / cam / "IR";
        if(!fs::exists(dir)) {
            dir = selectedDataDir() / cam / "IR_left";
        }
        if(!fs::exists(dir)) {
            dir = selectedDataDir() / cam / "IR_right";
        }
        const std::string key = makeCacheKey(2, cam, frameIdx);
        cv::Mat out = irCache_.getOrLoad(key, [&]() {
            const fs::path p = findFrameFile(dir, frameIdx, { ".png" });
            if(p.empty()) {
                return cv::Mat();
            }
            cv::Mat raw = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
            if(raw.empty()) {
                return cv::Mat();
            }
            cv::Mat bgr;
            if(raw.type() == CV_8UC1) {
                cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR);
            }
            else if(raw.type() == CV_16UC1) {
                double minv = 0.0, maxv = 0.0;
                cv::minMaxLoc(raw, &minv, &maxv);
                if(maxv <= minv) {
                    maxv = minv + 1.0;
                }
                cv::Mat tmp8;
                raw.convertTo(tmp8, CV_8UC1, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
                cv::cvtColor(tmp8, bgr, cv::COLOR_GRAY2BGR);
            }
            else {
                bgr = toBgrForDisplay(raw);
            }
            return bgr;
        });
        return out.clone();
    }

    CloudPointVecPtr loadSavedColorCloudFrame(int frameIdx) const {
        const fs::path dir = selectedDataDir() / "ColorCloudPoints";
        const std::string key = makeCacheKey(3, "ColorCloudPoints", frameIdx);
        return colorCloudCache_.getOrLoad(key, [&]() -> CloudPointVecPtr {
            const fs::path p = findFrameFile(dir, frameIdx, { ".ply" });
            if(p.empty()) {
                return {};
            }
            // Collection already saves a fused world-space color cloud. Prefer it over
            // rebuilding geometry from RGB-aligned depth, which is a resampled product.
            return loadColorCloudPly(p);
        });
    }

    CloudPointVecPtr buildPointCloudFrameData(int frameIdx, bool colorCloud) {
        bool allowSavedColorCloud = colorCloud;
        if(colorCloud) {
            for(const auto &source : sources_) {
                if(source.kind != ViewerSourceKind::Multiview || !sourceSupportsType(source, ViewerDataType::ColorCloud)) {
                    continue;
                }
                if(!source.visible) {
                    allowSavedColorCloud = false;
                    break;
                }
            }
            if(allowSavedColorCloud && !canUseSavedColorCloud()) {
                allowSavedColorCloud = false;
            }
        }
        const auto savedColorCloud = allowSavedColorCloud ? loadSavedColorCloudFrame(frameIdx) : CloudPointVecPtr{};
        if(savedColorCloud && !savedColorCloud->empty()) {
            return savedColorCloud;
        }

        auto out = std::make_shared<CloudPointVec>();
        const int step = std::max(1, cfg_.filters.pointCloudDecimationFactor > 0 ? cfg_.filters.pointCloudDecimationFactor : 2);
        const auto activeSources = activeMultiviewSourcesForType(colorCloud ? ViewerDataType::ColorCloud : ViewerDataType::PointCloud);
        for(const auto *source : activeSources) {
            if(!source) {
                continue;
            }
            const std::string &cam = source->camKey;
            const cv::Mat depth16 = loadDepthFrame(cam, frameIdx);
            if(depth16.empty() || depth16.type() != CV_16UC1) {
                continue;
            }
            const ExtrinsicCamToWorld ex = effectiveExtrinsicForFrame(cam, frameIdx);

            cv::Mat rgb;
            AlignMapCache *align = nullptr;
            AlignMapCache cache;
            const OBCameraIntrinsic *pointIntr = nullptr;
            if(colorCloud) {
                rgb = loadRgbFrame(cam, frameIdx);
                const auto *rp = findByCamKeyVariants(taskCamParams_.rgbToDepth, cam);
                pointIntr = pointCloudIntrinsicForDepth(cam, depth16, rgb);
                if(!rgb.empty() && !isAlignedDepthToRgb(depth16, rgb) && (!rp || !rp->valid)) {
                    rgb.release();
                }
                if(!rgb.empty() && !isAlignedDepthToRgb(depth16, rgb)) {
                    buildDepthToColorMap(cache, frameIdx, depth16, rgb.cols, rgb.rows, *rp, cfg_.maxDepth, 1.0f);
                    align = &cache;
                }
            }
            if(!pointIntr) {
                pointIntr = pointCloudIntrinsicForDepth(cam, depth16, rgb);
            }
            if(!pointIntr || pointIntr->fx <= 0.0f || pointIntr->fy <= 0.0f) {
                continue;
            }

            const float fx = pointIntr->fx;
            const float fy = pointIntr->fy;
            const float cx = pointIntr->cx;
            const float cy = pointIntr->cy;
            const float sMm = 1.0f;
            for(int y = 0; y < depth16.rows; y += step) {
                const uint16_t *row = depth16.ptr<uint16_t>(y);
                for(int x = 0; x < depth16.cols; x += step) {
                    const uint16_t d = row[x];
                    if(d == 0) {
                        continue;
                    }
                    const float depthMm = static_cast<float>(d) * sMm;
                    const float z = depthMm * 0.001f;
                    if(!(z >= 0.2f && z <= cfg_.maxDepth)) {
                        continue;
                    }
                    const float xCam = (static_cast<float>(x) - cx) * z / fx;
                    const float yCam = (static_cast<float>(y) - cy) * z / fy;
                    const cv::Vec3f pw = ex.R * cv::Vec3f(xCam, yCam, z) + ex.t;
                    CloudPoint cp;
                    cp.p = pw;
                    if(colorCloud && !rgb.empty()) {
                        if(isAlignedDepthToRgb(depth16, rgb)) {
                            cp.c = rgb.at<cv::Vec3b>(y, x);
                            cp.hasColor = true;
                        }
                        else if(align && align->depthW == depth16.cols && align->depthH == depth16.rows) {
                            const int32_t packedUv = align->depthToColor[static_cast<size_t>(y) * static_cast<size_t>(align->depthW) + static_cast<size_t>(x)];
                            int u = -1;
                            int v = -1;
                            if(unpackXY(packedUv, u, v) && u >= 0 && u < rgb.cols && v >= 0 && v < rgb.rows) {
                                cp.c = rgb.at<cv::Vec3b>(v, u);
                                cp.hasColor = true;
                            }
                        }
                    }
                    out->push_back(cp);
                }
            }
        }
        return out;
    }

    CloudPointVecPtr loadPointCloudFrameCached(int frameIdx, bool colorCloud) {
        const std::string key = pointCloudFrameKey(frameIdx, colorCloud);
        auto &cache = colorCloud ? fusedColorCloudFrameCache_ : pointCloudFrameCache_;
        return cache.getOrLoad(key, [&]() {
            return buildPointCloudFrameData(frameIdx, colorCloud);
        });
    }

    JointWorldVecPtr buildJointWorldPointsFrame(int frameIdx) {
        auto out = std::make_shared<JointWorldVec>();
        const auto activeSources = activeMultiviewSourcesForType(ViewerDataType::PointCloud);
        for(const auto *source : activeSources) {
            if(!source) {
                continue;
            }
            const std::string &cam = source->camKey;
            const auto pts2d = labelsForFrameCamAt(frameIdx, cam);
            if(pts2d.empty()) {
                continue;
            }
            const cv::Mat rgb = loadRgbFrame(cam, frameIdx);
            const cv::Mat depth16 = loadDepthFrame(cam, frameIdx);
            if(rgb.empty() || depth16.empty()) {
                continue;
            }
            const OBCameraIntrinsic *pointIntr = pointCloudIntrinsicForDepth(cam, depth16, rgb);
            if(!pointIntr || pointIntr->fx <= 0.0f || pointIntr->fy <= 0.0f) {
                continue;
            }

            const ExtrinsicCamToWorld ex = effectiveExtrinsicForFrame(cam, frameIdx);

            for(const auto &pt : pts2d) {
                if(!(pt.x >= 0.0f && pt.y >= 0.0f)) {
                    continue;
                }
                int dx = -1;
                int dy = -1;
                if(!mapRgbPixelToDepthPixel(cam, depth16, rgb.cols, rgb.rows, frameIdx, static_cast<int>(std::lround(pt.x)), static_cast<int>(std::lround(pt.y)), dx, dy)) {
                    continue;
                }
                if(dx < 0 || dx >= depth16.cols || dy < 0 || dy >= depth16.rows) {
                    continue;
                }
                const uint16_t d = depth16.at<uint16_t>(dy, dx);
                if(d == 0) {
                    continue;
                }
                const float z = static_cast<float>(d) * 0.001f;
                if(!(z >= 0.2f && z <= cfg_.maxDepth)) {
                    continue;
                }
                const float xCam = (static_cast<float>(dx) - pointIntr->cx) * z / pointIntr->fx;
                const float yCam = (static_cast<float>(dy) - pointIntr->cy) * z / pointIntr->fy;
                const cv::Vec3f pw = ex.R * cv::Vec3f(xCam, yCam, z) + ex.t;
                if(std::isfinite(pw[0]) && std::isfinite(pw[1]) && std::isfinite(pw[2])) {
                    out->push_back(pw);
                }
            }
        }
        return out;
    }

    JointWorldVecPtr loadJointWorldPointsCached(int frameIdx) {
        const std::string key = jointWorldFrameKey(frameIdx);
        return jointWorldCache_.getOrLoad(key, [&]() {
            return buildJointWorldPointsFrame(frameIdx);
        });
    }

    std::vector<cv::Point2f> labelsForFrameCam(const std::string &cam) const {
        auto itF = labelsByFrame_.find(currentFrame_);
        if(itF == labelsByFrame_.end()) {
            return {};
        }
        const auto *pts = findByCamKeyVariants(itF->second, cam);
        if(!pts) {
            return {};
        }
        return *pts;
    }

    void drawLabelsOnRgb(cv::Mat &rgb, const std::string &cam) {
        if(rgb.empty()) {
            return;
        }
        const auto pts = labelsForFrameCam(cam);
        for(const auto &p: pts) {
            cv::circle(rgb, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        }
    }

    AlignMapCache &getAlignCache(const std::string &cam, const cv::Mat &depth16, int rgbW, int rgbH, int frameIdx) {
        auto &cache = alignCache_[cam];
        const auto *p = findByCamKeyVariants(taskCamParams_.rgbToDepth, cam);
        if(!p || !p->valid) {
            return cache;
        }
        if(cache.frameIdx == frameIdx && cache.rgbW == rgbW && cache.rgbH == rgbH && !cache.colorToDepth.empty()) {
            return cache;
        }
        const float sMm = 1.0f;
        buildDepthToColorMap(cache, frameIdx, depth16, rgbW, rgbH, *p, cfg_.maxDepth, sMm);
        return cache;
    }

    bool mapRgbPixelToDepthPixel(const std::string &cam, const cv::Mat &depth16, int rgbW, int rgbH, int frameIdx, int rgbX, int rgbY, int &depthX, int &depthY) {
        if(isAlignedDepthToRgb(depth16, rgbW, rgbH)) {
            if(rgbX >= 0 && rgbX < depth16.cols && rgbY >= 0 && rgbY < depth16.rows) {
                depthX = rgbX;
                depthY = rgbY;
                return true;
            }
            return false;
        }
        if(!findByCamKeyVariants(taskCamParams_.rgbToDepth, cam)) {
            return false;
        }
        AlignMapCache &cache = getAlignCache(cam, depth16, rgbW, rgbH, frameIdx);
        return lookupDepthForRgb(cache, rgbX, rgbY, depthX, depthY);
    }

    const OBCameraIntrinsic *pointCloudIntrinsicForDepth(const std::string &cam, const cv::Mat &depth16, const cv::Mat &rgb) const {
        const auto *rgbParams = findByCamKeyVariants(taskCamParams_.rgb, cam);
        const bool alignedWithRgbFrame = isAlignedDepthToRgb(depth16, rgb);
        const bool alignedWithRgbConfig =
            !depth16.empty() && rgbParams && rgbParams->valid && rgbParams->intrinsic.width > 0 && rgbParams->intrinsic.height > 0 && depth16.cols == rgbParams->intrinsic.width
            && depth16.rows == rgbParams->intrinsic.height;
        if((alignedWithRgbFrame || alignedWithRgbConfig) && rgbParams && rgbParams->valid && rgbParams->intrinsic.fx > 0.0f && rgbParams->intrinsic.fy > 0.0f) {
            return &rgbParams->intrinsic;
        }
        const auto *depthParams = findByCamKeyVariants(taskCamParams_.depth, cam);
        if(depthParams && depthParams->valid && depthParams->intrinsic.fx > 0.0f && depthParams->intrinsic.fy > 0.0f) {
            return &depthParams->intrinsic;
        }
        return nullptr;
    }

    void drawLabelsOnDepth(cv::Mat &depthBgr, const cv::Mat &depth16, const std::string &cam) {
        if(depthBgr.empty() || depth16.empty()) {
            return;
        }
        const cv::Mat rgb = loadRgbFrame(cam, currentFrame_);
        const int rgbW = rgb.empty() ? 0 : rgb.cols;
        const int rgbH = rgb.empty() ? 0 : rgb.rows;
        if(rgbW <= 0 || rgbH <= 0) {
            return;
        }
        const auto pts = labelsForFrameCam(cam);
        for(const auto &p: pts) {
            int dx = -1, dy = -1;
            if(mapRgbPixelToDepthPixel(cam, depth16, rgbW, rgbH, currentFrame_, static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)), dx, dy)) {
                if(dx >= 0 && dx < depth16.cols && dy >= 0 && dy < depth16.rows) {
                    cv::circle(depthBgr, cv::Point(dx, dy), 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
                }
            }
        }
    }

    void drawPointCloud(cv::Mat &ui) {
        const auto cloudPtr = loadPointCloudFrameCached(currentFrame_, dataType_ == ViewerDataType::ColorCloud);
        JointWorldVec joints;
        if(showLabels_ && labelsAvailable_) {
            const auto jointsPtr = loadJointWorldPointsCached(currentFrame_);
            if(jointsPtr) {
                joints = clusterCandidatesMedoids(*jointsPtr, 0.008f);
            }
        }
        const std::vector<CloudPoint> emptyCloud;
        const cv::Mat pc = renderPointCloudCanvas(cloudPtr ? *cloudPtr : emptyCloud,
                                                  joints,
                                                  pcView_,
                                                  cv::Size(mainRect_.width, mainRect_.height),
                                                  cfg_.differentColor);
        pc.copyTo(ui(mainRect_));
    }

    std::vector<cv::Vec3f> computeJointWorldPoints() {
        const auto joints = loadJointWorldPointsCached(currentFrame_);
        const std::vector<cv::Vec3f> candidates = joints ? *joints : std::vector<cv::Vec3f>{};
        const float mergeR = 0.008f;
        if(candidates.empty()) {
            return {};
        }
        std::vector<std::vector<cv::Vec3f>> clusters;
        clusters.reserve(candidates.size());
        for(const auto &p: candidates) {
            bool assigned = false;
            for(auto &cl: clusters) {
                if(cl.empty()) {
                    continue;
                }
                const cv::Vec3f d = p - cl.front();
                if(std::sqrt(d.dot(d)) <= mergeR) {
                    cl.push_back(p);
                    assigned = true;
                    break;
                }
            }
            if(!assigned) {
                clusters.push_back({ p });
            }
        }
        std::vector<cv::Vec3f> out;
        out.reserve(clusters.size());
        for(const auto &cl: clusters) {
            if(cl.empty()) {
                continue;
            }
            out.push_back(medoidPoint(cl));
        }
        return out;
    }

private:
    AppConfig cfg_;
    const std::atomic_bool *cancel_;

    std::string winName_;
    CvMouseState ms_;
    MainMouseContext mainMouseCtx_;
    ViewerViewState pcView_;
    bool pcAllowMouse_ = false;
    PointCloudMouseContext pcMouseCtx_;

    fs::path dataRoot_;
    bool shouldExit_ = false;
    std::string statusLine_;

    std::vector<SubjectEntry> subjects_;
    int leftScrollY_ = 0;
    std::string selectedSubject_;
    std::string selectedTask_;
    std::string selectedEpisode_;

    std::vector<ViewerSource> sources_;
    std::vector<std::string> cameras_;
    std::vector<ViewerDataType> availableTypes_;
    ViewerDataType dataType_ = ViewerDataType::RGB;
    bool dropdownOpen_ = false;
    cv::Rect dropdownAnchor_;
    bool labelsAvailable_ = false;
    bool showLabels_ = false;
    int totalFrames_ = 0;
    int currentFrame_ = 0;
    bool playing_ = false;
    std::chrono::steady_clock::time_point lastStep_;

    cv::Rect leftPanel_;
    cv::Rect topBar_;
    cv::Rect bottomBar_;
    cv::Rect mainRect_;

    CameraParamsBundle taskCamParams_;
    std::unordered_map<std::string, ExtrinsicCamToWorld> extrinsics_;
    std::unordered_map<int, std::unordered_map<std::string, std::vector<cv::Point2f>>> labelsByFrame_;
    bool headCamPoseAvailable_ = false;
    std::string headPoseCamInput_;
    bool headPoseCamFieldActive_ = false;
    cv::Rect headPoseCamFieldRect_;
    mutable std::mutex headCamPoseMtx_;
    mutable fs::path headCamPoseLoadedJsonPath_;
    mutable std::unordered_map<int, std::optional<ExtrinsicCamToWorld>> headCamPoseFrameCache_;

    MatLruCache rgbCache_{ 900 };
    MatLruCache depthCache_{ 900 };
    MatLruCache irCache_{ 600 };
    mutable CloudLruCache colorCloudCache_{ 180 };
    mutable CloudLruCache pointCloudFrameCache_{ 180 };
    mutable CloudLruCache fusedColorCloudFrameCache_{ 180 };
    mutable JointLruCache jointWorldCache_{ 180 };
    std::unordered_map<std::string, AlignMapCache> alignCache_;

    std::atomic_bool preloadStop_{ false };
    std::thread preloadThread_;
    ViewerViewState preloadView_;
    bool preloadReadyDepth_ = false;
    bool preloadReadyPointCloud_ = false;
    bool preloadReadyColorCloud_ = false;
    mutable std::mutex preloadMtx_;
    std::vector<cv::Mat> preDepthCanvas_;
    std::vector<cv::Mat> preDepthCanvasL_;
    std::vector<cv::Mat> prePointCloudCanvas_;
    std::vector<cv::Mat> prePointCloudCanvasL_;
    std::vector<cv::Mat> preColorCloudCanvas_;
    std::vector<cv::Mat> preColorCloudCanvasL_;
    std::string preloadSpec_;
    int preloadStartFrame_ = 0;
    int preloadSpan_ = 0;
    std::chrono::steady_clock::time_point preloadLastRestart_{};
};

int run_viewer(const AppConfig &cfg, const std::atomic_bool *cancel) {
    try {
        DatasetViewer viewer(cfg, cancel);
        return viewer.run();
    }
    catch(...) {
        return 1;
    }
}

}  // namespace sync_app
