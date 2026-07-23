#include "task_backend_client.hpp"

#include "utils/cJSON.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <sstream>

#if defined(_WIN32)
#error "TaskBackendClient currently requires POSIX sockets"
#endif

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace sync_app {

namespace {

struct ParsedUrl {
    std::string host;
    std::string port = "80";
    std::string basePath;
};

struct HttpResponse {
    int status = 0;
    std::string body;
};

static void setError(std::string *errorMessage, const std::string &message) {
    if(errorMessage) {
        *errorMessage = message;
    }
}

static std::string trimAscii(std::string s) {
    auto isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while(!s.empty() && isSpace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while(!s.empty() && isSpace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

static std::string trimSlashesRight(std::string s) {
    while(s.size() > 1 && s.back() == '/') {
        s.pop_back();
    }
    return s;
}

static bool parseHttpUrl(const std::string &url, ParsedUrl &out, std::string *errorMessage) {
    std::string s = trimAscii(url);
    const std::string prefix = "http://";
    if(s.compare(0, prefix.size(), prefix) != 0) {
        setError(errorMessage, "Only http:// task backend URLs are supported: " + url);
        return false;
    }
    s = s.substr(prefix.size());
    const auto slashPos = s.find('/');
    std::string authority = (slashPos == std::string::npos) ? s : s.substr(0, slashPos);
    out.basePath = (slashPos == std::string::npos) ? "" : trimSlashesRight(s.substr(slashPos));
    if(authority.empty()) {
        setError(errorMessage, "Task backend URL host is empty: " + url);
        return false;
    }

    if(authority.front() == '[') {
        const auto close = authority.find(']');
        if(close == std::string::npos) {
            setError(errorMessage, "Invalid IPv6 task backend URL: " + url);
            return false;
        }
        out.host = authority.substr(1, close - 1);
        if(close + 1 < authority.size()) {
            if(authority[close + 1] != ':') {
                setError(errorMessage, "Invalid task backend URL authority: " + url);
                return false;
            }
            out.port = authority.substr(close + 2);
        }
    }
    else {
        const auto colon = authority.rfind(':');
        if(colon != std::string::npos) {
            out.host = authority.substr(0, colon);
            out.port = authority.substr(colon + 1);
        }
        else {
            out.host = authority;
        }
    }
    if(out.host.empty() || out.port.empty()) {
        setError(errorMessage, "Invalid task backend URL: " + url);
        return false;
    }
    return true;
}

static std::string urlEncode(const std::string &s) {
    static const char *hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    for(unsigned char ch: s) {
        const bool safe = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
                          || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_'
                          || ch == '.' || ch == '~';
        if(safe) {
            out.push_back(static_cast<char>(ch));
        }
        else {
            out.push_back('%');
            out.push_back(hex[(ch >> 4) & 0xF]);
            out.push_back(hex[ch & 0xF]);
        }
    }
    return out;
}

static std::string joinBackendPath(const ParsedUrl &url, const std::string &pathAndQuery) {
    if(url.basePath.empty()) {
        return pathAndQuery;
    }
    if(pathAndQuery.empty() || pathAndQuery.front() != '/') {
        return url.basePath + "/" + pathAndQuery;
    }
    return url.basePath + pathAndQuery;
}

static bool sendAll(int fd, const std::string &data, std::string *errorMessage) {
    size_t sent = 0;
    while(sent < data.size()) {
        const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
        if(n < 0) {
            if(errno == EINTR) {
                continue;
            }
            setError(errorMessage, "HTTP send failed: " + std::string(std::strerror(errno)));
            return false;
        }
        if(n == 0) {
            setError(errorMessage, "HTTP send failed: connection closed");
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

static bool recvAll(int fd, std::string &out, std::string *errorMessage) {
    char buf[4096];
    while(true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if(n < 0) {
            if(errno == EINTR) {
                continue;
            }
            setError(errorMessage, "HTTP receive failed: " + std::string(std::strerror(errno)));
            return false;
        }
        if(n == 0) {
            return true;
        }
        out.append(buf, static_cast<size_t>(n));
    }
}

static bool parseHttpResponse(const std::string &raw, HttpResponse &out, std::string *errorMessage) {
    const auto lineEnd = raw.find("\r\n");
    if(lineEnd == std::string::npos) {
        setError(errorMessage, "Invalid HTTP response: missing status line");
        return false;
    }
    const std::string statusLine = raw.substr(0, lineEnd);
    std::istringstream iss(statusLine);
    std::string httpVersion;
    iss >> httpVersion >> out.status;
    if(out.status <= 0) {
        setError(errorMessage, "Invalid HTTP response status: " + statusLine);
        return false;
    }
    const auto headerEnd = raw.find("\r\n\r\n");
    if(headerEnd == std::string::npos) {
        setError(errorMessage, "Invalid HTTP response: missing header terminator");
        return false;
    }
    out.body = raw.substr(headerEnd + 4);
    return true;
}

static bool httpRequest(const std::string &baseUrl,
                        int timeoutMs,
                        const std::string &method,
                        const std::string &pathAndQuery,
                        const std::string &body,
                        HttpResponse &response,
                        std::string *errorMessage) {
    ParsedUrl url;
    if(!parseHttpUrl(baseUrl, url, errorMessage)) {
        return false;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    const int gai = ::getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &res);
    if(gai != 0) {
        setError(errorMessage, "Task backend DNS lookup failed: " + std::string(gai_strerror(gai)));
        return false;
    }

    int fd = -1;
    std::string connectError;
    for(addrinfo *it = res; it != nullptr; it = it->ai_next) {
        fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if(fd < 0) {
            connectError = std::strerror(errno);
            continue;
        }
        const int seconds = std::max(1, timeoutMs / 1000);
        timeval tv{};
        tv.tv_sec = seconds;
        tv.tv_usec = static_cast<suseconds_t>((timeoutMs % 1000) * 1000);
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if(::connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
            break;
        }
        connectError = std::strerror(errno);
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);

    if(fd < 0) {
        setError(errorMessage, "Task backend connection failed: " + connectError);
        return false;
    }

    const std::string target = joinBackendPath(url, pathAndQuery);
    std::ostringstream req;
    req << method << " " << target << " HTTP/1.1\r\n"
        << "Host: " << url.host << ":" << url.port << "\r\n"
        << "Accept: application/json\r\n"
        << "Connection: close\r\n";
    if(method == "POST") {
        req << "Content-Type: application/json\r\n";
    }
    req << "Content-Length: " << body.size() << "\r\n\r\n"
        << body;

    std::string rawResponse;
    bool ok = sendAll(fd, req.str(), errorMessage) && recvAll(fd, rawResponse, errorMessage);
    ::close(fd);
    if(!ok) {
        return false;
    }
    return parseHttpResponse(rawResponse, response, errorMessage);
}

static std::string jsonString(cJSON *obj, const char *key) {
    auto *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(item && cJSON_IsString(item) && item->valuestring) {
        return item->valuestring;
    }
    return "";
}

static int jsonInt(cJSON *obj, const char *key, int fallback = 0) {
    auto *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if(item && cJSON_IsNumber(item)) {
        return static_cast<int>(item->valuedouble);
    }
    return fallback;
}

static std::string backendErrorFromBody(const std::string &body) {
    cJSON *root = cJSON_Parse(body.c_str());
    if(!root) {
        return body.empty() ? std::string("empty response body") : body;
    }
    std::string error = jsonString(root, "error");
    cJSON_Delete(root);
    return error.empty() ? body : error;
}

static bool ensureSuccess(const HttpResponse &response, std::string *errorMessage) {
    if(response.status >= 200 && response.status < 300) {
        return true;
    }
    std::ostringstream oss;
    oss << "Task backend HTTP " << response.status << ": " << backendErrorFromBody(response.body);
    setError(errorMessage, oss.str());
    return false;
}

static std::string printJson(cJSON *root) {
    char *printed = cJSON_PrintUnformatted(root);
    if(!printed) {
        return "{}";
    }
    std::string out(printed);
    cJSON_free(printed);
    return out;
}

static bool parseTasksPayload(const std::string &body,
                              std::vector<TaskBackendTask> &tasksOut,
                              std::string *errorMessage) {
    cJSON *root = cJSON_Parse(body.c_str());
    if(!root) {
        setError(errorMessage, "Task backend returned invalid JSON");
        return false;
    }
    auto *tasks = cJSON_GetObjectItemCaseSensitive(root, "tasks");
    if(!tasks || !cJSON_IsArray(tasks)) {
        cJSON_Delete(root);
        setError(errorMessage, "Task backend response missing tasks array");
        return false;
    }
    std::vector<TaskBackendTask> parsed;
    cJSON *item = nullptr;
    cJSON_ArrayForEach(item, tasks) {
        if(!item || !cJSON_IsObject(item)) {
            continue;
        }
        TaskBackendTask task;
        task.taskName = jsonString(item, "task_name");
        task.descriptionCn = jsonString(item, "description_cn");
        task.descriptionEn = jsonString(item, "description_en");
        task.completed = std::max(0, jsonInt(item, "completed", 0));
        task.total = std::max(0, jsonInt(item, "total", 0));
        if(!task.taskName.empty()) {
            parsed.push_back(std::move(task));
        }
    }
    cJSON_Delete(root);
    tasksOut = std::move(parsed);
    return true;
}

}  // namespace

TaskBackendClient::TaskBackendClient(std::string baseUrl, int timeoutMs)
    : baseUrl_(trimSlashesRight(trimAscii(std::move(baseUrl)))),
      timeoutMs_(std::max(500, timeoutMs)) {
    if(baseUrl_.empty()) {
        baseUrl_ = "http://127.0.0.1:8765";
    }
}

bool TaskBackendClient::getTasks(const std::string &subjectId,
                                 std::vector<TaskBackendTask> &tasksOut,
                                 std::string *errorMessage) const {
    HttpResponse response;
    const std::string path = "/api/v1/tasks?subject_id=" + urlEncode(subjectId);
    if(!httpRequest(baseUrl_, timeoutMs_, "GET", path, "", response, errorMessage)) {
        return false;
    }
    if(!ensureSuccess(response, errorMessage)) {
        return false;
    }
    return parseTasksPayload(response.body, tasksOut, errorMessage);
}

bool TaskBackendClient::reserveEpisode(const std::string &clientId,
                                       const std::string &subjectId,
                                       const std::string &taskName,
                                       TaskEpisodeReservation &reservationOut,
                                       std::string *errorMessage) const {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "client_id", clientId.c_str());
    cJSON_AddStringToObject(root, "subject_id", subjectId.c_str());
    cJSON_AddStringToObject(root, "task_name", taskName.c_str());
    const std::string body = printJson(root);
    cJSON_Delete(root);

    HttpResponse response;
    if(!httpRequest(baseUrl_, timeoutMs_, "POST", "/api/v1/episodes/reserve", body, response, errorMessage)) {
        return false;
    }
    if(!ensureSuccess(response, errorMessage)) {
        return false;
    }

    cJSON *parsed = cJSON_Parse(response.body.c_str());
    if(!parsed) {
        setError(errorMessage, "Task backend reserve returned invalid JSON");
        return false;
    }
    TaskEpisodeReservation reservation;
    reservation.reservationId = jsonString(parsed, "reservation_id");
    reservation.taskName = jsonString(parsed, "task_name");
    reservation.episodeNumber = jsonInt(parsed, "episode_number", 0);
    cJSON_Delete(parsed);
    if(reservation.reservationId.empty() || reservation.taskName.empty() || reservation.episodeNumber <= 0) {
        setError(errorMessage, "Task backend reserve response is missing reservation_id/task_name/episode_number");
        return false;
    }
    reservationOut = std::move(reservation);
    return true;
}

bool TaskBackendClient::confirmEpisode(const std::string &reservationId,
                                       const std::string &subjectId,
                                       const std::string &taskName,
                                       int episodeNumber,
                                       const std::string &localPath,
                                       double durationSeconds,
                                       int frameCount,
                                       const std::string &idempotencyKey,
                                       std::vector<TaskBackendTask> &tasksOut,
                                       std::string *errorMessage) const {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "reservation_id", reservationId.c_str());
    cJSON_AddStringToObject(root, "subject_id", subjectId.c_str());
    cJSON_AddStringToObject(root, "task_name", taskName.c_str());
    cJSON_AddNumberToObject(root, "episode_number", episodeNumber);
    cJSON_AddStringToObject(root, "local_path", localPath.c_str());
    if(durationSeconds > 0.0) {
        cJSON_AddNumberToObject(root, "duration_seconds", durationSeconds);
    }
    if(frameCount > 0) {
        cJSON_AddNumberToObject(root, "frame_count", frameCount);
    }
    cJSON_AddStringToObject(root, "idempotency_key", idempotencyKey.c_str());
    const std::string body = printJson(root);
    cJSON_Delete(root);

    HttpResponse response;
    if(!httpRequest(baseUrl_, timeoutMs_, "POST", "/api/v1/episodes/confirm", body, response, errorMessage)) {
        return false;
    }
    if(!ensureSuccess(response, errorMessage)) {
        return false;
    }
    return parseTasksPayload(response.body, tasksOut, errorMessage);
}

bool TaskBackendClient::releaseEpisode(const std::string &reservationId,
                                       const std::string &subjectId,
                                       const std::string &taskName,
                                       std::string *errorMessage) const {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "reservation_id", reservationId.c_str());
    cJSON_AddStringToObject(root, "subject_id", subjectId.c_str());
    cJSON_AddStringToObject(root, "task_name", taskName.c_str());
    const std::string body = printJson(root);
    cJSON_Delete(root);

    HttpResponse response;
    if(!httpRequest(baseUrl_, timeoutMs_, "POST", "/api/v1/episodes/release", body, response, errorMessage)) {
        return false;
    }
    return ensureSuccess(response, errorMessage);
}

}  // namespace sync_app
