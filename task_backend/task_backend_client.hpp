#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace sync_app {

struct TaskBackendTask {
    std::string taskName;
    std::string descriptionCn;
    std::string descriptionEn;
    int completed = 0;
    int total = 0;
    std::string claimedBySubject;
    bool claimedByOther = false;
};

struct TaskEpisodeReservation {
    std::string reservationId;
    std::string taskName;
    int episodeNumber = 0;
};

struct TaskUploadStatus {
    bool        available = false;
    std::string episodeId;
    std::string jobId;
    std::string jobStatus;
    std::string phase;
    double      percent = 0.0;
    uint64_t    copiedBytes = 0;
    uint64_t    totalBytes = 0;
    int         filesDone = 0;
    int         filesTotal = 0;
    std::string collectionPath;
    std::string nasUri;
    std::string error;
    std::string updatedAt;
};

class TaskBackendClient {
public:
    explicit TaskBackendClient(std::string baseUrl, int timeoutMs = 3000);

    const std::string &baseUrl() const { return baseUrl_; }

    bool getTasks(const std::string &subjectId,
                  std::vector<TaskBackendTask> &tasksOut,
                  std::string *errorMessage = nullptr) const;

    bool reserveEpisode(const std::string &clientId,
                        const std::string &subjectId,
                        const std::string &taskName,
                        TaskEpisodeReservation &reservationOut,
                        std::string *errorMessage = nullptr) const;

    bool confirmEpisode(const std::string &reservationId,
                        const std::string &subjectId,
                        const std::string &taskName,
                        int episodeNumber,
                        const std::string &collectionPath,
                        double durationSeconds,
                        int frameCount,
                        const std::string &idempotencyKey,
                        std::vector<TaskBackendTask> &tasksOut,
                        std::string *errorMessage = nullptr) const;

    bool releaseEpisode(const std::string &reservationId,
                        const std::string &subjectId,
                        const std::string &taskName,
                        std::string *errorMessage = nullptr) const;

    bool getUploadStatus(const std::string &episodeId,
                         TaskUploadStatus &statusOut,
                         std::string *errorMessage = nullptr) const;

private:
    std::string baseUrl_;
    int timeoutMs_ = 3000;
};

}  // namespace sync_app
