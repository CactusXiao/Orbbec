#pragma once

#include <string>
#include <vector>

namespace sync_app {

struct TaskBackendTask {
    std::string taskName;
    std::string descriptionCn;
    std::string descriptionEn;
    int completed = 0;
    int total = 0;
};

struct TaskEpisodeReservation {
    std::string reservationId;
    std::string taskName;
    int episodeNumber = 0;
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
                        const std::string &localPath,
                        const std::string &idempotencyKey,
                        std::vector<TaskBackendTask> &tasksOut,
                        std::string *errorMessage = nullptr) const;

    bool releaseEpisode(const std::string &reservationId,
                        const std::string &subjectId,
                        const std::string &taskName,
                        std::string *errorMessage = nullptr) const;

private:
    std::string baseUrl_;
    int timeoutMs_ = 3000;
};

}  // namespace sync_app
