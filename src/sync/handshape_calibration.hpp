#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace sync_app {
namespace handshape {
inline constexpr const char *taskName = "task_handshapeCalibration";
inline constexpr const char *episodeName = "episode1";

inline std::filesystem::path episodePath(const std::filesystem::path &root, const std::string &subject) {
    namespace fs = std::filesystem;
    if(root.empty() || subject.empty() || subject == "." || subject == ".."
       || subject.find_first_of("/\\") != std::string::npos) {
        throw std::runtime_error("Invalid handshape calibration root or subject ID");
    }
    fs::path path = root;
    for(const auto &part: {subject, std::string(taskName), std::string(episodeName)}) {
        path /= part;
        if(fs::is_symlink(fs::symlink_status(path))) {
            throw std::runtime_error("Handshape calibration path must not contain symlinks: " + path.string());
        }
    }
    return path;
}

inline void prepareEpisode(const std::filesystem::path &root, const std::string &subject, bool overwrite) {
    const auto path = episodePath(root, subject);
    if(overwrite) {
        std::filesystem::remove_all(path);
    }
    std::filesystem::create_directories(path);
}

// One capture at a time for a subject. The lock is outside the replaced episode.
class CaptureLock {
public:
    CaptureLock() = default;
    CaptureLock(const CaptureLock &) = delete;
    CaptureLock &operator=(const CaptureLock &) = delete;
    void acquire(const std::filesystem::path &root, const std::string &subject) {
        const auto path = episodePath(root, subject);
        std::filesystem::create_directories(path.parent_path());
#if defined(__unix__) || defined(__APPLE__)
        fd_ = ::open((path.parent_path() / ".capture.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if(fd_ < 0 || ::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            throw std::runtime_error("Handshape calibration is already open for this subject, or its lock is unavailable");
        }
#endif
    }
    ~CaptureLock() {
#if defined(__unix__) || defined(__APPLE__)
        if(fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); }
#endif
    }
private:
    int fd_ = -1;
};
}  // namespace handshape
}  // namespace sync_app
