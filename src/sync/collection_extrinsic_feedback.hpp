#pragma once

#include <string_view>

namespace sync_app::extrinsic_feedback {

inline constexpr const char *failureMessageKey = "extrinsic_check_failed";
inline constexpr const char *failureMessage = "外参检查失败，请调整相机后按F2重新采样";

inline bool resampleOnFailure(int key, bool resampleAllowed, std::string_view status) {
    if(!resampleAllowed || status != "fail" || key < 0) return false;
    // GTK waitKeyEx may include lock modifiers; waitKey returns the low byte.
    return (key & 0xFFFF) == 0xFFBF || key == 0xBF;
}

} // namespace sync_app::extrinsic_feedback
