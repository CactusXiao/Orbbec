#pragma once

#include "shared_utils.hpp"

namespace sync_app {

int run_calibration(const AppConfig &cfg, const std::atomic_bool *cancel);

}

