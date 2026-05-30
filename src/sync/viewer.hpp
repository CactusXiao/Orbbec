#pragma once

#include "shared_utils.hpp"

namespace sync_app {

int run_viewer(const AppConfig &cfg, const std::atomic_bool *cancel);

}

