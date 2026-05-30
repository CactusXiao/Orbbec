#pragma once

#include "shared_utils.hpp"

namespace sync_app {

int run_collection(const AppConfig &cfg, const std::atomic_bool *cancel);

}

