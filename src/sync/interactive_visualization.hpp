#pragma once

#include "shared_utils.hpp"

namespace sync_app {

enum class InteractiveExit {
    ReturnMenu,
    ReturnConfig,
    StartCollection,
    Quit
};

InteractiveExit run_interactive_visualization(const AppConfig &cfg, const std::atomic_bool *cancel);

}
