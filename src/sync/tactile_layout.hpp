#pragma once
#include <array>
#include <string>

namespace sync_app {
// Fabric electronic skin (tactile glove) manual, pp. 11–13. One-based IDs.
// Keep backend tactile_layout.py in agreement; regression tests compare both.
inline constexpr std::array<std::array<int, 12>, 5> kGloveRightFingers{{
    {{240,239,238,256,255,254,16,15,14,32,31,30}},
    {{237,236,235,253,252,251,13,12,11,29,28,27}},
    {{234,233,232,250,249,248,10,9,8,26,25,24}},
    {{231,230,229,247,246,245,7,6,5,23,22,21}},
    {{228,227,226,244,243,242,4,3,2,20,19,18}}
}};
inline constexpr std::array<std::array<int, 12>, 5> kGloveLeftFingers{{
    {{19,18,17,3,2,1,243,242,241,227,226,225}},
    {{22,21,20,6,5,4,246,245,244,230,229,228}},
    {{25,24,23,9,8,7,249,248,247,233,232,231}},
    {{28,27,26,12,11,10,252,251,250,236,235,234}},
    {{31,30,29,15,14,13,255,254,253,239,238,237}}
}};
inline constexpr std::array<int, 5> kGloveRightBends{{47,44,41,38,35}};
inline constexpr std::array<int, 5> kGloveLeftBends{{210,213,216,219,222}};

struct GloveChannel {
    int region = 0;
    const char *name = "Unmapped";
    const char *kind = "unmapped";
    int point = 0;
};

inline GloveChannel gloveChannel(const std::string &side, int id) {
    if(side != "left" && side != "right") return {};
    const bool right = side == "right";
    const auto &fingers = right ? kGloveRightFingers : kGloveLeftFingers;
    const auto &bends = right ? kGloveRightBends : kGloveLeftBends;
    constexpr const char *names[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
    for(int f = 0; f < 5; ++f) {
        if(id == bends[f]) return {f + 1, names[f], "bend", 1};
        for(int p = 0; p < 12; ++p)
            if(id == fingers[f][p]) return {f + 1, names[f], "pressure", p + 1};
    }
    const std::array<int, 5> starts = right ? std::array<int, 5>{{61,80,96,112,128}}
                                          : std::array<int, 5>{{207,191,175,159,143}};
    int point = 0;
    for(int row = 0; row < 5; ++row) {
        const int count = row == 0 ? 12 : 15;
        for(int col = 0; col < count; ++col) {
            ++point;
            if(id == starts[row] - col) return {6, "Palm", "pressure", point};
        }
    }
    return {};
}
} // namespace sync_app
