#include "collection_extrinsic_feedback.hpp"
#include <cassert>
#include <iostream>

using sync_app::extrinsic_feedback::resampleOnFailure;

int main() {
    for(int f2 : {0xFFBF, 0xBF, 0x10FFBF, 0x14FFBF}) {
        assert(resampleOnFailure(f2, true, "fail"));
        // Recording, draining, pending sampling and dialogs must gate resampling.
        assert(!resampleOnFailure(f2, false, "fail"));
        for(const char *status : {"pass", "warn", "inconclusive", "error", ""}) {
            assert(!resampleOnFailure(f2, true, status));
        }
    }
    for(int other : {-1, 0, int('2'), 0xFFBE, 0xBE, 0xFFC0, 0xC0, 0x140032}) {
        assert(!resampleOnFailure(other, true, "fail"));
    }
    std::cout << "PASS: F2 resamples only an actionable failed extrinsic check\n";
}
