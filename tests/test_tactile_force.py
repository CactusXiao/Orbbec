"""Hardware-free regression checks for calibration and saved force measurements."""
from pathlib import Path
import csv
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TactileForceTest(unittest.TestCase):
    def test_mapping_and_saved_csv(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            program = work / "test.cpp"
            program.write_text(r'''
#include "tactile.hpp"
#include "tactile_layout.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace sync_app;
int main(int argc, char **argv) {
    std::filesystem::path work = argv[1], repo = argv[2];
    auto fixture = work / "calibration.csv";
    std::filesystem::copy_file(repo / "tactile/5 (1).csv", fixture);
    std::filesystem::permissions(fixture, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    // Export the capture-side anatomy for comparison against the viewer map.
    std::ofstream layout(work / "layout.csv");
    layout << "side,id,region,kind,point\n";
    for(const std::string side : {"left", "right"})
        for(int id=1; id<=256; ++id) {
            auto c = gloveChannel(side, id);
            layout << side << "," << id << "," << c.region << "," << c.kind << "," << c.point << "\n";
        }
    TactileForceCalibration model;
    std::string error;
    auto module = createPosixSerialTactileModule();
    TactileModuleConfig config;
    config.handSide = "left"; // Defaults to type 2: fail before opening hardware.
    assert(!module->start(config, &error));
    assert(error.find("handSide must match") != std::string::npos);
    assert(model.load({fixture}, &error));
    assert(model.forceN(0) == 0);
    assert(std::abs(model.forceN(20) - 0.5336303022434962) < 1e-12);
    assert(std::abs(model.forceN(60) - 1.4053247333388768) < 1e-12);
    assert(std::abs(model.forceN(700) - 22.140611132788685) < 1e-9);
    assert(std::abs(model.forceN(1100) - 96.33705109110647) < 1e-9);
    assert(std::abs(model.forceN(1200) - 246.95503452513867) < 1e-9);
    assert(std::abs(model.forceN(1245) - 965.829767739969) < 1e-9);
    assert(std::isnan(model.forceN(-1)));
    assert(std::isnan(model.forceN(std::numeric_limits<double>::infinity())));
    assert(std::isnan(model.forceN(std::numeric_limits<double>::quiet_NaN())));
    assert(std::isnan(model.forceN(1257.0210)));
    assert(std::isnan(model.forceN(1258)));
    assert(std::isfinite(model.forceN(std::nextafter(1257.0210, 0.0))));
    // Forward Hill response from the image must round-trip through the inverse.
    for(double force : {0.1, 1.0, 10.0, 18.2084, 100.0, 250.0}) {
        const double pn = std::pow(force, 1.1685);
        const double adc = 1257.0210 * pn / (std::pow(18.2084, 1.1685) + pn);
        assert(std::abs(model.forceN(adc) - force) < 1e-8);
    }
    TactileSample sample;
    sample.representativeTimestampUs = 1234567;
    sample.representativeTimestampSec = 1.234567;
    sample.frame.side = "right";
    sample.frame.sensorType = 2;
    sample.frame.rawAdc.assign(256, 0);
    sample.frame.rawAdc[7] = 10;
    sample.frame.rawAdc[249] = 30;
    sample.frame.rawAdc[40] = 255; // A bend/unmapped channel must not affect force.
    assert(model.apply(sample.frame, &error));
    const double expectedForce = 0.9793085705056488;
    assert(std::abs(sample.frame.calibratedRegionForceN - expectedForce) < 1e-12);
    assert(sample.frame.forceCalibrationStatus == "right_middle_region");
    assert(!sample.frame.forceOutOfRange);
    assert(TactileRecorder::saveSamples({sample}, work / "samples", {}, nullptr, &error));
    std::ofstream saved(work / "collection.csv");
    saved << "sample_index";
    writeTactileMeasurementCsvHeader(saved);
    saved << "\n0";
    writeTactileMeasurementCsvValues(saved, sample.frame, 6);
    saved << "\n";
    saved.close();
    sample.frame.rawAdc[7] = 255;
    sample.frame.rawAdc[249] = 255;
    assert(model.apply(sample.frame));
    assert(!sample.frame.forceOutOfRange && sample.frame.calibratedRegionForceN == model.forceN(510));
    sample.frame.rawAdc.assign(256, 0);
    assert(model.apply(sample.frame));
    assert(sample.frame.calibratedRegionForceN == 0 && !sample.frame.forceOutOfRange);
    // The same IDs refer to 8 middle + 4 ring points on the left. Never convert.
    sample.frame.side = "left";
    sample.frame.sensorType = 1;
    for(size_t i : model.channelIndices()) sample.frame.rawAdc[i] = 50;
    assert(model.apply(sample.frame));
    assert(std::isnan(sample.frame.calibratedRegionForceN));
    assert(sample.frame.forceCalibrationStatus == "uncalibrated_hand");
    assert(sample.frame.rawAdc[7] == 50);
    assert(TactileRecorder::saveSamples({sample}, work / "left", {}, nullptr, &error));
    // Conflicting/unknown identity also must never receive right-hand N.
    sample.frame.side = "right";
    assert(model.apply(sample.frame) && std::isnan(sample.frame.calibratedRegionForceN));
    sample.frame.sensorType = 2;
    sample.frame.rawAdc.resize(10);
    assert(!model.apply(sample.frame));
    sample.frame.rawAdc.resize(250);
    assert(!model.apply(sample.frame)); // Even though all calibration IDs fit.
    // CSV force values and reversals must not refit or alter the image parameters.
    std::ofstream(fixture) << "time,force(N),sensor#8\n"
                          << "t,10,10\nt,30,10\nt,10,20\nt,40,30\n";
    assert(!model.load({fixture})); // Cannot apply this image fit to arbitrary IDs.
    for(const auto &bad : {"t,nan,10", "t,-1,10", "t,1,256", "t,1,2junk", "t,1"}) {
        std::ofstream(fixture) << "time,force(N),sensor#8\n" << bad << "\n";
        assert(!model.load({fixture}, &error));
        assert(!error.empty() && std::isnan(model.forceN(10)));
    }
    assert(!model.load({work / "missing.csv"}));
    assert(!model.load({}));
    // All three supplied CSVs, including GBK/CRLF, must use the same fixed fit.
    std::vector<std::filesystem::path> paths;
    for(int i = 1; i <= 3; ++i) {
        paths.push_back(repo / "tactile" / ("5 (" + std::to_string(i) + ").csv"));
        assert(model.load({paths.back()}, &error));
        assert(std::abs(model.forceN(1200) - 246.95503452513867) < 1e-9);
        assert(model.channelIndices().size() == 12);
        assert(model.channelIndices().front() == 7 && model.channelIndices().back() == 249);
    }
    assert(model.load(paths, &error));
    double previous = 0;
    for(int adc = 0; adc <= 1257; ++adc) {
        const double force = model.forceN(adc);
        assert(std::isfinite(force) && force >= previous);
        previous = force;
    }
    assert(model.forceN(0) == 0);
    // Extrapolation remains finite below a; saturation saves NaN, not a clipped force.
    sample.frame.rawAdc.assign(256, 0);
    for(size_t i : model.channelIndices()) sample.frame.rawAdc[i] = 100;
    assert(model.apply(sample.frame));
    assert(!sample.frame.forceOutOfRange);
    assert(std::abs(sample.frame.calibratedRegionForceN - 246.95503452513867) < 1e-9);
    for(size_t i : model.channelIndices()) sample.frame.rawAdc[i] = 104;
    assert(model.apply(sample.frame));
    assert(sample.frame.forceOutOfRange && std::isfinite(sample.frame.calibratedRegionForceN));
    for(size_t i : model.channelIndices()) sample.frame.rawAdc[i] = 108;
    assert(model.apply(sample.frame));
    assert(sample.frame.forceOutOfRange && std::isnan(sample.frame.calibratedRegionForceN));
    std::ofstream invalid(work / "saturated.csv");
    invalid << "sample_index";
    writeTactileMeasurementCsvHeader(invalid);
    invalid << "\n0";
    writeTactileMeasurementCsvValues(invalid, sample.frame, 6);
    invalid << "\n";
    assert(TactileRecorder::saveSamples({sample}, work / "saturated", {}, nullptr, &error));
}
''')
            executable = work / "test"
            subprocess.run([
                compiler, "-std=c++17", "-Wall", "-Wextra", "-pthread",
                "-I", str(ROOT / "src/sync"), str(program),
                str(ROOT / "src/sync/tactile.cpp"), "-o", str(executable),
            ], check=True, capture_output=True, text=True)
            result = subprocess.run([str(executable), str(work), str(ROOT)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (work / "collection.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(len(row), 260)
            self.assertNotIn("force_007_n", row)
            self.assertEqual(row["force_calibration_status"], "right_middle_region")
            self.assertEqual(row["calibrated_region_force_n"], "0.979309")
            self.assertEqual(row["raw_adc_040"], "255")
            with (work / "samples/samples/1.234567.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertNotIn("output_value", rows[7])
            self.assertEqual(rows[7]["region_name"], "Middle")
            self.assertEqual(rows[40]["channel_kind"], "bend")
            self.assertEqual(rows[40]["region_name"], "Middle")
            self.assertEqual(rows[7]["region_force_unit"], "N")
            self.assertEqual(rows[7]["raw_adc"], "10")
            from task_backend.tactile_layout import FINGERS, PALM_ROWS, BEND_IDS
            with (work / "layout.csv").open() as f:
                layout = list(csv.DictReader(f))
            for side in ("left", "right"):
                expected = {}
                for finger, ids in enumerate(FINGERS[side], 1):
                    expected.update({i: (finger, "pressure", point) for point, i in enumerate(ids, 1)})
                palm = [i for row in PALM_ROWS[side] for i in row]
                expected.update({i: (6, "pressure", point) for point, i in enumerate(palm, 1)})
                expected.update({i: (finger, "bend", 1) for finger, i in enumerate(BEND_IDS[side], 1)})
                for row in (r for r in layout if r["side"] == side):
                    self.assertEqual((int(row["region"]), row["kind"], int(row["point"])),
                                     expected.get(int(row["id"]), (0, "unmapped", 0)))
            with (work / "left/samples/1.234567.csv").open() as f:
                left = list(csv.DictReader(f))
            self.assertEqual(left[249]["region_name"], "Ring")
            self.assertEqual(left[215]["channel_kind"], "bend")
            self.assertEqual(left[7]["force_calibration_status"], "uncalibrated_hand")
            self.assertEqual(left[7]["calibrated_region_force_n"], "nan")
            with (work / "saturated.csv").open() as f:
                invalid = next(csv.DictReader(f))
            self.assertEqual(invalid["calibrated_region_force_n"], "nan")
            self.assertNotIn("force_007_n", invalid)
            self.assertEqual(invalid["force_out_of_range"], "1")
            self.assertEqual(invalid["raw_adc_007"], "108")
            with (work / "saturated/samples/1.234567.csv").open() as f:
                invalid_rows = list(csv.DictReader(f))
            self.assertNotIn("output_value", invalid_rows[7])
            self.assertEqual(invalid_rows[7]["force_out_of_range"], "1")


if __name__ == "__main__":
    unittest.main()
