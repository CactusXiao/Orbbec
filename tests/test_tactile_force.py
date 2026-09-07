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
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace sync_app;
int main(int argc, char **argv) {
    std::filesystem::path work = argv[1], repo = argv[2];
    auto fixture = work / "calibration.csv";
    std::ofstream(fixture) << "time,force(N),sensor#8,sensor#250\n"
                          << "t,0,0,0\nt,-0.24,0,0\nt,10,10,30\nt,20,20,60\n";
    TactileForceCalibration model;
    std::string error;
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
    sample.frame.rawAdc.assign(256, 0);
    sample.frame.rawAdc[7] = 10;
    sample.frame.rawAdc[249] = 30;
    sample.frame.rawAdc[40] = 255; // A bend/unmapped channel must not affect force.
    assert(model.apply(sample.frame, &error));
    const double expectedForce = 0.9793085705056488;
    assert(std::abs(sample.frame.calibratedRegionForceN - expectedForce) < 1e-12);
    assert(std::abs(sample.frame.outputValues[7] - expectedForce / 4) < 1e-12);
    assert(std::abs(sample.frame.outputValues[249] - expectedForce * 3 / 4) < 1e-12);
    assert(std::isnan(sample.frame.outputValues[40]));
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
    assert(sample.frame.forceOutOfRange && sample.frame.calibratedRegionForceN == model.forceN(510));
    sample.frame.rawAdc.assign(256, 0);
    assert(model.apply(sample.frame));
    assert(sample.frame.outputValues[7] == 0 && !sample.frame.forceOutOfRange);
    sample.frame.rawAdc.resize(10);
    assert(!model.apply(sample.frame));
    // CSV force values and reversals must not refit or alter the image parameters.
    std::ofstream(fixture) << "time,force(N),sensor#8\n"
                          << "t,10,10\nt,30,10\nt,10,20\nt,40,30\n";
    assert(model.load({fixture}));
    assert(std::abs(model.forceN(20) - 0.5336303022434962) < 1e-12);
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
    double allocated = 0;
    for(size_t i : model.channelIndices()) allocated += sample.frame.outputValues[i];
    assert(std::abs(allocated - sample.frame.calibratedRegionForceN) < 1e-9);
    for(size_t i : model.channelIndices()) sample.frame.rawAdc[i] = 104;
    assert(model.apply(sample.frame));
    assert(sample.frame.forceOutOfRange && std::isfinite(sample.frame.calibratedRegionForceN));
    for(size_t i : model.channelIndices()) sample.frame.rawAdc[i] = 108;
    assert(model.apply(sample.frame));
    assert(sample.frame.forceOutOfRange && std::isnan(sample.frame.calibratedRegionForceN));
    for(size_t i : model.channelIndices()) assert(std::isnan(sample.frame.outputValues[i]));
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
            self.assertEqual(len(row), 515)
            self.assertEqual(row["force_007_n"], "0.244827")
            self.assertEqual(row["force_249_n"], "0.734481")
            self.assertEqual(row["calibrated_region_force_n"], "0.979309")
            self.assertEqual(row["force_040_n"], "nan")
            self.assertEqual(row["raw_adc_040"], "255")
            with (work / "samples/samples/1.234567.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[7]["output_value"], "0.244827")
            self.assertEqual(rows[7]["force_unit"], "N")
            self.assertEqual(rows[7]["raw_adc"], "10")
            with (work / "saturated.csv").open() as f:
                invalid = next(csv.DictReader(f))
            self.assertEqual(invalid["calibrated_region_force_n"], "nan")
            self.assertEqual(invalid["force_007_n"], "nan")
            self.assertEqual(invalid["force_out_of_range"], "1")
            self.assertEqual(invalid["raw_adc_007"], "108")
            with (work / "saturated/samples/1.234567.csv").open() as f:
                invalid_rows = list(csv.DictReader(f))
            self.assertEqual(invalid_rows[7]["output_value"], "nan")
            self.assertEqual(invalid_rows[7]["force_out_of_range"], "1")


if __name__ == "__main__":
    unittest.main()
