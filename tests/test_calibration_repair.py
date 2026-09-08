"""Run hardware-independent calibration regression tests (C++17 and OpenCV required)."""
import pathlib
import shlex
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class CalibrationRepairTest(unittest.TestCase):
    def test_repair_and_safe_save(self):
        flags = shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "opencv4"], text=True))
        with tempfile.TemporaryDirectory(prefix="calibration-repair-") as tmp:
            executable = str(pathlib.Path(tmp) / "calibration_repair_test")
            subprocess.run([
                "c++", "-std=c++17", "-O0", "-g", "-I", str(ROOT / "src/sync"),
                str(ROOT / "tests/calibration_repair_test.cpp"),
                str(ROOT / "src/sync/utils/cJSON.c"), "-o", executable, *flags,
            ], check=True)
            subprocess.run([executable, tmp], check=True)


if __name__ == "__main__":
    unittest.main()
