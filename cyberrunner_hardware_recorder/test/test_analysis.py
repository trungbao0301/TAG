import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "cyberrunner_hardware_recorder"
    / "analyze.py"
)
SPEC = importlib.util.spec_from_file_location("passive_analysis", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_timing_summary_reports_rate_and_gap():
    times = np.asarray([0, 10, 20, 40], dtype=np.int64) * 1_000_000
    result = ANALYSIS._timing_summary(times)
    assert result["messages"] == 4
    assert result["rate_hz"] == 75.0
    assert result["estimated_missing_messages"] == 1


def test_range_summary_ignores_nonfinite_values():
    result = ANALYSIS._range_summary([1.0, float("nan"), 3.0])
    assert result["count"] == 2
    assert result["min"] == 1.0
    assert result["max"] == 3.0
