#!/usr/bin/env python3

"""Analyze a passive CyberRunner hardware recording session."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _floats(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _ints(rows, key):
    return np.asarray([int(row[key]) for row in rows], dtype=np.int64)


def _number(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_number(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_number(item) for item in value]
    return value


def _timing_summary(receipt_ns):
    receipt_ns = np.asarray(receipt_ns, dtype=np.int64)
    if receipt_ns.size < 2:
        return {
            "messages": int(receipt_ns.size),
            "duration_sec": 0.0,
            "rate_hz": None,
            "interval_ms": None,
            "estimated_missing_messages": None,
        }
    intervals = np.diff(receipt_ns).astype(np.float64) / 1e6
    duration = (receipt_ns[-1] - receipt_ns[0]) / 1e9
    median = float(np.median(intervals))
    expected = max(median, 1e-9)
    expected_messages = int(round(duration * 1000.0 / expected)) + 1
    return {
        "messages": int(receipt_ns.size),
        "duration_sec": float(duration),
        "rate_hz": float((receipt_ns.size - 1) / duration) if duration > 0 else None,
        "interval_ms": {
            "mean": float(np.mean(intervals)),
            "std": float(np.std(intervals)),
            "median": median,
            "p95": float(np.percentile(intervals, 95)),
            "p99": float(np.percentile(intervals, 99)),
            "max": float(np.max(intervals)),
        },
        "estimated_missing_messages": max(expected_messages - receipt_ns.size, 0),
        "gaps_over_2x_median": int(np.sum(intervals > 2.0 * expected)),
    }


def _range_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
    }


def _source_timing(rows):
    stamps = []
    for row in rows:
        value = row.get("source_stamp_ns", "")
        if not value:
            continue
        try:
            stamp = int(value)
        except ValueError:
            continue
        if stamp > 0:
            stamps.append(stamp)
    stamps = np.asarray(stamps, dtype=np.int64)
    result = _timing_summary(stamps)
    result["nonmonotonic_or_duplicate"] = (
        int(np.sum(np.diff(stamps) <= 0)) if stamps.size > 1 else 0
    )
    return result


def _matched_receipt_delay(camera_rows, subimg_rows):
    camera_receipts = {}
    for row in camera_rows:
        stamp = row.get("source_stamp_ns", "")
        if stamp and stamp not in camera_receipts:
            camera_receipts[stamp] = int(row["receipt_monotonic_ns"])
    delays = []
    for row in subimg_rows:
        stamp = row.get("source_stamp_ns", "")
        if stamp in camera_receipts:
            delays.append(
                (
                    int(row["receipt_monotonic_ns"]) - camera_receipts[stamp]
                )
                / 1e6
            )
    if not delays and camera_rows and subimg_rows:
        camera_ns = _ints(camera_rows, "receipt_monotonic_ns")
        subimg_ns = _ints(subimg_rows, "receipt_monotonic_ns")
        indices = np.searchsorted(camera_ns, subimg_ns, side="right") - 1
        valid = indices >= 0
        delays = ((subimg_ns[valid] - camera_ns[indices[valid]]) / 1e6).tolist()
        method = "latest_prior_camera_receipt"
        warning = (
            "Subimage headers are unstamped, so this is age since the latest "
            "camera receipt seen by this best-effort recorder. Missed camera "
            "receipts can overestimate it; it is not TCP round-trip latency."
        )
    else:
        method = "matching_source_stamp"
        warning = (
            "Receipt-to-receipt estimator pipeline delay for matching image "
            "header stamps; not TCP round-trip latency."
        )
    if not delays:
        return {"matched_messages": 0, "delay_ms": None, "method": "unavailable"}
    return {
        "matched_messages": len(delays),
        "delay_ms": _range_summary(delays),
        "method": method,
        "warning": warning,
    }


def _longest_false_run(receipt_ns, detected):
    start = None
    longest = 0.0
    for index, value in enumerate(detected):
        if not value and start is None:
            start = index
        if value and start is not None:
            longest = max(
                longest, (receipt_ns[index - 1] - receipt_ns[start]) / 1e9
            )
            start = None
    if start is not None and len(detected):
        longest = max(longest, (receipt_ns[-1] - receipt_ns[start]) / 1e9)
    return float(longest)


def _held_commands(
    state_ns, command_ns, command_values, lag_sec, max_command_age_sec=1.0
):
    query_ns = state_ns - int(lag_sec * 1e9)
    indices = np.searchsorted(command_ns, query_ns, side="right") - 1
    valid = indices >= 0
    age_ns = np.zeros(state_ns.size, dtype=np.int64)
    age_ns[valid] = query_ns[valid] - command_ns[indices[valid]]
    valid &= age_ns <= int(max_command_age_sec * 1e9)
    output = np.full((state_ns.size, command_values.shape[1]), np.nan)
    output[valid] = command_values[indices[valid]]
    return output


def _fit_response(y, commands):
    valid = np.isfinite(y) & np.all(np.isfinite(commands), axis=1)
    if np.sum(valid) < 20:
        return None
    x = np.column_stack([commands[valid], np.ones(np.sum(valid))])
    target = y[valid]
    coefficients, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
    prediction = x @ coefficients
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 0.0
    return {
        "gain_cmd_1": float(coefficients[0]),
        "gain_cmd_2": float(coefficients[1]),
        "intercept": float(coefficients[2]),
        "r2": float(r2),
        "samples": int(np.sum(valid)),
    }


def _passive_dynamics(state_rows, command_rows):
    warning = (
        "Closed-loop passive estimate only. Commands are policy-correlated with "
        "state; gains and delays are not causal identification results."
    )
    if len(state_rows) < 50 or len(command_rows) < 20:
        return {"available": False, "reason": "insufficient samples", "warning": warning}

    state_ns = _ints(state_rows, "receipt_monotonic_ns")
    command_ns = _ints(command_rows, "receipt_monotonic_ns")
    commands = np.column_stack(
        [_floats(command_rows, "vel_1"), _floats(command_rows, "vel_2")]
    )
    alpha = _floats(state_rows, "alpha")
    beta = _floats(state_rows, "beta")

    order = np.argsort(command_ns)
    command_ns = command_ns[order]
    commands = commands[order]
    if np.nanstd(commands[:, 0]) < 1e-6 and np.nanstd(commands[:, 1]) < 1e-6:
        return {
            "available": False,
            "reason": "commands have no useful variation",
            "warning": warning,
        }

    candidates = np.arange(0.0, 0.501, 0.01)
    results = {}
    for name, response, primary_index in (
        ("alpha", alpha, 0),
        ("beta", beta, 1),
    ):
        best = None
        for lag in candidates:
            held = _held_commands(state_ns, command_ns, commands, lag)
            fit = _fit_response(response, held)
            if fit is None:
                continue
            fit["delay_sec"] = float(lag)
            if best is None or fit["r2"] > best["r2"]:
                best = fit
        if best is None:
            results[name] = {"available": False}
            continue
        gains = np.asarray(
            [abs(best["gain_cmd_1"]), abs(best["gain_cmd_2"])],
            dtype=np.float64,
        )
        dominant_index = int(np.argmax(gains))
        secondary_index = 1 - dominant_index
        best["nominal_primary_command"] = primary_index + 1
        best["dominant_command"] = dominant_index + 1
        best["secondary_to_dominant_gain_ratio"] = (
            float(gains[secondary_index] / gains[dominant_index])
            if gains[dominant_index] > 1e-12
            else None
        )
        nominal_primary = gains[primary_index]
        nominal_cross = gains[1 - primary_index]
        best["nominal_cross_axis_gain_ratio"] = (
            float(nominal_cross / nominal_primary)
            if nominal_primary > 1e-12
            else None
        )
        best["gain_units"] = "radians per command unit"
        results[name] = best

    return {
        "available": True,
        "responses": results,
        "command_hold_max_age_sec": 1.0,
        "warning": warning,
    }


def analyze_session(session_dir):
    session_dir = Path(session_dir).expanduser().resolve()
    metadata = json.loads((session_dir / "session_metadata.json").read_text())
    camera_rows = _read_csv(session_dir / "camera_timing.csv")
    state_rows_all = _read_csv(session_dir / "state.csv")
    command_rows = _read_csv(session_dir / "motor_commands.csv")
    episode_rows = _read_csv(session_dir / "episodes.csv")
    primary_topic = metadata["topics"]["state"]
    subimg_topic = metadata["topics"].get("state_subimg")
    primary_rows = [
        row for row in state_rows_all if row["source_topic"] == primary_topic
    ]
    subimg_rows = [
        row for row in state_rows_all if row["source_topic"] == subimg_topic
    ]

    camera_ns = _ints(camera_rows, "receipt_monotonic_ns") if camera_rows else []
    primary_ns = _ints(primary_rows, "receipt_monotonic_ns") if primary_rows else []
    subimg_ns = _ints(subimg_rows, "receipt_monotonic_ns") if subimg_rows else []
    command_ns = _ints(command_rows, "receipt_monotonic_ns") if command_rows else []
    detected = (
        np.asarray([row["ball_detected"] == "1" for row in primary_rows], dtype=bool)
        if primary_rows
        else np.asarray([], dtype=bool)
    )

    limits = metadata.get("command_limits", {})
    limit_1 = abs(float(limits.get("vel_1", [-180, 180])[1]))
    limit_2 = abs(float(limits.get("vel_2", [-180, 180])[1]))
    vel_1 = _floats(command_rows, "vel_1")
    vel_2 = _floats(command_rows, "vel_2")
    session_duration = float(metadata.get("duration_sec") or 0.0)

    episodes = {
        "count": len(episode_rows),
        "official_dreamer_events": False,
        "inference_source": "ball_visibility",
    }
    if episode_rows:
        durations = _floats(episode_rows, "duration_sec")
        episodes.update(
            {
                "duration_sec": _range_summary(durations),
                "count_at_least_0_35_sec": int(np.sum(durations >= 0.35)),
                "count_at_least_1_sec": int(np.sum(durations >= 1.0)),
                "short_visibility_blips_under_0_35_sec": int(
                    np.sum(durations < 0.35)
                ),
                "outcomes": {
                    outcome: sum(row["outcome"] == outcome for row in episode_rows)
                    for outcome in sorted({row["outcome"] for row in episode_rows})
                },
            }
        )

    command_timing = _timing_summary(command_ns)
    command_timing["estimated_missing_messages"] = None
    command_timing["missing_estimate_reason"] = (
        "event-driven topic; silence is not evidence of a missing command"
    )

    summary = {
        "session": {
            "directory": str(session_dir),
            "start_utc": metadata.get("start_utc"),
            "end_utc": metadata.get("end_utc"),
            "duration_sec": metadata.get("duration_sec"),
        },
        "camera": {
            "metadata": metadata.get("camera_first_message"),
            "timing": _timing_summary(camera_ns),
            "source_stamp_timing": _source_timing(camera_rows),
            "saved_frames": metadata.get("message_counts", {}).get("saved_frames", 0),
        },
        "state": {
            "primary_topic": primary_topic,
            "timing": _timing_summary(primary_ns),
            "subimg_topic": subimg_topic,
            "subimg_timing": _timing_summary(subimg_ns),
            "ball_missing_samples": int(np.sum(~detected)),
            "ball_missing_fraction": (
                float(np.mean(~detected)) if detected.size else None
            ),
            "longest_continuous_missing_sec": (
                _longest_false_run(primary_ns, detected) if detected.size else None
            ),
            "x_b_m": _range_summary(_floats(primary_rows, "x_b")),
            "y_b_m": _range_summary(_floats(primary_rows, "y_b")),
            "alpha_rad": _range_summary(_floats(primary_rows, "alpha")),
            "beta_rad": _range_summary(_floats(primary_rows, "beta")),
            "camera_to_estimate_subimg_receipt_delay": _matched_receipt_delay(
                camera_rows, subimg_rows
            ),
        },
        "motor_commands": {
            "timing": command_timing,
            "session_average_rate_hz": (
                len(command_rows) / session_duration if session_duration > 0 else None
            ),
            "vel_1": _range_summary(vel_1),
            "vel_2": _range_summary(vel_2),
            "configured_limits": {"vel_1": limit_1, "vel_2": limit_2},
            "at_limit_samples": {
                "vel_1": int(np.sum(np.abs(vel_1) >= limit_1 - 1e-9)),
                "vel_2": int(np.sum(np.abs(vel_2) >= limit_2 - 1e-9)),
            },
            "at_limit_fraction": {
                "vel_1": (
                    float(np.mean(np.abs(vel_1) >= limit_1 - 1e-9))
                    if vel_1.size
                    else None
                ),
                "vel_2": (
                    float(np.mean(np.abs(vel_2) >= limit_2 - 1e-9))
                    if vel_2.size
                    else None
                ),
            },
            "median_event_cadence_hz": (
                1000.0 / command_timing["interval_ms"]["median"]
                if command_timing.get("interval_ms")
                else None
            ),
        },
        "passive_command_to_angle": _passive_dynamics(primary_rows, command_rows),
        "episodes": episodes,
        "limitations": [
            "No active motor excitation was performed.",
            "Delay, gain, and coupling estimates are preliminary closed-loop fits.",
            "Episode rows are inferred from ball visibility, not Dreamer event messages.",
            (
                "Estimated missing transport messages use median cadence because "
                "the recorded messages have no sequence counter."
            ),
            (
                "The latest-camera receipt age can be overestimated when the "
                "best-effort camera subscription misses a frame."
            ),
        ],
    }
    summary = _number(summary)
    (session_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (session_dir / "analysis_summary.md").write_text(_markdown(summary))
    return summary


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _timing_line(name, timing):
    missing = timing.get("estimated_missing_messages")
    missing_text = "n/a (event-driven)" if missing is None else str(missing)
    return (
        f"- {name}: {timing['messages']} messages, "
        f"{_fmt(timing['rate_hz'])} Hz, interval std "
        f"{_fmt((timing.get('interval_ms') or {}).get('std'))} ms, "
        f"p99 {_fmt((timing.get('interval_ms') or {}).get('p99'))} ms, "
        f"estimated missing {missing_text}"
    )


def _markdown(summary):
    camera = summary["camera"]
    state = summary["state"]
    commands = summary["motor_commands"]
    lines = [
        "# Passive CyberRunner hardware recording analysis",
        "",
        f"- Start: {summary['session']['start_utc']}",
        f"- End: {summary['session']['end_utc']}",
        f"- Duration: {_fmt(summary['session']['duration_sec'], 1)} s",
        "",
        "## Rates and jitter",
        "",
        _timing_line("Camera", camera["timing"]),
        _timing_line("Camera source stamps", camera["source_stamp_timing"]),
        _timing_line("State", state["timing"]),
        _timing_line("State with TCP subimage", state["subimg_timing"]),
        _timing_line("Motor commands", commands["timing"]),
        "",
        "## Observations and ranges",
        "",
        f"- Camera metadata: `{camera['metadata']}`",
        (
            f"- Missing ball observations: {state['ball_missing_samples']} "
            f"({_fmt(100 * state['ball_missing_fraction'] if state['ball_missing_fraction'] is not None else None)}%), "
            f"longest run {_fmt(state['longest_continuous_missing_sec'])} s"
        ),
        (
            f"- alpha: [{_fmt(state['alpha_rad']['min'], 6)}, "
            f"{_fmt(state['alpha_rad']['max'], 6)}] rad"
        ),
        (
            f"- beta: [{_fmt(state['beta_rad']['min'], 6)}, "
            f"{_fmt(state['beta_rad']['max'], 6)}] rad"
        ),
        (
            "- Latest camera receipt to estimate_subimg receipt age: "
            f"{_fmt((state['camera_to_estimate_subimg_receipt_delay'].get('delay_ms') or {}).get('mean'))} ms mean, "
            f"{_fmt((state['camera_to_estimate_subimg_receipt_delay'].get('delay_ms') or {}).get('p99'))} ms p99"
        ),
        (
            f"- vel_1: [{_fmt(commands['vel_1']['min'], 3)}, "
            f"{_fmt(commands['vel_1']['max'], 3)}], "
            f"limit hits {commands['at_limit_samples']['vel_1']} "
            f"({_fmt(100 * commands['at_limit_fraction']['vel_1'])}%)"
        ),
        (
            f"- vel_2: [{_fmt(commands['vel_2']['min'], 3)}, "
            f"{_fmt(commands['vel_2']['max'], 3)}], "
            f"limit hits {commands['at_limit_samples']['vel_2']} "
            f"({_fmt(100 * commands['at_limit_fraction']['vel_2'])}%)"
        ),
        (
            f"- Command rate while active: {_fmt(commands['timing']['rate_hz'])} Hz; "
            f"median event cadence: {_fmt(commands['median_event_cadence_hz'])} Hz; "
            f"whole-session average: {_fmt(commands['session_average_rate_hz'])} Hz; "
            f"maximum silent gap: "
            f"{_fmt(commands['timing']['interval_ms']['max'] / 1000.0)} s"
        ),
        "",
        "## Passive command-to-angle fit",
        "",
    ]
    dynamics = summary["passive_command_to_angle"]
    if not dynamics.get("available"):
        lines.append(f"- Not available: {dynamics.get('reason')}")
    else:
        for name, fit in dynamics["responses"].items():
            if not fit.get("available", True):
                lines.append(f"- {name}: unavailable")
                continue
            lines.append(
                f"- {name}: delay {_fmt(1000 * fit['delay_sec'], 1)} ms, "
                f"gains [{_fmt(fit['gain_cmd_1'], 8)}, "
                f"{_fmt(fit['gain_cmd_2'], 8)}] rad/command, "
                f"R2={_fmt(fit['r2'])}, dominant cmd_{fit['dominant_command']}, "
                f"secondary/dominant ratio "
                f"{_fmt(fit['secondary_to_dominant_gain_ratio'])}"
            )
    lines.extend(
        [
            "",
            f"> {dynamics['warning']}",
            "",
            "## Episode statistics",
            "",
            (
                f"- {summary['episodes']['count']} passively inferred episodes; "
                "no official Dreamer episode-event topic was present."
            ),
            (
                f"- {summary['episodes'].get('count_at_least_1_sec', 0)} inferred "
                "intervals lasted at least 1 second; "
                f"{summary['episodes'].get('short_visibility_blips_under_0_35_sec', 0)} "
                "were shorter than the visibility grace threshold."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    args = parser.parse_args(argv)
    summary = analyze_session(args.session_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
