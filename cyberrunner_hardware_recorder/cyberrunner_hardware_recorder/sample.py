#!/usr/bin/env python3

"""Export a small, sanitized excerpt from a hardware recording."""

import argparse
import csv
import json
import shutil
from pathlib import Path


CSV_FILES = (
    "camera_timing.csv",
    "state.csv",
    "motor_commands.csv",
    "episodes.csv",
)


def _copy_csv_excerpt(source, destination, max_rows):
    with source.open(newline="") as input_handle:
        reader = csv.reader(input_handle)
        with destination.open("w", newline="") as output_handle:
            writer = csv.writer(output_handle)
            for index, row in enumerate(reader):
                if index > max_rows:
                    break
                writer.writerow(row)


def _sanitize_json(source, destination):
    data = json.loads(source.read_text())
    if "session_directory" in data:
        data["session_directory"] = "<recording-root>/" + source.parent.name
    if "directory" in data.get("session", {}):
        data["session"]["directory"] = "<recording-root>/" + source.parent.name
    data.pop("host", None)
    data.pop("pid", None)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def export_sample(session_dir, output_dir, max_rows):
    session_dir = Path(session_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    for filename in CSV_FILES:
        source = session_dir / filename
        if source.exists():
            _copy_csv_excerpt(source, output_dir / filename, max_rows)

    for filename in ("session_metadata.json", "analysis_summary.json"):
        source = session_dir / filename
        if source.exists():
            _sanitize_json(source, output_dir / filename)

    for filename in ("topic_report.txt", "analysis_summary.md"):
        source = session_dir / filename
        if source.exists():
            shutil.copyfile(source, output_dir / filename)

    (output_dir / "SAMPLE_README.md").write_text(
        "# Passive recorder sample excerpt\n\n"
        f"Source session: `{session_dir.name}`\n\n"
        f"Each CSV contains its header and at most {max_rows} data rows. "
        "The full raw session and image payloads are intentionally not included. "
        "Session paths, host name, and PID are sanitized.\n"
    )
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--max-rows", type=int, default=200)
    args = parser.parse_args(argv)
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be nonnegative")
    print(export_sample(args.session_dir, args.output_dir, args.max_rows))


if __name__ == "__main__":
    main()
