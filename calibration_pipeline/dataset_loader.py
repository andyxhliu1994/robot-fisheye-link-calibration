"""Read-only access and frame-alignment inspection for a recording dataset."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FRAME_RE = re.compile(r"^frame_(\d{6})$")


def normalize_frame_id(frame_id: str | int) -> str:
    if isinstance(frame_id, int):
        if frame_id < 0:
            raise ValueError("Frame number must be nonnegative")
        return f"frame_{frame_id:06d}"
    value = str(frame_id)
    if FRAME_RE.fullmatch(value):
        return value
    if value.isdigit():
        return f"frame_{int(value):06d}"
    raise ValueError(f"Invalid frame id: {frame_id!r}")


def frame_number(frame_id: str) -> int:
    match = FRAME_RE.fullmatch(frame_id)
    if not match:
        raise ValueError(f"Invalid frame id: {frame_id!r}")
    return int(match.group(1))


def load_json_document(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load JSON, allowing only whole-line ``//`` comments as compatibility mode."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        return json.loads(text), {"comment_lines_ignored": 0, "strict_json": True}
    except json.JSONDecodeError as strict_error:
        lines = text.splitlines()
        comment_count = sum(line.lstrip().startswith("//") for line in lines)
        if not comment_count:
            raise strict_error
        filtered = "\n".join(
            line for line in lines if not line.lstrip().startswith("//")
        )
        try:
            document = json.loads(filtered)
        except json.JSONDecodeError:
            raise strict_error
        return document, {
            "comment_lines_ignored": comment_count,
            "strict_json": False,
        }


class DatasetLoader:
    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

    def load_json(self, relative_path: str | Path) -> tuple[Any, dict[str, Any]]:
        return load_json_document(self.root / relative_path)

    def camera_names(self) -> list[str]:
        camera_root = self.root / "cameras"
        if not camera_root.is_dir():
            return []
        return sorted(path.name for path in camera_root.iterdir() if path.is_dir())

    def csv_frame_ids(self, relative_path: str | Path) -> tuple[list[str], list[str]]:
        path = self.root / relative_path
        ids: list[str] = []
        invalid: list[str] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "frame_id" not in reader.fieldnames:
                raise ValueError(f"CSV lacks frame_id column: {path}")
            for row in reader:
                value = row.get("frame_id", "")
                try:
                    ids.append(normalize_frame_id(value))
                except ValueError:
                    invalid.append(value)
        return ids, invalid

    def glob_frame_ids(
        self, relative_dir: str | Path, suffix: str
    ) -> tuple[list[str], list[str]]:
        directory = self.root / relative_dir
        ids: list[str] = []
        invalid: list[str] = []
        if not directory.is_dir():
            return ids, [f"missing directory: {relative_dir}"]
        for path in sorted(directory.glob(f"*{suffix}")):
            try:
                ids.append(normalize_frame_id(path.stem))
            except ValueError:
                invalid.append(path.name)
        return ids, invalid

    @staticmethod
    def _source_summary(
        ids: Iterable[str], invalid: list[str], expected: set[str]
    ) -> dict[str, Any]:
        values = list(ids)
        counts = Counter(values)
        actual = set(values)
        duplicate_ids = sorted(key for key, count in counts.items() if count > 1)
        return {
            "count": len(values),
            "unique_count": len(actual),
            "first_frame_id": min(actual, key=frame_number) if actual else None,
            "last_frame_id": max(actual, key=frame_number) if actual else None,
            "missing_frame_ids": sorted(expected - actual, key=frame_number),
            "unexpected_frame_ids": sorted(actual - expected, key=frame_number),
            "duplicate_frame_ids": duplicate_ids,
            "invalid_entries": invalid,
            "aligned": not (expected - actual or actual - expected or duplicate_ids or invalid),
        }

    def inspect_frame_alignment(
        self, selected_frame_ids: Iterable[str] | None = None
    ) -> dict[str, Any]:
        raw_sources: dict[str, tuple[list[str], list[str]]] = {}
        for csv_name in ("joint_states.csv", "link_poses.csv", "board_pose_base.csv"):
            raw_sources[csv_name] = self.csv_frame_ids(csv_name)
        raw_sources["link_poses/*.json"] = self.glob_frame_ids("link_poses", ".json")
        for camera in self.camera_names():
            raw_sources[f"cameras/{camera}/rgb/*.jpg"] = self.glob_frame_ids(
                Path("cameras") / camera / "rgb", ".jpg"
            )
            raw_sources[f"cameras/{camera}/transform/*.json"] = self.glob_frame_ids(
                Path("cameras") / camera / "transform", ".json"
            )

        if selected_frame_ids is None:
            expected = set(raw_sources["joint_states.csv"][0])
        else:
            expected = {normalize_frame_id(value) for value in selected_frame_ids}
            raw_sources = {
                name: ([value for value in ids if value in expected], invalid)
                for name, (ids, invalid) in raw_sources.items()
            }
        sources = {
            name: self._source_summary(ids, invalid, expected)
            for name, (ids, invalid) in raw_sources.items()
        }
        return {
            "reference_source": "joint_states.csv",
            "expected_frame_count": len(expected),
            "expected_first_frame_id": min(expected, key=frame_number) if expected else None,
            "expected_last_frame_id": max(expected, key=frame_number) if expected else None,
            "sources": sources,
            "passed": bool(expected) and all(item["aligned"] for item in sources.values()),
        }

    def inspect_camera_configuration(self) -> dict[str, Any]:
        report: dict[str, Any] = {"passed": False, "errors": [], "warnings": []}
        try:
            config, metadata = self.load_json("camera_model_config.json")
            report["camera_model_config"] = {
                "readable": True,
                **metadata,
                "default_camera_model": config.get("default_camera_model"),
                "camera_pose_frame": config.get(
                    "camera_pose_frame", config.get("pose_camera_frame")
                ),
                "ray_frame": config.get("ray_frame"),
                "ray_frame_adapter": config.get("ray_frame_adapter"),
            }
            if not metadata["strict_json"]:
                report["warnings"].append(
                    "camera_model_config.json required whole-line // comment compatibility mode"
                )
            relative_calibration = config["default_calibration_file"]
            calibration, calibration_metadata = self.load_json(relative_calibration)
            required = ("distortion_center", "stretch_matrix", "taylor_coefficient")
            missing = [key for key in required if key not in calibration]
            report["calibration_file"] = {
                "path": relative_calibration,
                "readable": True,
                **calibration_metadata,
                "required_fields_present": not missing,
                "missing_fields": missing,
            }
            if config.get("default_camera_model") != "ocamcalib":
                report["errors"].append("Default camera model is not ocamcalib")
            if missing:
                report["errors"].append(f"Calibration file lacks fields: {missing}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        report["passed"] = not report["errors"]
        return report

