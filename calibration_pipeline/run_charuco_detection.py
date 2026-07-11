"""Run Milestone 2 ChArUco detection and optional debug-overlay generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Sequence

import cv2

from .charuco_config import CharucoBoardConfig
from .charuco_detector import CharucoDetector
from .dataset_loader import DatasetLoader, frame_number


def select_rgb_frames(rgb_dir: Path, max_frames: int | None, stride: int) -> list[Path]:
    if stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    frames = []
    for path in rgb_dir.glob("frame_*.jpg"):
        try:
            frame_number(path.stem)
        except ValueError:
            continue
        frames.append(path)
    frames.sort(key=lambda path: frame_number(path.stem))
    selected = frames[::stride]
    return selected if max_frames is None else selected[:max_frames]


def process_camera(
    dataset_root: Path,
    output_root: Path,
    camera_name: str,
    detector: CharucoDetector,
    max_frames: int | None,
    frame_stride: int,
    save_overlays: bool,
) -> dict:
    rgb_dir = dataset_root / "cameras" / camera_name / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    frame_paths = select_rgb_frames(rgb_dir, max_frames, frame_stride)
    detection_dir = output_root / "detections"
    detection_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = detection_dir / f"{camera_name}.jsonl"
    overlay_dir = output_root / "debug_overlays" / camera_name
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            detection = detector.detect(image, frame_path.stem, camera_name)
            record = detection.to_record()
            records.append(record)
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if save_overlays and image is not None:
                overlay = detector.draw_overlay(image, detection)
                destination = overlay_dir / frame_path.name
                if not cv2.imwrite(str(destination), overlay):
                    raise OSError(f"Could not write overlay: {destination}")

    valid_count = sum(record["valid"] for record in records)
    failures = Counter(record["reason"] for record in records if not record["valid"])
    return {
        "camera_name": camera_name,
        "jsonl_path": str(jsonl_path),
        "total_frames": len(records),
        "valid_frames": valid_count,
        "valid_ratio": valid_count / len(records) if records else 0.0,
        "mean_marker_count": fmean(record["marker_count"] for record in records) if records else 0.0,
        "mean_charuco_corner_count": (
            fmean(record["charuco_corner_count"] for record in records) if records else 0.0
        ),
        "failure_reasons": dict(failures.most_common()),
    }


def print_summary(summaries: list[dict]) -> None:
    print("ChArUco detection summary")
    for summary in summaries:
        failures = summary["failure_reasons"] or {"none": 0}
        failure_text = ", ".join(f"{key}={value}" for key, value in failures.items())
        print(
            f"- {summary['camera_name']}: total={summary['total_frames']}, "
            f"valid={summary['valid_frames']} ({summary['valid_ratio']:.1%}), "
            f"mean_markers={summary['mean_marker_count']:.2f}, "
            f"mean_charuco={summary['mean_charuco_corner_count']:.2f}, "
            f"failures=[{failure_text}]"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--camera")
    parser.add_argument("--save-overlays", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    loader = DatasetLoader(args.dataset)
    available_cameras = loader.camera_names()
    if args.camera:
        if args.camera not in available_cameras:
            raise SystemExit(
                f"Unknown camera {args.camera!r}; choices: {', '.join(available_cameras)}"
            )
        camera_names = [args.camera]
    else:
        camera_names = available_cameras
    if not camera_names:
        raise SystemExit("No camera directories found")

    config = CharucoBoardConfig.from_json(args.dataset / "charuco_board_config.json")
    detector = CharucoDetector(config)
    summaries = [
        process_camera(
            args.dataset,
            args.output,
            camera,
            detector,
            args.max_frames,
            args.frame_stride,
            args.save_overlays,
        )
        for camera in camera_names
    ]
    print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

