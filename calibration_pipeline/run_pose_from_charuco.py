"""Run ray-based T_camera_board estimation on sampled ChArUco frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Sequence

import cv2
import numpy as np

from .camera_frame_adapter import camera_frame_adapter_from_config
from .camera_model import OCamCalibCameraModel
from .charuco_config import CharucoBoardConfig
from .charuco_detector import CharucoDetection, CharucoDetector
from .dataset_loader import DatasetLoader, load_json_document
from .pose_from_charuco import BoardPoseEstimate, estimate_pose_from_rays
from .run_charuco_detection import select_rgb_frames
from .se3_utils import (
    invert_T,
    mat_from_t_q,
    rotation_error_deg,
    translation_error_m,
)


def load_camera_ray_components(dataset_root: Path):
    config, _ = load_json_document(dataset_root / "camera_model_config.json")
    if config.get("default_camera_model") != "ocamcalib":
        raise ValueError("Milestone 3 currently requires an ocamcalib camera model")
    calibration, _ = load_json_document(
        dataset_root / config["default_calibration_file"]
    )
    model = OCamCalibCameraModel.from_mapping(calibration)
    adapter = camera_frame_adapter_from_config(config["ray_frame_adapter"])
    camera_pose_frame = str(
        config.get("camera_pose_frame", config.get("pose_camera_frame", "unspecified"))
    )
    return model, adapter, camera_pose_frame


class GroundTruthBoardPoseEvaluator:
    """Post-estimation evaluation with an explicit Unity board-frame adapter."""

    def __init__(self, dataset_root: Path, board_config: CharucoBoardConfig) -> None:
        self.dataset_root = dataset_root
        self.board_poses: dict[str, np.ndarray] = {}
        board_csv = dataset_root / "board_pose_base.csv"
        if board_csv.is_file():
            with board_csv.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    self.board_poses[row["frame_id"]] = mat_from_t_q(
                        [float(row[key]) for key in ("tx", "ty", "tz")],
                        [float(row[key]) for key in ("qx", "qy", "qz", "qw")],
                    )
        # Unity's exported board GameObject pose is centered with +Y opposite
        # OpenCV ChArUco +Y. This proper rigid transform maps config/OpenCV board
        # coordinates into that evaluation-only Unity board frame.
        self.T_unity_board_charuco_board = np.eye(4, dtype=float)
        self.T_unity_board_charuco_board[:3, :3] = np.diag([1.0, -1.0, -1.0])
        self.T_unity_board_charuco_board[:3, 3] = [
            -board_config.board_width_m / 2.0,
            board_config.board_height_m / 2.0,
            0.0,
        ]

    def evaluate(
        self, camera_name: str, frame_id: str, estimated: np.ndarray
    ) -> dict[str, object]:
        camera_path = (
            self.dataset_root
            / "cameras"
            / camera_name
            / "transform"
            / f"{frame_id}.json"
        )
        if frame_id not in self.board_poses or not camera_path.is_file():
            return {"gt_evaluation_available": False}
        camera, _ = load_json_document(camera_path)
        T_base_camera = mat_from_t_q(
            camera["t_base_cam"], camera["q_base_cam_xyzw"]
        )
        T_camera_unity_board = invert_T(T_base_camera) @ self.board_poses[frame_id]
        T_camera_charuco_board_gt = (
            T_camera_unity_board @ self.T_unity_board_charuco_board
        )
        return {
            "gt_evaluation_available": True,
            "gt_board_frame_adapter": "unity_center_y_up_from_charuco_corner_y_down",
            "gt_translation_error_m": translation_error_m(
                estimated[:3, 3], T_camera_charuco_board_gt[:3, 3]
            ),
            "gt_rotation_error_deg": rotation_error_deg(
                estimated[:3, :3], T_camera_charuco_board_gt[:3, :3]
            ),
        }


def _failure_estimate(detection: CharucoDetection) -> BoardPoseEstimate:
    if detection.reason in {"image_read_failed", "unsupported_image_shape"}:
        reason = detection.reason
    else:
        reason = "insufficient_corners"
    return BoardPoseEstimate.failure(reason, detection.charuco_corner_count)


def _draw_pose_overlay(
    image: np.ndarray,
    detection: CharucoDetection,
    estimate: BoardPoseEstimate,
) -> np.ndarray:
    overlay = CharucoDetector.draw_overlay(image, detection)
    if (
        estimate.per_point_ray_error_deg is not None
        and detection.charuco_corners is not None
    ):
        points = np.asarray(detection.charuco_corners).reshape(-1, 2)
        for point, error in zip(points, estimate.per_point_ray_error_deg):
            if error < 0.2:
                color = (0, 255, 0)
            elif error < 1.0:
                color = (0, 220, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 6, color, 2, cv2.LINE_AA)
    pose_status = "POSE VALID" if estimate.valid else f"POSE INVALID: {estimate.reason}"
    error_text = (
        f"mean ray error={estimate.mean_ray_error_deg:.4f} deg"
        if estimate.mean_ray_error_deg is not None
        else "mean ray error=n/a"
    )
    color = (20, 230, 20) if estimate.valid else (20, 20, 240)
    for index, text in enumerate((pose_status, error_text)):
        origin = (20, 140 + index * 34)
        cv2.putText(
            overlay,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def process_camera_poses(
    dataset_root: Path,
    output_root: Path,
    camera_name: str,
    board_config: CharucoBoardConfig,
    detector: CharucoDetector,
    camera_model: OCamCalibCameraModel,
    frame_adapter,
    camera_pose_frame: str,
    max_frames: int | None,
    frame_stride: int,
    save_overlays: bool,
    gt_evaluator: GroundTruthBoardPoseEvaluator | None,
) -> dict[str, object]:
    rgb_dir = dataset_root / "cameras" / camera_name / "rgb"
    frame_paths = select_rgb_frames(rgb_dir, max_frames, frame_stride)
    pose_dir = output_root / "board_poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pose_dir / f"{camera_name}.jsonl"
    overlay_dir = output_root / "pose_overlays" / camera_name
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            detection = detector.detect(image, frame_path.stem, camera_name)
            if (
                image is None
                or detection.charuco_corner_count < 8
                or detection.charuco_ids is None
                or detection.charuco_corners is None
            ):
                estimate = _failure_estimate(detection)
            else:
                ids = np.asarray(detection.charuco_ids, dtype=int).reshape(-1)
                pixels = np.asarray(detection.charuco_corners, dtype=float).reshape(-1, 2)
                board_points = board_config.corner_points_for_ids(ids)
                height, width = image.shape[:2]
                rays = np.asarray(
                    [
                        frame_adapter(
                            camera_model.pixel_to_ray(u, v, width, height)
                        )
                        for u, v in pixels
                    ]
                )
                estimate = estimate_pose_from_rays(board_points, rays)
            record = estimate.to_record(frame_path.stem, camera_name, camera_pose_frame)
            if (
                gt_evaluator is not None
                and estimate.valid
                and estimate.T_camera_board is not None
            ):
                record.update(
                    gt_evaluator.evaluate(
                        camera_name, frame_path.stem, estimate.T_camera_board
                    )
                )
            else:
                record["gt_evaluation_available"] = False
            records.append(record)
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if save_overlays and image is not None:
                overlay = _draw_pose_overlay(image, detection, estimate)
                destination = overlay_dir / frame_path.name
                if not cv2.imwrite(str(destination), overlay):
                    raise OSError(f"Could not write pose overlay: {destination}")

    valid_records = [record for record in records if record["valid"]]
    ray_errors = [float(record["mean_ray_error_deg"]) for record in valid_records]
    evaluated = [
        record for record in valid_records if record.get("gt_evaluation_available")
    ]
    failures = Counter(
        str(record["reason"]) for record in records if not record["valid"]
    )
    return {
        "camera_name": camera_name,
        "jsonl_path": str(jsonl_path),
        "total_frames": len(records),
        "valid_poses": len(valid_records),
        "valid_ratio": len(valid_records) / len(records) if records else 0.0,
        "mean_charuco_corner_count": (
            fmean(float(record["charuco_corner_count"]) for record in records)
            if records
            else 0.0
        ),
        "mean_ray_error_deg": fmean(ray_errors) if ray_errors else None,
        "median_ray_error_deg": median(ray_errors) if ray_errors else None,
        "failure_counts": dict(failures.most_common()),
        "gt_evaluated_pose_count": len(evaluated),
        "mean_gt_translation_error_m": (
            fmean(float(record["gt_translation_error_m"]) for record in evaluated)
            if evaluated
            else None
        ),
        "mean_gt_rotation_error_deg": (
            fmean(float(record["gt_rotation_error_deg"]) for record in evaluated)
            if evaluated
            else None
        ),
    }


def print_summary(summaries: list[dict[str, object]]) -> None:
    print("Ray-based ChArUco board pose summary")
    for summary in summaries:
        failures = summary["failure_counts"] or {"none": 0}
        failure_text = ", ".join(f"{key}={value}" for key, value in failures.items())
        mean_ray = summary["mean_ray_error_deg"]
        median_ray = summary["median_ray_error_deg"]
        gt_translation = summary["mean_gt_translation_error_m"]
        gt_rotation = summary["mean_gt_rotation_error_deg"]
        print(
            f"- {summary['camera_name']}: total={summary['total_frames']}, "
            f"valid={summary['valid_poses']} ({summary['valid_ratio']:.1%}), "
            f"mean_charuco={summary['mean_charuco_corner_count']:.2f}, "
            f"mean_ray_deg={mean_ray:.4f}, median_ray_deg={median_ray:.4f}, "
            f"failures=[{failure_text}]"
            if mean_ray is not None and median_ray is not None
            else (
                f"- {summary['camera_name']}: total={summary['total_frames']}, "
                f"valid=0, mean_charuco={summary['mean_charuco_corner_count']:.2f}, "
                f"failures=[{failure_text}]"
            )
        )
        if gt_translation is not None and gt_rotation is not None:
            print(
                f"  GT evaluation: n={summary['gt_evaluated_pose_count']}, "
                f"mean_translation_m={gt_translation:.4f}, "
                f"mean_rotation_deg={gt_rotation:.4f}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--camera")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate estimates after optimization when Unity GT files exist",
    )
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

    board_config = CharucoBoardConfig.from_json(
        args.dataset / "charuco_board_config.json"
    )
    detector = CharucoDetector(board_config)
    model, adapter, camera_pose_frame = load_camera_ray_components(args.dataset)
    evaluator = None
    if args.evaluate_gt and (args.dataset / "board_pose_base.csv").is_file():
        evaluator = GroundTruthBoardPoseEvaluator(args.dataset, board_config)
    summaries = [
        process_camera_poses(
            args.dataset,
            args.output,
            camera,
            board_config,
            detector,
            model,
            adapter,
            camera_pose_frame,
            args.max_frames,
            args.frame_stride,
            args.save_overlays,
            evaluator,
        )
        for camera in camera_names
    ]
    print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
