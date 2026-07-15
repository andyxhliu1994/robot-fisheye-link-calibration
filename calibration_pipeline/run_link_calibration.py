"""Run Milestone 4 link association and static camera-mount estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset_loader import DatasetLoader, load_json_document
from .kinematics_provider import UnityLinkPoseProvider
from .link_calibrator import (
    BoardPoseObservation,
    CameraLinkCalibrationResult,
    rank_link_hypotheses,
)
from .se3_utils import mat_from_t_q, rotation_error_deg, translation_error_m


def load_board_pose_observations(path: Path) -> list[BoardPoseObservation]:
    if not path.is_file():
        raise FileNotFoundError(f"Board-pose JSONL does not exist: {path}")
    observations = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("valid"):
            continue
        frame_id = str(record["frame_id"])
        if frame_id in seen:
            raise ValueError(f"Duplicate frame_id {frame_id} in {path}:{line_number}")
        seen.add(frame_id)
        matrix = np.asarray(record["T_camera_board_rowmajor"], dtype=float)
        if matrix.size != 16:
            raise ValueError(f"Invalid T_camera_board in {path}:{line_number}")
        observations.append(
            BoardPoseObservation(
                frame_id=frame_id,
                T_camera_board=matrix.reshape(4, 4),
                charuco_corner_count=int(record.get("charuco_corner_count", 0)),
                mean_ray_error_deg=(
                    float(record["mean_ray_error_deg"])
                    if record.get("mean_ray_error_deg") is not None
                    else None
                ),
            )
        )
    return observations


def load_aligned_candidate_link_poses(
    provider: UnityLinkPoseProvider,
    observations: Sequence[BoardPoseObservation],
    candidate_links: Sequence[dict[str, Any]],
) -> dict[str, list[np.ndarray]]:
    aligned = {str(candidate["link_path_rel"]): [] for candidate in candidate_links}
    for observation in observations:
        poses = provider.get_candidate_link_poses(observation.frame_id)
        for path in aligned:
            if path not in poses:
                raise KeyError(f"Candidate link {path} missing at {observation.frame_id}")
            aligned[path].append(poses[path])
    return aligned


class GroundTruthMountEvaluator:
    """Optional post-ranking evaluator; never called by estimation functions."""

    def __init__(self, dataset_root: Path) -> None:
        setup, _ = load_json_document(dataset_root / "setup_used.json")
        self.mounts = {
            str(camera["camera_name"]): camera for camera in setup.get("cameras", [])
        }

    @staticmethod
    def _vector(document: dict[str, Any], keys: Sequence[str]) -> list[float]:
        return [float(document[key]) for key in keys]

    def evaluate(self, camera_record: dict[str, Any]) -> dict[str, Any]:
        camera_name = str(camera_record["camera_name"])
        mount = self.mounts.get(camera_name)
        matrix_values = camera_record.get("T_link_camera_rowmajor")
        if mount is None or matrix_values is None:
            return {"gt_evaluation_available": False}
        translation_document = mount["t_link_from_cam"]
        quaternion_document = mount["q_link_from_cam_xyzw"]
        T_link_camera_gt = mat_from_t_q(
            self._vector(translation_document, ("x", "y", "z")),
            self._vector(quaternion_document, ("x", "y", "z", "w")),
        )
        estimated = np.asarray(matrix_values, dtype=float).reshape(4, 4)
        return {
            "gt_evaluation_available": True,
            "gt_link_path_rel": str(mount["link_path_rel"]),
            "gt_best_link_correct": (
                camera_record.get("best_link_path_rel") == mount["link_path_rel"]
            ),
            "gt_T_link_camera_translation_error_m": translation_error_m(
                estimated[:3, 3], T_link_camera_gt[:3, 3]
            ),
            "gt_T_link_camera_rotation_error_deg": rotation_error_deg(
                estimated[:3, :3], T_link_camera_gt[:3, :3]
            ),
        }


def calibrate_camera(
    board_pose_dir: Path,
    camera_name: str,
    candidate_links: Sequence[dict[str, Any]],
    provider: UnityLinkPoseProvider,
    min_valid_poses: int,
) -> CameraLinkCalibrationResult:
    observations = load_board_pose_observations(
        board_pose_dir / f"{camera_name}.jsonl"
    )
    link_poses = load_aligned_candidate_link_poses(
        provider, observations, candidate_links
    )
    return rank_link_hypotheses(
        camera_name,
        observations,
        candidate_links,
        link_poses,
        min_valid_poses=min_valid_poses,
    )


def print_summary(records: Sequence[dict[str, Any]]) -> None:
    print("Link association and camera mount summary")
    for record in records:
        translation = record["board_consistency_translation_m"]
        rotation = record["board_consistency_rotation_deg"]
        print(
            f"- {record['camera_name']}: best={record['best_link']}, "
            f"second={record['second_best_link']}, margin={record['score_margin']:.6f}, "
            f"frames={record['num_valid_frames']}, "
            f"translation_mean_m={translation['mean']:.6f}, "
            f"rotation_mean_deg={rotation['mean']:.4f}"
            if record["success"]
            else (
                f"- {record['camera_name']}: FAILED ({record['failure_reason']}), "
                f"frames={record['num_valid_frames']}"
            )
        )
        if record.get("gt_evaluation_available"):
            print(
                f"  GT: link_correct={record['gt_best_link_correct']}, "
                f"translation_error_m={record['gt_T_link_camera_translation_error_m']:.6f}, "
                f"rotation_error_deg={record['gt_T_link_camera_rotation_error_deg']:.4f}"
            )
        if record.get("mount_fully_observable") is False:
            print(
                f"  Observability: rank={record['observability_rank']}/"
                f"{record['observability_parameter_count']}; mount uses a "
                "minimum-norm gauge in unobservable directions"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--board-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera")
    parser.add_argument("--min-valid-poses", type=int, default=10)
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the completed ranking against Unity mount GT",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_valid_poses < 3:
        raise SystemExit("--min-valid-poses must be at least 3")
    loader = DatasetLoader(args.dataset)
    available_cameras = loader.camera_names()
    if args.camera:
        if args.camera not in available_cameras:
            raise SystemExit(
                f"Unknown camera {args.camera!r}; choices: {', '.join(available_cameras)}"
            )
        camera_names = [args.camera]
    else:
        camera_names = [
            camera
            for camera in available_cameras
            if (args.board_poses / f"{camera}.jsonl").is_file()
        ]
    if not camera_names:
        raise SystemExit("No camera board-pose JSONLs were found")

    candidate_document, _ = load_json_document(args.dataset / "candidate_links.json")
    candidate_links = list(candidate_document["links"])
    provider = UnityLinkPoseProvider(args.dataset)
    output_dir = args.output / "link_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_records = []
    for camera_name in camera_names:
        result = calibrate_camera(
            args.board_poses,
            camera_name,
            candidate_links,
            provider,
            args.min_valid_poses,
        )
        record = result.to_record()
        camera_records.append(record)

    # Load optional ground truth only after every estimate and ranking is final.
    if args.evaluate_gt:
        evaluator = GroundTruthMountEvaluator(args.dataset)
        for record in camera_records:
            record.update(evaluator.evaluate(record))
    for record in camera_records:
        camera_name = str(record["camera_name"])
        (output_dir / f"{camera_name}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "schema_version": 1,
        "milestone": "link_association_and_camera_mount_estimation",
        "kinematics_provider": "UnityLinkPoseProvider",
        "base_frame": str(candidate_document.get("base_frame_name", "base")),
        "candidate_links_source": "candidate_links.json",
        "board_pose_source": str(args.board_poses),
        "ground_truth_used_for_estimation": False,
        "camera_count": len(camera_records),
        "cameras": camera_records,
    }
    summary_path = output_dir / "link_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(camera_records)
    print(f"- Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
