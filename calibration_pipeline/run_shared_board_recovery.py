"""Recover motion-limited camera mounts using a shared fixed board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_loader import DatasetLoader, load_json_document
from .kinematics_provider import UnityLinkPoseProvider
from .run_link_calibration import (
    GroundTruthMountEvaluator,
    load_board_pose_observations,
)
from .shared_board_recovery import (
    classify_camera_records,
    perform_shared_board_recovery,
)


def load_best_link_inputs(
    board_pose_dir: Path,
    provider: UnityLinkPoseProvider,
    camera_records: Sequence[Mapping[str, Any]],
    candidate_paths: set[str],
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    observations_by_camera = {}
    link_poses_by_camera: dict[str, list[np.ndarray]] = {}
    for record in camera_records:
        camera_name = str(record["camera_name"])
        link_path = str(record.get("best_link_path_rel") or "")
        if link_path not in candidate_paths:
            raise ValueError(
                f"Best link path for {camera_name} is not in candidate_links.json"
            )
        observations = load_board_pose_observations(
            board_pose_dir / f"{camera_name}.jsonl"
        )
        link_poses = []
        for observation in observations:
            candidate_poses = provider.get_candidate_link_poses(observation.frame_id)
            if link_path not in candidate_poses:
                raise KeyError(f"Link {link_path} missing at {observation.frame_id}")
            link_poses.append(candidate_poses[link_path])
        observations_by_camera[camera_name] = observations
        link_poses_by_camera[camera_name] = link_poses
    return observations_by_camera, link_poses_by_camera


def add_ground_truth_evaluation(
    summary: dict[str, Any], dataset_root: Path
) -> None:
    """Evaluate finalized recovered mounts; never called by estimation code."""
    evaluator = GroundTruthMountEvaluator(dataset_root)
    for result in summary["camera_results"]:
        matrix = result.get("T_link_camera_recovered_rowmajor")
        if not result.get("recovery_used") or matrix is None:
            continue
        evaluation = evaluator.evaluate(
            {
                "camera_name": result["camera_name"],
                "best_link_path_rel": result["best_link_path_rel"],
                "T_link_camera_rowmajor": matrix,
            }
        )
        result["gt_evaluation_available"] = evaluation.get(
            "gt_evaluation_available", False
        )
        result["gt_best_link_correct"] = evaluation.get("gt_best_link_correct")
        result["gt_recovered_translation_error_m"] = evaluation.get(
            "gt_T_link_camera_translation_error_m"
        )
        result["gt_recovered_rotation_error_deg"] = evaluation.get(
            "gt_T_link_camera_rotation_error_deg"
        )


def print_summary(summary: Mapping[str, Any]) -> None:
    anchors = ", ".join(summary["anchor_cameras"]) or "none"
    motion_limited = ", ".join(summary["motion_limited_cameras"]) or "none"
    print("Shared-board recovery summary")
    print(f"- Anchor cameras: {anchors}")
    print(f"- Motion-limited cameras: {motion_limited}")
    print(f"- Status: {summary['status']}")
    agreement = summary.get("anchor_agreement")
    if agreement is not None:
        print(
            f"- Anchor agreement: samples={agreement['sample_count']}, "
            f"translation_mean_m={agreement['translation_m']['mean']:.6f}, "
            f"rotation_mean_deg={agreement['rotation_deg']['mean']:.4f}"
        )
    for result in summary["camera_results"]:
        if result["recovery_used"]:
            translation = result["recovery_consistency_translation_m"]
            rotation = result["recovery_consistency_rotation_deg"]
            print(
                f"- Recovered {result['camera_name']}: link={result['best_link']}, "
                f"confidence={result['confidence']}, frames={result['num_valid_frames']}, "
                f"translation_mean_m={translation['mean']:.6f}, "
                f"rotation_mean_deg={rotation['mean']:.4f}"
            )
            if result.get("gt_evaluation_available"):
                print(
                    f"  GT: translation_error_m="
                    f"{result['gt_recovered_translation_error_m']:.6f}, "
                    f"rotation_error_deg="
                    f"{result['gt_recovered_rotation_error_deg']:.4f}"
                )
        else:
            print(f"- Not recovered {result['camera_name']}: {result['warning']}")
        if result.get("warning"):
            print(f"  Warning: {result['warning']}")
    for warning in summary["warnings"]:
        print(f"- Warning: {warning}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--link-calibration", type=Path, required=True)
    parser.add_argument("--board-poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera")
    parser.add_argument("--min-anchor-cameras", type=int, default=2)
    parser.add_argument("--allow-single-anchor", action="store_true")
    parser.add_argument(
        "--evaluate-gt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate finalized recovered mounts against Unity GT",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_anchor_cameras < 2:
        raise SystemExit("--min-anchor-cameras must be at least 2")
    loader = DatasetLoader(args.dataset)
    available_cameras = loader.camera_names()
    link_summary = json.loads(args.link_calibration.read_text(encoding="utf-8"))
    camera_records = list(link_summary.get("cameras", []))
    summary_cameras = {str(record.get("camera_name")) for record in camera_records}
    if args.camera:
        if args.camera not in available_cameras or args.camera not in summary_cameras:
            raise SystemExit(f"Unknown camera {args.camera!r}")

    anchors, motion_limited, _ = classify_camera_records(camera_records)
    if args.camera:
        motion_limited = [
            record
            for record in motion_limited
            if str(record["camera_name"]) == args.camera
        ]
    can_recover = len(anchors) >= args.min_anchor_cameras or (
        len(anchors) == 1 and args.allow_single_anchor
    )
    if motion_limited and can_recover:
        required_records = list(anchors) + list(motion_limited)
        candidate_document, _ = load_json_document(
            args.dataset / "candidate_links.json"
        )
        candidate_paths = {
            str(candidate["link_path_rel"])
            for candidate in candidate_document.get("links", [])
        }
        provider = UnityLinkPoseProvider(args.dataset)
        observations, link_poses = load_best_link_inputs(
            args.board_poses, provider, required_records, candidate_paths
        )
    else:
        observations, link_poses = {}, {}
    summary = perform_shared_board_recovery(
        link_summary,
        observations,
        link_poses,
        input_summary_path=str(args.link_calibration),
        min_anchor_cameras=args.min_anchor_cameras,
        allow_single_anchor=args.allow_single_anchor,
        target_camera=args.camera,
    )
    # Ground truth is loaded only after classification and recovery are final.
    if args.evaluate_gt:
        add_ground_truth_evaluation(summary, args.dataset)

    output_dir = args.output / "shared_board_recovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "shared_board_recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"- Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
