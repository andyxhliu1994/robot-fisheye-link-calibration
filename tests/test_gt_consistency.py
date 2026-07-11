import json

from calibration_pipeline.evaluation import evaluate_gt_consistency
from calibration_pipeline.run_integrity_check import build_report


def test_gt_composition_consistency_across_sequence_and_all_cameras():
    frames = ["frame_000000", "frame_000471", "frame_000941"]
    report = evaluate_gt_consistency("dataset", frames)
    assert report["evaluated_composition_count"] == 15
    assert report["passed"], json.dumps(report, indent=2)


def test_integrity_report_builder_on_debug_subset():
    report = build_report(
        dataset_root=__import__("pathlib").Path("dataset"),
        frame_start=0,
        frame_stop=10,
        frame_step=2,
        gt_sample_count=3,
    )
    assert report["selection"]["selected_frame_count"] == 5
    assert report["gt_transform_consistency"]["evaluated_composition_count"] == 15
    assert report["passed"], json.dumps(report, indent=2)

