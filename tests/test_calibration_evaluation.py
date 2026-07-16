import csv
import json
from pathlib import Path

import numpy as np

from calibration_pipeline.calibration_evaluation import plots as evaluation_plots
from calibration_pipeline.calibration_evaluation.metrics import (
    LINK_RANKING_COLUMNS,
    PAIRWISE_COLUMNS,
    PER_CAMERA_COLUMNS,
    collect_calibration_metrics,
)
from calibration_pipeline.calibration_evaluation.report import build_report
from calibration_pipeline.run_calibration_evaluation import main


def write_json(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def transform(x=0.0, y=0.0, z=0.0):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix.reshape(-1).tolist()


def write_fixture(root: Path, cameras=("Fisheye180_Cam1", "Fisheye180_Cam2")):
    dataset = root / "dataset"
    outputs = root / "outputs"
    write_json(
        dataset / "session_summary.json",
        {
            "session_name": "experiment_001",
            "setup_id": "setup_a",
            "frame_count_so_far": 2,
            "camera_names": list(cameras),
        },
    )
    write_json(
        dataset / "camera_model_config.json",
        {"default_camera_model": "ocamcalib"},
    )
    write_json(
        dataset / "charuco_board_config.json",
        {
            "dictionary": "DICT_4X4_1000",
            "squares_x": 10,
            "squares_y": 7,
            "square_length_m": 0.08,
            "marker_length_m": 0.056,
        },
    )
    final_cameras = []
    link_cameras = []
    for index, camera in enumerate(cameras):
        final_cameras.append(
            {
                "camera_name": camera,
                "attached_link": f"robot/link_{index}",
                "attached_link_name": f"link_{index}",
                "calibration_source": "independent_link_calibration",
                "confidence": "high",
                "warnings": [],
                "observability": {
                    "rank": 12,
                    "max_rank": 12,
                    "motion_limited": False,
                },
            }
        )
        link_cameras.append(
            {
                "camera_name": camera,
                "best_link": f"link_{index}",
                "best_link_path_rel": f"robot/link_{index}",
                "best_score": 0.01 + index * 0.001,
                "second_best_score": 0.2,
                "score_margin": 0.19 - index * 0.001,
                "observability_rank": 12,
                "observability_parameter_count": 12,
                "mount_fully_observable": True,
                "hypotheses": [
                    {
                        "rank": 1,
                        "link_name": f"link_{index}",
                        "link_path_rel": f"robot/link_{index}",
                        "success": True,
                        "score": 0.01 + index * 0.001,
                        "translation_mean_m": 0.003,
                        "translation_median_m": 0.002,
                        "translation_max_m": 0.006,
                        "rotation_mean_deg": 0.3,
                        "rotation_median_deg": 0.2,
                        "rotation_max_deg": 0.7,
                        "num_valid_frames": 2,
                    }
                ],
            }
        )
        detection_dir = outputs / "detections"
        detection_dir.mkdir(parents=True, exist_ok=True)
        (detection_dir / f"{camera}.jsonl").write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "camera_name": camera,
                        "frame_id": "frame_000000",
                        "valid": True,
                        "marker_count": 10,
                        "charuco_corner_count": 20,
                        "reason": "ok",
                    },
                    {
                        "camera_name": camera,
                        "frame_id": "frame_000001",
                        "valid": False,
                        "marker_count": 0,
                        "charuco_corner_count": 0,
                        "reason": "no_markers",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        board_dir = outputs / "board_poses"
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / f"{camera}.jsonl").write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {
                        "camera_name": camera,
                        "frame_id": "frame_000000",
                        "valid": True,
                        "charuco_corner_count": 20,
                        "mean_ray_error_deg": 0.02,
                        "reason": "ok",
                    },
                    {
                        "camera_name": camera,
                        "frame_id": "frame_000001",
                        "valid": False,
                        "charuco_corner_count": 0,
                        "mean_ray_error_deg": None,
                        "reason": "insufficient_corners",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
    write_json(
        outputs / "final_calibration" / "final_calibration.json",
        {"camera_count": len(cameras), "cameras": final_cameras},
    )
    write_json(
        outputs / "link_calibration" / "link_calibration_summary.json",
        {"camera_count": len(cameras), "cameras": link_cameras},
    )
    write_json(
        outputs
        / "shared_board_recovery"
        / "shared_board_recovery_summary.json",
        {
            "anchor_cameras": list(cameras),
            "motion_limited_cameras": [],
            "camera_results": [],
            "warnings": [],
        },
    )
    write_json(
        outputs
        / "depth_model_compat"
        / "depth_model_compatibility_report.json",
        {
            "parser_compatibility_smoke_test_passed": True,
            "relative_pose_computation_smoke_test_passed": True,
        },
    )
    return dataset, outputs


def write_static_validation(outputs: Path, cameras):
    write_json(
        outputs
        / "final_calibration"
        / "final_static_calibration_validation.json",
        {
            "gt_validation_available": True,
            "cameras": [
                {
                    "camera_name": name,
                    "gt_attached_link_correct": True,
                    "gt_T_link_camera_translation_error_m": 0.01 + index * 0.001,
                    "gt_T_link_camera_rotation_error_deg": 0.5 + index * 0.1,
                }
                for index, name in enumerate(cameras)
            ],
        },
    )


def write_pose_validation(outputs: Path, cameras):
    write_json(
        outputs / "final_calibration" / "final_camera_pose_validation.json",
        {
            "gt_validation_available": True,
            "frame_count": 2,
            "cameras": [
                {
                    "camera_name": name,
                    "translation_error_m": {"mean": 0.01, "median": 0.009, "max": 0.02},
                    "rotation_error_deg": {"mean": 0.7, "median": 0.6, "max": 1.0},
                }
                for name in cameras
            ],
        },
    )


def write_pair_inputs(dataset: Path, outputs: Path, cameras):
    for camera_index, camera in enumerate(cameras):
        for frame_index in range(2):
            frame_id = f"frame_{frame_index:06d}"
            truth = transform(0.2 * camera_index, 0.01 * frame_index, 0.0)
            predicted = transform(
                0.2 * camera_index + 0.002 * (camera_index + 1),
                0.01 * frame_index,
                0.0,
            )
            write_json(
                dataset / "cameras" / camera / "transform" / f"{frame_id}.json",
                {"T_base_cam_rowmajor": truth},
            )
            write_json(
                outputs
                / "depth_model_compat"
                / "transforms"
                / camera
                / f"{frame_id}.json",
                {"T_base_cam_rowmajor": predicted},
            )
    write_json(
        outputs / "depth_model_compat" / "depth_model_pose_validation.json",
        {
            "gt_validation_available": True,
            "absolute_pose": {"frames_evaluated_total": 4, "cameras": []},
            "relative_pose": {},
        },
    )


def test_minimal_final_calibration_creates_per_camera_metrics(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    result = collect_calibration_metrics(dataset, outputs, evaluate_gt=False)
    assert result["camera_count"] == 2
    assert result["per_camera"][0]["attached_link"].startswith("robot/link_")
    assert result["per_camera"][0]["detection_valid_ratio"] == 0.5
    assert result["per_camera"][0]["board_pose_mean_ray_error_deg"] == 0.02


def test_gt_free_mode_succeeds_without_validation_files(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    result = collect_calibration_metrics(dataset, outputs, evaluate_gt=False)
    assert result["global_gt_based_metrics"] is None
    assert result["gt_metrics_available"] is False
    assert all(
        row["gt_T_link_camera_translation_error_m"] is None
        for row in result["per_camera"]
    )


def test_gt_mode_reads_static_validation(tmp_path):
    cameras = ("Fisheye180_Cam1", "Fisheye180_Cam2")
    dataset, outputs = write_fixture(tmp_path, cameras)
    write_static_validation(outputs, cameras)
    result = collect_calibration_metrics(dataset, outputs, evaluate_gt=True)
    assert result["per_camera"][0]["gt_attached_link_correct"] is True
    assert result["global_gt_based_metrics"]["link_association_top1_accuracy"] == 1.0


def test_gt_mode_reads_camera_pose_validation(tmp_path):
    cameras = ("Fisheye180_Cam1", "Fisheye180_Cam2")
    dataset, outputs = write_fixture(tmp_path, cameras)
    write_pose_validation(outputs, cameras)
    result = collect_calibration_metrics(dataset, outputs, evaluate_gt=True)
    assert result["per_camera"][0]["gt_T_base_cam_translation_error_mean_m"] == 0.01
    assert result["per_camera"][0]["gt_T_base_cam_rotation_error_max_deg"] == 1.0


def test_pairwise_metrics_csv_is_created_from_relative_inputs(tmp_path):
    cameras = ("Fisheye180_Cam1", "Fisheye180_Cam2")
    dataset, outputs = write_fixture(tmp_path, cameras)
    write_pair_inputs(dataset, outputs, cameras)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--outputs", str(outputs), "--output", str(output), "--evaluate-gt", "--no-save-plots"]) == 0
    with (output / "pairwise_relative_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(int(row["pair_count"]) == 2 for row in rows)
    assert {row["source_camera"] for row in rows} == set(cameras)


def test_missing_optional_inputs_warn_instead_of_crashing(tmp_path):
    dataset = tmp_path / "dataset"
    outputs = tmp_path / "outputs"
    write_json(
        outputs / "final_calibration" / "final_calibration.json",
        {
            "cameras": [
                {
                    "camera_name": "camera_1",
                    "attached_link": "robot/link",
                    "confidence": "low",
                    "warnings": ["limited"],
                    "observability": {},
                }
            ]
        },
    )
    result = collect_calibration_metrics(dataset, outputs, evaluate_gt=False)
    assert result["camera_count"] == 1
    assert result["warnings"]


def test_csv_schemas_contain_required_columns(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--outputs", str(outputs), "--output", str(output), "--no-save-plots"]) == 0
    with (output / "per_camera_metrics.csv").open(newline="") as stream:
        per_camera_header = next(csv.reader(stream))
    with (output / "link_ranking_table.csv").open(newline="") as stream:
        ranking_header = next(csv.reader(stream))
    with (output / "pairwise_relative_metrics.csv").open(newline="") as stream:
        pairwise_header = next(csv.reader(stream))
    assert set(PER_CAMERA_COLUMNS) <= set(per_camera_header)
    assert set(LINK_RANKING_COLUMNS) <= set(ranking_header)
    assert set(PAIRWISE_COLUMNS) <= set(pairwise_header)


def test_summary_json_has_aggregation_ready_top_level_keys(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    result = collect_calibration_metrics(dataset, outputs, evaluate_gt=False)
    required = {
        "input_paths",
        "output_paths",
        "gt_evaluation_enabled",
        "gt_metrics_available",
        "camera_count",
        "per_camera",
        "global_gt_free_metrics",
        "global_gt_based_metrics",
        "warnings",
        "plot_paths",
        "csv_paths",
        "report_path",
        "experiment_id",
        "setup_name",
        "fov",
        "camera_model",
        "board_config",
        "dataset_frame_count",
        "evaluated_frame_count",
    }
    assert required <= set(result)
    assert result["experiment_id"] == "experiment_001"
    assert result["fov"] == 180


def test_report_contains_required_section_headings(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--outputs", str(outputs), "--output", str(output), "--no-save-plots"]) == 0
    report = (output / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Run metadata",
        "## Executive summary",
        "## Per-camera summary",
        "## Detection and board pose quality",
        "## Link association and observability",
        "## Static mount calibration accuracy",
        "## Runtime `T_base_cam` validation",
        "## Relative pose validation for depth-model warping",
        "## GT-free quality indicators",
        "## Output files",
        "## Notes for interpretation",
    ):
        assert heading in report


def test_plot_generation_runs_on_minimal_gt_free_data(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--outputs", str(outputs), "--output", str(output), "--save-plots"]) == 0
    summary = json.loads((output / "calibration_metrics_summary.json").read_text())
    assert len(summary["plot_paths"]) >= 7
    assert (output / "plots" / "detection_valid_ratio.png").is_file()


def test_cli_smoke_writes_summary_csvs_and_report(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--outputs", str(outputs), "--output", str(output), "--experiment-id", "custom", "--no-save-plots"]) == 0
    assert (output / "calibration_metrics_summary.json").is_file()
    assert (output / "per_camera_metrics.csv").is_file()
    assert (output / "link_ranking_table.csv").is_file()
    assert (output / "gt_free_quality_metrics.csv").is_file()
    assert (output / "report.md").is_file()
    summary = json.loads((output / "calibration_metrics_summary.json").read_text())
    assert summary["experiment_id"] == "custom"


def test_explicit_no_gt_cli_mode_succeeds(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--outputs", str(outputs), "--output", str(output), "--no-evaluate-gt", "--no-save-plots"]) == 0
    summary = json.loads((output / "calibration_metrics_summary.json").read_text())
    assert summary["gt_evaluation_enabled"] is False
    assert summary["global_gt_based_metrics"] is None


def test_link_heatmap_plots_predicted_and_gt_markers(tmp_path):
    ranking = [
        {"camera_name": "camera_1", "link_name": "link_a", "score": 0.01},
        {"camera_name": "camera_1", "link_name": "link_b", "score": 0.2},
    ]
    camera_rows = [
        {
            "camera_name": "camera_1",
            "attached_link_name": "link_a",
            "gt_attached_link_name": "link_a",
        }
    ]
    path = tmp_path / "link_score_heatmap.png"
    assert evaluation_plots._link_score_heatmap(ranking, camera_rows, path) == str(path)
    assert path.is_file()
    _, links, _, annotations, gt_available = (
        evaluation_plots._build_link_score_heatmap_data(ranking, camera_rows)
    )
    assert gt_available is True
    assert annotations[0, links.index("link_a")] == "0.01\n★✓"


def test_link_heatmap_plots_predicted_marker_without_gt(tmp_path):
    ranking = [
        {"camera_name": "camera_1", "link_name": "link_a", "score": 0.01},
        {"camera_name": "camera_1", "link_name": "link_b", "score": 0.2},
    ]
    camera_rows = [{"camera_name": "camera_1", "attached_link_name": "link_a"}]
    path = tmp_path / "link_score_heatmap.png"
    assert evaluation_plots._link_score_heatmap(ranking, camera_rows, path) == str(path)
    _, links, _, annotations, gt_available = (
        evaluation_plots._build_link_score_heatmap_data(ranking, camera_rows)
    )
    assert gt_available is False
    assert "★" in annotations[0, links.index("link_a")]
    assert "✓" not in "".join(annotations.reshape(-1))


def test_link_heatmap_uses_log_colors_and_raw_score_annotations():
    ranking = [
        {"camera_name": "camera_1", "link_name": "link_a", "score": 0.01},
        {"camera_name": "camera_1", "link_name": "link_b", "score": 1.0},
    ]
    rows = [{"camera_name": "camera_1", "attached_link_name": "link_a"}]
    _, links, colors, annotations, _ = (
        evaluation_plots._build_link_score_heatmap_data(ranking, rows)
    )
    best_index = links.index("link_a")
    assert colors[0, best_index] == np.log10(
        0.01 + evaluation_plots.LINK_SCORE_LOG_EPSILON
    )
    assert annotations[0, best_index].startswith("0.01")


def test_heatmap_text_color_contrasts_with_viridis_cells():
    viridis = evaluation_plots.plt.get_cmap("viridis")
    assert evaluation_plots._readable_text_color(viridis(0.0)) == "white"
    assert evaluation_plots._readable_text_color(viridis(1.0)) == "#111111"


def test_static_translation_bar_converts_metres_to_millimetres(monkeypatch, tmp_path):
    captured = {}

    def capture(fig, path):
        axis = fig.axes[0]
        captured["heights"] = [bar.get_height() for bar in axis.patches]
        captured["labels"] = [text.get_text() for text in axis.texts]
        captured["ylabel"] = axis.get_ylabel()
        captured["title"] = axis.get_title()
        evaluation_plots.plt.close(fig)
        return str(path)

    monkeypatch.setattr(evaluation_plots, "_save", capture)
    evaluation_plots._bar_plot(
        [{"camera_name": "camera_1", "error_m": 0.0141}],
        "error_m",
        tmp_path / "translation.png",
        title="Static T_link_camera translation error",
        ylabel="Translation error (mm)",
        value_scale=1000.0,
        value_label=lambda value: f"{value:.1f} mm",
    )
    assert captured["heights"] == [14.1]
    assert captured["labels"] == ["14.1 mm"]
    assert captured["ylabel"] == "Translation error (mm)"
    assert captured["title"] == "Static T_link_camera translation error"


def test_static_rotation_bar_includes_numeric_labels(monkeypatch, tmp_path):
    captured = {}

    def capture(fig, path):
        axis = fig.axes[0]
        captured["labels"] = [text.get_text() for text in axis.texts]
        captured["title"] = axis.get_title()
        evaluation_plots.plt.close(fig)
        return str(path)

    monkeypatch.setattr(evaluation_plots, "_save", capture)
    evaluation_plots._bar_plot(
        [{"camera_name": "camera_1", "error_deg": 0.84}],
        "error_deg",
        tmp_path / "rotation.png",
        title="Static T_link_camera rotation error",
        ylabel="Rotation error (deg)",
        value_label=lambda value: f"{value:.2f}°",
    )
    assert captured["labels"] == ["0.84°"]
    assert captured["title"] == "Static T_link_camera rotation error"


def test_report_explains_link_markers_and_plot_units(tmp_path):
    cameras = ("Fisheye180_Cam1", "Fisheye180_Cam2")
    dataset, outputs = write_fixture(tmp_path, cameras)
    write_static_validation(outputs, cameras)
    summary = collect_calibration_metrics(dataset, outputs, evaluate_gt=True)
    report = build_report(summary)
    assert "★ = predicted/best link; ✓ = GT link" in report
    assert "All predicted links match GT." in report
    assert "displayed in millimetres" in report
    assert "CSV and JSON translation metrics remain in metres" in report


def test_plot_metadata_does_not_change_csv_schema(tmp_path):
    cameras = ("Fisheye180_Cam1", "Fisheye180_Cam2")
    dataset, outputs = write_fixture(tmp_path, cameras)
    write_static_validation(outputs, cameras)
    output = tmp_path / "evaluation"
    assert main(
        [
            "--dataset",
            str(dataset),
            "--outputs",
            str(outputs),
            "--output",
            str(output),
            "--evaluate-gt",
            "--no-save-plots",
        ]
    ) == 0
    with (output / "per_camera_metrics.csv").open(newline="") as stream:
        assert next(csv.reader(stream)) == PER_CAMERA_COLUMNS
    assert "gt_attached_link" not in PER_CAMERA_COLUMNS
    assert "gt_attached_link_name" not in PER_CAMERA_COLUMNS


def test_no_gt_evaluation_still_generates_predicted_link_heatmap(tmp_path):
    dataset, outputs = write_fixture(tmp_path)
    output = tmp_path / "evaluation"
    assert main(
        [
            "--dataset",
            str(dataset),
            "--outputs",
            str(outputs),
            "--output",
            str(output),
            "--no-evaluate-gt",
        ]
    ) == 0
    assert (output / "plots" / "link_score_heatmap.png").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "GT link markers are unavailable" in report
