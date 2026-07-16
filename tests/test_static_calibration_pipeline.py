import json
from pathlib import Path

import pytest

from calibration_pipeline import run_static_calibration_pipeline as wrapper


def make_board_poses(root: Path, camera_name: str = "camera_1") -> Path:
    board_poses = root / "board_poses"
    board_poses.mkdir(parents=True)
    (board_poses / f"{camera_name}.jsonl").write_text("{}\n", encoding="utf-8")
    return board_poses


def base_argv(tmp_path: Path, board_poses: Path) -> list[str]:
    return [
        "--dataset",
        str(tmp_path / "dataset"),
        "--board-poses",
        str(board_poses),
        "--output",
        str(tmp_path / "outputs"),
    ]


def install_fake_stages(monkeypatch, output: Path, calls: list[tuple[str, list[str]]]):
    paths = wrapper.standard_output_paths(output)

    def link_main(arguments):
        calls.append(("link", list(arguments)))
        paths.link_calibration.parent.mkdir(parents=True, exist_ok=True)
        paths.link_calibration.write_text('{"cameras": []}', encoding="utf-8")
        return 0

    def recovery_main(arguments):
        calls.append(("recovery", list(arguments)))
        paths.shared_board_recovery.parent.mkdir(parents=True, exist_ok=True)
        paths.shared_board_recovery.write_text(
            '{"camera_results": []}', encoding="utf-8"
        )
        return 0

    def final_main(arguments):
        calls.append(("final", list(arguments)))
        paths.final_calibration.parent.mkdir(parents=True, exist_ok=True)
        paths.final_calibration.write_text(
            json.dumps(
                {
                    "camera_count": 1,
                    "cameras": [
                        {
                            "camera_name": "camera_1",
                            "attached_link": "robot/link_1",
                            "calibration_source": "shared_board_recovery",
                            "confidence": "high",
                            "warnings": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(wrapper.run_link_calibration, "main", link_main)
    monkeypatch.setattr(wrapper.run_shared_board_recovery, "main", recovery_main)
    monkeypatch.setattr(wrapper.run_final_calibration_export, "main", final_main)
    return paths


def test_argument_parsing_and_dry_run_smoke(tmp_path, capsys):
    board_poses = make_board_poses(tmp_path)
    arguments = base_argv(tmp_path, board_poses) + [
        "--camera",
        "camera_1",
        "--min-valid-poses",
        "12",
        "--min-anchor-cameras",
        "3",
        "--allow-single-anchor",
        "--dry-run",
    ]
    assert wrapper.main(arguments) == 0
    output = capsys.readouterr().out
    assert "Static calibration pipeline dry run." in output
    assert "calibration_pipeline.run_link_calibration" in output
    assert "--min-valid-poses 12" in output
    assert "--min-anchor-cameras 3" in output
    assert "--allow-single-anchor" in output


def test_missing_board_poses_fails_with_clear_message(tmp_path):
    with pytest.raises(SystemExit, match=wrapper.MISSING_BOARD_POSES_MESSAGE):
        wrapper.main(base_argv(tmp_path, tmp_path / "missing"))


def test_no_gt_mode_is_passed_to_every_stage(tmp_path, monkeypatch):
    board_poses = make_board_poses(tmp_path)
    calls = []
    install_fake_stages(monkeypatch, tmp_path / "outputs", calls)
    assert wrapper.main(base_argv(tmp_path, board_poses) + ["--no-evaluate-gt"]) == 0
    assert all("--no-evaluate-gt" in arguments for _, arguments in calls)
    assert all("--evaluate-gt" not in arguments for _, arguments in calls)


def test_evaluate_gt_mode_is_passed_to_every_stage(tmp_path, monkeypatch):
    board_poses = make_board_poses(tmp_path)
    calls = []
    install_fake_stages(monkeypatch, tmp_path / "outputs", calls)
    assert wrapper.main(base_argv(tmp_path, board_poses) + ["--evaluate-gt"]) == 0
    assert all("--evaluate-gt" in arguments for _, arguments in calls)
    assert all("--no-evaluate-gt" not in arguments for _, arguments in calls)


def test_standard_output_paths_match_lower_level_clis(tmp_path):
    paths = wrapper.standard_output_paths(tmp_path / "outputs")
    assert paths.link_calibration == (
        tmp_path / "outputs/link_calibration/link_calibration_summary.json"
    )
    assert paths.shared_board_recovery == (
        tmp_path
        / "outputs/shared_board_recovery/shared_board_recovery_summary.json"
    )
    assert paths.final_calibration == (
        tmp_path / "outputs/final_calibration/final_calibration.json"
    )


def test_final_summary_includes_calibration_path_and_camera_details(
    tmp_path, monkeypatch, capsys
):
    board_poses = make_board_poses(tmp_path)
    calls = []
    paths = install_fake_stages(monkeypatch, tmp_path / "outputs", calls)
    assert wrapper.main(base_argv(tmp_path, board_poses)) == 0
    output = capsys.readouterr().out
    assert f"- final calibration: {paths.final_calibration}" in output
    assert "- cameras: 1" in output
    assert "attached_link=robot/link_1" in output
    assert "source=shared_board_recovery" in output
    assert "confidence=high" in output
    assert "shared-board recovery used: yes" in output


def test_wrapper_calls_existing_cli_entrypoints_in_order_with_pass_throughs(
    tmp_path, monkeypatch
):
    board_poses = make_board_poses(tmp_path)
    calls = []
    install_fake_stages(monkeypatch, tmp_path / "outputs", calls)
    arguments = base_argv(tmp_path, board_poses) + [
        "--camera",
        "camera_1",
        "--min-valid-poses",
        "14",
        "--min-anchor-cameras",
        "4",
        "--allow-single-anchor",
    ]
    assert wrapper.main(arguments) == 0
    assert [name for name, _ in calls] == ["link", "recovery", "final"]
    link_arguments = calls[0][1]
    recovery_arguments = calls[1][1]
    final_arguments = calls[2][1]
    assert link_arguments[link_arguments.index("--min-valid-poses") + 1] == "14"
    assert recovery_arguments[
        recovery_arguments.index("--min-anchor-cameras") + 1
    ] == "4"
    assert "--allow-single-anchor" in recovery_arguments
    assert "--camera" in link_arguments and "--camera" in recovery_arguments
    assert "--camera" not in final_arguments
