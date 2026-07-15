import json

from calibration_pipeline.dataset_loader import DatasetLoader, load_json_document


def test_json_comment_compatibility_is_explicit(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"value": 1}\n// {"example": 2}\n', encoding="utf-8")
    document, metadata = load_json_document(path)
    assert document == {"value": 1}
    assert metadata == {"comment_lines_ignored": 1, "strict_json": False}


def test_dataset_alignment_and_configuration():
    loader = DatasetLoader("dataset")
    report = loader.inspect_frame_alignment()
    assert report["passed"], json.dumps(report, indent=2)

    session_summary, _ = loader.load_json("session_summary.json")
    recorded_frame_count = session_summary.get("frame_count_so_far")
    if recorded_frame_count is not None:
        assert report["expected_frame_count"] == int(recorded_frame_count)
    else:
        assert report["expected_frame_count"] > 0
    assert len(loader.camera_names()) == 5

    config = loader.inspect_camera_configuration()
    assert config["passed"], json.dumps(config, indent=2)
    assert config["camera_model_config"]["comment_lines_ignored"] == 9
