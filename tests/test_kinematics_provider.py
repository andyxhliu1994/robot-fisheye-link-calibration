import numpy as np

from calibration_pipeline.kinematics_provider import UnityLinkPoseProvider


def test_unity_provider_returns_all_candidate_link_poses():
    provider = UnityLinkPoseProvider("dataset")
    poses = provider.get_candidate_link_poses("frame_000000")
    assert len(poses) == 6
    assert all(value.shape == (4, 4) for value in poses.values())
    assert all(np.allclose(value[3], [0, 0, 0, 1]) for value in poses.values())


def test_unity_provider_accepts_numeric_frame_id():
    provider = UnityLinkPoseProvider("dataset")
    assert provider.frame_path(12).name == "frame_000012.json"

