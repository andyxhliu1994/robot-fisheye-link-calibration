def test_required_environment_imports():
    import cv2
    import numpy
    import pandas
    import pytest
    import scipy

    assert numpy.__version__
    assert scipy.__version__
    assert pandas.__version__
    assert pytest.__version__
    assert cv2.__version__
    assert hasattr(cv2, "aruco")


def test_only_contrib_opencv_distribution_is_declared():
    requirements = open("requirements.txt", encoding="utf-8").read().splitlines()
    assert any(line.startswith("opencv-contrib-python") for line in requirements)
    assert not any(line.startswith("opencv-python") for line in requirements)

