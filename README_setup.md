# Calibration pipeline setup

The commands below are run from the session root, where `calibration_pipeline/`,
`dataset/`, `outputs/`, and `tests/` are siblings. Use a local virtual environment;
do not install these dependencies into the system Python. Only
`opencv-contrib-python` is installed because installing it together with
`opencv-python` can produce conflicting `cv2` packages.

## macOS (uv, recommended)

Install Python 3.11 and uv if they are not already available, then run:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[test]'
uv lock
pytest
```

## Ubuntu (uv, recommended)

After installing Python 3.11 and uv:

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv libgl1 libglib2.0-0
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[test]'
uv lock
pytest
```

## pip-only alternative (macOS or Ubuntu)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```

Run the first-milestone integrity check with:

```bash
python -m calibration_pipeline.run_integrity_check \
  --dataset ./dataset \
  --output ./outputs/integrity_report.json
```

The command reads raw data but never modifies it. Generated files belong under
`outputs/`. It exits nonzero if frame alignment, configuration loading, or the
ground-truth composition check fails.

