# Robot-mounted fisheye camera calibration

This repository calibrates robot-mounted fisheye cameras from synchronized
ChArUco images and robot link poses. It determines which candidate robot link
each camera is attached to and estimates the static camera mount transform
`T_link_camera`.

The main deployment artifact is:

```text
outputs/final_calibration/final_calibration.json
```

For every camera it contains `camera_name`, `attached_link`, and the static
`T_link_camera`, plus confidence, warnings, observability, and frame/adapter
metadata. Ground-truth error metrics are kept in separate evaluation-only
reports.

## Transform rule

All transforms use:

```text
T_A_B means p_A = T_A_B @ p_B
```

The depth model does **not** consume static `T_link_camera` directly. At runtime,
obtain the attached link pose from robot forward kinematics and compose:

```text
T_base_cam(t) = T_base_link(t) @ T_link_camera
```

The resulting per-frame `T_base_cam` is the depth-model camera pose.

## Quick start

Use Python 3.10–3.12 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Place a recording under `./dataset/`; never edit it through the pipeline. All
generated files are written under `./outputs/` and are ignored by Git.

Start with the [new-dataset quickstart](docs/QUICKSTART_NEW_DATASET.md), then use
the [full command sequence](docs/RUN_PIPELINE.md).

After board poses have been generated, routine static calibration can run the
three mount-calibration/export stages with one command:

```bash
python -m calibration_pipeline.run_static_calibration_pipeline \
  --dataset ./dataset \
  --board-poses ./outputs/board_poses \
  --output ./outputs \
  --evaluate-gt
```

Use `--no-evaluate-gt` for real/no-GT data. The separate lower-level commands
remain available for research and debugging.

Summarize one completed experiment as JSON, CSV tables, plots, and Markdown:

```bash
python -m calibration_pipeline.run_calibration_evaluation \
  --dataset ./dataset \
  --outputs ./outputs \
  --output ./outputs/evaluation \
  --evaluate-gt
```

This reporting command does not alter calibration results. Use
`--no-evaluate-gt` when Unity/offline validation reports are unavailable.
The candidate-link heatmap uses log-scale colors with raw residual-score
annotations: ★ marks the predicted link and ✓ marks the GT link when available.
Translation-error plots use millimetres and error bars carry numeric labels for
presentation readability; JSON and CSV translation metrics remain in metres.

## Documentation

- [Dataset format and required inputs](docs/DATASET_FORMAT.md)
- [Complete pipeline commands and troubleshooting](docs/RUN_PIPELINE.md)
- [Output files and deployment artifact](docs/OUTPUTS.md)
- [Frames, transform directions, and adapters](docs/FRAME_CONVENTIONS.md)
- [Quickstart for Unity and real-robot datasets](docs/QUICKSTART_NEW_DATASET.md)
- [New-dataset checklist](docs/NEW_DATASET_CHECKLIST.md)
- [Input-file samples and templates](docs/INPUT_FILE_SAMPLES.md)
- [Implemented milestones](MILESTONES.md)

The current real-robot input boundary is per-frame `T_base_link` for every
candidate link. Producing those transforms directly from ROS, URDF, DH
parameters, or a vendor SDK is outside the current offline pipeline.
