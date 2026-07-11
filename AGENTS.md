# Project rules for Codex

- Use this repository's current folder as the session root. The
  `calibration_pipeline/`, `dataset/`, `outputs/`, and `tests/` directories are
  siblings under this root.
- Treat `dataset/` as local raw data. Do not modify, move, rename, or delete any
  dataset file.
- Write every generated report, cache, detection, overlay, and plot only under
  `outputs/`.
- Work only within the milestone explicitly requested. Do not begin a later
  milestone without user direction.
- Run pytest before finishing a milestone.
- Run the CLI relevant to the milestone before finishing it.
- At handoff, summarize changed files, the CLI result, and the test results.

