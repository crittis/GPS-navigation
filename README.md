# GPS Hybrid Optimization

This repository contains a synthetic GPS pseudorange estimator with
Gauss–Newton, Levenberg–Marquardt, BFGS, geometry diagnostics, and a bounded
hybrid orchestration policy. Satellite directions are generated as synthetic
local ENU rays and transformed to ECEF; they are not orbital ephemerides or a
real visibility model.

## Run the tests

```powershell
D:\MyProjects\DAAPRO2\.venv\Scripts\python.exe -m pytest -q
```

## Run a debug benchmark

The debug benchmark uses 500 deterministic scenarios, shared across all
algorithm configurations, and writes raw rows and summaries under
`results/benchmark_debug`.

```powershell
D:\MyProjects\DAAPRO2\.venv\Scripts\python.exe -m gps_hybrid.benchmarking `
  --scenario-count 500 `
  --output-dir results/benchmark_debug
```

For a quick smoke run:

```powershell
D:\MyProjects\DAAPRO2\.venv\Scripts\python.exe -m gps_hybrid.benchmarking `
  --scenario-count 20 `
  --output-dir results/benchmark_smoke
```

Optional figures require Matplotlib in the selected environment:

```powershell
D:\MyProjects\DAAPRO2\.venv\Scripts\python.exe -m gps_hybrid.benchmarking `
  --scenario-count 500 `
  --output-dir results/benchmark_debug `
  --plot
```

The benchmark uses a seeded, stratified assignment across the experimental
parameter combinations. Test counts are allocated by largest remainder:
proportional floors are assigned first, then remaining test slots go to the
largest fractional remainders with seeded tie-breaking. A one-scenario smoke run is explicitly development-only;
set `require_both_splits=True` in Python when a held-out test split is required.
Summary files are split-specific: `summaries/development_*_summary.csv` contains
development rows, while `summaries/test_*_summary.csv` contains the final
held-out test rows. The corresponding `*_ablation_summary.csv` files also pair
only scenarios within that split; the raw CSVs remain shared and uncombined.
Metadata records factor and full-combination coverage for each split. When a
small scenario count cannot cover every combination, the metadata reports that
limitation instead of implying complete stratification.
Position and clock-bias initialization perturbations are independent. Satellite
directions are synthetic ENU rays, not real GPS visibility or orbital geometry.
Configured outlier rates are retained separately from realized affected counts,
fractions, and labels. Failures and missing metrics are retained rather than
dropped, and scenario truth is used only for post-solve metrics, never by a
solver or selector.
Final residual RMS is recomputed from the final estimate; the solver-reported
value is retained separately for consistency checks. Timing uses outer
`perf_counter` measurements without warm-up runs and excludes generation and
file I/O. Generated output is intentionally not part of the source tree's
required test fixtures.

The development-only policy review and pending configuration proposal are in
[the Phase 6.6 report](docs/phase_6_6_development_configuration_report.md) and
[the proposed configuration artifact](docs/phase_6_6_proposed_configuration.json).
