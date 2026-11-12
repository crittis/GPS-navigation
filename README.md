# GPS Hybrid Optimization

A Python project for estimating a GPS receiver's position from synthetic pseudorange measurements.

The project includes three numerical solvers:

- **Gauss–Newton (GN)**
- **Levenberg–Marquardt (LM)**
- **BFGS**

It also includes satellite-geometry checks, a controller that can move between solvers under set conditions, a final solution-validity check, and benchmark tools.

The satellite directions used in the experiments are generated as local ENU (east, north, up) directions and converted to ECEF coordinates. They are synthetic directions, not real satellite orbits, ephemerides, or a model of satellite visibility.

## Requirements

- Python 3.10 or later
- NumPy
- Pytest (for running tests)
- Matplotlib (only if benchmark plots are required)

## Setup on Windows

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

If you want to generate benchmark figures, install Matplotlib too:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
```

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Run a benchmark

The benchmark runs the same generated scenarios across the available solver approaches. By default, it uses 500 scenarios and writes the results to `results/benchmark_debug`.

```powershell
.\.venv\Scripts\python.exe -m gps_hybrid.benchmarking `
  --scenario-count 500 `
  --output-dir results/benchmark_debug
```

For a smaller smoke run:

```powershell
.\.venv\Scripts\python.exe -m gps_hybrid.benchmarking `
  --scenario-count 20 `
  --output-dir results/benchmark_smoke
```

To create plots as well, add `--plot`:

```powershell
.\.venv\Scripts\python.exe -m gps_hybrid.benchmarking `
  --scenario-count 500 `
  --output-dir results/benchmark_debug `
  --plot
```

## Benchmark output

Each benchmark run creates:

- `raw/per_run_results.csv` – results for each scenario and solver approach.
- `raw/per_stage_results.csv` – details of the solver stages.
- `raw/scenario_manifest.csv` – scenario settings and split assignments.
- `summaries/` – summary CSV files for the development and test splits.
- `experiment_metadata.json` – benchmark settings and scenario-coverage information.
- `figures/` – plots, when `--plot` is used.

The benchmark uses a fixed random seed by default, so the same settings can be run again. Scenarios are assigned to development and test splits using a seeded, stratified procedure. The test split is kept separate in the summary files. If a small scenario count cannot cover all parameter combinations, the metadata records the coverage limitation.

A one-scenario run is development-only and does not create a held-out test split. When creating a benchmark through the Python API, set `require_both_splits=True` in `BenchmarkConfig` if both splits are required.

Position and clock-bias initialisation perturbations are generated independently. The benchmark records configured outlier rates separately from the number of outliers that actually occur. Failed runs and missing metrics are kept in the output rather than removed.

The final residual RMS is calculated again from the final estimate. The solver-reported residual value is also retained for comparison. Runtime measurements exclude scenario generation and file writing.

## Project structure

```text
.
├── docs/
├── src/
│   └── gps_hybrid/
│       ├── solvers/
│       ├── benchmarking.py
│       ├── coordinates.py
│       ├── geometry.py
│       ├── gps_model.py
│       ├── hybrid.py
│       ├── models.py
│       ├── scenario.py
│       ├── selection.py
│       └── solution_validity.py
├── tests/
├── pyproject.toml
└── README.md
```

The development configuration review and its proposed settings are documented here:

- [Phase 6.6 development configuration report](docs/phase_6_6_development_configuration_report.md)
- [Proposed configuration](docs/phase_6_6_proposed_configuration.json)
