import csv
import json
from pathlib import Path

import numpy as np
import pytest

import gps_hybrid.benchmarking as benchmarking
from gps_hybrid.benchmarking import (
    ALGORITHM_NAMES,
    BenchmarkConfig,
    generate_benchmark_scenario,
    generate_benchmark_scenarios,
    run_benchmark,
    write_benchmark_outputs,
)
from gps_hybrid.scenario import ScenarioConfig
from gps_hybrid.solvers import SolverResult


def _small_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        scenario_count=8,
        seed=1234,
        test_fraction=0.25,
        satellite_counts=(4, 6),
        noise_stds_m=(1.0,),
        position_perturbations_m=(10.0,),
        geometry_categories=("good", "poor", "rank_deficient"),
        outlier_rates=(0.0,),
    )


def test_identical_seeds_reproduce_scenarios_and_different_seeds_do_not():
    kwargs = dict(
        scenario_id="same", split="development", seed=7, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
    )
    first = generate_benchmark_scenario(**kwargs)
    second = generate_benchmark_scenario(**kwargs)
    different = generate_benchmark_scenario(**{**kwargs, "seed": 8})
    np.testing.assert_array_equal(
        first.scenario.observations.satellite_positions_ecef,
        second.scenario.observations.satellite_positions_ecef,
    )
    np.testing.assert_array_equal(first.scenario.observations.pseudoranges, second.scenario.observations.pseudoranges)
    assert not np.array_equal(
        first.scenario.observations.pseudoranges,
        different.scenario.observations.pseudoranges,
    )


def test_scenario_data_is_reused_across_all_algorithms(monkeypatch):
    config = _small_config()
    seen = []
    original = benchmarking._execute_algorithm

    def capture(scenario, algorithm, run_config):
        seen.append((scenario.scenario_id, algorithm, id(scenario.scenario.observations)))
        return original(scenario, algorithm, run_config)

    monkeypatch.setattr(benchmarking, "_execute_algorithm", capture)
    results = run_benchmark(config)
    for scenario_id in {item[0] for item in seen}:
        ids = {item[2] for item in seen if item[0] == scenario_id}
        assert len(ids) == 1
    assert len(results.run_rows) == config.scenario_count * len(ALGORITHM_NAMES)


def test_all_algorithms_are_recorded_and_unobservable_cases_are_not_skipped():
    results = run_benchmark(_small_config())
    assert {row["algorithm"] for row in results.run_rows} == set(ALGORITHM_NAMES)
    rank_rows = [row for row in results.run_rows if row["geometry_category"] == "rank_deficient"]
    assert rank_rows
    assert all(row["is_unobservable"] for row in rank_rows)
    assert len(results.manifest_rows) == 8


def test_truth_is_not_passed_to_standalone_solver(monkeypatch):
    scenario = generate_benchmark_scenario(
        scenario_id="one", split="test", seed=10, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
    )
    calls = []

    def fake_solver(observations, initial_state, config):
        calls.append((observations, initial_state, config))
        assert not hasattr(observations, "truth")
        assert not hasattr(observations, "receiver_state")
        return SolverResult(
            np.asarray(initial_state), 1.0, 1.0, 1, 1, 0.01,
            "converged", None, (),
        )

    monkeypatch.setitem(benchmarking.STANDALONE_SOLVERS, "GN", fake_solver)
    outcome = benchmarking._execute_algorithm(scenario, "GN", _small_config())
    assert outcome.status == "converged"
    assert len(calls) == 1


def test_invalid_solver_output_is_recorded_with_missing_accuracy_metrics(monkeypatch):
    scenario = generate_benchmark_scenario(
        scenario_id="one", split="test", seed=10, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
    )

    def invalid_solver(observations, initial_state, config):
        return SolverResult(
            np.array([np.nan, 0.0, 0.0, 0.0]), 1.0, 1.0, 1, 1, 0.01,
            "converged", None, (),
        )

    monkeypatch.setitem(benchmarking.STANDALONE_SOLVERS, "GN", invalid_solver)
    outcome = benchmarking._execute_algorithm(scenario, "GN", _small_config())
    row = benchmarking._run_row(scenario, "GN", outcome)
    assert outcome.status == "numerical_failure"
    assert row["is_converged"] is False
    assert row["position_error_m"] is None
    assert row["clock_bias_error_m"] is None


def test_observations_are_not_modified_by_benchmark():
    scenario = generate_benchmark_scenario(
        scenario_id="one", split="test", seed=10, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
    )
    satellites = scenario.scenario.observations.satellite_positions_ecef.copy()
    pseudoranges = scenario.scenario.observations.pseudoranges.copy()
    benchmarking.run_benchmark(BenchmarkConfig(
        scenario_count=1, satellite_counts=(6,), noise_stds_m=(1.0,),
        position_perturbations_m=(10.0,), geometry_categories=("good",),
    ))
    np.testing.assert_array_equal(scenario.scenario.observations.satellite_positions_ecef, satellites)
    np.testing.assert_array_equal(scenario.scenario.observations.pseudoranges, pseudoranges)


def test_output_csv_schema_metadata_and_summaries(tmp_path):
    results = run_benchmark(_small_config())
    output = write_benchmark_outputs(results, tmp_path / "results")
    expected = [
        output / "raw" / "scenario_manifest.csv",
        output / "raw" / "per_run_results.csv",
        output / "raw" / "per_stage_results.csv",
        *(output / "summaries" / f"{split}_{name}_summary.csv"
          for split in ("development", "test")
          for name in ("overall", "geometry", "noise", "satellite_count",
                       "initial_perturbation", "outlier_condition", "ablation")),
        output / "experiment_metadata.json",
        output / "figures",
    ]
    assert all(path.exists() for path in expected)
    with (output / "raw" / "per_run_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scenario_id", "algorithm", "condition_number", "pdop", "position_error_m",
                "convergence_status", "iterations", "function_evaluations", "solver_switches"}
    assert required <= set(rows[0])
    with (output / "experiment_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["scenario_count"] == 8
    assert "configuration" in metadata
    assert "split_coverage" in metadata


def test_split_summaries_do_not_mix_different_outcomes(tmp_path):
    rows = []
    for split, error in (("development", 1.0), ("test", 99.0)):
        rows.append({
            "scenario_id": split, "split": split, "algorithm": "GN",
            "geometry_category": "good", "noise_std_m": 1.0,
            "satellite_count": 4, "initial_perturbation_m": 10.0,
            "outlier_condition": "none", "is_converged": split == "development",
            "is_failure": split == "test", "is_unobservable": False,
            "is_numerical_failure": split == "test", "position_error_m": error,
            "clock_bias_error_m": None, "residual_rms_m": error,
            "iterations": 1, "function_evaluations": 1,
            "execution_time_seconds": 0.1, "rejected_steps": 0,
            "solver_switches": 0,
        })
    result = benchmarking.BenchmarkResults(tuple(), tuple(rows), tuple(), {})
    output = write_benchmark_outputs(result, tmp_path / "split-results")
    with (output / "summaries" / "test_overall_summary.csv").open(newline="", encoding="utf-8") as handle:
        test_summary = next(csv.DictReader(handle))
    with (output / "summaries" / "development_overall_summary.csv").open(newline="", encoding="utf-8") as handle:
        development_summary = next(csv.DictReader(handle))
    assert float(test_summary["position_error_m_mean"]) == 99.0
    assert float(development_summary["position_error_m_mean"]) == 1.0


def test_summary_statistics_preserve_missing_values_and_are_correct():
    rows = [
        {"algorithm": "GN", "is_converged": True, "is_failure": False,
         "is_unobservable": False, "is_numerical_failure": False,
         "position_error_m": 1.0, "clock_bias_error_m": None, "residual_rms_m": 2.0,
         "iterations": 2, "function_evaluations": 3, "execution_time_seconds": 0.1,
         "rejected_steps": 0, "solver_switches": 0},
        {"algorithm": "GN", "is_converged": False, "is_failure": True,
         "is_unobservable": False, "is_numerical_failure": True,
         "position_error_m": 3.0, "clock_bias_error_m": None, "residual_rms_m": None,
         "iterations": 4, "function_evaluations": 5, "execution_time_seconds": 0.2,
         "rejected_steps": 1, "solver_switches": 0},
    ]
    summary = benchmarking._summary_rows(rows, ("algorithm",))[0]
    assert summary["scenario_algorithm_rows"] == 2
    assert summary["converged_count"] == 1
    assert summary["converged_rate"] == 0.5
    assert summary["position_error_m_mean"] == 2.0
    assert summary["clock_bias_error_m_missing_count"] == 2
    assert summary["residual_rms_m_missing_count"] == 1


def test_development_test_split_is_reproducible_and_disjoint():
    config = _small_config()
    first = generate_benchmark_scenarios(config)
    second = generate_benchmark_scenarios(config)
    assert [scenario.split for scenario in first] == [scenario.split for scenario in second]
    assert {scenario.scenario_id for scenario in first if scenario.split == "development"}.isdisjoint(
        {scenario.scenario_id for scenario in first if scenario.split == "test"}
    )
    assert {scenario.split for scenario in first} == {"development", "test"}


def test_split_is_seeded_stratified_and_representative():
    config = BenchmarkConfig(
        scenario_count=60, seed=31, test_fraction=0.2,
        satellite_counts=(4, 6), noise_stds_m=(1.0, 10.0),
        position_perturbations_m=(10.0,), geometry_categories=("good", "poor"),
    )
    scenarios = generate_benchmark_scenarios(config)
    assert {item.split for item in scenarios} == {"development", "test"}
    for split in ("development", "test"):
        combinations = {
            (item.scenario_config.satellite_count,
             item.scenario_config.pseudorange_noise_std_m,
             item.requested_geometry_category)
            for item in scenarios if item.split == split
        }
        assert combinations == {(4, 1.0, "good"), (4, 1.0, "poor"),
                                (4, 10.0, "good"), (4, 10.0, "poor"),
                                (6, 1.0, "good"), (6, 1.0, "poor"),
                                (6, 10.0, "good"), (6, 10.0, "poor")}


def test_small_split_records_incomplete_combination_coverage():
    config = BenchmarkConfig(
        scenario_count=3, seed=41, test_fraction=1 / 3,
        satellite_counts=(4, 6), noise_stds_m=(1.0, 10.0),
        position_perturbations_m=(10.0,), geometry_categories=("good", "poor"),
    )
    results = run_benchmark(config)
    coverage = results.metadata["split_coverage"]
    assert coverage["development"]["total_parameter_combinations"] == 8
    assert coverage["test"]["total_parameter_combinations"] == 8
    assert not coverage["test"]["represented_all_combinations"]
    assert coverage["test"]["missing_parameter_combinations"]
    assert "small scenario counts" in coverage["limitation"]


def test_largest_remainder_split_size_and_stratum_caps():
    config = BenchmarkConfig(
        scenario_count=10, seed=123, test_fraction=0.3,
        satellite_counts=(4, 6), noise_stds_m=(1.0,),
        position_perturbations_m=(10.0,), geometry_categories=("good", "poor", "rank_deficient"),
    )
    scenarios = generate_benchmark_scenarios(config)
    assert sum(item.split == "test" for item in scenarios) == 3
    counts = {}
    for item in scenarios:
        key = (item.scenario_config.satellite_count, item.requested_geometry_category)
        counts.setdefault(key, [0, 0])
        counts[key][item.split == "test"] += 1
    assert all(test <= total for total, test in counts.values())


def test_largest_remainder_tie_breaking_is_seeded_and_reproducible():
    config = BenchmarkConfig(
        scenario_count=6, seed=10, test_fraction=1 / 3,
        satellite_counts=(4, 6, 8), noise_stds_m=(1.0,),
        position_perturbations_m=(10.0,), geometry_categories=("good",),
    )
    first = generate_benchmark_scenarios(config)
    second = generate_benchmark_scenarios(config)
    assert [item.split for item in first] == [item.split for item in second]
    assert sum(item.split == "test" for item in first) == 2
    # Each stratum has two members and the two test slots must be distributed
    # proportionally as one per stratum until the requested slots are exhausted.
    test_strata = {
        item.scenario_config.satellite_count
        for item in first if item.split == "test"
    }
    assert len(test_strata) == 2


def test_more_strata_than_test_scenarios_reports_partial_coverage():
    config = BenchmarkConfig(
        scenario_count=5, seed=4, test_fraction=0.2,
        satellite_counts=(4, 6, 8, 10), noise_stds_m=(1.0,),
        position_perturbations_m=(10.0,), geometry_categories=("good",),
    )
    results = run_benchmark(config)
    assert sum(item.split == "test" for item in generate_benchmark_scenarios(config)) == 1
    test_coverage = results.metadata["split_coverage"]["test"]
    assert test_coverage["missing_parameter_combinations"]
    assert not test_coverage["represented_all_combinations"]


def test_split_coverage_reports_missing_configured_factor_levels():
    config = BenchmarkConfig(
        scenario_count=2, seed=8, test_fraction=0.5,
        satellite_counts=(4, 6, 8), noise_stds_m=(1.0, 10.0),
        position_perturbations_m=(10.0, 100.0), geometry_categories=("good", "poor"),
        outlier_rates=(0.0, 1.0),
    )
    results = run_benchmark(config)
    test_factors = results.metadata["split_coverage"]["test"]["factor_coverage"]
    assert test_factors["satellite_count"]["missing_configured_values"]
    assert test_factors["noise_std_m"]["missing_configured_values"]
    assert test_factors["initial_perturbation_m"]["missing_configured_values"]
    assert test_factors["geometry_category"]["missing_configured_values"]
    assert test_factors["outlier_rate"]["missing_configured_values"]
    assert not test_factors["satellite_count"]["represented_all_configured_values"]


def test_position_perturbation_does_not_change_clock_bias_perturbation():
    common = dict(scenario_id="one", split="development", seed=10, satellite_count=6,
                  noise_std_m=1.0, geometry_category="good", clock_bias_perturbation_m=17.0)
    low = generate_benchmark_scenario(position_perturbation_m=10.0, **common)
    high = generate_benchmark_scenario(position_perturbation_m=1000.0, **common)
    assert low.scenario_config.initial_bias_perturbation_std_m == 17.0
    assert high.scenario_config.initial_bias_perturbation_std_m == 17.0


@pytest.mark.parametrize("kwargs", [
    {"noise_stds_m": ()}, {"position_perturbations_m": ()}, {"outlier_rates": ()},
])
def test_empty_experimental_collections_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"split": "validation"}, {"geometry_category": "orbital"},
    {"noise_std_m": np.nan}, {"clock_bias_perturbation_m": -1.0},
])
def test_direct_scenario_generation_validates_inputs(kwargs):
    base = dict(scenario_id="one", split="development", seed=10, satellite_count=6,
                noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good")
    with pytest.raises(ValueError):
        generate_benchmark_scenario(**{**base, **kwargs})


def test_both_split_requirement_is_explicit_for_small_runs():
    with pytest.raises(ValueError, match="both development and test"):
        generate_benchmark_scenarios(BenchmarkConfig(scenario_count=1, require_both_splits=True))


def test_outlier_labels_distinguish_none_partial_and_all():
    assert benchmarking._outlier_condition(0.0) == "none"
    assert benchmarking._outlier_condition(0.5) == "partial_observations"
    assert benchmarking._outlier_condition(1.0) == "all_observations"


@pytest.mark.parametrize("rate, expected_count, expected_fraction, expected_condition", [
    (0.0, 0, 0.0, "none"),
    (1.0, 6, 1.0, "all_observations"),
])
def test_realized_outlier_information_is_deterministic(rate, expected_count, expected_fraction, expected_condition):
    scenario = generate_benchmark_scenario(
        scenario_id="outliers", split="development", seed=10, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
        outlier_rate=rate, outlier_magnitude_m=1000.0,
    )
    assert scenario.realized_outlier_count == expected_count
    assert scenario.realized_outlier_fraction == expected_fraction
    assert scenario.realized_outlier_condition == expected_condition
    manifest = benchmarking._manifest_row(scenario)
    assert manifest["outlier_rate"] == rate
    assert manifest["realized_outlier_count"] == expected_count
    assert manifest["realized_outlier_condition"] == expected_condition


def test_solver_exception_and_invalid_result_have_no_metrics(monkeypatch):
    scenario = generate_benchmark_scenario(
        scenario_id="failure", split="test", seed=10, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
    )

    def raising_solver(observations, initial_state, config):
        raise RuntimeError("boom")

    def invalid_solver(observations, initial_state, config):
        return SolverResult(np.array([np.nan, 0.0, 0.0, 0.0]), 1.0, 1.0, 1, 1, 0.01,
                            "converged", None, ())

    for solver in (raising_solver, invalid_solver):
        monkeypatch.setitem(benchmarking.STANDALONE_SOLVERS, "GN", solver)
        outcome = benchmarking._execute_algorithm(scenario, "GN", _small_config())
        assert outcome.status == "numerical_failure"
        assert outcome.estimated_state is None
        assert outcome.final_objective is None
        assert outcome.residual_rms is None
        assert outcome.solver_reported_residual_rms is None
        assert outcome.iterations == 0
        assert outcome.function_evaluations == 0
        assert np.isfinite(outcome.execution_time_seconds)
        assert outcome.execution_time_seconds >= 0.0


def test_paired_summary_is_hand_calculated():
    rows = []
    for algorithm, position, converged, elapsed in (
        ("GN", 10.0, True, 1.0), ("fixed_gn_adaptive", 4.0, False, 3.0),
        ("initial_selection_only", 8.0, True, 2.0), ("full_hybrid", 5.0, True, 4.0),
    ):
        rows.append({"scenario_id": "s1", "algorithm": algorithm,
                     "position_error_m": position, "execution_time_seconds": elapsed,
                     "is_converged": converged})
    summaries = benchmarking._paired_summary_rows(rows)
    gn_pair = next(item for item in summaries if item["left_algorithm"] == "GN")
    assert gn_pair["available_pairs"] == 1
    assert gn_pair["position_error_m_difference_mean"] == 6.0
    assert gn_pair["convergence_outcome_difference_mean"] == 1.0
    assert gn_pair["execution_time_seconds_difference_mean"] == -2.0


def test_final_residual_rms_is_recomputed_and_solver_value_is_retained():
    scenario = generate_benchmark_scenario(
        scenario_id="one", split="test", seed=10, satellite_count=6,
        noise_std_m=1.0, position_perturbation_m=10.0, geometry_category="good",
    )
    estimate = np.asarray(scenario.scenario.initial_estimate, dtype=float)
    outcome = benchmarking._AlgorithmOutcome(
        "converged", None, estimate, 12.0, 999.0, 3.0, 3, 4, 0.1, 0, 0, ("GN",), tuple(),
    )
    row = benchmarking._run_row(scenario, "GN", outcome)
    expected = float(np.sqrt(np.mean(benchmarking.residual(
        estimate, scenario.scenario.observations,
    ) ** 2)))
    assert row["residual_rms_m"] == expected
    assert row["solver_reported_residual_rms_m"] == 3.0


def test_timing_metrics_are_finite_and_nonnegative():
    results = run_benchmark(BenchmarkConfig(
        scenario_count=1, satellite_counts=(6,), noise_stds_m=(1.0,),
        position_perturbations_m=(10.0,), geometry_categories=("good",),
    ))
    for row in results.run_rows:
        assert row["execution_time_seconds"] is not None
        assert np.isfinite(row["execution_time_seconds"])
        assert row["execution_time_seconds"] >= 0.0


def test_validity_accounting_uses_explicit_populations_and_preserves_dimensions():
    rows = [
        {
            "algorithm": "GN", "convergence_status": "converged", "is_converged": True,
            "is_failure": False, "is_unobservable": False, "is_numerical_failure": False,
            "validity_assessment_status": "valid_under_configured_policy",
            "validity_fit_status": "pass", "validity_geometry_status": "acceptable_conditioning",
            "validity_physical_status": "pass", "validity_physical_waived": False,
            "position_error_m": 10.0,
        },
        {
            "algorithm": "GN", "convergence_status": "converged", "is_converged": True,
            "is_failure": False, "is_unobservable": False, "is_numerical_failure": False,
            "validity_assessment_status": "invalid_under_configured_policy",
            "validity_fit_status": "fail", "validity_geometry_status": "poor_observability",
            "validity_physical_status": "fail", "validity_physical_waived": False,
            "position_error_m": 20.0,
        },
        {
            "algorithm": "GN", "convergence_status": "converged", "is_converged": True,
            "is_failure": False, "is_unobservable": True, "is_numerical_failure": False,
            "validity_assessment_status": "valid_under_configured_policy",
            "validity_fit_status": "pass", "validity_geometry_status": "acceptable_conditioning",
            "validity_physical_status": "not_applicable", "validity_physical_waived": True,
            "position_error_m": 30.0,
        },
        {
            "algorithm": "GN", "convergence_status": "max_iterations", "is_converged": False,
            "is_failure": True, "is_unobservable": True, "is_numerical_failure": False,
            "validity_assessment_status": "incomplete_assessment",
            "validity_fit_status": "not_evaluated", "validity_geometry_status": "rank_deficient",
            "validity_physical_status": "not_assessable", "validity_physical_waived": False,
            "position_error_m": None,
        },
        {
            "algorithm": "GN", "convergence_status": "converged", "is_converged": True,
            "is_failure": False, "is_unobservable": False, "is_numerical_failure": False,
            "validity_assessment_status": "incomplete_assessment",
            "validity_fit_status": "pass", "validity_geometry_status": "acceptable_conditioning",
            "validity_physical_status": "not_applicable", "validity_physical_waived": False,
            "position_error_m": None,
        },
    ]
    summary = benchmarking._summary_rows(rows, ("algorithm",))[0]
    assert (summary["N_all"], summary["N_converged"], summary["N_complete"]) == (5, 4, 3)
    assert (summary["valid_count"], summary["invalid_count"], summary["incomplete_count"]) == (2, 1, 2)
    assert summary["assessment_accounting_invariant"] is True
    assert summary["valid_rate_all_runs"] == 0.4
    assert summary["valid_rate_converged_runs"] == 0.5
    assert summary["invalid_rate_complete_assessments"] == pytest.approx(1 / 3)
    assert summary["incomplete_rate_all_runs"] == 0.4
    assert summary["validity_physical_status_evaluated_count"] == 2
    assert summary["validity_physical_status_failure_count"] == 1
    assert summary["validity_physical_status_not_applicable_or_waived_count"] == 1
    assert summary["validity_physical_waived_count"] == 1
    assert summary["validity_physical_status_unavailable_count"] == 2
    assert summary["validity_poor_observability_evaluated_count"] == 4
    assert summary["validity_poor_observability_failure_count"] == 1
    assert summary["validity_poor_observability_excluded_rank_deficient_count"] == 1
    assert summary["validity_rank_deficiency_evaluated_count"] == 5
    assert summary["validity_rank_deficiency_failure_count"] == 1
    assert summary["validity_fit_status_evaluated_count"] == 4
    assert summary["validity_fit_status_failure_count"] == 1
    assert summary["validity_fit_not_evaluated_count"] == 1
    assert summary["position_error_valid_only_m_mean"] == 20.0
    assert summary["position_error_converged_only_m_mean"] == 20.0
    assert summary["position_error_converged_only_m_missing_count"] == 1


def test_physical_summary_counts_unavailable_unrestricted_separately_from_waived():
    common = {
        "convergence_status": "converged", "is_converged": True,
        "is_failure": False, "is_unobservable": False, "is_numerical_failure": False,
        "validity_assessment_status": "incomplete_assessment",
        "validity_fit_status": "not_evaluated",
        "validity_geometry_status": "not_assessable", "position_error_m": None,
    }
    rows = [
        {**common, "algorithm": "GN", "validity_physical_status": "not_assessable",
         "validity_physical_waived": False},
        {**common, "algorithm": "GN", "validity_physical_status": "not_applicable",
         "validity_physical_waived": False},
        {**common, "algorithm": "GN", "validity_physical_status": "not_applicable",
         "validity_physical_waived": True},
    ]
    summary = benchmarking._summary_rows(rows, ("algorithm",))[0]
    assert summary["validity_physical_status_unavailable_count"] == 2
    assert summary["validity_physical_waived_count"] == 1
    assert summary["validity_physical_status_evaluated_count"] == 0


def test_benchmark_integrates_validity_and_keeps_split_metadata_separate(tmp_path):
    from gps_hybrid.solution_validity import (
        OperatingDomainConfig,
        SolutionValidityConfig,
    )

    config = BenchmarkConfig(
        scenario_count=4, seed=2026, test_fraction=0.25,
        satellite_counts=(8,), noise_stds_m=(0.0,),
        position_perturbations_m=(0.0,), geometry_categories=("good",),
        outlier_rates=(0.0,),
        validity_config=SolutionValidityConfig(
            operating_domain=OperatingDomainConfig("terrestrial", -1_000.0, 100_000.0),
            max_residual_rms_m=10.0,
            max_absolute_residual_m=50.0,
        ),
    )
    results = run_benchmark(config, output_dir=tmp_path / "validity-smoke")
    assert len(results.run_rows) == config.scenario_count * len(ALGORITHM_NAMES)
    for row in results.run_rows:
        assert row["convergence_status"] in {
            "converged", "rank_deficient", "unobservable", "unobservable_geometry",
            "insufficient_observations", "numerical_failure", "line_search_failed",
            "max_iterations", "max_total_iterations", "max_transitions", "solver_failure",
        }
        assert row["validity_assessment_status"] in {
            "valid_under_configured_policy", "invalid_under_configured_policy",
            "incomplete_assessment",
        }
        assert "validity_max_absolute_residual_m" in row
        assert "validity_final_numerical_rank" in row
    for split in ("development", "test"):
        groups = results.metadata["validity_accounting_by_split"][split]
        assert all(item["assessment_accounting_invariant"] for item in groups)
        assert all(item["N_all"] == sum(row["split"] == split and row["algorithm"] == item["algorithm"]
                                         for row in results.run_rows) for item in groups)
    raw_path = tmp_path / "validity-smoke" / "raw" / "per_run_results.csv"
    with raw_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(results.run_rows)
    assert "validity_assessment_status" in csv_rows[0]
    assert all(row["split"] in ("development", "test") for row in csv_rows)


@pytest.mark.parametrize("kwargs", [
    {"scenario_count": 0}, {"seed": -1}, {"test_fraction": 0.0},
    {"satellite_counts": (3,)}, {"noise_stds_m": (np.nan,)},
    {"outlier_rates": (1.1,)},
])
def test_benchmark_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)
