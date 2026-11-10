from dataclasses import replace

import numpy as np
import pytest

import gps_hybrid.hybrid as hybrid
from gps_hybrid.geometry import GeometryDiagnostics, diagnose_geometry
from gps_hybrid.models import GPSObservations
from gps_hybrid.selection import SelectionConfig, SolverSelection
from gps_hybrid.solvers import IterationRecord, SolverConfig, SolverResult
from gps_hybrid.scenario import ScenarioConfig, generate_scenario

from test_geometry import RECEIVER, observations_from_enu, well_conditioned_observations


def _record(iteration, objective, accepted):
    return IterationRecord(iteration, objective, 1.0, 1.0, 1.0, accepted)


def _result(state, status, records=(), iterations=None, evaluations=1, reason="synthetic"):
    records = tuple(records)
    return SolverResult(
        np.asarray(state, dtype=float), 10.0, 1.0,
        len(records) if iterations is None else iterations,
        evaluations, 0.01, status, None if status == "converged" else reason, records,
    )


def _selection(name, solver, diagnostics):
    return SolverSelection(name.lower().replace("-", "").replace(" ", "_"), name, solver, "synthetic selection", diagnostics)


def _patch_selection(monkeypatch, name, solver, diagnostics):
    monkeypatch.setattr(
        hybrid, "select_initial_solver",
        lambda diagnostics, config: _selection(name, solver, diagnostics),
    )


def test_well_conditioned_actual_run_converges():
    scenario = generate_scenario(ScenarioConfig(seed=5, pseudorange_noise_std_m=0.0))
    result = hybrid.run_hybrid(scenario.observations, scenario.initial_estimate)
    assert result.converged
    assert result.solver_sequence == ("Gauss-Newton",)
    assert np.all(np.isfinite(result.estimated_state))


def test_poorly_conditioned_actual_run_uses_lm():
    directions = np.array([
        [1.0, 0.0, 0.0], [1.0, 1e-4, 1e-4],
        [1.0, -1e-4, 2e-4], [1.0, 2e-4, -1e-4],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    observations = observations_from_enu(directions)
    observations = GPSObservations(
        observations.satellite_positions_ecef,
        observations.pseudoranges + np.array([1.0, -2.0, 3.0, -1.0]),
    )
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.converged
    assert result.solver_sequence == ("Levenberg-Marquardt",)
    assert np.all(np.isfinite(result.estimated_state))


def test_initial_insufficient_observations_terminate_without_solver():
    observations = observations_from_enu(np.eye(3))
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "insufficient_observations"
    assert result.stage_results == ()
    assert result.solver_switches == 0


def test_initial_rank_deficiency_terminates_without_solver():
    observations = observations_from_enu(np.tile([1.0, 0.0, 0.0], (4, 1)))
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "unobservable_geometry"
    assert result.stage_results == ()


def test_gn_repeated_rejection_switches_to_lm_and_warm_starts(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)
    gn_state = RECEIVER + np.array([10.0, 0.0, 0.0, 1.0])
    lm_state = RECEIVER + np.array([1.0, 0.0, 0.0, 0.1])
    calls = []

    def fake_gn(obs, state, config):
        calls.append(("GN", state.copy(), config.max_iterations))
        return _result(gn_state, "line_search_failed", [
            _record(1, 20.0, False), _record(2, 20.0, False),
        ])

    def fake_lm(obs, state, config):
        calls.append(("LM", state.copy(), config.max_iterations))
        return _result(lm_state, "converged", [_record(1, 1.0, True)])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Levenberg-Marquardt", fake_lm)
    result = hybrid.run_hybrid(
        observations, RECEIVER,
        hybrid.HybridConfig(stage_iteration_budget=2, max_total_iterations=10),
    )
    assert result.converged
    assert result.solver_sequence == ("Gauss-Newton", "Levenberg-Marquardt")
    np.testing.assert_array_equal(calls[1][1], gn_state)
    assert result.transition_history[0].previous_solver == "Gauss-Newton"
    assert result.transition_history[0].next_solver == "Levenberg-Marquardt"
    assert result.transition_history[0].rejected_steps == 2


def test_single_rejection_does_not_switch(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "max_iterations", [_record(1, 20.0, False)])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "max_iterations"
    assert result.solver_sequence == ("Gauss-Newton",)
    assert result.transition_history == ()


def test_gn_stagnation_switches_to_lm(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "max_iterations", [
            _record(1, 10.0, True), _record(2, 10.0, True),
            _record(3, 10.0, True), _record(4, 10.0, True),
        ])

    def fake_lm(obs, state, config):
        return _result(state, "converged", [_record(1, 1.0, True)])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Levenberg-Marquardt", fake_lm)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.converged
    assert result.solver_sequence == ("Gauss-Newton", "Levenberg-Marquardt")
    assert "stagnation" in result.transition_history[0].trigger


def test_converged_stage_cannot_override_rank_deficient_final_geometry(monkeypatch):
    observations = well_conditioned_observations()
    good = diagnose_geometry(observations, RECEIVER)
    bad = replace(good, numerical_rank=3, full_rank=False,
                  condition_number=np.inf, pdop=None,
                  explanation="final geometry is rank deficient")
    diagnostics = iter([good, bad])
    monkeypatch.setattr(hybrid, "diagnose_geometry", lambda *args, **kwargs: next(diagnostics))

    def fake_gn(obs, state, config):
        return _result(state, "converged", [_record(1, 1.0, True)])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, good)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "unobservable_geometry"
    assert not result.converged
    assert result.final_geometry == bad


def test_invalid_output_after_valid_stage_preserves_previous_metrics(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)
    valid_state = RECEIVER + np.array([3.0, 0.0, 0.0, 0.2])

    def fake_gn(obs, state, config):
        return SolverResult(valid_state, 12.0, 2.0, 2, 4, 0.01,
                            "line_search_failed", "synthetic retry", (
                                _record(1, 12.0, False), _record(2, 12.0, False),
                            ))

    def fake_lm(obs, state, config):
        return SolverResult(np.array([np.nan, 0.0, 0.0, 0.0]), 999.0, 999.0,
                            1, 1, 0.01, "max_iterations", "invalid state", ())

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Levenberg-Marquardt", fake_lm)
    result = hybrid.run_hybrid(observations, RECEIVER)
    np.testing.assert_array_equal(result.estimated_state, valid_state)
    assert result.final_objective == 12.0
    assert result.final_residual_rms == 2.0
    assert result.convergence_status == "numerical_failure"
    assert result.solver_sequence == ("Gauss-Newton",)


def test_invalid_output_on_first_stage_uses_nan_metrics(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return SolverResult(np.array([np.nan, 0.0, 0.0, 0.0]), 999.0, 999.0,
                            1, 1, 0.01, "max_iterations", "invalid state", ())

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    np.testing.assert_array_equal(result.estimated_state, RECEIVER)
    assert np.isnan(result.final_objective)
    assert np.isnan(result.final_residual_rms)
    assert result.convergence_status == "numerical_failure"
    assert result.stage_results == ()


def test_solver_result_validation_rejects_invalid_metadata():
    valid = _result(RECEIVER, "converged", [_record(1, 1.0, True)])
    cases = [
        (replace(valid, iterations=True), "iterations"),
        (replace(valid, function_evaluations=-1), "function_evaluations"),
        (replace(valid, execution_time_seconds=np.nan), "execution time"),
        (replace(valid, final_objective=np.nan), "converged"),
        (replace(valid, convergence_history=[valid.convergence_history[0]]), "not a tuple"),
        (replace(valid, convergence_history=(replace(valid.convergence_history[0], objective=np.nan),)), "history objective"),
    ]
    for result, expected in cases:
        reason = hybrid._invalid_result_reason(result)
        assert reason is not None and expected in reason


def test_solver_result_validation_rejects_non_solver_result():
    assert hybrid._invalid_result_reason(object()) == "Solver returned an invalid result object"


def test_solver_exceeding_stage_budget_is_rejected_without_counting_stage(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "max_iterations", [_record(1, 10.0, True)], iterations=config.max_iterations + 1)

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(
        observations, RECEIVER,
        hybrid.HybridConfig(stage_iteration_budget=2, max_total_iterations=10),
    )
    assert result.convergence_status == "budget_integrity_failure"
    assert result.total_iterations == 0
    assert result.stage_results == ()
    np.testing.assert_array_equal(result.estimated_state, RECEIVER)
    assert np.isnan(result.final_objective)
    assert "exceeding" in result.failure_reason


def test_max_iterations_with_meaningful_progress_remains_distinct(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "max_iterations", [
            _record(1, 10.0, True), _record(2, 5.0, True), _record(3, 1.0, True),
        ])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "max_iterations"
    assert result.transition_history == ()


def test_max_iterations_with_insufficient_history_remains_distinct(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "max_iterations", ())

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "max_iterations"
    assert result.transition_history == ()


def test_lm_repeated_rejection_can_fallback_to_bfgs(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_lm(obs, state, config):
        return _result(state, "max_iterations", [
            _record(1, 10.0, False), _record(2, 10.0, False),
        ])

    def fake_bfgs(obs, state, config):
        return _result(state, "converged", [_record(1, 1.0, True)])

    _patch_selection(monkeypatch, "Levenberg-Marquardt", fake_lm, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Levenberg-Marquardt", fake_lm)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "BFGS", fake_bfgs)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.converged
    assert result.solver_sequence == ("Levenberg-Marquardt", "BFGS")


def test_bfgs_fallback_disabled_prevents_switch(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_lm(obs, state, config):
        return _result(state, "max_iterations", [
            _record(1, 10.0, False), _record(2, 10.0, False),
        ])

    _patch_selection(monkeypatch, "Levenberg-Marquardt", fake_lm, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Levenberg-Marquardt", fake_lm)
    result = hybrid.run_hybrid(
        observations, RECEIVER,
        hybrid.HybridConfig(enable_bfgs_fallback=False),
    )
    assert result.convergence_status == "solver_failure"
    assert result.solver_sequence == ("Levenberg-Marquardt",)


def test_converged_solver_does_not_switch(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "converged", [_record(1, 1.0, True)])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.converged
    assert result.solver_switches == 0


def test_transition_limit_is_respected(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "line_search_failed", [
            _record(1, 10.0, False), _record(2, 10.0, False),
        ])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(
        observations, RECEIVER, hybrid.HybridConfig(max_transitions=0),
    )
    assert result.convergence_status == "max_transitions"
    assert result.solver_switches == 0


def test_total_iteration_budget_is_not_reset_after_stage(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "line_search_failed", [
            _record(1, 10.0, False), _record(2, 10.0, False),
        ], iterations=2)

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(
        observations, RECEIVER,
        hybrid.HybridConfig(max_total_iterations=2, stage_iteration_budget=2),
    )
    assert result.convergence_status == "max_total_iterations"
    assert result.total_iterations == 2


def test_invalid_returned_state_is_rejected(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result([np.nan, 0.0, 0.0, 0.0], "max_iterations")

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "numerical_failure"
    assert "non-finite" in result.failure_reason


def test_geometry_is_rechecked_before_switch(monkeypatch):
    observations = well_conditioned_observations()
    good = diagnose_geometry(observations, RECEIVER)
    bad = replace(good, numerical_rank=3, full_rank=False, condition_number=np.inf,
                  pdop=None, explanation="became rank deficient")
    diagnostics = iter([good, bad])
    monkeypatch.setattr(hybrid, "diagnose_geometry", lambda *args, **kwargs: next(diagnostics))

    def fake_gn(obs, state, config):
        return _result(state, "line_search_failed", [
            _record(1, 10.0, False), _record(2, 10.0, False),
        ])

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, good)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert result.convergence_status == "unobservable_geometry"
    assert result.solver_switches == 0


def test_solver_exception_is_recorded_as_numerical_failure(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def failing_gn(obs, state, config):
        raise np.linalg.LinAlgError("synthetic failure")

    _patch_selection(monkeypatch, "Gauss-Newton", failing_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", failing_gn)
    result = hybrid.run_hybrid(
        observations, RECEIVER, hybrid.HybridConfig(max_transitions=0),
    )
    assert result.convergence_status == "max_transitions"
    assert "numerical failure" in result.failure_reason
    assert "synthetic failure" in result.stage_results[0].result.failure_reason


@pytest.mark.parametrize("kwargs", [
    {"max_total_iterations": 0}, {"max_transitions": -1},
    {"stage_iteration_budget": 0}, {"consecutive_failure_threshold": 0},
    {"objective_progress_tolerance": np.nan},
    {"enable_bfgs_fallback": 1}, {"max_total_iterations": True},
])
def test_hybrid_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        hybrid.HybridConfig(**kwargs)


def test_stage_histories_and_counters_are_aggregated(monkeypatch):
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)

    def fake_gn(obs, state, config):
        return _result(state, "line_search_failed", [
            _record(1, 10.0, False), _record(2, 10.0, False),
        ], evaluations=4)

    def fake_lm(obs, state, config):
        return _result(state, "converged", [_record(1, 1.0, True)], evaluations=3)

    _patch_selection(monkeypatch, "Gauss-Newton", fake_gn, diagnostics)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Gauss-Newton", fake_gn)
    monkeypatch.setitem(hybrid.SOLVER_FUNCTIONS, "Levenberg-Marquardt", fake_lm)
    result = hybrid.run_hybrid(observations, RECEIVER)
    assert len(result.stage_results) == 2
    assert sum(len(stage.result.convergence_history) for stage in result.stage_results) == 3
    assert result.total_iterations == 3
    assert result.total_function_evaluations == 7
    assert result.execution_time_seconds == pytest.approx(0.02)
