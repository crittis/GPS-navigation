import numpy as np
import pytest

from gps_hybrid.gps_model import jacobian, objective, residual
from gps_hybrid.models import GPSObservations
from gps_hybrid.scenario import ScenarioConfig, generate_scenario
from gps_hybrid.solvers import (
    SolverConfig,
    solve_bfgs,
    solve_gauss_newton,
    solve_levenberg_marquardt,
)
import gps_hybrid.solvers.levenberg_marquardt as lm_module
from gps_hybrid.solvers.base import Evaluator


SOLVERS = [solve_gauss_newton, solve_levenberg_marquardt, solve_bfgs]


def test_gps_residual_and_objective_match_definition():
    satellites = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0], [-10.0, 0.0, 0.0]])
    state = np.array([1.0, 2.0, 3.0, 4.0])
    observed = np.linalg.norm(satellites - state[:3], axis=1) + state[3] + np.array([1, -2, 3, -4])
    obs = GPSObservations(satellites, observed)
    expected = np.array([1.0, -2.0, 3.0, -4.0])
    np.testing.assert_allclose(residual(state, obs), expected)
    assert objective(state, obs) == pytest.approx(15.0)


def test_analytical_jacobian_matches_central_finite_difference():
    scenario = generate_scenario(ScenarioConfig(seed=17))
    state = scenario.initial_estimate.copy()
    analytic = jacobian(state, scenario.observations)
    numeric = np.empty_like(analytic)
    eps = 1e-3
    for k in range(4):
        plus, minus = state.copy(), state.copy()
        plus[k] += eps
        minus[k] -= eps
        numeric[:, k] = (residual(plus, scenario.observations) -
                         residual(minus, scenario.observations)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_recover_noise_free_truth(solver):
    scenario = generate_scenario(ScenarioConfig(seed=5, pseudorange_noise_std_m=0.0))
    result = solver(scenario.observations, scenario.initial_estimate)
    assert result.converged, result.failure_reason
    np.testing.assert_allclose(result.estimated_state, scenario.truth.receiver_state, atol=1e-5)
    assert result.final_objective < 1e-12
    assert result.final_residual_rms < 1e-6
    assert result.iterations > 0
    assert result.function_evaluations > 0
    assert result.execution_time_seconds >= 0
    assert result.failure_reason is None
    assert result.convergence_history


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_minimize_noisy_scenario_without_using_truth_as_input(solver):
    scenario = generate_scenario(ScenarioConfig(seed=6, pseudorange_noise_std_m=2.0))
    # Solver interface receives observations and a supplied initial estimate only.
    result = solver(scenario.observations, scenario.initial_estimate)
    assert result.converged, result.failure_reason
    assert result.final_objective >= 0
    assert result.final_residual_rms == pytest.approx(
        np.sqrt(2.0 * result.final_objective / scenario.observations.pseudoranges.size)
    )
    assert np.linalg.norm(result.estimated_state - scenario.truth.receiver_state) < 20.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_report_convergence_with_high_pseudorange_noise(solver):
    scenario = generate_scenario(ScenarioConfig(seed=19, pseudorange_noise_std_m=1000.0))
    result = solver(scenario.observations, scenario.initial_estimate)
    assert result.converged, result.failure_reason
    assert result.final_residual_rms > 100.0
    assert np.isfinite(result.final_objective)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_report_rank_deficient_geometry(solver):
    receiver = np.array([6_378_137.0, 0.0, 0.0])
    satellites = receiver + np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
    obs = GPSObservations(satellites, np.array([20.0, 21.0, 22.0, 23.0]))
    result = solver(obs, np.array([*receiver, 0.0]))
    assert result.convergence_status == "rank_deficient"
    assert result.failure_reason


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_handle_large_initial_position_error(solver):
    scenario = generate_scenario(ScenarioConfig(
        seed=5, pseudorange_noise_std_m=0,
        initial_position_perturbation_std_m=0, initial_bias_perturbation_std_m=0,
    ))
    initial = scenario.truth.receiver_state + np.array([1e6, -1e6, 8e5, 2e5])
    result = solver(scenario.observations, initial)
    assert result.converged, result.failure_reason
    np.testing.assert_allclose(result.estimated_state, scenario.truth.receiver_state, atol=1e-4)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_accept_poor_but_full_rank_geometry(solver):
    truth = np.array([6_378_137.0, 0.0, 0.0, 75.0])
    directions = np.array([
        [1.0, 0.0, 1e-5], [-1.0, 0.0, 2e-5], [0.0, 1.0, -1e-5],
        [0.0, -1.0, 2e-5], [0.7, 0.7, 3e-5],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    satellites = truth[:3] + 22_000_000.0 * directions
    observations = GPSObservations(
        satellites, np.linalg.norm(satellites - truth[:3], axis=1) + truth[3]
    )
    result = solver(observations, truth + np.array([100.0, -100.0, 50.0, 25.0]))
    assert result.converged, result.failure_reason
    assert result.convergence_status != "rank_deficient"


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_terminate_at_configured_iteration_limit(solver):
    scenario = generate_scenario(ScenarioConfig(
        seed=5, pseudorange_noise_std_m=0,
        initial_position_perturbation_std_m=0, initial_bias_perturbation_std_m=0,
    ))
    initial = scenario.truth.receiver_state + np.array([1e6, -1e6, 8e5, 2e5])
    result = solver(scenario.observations, initial, SolverConfig(max_iterations=1))
    assert result.convergence_status == "max_iterations"
    assert result.iterations == 1
    assert result.failure_reason == "Maximum iteration count reached"


@pytest.mark.parametrize("field", [
    "gradient_tolerance", "step_tolerance", "objective_tolerance", "rank_rcond",
    "lm_initial_damping", "lm_acceptance_threshold", "lm_damping_increase",
    "lm_damping_decrease_factor", "line_search_contraction", "line_search_armijo",
    "line_search_min_step",
])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "invalid"])
def test_solver_config_rejects_non_numeric_or_non_finite_values(field, value):
    with pytest.raises(ValueError):
        SolverConfig(**{field: value})


def test_solver_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="lm_damping_increase"):
        SolverConfig(lm_damping_increase=0.99)
    with pytest.raises(ValueError, match="max_iterations"):
        SolverConfig(max_iterations=True)
    with pytest.raises(ValueError, match="rank_rcond"):
        SolverConfig(rank_rcond=1.1)
    with pytest.raises(ValueError, match="line_search_min_step"):
        SolverConfig(line_search_min_step=1.1)


@pytest.mark.parametrize("solver", [solve_gauss_newton, solve_bfgs])
def test_line_search_failure_is_reported(solver, monkeypatch):
    original = Evaluator.objective
    calls = {"count": 0}

    def objective_that_never_decreases(self, state):
        calls["count"] += 1
        current_r, current_f = original(self, state)
        return current_r, current_f if calls["count"] == 1 else current_f + 1e100

    monkeypatch.setattr(Evaluator, "objective", objective_that_never_decreases)
    scenario = generate_scenario(ScenarioConfig(seed=3))
    result = solver(
        scenario.observations, scenario.initial_estimate,
        SolverConfig(gradient_tolerance=1e-30, step_tolerance=1e-30,
                     objective_tolerance=1e-30),
    )
    assert result.convergence_status == "line_search_failed"
    assert result.failure_reason
    assert result.convergence_history and not result.convergence_history[-1].accepted


def test_lm_repeated_rejection_grows_damping():
    scenario = generate_scenario(ScenarioConfig(seed=3))
    result = solve_levenberg_marquardt(
        scenario.observations, scenario.initial_estimate,
        SolverConfig(max_iterations=4, lm_acceptance_threshold=1e100),
    )
    assert result.convergence_status == "max_iterations"
    rejected = [record for record in result.convergence_history if not record.accepted]
    assert len(rejected) == 4
    assert all(rejected[i + 1].damping > rejected[i].damping for i in range(3))


def _assert_lm_history_is_finite(result):
    for item in result.convergence_history:
        assert all(np.isfinite(value) for value in (
            item.iteration, item.objective, item.residual_rms,
            item.gradient_inf_norm, item.step_norm, item.step_scale,
            item.damping, item.gain_ratio,
        ))


def test_lm_rejects_finite_step_whose_euclidean_norm_is_unrepresentable(monkeypatch):
    monkeypatch.setattr(Evaluator, "objective", lambda _self, _state: (np.ones(4), 2.0))
    monkeypatch.setattr(Evaluator, "jacobian", lambda _self, _state: np.eye(4))
    monkeypatch.setattr(
        np.linalg, "lstsq",
        lambda *_args, **_kwargs: (
            np.array([1.3e308, 1.3e308, 0.0, 0.0]), np.array([]), 4, np.ones(4)
        ),
    )
    observations = GPSObservations(np.eye(4, 3), np.ones(4))
    result = solve_levenberg_marquardt(observations, np.zeros(4))
    assert result.convergence_status == "numerical_failure"
    assert "step norm exceeds floating-point range" in result.failure_reason
    _assert_lm_history_is_finite(result)


def test_lm_invalid_accepted_damping_reduction_is_numerical_failure(monkeypatch):
    scenario = generate_scenario(ScenarioConfig(seed=5, pseudorange_noise_std_m=0.0))
    monkeypatch.setattr(lm_module, "_accepted_damping", lambda *_args: float("nan"))
    result = solve_levenberg_marquardt(scenario.observations, scenario.initial_estimate)
    assert result.convergence_status == "numerical_failure"
    assert "damping reduction produced an invalid value" in result.failure_reason
    _assert_lm_history_is_finite(result)


def test_lm_damping_near_float_limit_fails_before_update_overflow(monkeypatch):
    scenario = generate_scenario(ScenarioConfig(seed=3))
    monkeypatch.setattr(lm_module, "_initial_damping_scales", lambda _j: np.ones(4))
    result = solve_levenberg_marquardt(
        scenario.observations, scenario.initial_estimate,
        SolverConfig(max_iterations=3, lm_initial_damping=1e308,
                     gradient_tolerance=1e-30, step_tolerance=1e-30,
                     objective_tolerance=1e-30),
    )
    assert result.convergence_status == "numerical_failure"
    assert "damping update would exceed floating-point range" in result.failure_reason
    assert result.convergence_history and not result.convergence_history[-1].accepted
    _assert_lm_history_is_finite(result)


def test_lm_rejects_nonfinite_scaling_or_initial_damping():
    scenario = generate_scenario(ScenarioConfig(seed=3))
    config = SolverConfig(lm_initial_damping=1e308)
    # The extreme but finite diagonal scaling makes lambda * scale
    # unrepresentable during initialization; it must fail deliberately.
    original = lm_module._initial_damping_scales
    lm_module._initial_damping_scales = lambda _j: np.full(4, 1e308)
    try:
        result = solve_levenberg_marquardt(
            scenario.observations, scenario.initial_estimate, config,
        )
    finally:
        lm_module._initial_damping_scales = original
    assert result.convergence_status == "numerical_failure"
    assert "Initial LM damping" in result.failure_reason
    assert "not representable" in result.failure_reason


def test_lm_nonfinite_scaling_values_report_numerical_failure(monkeypatch):
    scenario = generate_scenario(ScenarioConfig(seed=3))
    monkeypatch.setattr(lm_module, "_initial_damping_scales", lambda _j: np.full(4, np.nan))
    result = solve_levenberg_marquardt(scenario.observations, scenario.initial_estimate)
    assert result.convergence_status == "numerical_failure"
    assert "scaling is non-finite" in result.failure_reason


@pytest.mark.parametrize("reductions", [(np.inf, 1.0), (1.0, np.nan)])
def test_lm_nonfinite_reductions_report_numerical_failure(monkeypatch, reductions):
    scenario = generate_scenario(ScenarioConfig(seed=3))
    monkeypatch.setattr(lm_module, "_compute_reductions", lambda *_args: reductions)
    result = solve_levenberg_marquardt(scenario.observations, scenario.initial_estimate)
    assert result.convergence_status == "numerical_failure"
    assert "predicted or actual reduction is non-finite" in result.failure_reason


def test_lm_nonfinite_gain_ratio_reports_numerical_failure(monkeypatch):
    scenario = generate_scenario(ScenarioConfig(seed=3))
    monkeypatch.setattr(lm_module, "_gain_ratio", lambda _actual, _predicted: float("inf"))
    result = solve_levenberg_marquardt(scenario.observations, scenario.initial_estimate)
    assert result.convergence_status == "numerical_failure"
    assert "gain ratio is non-finite" in result.failure_reason


def test_lm_nonfinite_step_reports_numerical_failure(monkeypatch):
    scenario = generate_scenario(ScenarioConfig(seed=3))
    monkeypatch.setattr(
        np.linalg, "lstsq",
        lambda *_args, **_kwargs: (np.full(4, np.nan), np.array([]), 4, np.ones(4)),
    )
    result = solve_levenberg_marquardt(scenario.observations, scenario.initial_estimate)
    assert result.convergence_status == "numerical_failure"
    assert "invalid step" in result.failure_reason


def test_lm_nonfinite_candidate_state_reports_numerical_failure(monkeypatch):
    monkeypatch.setattr(Evaluator, "objective", lambda _self, _state: (np.ones(4), 2.0))
    monkeypatch.setattr(Evaluator, "jacobian", lambda _self, _state: np.eye(4))
    monkeypatch.setattr(
        np.linalg, "lstsq",
        lambda *_args, **_kwargs: (np.array([1e308, 0.0, 0.0, 0.0]), np.array([]), 4, np.ones(4)),
    )
    observations = GPSObservations(np.eye(4, 3), np.ones(4))
    result = solve_levenberg_marquardt(
        observations, np.array([1.7e308, 0.0, 0.0, 0.0]),
    )
    assert result.convergence_status == "numerical_failure"
    assert "candidate state is not representable" in result.failure_reason


def test_numerical_failure_is_reported(monkeypatch):
    def fail_lstsq(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic least-squares failure")

    monkeypatch.setattr(np.linalg, "lstsq", fail_lstsq)
    scenario = generate_scenario(ScenarioConfig(seed=3))
    result = solve_gauss_newton(scenario.observations, scenario.initial_estimate)
    assert result.convergence_status == "numerical_failure"
    assert "synthetic least-squares failure" in result.failure_reason

