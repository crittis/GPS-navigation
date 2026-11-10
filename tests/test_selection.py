import numpy as np
import pytest

from gps_hybrid.geometry import diagnose_geometry
from gps_hybrid.selection import SelectionConfig, select_initial_solver

from test_geometry import RECEIVER, observations_from_enu, well_conditioned_observations


def poorly_conditioned_observations():
    directions = np.array([
        [1.0, 0.0, 0.0], [1.0, 1e-4, 1e-4],
        [1.0, -1e-4, 2e-4], [1.0, 2e-4, -1e-4],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return observations_from_enu(directions)


def test_well_conditioned_geometry_selects_gauss_newton():
    decision = select_initial_solver(diagnose_geometry(well_conditioned_observations(), RECEIVER))
    assert decision.decision == "gauss_newton"
    assert decision.solver_name == "Gauss-Newton"
    assert decision.solver is not None
    assert "Full-rank" in decision.reason


def test_poorly_conditioned_geometry_selects_lm_not_bfgs():
    diagnostics = diagnose_geometry(poorly_conditioned_observations(), RECEIVER)
    decision = select_initial_solver(diagnostics, SelectionConfig(condition_number_threshold=1e4))
    assert decision.decision == "levenberg_marquardt"
    assert decision.solver_name == "Levenberg-Marquardt"
    assert decision.solver is not None
    assert "condition-number" in decision.reason
    assert "BFGS" not in decision.reason


def test_rank_deficient_geometry_is_unsolvable_without_solver():
    observations = observations_from_enu(np.tile([1.0, 0.0, 0.0], (4, 1)))
    decision = select_initial_solver(diagnose_geometry(observations, RECEIVER))
    assert decision.decision == "unobservable"
    assert decision.solver is None
    assert not decision.solvable
    assert decision.reason


def test_insufficient_geometry_is_distinct_and_without_solver():
    decision = select_initial_solver(diagnose_geometry(observations_from_enu(np.eye(3)), RECEIVER))
    assert decision.decision == "insufficient_observations"
    assert decision.solver is None
    assert "four" in decision.reason


def test_condition_threshold_customization_changes_decision():
    diagnostics = diagnose_geometry(poorly_conditioned_observations(), RECEIVER)
    assert select_initial_solver(diagnostics, SelectionConfig(condition_number_threshold=1e12)).decision == "gauss_newton"
    assert select_initial_solver(diagnostics, SelectionConfig(condition_number_threshold=1.0)).decision == "levenberg_marquardt"


@pytest.mark.parametrize("threshold", [True, 0.5, np.nan, np.inf, "1e8"])
def test_invalid_selection_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        SelectionConfig(condition_number_threshold=threshold)


def test_every_selection_decision_has_an_explanation_and_bfgs_is_not_automatic():
    cases = [
        diagnose_geometry(well_conditioned_observations(), RECEIVER),
        diagnose_geometry(poorly_conditioned_observations(), RECEIVER),
        diagnose_geometry(observations_from_enu(np.tile([1.0, 0.0, 0.0], (4, 1))), RECEIVER),
        diagnose_geometry(observations_from_enu(np.eye(3)), RECEIVER),
    ]
    decisions = [select_initial_solver(case) for case in cases]
    assert all(decision.reason.strip() for decision in decisions)
    assert all(decision.solver_name != "BFGS" for decision in decisions)
