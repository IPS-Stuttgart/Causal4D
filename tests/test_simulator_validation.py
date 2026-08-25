from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    SimulatorConfig,
    WorldCondition,
    simulate,
)


def _parameters() -> PhysicalParameters:
    return PhysicalParameters(stiffness=2.0, damping=0.4, contact_gain=1.1)


def _graph(**overrides: object) -> GraphObject:
    values: dict[str, object] = {
        "name": "line",
        "rest_positions": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        "edges": ((0, 1),),
        "mass": 1.0,
        "support_stiffness": 0.2,
        "true_parameters": _parameters(),
        "sensor_nodes": (0, 1),
    }
    values.update(overrides)
    return GraphObject(**values)


def _action(**overrides: object) -> Action:
    values: dict[str, object] = {
        "action_id": "push",
        "split": "test",
        "contact_nodes": (0,),
        "commanded_forces": np.ones((4, 1, 2), dtype=float),
    }
    values.update(overrides)
    return Action(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stiffness", 0.0),
        ("stiffness", -1.0),
        ("stiffness", np.nan),
        ("damping", np.inf),
        ("contact_gain", True),
    ],
)
def test_physical_parameters_reject_invalid_scalars(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "stiffness": 2.0,
        "damping": 0.4,
        "contact_gain": 1.1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        PhysicalParameters(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "  "}, "name must be a non-empty string"),
        ({"mass": np.nan}, "mass must be finite"),
        ({"mass": 0.0}, "mass must be positive"),
        ({"support_stiffness": np.inf}, "support_stiffness must be finite"),
        ({"support_stiffness": -1.0}, "support_stiffness must be non-negative"),
        ({"edges": ((0, 1), (1, 0))}, "edges must not contain duplicates"),
        ({"edges": ((False, 1),)}, "edge node indices must be integers"),
        ({"sensor_nodes": (0, 0)}, "sensor_nodes must not contain duplicates"),
        ({"sensor_nodes": (False, 1)}, "sensor node indices must be integers"),
    ],
)
def test_graph_object_rejects_ambiguous_or_nonfinite_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _graph(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"action_id": ""}, "action_id must be a non-empty string"),
        (
            {
                "contact_nodes": (),
                "commanded_forces": np.ones((4, 0, 2), dtype=float),
            },
            "contact_nodes must be non-empty",
        ),
        (
            {
                "contact_nodes": (0, 0),
                "commanded_forces": np.ones((4, 2, 2), dtype=float),
            },
            "contact_nodes must not contain duplicates",
        ),
        ({"contact_nodes": (-1,)}, "contact_nodes must contain non-negative integers"),
        (
            {"contact_nodes": (True,)},
            "contact_nodes must contain non-negative integers",
        ),
    ],
)
def test_action_rejects_invalid_identity_and_contact_support(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _action(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "name must be a non-empty string"),
        ({"contact_gain_multiplier": np.nan}, "contact_gain_multiplier must be finite"),
        ({"contact_delay_steps": 1.5}, "contact_delay_steps must be an integer"),
        ({"contact_delay_steps": True}, "contact_delay_steps must be an integer"),
        ({"shift_contact_nodes": 1}, "shift_contact_nodes must be boolean"),
        ({"contact_spread": np.nan}, "contact_spread must be finite"),
        ({"nonlinear_stiffening": np.inf}, "nonlinear_stiffening must be finite"),
    ],
)
def test_world_condition_rejects_nonfinite_or_ambiguous_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {"name": "condition"}
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        WorldCondition(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frame_count": True}, "frame_count must be an integer"),
        ({"dt": np.nan}, "dt must be finite"),
        ({"velocity_drag": np.inf}, "velocity_drag must be finite"),
    ],
)
def test_simulator_config_rejects_nonfinite_or_boolean_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SimulatorConfig(**overrides)


def test_action_contact_nodes_are_checked_against_graph_at_binding_time() -> None:
    graph = _graph()
    action = _action(contact_nodes=(2,))

    with pytest.raises(ValueError, match="invalid contact node"):
        simulate(
            graph,
            action,
            _parameters(),
            WorldCondition(name="condition"),
            SimulatorConfig(frame_count=5),
        )


def test_validation_does_not_change_valid_simulation_or_serialization() -> None:
    graph = _graph()
    action = _action()
    condition = WorldCondition(name="condition", contact_spread=0.2)
    config = SimulatorConfig(frame_count=5, dt=0.03, velocity_drag=0.18)
    trajectory = simulate(graph, action, _parameters(), condition, config)

    assert trajectory.shape == (5, 2, 2)
    assert np.all(np.isfinite(trajectory))
    assert graph.as_dict()["edges"] == [[0, 1]]
    assert action.as_dict()["contact_nodes"] == [0]
    assert replace(condition, contact_spread=0.0).contact_spread == 0.0
