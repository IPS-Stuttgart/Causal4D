"""Sealed sensor-reveal replay for proof-carrying active perception.

Synchronized public sensor channels remain challenge-owned until a frozen
sequential policy requests them. Only the realized reveal path is serialized;
terminal physical losses are opened after that trace is sealed and independently
verified.
"""

from ._sensor_reveal_case import (
    SensorRevealManifest,
    SensorRevealTruth,
    seal_sensor_reveal_case,
)
from ._sensor_reveal_execution import (
    SensorRevealEvent,
    SensorRevealScore,
    SensorRevealTrace,
    execute_sensor_reveal_plan,
    score_sensor_reveal_trace,
)
from ._sensor_reveal_submission import (
    SensorRevealPlan,
    SensorRevealSubmission,
    build_sensor_reveal_plan,
    validate_sensor_reveal_submission,
)

__all__ = [
    "SensorRevealEvent",
    "SensorRevealManifest",
    "SensorRevealPlan",
    "SensorRevealScore",
    "SensorRevealSubmission",
    "SensorRevealTrace",
    "SensorRevealTruth",
    "build_sensor_reveal_plan",
    "execute_sensor_reveal_plan",
    "score_sensor_reveal_trace",
    "seal_sensor_reveal_case",
    "validate_sensor_reveal_submission",
]
