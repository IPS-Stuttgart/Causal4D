"""Version 1 of Causal4D's supported inference API.

The surface follows the scientific pipeline directly: factual intervention
abduction, explicit counterfactual action, and physical-posterior projection.
Portable contracts and archive I/O live in ``causal4d.artifacts.v1``.
"""

from __future__ import annotations

from typing import Final

from causal4d.counterfactual import (
    apply_counterfactual_operator,
    project_physical_posterior,
)
from causal4d.hierarchical_abduction import (
    HierarchicalAbductionResult,
    abduct_hierarchical_interventions,
)
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    factual_joint_weights,
)


PUBLIC_API_NAME: Final = "causal4d.inference.v1"
PUBLIC_API_VERSION: Final = 1

__all__ = [
    "PUBLIC_API_NAME",
    "PUBLIC_API_VERSION",
    "FactualAbductionConfig",
    "HierarchicalAbductionResult",
    "abduct_factual_intervention",
    "abduct_hierarchical_interventions",
    "apply_counterfactual_operator",
    "factual_joint_weights",
    "project_physical_posterior",
]
