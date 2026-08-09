from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRANSPORT_FILES = (
    ROOT / ".github" / "workflows" / "apply-artifact-index-hardening.yml",
    ROOT / "scripts" / "ci" / "apply_artifact_index_hardening.py",
)


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one replacement in {path.relative_to(ROOT)}; found {count}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _update_rollout_bank() -> None:
    path = ROOT / "src" / "causal4d" / "rollout_bank.py"
    _replace_once(
        path,
        "from causal4d.immutable_array import readonly_array as _readonly_array\n",
        "from causal4d.immutable_array import (\n"
        "    readonly_array as _readonly_array,\n"
        "    readonly_integer_array as _readonly_integer_array,\n"
        ")\n",
    )
    _replace_once(
        path,
        "        nodes = _readonly_array(self.node_indices, dtype=int)\n",
        "        nodes = _readonly_integer_array(\n"
        "            self.node_indices,\n"
        "            name=\"node_indices\",\n"
        "        )\n",
    )
    _replace_once(
        path,
        "    def update_from_observations(\n",
        "    def update_from_observations_legacy_v1(\n",
    )
    _replace_once(
        path,
        '        """Update only from frames before ``prefix_frame_count``."""\n',
        '        """Run the registered legacy-v1 dense prefix update."""\n',
    )
    _replace_once(
        path,
        "        nodes = np.asarray(\n"
        "            tuple(range(self.node_count))\n"
        "            if observed_nodes is None\n"
        "            else tuple(observed_nodes),\n"
        "            dtype=int,\n"
        "        )\n"
        "        if (\n"
        "            nodes.ndim != 1\n"
        "            or not len(nodes)\n"
        "            or np.any(nodes < 0)\n"
        "            or np.any(nodes >= self.node_count)\n"
        "        ):\n"
        "            raise ValueError(\"observed_nodes must identify available rollout nodes\")\n",
        "        raw_nodes = (\n"
        "            tuple(range(self.node_count))\n"
        "            if observed_nodes is None\n"
        "            else tuple(observed_nodes)\n"
        "        )\n"
        "        nodes = _readonly_integer_array(raw_nodes, name=\"observed_nodes\")\n"
        "        if (\n"
        "            nodes.ndim != 1\n"
        "            or not len(nodes)\n"
        "            or np.any(nodes < 0)\n"
        "            or np.any(nodes >= self.node_count)\n"
        "            or len(np.unique(nodes)) != len(nodes)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"observed_nodes must uniquely identify available rollout nodes\"\n"
        "            )\n",
    )
    marker = "    def _interpolated_nodes(self, evidence: SparseTrajectoryEvidence) -> np.ndarray:\n"
    wrapper = '''    def update_from_observations(
        self,
        observations_m: np.ndarray,
        *,
        prefix_frame_count: int,
        scale_m: float,
        likelihood_power: float = 1.0,
        dynamic_likelihood_weight: float = 0.0,
        degrees_of_freedom: float = 4.0,
        mask: np.ndarray | None = None,
        observed_nodes: Sequence[int] | None = None,
        base_weights: np.ndarray | None = None,
        particle_discrepancy_m: np.ndarray | None = None,
        particle_discrepancy_variance_m2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Backward-compatible alias for the registered legacy-v1 update."""

        return self.update_from_observations_legacy_v1(
            observations_m,
            prefix_frame_count=prefix_frame_count,
            scale_m=scale_m,
            likelihood_power=likelihood_power,
            dynamic_likelihood_weight=dynamic_likelihood_weight,
            degrees_of_freedom=degrees_of_freedom,
            mask=mask,
            observed_nodes=observed_nodes,
            base_weights=base_weights,
            particle_discrepancy_m=particle_discrepancy_m,
            particle_discrepancy_variance_m2=(
                particle_discrepancy_variance_m2
            ),
        )

'''
    _replace_once(path, marker, wrapper + marker)


def _update_prefix_likelihood() -> None:
    path = ROOT / "src" / "causal4d" / "prefix_likelihood.py"
    _replace_once(
        path,
        "import numpy as np\n\nfrom causal4d.weighting import log_weights_from_probabilities\n",
        "import numpy as np\n\n"
        "from causal4d.immutable_array import readonly_integer_array\n"
        "from causal4d.weighting import log_weights_from_probabilities\n",
    )
    _replace_once(
        path,
        "    nodes = np.asarray(\n"
        "        tuple(range(bank.node_count))\n"
        "        if observed_nodes is None\n"
        "        else tuple(observed_nodes),\n"
        "        dtype=int,\n"
        "    )\n",
        "    raw_nodes = (\n"
        "        tuple(range(bank.node_count))\n"
        "        if observed_nodes is None\n"
        "        else tuple(observed_nodes)\n"
        "    )\n"
        "    nodes = readonly_integer_array(raw_nodes, name=\"observed_nodes\")\n",
    )


def _update_phystwin_backend() -> None:
    path = ROOT / "src" / "causal4d" / "phystwin_backend.py"
    _replace_once(
        path,
        "from causal4d.immutable_array import readonly_array as _readonly_array\n",
        "from causal4d.immutable_array import (\n"
        "    readonly_array as _readonly_array,\n"
        "    readonly_integer_array as _readonly_integer_array,\n"
        ")\n"
        "from causal4d.numpy_archive import load_numpy_archive\n",
    )
    _replace_once(
        path,
        "        indices = _readonly_array(self.grid_indices, dtype=int)\n",
        "        indices = _readonly_integer_array(\n"
        "            self.grid_indices,\n"
        "            name=\"grid_indices\",\n"
        "        )\n",
    )
    _replace_once(
        path,
        "    weight_key: str | None = None,\n"
        "    support_method: SupportMethod = \"top_mass\",\n",
        "    weight_key: str | None = None,\n"
        "    support_method: SupportMethod = \"top_mass\",\n"
        "    expected_sha256: str | None = None,\n",
    )
    old = '''    with np.load(profile_path, allow_pickle=False) as archive:
        required = {"object_log_scales", "controller_log_scales"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "parameter profile is missing: " + ", ".join(sorted(missing))
            )
        object_grid = np.asarray(archive["object_log_scales"], dtype=float)
        controller_grid = np.asarray(archive["controller_log_scales"], dtype=float)
        if weight_key is None:
            available = [
                key
                for key in ("prediction_weights", "posterior_weights")
                if key in archive.files
            ]
            if not available:
                raise ValueError("parameter profile has no posterior weight grid")
            selected_weight_key = available[0]
        else:
            selected_weight_key = weight_key
            if selected_weight_key not in archive.files:
                raise ValueError(f"parameter profile has no {selected_weight_key!r}")
        weight_grid = np.asarray(archive[selected_weight_key], dtype=float)
        source_prediction_grid = (
            np.asarray(archive["source_prediction_weights"], dtype=float)
            if "source_prediction_weights" in archive.files
            else None
        )
        posterior_grid = (
            np.asarray(archive["posterior_weights"], dtype=float)
            if "posterior_weights" in archive.files
            else None
        )
'''
    new = '''    profile = load_numpy_archive(
        profile_path,
        expected_sha256=expected_sha256,
        name="Bayesian-PhysTwin parameter profile",
    )
    archive = profile.arrays
    required = {"object_log_scales", "controller_log_scales"}
    missing = required - set(archive)
    if missing:
        raise ValueError(
            "parameter profile is missing: " + ", ".join(sorted(missing))
        )
    object_grid = np.asarray(archive["object_log_scales"], dtype=float)
    controller_grid = np.asarray(archive["controller_log_scales"], dtype=float)
    if weight_key is None:
        available = [
            key
            for key in ("prediction_weights", "posterior_weights")
            if key in archive
        ]
        if not available:
            raise ValueError("parameter profile has no posterior weight grid")
        selected_weight_key = available[0]
    else:
        selected_weight_key = weight_key
        if selected_weight_key not in archive:
            raise ValueError(f"parameter profile has no {selected_weight_key!r}")
    weight_grid = np.asarray(archive[selected_weight_key], dtype=float)
    source_prediction_grid = (
        np.asarray(archive["source_prediction_weights"], dtype=float)
        if "source_prediction_weights" in archive
        else None
    )
    posterior_grid = (
        np.asarray(archive["posterior_weights"], dtype=float)
        if "posterior_weights" in archive
        else None
    )
'''
    _replace_once(path, old, new)


def _update_intervention_abduction() -> None:
    _replace_once(
        ROOT / "src" / "causal4d" / "intervention_abduction.py",
        "            joint_weights = bank.update_from_observations(\n",
        "            joint_weights = bank.update_from_observations_legacy_v1(\n",
    )


def _update_changelog() -> None:
    _replace_once(
        ROOT / "CHANGELOG.md",
        "### Fixed\n\n",
        "### Fixed\n\n"
        "- Read trusted pickle and Bayesian-PhysTwin NumPy archives from one "
        "descriptor-bound, symlink-free snapshot; reject duplicate, unsafe, "
        "oversized, object-dtype, or digest-mismatched NPZ inputs before use.\n"
        "- Reject lossy Boolean, string, and floating-point coercion at sparse "
        "trajectory, observed-node, and Bayesian-PhysTwin grid-index boundaries.\n"
        "- Name the registered dense factual update explicitly as "
        "`update_from_observations_legacy_v1` while retaining the historical "
        "method as an exact compatibility alias.\n",
    )


def main() -> None:
    _update_rollout_bank()
    _update_prefix_likelihood()
    _update_phystwin_backend()
    _update_intervention_abduction()
    _update_changelog()
    for path in TRANSPORT_FILES:
        path.unlink()


if __name__ == "__main__":
    main()
