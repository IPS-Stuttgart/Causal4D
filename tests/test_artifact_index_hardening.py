from __future__ import annotations

import hashlib
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from causal4d.artifact_io import ArtifactFileSnapshot, ArtifactValidationError
from causal4d.numpy_archive import load_numpy_archive
from causal4d.prefix_likelihood import prefix_component_log_likelihood
from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence
from causal4d.trusted_pickle import load_trusted_pickle


def _bank() -> JointRolloutBank:
    return JointRolloutBank(
        hypothesis_ids=("h0",),
        hypothesis_metadata=({},),
        hypothesis_prior_weights=np.asarray([1.0]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=np.zeros((1, 1, 4, 2, 3), dtype=np.float32),
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_payload(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def test_trusted_pickle_unpickles_the_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "value.pkl"
    path.write_bytes(pickle.dumps({"value": "path"}))
    payload = pickle.dumps({"value": "snapshot"})
    snapshot = ArtifactFileSnapshot(
        path=path,
        payload=payload,
        sha256=_sha256(payload),
        byte_count=len(payload),
    )

    monkeypatch.setattr(
        "causal4d.trusted_pickle.read_regular_file_beneath",
        lambda *_args, **_kwargs: snapshot,
    )

    assert load_trusted_pickle(
        path,
        allow_unsafe_pickle=True,
        expected_sha256=snapshot.sha256,
    ) == {"value": "snapshot"}


def test_numpy_archive_loads_one_content_verified_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "profile.npz"
    np.savez(path, values=np.asarray([1.0, 2.0]))
    payload = path.read_bytes()

    loaded = load_numpy_archive(path, expected_sha256=_sha256(payload))

    np.testing.assert_array_equal(loaded.arrays["values"], [1.0, 2.0])
    assert loaded.snapshot.payload == payload
    assert not loaded.arrays["values"].flags.writeable
    with pytest.raises(ValueError):
        loaded.arrays["values"].setflags(write=True)


def test_numpy_archive_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "profile.npz"
    np.savez(path, values=np.asarray([1.0]))

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_numpy_archive(path, expected_sha256="0" * 64)


def test_numpy_archive_rejects_duplicate_members(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    member = _npy_payload(np.asarray([1.0]))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, mode="w") as archive:
            archive.writestr("values.npy", member)
            archive.writestr("values.npy", member)

    with pytest.raises(ArtifactValidationError, match="duplicate ZIP members"):
        load_numpy_archive(path)


def test_numpy_archive_rejects_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    np.savez(path, values=np.asarray([{"unsafe": True}], dtype=object))

    with pytest.raises(ArtifactValidationError, match="without pickle support"):
        load_numpy_archive(path)


def test_numpy_archive_rejects_symlinked_parent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    profile = source / "profile.npz"
    np.savez(profile, values=np.asarray([1.0]))
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ArtifactValidationError):
        load_numpy_archive(linked / "profile.npz")


@pytest.mark.parametrize("indices", [[0.0], [True], ["0"]])
def test_sparse_evidence_rejects_lossy_node_indices(indices: list[object]) -> None:
    with pytest.raises(ValueError, match="node_indices must contain integers"):
        SparseTrajectoryEvidence(
            positions_m=np.zeros((2, 1, 3)),
            node_indices=np.asarray(indices),
            rollout_frame_indices=np.asarray([0.0, 1.0]),
            compare_displacements=False,
        )


@pytest.mark.parametrize("indices", [[0.0], [True], ["0"], [0, 0]])
def test_legacy_update_rejects_invalid_or_duplicate_nodes(
    indices: list[object],
) -> None:
    bank = _bank()
    observations = np.zeros((4, 2, 3))
    with pytest.raises(ValueError, match="observed_nodes"):
        bank.update_from_observations_legacy_v1(
            observations,
            prefix_frame_count=3,
            scale_m=0.01,
            observed_nodes=indices,
        )


@pytest.mark.parametrize("indices", [[0.0], [True], ["0"], [0, 0]])
def test_normalized_update_rejects_invalid_or_duplicate_nodes(
    indices: list[object],
) -> None:
    bank = _bank()
    observations = np.zeros((4, 2, 3))
    with pytest.raises(ValueError, match="observed_nodes"):
        prefix_component_log_likelihood(
            bank,
            observations,
            prefix_frame_count=3,
            observed_nodes=indices,
        )


def test_legacy_alias_preserves_registered_numerics() -> None:
    bank = _bank()
    observations = np.zeros((4, 2, 3))
    explicit = bank.update_from_observations_legacy_v1(
        observations,
        prefix_frame_count=3,
        scale_m=0.01,
    )
    compatibility = bank.update_from_observations(
        observations,
        prefix_frame_count=3,
        scale_m=0.01,
    )
    np.testing.assert_array_equal(compatibility, explicit)


def test_bayesian_phystwin_grid_indices_require_exact_integers() -> None:
    pytest.importorskip("bayesian_phystwin")
    from causal4d.phystwin_backend import BayesianPhysTwinParticles

    with pytest.raises(ValueError, match="grid_indices must contain integers"):
        BayesianPhysTwinParticles(
            log_scales=np.asarray([[0.0, 0.0]]),
            weights=np.asarray([1.0]),
            grid_indices=np.asarray([[0.0, 0.0]]),
            source_weight_key="posterior_weights",
            retained_probability_mass=1.0,
        )


def test_bayesian_phystwin_profile_accepts_a_bound_digest(tmp_path: Path) -> None:
    pytest.importorskip("bayesian_phystwin")
    from causal4d.phystwin_backend import load_bayesian_phystwin_particles

    profile = tmp_path / "profile.npz"
    np.savez(
        profile,
        object_log_scales=np.asarray([-0.2, 0.2]),
        controller_log_scales=np.asarray([-0.1, 0.1]),
        posterior_weights=np.asarray([[0.1, 0.2], [0.6, 0.1]]),
    )
    digest = _sha256(profile.read_bytes())

    particles = load_bayesian_phystwin_particles(
        profile,
        maximum_count=2,
        expected_sha256=digest,
    )
    assert len(particles.weights) == 2
