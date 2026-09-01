from pathlib import Path

from causal4d_public.cli.pokeflex_realized_load_source import (
    _official_public_archive_root,
)


def test_pokeflex_realized_load_uses_the_frozen_poking_archive_root() -> None:
    mounted = Path("/mnt/lexar4tb/pokeflex")
    assert _official_public_archive_root(mounted) == mounted / "poking"
