#!/usr/bin/env python3
"""Compatibility entry point for legacy numeric aligned-episode directories."""

from __future__ import annotations

import re

import audit_deform360_fragments as audit

# Current official outputs use episode_0000. Some retained legacy processing
# trees use bare numeric names or add a suffix. This remains restricted to
# direct object children and only changes structural episode discovery.
audit.EPISODE_RE = re.compile(
    r"^(?:episode[_-]?)?(\d+)(?:[^0-9].*)?$",
    re.IGNORECASE,
)


if __name__ == "__main__":
    raise SystemExit(audit.main())
