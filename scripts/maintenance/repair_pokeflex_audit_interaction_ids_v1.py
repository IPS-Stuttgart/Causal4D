#!/usr/bin/env python3
"""Repair the PokeFlex audit to use action-qualified interaction identities."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts/remote/audit_pokeflex_probe_challenge_folds_gpuserver4090.py"
VERIFY = ROOT / "scripts/ci/verify_pokeflex_probe_challenge_fold_audit.py"
TEST = ROOT / "tests/test_pokeflex_probe_challenge_fold_audit.py"
DOC = ROOT / "docs/pokeflex_probe_challenge_fold_audit_v1.md"
REQUEST = ROOT / ".github/requests/pokeflex-probe-challenge-fold-audit-gpuserver4090-v1.json"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    replace_once(
        AUDIT,
        "    take_id: str\n    action_class: str\n",
        "    raw_take_id: str\n    take_id: str\n    action_class: str\n",
        "ArchiveRecord identity fields",
    )
    replace_once(
        AUDIT,
        '            "take_id": self.take_id,\n            "action_class": self.action_class,\n',
        '            "raw_take_id": self.raw_take_id,\n'
        '            "take_id": self.take_id,\n'
        '            "action_class": self.action_class,\n',
        "archive JSON identity fields",
    )
    replace_once(
        AUDIT,
        "    object_id, take_id, action_class, take_index = parse_archive_identity(path, root)\n",
        "    object_id, raw_take_id, action_class, take_index = parse_archive_identity(\n"
        "        path, root\n"
        "    )\n",
        "parsed raw take identity",
    )
    replace_once(
        AUDIT,
        "        object_id=object_id,\n        take_id=take_id,\n        action_class=action_class,\n",
        "        object_id=object_id,\n"
        "        raw_take_id=raw_take_id,\n"
        "        take_id=f\"{action_class}:{raw_take_id}\",\n"
        "        action_class=action_class,\n",
        "action-qualified take identity",
    )

    replace_once(
        VERIFY,
        '                    "take_id",\n                    "object_id",\n',
        '                    "raw_take_id",\n'
        '                    "take_id",\n'
        '                    "object_id",\n',
        "verifier required raw identity",
    )
    replace_once(
        VERIFY,
        "        take_key = f\"{record['object_id']}::{record['take_id']}\"\n"
        "        require(take_key not in archive_take_ids, f\"duplicate object/take identity: {take_key}\")\n"
        "        archive_take_ids.add(take_key)\n",
        "        take_id = record[\"take_id\"]\n"
        "        expected_prefix = f\"{record['action_class']}:\"\n"
        "        require(\n"
        "            isinstance(take_id, str) and take_id.startswith(expected_prefix),\n"
        "            f\"take identity is not action-qualified: {take_id}\",\n"
        "        )\n"
        "        require(\n"
        "            take_id not in archive_take_ids,\n"
        "            f\"duplicate action-qualified take identity: {take_id}\",\n"
        "        )\n"
        "        archive_take_ids.add(take_id)\n",
        "verifier action-qualified uniqueness",
    )

    replace_once(
        TEST,
        '    assert audit["summary"]["frozen_fold_count"] == 4\n',
        '    assert audit["summary"]["frozen_fold_count"] == 4\n'
        '    assert all(\n'
        '        record["take_id"].startswith(f"{record[\'action_class\']}:")\n'
        '        for record in audit["archives"]\n'
        '    )\n'
        '    assert all(\n'
        '        interaction.startswith("poking:")\n'
        '        for fold in audit["frozen_folds"]\n'
        '        for interaction in fold["candidate_probe_take_ids"]\n'
        '    )\n',
        "synthetic canonical identity assertions",
    )

    replace_once(
        DOC,
        "For each parsed object identity, complete poking takes are ordered by a\n"
        "content-independent SHA-256 ordering using the registered salt. One poke is\n",
        "Poking and dropping folders reuse raw stems such as `Object_T1`. The audit\n"
        "therefore assigns the canonical action-qualified identities `poking:Object_T1`\n"
        "and `dropping:Object_T1`; raw stems remain metadata only. For each parsed\n"
        "object identity, complete poking interactions are ordered by a\n"
        "content-independent SHA-256 ordering using the registered salt. One poke is\n",
        "document action-qualified identity",
    )

    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["technical_revision"] = 3
    request["retry_reason"] = (
        "qualify duplicate raw T1/T2/T3 stems by action class after a successful "
        "metadata-only scan; no archive member payload was opened"
    )
    request["prior_run"] = {
        "workflow_run_id": 33589834614,
        "audit_id": "73d63e1d1980c723346b55288ef75a3535bde814d5d2603bcc6d4c0a21cb68fc",
        "dataset_gate_passed": True,
        "failure_stage": "post-audit verifier only",
        "archive_member_payload_opened": False,
    }
    REQUEST.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PokeFlex audit interaction identities repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
