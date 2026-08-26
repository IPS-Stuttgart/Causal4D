# Single-operator v5 bootstrap evidence - 2026-08-26

This directory preserves the sanitized result of the owner-only v5 identity
bootstrap. It is operational evidence, not physical evidence, and it does not
advance the source-panel or confirmatory execution counts.

## Execution identity

- Workflow run: `32925318021` (success)
- Reviewed main commit: `c4abf3ebe599ba6d2c009d242cab493d95da88af`
- GitHub artifact: `causal4d-single-operator-v5-owner-bootstrap-v2`
- GitHub artifact ID: `9597158446`
- GitHub archive digest: `sha256:15cd19a517f75582984c41a5d0e8b64397ff3dd6325d9aa15c382cbbe2d482ab`
- Report SHA-256: `8cd6f2d66d954b49f2f021ea936b3485e70420defa0816b60898f22098092c1b`
- Built wheel SHA-256: `0833d7e27efae963c28c186e697595ea20811b703f7756611db3b14c50ed9f6a`

The committed `report.json` is byte-identical to the workflow's `report.txt`.
Its canonical digest, computed with `report_sha256` omitted, recomputes to the
recorded value.

## Result

The run created a fresh private owner HMAC identity because the historical
private identity material was unavailable. It intentionally claims neither
historical identity-digest continuity nor independent attestation.

`dataset_modified = true` records creation of the target dataset scaffold and
registry binding. It does not mean that an experimental observation or outcome
was collected. The report also records:

- `physical_evidence_increment = 0`
- `target_outcomes_used = false`
- `device_nodes_opened = false`
- `physical_command_sent = false`
- `registered_method_changed = false`

The registered next action is `complete_object_registration`. The contact
registration must still pass anatomical review before the slip/reset pilot is
eligible to run.

## Current registration boundary

The first generated contact proposal was rejected during anatomical review:
its red region was a backpaw, and its yellow torso region was too low. This
bootstrap result does not approve that proposal and cannot be used to bypass
the schema-4 object/contact registration gate.
