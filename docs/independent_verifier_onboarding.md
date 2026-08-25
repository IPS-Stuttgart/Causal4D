# Independent verifier onboarding

## Purpose and current boundary

The registered Causal4D same-object physical protocol requires a genuinely
independent person to attest the final method freeze. The current truthful
operator registry contains one person only, so the authoritative next action is
`stop_independent_verifier_unavailable`. Object registration, source-panel
acquisition, gate approval, freeze attestation, and confirmatory execution remain
forbidden until the governance boundary is resolved.

This guide narrows the verifier role so that recruiting a verifier does not
require transferring model-development, robot-operation, or data-analysis work.
It does not register a person, infer an identity, relax independence, or
authorize physical acquisition.

## Recruitment materials

Two copy-ready aids reduce the administrative burden without changing the
protocol:

- [`independent_verifier_invitation_template.md`](independent_verifier_invitation_template.md)
  is a private invitation that explains the bounded role and current `0/36`
  evidence state; and
- [`independent_verifier_self_declaration_template.md`](independent_verifier_self_declaration_template.md)
  is a private, candidate-completed consent and independence declaration.

Only the blank templates belong in Git. A completed invitation, declaration,
signature, stable principal, raw email address, personnel number, or HMAC secret
must remain outside the repository and acquisition dataset. Receiving a private
declaration does not itself register the candidate, authorize physical work, or
constitute the final attestation.

## Bounded verifier role

The independent verifier is responsible only for checking and attesting the
frozen scientific and software boundary. The verifier does **not** need to:

- implement or tune Causal4D, BayesianPhysTwin, or Prob4D;
- select estimators, thresholds, exclusions, contact regions, or command
  profiles;
- operate the robot, cameras, or physical object;
- inspect target outcomes;
- perform the confirmatory analysis; or
- become a general project contributor.

The verifier's claim-bearing responsibilities are limited to:

1. provide their own identity and independence declaration through the
   registered operator-registry process;
2. verify the exact protocol, acquisition schedule, analysis manifest, source
   and package identities, and deployed checkout;
3. run or independently witness the existing fail-closed freeze-attestation
   command;
4. verify that the attested person differs from the freezer at the person level;
5. preserve the resulting immutable attestation; and
6. refuse attestation when any required identity, artifact, or chronology check
   fails.

## Eligibility

A suitable verifier must be a real, distinct person who can truthfully state
that they did not select or tune the frozen Causal4D method using the
confirmatory target outcomes. A colleague outside the method-development effort,
an institutional research-data or reproducibility officer, or an external
laboratory member may be suitable when the declaration is accurate.

The following are not independent verification:

- a second username, email address, alias, or operator ID for the same person;
- an identity invented or inferred from Git metadata, issue history, workflow
  authorship, or institutional affiliation;
- a service account, bot, language model, or unattended workflow;
- a person whose name was entered without their knowledge and consent; or
- an attestation performed after target outcomes were used to modify the method.

Causal4D validates declarations and artifact bindings. It must never invent the
participant roster.

## Minimal onboarding inputs

The principal investigator and verifier should prepare only the information
required by the versioned operator-registry contract:

- one project-local operator ID for the verifier;
- the verifier's self-supplied principal identity material;
- the `independent_verifier` role and no incompatible claim-bearing role;
- the verifier's explicit acknowledgement of the protocol and target-access
  boundary; and
- the registry seal produced by the existing CLI.

Private principal and HMAC material stays outside published artifacts. Public
records contain only the protocol-bound operator IDs, roles, person-level
identity commitments, artifact identities, and the resulting availability
status.

## Onboarding and attestation sequence

All commands must run from the exact deployed checkout and dataset selected for
the physical study. Placeholder identities below must be replaced only after the
real participants themselves have supplied and reviewed the required private
material.

1. Recompute the current decision and preserve the hash-verified result:

   ```bash
   causal4d protocol readiness next-action \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1
   ```

   Until a truthful sealed registry validates, the action must remain
   `stop_independent_verifier_unavailable` and must not authorize physical work.

2. Send the bounded invitation privately. The candidate reviews this guide and
   the blank self-declaration, then replies and completes the declaration in
   their own words. Do not enter a candidate into the registry before their
   informed consent.

3. Scaffold the protocol-bound registry template through the registered CLI:

   ```bash
   causal4d protocol readiness scaffold-operator-registry \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1
   ```

4. The authorized identity custodian derives the person-level commitment using
   the registered institution-held, domain-separated HMAC procedure. Edit only
   the `operators` array in:

   ```text
   /data/causal4d-sloth-multi-action-v1/preacquisition/operator_registry.template.json
   ```

   The real verifier entry must use the consented project-local operator ID,
   role `independent_verifier`, and the derived person-level commitment. Do not
   copy raw identity or the completed declaration into the template.

5. Seal the roster exactly once with the registered freezer identity:

   ```bash
   causal4d protocol readiness seal-operator-registry \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1 \
     /data/causal4d-sloth-multi-action-v1/preacquisition/operator_registry.template.json \
     --sealed-by "<registered-freezer-id>"
   ```

6. Recompute the next action. The governance blocker must disappear before any
   object registration, source-panel acquisition, gate approval, freeze
   attestation, or confirmatory physical operation is attempted:

   ```bash
   causal4d protocol readiness next-action \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1
   ```

7. After all source-only readiness gates pass, seal the exact method freeze:

   ```bash
   causal4d protocol freeze seal \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1/method_freeze.json \
     --frozen-by "<registered-freezer-id>"
   ```

8. The distinct verifier independently validates and attests that freeze:

   ```bash
   causal4d protocol freeze attest \
     /data/causal4d-sloth-multi-action-v1/method_freeze.json \
     configs/causal4d/sloth_multi_action_v1.json \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
     --verified-by "<registered-independent-verifier-id>"
   ```

9. Obtain the final hash-verified readiness decision. Confirmatory execution 1
   remains forbidden unless both `ready=true` and
   `first_confirmatory_execution_allowed=true` are present.

A verifier may rerun read-only validation and inspect the reviewer-facing
reproduction bundle. They must not approve a changed commit, wheel, protocol,
analysis manifest, or evidence tree under an earlier attestation.

## Alternative when no verifier is available

The current independently attested protocol cannot be satisfied by one person.
When no real verifier can be recruited, the only valid alternative is a
separately reviewed and versioned protocol amendment created **before target
access and before physical acquisition**. Such an amendment must:

- state explicitly that pre-acquisition independent attestation is unavailable;
- remove every claim that depends on that attestation;
- preserve the estimator, action/contact design, split, thresholds, exclusions,
  target-access boundary, and reporting obligations unless separately justified
  before target access;
- retain cryptographic freezing, immutable evidence publication, and complete
  negative-result reporting; and
- use a new protocol and amendment identity rather than rewriting the existing
  v4 record.

Later independent review of a completed bundle is useful but is not equivalent
to pre-acquisition independent attestation. Creating multiple identities for one
person is never an acceptable substitute.

## Relationship to the primary experiment

The primary acquisition candidate already declares Prob4D unused and semantic
reweighting excluded. Recruiting a verifier or versioning the governance
amendment must not reopen those method decisions or introduce a new estimator.
The decisive scientific milestone remains the registered 18-session,
36-execution physical study and either its positive result or its complete
negative or bounded result.

See also:

- [`operator_identity_registry.md`](operator_identity_registry.md)
- [`single_operator_registry_correction.md`](single_operator_registry_correction.md)
- [`causal4d_real_experiment_milestone.md`](causal4d_real_experiment_milestone.md)
- [`causal4d_paper_scope.md`](causal4d_paper_scope.md)
- [governance blocker issue #377](https://github.com/IPS-Stuttgart/Causal4D/issues/377)
- [primary milestone issue #25](https://github.com/IPS-Stuttgart/Causal4D/issues/25)
