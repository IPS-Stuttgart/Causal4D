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
the physical study. Placeholder identities below must be replaced only by the
real participants themselves.

1. Recompute the current decision and preserve the hash-verified result:

   ```bash
   causal4d protocol readiness next-action \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1
   ```

2. Scaffold or update the operator-registry template through the registered
   readiness route. Do not edit the sealed registry in place.

3. Have the verifier supply and review their own identity and independence
   declaration. Seal the corrected registry with the real principal
   investigator identity.

4. Recompute the next action. The governance blocker must disappear before any
   physical or claim-bearing operation is attempted.

5. After all source-only readiness gates pass, seal the exact method freeze:

   ```bash
   causal4d protocol freeze seal \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1/method_freeze.json \
     --frozen-by "<registered-freezer-id>"
   ```

6. The distinct verifier independently validates and attests that freeze:

   ```bash
   causal4d protocol freeze attest \
     /data/causal4d-sloth-multi-action-v1/method_freeze.json \
     configs/causal4d/sloth_multi_action_v1.json \
     /opt/causal4d-frozen \
     /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
     --verified-by "<registered-independent-verifier-id>"
   ```

7. Obtain the final hash-verified readiness decision. Confirmatory execution 1
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

- [`single_operator_registry_correction.md`](single_operator_registry_correction.md)
- [`causal4d_real_experiment_milestone.md`](causal4d_real_experiment_milestone.md)
- [`causal4d_paper_scope.md`](causal4d_paper_scope.md)
- [primary milestone issue #25](https://github.com/IPS-Stuttgart/Causal4D/issues/25)
