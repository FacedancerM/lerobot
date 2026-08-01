# Task 1 PickLift SmolVLA matched-pair Phase A

This directory freezes the offline-only candidate contract for a matched
SmolVLA comparison:

- **A:** frozen Real24 only.
- **B:** the frozen Combined72 dataset (Real24 + Quest-Sim48-v7) with an
  exact 32 Real / 32 Sim quota in every batch.

Phase A does not train a model. It freezes the two data identities, the exact
official SmolVLA and transitive SmolVLM revisions, the local dependency lock,
the front-only feature contract, a common batch size, and the predeclared run
order. `phase_a_preflight_result.json` records the completed loader and CUDA
load/backward probes. `verify_phase_a.py` rechecks the frozen local inputs.

Training remains blocked until research-control accepts
`candidate_plan.json` and explicitly authorizes Phase B. The smoke runs and
the two full runs must use new output directories and must never resume from
one another:

1. Real24 smoke, 500 steps.
2. Combined72 smoke, 500 steps.
3. Real24 full, 20,000 steps from the exact base.
4. Combined72 full, 20,000 steps from the exact base.

Only step 20,000 is the selected candidate checkpoint. There is no hardware,
Quest, Remote service, MuJoCo rollout, evaluation, checkpoint selection, or
push in this phase.

