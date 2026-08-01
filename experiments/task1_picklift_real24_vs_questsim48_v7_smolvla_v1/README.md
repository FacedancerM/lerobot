# Task 1 PickLift SmolVLA matched pair

This experiment owns the offline-only matched SmolVLA comparison frozen by
research-control:

- **A:** frozen Real24 only.
- **B:** frozen Combined72 (Real24 + Quest-Sim48-v7), with exactly 32 Real
  and 32 Sim samples in every batch consumed by an optimizer step.

Phase A froze and verified the datasets, official SmolVLA/SmolVLM revisions,
dependency lock, front-only feature contract, batch size, sampler, and run
order without taking an optimizer step.

Phase B then completed the predeclared order without retries or resume:

1. A smoke: 500 steps.
2. B smoke: 500 steps.
3. A full: 20,000 steps, initialized independently from the exact base.
4. B full: 20,000 steps, initialized independently from the exact base.

Only the two step-20,000 checkpoints are selected. Their paths and SHA-256
identities are recorded in `phase_b_result_index.json`. The immutable evidence
root is:

`/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_vs_questsim48_v7_smolvla_v1/phase_b_v1/matched_pair_result_v1`

`finalize_phase_b_results.py` verifies each run and creates the matched-pair
summary and primary inventory. `verify_phase_b_results.py` independently
recomputes the listed hashes, checkpoint inventories, sample counts, and
offline finite-action checks. Both scripts refuse to overwrite existing
frozen results.

This phase contains no hardware, Quest, Remote, MuJoCo, LocalSim, or
real-robot rollout. Training losses are engineering diagnostics only. The two
checkpoints have no performance meaning until a separately frozen matched
evaluation is completed; this directory does not contain a paper result.
