# Task1 SmolVLA matched-pair Phase B freeze

Date: 2026-08-01

## Confirmed engineering facts

The Ubuntu Phase A owner worktree is clean at commit
`f2dca70a648294df5ffc8016ff732d863a1d4f0c`. Its tracked candidate plan,
preflight result, and verification result match their committed bytes. An
independent rerun verified the Real24 and Combined72 identities, official
Dataset loading, the exact base and transitive VLM inventories, dependency
locks, deterministic 32 Real / 32 Sim batches, and the batch-64 CUDA
forward/backward probe. Phase A executed zero optimizer steps and accessed no
hardware.

The official base contains three placeholder camera inputs and names the VLM
and tokenizer through a mutable repository reference. Therefore Phase B must,
before its first optimizer step, replace the image feature contract with the
single frozen `observation.images.front` input and bind both VLM and tokenizer
to the same frozen local snapshot. These are reproducibility and input-contract
gates, not model changes.

## Frozen decision

Authorize exactly this offline matched pair:

- A: frozen Real24-only;
- B: frozen Combined72 (the same Real24 plus Quest-Sim48-v7), with every batch
  containing exactly 32 Real and 32 Sim samples.

Both use the same SmolVLA base and VLM snapshots, exact training task text,
front-only image plus six-dimensional state, six-dimensional action, 20 Hz,
chunk/action horizon 50, seed 1000, batch 64, official base optimizer and
scheduler, and independent initialization from the same base. Execution order
is A smoke 500, B smoke 500, A full 20k, B full 20k. Only step 20k may be
selected; observed loss or intermediate checkpoints cannot change the choice.

The exact offline authorization token is:

`GO_TASK1_SMOLVLA_MATCHED_PAIR_PHASE_B_V1`

It authorizes no hardware, rollout, Dataset mutation, Local pilot use,
training extension, checkpoint switching, or push.

## Research boundary

Passing smoke, completing training, or producing finite offline actions is
engineering evidence only. It does not establish SmolVLA performance, a
Sim-to-Real benefit, or a paper result. Any downstream simulation or real
comparison requires a separately frozen evaluation contract.

## Files

- `experiment-design/task1-picklift-real24-vs-questsim48-v7-smolvla-v1.json`
  SHA-256 `4e92dc821d7e3ea334c18b7278c8e918fced82bf0a833d29d55307faea9cdc7e`.
- `manifests/task1-picklift-smolvla-matched-pair-phase-a-freeze-v1.json`
  SHA-256 `181054c01265f52281e2e2349cd611740939edf0264f1c6b7f4866a11e9d0687`.
