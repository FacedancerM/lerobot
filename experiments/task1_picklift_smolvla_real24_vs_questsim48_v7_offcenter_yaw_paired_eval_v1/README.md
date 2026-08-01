# Task1 SmolVLA matched real Eval-v2

Fresh paired real evaluation of the two frozen step-20000 SmolVLA checkpoints.
It copies the exact frozen 12-pose Eval-v2 schedule, success rule, Real camera,
ready pose, 3-second/20 Hz official-send ready-return, 30-second full policy
window, operator annotation, and canonical-video review contract.

Each pose receives one A and one B trial. First model alternates B/A then A/B,
so each model is first at six poses. Order, poses, checkpoints, prompt and
scoring cannot adapt to outcomes. The evidence root is fresh and no ACT trial
evidence is reused.

A: `916f0111caf004cafd37f8fd9db9b6e5aa35c1d125cf59ea5cd63b8f19897c83`

B: `1a1dbc6d553c9501a76d4ed4b9cc3bd7b84724216086432868d7ab6b2fa5c185`

Software preparation and dynamic checkpoint adapter smoke must finish with all
hardware-access flags false. Hardware execution is one trial at a time and is
blocked until explicit onsite GO.
