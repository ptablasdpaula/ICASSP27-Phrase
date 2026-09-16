# Gradual direction rotation after a plateau

User follow-up, 2026-09-16. Begin at the known four-direction plateau from
C04-T0005; amplitudes stay fixed at 0.8 and onset logits remain independent.

At exploration update s, use L_s = sum_q w_q(s) ell_q with weights summing to 1.
Weights start uniform, then smoothly concentrate on successive corners in
clockwise order ↗ → ↘ → ↙ → ↖. Specifically, angles in implementation order
(↗,↘,↖,↙) are (0, pi/2, 3pi/2, pi), theta=2pi*s/800,
p=softmax(4*cos(theta-angle)), a=0.9*min(s/100,1), and w=(1-a)/4+a*p.
Every direction retains positive weight. Run 1600 updates (two rotations).
This is an exploratory schedule, not a schedule established by prior work.

Use fresh Adam at LR .05, betas (.9,.999), eps 1e-8; divide the objective by the
fixed original canonical loss at the restart. Do not rollback or reduce LR
while the objective changes. Keep a checkpoint only when the **original
uniform four-direction CeL** improves. Then restore that checkpoint and polish
using canonical CeL and the registered Adam/rollback schedule (max 3000 updates).

A matched control uses the identical 1600-update exploration and polish with
constant uniform weights. It separates changed weighting from extra updates or
a different plateau schedule. Neither target coordinates nor matched errors
choose checkpoints. Record weighted and canonical losses separately; record
per-direction and combined raw-coordinate gradient norms every 100 updates.
A larger gradient or a change in weighted loss alone is not escape. Check actual
pitch/onset recovery and the unchanged canonical loss.

[SCRAPL](https://arxiv.org/html/2602.11145v1), Sections 3.1–3.4, samples scattering
paths to estimate a full-loss gradient, with path-wise Adam/SAGA and an
importance-sampling heuristic. That motivates varying loss components, but
this deterministic post-plateau rotation is not an implementation of SCRAPL,
its importance sampler, or its optimiser, and is not claimed to inherit its
guarantees. Here all four terms are evaluated; there is no compute saving.

Run `python scripts/test_rotating_directions.py` with the project environment.
Results pending. This selected case cannot settle the value of adaptive
weighting across targets or other rotation schedules.
