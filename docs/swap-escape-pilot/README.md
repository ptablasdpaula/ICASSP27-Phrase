# Escape from a swapped four-event fit

2026-09-16 follow-up to the amplitude and ordered-time pilots. Same target
C04-T0005, same fixed-amplitude failed fit, same 2.048 padding. All amplitudes
remain 0.8 and times use independent bounded logits. No production/paper change.

Seven loss restarts test upward frequency accumulation (bottom to top) in
forward time (↗, cel_01), backward time (↖, cel_04), both time directions
(↗↖, cel_05), and the latter with Log-Weighing (cel_05_lw). Comparators are
downward frequency in both time directions (↘↙, cel_10), all four with
Log-Weighing (cel_15_lw), and forward time in both frequency directions with
Log-Weighing (↗↘, cel_03_lw). This is a selected subset, not a full ranking.

All restart at identical failed physical controls with fresh Adam moments,
the registered initial LR 0.05 and rollback/stop schedule, at most 3000 updates.
Each objective conditions by its own initial loss. Comparisons use matched
pitch/onset errors and the original canonical four-direction CeL evaluated
at each run's strict-best state; different objectives' raw values are not
compared directly. Retain all evaluated trajectories and all tried variants.

An additional intervention holds onsets fixed, scores all six pairwise swaps
of the four current pitches plus the unchanged candidate using canonical CeL,
and chooses the minimum. Target coordinates are not used for proposal selection;
only target audio through the same objective. Resume canonical CeL Adam from
that candidate. This is an extra discrete search step, not pure gradient descent,
and uses six extra proposal evaluations beyond the unchanged candidate.

Shared experimental fitter was checked against production and used by both
preceding pilots. Direction implementations were validated in the completed
[direction screen](../cel-gradient-screen/README.md). The runner asserts fixed
amplitudes at every evaluation. Run `python scripts/test_swap_escape.py` with
the project environment and PYTHONHOME set to `.pixi/envs/default`.

Results pending. A success here is not evidence of general success across targets.

## Adaptive follow-up: relax before judging a swap

The greedy proposal screen selected the unchanged state: all six swaps initially
increase CeL. Therefore also optimise **all six** swapped candidates under canonical
CeL before selecting by their final strict-best losses (including the unchanged
baseline in selection). Same full registered schedule, fixed amplitudes and
independent times. This is a six-restart search with more compute, explicitly
not an equal-budget loss comparison. Enumeration is lexicographic; no target
coordinates select a proposal. Each trajectory is stored under `relaxed-swaps/`.
Run `python scripts/test_relaxed_swaps.py` in the same environment.
