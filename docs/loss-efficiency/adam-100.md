# Actual Adam efficiency benchmark (2026-09-24)

Replaces the earlier render–loss–backward microbenchmark with 100 genuine Adam updates, using the 150 frozen single-note recovery targets in 15 batches of 10. Losses: SS, DDSP MSS (`mss`, not SmoMSS), published composite SOT, sliced TFW2, and four-direction CeL.

Same 16 kHz synthesiser, target data, bounded pitch/onset parameters and initialisation as recovery. Adam uses LR .05, betas (.9, .999), epsilon 1e-8, zero weight decay, and each target's initial-loss normalisation. Lowest-loss parameters are tracked without rollback. Neither the 200-update plateau patience nor the 1000-update early-stop patience can trigger within this horizon; scheduler bookkeeping is omitted. Five disposable warm-up updates precede a fresh 100-update run, with both parameters and optimiser state reset. Target rendering/precomputation, optimiser construction and the diagnostic final evaluation are outside timing. The measured loop includes rendering, loss, backward, Adam updates and best-iterate tracking.

All jobs require NVIDIA A100-PCIE-40GB. Runtime is the mean per batch update over 100 steps; GPU memory is peak allocated memory over those steps. Report the median across 15 batches for each loss, and ratios to SS. Raw records also retain initial/final/best losses, final parameters and the actual Adam step count, making the optimisation verifiable.

Submitted:
- 28141106, tasks 0–1, gpushort: SS, MSS.
- 28141107, tasks 2–4, andrena: SOT, TFW2, CeL.
- Files: `results/loss-efficiency/benchmark-<job>_<task>.json`; partial progress is in matching `.log` files.

Paper table values are still the previous measurements until these jobs finish. Do not relabel the old SmoMSS measurements as MSS.

At the user's request, dependent CPU job 28141715 runs after successful completion of both SS/MSS jobs. It moves tasks 2–4 from Andrena to gpushort only if still pending, confirms cancellation before creating a replacement, and leaves running/completed tasks untouched. Its audit is `results/loss-efficiency/queue-fallback.json`. Replacements retain the A100 40 GB restriction and original array task/loss mapping.
