# Execution and validation

Completed all 173 targets / 13,099 candidates on CPU float64 using the compiled
TorchLPC backend, PyTorch 2.7.1+cu126, with onset FFT length 16,384 (2.048 padding).
The 16 elementary terms reconstruct 510 equal-weight subset objectives exactly.
There are zero ambiguous assignments and zero nonfinite variant/candidate gradients.

Validation:

- All 10 targeted tests passed (the original screen tests plus two new tests).
  New checks compare orthogonal sums and Log-Weighing against explicit grid slices,
  mixed-subset gradients against direct differentiation and finite differences,
  and exact-match losses/gradients against zero.
- Every target's first batch independently reproduced all eight archived diagonal
  elementary values and gradients before its new results were accepted.
- All 31,500 original summary rows reproduce their means, sample SDs, medians and
  phrase counts; maximum numerical difference is **0.0**.
- All 173 raw NPZ files share the accepted source signature. The ZIP archive
  passes its integrity check. Per-shard hashes are in `provenance.json`.
- Changed Python files pass Ruff. The selected heatmap and interactive table were
  visually inspected. Chromium checks confirmed 255/510 rows for uniform/all
  weighting, 128 rows requiring ↑ and ↗ across both weightings, and correct
  orthogonal-only filtering, with no page JavaScript errors.

Slurm job 27186120 ran for 17m10s (59m02s total CPU including the replacement
worker). Its batch exit status is failed because the first worker rejected the
initial benchmark shard after a source-format change; that shard was excluded,
and the worker was restarted as step 27186120.0, which completed successfully.
No rejected benchmark data enter the report. Follow-up job 27186150 completed
successfully in 41s, revalidating all 173 checkpoint signatures. Aggregation and
plot generation ran locally. No jobs from this campaign remain active.

The compute source is commit `8c5a467`; provenance additionally records the exact
compute signature and reporting-script hash. Full statistics are compressed as
`summary.csv.gz` to avoid committing a very large uncompressed CSV. The standalone
`explorer.html` needs no server or external assets; open it in a browser. All
subsets also appear in the 60-page `all-subsets.pdf`.

These are local-gradient measurements on the earlier frozen screening design.
No extra gradient-descent runs, amplitude fitting, subset selection on held-out
phrases, or paper changes were made in this experiment.
