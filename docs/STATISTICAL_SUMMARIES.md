# Statistical summaries and denominators

Pareto: 84 configurations x 5 planned seeds = 420 physical runs.
Equal iterations and equal samples are two views, not 840 independent runs.

Endpoint summaries first select status=completed AND finite=true. Medians, Q1/Q3
with linear interpolation at index (n-1)*p, means, sample standard deviations and
IQR use only that set. Missing/nonfinite endpoints are not replaced by earlier
states, zero or infinity. The aggregator then asserts every included metric is
finite; it does not silently drop an unexpected NaN from a completed record.
An empty selection gives None/blank statistics; standard deviation is undefined
for a single observation.

For BOTH comparison modes:
- Every SGD cell: n_seeds_total=5, n_completed=3, n_failed=2,
  n_physical_failed=2, completion_rate=3/5.
- Every MoM/clipped cell: 5,5,0,0,1 respectively.
- final_parameter_error, final_loss, maximum_parameter_error and wall_clock_seconds
  each use n=3 for SGD and n=5 for MoM/clipped.
- Even maximum_parameter_error is conditional on endpoint completion. Finite
  excursions from a subsequently failed run remain in its individual record but
  do not enter this completed-run aggregate.
- clipping_fraction uses n=5 for clipped SGD; n=0/not applicable for SGD/MoM.
- All 168 exact configuration/mode denominators are in
  results/pareto/audit/summary_denominators.csv.

A failure after an earlier finite comparison endpoint would not invalidate that
endpoint. Failure before/at the endpoint makes it unavailable. In these data all
24 failures are unavailable in both modes. Their endpoint error/loss are NaN;
endpoint timing is None, while observed elapsed time and failed-attempt sample
counts are retained separately.

Physical completion percentages use all planned runs: SGD 36/60, MoM 60/60,
clipped SGD 300/300; overall 396/420. These are not the same denominators as
configuration-specific completed-endpoint statistics.

Trajectory curves include only available rows with finite=true and a finite,
strictly positive metric at that iteration. Failed attempts and later missing
rows are excluded. SGD curve denominators go 5 -> 4 -> 3 at the recorded failures;
other curves use 5. The 216 exact metric/iteration intervals are in
results/pareto/audit/trajectory_denominators.csv.
An apparent decrease of the median after a seed fails does not show its recovery.

CNN: each point/final summary requires all three finite seeds 42,123,2026.
Missing or nonfinite seed observations are rejected rather than dropped.
NumPy linear quantiles define median/Q1/Q3; IQR=Q3-Q1. All 24 seed/method runs
completed. Bands describe seed variability, not confidence intervals.
Diagnostics average within each 500-update seed/window before three-seed
aggregation. Nonfinite/undefined diagnostic values are excluded within that
window; defined_counts reports the number used and empty windows are rejected.
Updates within a seed are not independent seed replicates.
