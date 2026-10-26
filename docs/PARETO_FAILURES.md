# Pareto numerical failures

The alpha=2.1 experiment has 420 terminal records: 396 completed, 24 numerical
failures. Every failure is SGD, seeds 1 and 4, across four schedules and three
reference K settings. Reference K defines the matched baseline batch/budget:
SGD itself has no MoM block count and no clipping threshold.

Every exact failure reason is "nonfinite per-observation loss".
Terminal predictions/residuals were not flagged nonfinite; loss_nonfinite is true.
The check aborts before evaluating the failing minibatch's gradient or update.
The false gradient flag therefore does not establish that this gradient was
computed. The event is consistent with float32 squared-loss overflow on finite
predictions/residuals, not proof of mathematical divergence.

| Regime | Method | Reference K | Clipping threshold | Failures / planned | Failed seed IDs | Failure iterations, seed order | Exact reason |
|---|---|---:|---|---:|---|---|---|
| R1-LOG | sgd | 11 | N/A | 2/5 | 1;4 | 1156;239 | nonfinite per-observation loss |
| R1-LOG | sgd | 31 | N/A | 2/5 | 1;4 | 624;84 | nonfinite per-observation loss |
| R1-LOG | sgd | 51 | N/A | 2/5 | 1;4 | 380;53 | nonfinite per-observation loss |
| R2-LOG | sgd | 11 | N/A | 2/5 | 1;4 | 1224;247 | nonfinite per-observation loss |
| R2-LOG | sgd | 31 | N/A | 2/5 | 1;4 | 405;85 | nonfinite per-observation loss |
| R2-LOG | sgd | 51 | N/A | 2/5 | 1;4 | 247;56 | nonfinite per-observation loss |
| R3-CONST-LOG | sgd | 11 | N/A | 2/5 | 1;4 | 2048;241 | nonfinite per-observation loss |
| R3-CONST-LOG | sgd | 31 | N/A | 2/5 | 1;4 | 413;612 | nonfinite per-observation loss |
| R3-CONST-LOG | sgd | 51 | N/A | 2/5 | 1;4 | 252;57 | nonfinite per-observation loss |
| R3-CONST-RM | sgd | 11 | N/A | 2/5 | 1;4 | 2048;241 | nonfinite per-observation loss |
| R3-CONST-RM | sgd | 31 | N/A | 2/5 | 1;4 | 413;612 | nonfinite per-observation loss |
| R3-CONST-RM | sgd | 51 | N/A | 2/5 | 1;4 | 252;57 | nonfinite per-observation loss |

Largest finite transient parameter error among completed runs: 136370935.70890108.
It remains a finite value, not a failure. Last finite errors preceding failed
attempts range from 1.19244872663939e+26 to 9.419496639254835e+28; those values are preserved.
There is no arbitrary magnitude threshold for failure classification.

All 24 full failure records and both comparison views remain inside
results/pareto/plot_inputs.json.gz. The audit CSVs identify their run IDs, failure
iterations/flags, original trajectory SHA256 and last finite excursions.
results/pareto/failed_terminal_rows.csv contains the original terminal CSV rows.
No failed trajectory was retrained, rescued, replaced or converted into a success.

Regenerate the four audit reports in a new directory:

    python -B scripts/audit_pareto_failures.py --output ../pareto_failure_audit

This uses only preserved numerical records and stdlib; it does not import torch,
generate datasets, compute gradients or invoke training.
