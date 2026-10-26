# MoM-SGD: SIOPT manuscript research code

This repository contains two completed experiments: raw signed Pareto regression
(dimension 10, alpha=2.1) and a bias-free CIFAR-10 CNN comparing SGD, normalized SGD,
five clipped-SGD settings and MoM-SGD. License and citation approval remain pending.
Figure selection is provisional; all manuscript cross-references are pending.

## Structure

- experiments/, src/: unchanged scientific implementations and frozen analysis.
- results/pareto/plot_inputs.json.gz: 420 full terminal records, both comparison
  summaries and exact finite/parameter-error/loss numerical plotting histories.
- results/pareto/: fingerprints, original terminal failure rows, four derived
  audit CSVs, schedule/budget audits and original aggregate reference tables.
- results/cnn/seed42, seed123, seed2026/: per-method compressed evaluation and
  optimization CSVs, sanitized original configurations and completion metadata.
- results/cnn/reference/: original three-seed summaries.
- results/cnn/runtime/: separate controlled computational-cost audit tables.
- figures/main/, figures/supplementary/: 22 proposed PDF/PNG pairs copied unchanged.
- provenance/: original source identities, numerical input hashes, recovery evidence.
- docs/: numerical failures, exact denominators, figure/file selection and final tree.
- scripts/, tests/: reconstruction, standalone failure audit and lightweight tests.

## Installation

Recorded environment: regular Windows x64 CPython 3.14.6; torch 2.14.0+cpu,
torchvision 0.29.0+cpu, numpy 2.5.3, matplotlib 3.11.1, pytest 9.1.1.
requirements-tested.txt records 27 pinned installed distributions.
requirements.txt lists direct dependencies. Install the recorded CPU builds
explicitly before the base-version lock, which accepts those +cpu versions.
Create an environment outside the repository, without system-site-packages or
path files pointing at another environment. PowerShell, from this directory:

    py -3.14 -m venv ../mom-sgd-release-venv
    $releasePython = (Resolve-Path ../mom-sgd-release-venv/Scripts/python.exe).Path
    & $releasePython -m pip install --no-cache-dir --only-binary=:all: --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu" "torchvision==0.29.0+cpu"
    & $releasePython -m pip install --no-cache-dir --only-binary=:all: -r requirements-tested.txt
    & $releasePython -m pip check
    & $releasePython -B scripts/verify_manifest.py
    & $releasePython -B -m pytest -q --basetemp ../release_test_temp

A truly fresh installation, pip check, all 51 lightweight tests, standalone failure
audit and the combined reconstruction command passed. See VALIDATION.md for evidence
and limits; full training and clean dataset/cache preparation were not run.

## Figures and summaries without training

No downloaded dataset, transformed cache or checkpoint is required:

    python -B scripts/reconstruct.py all --output ../reconstructed_release
    python -B scripts/audit_pareto_failures.py --output ../pareto_failure_audit

Use new output directories; existing destinations are rejected.
The combined command reconstructs ALL 51 original figure pairs: CNN 5/Pareto 46.
The smaller publication selection does not restrict the other 29 figures.
Numerical aggregates are compared to retained original reference CSVs.
Figures appear under cnn/figures and pareto/figures in the chosen output.
CNN exports final_per_seed_metrics, final_median_iqr_summary, pointwise
training/test loss/accuracy and gradient_diagnostics_summary CSVs.
Pareto exports both per-run comparison CSVs and five-seed aggregate tables.
All per-seed numerical records, failures and configurations remain available.

## Pareto protocol and failures

84 configurations x five seed pairs = 420 physical runs, 3000 nominal updates.
K=11,31,51; clipping thresholds=1,5,7,10,100; four frozen schedules.
Seed pairs (data,training): (12345,54321), (27182,31415), (16180,14142),
(42424,73737), (86753,97531). Data/parameters/updates are float32; gradient
norms accumulate in float64. Two comparison modes share the physical runs.

396 runs completed; 24 SGD runs stopped with "nonfinite per-observation loss".
No MoM or clipped-SGD run failed. These are finite-precision numerical events,
not proofs of mathematical divergence. Large finite errors remain finite records;
no arbitrary error threshold reclassifies them. Failed attempts are terminal,
not rescued/restarted as successes. Run IDs, failure iterations/flags, sampled
observations, observed times and finite excursions are retained.
Unavailable endpoint errors/losses are NaN, not the last finite state.
Original JSON permits NaN/Infinity to preserve those records.

Endpoint medians/Q1/Q3 condition on completed finite endpoints: 3/5 seeds for
each SGD setting and 5/5 for MoM/clipped SGD, in both modes. Completion percentages
still use all five planned seeds. Trajectory bands use available finite strictly
positive values per iteration; SGD denominators can drop 5 -> 4 -> 3.
docs/STATISTICAL_SUMMARIES.md and audit CSVs state every denominator.
Bands describe seed variability, not confidence intervals.
Two comparison modes are not 840 independent runs.

## CNN protocol and summaries

Seeds=42,123,2026; CPU/four threads/float32; bias-free CNN, 94,304 parameters.
15000 updates, batch=100, MoM K=5/m=20; no momentum/weight decay/regularization.
Learning rate=.1 at 1-4999, .05 at 5000-9999, .025 at 10000-15000.
Eight methods: SGD, normalized SGD, clipping at .25,.5,2,4,8, and MoM.
Full train/test evaluation every 500 updates including 0; evaluation batch=128.
All 24 seed/method histories completed.

Pointwise/final quartiles require exactly three finite seed observations and use
NumPy linear quantiles; IQR=Q3-Q1. Missing/nonfinite seeds are rejected.
Diagnostics average within each nonoverlapping 500-update seed/window before
three-seed aggregation; defined_counts records within-window finite observations.
Sequential training time is not pooled as controlled benchmark replicates.

## Dataset and expensive new experiments

These commands are for future reproduction. Full training is unnecessary for
paper reconstruction and was not executed during publication preparation.

    python scripts/prepare_cifar10.py
    python experiments/pareto_d10_final.py --audit-only
    python experiments/pareto_d10_final.py --output-dir runs/pareto_alpha2p1 --threads 2
    python experiments/cifar10_siopt_final.py preflight --architecture cnn --seed 42 --threads 4 --device cpu --output runs/preflight42
    python experiments/cifar10_siopt_final.py run --architecture cnn --seed 42 --threads 4 --device cpu --preflight runs/preflight42 --output runs/seed42

Repeat CNN seed-specific commands for 123 and 2026. Preflight performs tiny smoke
updates and benchmarking; it is separate from full scientific training.
CIFAR preprocessing: ToTensor, default float32 conversion, Normalize with
mean [.491,.482,.447], std [.247,.243,.262], no augmentation.
Official dataset files and exact transformed-sample caches stay local/ignored.
Dataset licensing is unchanged.

## Provenance and limits

Scientific source and numerical inputs are byte-identical to the audited
candidate; Git newline conversion is disabled to preserve hashes.
The original repository, audited candidate, results and all unselected figures
remain intact. No obsolete sensitivity script is a dependency.
Original source hashes identify completed runs; they are not hashes of the
reorganized candidate. Legacy checkpoints require legacy source; candidate-created
checkpoints can resume under their own source identities.

Recovered alpha=2.1 code matches the cached compiled module recursively, and all
five raw-data/noise fingerprint pairs match. Original byte-level source SHA256
remains unrecovered. This is not a full training rerun or cross-platform guarantee.
Original checkpoint/cache/hardware validation needs omitted local artifacts.
LICENSE.draft is not an applied license; CITATION.cff is intentionally incomplete.
