# Fresh-environment release validation

Validated on 2026-10-25, Windows x64, regular CPython 3.14.6.
A new virtual environment was created without system-site-packages or path files
pointing to existing installed distributions. PYTHONPATH/PYTHONHOME and inherited
pip-index configuration were excluded from the validation subprocess environment.
The source copy contained only the proposed publication files.

Actual package downloads/installations:
- torch 2.14.0+cpu and torchvision 0.29.0+cpu from the official CPU index.
- All remaining pinned dependencies downloaded and installed from package indexes.
- No reuse of an existing environment's installed scientific distributions.
- pip check: passed, no dependency conflicts.
- Exact runtime versions and scientific module paths: checked and passed.

Executed checks:
- Manifest verification before and after tests: passed.
- Lightweight suite: 51 passed in 63.52s (0:01:03).
- Pareto frozen configuration --audit-only: passed.
- Standalone failure audit: passed; all four generated CSVs are byte-identical
  to the included failure/denominator reports.
- Combined reconstruct.py all command: PASSED for CNN AND Pareto.
- Every reconstructed numerical aggregate matched its retained reference table.
- Generated CNN 5 and Pareto 46 PDF/PNG pairs, all 102 valid file signatures.

The suite includes synthetic update/resume/failure checks, MoM estimator tests,
seed/architecture/schedule checks, three-seed aggregation and figure generation,
and reproduction of all five original Pareto raw-data/noise fingerprint pairs.
No scientific training trajectory was rerun. Full training was not executed.

Commands were run in a separate validation source copy, writing only new validation
outputs. Scientific source, retained numerical inputs and test files were compared
byte for byte with both this proposal and the unchanged audited candidate.
Documentation/report metadata were finalized afterward; no scientific code,
tests, plotting code or numerical inputs changed after validation.
The original mom-sgd repository and publication_repo_candidate remain unchanged.

Machine-readable evidence: provenance/release_validation.json.
Raw local validation logs/environment live outside the publication proposal.
They are not public inputs and are not required at runtime.

Still unverified/pending:
- Clean CIFAR-10 download/cache creation and any full-training reproduction.
- Other operating systems/hardware and bitwise cross-platform training behavior.
- Original Pareto byte-level source hash recovery.
- Final manuscript figure references, copyright/license approval and citation.
