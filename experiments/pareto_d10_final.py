"""Canonical raw Pareto d=10 experiment and five-seed analysis.

Use --audit-only to inspect the frozen grid, --analyze to rebuild reports from
canonical terminal runs, or run without either flag to train and then analyze.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import statistics
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.mom import mom_gradient
from src.optimizers import gradient_norm, clipped_sgd_step_, sgd_step_
from src.schedules import (
    regime1_learning_rate, regime1_block_size,
    regime2_learning_rate, regime2_block_size,
    regime3_log_learning_rate, regime3_rm_learning_rate,
)

NUM_SAMPLES, DIMENSION, NUM_ITERATIONS = 1_000_000, 10, 3000
PARETO_ALPHA, PARETO_XMIN = 2.1, 1.0
ERROR_VARIANCE, ERROR_STD = 4.0, 2.0
DATA_SEED, TRAINING_SEED = 12345, 54321
DTYPE, DEVICE = torch.float32, "cpu"
K_VALUES = [11, 31, 51]
GAMMA_VALUES_RAW = [1.0, 5.0, 7.0, 10.0, 100.0]
BETA, REGIME3_BLOCK_SIZE = 0.001, 100
THEORETICAL_SD = math.sqrt(PARETO_ALPHA * PARETO_XMIN**2 / (PARETO_ALPHA - 2))
FEATURE_SETTINGS = ("raw",)
COMPARISON_MODES = ("equal_iterations", "equal_samples")
TOTAL_ACTUAL_RUNS = 420
CONFIGURATIONS_PER_SEED = 84
SEED_PAIRS = {0: (12345, 54321), 1: (27182, 31415), 2: (16180, 14142),
              3: (42424, 73737), 4: (86753, 97531)}
SEED_IDS = (0, 1, 2, 3, 4)
HISTORICAL_DIRECTORIES = tuple(ROOT / "results" / name for name in (
    "pareto_features_d10_all_regimes_sensitivity",
    "pareto_features_d10_all_regimes_sensitivity_v2",
    "pareto_features_d10_confirmatory_5seeds",
    "pareto_features_d10_nonfinite_diagnostics",
    "pareto_features_d10_norm_fix_validation",
))
OUTPUT_DIRECTORY = ROOT / "results" / "pareto_d10_final"
SCHEDULES = {
    "regime1": ("regime1", "R1-LOG", regime1_learning_rate, regime1_block_size),
    "regime2": ("regime2", "R2-LOG", regime2_learning_rate, regime2_block_size),
    "regime3_a": ("regime3", "R3-CONST-LOG", regime3_log_learning_rate, None),
    "regime3_b": ("regime3", "R3-CONST-RM", regime3_rm_learning_rate, None),
}
IDENTITY_FIELDS = ["seed_id", "data_seed", "training_seed", "feature_scaling", "regime", "regime_variant",
                   "reference_k", "method", "k", "gamma", "beta", "lr_multiplier"]
TRAJECTORY_FIELDS = [
    "iteration", "samples_processed", "theta_error", "loss", "learning_rate",
    "block_size", "batch_size", "gradient_norm", "gradient_norm_before_clipping",
    "clipping_active", "clipping_factor", "cumulative_clipping_fraction",
    "master_batch_sha256", "elapsed_seconds", "finite", "failure_reason",
    "theta_nonfinite", "gradient_nonfinite", "loss_nonfinite",
    "prediction_nonfinite", "residual_nonfinite",
    "state_loss", "theta", "cumulative_mom_estimator_seconds", "cumulative_block_gradients",
]


def squared_loss(predictions, targets):
    """Per-observation squared loss, with no factor 1/2."""
    return (predictions - targets) ** 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mom_block_size(schedule, n):
    raw = SCHEDULES[schedule][3]
    if raw is None:
        return REGIME3_BLOCK_SIZE
    return 100 + raw(n) - raw(1)


def baseline_batch(schedule, k):
    return max(1, k * mom_block_size(schedule, NUM_ITERATIONS) // 2)


def budget_row(schedule, k):
    batches = [k * mom_block_size(schedule, n) for n in range(1, NUM_ITERATIONS + 1)]
    baseline = baseline_batch(schedule, k)
    budget = baseline * NUM_ITERATIONS
    total, updates = 0, 0
    for batch in batches:
        if total + batch > budget:
            break
        total += batch
        updates += 1
    require(0 < updates < NUM_ITERATIONS, "Equal samples must reduce MoM updates")
    require(total <= budget < total + batches[updates], "Budget prefix is not maximal")
    return dict(schedule=schedule, regime=SCHEDULES[schedule][0],
                regime_variant=SCHEDULES[schedule][1], K=k,
                reference_k=k, m_1=batches[0]//k, m_3000=batches[-1]//k,
                mom_total_samples_at_3000=sum(batches), baseline_total_samples_at_3000=budget,
                T_equal_samples=updates, mom_samples_at_T_equal_samples=total,
                m_initial=mom_block_size(schedule, 1), m_final=mom_block_size(schedule, NUM_ITERATIONS),
                mom_batch_initial=batches[0], mom_batch_final=batches[-1], baseline_batch=baseline,
                mom_samples_3000_iterations=sum(batches), baseline_samples_3000_iterations=budget,
                equal_sample_mom_iterations=updates, equal_sample_actual_mom_samples=total)


def schedule_audit():
    rows = []
    for name, (regime, variant, rate, raw_block) in SCHEDULES.items():
        row = dict(schedule=name, regime=regime, regime_variant=variant,
                   learning_rate_function=f"{rate.__module__}.{rate.__name__}",
                   block_size_function=f"{raw_block.__module__}.{raw_block.__name__}" if raw_block else "none: experiment constant 100",
                   experiment_block_rule="100 + raw(n) - raw(1)" if raw_block else "REGIME3_BLOCK_SIZE = 100")
        for n in (1, 10, 100, 1000, 3000):
            row[f"eta_{n}"] = rate(n)
            # There is no source R3 block-size function; blank is intentional.
            row[f"raw_schedule_m_{n}"] = raw_block(n) if raw_block else None
            row[f"experiment_m_{n}"] = mom_block_size(name, n)
            row[f"m_{n}"] = mom_block_size(name, n)
        rows.append(row)
    return rows


def configurations(seed_ids=SEED_IDS):
    rows = []
    for seed_id in seed_ids:
        data_seed, training_seed = SEED_PAIRS[seed_id]
        for schedule, (regime, variant, _, _) in SCHEDULES.items():
            for k in K_VALUES:
                for method, gamma in [("sgd", None), ("mom", None),
                                      *(("clipped", g) for g in GAMMA_VALUES_RAW)]:
                    rows.append(dict(seed_id=seed_id, data_seed=data_seed,
                        training_seed=training_seed, feature_scaling="raw",
                        schedule=schedule, regime=regime, regime_variant=variant,
                        reference_k=k, method=method, k=k if method == "mom" else None,
                        gamma=gamma, beta=None, lr_multiplier=None,
                        planned_iterations=NUM_ITERATIONS))
    return rows


def run_id(config):
    identity = [config[name] for name in IDENTITY_FIELDS]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def run_directory(root, config):
    suffix = f"clipped_gamma{config['gamma']:g}" if config["method"] == "clipped" else config["method"]
    return root / "trajectories" / f"seed{config['seed_id']}" / config["feature_scaling"] / config["schedule"] / f"K{config['reference_k']}" / suffix


def validate_settings():
    require((NUM_SAMPLES, DIMENSION, NUM_ITERATIONS, PARETO_ALPHA, PARETO_XMIN,
             ERROR_VARIANCE, ERROR_STD, BETA) == (1_000_000, 10, 3000, 2.1, 1.0, 4.0, 2.0, .001),
            "Experiment constants changed")
    require(ERROR_STD**2 == ERROR_VARIANCE and math.isclose(THEORETICAL_SD, math.sqrt(21)), "Noise/normalization mismatch")
    require((DATA_SEED, TRAINING_SEED, DTYPE, DEVICE) == (12345, 54321, torch.float32, "cpu"), "Frozen seeds/dtype/device changed")
    require(torch.arange(1, DIMENSION+1, dtype=DTYPE).tolist() == list(range(1,11)), "Incorrect theta_star")
    require(FEATURE_SETTINGS == ("raw",) and COMPARISON_MODES == ("equal_iterations", "equal_samples"), "Incorrect comparison settings")
    require(K_VALUES == [11, 31, 51], "Incorrect frozen K grid")
    require(GAMMA_VALUES_RAW == [1.0, 5.0, 7.0, 10.0, 100.0], "Incorrect frozen gamma grid")
    require(SEED_IDS == (0, 1, 2, 3, 4) and SEED_PAIRS[0] == (12345, 54321), "Incorrect seeds")
    require(len(set(SEED_PAIRS.values())) == 5, "Duplicate seed pairs")
    require(list(SCHEDULES) == ["regime1", "regime2", "regime3_a", "regime3_b"], "Incorrect schedule grid")
    require([entry[2] for entry in SCHEDULES.values()] == [regime1_learning_rate, regime2_learning_rate, regime3_log_learning_rate, regime3_rm_learning_rate], "Incorrect source schedules")
    require([entry[3] for entry in SCHEDULES.values()] == [regime1_block_size, regime2_block_size, None, None], "Incorrect source block schedules")
    for name, (_, _, _, raw_block) in SCHEDULES.items():
        require(mom_block_size(name, 1) == 100, "Initial block size is not 100")
        for n in range(1, NUM_ITERATIONS + 1):
            expected = 100 + raw_block(n) - raw_block(1) if raw_block else 100
            require(mom_block_size(name, n) == expected, "Incorrect shifted/constant blocks")
        for k in K_VALUES:
            require(baseline_batch(name, k) == k * mom_block_size(name, NUM_ITERATIONS) // 2, "Incorrect baseline batch")
            if raw_block is None:
                require(baseline_batch(name, k) == 50*k, "Incorrect Regime-3 baseline")
            budget_row(name, k)
    require(SEED_PAIRS == {0: (12345, 54321), 1: (27182, 31415), 2: (16180, 14142),
                           3: (42424, 73737), 4: (86753, 97531)}, "Frozen seed pairs changed")
    probe = torch.tensor([1e30, -1e30], dtype=DTYPE)
    require(gradient_norm(probe).dtype == torch.float64 and bool(torch.isfinite(gradient_norm(probe))),
            "Stable gradient norm not enabled")
    grid = configurations()
    require(TOTAL_ACTUAL_RUNS == len(grid) == 420, "Expected 420 final physical runs")
    require(len({run_id(c) for c in grid}) == 420, "Duplicate run IDs")
    require(len({run_directory(OUTPUT_DIRECTORY, c) for c in grid}) == 420, "Duplicate run directories")
    for seed_id in SEED_IDS:
        seed = [c for c in grid if c["seed_id"] == seed_id]
        require(len(seed) == CONFIGURATIONS_PER_SEED == 84, "Expected 84 configurations per seed")
        for schedule in SCHEDULES:
            for k in K_VALUES:
                cell = [c for c in seed if c["schedule"] == schedule and c["reference_k"] == k]
                require([c["method"] for c in cell] == ["sgd", "mom"] + ["clipped"] * 5,
                        "Incorrect frozen cell")
    require(all(c["feature_scaling"] == "raw" and c["method"] != "normalized" for c in grid),
            "Normalized features and NSGD must be absent")
    return grid


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def generate_underlying_data(data_seed):
    generator = torch.Generator(device=DEVICE).manual_seed(data_seed)
    u = torch.rand((NUM_SAMPLES, DIMENSION), generator=generator, dtype=DTYPE, device=DEVICE)
    # Same uniform-endpoint guard as the existing experiment; never clamp X.
    z = PARETO_XMIN * u.clamp_min(torch.finfo(DTYPE).tiny).pow(-1 / PARETO_ALPHA)
    signs = 2 * torch.randint(0, 2, u.shape, generator=generator, device=DEVICE).to(DTYPE) - 1
    raw = signs * z
    epsilon = ERROR_STD * torch.randn(NUM_SAMPLES, generator=generator, dtype=DTYPE, device=DEVICE)
    require(bool(torch.isfinite(raw).all() and torch.isfinite(epsilon).all()), "Nonfinite generated data")
    return raw, epsilon


def make_master(schedule, k, training_seed):
    # Reset for each cell. Mode and feature scaling intentionally do not enter
    # the seed: matching cells have identical streams across those comparisons.
    # Analysis modes consume prefixes of these same physical trajectories.
    generator = torch.Generator().manual_seed(training_seed)
    width = max(baseline_batch(schedule, k), *(k * mom_block_size(schedule, n)
                                             for n in range(1, NUM_ITERATIONS + 1)))
    master = torch.stack([torch.randint(NUM_SAMPLES, (width,), generator=generator)
                          for _ in range(NUM_ITERATIONS)])
    require(master.shape == (NUM_ITERATIONS, width), "Invalid master shape")
    require(bool(((master >= 0) & (master < NUM_SAMPLES)).all()), "Invalid master indices")
    return master, [digest(row) for row in master]


def scaled_dataset(raw, epsilon, theta_star, scaling):
    """Use the same raw realization/noise, with a response matching actual X."""
    require(scaling in FEATURE_SETTINGS, "Invalid feature setting")
    X = raw
    y = X @ theta_star + epsilon
    require(bool(torch.isfinite(X).all() and torch.isfinite(y).all()), "Nonfinite generated data")
    return X, y


MAX_REPLACE_RETRIES = 20
REPLACE_RETRY_DELAY = 0.25


def replace_with_retry(temporary, path):
    """Attempt atomic replacement up to 20 times; preserve the temp on failure."""
    for attempt in range(1, MAX_REPLACE_RETRIES + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == MAX_REPLACE_RETRIES:
                print(
                    f"ERROR: could not replace {path.name} after {attempt} attempts.\n"
                    f"Latest complete temporary file preserved at:\n{temporary}",
                    flush=True,
                )
                raise
            print(f"Windows file-lock retry {attempt}/{MAX_REPLACE_RETRIES}: {path.name}",
                  flush=True)
            time.sleep(REPLACE_RETRY_DELAY)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def atomic_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


@contextmanager
def exclusive_run(root):
    """OS lock releases on crashes; prevents concurrent writers to this grid."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_manifest(root, grid, threads):
    sources = [Path(__file__),
               *(ROOT / "src" / name for name in ("mom.py", "optimizers.py", "schedules.py"))]
    manifest = dict(version=1, experiment="pareto_d10_final",
                    norm_accumulation="float64; data, parameters, gradients and updates remain float32",
                    terminal_outcomes="completed or genuinely failed; failed runs are never retrained", grid=grid,
                    seed_pairs={str(k): list(v) for k, v in SEED_PAIRS.items()},
                    dtype=str(DTYPE), device=DEVICE, threads=threads, torch_version=str(torch.__version__),
                    loss="(prediction-target)**2, no factor 1/2; defined in this experiment",
                    trajectory_loss="pre-update minibatch MSE; state_loss and theta saved at analysis endpoints",
                    final_loss="post-update full-dataset MSE at exact analysis endpoint",
                    analysis="one physical run; two derived views, no reruns or interpolation",
                    sample_accounting="drawn minibatch observations, including a terminal failed attempt",
                    timing="sum of per-iteration perf_counter intervals; includes gradient/update and diagnostics, excludes CSV/checkpoint writing, progress printing, dataset/master generation and plotting",
                    maximum_loss="maximum pre-update minibatch MSE in the analyzed trajectory prefix",
                    missing_thresholds="NaN; samples and iterations refer to the first strict threshold crossing",
                    block_timing="amortized MoM estimator time / number of block gradients; includes aggregation and finite checks",
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    schedules=schedule_audit())
    path = root / "manifest.json"
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == manifest, "Resume settings/code differ; use a fresh --output-dir")
    else:
        require(not any(p.name != ".run.lock" for p in root.iterdir()), "Refusing to adopt existing outputs without a manifest")
        atomic_json(path, manifest)


def load_records(root, grid):
    # The master summary is the resume index. Atomic per-run commits additionally
    # recover a crash after a run finished but before its summary row was written.
    summary_path = root / "summaries" / "all_runs_summary.csv"
    indexed = {}
    valid_ids = {run_id(config) for config in grid}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = row["run_id"]
                require(key in valid_ids and key not in indexed, "Unknown/duplicate run in master summary")
                if row["status"] in ("completed", "failed"):
                    indexed[key] = row
    records = {}
    for config in grid:
        directory = run_directory(root, config)
        path = directory / "result.json"
        if not path.exists():
            require(run_id(config) not in indexed, "Terminal summary row lacks its commit; refusing to rerun it")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        require(all(record[key] == config[key] for key in config), "Checkpoint configuration mismatch")
        require(record["status"] in ("completed", "failed"), "Invalid terminal status")
        if run_id(config) in indexed:
            require(record["status"] == indexed[run_id(config)]["status"], "Terminal summary/commit status mismatch")
        trajectory = directory / "trajectory.csv"
        require(trajectory.exists() and hashlib.sha256(trajectory.read_bytes()).hexdigest() == record["trajectory_sha256"], "Terminal trajectory missing or corrupt; refusing to rerun it")
        require(0 < record["actual_iterations"] <= config["planned_iterations"], "Invalid checkpoint length")
        require((record["status"] == "completed") == bool(record["finite"]), "Invalid checkpoint finite flag")
        actual = record["actual_iterations"]
        expected_samples = sum(config["k"] * mom_block_size(config["schedule"], n)
                               for n in range(1, actual + 1)) if config["method"] == "mom" else actual * baseline_batch(config["schedule"], config["reference_k"])
        require(record["total_samples_processed"] == expected_samples, "Invalid checkpoint sample accounting")
        if record["finite"]:
            require(record["actual_iterations"] == config["planned_iterations"], "Incomplete run marked completed")
        else:
            require(record["failure_iteration"] == actual and bool(record["failure_reason"]), "Invalid failure checkpoint")
        records[run_id(config)] = record
    return records


def save_summaries(root, records):
    if not records:
        return
    rows = [{key: value for key, value in record.items() if key != "analysis_summaries"}
            for record in records.values()]
    atomic_csv(root / "summaries" / "all_runs_summary.csv", rows)
    fields = ["run_id", *IDENTITY_FIELDS, "status", "actual_iterations", "total_samples_processed",
              "wall_clock_seconds", "wall_clock_seconds_per_iteration",
              "wall_clock_seconds_per_1e6_samples", "wall_clock_seconds_per_block_gradient",
              "seconds_per_iteration", "seconds_per_1e6_samples", "seconds_per_block_gradient",
              "equal_iterations_wall_clock_seconds", "equal_samples_wall_clock_seconds"]
    atomic_csv(root / "summaries" / "timing_summary.csv", [{key: row[key] for key in fields} for row in rows])
    for mode in COMPARISON_MODES:
        atomic_csv(root / "summaries" / f"{mode}_summary.csv",
                   [record["analysis_summaries"][mode] for record in records.values()])


def analysis_iteration(config, mode):
    require(mode in COMPARISON_MODES, "Unknown analysis mode")
    if config["method"] == "mom" and mode == "equal_samples":
        return budget_row(config["schedule"], config["reference_k"])["equal_sample_mom_iterations"]
    return NUM_ITERATIONS


def transient_metrics(trajectory):
    """Use recorded values, retaining finite excursions and terminal infinities."""
    def maximum(field):
        values = [row[field] for row in trajectory
                  if row[field] is not None and not math.isnan(row[field])]
        return max(values) if values else float("nan")

    result = dict(maximum_gradient_norm=maximum("gradient_norm"), maximum_loss=maximum("loss"))
    for exponent in (1, 2, 3):
        label = f"1e-{exponent}"
        first = next((row for row in trajectory if row["finite"]
                      and row["theta_error"] is not None and row["theta_error"] < 10**(-exponent)), None)
        iteration = first["iteration"] if first else float("nan")
        samples = first["samples_processed"] if first else float("nan")
        result[f"first_iteration_parameter_error_below_{label}"] = iteration
        result[f"first_samples_processed_parameter_error_below_{label}"] = samples
        result[f"first_iteration_below_{label}"] = iteration  # Preserve prior column aliases.
    finite_errors = [r["theta_error"] for r in trajectory
                     if r["theta_error"] is not None and math.isfinite(r["theta_error"])]
    finite_norms = [r["gradient_norm"] for r in trajectory
                    if r["gradient_norm"] is not None and math.isfinite(r["gradient_norm"])]
    result["last_finite_parameter_error"] = finite_errors[-1] if finite_errors else None
    result["maximum_finite_gradient_norm"] = max(finite_norms) if finite_norms else None
    for field in ("theta_nonfinite", "gradient_nonfinite", "loss_nonfinite",
                  "prediction_nonfinite", "residual_nonfinite"):
        result[field] = any(r[field] for r in trajectory)
    return result


def derive_comparison(record, trajectory, mode):
    """Pure prefix analysis: never computes gradients, updates, or interpolation.

    A failure after T does not invalidate the already recorded state at T.
    A failure before/at T produces an unavailable endpoint, not a substitute
    earlier state or a misleading shorter-run timing.
    """
    target = analysis_iteration(record, mode)
    prefix = [row for row in trajectory if row["iteration"] <= target]
    require(bool(prefix), "Cannot analyze an empty trajectory")
    last = prefix[-1]
    reached = last["iteration"] == target and all(row["finite"] for row in prefix)
    if reached:
        require(last["state_loss"] is not None and last["theta"] is not None,
                "Missing saved endpoint state/loss")
    if mode == "equal_samples":
        ceiling = baseline_batch(record["schedule"], record["reference_k"]) * NUM_ITERATIONS
        require(last["samples_processed"] <= ceiling, "Analysis prefix exceeds baseline budget")
    errors = [row for row in prefix if row["theta_error"] is not None and math.isfinite(row["theta_error"])]
    peak = max(errors, key=lambda row: row["theta_error"]) if errors else None
    norms = [row["gradient_norm"] for row in prefix
             if row["gradient_norm"] is not None and not math.isnan(row["gradient_norm"])]
    clipped = [row for row in prefix if row["clipping_factor"] is not None]
    elapsed = last["elapsed_seconds"] if reached else None
    failed = next((row for row in prefix if not row["finite"]), None)
    result = {key: record[key] for key in [*IDENTITY_FIELDS, "schedule", "run_id"]}
    result.update(
        comparison_mode=mode, physical_run_status=record["status"],
        status="completed" if reached else "failed", finite=reached,
        nominal_iterations=NUM_ITERATIONS, planned_iterations=target,
        actual_iterations=last["iteration"], endpoint_iteration=target,
        endpoint_available=reached, total_samples_processed=last["samples_processed"],
        block_size_initial=prefix[0]["block_size"], block_size_final=last["block_size"],
        batch_size_initial=prefix[0]["batch_size"], batch_size_final=last["batch_size"],
        final_parameter_error=last["theta_error"] if reached else float("nan"),
        final_loss=last["state_loss"] if reached else float("nan"),
        final_minibatch_loss=last["loss"], theta=last["theta"] if reached else None,
        maximum_parameter_error=peak["theta_error"] if peak else None,
        iteration_of_maximum_parameter_error=peak["iteration"] if peak else None,
        samples_at_maximum_parameter_error=peak["samples_processed"] if peak else None,
        maximum_gradient_norm_observed=max(norms) if norms else None,
        clipping_fraction=sum(row["clipping_active"] for row in clipped)/len(clipped) if clipped else None,
        mean_clipping_factor=sum(row["clipping_factor"] for row in clipped)/len(clipped) if clipped else None,
        wall_clock_seconds=elapsed,
        wall_clock_seconds_per_iteration=elapsed/target if reached else None,
        wall_clock_seconds_per_1e6_samples=elapsed/last["samples_processed"]*1e6 if reached else None,
        wall_clock_seconds_per_block_gradient=last["cumulative_mom_estimator_seconds"]/last["cumulative_block_gradients"]
        if reached and record["method"] == "mom" else None,
        observed_wall_clock_seconds=last["elapsed_seconds"],
        failure_iteration=failed["iteration"] if failed else None,
        failure_samples_processed=failed["samples_processed"] if failed else None,
        failure_reason=failed["failure_reason"] if failed else "",
    )
    result.update(transient_metrics(prefix))
    result.update(
        sample_budget=baseline_batch(record["schedule"], record["reference_k"]) * NUM_ITERATIONS,
        comparison_iteration=target, samples_processed=last["samples_processed"],
        parameter_error=result["final_parameter_error"], loss=result["final_loss"],
        cumulative_wall_clock_seconds=elapsed,
        seconds_per_iteration=result["wall_clock_seconds_per_iteration"],
        seconds_per_1e6_samples=result["wall_clock_seconds_per_1e6_samples"],
        seconds_per_block_gradient=result["wall_clock_seconds_per_block_gradient"],
    )
    return result


def train(config, X, y, theta_star, master, hashes, directory):
    directory.mkdir(parents=True, exist_ok=True)
    theta = torch.zeros_like(theta_star, requires_grad=True)
    rows, total, successful, active_count, factor_sum = [], 0, 0, 0, 0.0
    factor_count, largest_norm, mom_seconds, block_calls = 0, None, 0.0, 0
    method, schedule, k = config["method"], config["schedule"], config["reference_k"]
    equal_sample_iteration = analysis_iteration(config, "equal_samples")
    endpoints = {equal_sample_iteration, NUM_ITERATIONS}
    peak_error = peak_iteration = peak_samples = None
    crossings = {"first_iteration_below_1e-2": None, "first_iteration_below_1e-3": None}

    def check_gradient(gradient):
        # Autograd calls this for EVERY block gradient, not just the median.
        if not bool(torch.isfinite(gradient).all()):
            row["gradient_nonfinite"] = True
            row["gradient_norm"] = gradient_norm(gradient).item()
            raise FloatingPointError("nonfinite block/minibatch gradient")
        return gradient

    def checked_loss(predictions, targets):
        row["prediction_nonfinite"] = not bool(torch.isfinite(predictions).all())
        row["residual_nonfinite"] = not bool(torch.isfinite(predictions.detach() - targets).all())
        values = squared_loss(predictions, targets)
        if not bool(torch.isfinite(values).all()):
            row["loss_nonfinite"] = True
            row["loss"] = values.mean().item()
            raise FloatingPointError("nonfinite per-observation loss")
        return values

    hook = theta.register_hook(check_gradient)
    elapsed = 0.0  # Sum timed iterations, excluding CSV IO and progress output.
    final_loss = float("nan")
    try:
        for n in range(1, config["planned_iterations"] + 1):
            iteration_started = time.perf_counter()
            m = mom_block_size(schedule, n) if method == "mom" else None
            batch = k * m if method == "mom" else baseline_batch(schedule, k)
            rate = SCHEDULES[schedule][2](n)
            indices = master[n - 1, :batch]
            require(indices.numel() == batch and digest(master[n - 1]) == hashes[n - 1], "Master index stream changed")
            total += batch
            row = dict.fromkeys(TRAJECTORY_FIELDS)
            row.update(theta_nonfinite=False, gradient_nonfinite=False, loss_nonfinite=False,
                       prediction_nonfinite=False, residual_nonfinite=False)
            row.update(iteration=n, samples_processed=total, learning_rate=rate, block_size=m,
                       batch_size=batch, master_batch_sha256=hashes[n - 1], finite=True, failure_reason="")
            try:
                xb, yb = X[indices], y[indices]
                if method == "mom":
                    with torch.no_grad():
                        loss = checked_loss(xb @ theta, yb).mean()
                    row["loss"] = loss.item()
                    before = time.perf_counter()
                    try:
                        gradient = mom_gradient(theta, xb, yb, checked_loss, k, m)
                    finally:
                        mom_seconds += time.perf_counter() - before
                    block_calls += k
                else:
                    loss = checked_loss(xb @ theta, yb).mean()
                    row["loss"] = loss.item()
                    gradient = torch.autograd.grad(loss, theta)[0]
                norm_tensor = gradient_norm(gradient)
                norm = norm_tensor.item()
                row["gradient_norm"] = norm
                row["loss_nonfinite"] = not math.isfinite(row["loss"])
                if not math.isfinite(norm) or row["loss_nonfinite"]:
                    raise FloatingPointError("nonfinite gradient norm or mean loss")
                if method == "clipped":
                    factor = 1.0 if norm == 0 else torch.clamp(config["gamma"] / norm_tensor, max=1).item()
                    active = int(norm > config["gamma"])
                    row.update(gradient_norm_before_clipping=norm, clipping_active=active, clipping_factor=factor)
                    active_count += active
                    factor_sum += factor
                    factor_count += 1
                    row["cumulative_clipping_fraction"] = active_count / factor_count
                    clipped_sgd_step_(theta, gradient, rate, config["gamma"])
                else:
                    sgd_step_(theta, gradient, rate)
                with torch.no_grad():
                    row["theta_error"] = torch.linalg.vector_norm((theta - theta_star).double()).item()
                row["theta_nonfinite"] = not bool(torch.isfinite(theta).all())
                if row["theta_nonfinite"] or not math.isfinite(row["theta_error"]):
                    raise FloatingPointError("nonfinite theta or parameter error")
                successful += 1
                if n in endpoints:
                    with torch.no_grad():
                        final_loss = squared_loss(X @ theta, y).mean().item()
                    row["state_loss"] = final_loss
                    row["theta"] = json.dumps(theta.detach().tolist())
                    if not math.isfinite(final_loss):
                        row["loss_nonfinite"] = True
                        raise FloatingPointError("nonfinite endpoint full-dataset loss")
            except FloatingPointError as error:
                row.update(finite=False, failure_reason=str(error))
            observed_norm = row["gradient_norm"]
            if observed_norm is not None and not math.isnan(observed_norm):
                largest_norm = observed_norm if largest_norm is None else max(largest_norm, observed_norm)
            error = row["theta_error"]
            if error is not None and math.isfinite(error) and (peak_error is None or error > peak_error):
                peak_error, peak_iteration, peak_samples = error, n, total
            if row["finite"]:
                for threshold, field in ((1e-2, "first_iteration_below_1e-2"), (1e-3, "first_iteration_below_1e-3")):
                    if crossings[field] is None and error < threshold:
                        crossings[field] = n
            elapsed += time.perf_counter() - iteration_started
            row["elapsed_seconds"] = elapsed
            row["cumulative_mom_estimator_seconds"] = mom_seconds
            row["cumulative_block_gradients"] = block_calls
            rows.append(row)
            if n % 100 == 0 or n in endpoints or not row["finite"]:
                atomic_csv(directory / "trajectory.csv", rows, TRAJECTORY_FIELDS)
                print(f"{directory.relative_to(directory.parents[4])}: {n}/{config['planned_iterations']} "
                      f"samples={total} error={error} training_elapsed={elapsed:.1f}s", flush=True)
            if not row["finite"]:
                break  # Save the failed attempt; no rescue, restart, or skipped update.
    except KeyboardInterrupt:
        # Persist every fully recorded update, without marking interruption as
        # numerical failure or committing a completed/failed run.
        if rows:
            atomic_csv(directory / "trajectory.csv", rows, TRAJECTORY_FIELDS)
        print(f"Interrupted current run: saved {len(rows)} recorded updates. "
              "This unfinished configuration will restart deterministically on resume.", flush=True)
        raise
    finally:
        hook.remove()
    elapsed = rows[-1]["elapsed_seconds"]
    final = rows[-1]
    finite = final["finite"]
    count = len(rows)
    result = dict(config, run_id=run_id(config), status="completed" if finite else "failed",
        nominal_iterations=NUM_ITERATIONS, actual_iterations=count, successful_updates=successful,
        endpoint_iteration=NUM_ITERATIONS, samples_processed=total,
        block_size_initial=rows[0]["block_size"], block_size_final=final["block_size"],
        batch_size_initial=rows[0]["batch_size"], batch_size_final=final["batch_size"],
        total_samples_processed=total, final_parameter_error=final["theta_error"] if finite else float("nan"),
        final_loss=final_loss if finite else float("nan"), final_minibatch_loss=final["loss"],
        maximum_parameter_error=peak_error, iteration_of_maximum_parameter_error=peak_iteration,
        samples_at_maximum_parameter_error=peak_samples, **crossings,
        maximum_gradient_norm_observed=largest_norm,
        clipping_fraction=active_count/factor_count if factor_count else None,
        mean_clipping_factor=factor_sum/factor_count if factor_count else None,
        wall_clock_seconds=elapsed, wall_clock_seconds_per_iteration=elapsed/count,
        wall_clock_seconds_per_1e6_samples=elapsed/total*1e6,
        wall_clock_seconds_per_block_gradient=mom_seconds/block_calls if block_calls and finite else None,
        mom_estimator_seconds=mom_seconds if method == "mom" else None,
        finite=finite, failure_iteration=None if finite else count,
        failure_samples_processed=None if finite else total, failure_reason=final["failure_reason"],
        trajectory_sha256=hashlib.sha256((directory / "trajectory.csv").read_bytes()).hexdigest())
    result.update(transient_metrics(rows))
    result.update(seconds_per_iteration=result["wall_clock_seconds_per_iteration"],
                  seconds_per_1e6_samples=result["wall_clock_seconds_per_1e6_samples"],
                  seconds_per_block_gradient=result["wall_clock_seconds_per_block_gradient"])
    result["analysis_summaries"] = {mode: derive_comparison(result, rows, mode) for mode in COMPARISON_MODES}
    result["equal_iterations_wall_clock_seconds"] = result["analysis_summaries"]["equal_iterations"]["wall_clock_seconds"]
    result["equal_samples_wall_clock_seconds"] = result["analysis_summaries"]["equal_samples"]["wall_clock_seconds"]
    result["wall_clock_at_equal_samples"] = result["equal_samples_wall_clock_seconds"]
    atomic_json(directory / "result.json", result)
    return result


def method_label(row):
    if row["method"] == "clipped":
        return f"clipped gamma={row['gamma']:g}"
    return row["method"]


def report_progress(records, session_started, result=None):
    completed = len(records)
    elapsed = time.perf_counter() - session_started
    training_seconds = sum(row["wall_clock_seconds"] for row in records.values())
    eta = training_seconds/completed * (TOTAL_ACTUAL_RUNS-completed) if completed else None
    if result is not None:
        print(f"[{completed}/{TOTAL_ACTUAL_RUNS}] {result['feature_scaling']} | {result['regime_variant']} | "
              f"K={result['reference_k']} | {method_label(result)} | finite={result['finite']} | "
              f"final_error={result['final_parameter_error']} | max_error={result['maximum_parameter_error']} | "
              f"samples={result['total_samples_processed']} | time={result['wall_clock_seconds']:.3f}s", flush=True)
    print(f"completed_runs={completed}/{TOTAL_ACTUAL_RUNS} ({100*completed/TOTAL_ACTUAL_RUNS:.2f}%) | "
          f"remaining_runs={TOTAL_ACTUAL_RUNS-completed} | session_elapsed={elapsed:.1f}s | "
          f"cumulative_training_seconds={training_seconds:.1f} | "
          f"ETA_seconds={eta if eta is not None else 'pending first run'} "
          "(completed-run mean training time; excludes remaining setup/plotting)", flush=True)



def protect_output(output, exploratory):
    output, exploratory = output.resolve(), exploratory.resolve()
    require(output != exploratory and exploratory not in output.parents
            and output not in exploratory.parents, 'Output must be separate from exploratory results')


def validate_destination(root):
    for historical in HISTORICAL_DIRECTORIES:
        protect_output(root, historical)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--audit-only', action='store_true', help='Read-only validation; no training')
    parser.add_argument('--analyze', action='store_true', help='Regenerate tables and figures from canonical runs')
    args = parser.parse_args(argv)
    require(args.threads > 0, 'threads must be positive')
    validate_destination(args.output_dir)
    torch.set_num_threads(args.threads)
    grid = validate_settings()
    print(f'RAW only; K={K_VALUES}; gamma={GAMMA_VALUES_RAW}; regimes=4')
    print('84 configurations per seed; 5 seeds; 420 physical runs, all run from scratch under corrected code')
    print('Final combined seeds=5; final combined configurations=420')
    print('NSGD absent; normalized-feature runs absent; no historical results reused')
    print('Stable float64 norm accumulation enabled; data/model/gradients/updates remain float32')
    print('Genuine failures are terminal outcomes; all historical destinations excluded')
    print(f'Seed pairs: {SEED_PAIRS}; output={args.output_dir.resolve()}')
    for row in schedule_audit():
        print(json.dumps(row), flush=True)
    for schedule in SCHEDULES:
        for k in K_VALUES:
            print(json.dumps(budget_row(schedule, k)), flush=True)
    if args.audit_only:
        return
    if args.analyze:
        analyze(args.output_dir, grid)
        return
    root = args.output_dir
    session_started = time.perf_counter()
    with exclusive_run(root):
        check_manifest(root, grid, args.threads)
        atomic_csv(root / 'tables' / 'schedule_audit.csv', schedule_audit())
        atomic_csv(root / 'tables' / 'batch_budget_audit.csv', [budget_row(s, k) for s in SCHEDULES for k in K_VALUES])
        records = load_records(root, grid)
        try:
            save_summaries(root, records)
            for seed_id in SEED_IDS:
                pending = [c for c in grid if c['seed_id'] == seed_id and run_id(c) not in records]
                if not pending:
                    continue
                data_seed, training_seed = SEED_PAIRS[seed_id]
                raw, epsilon = generate_underlying_data(data_seed)
                fingerprint = dict(seed_id=seed_id, data_seed=data_seed, training_seed=training_seed,
                                   X_raw_sha256=digest(raw), epsilon_sha256=digest(epsilon),
                                   noise_std=ERROR_STD, theoretical_sd=THEORETICAL_SD)
                path = root / f'data_fingerprints_seed{seed_id}.json'
                if path.exists():
                    require(json.loads(path.read_text(encoding='utf-8')) == fingerprint, 'Dataset changed on resume')
                else:
                    require(not any(r['seed_id'] == seed_id for r in records.values()), 'Missing data fingerprint')
                    atomic_json(path, fingerprint)
                theta_star = torch.arange(1, DIMENSION + 1, dtype=DTYPE, device=DEVICE)
                X, y = scaled_dataset(raw, epsilon, theta_star, 'raw')
                for schedule in SCHEDULES:
                    for k in K_VALUES:
                        cell = [c for c in pending if c['schedule'] == schedule and c['reference_k'] == k]
                        if not cell:
                            continue
                        master, hashes = make_master(schedule, k, training_seed)
                        for config in cell:
                            print(f'Starting {len(records)+1}/{len(grid)}: {config}', flush=True)
                            result = train(config, X, y, theta_star, master, hashes, run_directory(root, config))
                            records[run_id(config)] = result
                            save_summaries(root, records)
                            report_progress(records, session_started, result)
                        del master, hashes
                del X, y, raw, epsilon
            require(len(records) == 420, 'Missing terminal configurations')
            analyze(root, grid, locked=True)
            print(f'Saved {len(records)} completed/failed runs to {root}', flush=True)
        except KeyboardInterrupt:
            records = load_records(root, grid)
            save_summaries(root, records)
            print('Stopped by Ctrl+C; completed/failed runs are intact. Resume with the same command.', flush=True)
            report_progress(records, session_started)
            return 130


ANALYSIS_METRICS = ('final_parameter_error', 'final_loss', 'maximum_parameter_error',
                    'wall_clock_seconds', 'clipping_fraction')


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    low = int(x)
    return values[low] + (x - low) * (values[min(low + 1, len(values) - 1)] - values[low])


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def load_trajectory(root, record):
    booleans = {'finite', 'theta_nonfinite', 'gradient_nonfinite', 'loss_nonfinite',
                'prediction_nonfinite', 'residual_nonfinite'}
    strings = {'master_batch_sha256', 'failure_reason', 'theta'}
    integers = {'iteration', 'samples_processed', 'block_size', 'batch_size',
                'clipping_active', 'cumulative_block_gradients'}
    path = run_directory(root, record) / 'trajectory.csv'
    history = []
    with path.open(newline='', encoding='utf-8') as stream:
        for raw in csv.DictReader(stream):
            require(set(raw) == set(TRAJECTORY_FIELDS), 'Trajectory schema mismatch')
            row = {}
            for key, value in raw.items():
                if key in booleans:
                    require(value in ('True', 'False'), 'Invalid trajectory boolean')
                    row[key] = value == 'True'
                elif key in strings:
                    row[key] = value if key != 'theta' or value else None
                else:
                    row[key] = (int(value) if key in integers else float(value)) if value else None
            history.append(row)
    require(len(history) == record['actual_iterations'] and bool(history), 'Trajectory length mismatch')
    for n, row in enumerate(history, 1):
        require(row['iteration'] == n, 'Nonconsecutive trajectory')
        require(row['learning_rate'] == SCHEDULES[record['schedule']][2](n), 'Learning-rate mismatch')
        require(row['finite'] or n == len(history), 'Failure must be terminal')
    terminal = history[-1]
    require(terminal['finite'] == record['finite'], 'Terminal finite flag mismatch')
    if record['status'] == 'failed':
        require(any(terminal[k] for k in booleans - {'finite'}), 'Failed run lacks nonfiniteness evidence')
        require(not math.isfinite(record['final_parameter_error']) and
                not math.isfinite(record['final_loss']), 'Failed run fabricates an endpoint')
    require(set(record['analysis_summaries']) == set(COMPARISON_MODES), 'Missing analysis mode')
    for mode in COMPARISON_MODES:
        derived = derive_comparison(record, history, mode)
        require(equivalent(derived, record['analysis_summaries'][mode]),
                'Stored analysis differs from trajectory')
    require(all(equivalent(record[k], v) for k, v in transient_metrics(history).items()),
            'Transient diagnostics mismatch')
    return history


def aggregate(rows):
    fields = [k for k in IDENTITY_FIELDS if k not in ('seed_id', 'data_seed', 'training_seed')]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in fields), []).append(row)
    require(len(groups) == 84, 'Expected 84 configuration groups')
    output = []
    for key, group in groups.items():
        require(len(group) == 5 and {r['seed_id'] for r in group} == set(SEED_IDS),
                'Missing or duplicate seed')
        completed = [r for r in group if r['status'] == 'completed' and r['finite'] is True]
        row = dict(zip(fields, key))
        row.update(schedule=group[0]['schedule'], comparison_mode=group[0]['comparison_mode'],
                   n_seeds_total=5, n_completed=len(completed), n_failed=5-len(completed),
                   completion_rate=len(completed)/5,
                   n_physical_failed=sum(r['physical_run_status'] == 'failed' for r in group),
                   statistics_condition='completed finite analysis endpoints only')
        for metric in ANALYSIS_METRICS:
            values = [r[metric] for r in completed] if metric != 'clipping_fraction' or row['method'] == 'clipped' else []
            require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
                    f'Nonfinite completed metric: {metric}')
            row['n_finite_' + metric] = len(values)
            stats = (dict(median=statistics.median(values), q25=quantile(values, .25),
                          q75=quantile(values, .75), mean=statistics.mean(values),
                          std=statistics.stdev(values) if len(values) > 1 else None)
                     if values else dict.fromkeys(('median', 'q25', 'q75', 'mean', 'std')))
            stats['iqr'] = stats['q75'] - stats['q25'] if values else None
            row.update({f'{metric}_completed_{name}': value for name, value in stats.items()})
        row['n_finite_final_error'] = row['n_finite_final_parameter_error']
        output.append(row)
    return output


def analyze(root, grid, locked=False):
    require(root.exists(), 'Canonical results do not exist')
    if not locked:
        with exclusive_run(root):
            return analyze(root, grid, locked=True)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['experiment'] == 'pareto_d10_final' and manifest['grid'] == grid,
            'Canonical manifest or grid mismatch')
    require(manifest['seed_pairs'] == {str(k): list(v) for k, v in SEED_PAIRS.items()} and
            manifest['schedules'] == schedule_audit() and manifest['dtype'] == 'torch.float32' and
            manifest['device'] == 'cpu', 'Scientific settings mismatch')
    for name, digest_value in manifest['source_sha256'].items():
        require(hashlib.sha256((ROOT / name.replace('\\', '/')).read_bytes()).hexdigest() == digest_value,
                f'Source mismatch: {name}')
    records = load_records(root, grid)
    require(len(records) == TOTAL_ACTUAL_RUNS, f'Expected 420 terminal runs; found {len(records)}')
    expected = {run_directory(root, c).resolve() / 'result.json' for c in grid}
    require(all(p.resolve() in expected for p in (root / 'trajectories').rglob('result.json')),
            'Unexpected physical record path')
    histories = {key: load_trajectory(root, record) for key, record in records.items()}
    save_summaries(root, records)
    tables = root / 'tables'
    summaries = root / 'summaries'
    for mode in COMPARISON_MODES:
        rows = [record['analysis_summaries'][mode] for record in records.values()]
        atomic_csv(summaries / f'{mode}.csv', rows)
        atomic_csv(tables / f'five_seed_aggregate_{mode}.csv', aggregate(rows))
    plot_final(root, records, histories)
    print(f'Validated and analyzed {len(records)} terminal runs; tables and figures saved to {root}')


def plot_final(root, records, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    regimes = [(key, value[1]) for key, value in SCHEDULES.items()]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f'{name}.png', dpi=180)
        fig.savefig(figures / f'{name}.pdf')
        plt.close(fig)
    for schedule, label in regimes:
        for method in ('sgd', 'mom', 'clipped'):
            for k_ref in (K_VALUES if method == 'clipped' else (None,)):
                configs = [(k, gamma) for k in K_VALUES for gamma in
                           (GAMMA_VALUES_RAW if method == 'clipped' else (None,))
                           if k_ref is None or k == k_ref]
                for metric, ylabel in (('theta_error', 'Parameter error'), ('loss', 'Minibatch loss')):
                    fig, ax = plt.subplots(figsize=(7.2, 4.8))
                    plotted = []
                    for i, (k, gamma) in enumerate(configs):
                        selected = [histories[key] for key, r in records.items()
                                    if r['schedule'] == schedule and r['method'] == method and
                                    r['reference_k'] == k and r['gamma'] == gamma]
                        require(len(selected) == 5, 'Missing trajectory seed')
                        xs, med, low, high = [], [], [], []
                        for n in range(1, NUM_ITERATIONS + 1):
                            values = [h[n-1][metric] for h in selected if len(h) >= n and
                                      h[n-1]['finite'] and h[n-1][metric] is not None and
                                      math.isfinite(h[n-1][metric]) and h[n-1][metric] > 0]
                            if values:
                                xs.append(n); med.append(statistics.median(values))
                                low.append(quantile(values, .25)); high.append(quantile(values, .75))
                        if xs:
                            title = f'gamma={gamma:g}' if method == 'clipped' else f'K={k}' if method == 'mom' else f'reference K={k}'
                            color = colors[i % len(colors)]
                            ax.plot(xs, med, label=title, color=color, linewidth=1.2)
                            ax.fill_between(xs, low, high, color=color, alpha=.18)
                            plotted.extend(high)
                    ax.set(xlabel='Iteration', ylabel=ylabel, title=f'{label}: {"clipped SGD" if method == "clipped" else "MoM-SGD" if method == "mom" else "SGD"}')
                    if metric == 'theta_error':
                        ax.set_yscale('log')
                    elif plotted:
                        ax.set_yscale('symlog', linthresh=10)
                    ax.legend(fontsize=8)
                    suffix = f'_refK{k_ref}' if k_ref is not None else ''
                    save(fig, f'{schedule}_{method}{suffix}_{metric}')
    rows = aggregate([r['analysis_summaries']['equal_iterations'] for r in records.values()])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for schedule, label in regimes:
        chosen = sorted((r for r in rows if r['schedule'] == schedule and r['method'] == 'mom'),
                        key=lambda r: r['reference_k'])
        ax.errorbar([r['reference_k'] for r in chosen],
                    [r['final_parameter_error_completed_median'] for r in chosen],
                    yerr=[[r['final_parameter_error_completed_median']-r['final_parameter_error_completed_q25'] for r in chosen],
                          [r['final_parameter_error_completed_q75']-r['final_parameter_error_completed_median'] for r in chosen]],
                    marker='o', capsize=3, label=label)
    ax.set(xlabel='MoM-SGD K', ylabel='Final parameter error')
    ax.legend()
    save(fig, 'mom_k_sensitivity')
    for metric, name, ylabel in (('final_parameter_error', 'robustness_parameter_sensitivity', 'Final parameter error'),
                                 ('maximum_parameter_error', 'stability', 'Maximum parameter error'),
                                 ('wall_clock_seconds', 'wall_clock_cost', 'Seconds')):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for schedule, label in regimes:
            chosen = [r for r in rows if r['schedule'] == schedule and r['method'] == 'clipped' and r['reference_k'] == 31]
            chosen.sort(key=lambda r: r['gamma'])
            ax.errorbar([r['gamma'] for r in chosen], [r[f'{metric}_completed_median'] for r in chosen],
                        yerr=[[r[f'{metric}_completed_median']-r[f'{metric}_completed_q25'] for r in chosen],
                              [r[f'{metric}_completed_q75']-r[f'{metric}_completed_median'] for r in chosen]],
                        marker='o', capsize=3, label=label)
        ax.set(xscale='log', xlabel='clipped SGD gamma (reference K=31)', ylabel=ylabel)
        ax.legend()
        save(fig, name)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ('sgd', 'mom', 'clipped')
    counts = [(sum(r['status'] == 'completed' for r in records.values() if r['method'] == m),
               sum(r['method'] == m for r in records.values())) for m in methods]
    ax.bar(('SGD', 'MoM-SGD', 'clipped SGD'), [100*a/b for a, b in counts])
    ax.set(ylabel='Completed physical runs (%)', ylim=(0, 105))
    for i, (a, b) in enumerate(counts):
        ax.text(i, 100*a/b + 1, f'{a}/{b}', ha='center')
    save(fig, 'completion_rate')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mode in COMPARISON_MODES:
        mode_rows = aggregate([r['analysis_summaries'][mode] for r in records.values()])
        chosen = [r for r in mode_rows if r['method'] == 'mom' and r['reference_k'] == 31]
        ax.plot([r['regime_variant'] for r in chosen],
                [r['final_parameter_error_completed_median'] for r in chosen], marker='o', label=mode.replace('_', ' '))
    ax.set(yscale='log', ylabel='Median final parameter error', xlabel='Regime')
    ax.legend()
    save(fig, 'equal_iterations_vs_equal_samples')


if __name__ == '__main__':
    raise SystemExit(main())
