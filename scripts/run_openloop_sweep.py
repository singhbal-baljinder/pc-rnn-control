#!/usr/bin/env python3
"""
Multi-dataset, multi-seed open-loop online-learning comparison.

The seed-robustness analysis for the open-loop study. It keeps ONLY the
analysis. It keeps ONLY the online forward-prediction core (predict ->
denormalize -> RMSE -> time -> update) and removes everything not needed
for this paper:

  * noise injection (it was already inert: `n` was always 0.0)
  * all plotting
  * all plotting

What it adds, relative to the original single-dataset script:

  * a loop over a LIST of datasets (loaded via robot_data.load_robot_data)
  * a loop over a LIST of seeds
  * a deterministic-method skip (e.g. LWPR / LoG-GP run once, since they
    have no random initialisation and are data-determined)
  * one pickle PER DATASET, containing every (method, seed) run plus an
    across-seed aggregation (ranking stability + seed variability), which
    is the evidence the open-loop seed argument rests on.
  * an N-sweep ablation, restricted to methods where iterating N within a
    single timestep is well-posed: PC-RNN-I (always) and CerebNet (default).
    See N_SWEEP / N_SWEEP_BASE_METHODS below.

On the meaning of N, and why most methods are NOT swept
-------------------------------------------------------
For PC-RNN-I, N iterates an INFERENCE stage (refine the latent state for the
current timestep by gradient descent on the free energy, with the observation
held fixed) and then a LEARNING stage -- the temporal dynamics advance exactly
once. CerebNet can likewise re-fit its linear readout on the current sample
without advancing the reservoir. For these, N is a genuine within-timestep
refinement and is swept.

The closed-form online learners (RTRL, UORO, KF-RTRL, R-KF-RTRL, RFLO,
E/F-BPTT) compute their (approximate) gradient in ONE shot; there is no inner
solver to iterate. The only multi-step mechanism available in the framework
(the `num_iters` loop in simulation.py wrapping forward_pass + train_step)
re-advances the hidden state AND the influence-matrix recursion on a frozen
input and applies N SGD steps -- which corrupts the one-step-ahead semantics
rather than refining the estimate. Sweeping N for those would handicap them,
not test them fairly, so they are reported at N=1 only (still multi-seed).
the available mechanism corrupts them the same way; add them to
N_SWEEP_BASE_METHODS only after verifying that `num_iters` iterates the
auxiliary solver rather than the simulation loop.

Timing note: to keep the per-step timings comparable to the original
(Table II), the timed prediction region includes the denormalisation
step exactly as before.

Interfaces preserved from the original:
    model = OnlineLearningModel(method_name, input_dim=..., output_dim=..., **kwargs)
    y_pred = model.predict(x_sample)            # normalised output
    model.update(x_sample, y_sample, prior=None)
"""

from __future__ import annotations

import os

# Must be set before any CUDA context is created (matches the original).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import hashlib
import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import tqdm

from pc_rnn_control.methods.dataset import load_robot_data, RobotDataset
from pc_rnn_control.methods.online_learning import OnlineLearningModel

# Same-seed -> same-result. Different seeds -> different runs (what we want).
try:
    torch.use_deterministic_algorithms(True)
except Exception as _e:  # some builds/ops lack a deterministic impl
    print(f"[warn] could not enable deterministic algorithms: {_e}")

METERS_TO_MILLIMETERS = 1000.0


# ----------------------------------------------------------------------------
# Configuration  -- edit this block for your experiments
# ----------------------------------------------------------------------------

# Each entry is forwarded to robot_data.load_robot_data(**entry).
# Add as many datasets as you like; one pickle is written per entry.
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# The SAME five test trajectories the closed-loop study uses, at both control
# frequencies: 5 x 2 = 10 open-loop datasets. Averaging a method over ten runs
# per seed rather than two makes the reported RMSE far less hostage to any one
# trajectory, and keeps the open- and closed-loop halves on the same motions.
# n_samples=None: use each trajectory in full (~1.0-1.6k steps).
_OPENLOOP_TRAJECTORIES = ["v5", "v19", "v21", "v23", "v35"]
_OPENLOOP_FREQS_HZ = [10, 20]
_OPENLOOP_DIR = os.path.join(
    _ROOT, "data", "open_loop"
)

_OPENLOOP_CANDIDATES: List[dict] = [
    dict(
        robot_name="spiraling_robot",
        data_dir=_OPENLOOP_DIR,
        filename=f"spiraling_babbling_{hz}_Hz_test_trajectory_{traj}_mvel0.5",
        n_samples=None,
        freq=float(hz),
    )
    for hz in _OPENLOOP_FREQS_HZ
    for traj in _OPENLOOP_TRAJECTORIES
]


def _one_per_trajectory(candidates: List[dict]) -> List[dict]:
    """Keep the FIRST listed recording of each trajectory -- one per trajectory.

    The (trajectory x frequency) grid does not give equal evidence per
    trajectory: load_robot_data ignores `freq` for spiraling_robot, and as of
    2026-08-26 the 10 Hz and 20 Hz files are byte-identical for v5, v19 and v23
    while genuinely differing for v21 and v35. Keeping every distinct file would
    therefore weight v21 and v35 twice as heavily as the rest in every average.
    One recording per trajectory makes the trajectory the unit of analysis and
    weights all five equally, at the cost of not using v21/v35's second file.
    """
    seen, keep = set(), []
    for d in candidates:
        path = os.path.join(d["data_dir"], d["filename"] + ".npz")
        traj = d["filename"].rsplit("test_trajectory_", 1)[-1].split("_")[0]
        if not os.path.exists(path):
            print(f"[datasets] missing, skipped: {d['filename']}")
            continue
        if traj in seen:
            print(f"[datasets] {traj} already covered, skipped: {d['filename']}")
            continue
        seen.add(traj)
        keep.append(d)
    return keep


DATASETS: List[dict] = _one_per_trajectory(_OPENLOOP_CANDIDATES)


# Seeds for the stochastic methods. >=5 gives a usable mean/std for the
# ranking-stability argument; deterministic methods use only the first.
SEEDS: List[int] = [0, 1, 2, 3, 4]

# Base methods to evaluate at their default N=1 (the full open-loop
# benchmark, Fig. 2). Comment out rows you do not need.
BASE_METHODS: List[str] = [
    # --- echo-state baseline ---
    "cerebnet",
    # --- online RNN family (unified framework) ---
    "rnn-rtrl",
    "rnn-uoro",
    "rnn-kf_rtrl",
    "rnn-reverse_kf_rtrl",
    "rnn-rflo",
    "rnn-dni",
    "rnn-kernl",
    "rnn-efficient_bptt",
    "rnn-future_bptt",
    # --- non-parametric (deterministic) ---
    "lwpr",
    "log_gp",
]

# Number of per-step iterations N to sweep -- the SAME four values used in the
# closed-loop experiments.
N_SWEEP: List[int] = [1, 2, 5, 10]

# WHICH methods are swept over N, and WHY (see audit of the online-learning
# framework). Iterating N within a single timestep is only well-posed where the
# extra steps REFINE the current-timestep estimate WITHOUT re-advancing the
# recurrent dynamics or the influence-matrix recursion:
#
#   * PC-RNN-I (proposed): N iterates internal VFE inference + learning for the
#     current timestep; the dynamics advance exactly once. Native and correct.
#     -> ALWAYS swept (handled separately below).
#
#   * CerebNet: the linear echo-state READOUT can be re-fit on the current
#     sample without advancing the reservoir dynamics. Well-posed.
#     -> swept (default member of N_SWEEP_BASE_METHODS).
#
#   * Closed-form online learners (RTRL, UORO, KF-RTRL, R-KF-RTRL, RFLO,
#     E/F-BPTT): compute their (approximate) gradient in ONE shot -- there is no
#     inner solver to iterate. The only available multi-step mechanism (the
#     `num_iters` loop in simulation.py) re-runs forward_pass + train_step,
#     which RE-ADVANCES the hidden state and the influence matrix on a frozen
#     input and applies N SGD steps -- corrupting the one-step-ahead semantics.
#     -> NOT swept; reported at N=1 only (still multi-seed).
#
#   * KeRNL / DNI (numerical): a within-timestep N could in principle iterate
#     their auxiliary SGD sub-solver, but the available mechanism corrupts
#     state the same way. -> NOT swept by default.
#
#   * LWPR / LoG-GP (non-parametric, deterministic): no iteration concept.
#     -> N=1, single seed (see DETERMINISTIC_METHODS).
#
# Edit this set to change which BASE methods are swept. PC-RNN-I is always
# swept regardless. To disable the ablation entirely, set N_SWEEP = [1].
N_SWEEP_BASE_METHODS = {"cerebnet"}

# Methods with NO random initialisation: identical across seeds, so we run
# them once. They are also NOT swept over N. Everything else is seed-sensitive
# (random init and/or random updates, e.g. UORO) and gets the full seed sweep.
#
# VERIFIED 2026-08-26 by running each method at seeds {0, 7, 12345} in separate
# processes and comparing mean_rmse bit-for-bit:
#   lwpr   -> 0.009673591809868 at every seed. Deterministic: the LWPR C library
#             statistics (no stored data to assign), and receptive-field
#             creation and removal are threshold tests on w_gen / w_prune.
#             threshold tests on w_gen / w_prune. Stays here.
#   log_gp -> 0.011759996, 0.011782498, 0.011813501. NOT deterministic: LoGGP
#             routes each point to a child model with random.random() (log_gp.py
#             lines 133 and 213), which _set_all_seeds() does seed. REMOVED, so
#             it now gets the full seed sweep like the other stochastic methods.
DETERMINISTIC_METHODS: set = set()

# LWPR has no RNG, but it does have a free initialisation: the initial distance
# metric init_D and the initial 2nd-order learning rate init_alpha (previously
# hardcoded at 40 and 250). Seeding THOSE puts LWPR on the same footing as the
# stochastic learners -- for them the seed sets the random initial weights, for
# LWPR it sets the initial metric -- so its across-seed spread means the same
# thing and belongs in the same column.
#
# Each is scaled by an independent log-uniform factor spanning one octave either
# way, so the nominal value is the median of the distribution and the parameter
# is perturbed symmetrically on the multiplicative scale that suits a distance
# metric and a learning rate. Set LWPR_SEED_INIT = False to pin the nominal.
LWPR_SEED_INIT = True
LWPR_INIT_D_NOMINAL = 40.0
LWPR_INIT_ALPHA_NOMINAL = 250.0
LWPR_INIT_OCTAVES = 1.0  # factor in [2**-1, 2**+1] = [0.5, 2.0]


def _seeded_kwargs(method_name: str, kwargs: dict, seed: int) -> dict:
    """Per-seed model kwargs. Only LWPR is affected: its initialisation is
    drawn from `seed` via a dedicated Generator, so it is reproducible and
    independent of whatever else consumed the global RNGs."""
    if method_name != "lwpr" or not LWPR_SEED_INIT:
        return kwargs
    rng = np.random.default_rng(seed)
    k = dict(kwargs)
    k["init_D"] = LWPR_INIT_D_NOMINAL * 2.0 ** rng.uniform(
        -LWPR_INIT_OCTAVES, LWPR_INIT_OCTAVES
    )
    k["init_alpha"] = LWPR_INIT_ALPHA_NOMINAL * 2.0 ** rng.uniform(
        -LWPR_INIT_OCTAVES, LWPR_INIT_OCTAVES
    )
    return k

# Store the full (T, n_y) prediction traces in the pickle (needed to redraw
# Fig. 2). Set False to keep only the scalar metrics and shrink the pickle.
STORE_PREDICTIONS = True

OUTPUT_DIR = os.path.join(_ROOT, "data", "open_loop")


# ----------------------------------------------------------------------------
# Per-method hyper-parameters (carried over from the original, trimmed)
# ----------------------------------------------------------------------------


def _pc_args(num_iters: int) -> dict:
    """PC-RNN-I config; only the number of inference/learning steps varies."""
    return {
        "infer_precision": True,
        "learn_Wr": True,
        "learn_Win": True,
        "learn_Wout": True,
        "use_output_bias": True,
        "use_input_bias": True,
        "use_known_input": True,
        "num_inference_iters": num_iters,
        "num_learning_iters": num_iters,
        "Sigma_x_init": 1e1,
        "Sigma_y_init": 1e1,
        "activation_fun": "sigmoid",
    }


ARG_CEREBNET = {
    "hidden_size": 100,
    "eta": 0.001,
    "lambda_reg": 0.001,
    "num_iters": 1,
}


def _rnn_args(num_iters: int = 1) -> dict:
    """RNN-family / non-parametric config. `num_iters` is left at 1 for the
    closed-form online learners (RTRL/UORO/KF-RTRL/R-KF-RTRL/RFLO/BPTT), for
    which within-timestep iteration is not well-posed (see config note). It is
    only varied for methods in N_SWEEP_BASE_METHODS."""
    return {"num_iters": num_iters, "activation_function": "sigmoid"}


@dataclass
class MethodSpec:
    """One experiment: a unique label, the model name to construct, its
    kwargs, the family it belongs to (so N-variants of the same method group
    together), and whether it is seed-deterministic (run once, no N sweep)."""

    label: str  # unique key in the results dict / ranking table
    method_name: str  # exact name passed to OnlineLearningModel
    kwargs: dict
    family: str  # groups N-variants of the same method
    deterministic: bool = False


def build_specs() -> List[MethodSpec]:
    """Assemble the experiment list from the config block above.

    PC-RNN-I is always swept over N_SWEEP. Among the base methods, only those
    in N_SWEEP_BASE_METHODS are swept (where within-timestep iteration is
    well-posed); the rest run at N=1. Deterministic methods run once.
    """
    specs: List[MethodSpec] = []

    # 1) PC-RNN-I (proposed) -- ALWAYS swept over N, grouped under one family
    #    so the N-curve is reported together.
    PC_FAMILY = "temporal_pc"
    for n in N_SWEEP:
        name = f"{PC_FAMILY}_N{n}"
        specs.append(
            MethodSpec(
                label=name,
                method_name=name,
                kwargs=_pc_args(n),
                family=PC_FAMILY,
                deterministic=False,
            )
        )

    # 2) Base methods.
    #    - deterministic (LWPR/LoG-GP): single N=1 run, single seed.
    #    - in N_SWEEP_BASE_METHODS (where N is well-posed, e.g. CerebNet):
    #      swept over N_SWEEP; the N=1 entry keeps the bare label, N>1 entries
    #      are labelled `<method>_N<n>` but construct the SAME model.
    #    - everyone else (closed-form online learners): N=1 only, still
    #      multi-seed. Iterating N for these is NOT well-posed (see config note).
    for m in BASE_METHODS:
        det = m in DETERMINISTIC_METHODS
        base_kw = dict(ARG_CEREBNET) if m == "cerebnet" else _rnn_args(1)

        if det:
            specs.append(
                MethodSpec(
                    label=m, method_name=m, kwargs=base_kw, family=m, deterministic=True
                )
            )
        elif m in N_SWEEP_BASE_METHODS:
            for n in N_SWEEP:
                kw = dict(base_kw)
                kw["num_iters"] = n
                label = m if n == 1 else f"{m}_N{n}"
                specs.append(
                    MethodSpec(
                        label=label,
                        method_name=m,
                        kwargs=kw,
                        family=m,
                        deterministic=False,
                    )
                )
        else:
            # N not well-posed for this method: run N=1 only (multi-seed).
            specs.append(
                MethodSpec(
                    label=m,
                    method_name=m,
                    kwargs=base_kw,
                    family=m,
                    deterministic=False,
                )
            )

    return specs


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------


def _set_all_seeds(seed: int) -> None:
    import random as _py_random

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _py_random.seed(seed)


def run_method(
    method_name: str,
    kwargs: dict,
    X_norm: np.ndarray,
    y_norm: np.ndarray,
    y_un: np.ndarray,
    denorm_fn,
    seed: int,
) -> dict:
    """
    Run one online forward-prediction pass of one method at one seed.

    Returns a dict of metrics (and, if STORE_PREDICTIONS, the traces).
    On failure, returns {"error": <message>, ...} so the sweep continues.
    """
    num_samples = X_norm.shape[0]
    dim_input = X_norm.shape[1]
    dim_output = y_norm.shape[1]

    try:
        # 1) Seed BEFORE construction so the random init is reproducible.
        _set_all_seeds(seed)
        kwargs = _seeded_kwargs(method_name, kwargs, seed)
        model = OnlineLearningModel(
            method_name, input_dim=dim_input, output_dim=dim_output, **kwargs
        )

        Y_pred = np.zeros((num_samples, dim_output))
        Y_pred_un = np.zeros((num_samples, dim_output))
        rmse_values = np.zeros(num_samples)
        pred_times = np.zeros(num_samples)
        update_times = np.zeros(num_samples)

        # 2) Seed AGAIN before the loop so any update-time randomness
        #    (e.g. UORO's random projections) starts from a known state.
        _set_all_seeds(seed)

        for i in tqdm.tqdm(
            range(num_samples), desc=f"{method_name}[seed {seed}]", leave=False
        ):
            x_sample = X_norm[i, :]
            y_sample = y_norm[i, :]
            y_sample_un = y_un[i, :]

            # --- Prediction (timed region matches the original: predict + denorm)
            t0 = time.perf_counter()
            Y_pred[i, :] = model.predict(x_sample)
            Y_pred_un[i, :] = denorm_fn(Y_pred[i, :])
            pred_times[i] = time.perf_counter() - t0

            # Per-step RMSE in metres, across the configuration dimensions.
            rmse_values[i] = np.sqrt(
                np.mean((Y_pred_un[i, :] - y_sample_un.reshape(1, -1)) ** 2)
            )

            # --- Online update (timed)
            t1 = time.perf_counter()
            model.update(x_sample, y_sample, prior=None)
            update_times[i] = time.perf_counter() - t1

        out = {
            "mean_rmse": float(np.mean(rmse_values)),  # metres
            "std_rmse": float(np.std(rmse_values)),  # metres (within-run variability)
            "mean_pred_time": float(np.mean(pred_times)),
            "mean_update_time": float(np.mean(update_times)),
            "p99_total_time": float(np.percentile(pred_times + update_times, 99)),
            "error": None,
        }
        if STORE_PREDICTIONS:
            out.update(
                {
                    "rmse": rmse_values.astype(np.float32),
                    "Y_pred": Y_pred.astype(np.float32),
                    "Y_pred_un": Y_pred_un.astype(np.float32),
                    "pred_times": pred_times.astype(np.float32),
                    "update_times": update_times.astype(np.float32),
                }
            )
        return out

    except Exception as e:
        return {"error": str(e), "mean_rmse": np.nan, "std_rmse": np.nan}


def aggregate_across_seeds(per_seed: Dict[int, dict], deterministic: bool) -> dict:
    """Summarise a method's runs across seeds (the seed-robustness evidence)."""
    ok = {s: r for s, r in per_seed.items() if r.get("error") is None}
    mean_rmse = np.array([r["mean_rmse"] for r in ok.values()], dtype=float)
    std_rmse = np.array([r["std_rmse"] for r in ok.values()], dtype=float)

    agg = {
        "deterministic": deterministic,
        "n_seeds_ok": int(mean_rmse.size),
        "seeds": sorted(ok.keys()),
        "mean_rmse_per_seed": mean_rmse,  # metres
        "std_rmse_per_seed": std_rmse,
        "n_failed": int(len(per_seed) - len(ok)),
    }
    if mean_rmse.size:
        m = float(mean_rmse.mean())
        s = float(mean_rmse.std())
        agg.update(
            {
                "mean_rmse_across_seeds": m,  # ranking statistic
                "std_rmse_across_seeds": s,  # KEY: seed variability of the mean
                "cv_rmse_across_seeds": float(s / m) if m > 0 else np.nan,
                "mean_std_rmse_across_seeds": float(std_rmse.mean()),
                "std_std_rmse_across_seeds": float(std_rmse.std()),
            }
        )
    return agg


def run_dataset(data: RobotDataset, specs: List[MethodSpec]) -> dict:
    """Run all specs x seeds on one dataset and return the save package."""
    print("\n" + "=" * 78)
    print(f"DATASET: {data.robot_name} / {data.filename}")
    print(f"  X={data.X.shape}  y={data.y.shape}  (n_u={data.n_u}, n_y={data.n_y})")
    print("=" * 78)

    results: Dict[str, Dict[int, dict]] = {}
    aggregate: Dict[str, dict] = {}

    for spec in specs:
        seeds_to_run = SEEDS[:1] if spec.deterministic else SEEDS

        tag = (
            " (deterministic: 1 seed)"
            if spec.deterministic
            else f" ({len(seeds_to_run)} seeds)"
        )
        n_iters = spec.kwargs.get(
            "num_iters", spec.kwargs.get("num_inference_iters", 1)
        )
        model_tag = (
            ""
            if spec.label == spec.method_name
            else f"  [model={spec.method_name}, N={n_iters}]"
        )
        print(f"\n-> {spec.label}{tag}{model_tag}")

        results[spec.label] = {}
        for seed in seeds_to_run:
            res = run_method(
                spec.method_name,
                spec.kwargs,
                data.X,
                data.y,
                data.y_un,
                data.denormalize,
                seed,
            )
            results[spec.label][seed] = res
            if res.get("error") is None:
                print(
                    f"   seed {seed:>3}: "
                    f"mean RMSE {res['mean_rmse'] * METERS_TO_MILLIMETERS:8.4f} mm "
                    f"(within-run std {res['std_rmse'] * METERS_TO_MILLIMETERS:7.4f} mm)"
                )
            else:
                print(f"   seed {seed:>3}: FAILED -> {res['error']}")

        agg = aggregate_across_seeds(results[spec.label], spec.deterministic)
        agg["model_name"] = spec.method_name
        agg["family"] = spec.family
        agg["num_iters"] = n_iters
        aggregate[spec.label] = agg

    _print_ranking(aggregate)
    _print_n_sweep(aggregate)

    return {
        "results": results,
        "aggregate": aggregate,
        "metadata": {
            "robot_name": data.robot_name,
            "filename": data.filename,
            "freq": data.freq,
            "pos_min": data.pos_min,
            "pos_max": data.pos_max,
            "Y_real": data.y,
            "Y_real_un": data.y_un,
            "seeds": SEEDS,
            "specs": [
                {
                    "label": s.label,
                    "method_name": s.method_name,
                    "family": s.family,
                    "num_iters": s.kwargs.get(
                        "num_iters", s.kwargs.get("num_inference_iters", 1)
                    ),
                    "deterministic": s.deterministic,
                }
                for s in specs
            ],
            "n_sweep": N_SWEEP,
            "deterministic_methods": sorted(DETERMINISTIC_METHODS),
            "store_predictions": STORE_PREDICTIONS,
        },
    }


def _print_ranking(aggregate: Dict[str, dict]) -> None:
    """Across-seed ranking table: the seed-robustness summary."""
    print("\n" + "-" * 78)
    print("ACROSS-SEED RANKING  (mean RMSE +/- seed std, mm)")
    print("-" * 78)
    rows = [(m, a) for m, a in aggregate.items() if "mean_rmse_across_seeds" in a]
    rows.sort(key=lambda x: x[1]["mean_rmse_across_seeds"])
    for rank, (method, a) in enumerate(rows, 1):
        mm = a["mean_rmse_across_seeds"] * METERS_TO_MILLIMETERS
        ss = a["std_rmse_across_seeds"] * METERS_TO_MILLIMETERS
        cv = a.get("cv_rmse_across_seeds", float("nan")) * 100.0
        det = "  [det]" if a["deterministic"] else ""
        print(
            f"{rank:2d}. {method:28s} {mm:8.4f} +/- {ss:6.4f}  "
            f"(CV {cv:5.2f}%, n={a['n_seeds_ok']}){det}"
        )
    print("-" * 78)


def _print_n_sweep(aggregate: Dict[str, dict]) -> None:
    """'Does N help?' table: group entries by family and print mean RMSE
    (+/- seed std) as a function of N. Covers every method swept over more
    than one N -- PC-RNN-I and every applicable baseline (incl. CerebNet)."""
    families: Dict[str, List[tuple]] = {}
    for label, a in aggregate.items():
        if "mean_rmse_across_seeds" not in a:
            continue
        families.setdefault(a.get("family", label), []).append(
            (a.get("num_iters", 1), label, a)
        )
    multi = {k: v for k, v in families.items() if len({n for n, _, _ in v}) > 1}
    if not multi:
        return

    print("\n" + "-" * 78)
    print("DOES N HELP?  mean RMSE +/- seed std (mm) vs. number of per-step iters")
    print("-" * 78)
    for family in sorted(multi):
        entries = sorted(multi[family], key=lambda x: x[0])
        print(f"{family}:")
        best_n, best_mm = None, float("inf")
        for n, label, a in entries:
            mm = a["mean_rmse_across_seeds"] * METERS_TO_MILLIMETERS
            ss = a["std_rmse_across_seeds"] * METERS_TO_MILLIMETERS
            print(f"    N={n:<3d} {mm:8.4f} +/- {ss:6.4f}")
            if mm < best_mm:
                best_mm, best_n = mm, n
        base = next((a for n, _, a in entries if n == 1), None)
        if base is not None and best_n is not None:
            base_mm = base["mean_rmse_across_seeds"] * METERS_TO_MILLIMETERS
            delta = base_mm - best_mm
            verdict = (
                f"improves by {delta:.4f} mm at N={best_n}"
                if delta > 0
                else "no improvement from N>1"
            )
            print(f"    -> {verdict}")
    print("-" * 78)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    specs = build_specs()
    print(f"Seeds: {SEEDS}")
    print(f"Experiments: {len(specs)}  |  Datasets: {len(DATASETS)}")
    print(
        f"N sweep: {N_SWEEP} for PC-RNN-I"
        + (f" + {sorted(N_SWEEP_BASE_METHODS)}" if N_SWEEP_BASE_METHODS else "")
        + " (all other methods run at N=1)"
    )

    for entry in DATASETS:
        data = load_robot_data(**entry)
        package = run_dataset(data, specs)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"results_multiseed_{data.robot_name}_{data.filename}.pkl",
        )
        with open(out_path, "wb") as f:
            pickle.dump(package, f)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
