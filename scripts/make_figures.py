"""Figures and tables for the paper, computed from the recorded runs.

Reads the closed-loop recordings under data/closed_loop/ and the multi-seed
open-loop results under data/open_loop/, and writes every figure and .tex
fragment the paper includes.

HOW TO RUN
  python scripts/make_figures.py
  Figures land in the working directory, table fragments in tables/. Set
  CACHE_PKL below to reuse a precomputed DataFrame instead of the raw .npz.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import rankdata
from scipy.optimize import least_squares

try:
    from statsmodels.stats.multitest import multipletests

    _HAVE_SM = True
except Exception:
    _HAVE_SM = False

# Needed ONLY when recomputing from raw .npz; NOT needed when loading the cache.
# The two imports are kept separate on purpose: the geometry helpers are also
# used by the per-trajectory figures (which run off the cache), while the LSTM
# checkpoint loader is needed only by the full recompute path.
try:
    from pc_rnn_control.robot.frames import (
        create_reference_frame,
        change_reference_frame_dataset,
        un_normalize_it,
    )

    _HAVE_FRAME = True
except Exception as _e:
    print(f"[warn] frame helpers unavailable ({_e}); raw-.npz figures will be skipped.")
    _HAVE_FRAME = False

try:
    from pc_rnn_control.robot.policy import load_checkpoint

    _HAVE_CKPT = True
except Exception:
    _HAVE_CKPT = False

_HAVE_PKG = _HAVE_FRAME and _HAVE_CKPT

# If you only have the cached DataFrame, point this at it (else leave None):
CACHE_PKL = None
F1_LOG_Y = False  # linear y (main result, easier to read); set True to de-crowd
F1_STYLE = "families"  # "families": family markers + legend (linear, no zoom)
# "labeled":  per-point text labels + zoom inset (previous)
F1_PRED_METRIC = "residual"  # "residual": RMSE(pred-obs) and its std -- the SAME
#   prediction-error metric used in the open-loop table,
#   applied consistently to closed loop. [DEFAULT]
# "magnitude": RMSE/std of preds themselves; only if you
#   want the predicted-error magnitude instead.

# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS  ──  copied verbatim from make_figures.py
# ═════════════════════════════════════════════════════════════════════════
INDEX_OBS_FRAMES = [1, 3, 5, 7]
INDICES_OBS_POS = [i for k in INDEX_OBS_FRAMES for i in range(3 * k, 3 * k + 3)]
METERS_TO_MILLIMETERS = 1000
MOTOR_REVOLUTION_PER_STEP_RAD = 1 / 4096
FAULT_START_STEP = 400
FAULT_END_STEP = 900

TRIAL_INDICES = [0, 20, 40, 60]
TRIAL_INDICES_N2 = [i for i in TRIAL_INDICES if i < 20]
TRIAL_INDICES_N5 = [i for i in TRIAL_INDICES if 20 <= i < 40]
TRIAL_INDICES_N1 = [i for i in TRIAL_INDICES if 40 <= i < 60]
TRIAL_INDICES_N10 = [i for i in TRIAL_INDICES if 60 <= i < 80]

TARGET_DATASETS = ["v5", "v19", "v21", "v23", "v35"]
TARGET_CONDITIONS = ["10Hz Nominal", "10Hz Fault", "20Hz Nominal", "20Hz Fault"]

METHOD_FILENAME = {
    k: "temporal_pc"
    for k in [
        "temporal_pc_N1",
        "temporal_pc_N2",
        "temporal_pc_N5",
        "temporal_pc_N10",
    ]
}
DISPLAY_NAMES = {
    "open_loop": "OL-C",
    "baseline": "NCL-C",
    "log_gp": "LoG-GP",
    "lwpr": "LWPR",
    "rnn-uoro": "UORO",
    "rnn-rtrl": "RTRL",
    "rnn-kf_rtrl": "KF-RTRL",
    "rnn-reverse_kf_rtrl": "Rev-KF-RTRL",
    "rnn-rflo": "RFLO",
    "rnn-dni": "DNI",
    "rnn-efficient_bptt": "E-BPTT",
    "rnn-future_bptt": "F-BPTT",
    "rnn-kernl": "KERNL",
    "temporal_pc_N1": "PC-RNN-I\n(N=1) (ours)",
    "temporal_pc_N2": "PC-RNN-I\n(N=2) (ours)",
    "temporal_pc_N5": "PC-RNN-I\n(N=5) (ours)",
    "temporal_pc_N10": "PC-RNN-I\n(N=10) (ours)",
    "cerebnet": "CerebNet",
}
METHODS_ORDER = [
    "open_loop",
    "baseline",
    "log_gp",
    "lwpr",
    "rnn-uoro",
    "temporal_pc_N1",
    "temporal_pc_N2",
    "temporal_pc_N5",
    "temporal_pc_N10",
    "cerebnet",
]
COLOR_MAP = {
    "open_loop": "#16be40",
    "baseline": "#c611f3",
    "log_gp": "C0",
    "lwpr": "C1",
    "rnn-uoro": "#86837c",
    "temporal_pc_N1": "#db2680",
    "temporal_pc_N2": "#b8860b",
    "temporal_pc_N5": "#f80909",
    "temporal_pc_N10": "C4",
    "cerebnet": "#09dcf8",
}
_SHORT = {
    "PC-RNN-I (N=1) (ours)": "PC (N=1)",
    "PC-RNN-I (N=2) (ours)": "PC (N=2)",
    "PC-RNN-I (N=5) (ours)": "PC (N=5)",
    "PC-RNN-I (N=10) (ours)": "PC (N=10)",
}


# ═════════════════════════════════════════════════════════════════════════
# HELPERS + LOADER  ──  copied verbatim from make_figures.py
# ═════════════════════════════════════════════════════════════════════════
def _method_stem(m):
    return METHOD_FILENAME.get(m, m)


def _method_trial_indices(m):
    return {
        "temporal_pc_N2": TRIAL_INDICES_N2,
        "temporal_pc_N5": TRIAL_INDICES_N5,
        "temporal_pc_N1": TRIAL_INDICES_N1,
        "temporal_pc_N10": TRIAL_INDICES_N10,
    }.get(m, TRIAL_INDICES)


def transform(data):
    if not _HAVE_FRAME:
        raise ImportError(
            "transform() needs pc_rnn_control.robot.frames "
            "(frame helpers); see the import warning above."
        )
    pos, quat = np.asarray(data["pos_array"]), np.asarray(data["quat_array"])
    G_T_T1 = create_reference_frame(pos[0, :3], quat[0, :4])
    t1_pos, _ = change_reference_frame_dataset(pos, quat, G_T_T1)
    return t1_pos * METERS_TO_MILLIMETERS


def get_variance_weights(pos_ref_12d):
    stds = np.std(pos_ref_12d, axis=0)
    stds += 1e-6
    return stds / np.sum(stds)


def collect_comprehensive_data(force_recompute=False, use_weighted_error=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "..", "data", "closed_loop_cache.pkl")

    _NEW_COLS = [
        "per_marker_mean_error",
        "per_marker_p95_error",
        "Mean_Euclidean",
        "Pred_Mag_RMSE",
        "Pred_Mag_Std",
        "Total_P99_ms",
        "Total_PctUnder",
        "Total_PerStep_ms",
    ]
    df = None
    if not force_recompute and os.path.exists(cache_path):
        df_cand = pd.read_pickle(cache_path)
        missing = [c for c in _NEW_COLS if c not in df_cand.columns]
        if missing:
            print(f"  Cache stale (missing: {missing}). Re-computing…")
        else:
            print(f"Loading cache: {cache_path}")
            df = df_cand

    if df is None:
        if not _HAVE_PKG:
            raise ImportError(
                "Recompute needs the pc_rnn_control package. Either run on "
                "the robot machine, or load the cache via CACHE_PKL."
            )
        print("Processing raw datasets with Mean Euclidean metric…")
        base_data_dir = os.path.join(script_dir, "..", "data")
        all_datasets = ["v5", "v19", "v21", "v23", "v35"]
        all_conditions = {
            "10Hz Nominal": {
                "dir": "closed_loop",
                "cond": "",
                "freq": "10Hz",
                "state": "Nominal",
            },
            "10Hz Fault": {
                "dir": "closed_loop",
                "cond": "_fault",
                "freq": "10Hz",
                "state": "Fault",
            },
            "20Hz Nominal": {
                "dir": "closed_loop",
                "cond": "",
                "freq": "20Hz",
                "state": "Nominal",
            },
            "20Hz Fault": {
                "dir": "closed_loop",
                "cond": "_fault",
                "freq": "20Hz",
                "state": "Fault",
            },
        }
        records = []
        _ckpt = os.path.join(base_data_dir, "closed_loop", "lstm_inverse_dynamics.pth")
        if not os.path.exists(_ckpt):
            return pd.DataFrame()
        _mdl = load_checkpoint(
            _ckpt, {"input_dim": 15, "action_dim": 3, "device": "cpu"}
        )
        P_MIN, P_MAX = np.asarray(_mdl.pos_min), np.asarray(_mdl.pos_max)

        for cond_name, cfg in all_conditions.items():
            folder = os.path.join(base_data_dir, cfg["dir"])
            freq_hz = float(cfg["freq"].replace("Hz", ""))
            for d_id in all_datasets:
                d_name = (
                    f"spiraling_babbling_{cfg['freq'].replace('Hz','_Hz')}"
                    f"_test_trajectory_{d_id}_mvel0.5"
                )
                ref_p = os.path.join(folder, f"{d_name}.npz")
                if not os.path.exists(ref_p):
                    continue
                data_ref = np.load(ref_p, allow_pickle=True)
                pos_ref = transform(data_ref)
                ref_act = (
                    np.asarray(data_ref["motor_pos"])
                    if "motor_pos" in data_ref
                    else None
                )

                b0_p = os.path.join(
                    folder, f"{d_name}_closed_loop_seq0{cfg['cond']}.npz"
                )
                avg_base = None
                # Metric A: whole-body error = MEAN of the four per-marker Euclidean
                # distances (not the 12-coordinate norm). Optional weighting is now
                # per-marker (length 4); with the default use_weighted_error=False it
                # is uniform, so avg_base is the plain mean of the 4 marker distances.
                if use_weighted_error:
                    _w12 = get_variance_weights(pos_ref[:, INDICES_OBS_POS])  # len-12
                    Wm = _w12.reshape(len(INDEX_OBS_FRAMES), 3).sum(axis=1)  # len-4
                    Wm = Wm / Wm.sum() * len(INDEX_OBS_FRAMES)  # mean->1
                else:
                    Wm = np.ones(len(INDEX_OBS_FRAMES))
                if os.path.exists(b0_p):
                    b0_pos = transform(np.load(b0_p, allow_pickle=True))
                    nb = min(len(pos_ref), len(b0_pos))
                    diff_b = (
                        b0_pos[:nb, INDICES_OBS_POS] - pos_ref[:nb, INDICES_OBS_POS]
                    )
                    md_b = np.linalg.norm(
                        diff_b.reshape(nb, len(INDEX_OBS_FRAMES), 3), axis=2
                    )  # (nb,4)
                    avg_base = float(np.mean(np.mean(Wm * md_b, axis=1)))

                for m in METHODS_ORDER:
                    trials = (
                        _method_trial_indices(m) if m.startswith("temporal_pc") else [0]
                    )
                    for i in trials:
                        if m == "baseline":
                            fname = f"{d_name}_closed_loop_seq{i}{cfg['cond']}.npz"
                        elif m == "open_loop":
                            fname = f"{d_name}_open_loop_seq{i}{cfg['cond']}.npz"
                        else:
                            fname = f"{d_name}_seq{_method_stem(m)}_adaptive{i}{cfg['cond']}.npz"
                        p = os.path.join(folder, fname)
                        if not os.path.exists(p):
                            continue

                        data = np.load(p, allow_pickle=True)
                        pos_act = transform(data)
                        n = min(len(pos_ref), len(pos_act))

                        diff = (
                            pos_act[:n, INDICES_OBS_POS] - pos_ref[:n, INDICES_OBS_POS]
                        )
                        diff_full = pos_act[:n] - pos_ref[:n]
                        rms_per_comp = np.sqrt(np.mean(diff_full**2, axis=0))

                        diff_3d = diff.reshape(n, len(INDEX_OBS_FRAMES), 3)
                        marker_d = np.linalg.norm(diff_3d, axis=2)  # (n, 4)

                        # Metric A: per-step whole-body error = MEAN of the four
                        # per-marker Euclidean distances at that timestep (optionally
                        # marker-weighted). mean_euclidean, the p95 and the CDF array
                        # (err_per_step_array) all derive from this single quantity,
                        # so they are consistent with per_marker_mean_error and with
                        # the metric-A Gain above.
                        err_per_step = np.mean(Wm * marker_d, axis=1)  # (n,)
                        mean_euclidean = float(np.mean(err_per_step))
                        p95_euclidean = float(np.percentile(err_per_step, 95))
                        max_dev = np.max(marker_d, axis=1)

                        pred_rmse, smooth_pred = np.nan, np.nan
                        pred_rms_per_step = np.full(n, np.nan)
                        pred_mag_rmse, pred_mag_std = np.nan, np.nan
                        if (
                            "fw_model_predictions" in data
                            and "fw_model_observations" in data
                        ):
                            preds = np.asarray(data["fw_model_predictions"])
                            obs = np.asarray(data["fw_model_observations"])
                            n_p = min(len(preds), len(obs))
                            # preds and obs are already ERROR signals (differences) in
                            # normalised space, so the affine +P_MIN offset already
                            # cancelled when the difference was formed. Recovering mm is
                            # therefore PURE per-coordinate scaling by the range
                            # (P_MAX-P_MIN), then m->mm -- NOT un_normalize_it, which is
                            # an affine position-restorer and would wrongly re-add P_MIN
                            # to a displacement.
                            pos_scale = (
                                P_MAX - P_MIN
                            ) * METERS_TO_MILLIMETERS  # len-12, mm per normalised unit
                            # (A) residual: how well the forward model predicts the
                            #     observation -- RMSE(preds - obs). [model accuracy]
                            dm = (preds[:n_p] - obs[:n_p]) * pos_scale
                            pred_rms_per_step = np.sqrt(np.mean(dm**2, axis=1))
                            pred_rmse = float(np.mean(pred_rms_per_step))
                            smooth_pred = float(np.std(pred_rms_per_step))
                            # (B) magnitude: the predicted error signal itself -- "how
                            #     much error is anticipated". preds is a normalised error,
                            #     so its magnitude is the same pure per-coordinate scaling.
                            pm = preds[:n_p] * pos_scale
                            pred_mag_per_step = np.sqrt(np.mean(pm**2, axis=1))
                            pred_mag_rmse = float(np.mean(pred_mag_per_step))
                            pred_mag_std = float(np.std(pred_mag_per_step))

                        avg_cyc = max_cyc = pct_over = np.nan
                        avg_pu = max_pu = np.nan
                        avg_fix = max_fix = np.nan
                        lim = 1000.0 / freq_hz
                        if "total_time_cycle" in data:
                            tc = np.asarray(data["total_time_cycle"]) * 1000
                            avg_cyc = float(np.mean(tc))
                            max_cyc = float(np.max(tc))
                            pct_over = float(np.mean(tc > lim) * 100)
                        if "time_predictions_and_update" in data:
                            tpu = np.asarray(data["time_predictions_and_update"]) * 1000
                            avg_pu = float(np.mean(tpu))
                            max_pu = float(np.max(tpu))
                        if "fixed_model_inference_time" in data:
                            fmi = np.asarray(data["fixed_model_inference_time"]) * 1000
                            avg_fix = float(np.mean(fmi))
                            max_fix = float(np.max(fmi))
                        # Per-step TOTAL computation = PU + fixed-policy inference,
                        # aligned by length (arrays may differ by one sample). p99 and
                        # the fraction of steps within the control period are the honest
                        # tail statistics; max alone is outlier-dominated.
                        tot_p99 = pct_under = np.nan
                        tot_per_step = None
                        if (
                            "time_predictions_and_update" in data
                            and "fixed_model_inference_time" in data
                        ):
                            n_t = min(len(tpu), len(fmi))
                            tot_per_step = tpu[:n_t] + fmi[:n_t]
                            tot_p99 = float(np.percentile(tot_per_step, 99))
                            pct_under = float(np.mean(tot_per_step <= lim) * 100)

                        records.append(
                            {
                                "Method": m,
                                "Display": DISPLAY_NAMES[m],
                                "Traj": d_id,
                                "Cond": cond_name,
                                "Freq": cfg["freq"],
                                "State": cfg["state"],
                                "ref_act_array": ref_act,
                                "motor_pos": (
                                    np.asarray(data["motor_pos"])
                                    if "motor_pos" in data
                                    else None
                                ),
                                "rms_per_component": rms_per_comp,
                                "err_per_step_array": err_per_step,
                                "Pred_RMSE_Per_Step": pred_rms_per_step,
                                "per_marker_mean_error": np.mean(marker_d, axis=0),
                                "per_marker_p95_error": np.percentile(
                                    marker_d, 95, axis=0
                                ),
                                "Mean_Euclidean": mean_euclidean,
                                "P95_Euclidean": p95_euclidean,
                                "Mean_MaxSpatialDev": float(np.mean(max_dev)),
                                "P95_MaxSpatialDev": float(np.percentile(max_dev, 95)),
                                "Smoothness": float(np.std(err_per_step)),
                                "Pred_RMSE": pred_rmse,
                                "Smoothness_Pred": smooth_pred,
                                "Pred_Mag_RMSE": pred_mag_rmse,
                                "Pred_Mag_Std": pred_mag_std,
                                "Gain": (
                                    ((avg_base - mean_euclidean) / avg_base * 100)
                                    if avg_base
                                    else np.nan
                                ),
                                "AvgCycleTime_ms": avg_cyc,
                                "MaxCycleTime_ms": max_cyc,
                                "PctOvershoot": pct_over,
                                "AvgPU_ms": avg_pu,
                                "MaxPU_ms": max_pu,
                                "AvgFixed_ms": avg_fix,
                                "MaxFixed_ms": max_fix,
                                "SumMax_ms": (max_pu if not np.isnan(max_pu) else 0)
                                + (max_fix if not np.isnan(max_fix) else 0),
                                "Total_P99_ms": tot_p99,
                                "Total_PctUnder": pct_under,
                                "Total_PerStep_ms": tot_per_step,
                                "Budget_ms": lim,
                            }
                        )
        df = pd.DataFrame(records)
        df.to_pickle(cache_path)
        print(f"Cached to: {cache_path}")

    df = df[df["Traj"].isin(TARGET_DATASETS) & df["Cond"].isin(TARGET_CONDITIONS)]
    return df.reset_index(drop=True)


def load_dataframe():
    _NEED = ["Pred_Mag_RMSE", "Pred_Mag_Std", "per_marker_mean_error", "Mean_Euclidean"]
    if CACHE_PKL and os.path.exists(CACHE_PKL):
        print(f"Loading cache: {CACHE_PKL}")
        df = pd.read_pickle(CACHE_PKL)
        missing = [c for c in _NEED if c not in df.columns]
        if missing:
            print(f"  Cache is stale (missing {missing}). Recomputing from raw data…")
            return collect_comprehensive_data(force_recompute=True)
        return df[
            df["Traj"].isin(TARGET_DATASETS) & df["Cond"].isin(TARGET_CONDITIONS)
        ].reset_index(drop=True)
    return collect_comprehensive_data(force_recompute=False)


# ═════════════════════════════════════════════════════════════════════════
# PLOTTING.  Everything is computed from the loaded DataFrame `df`.
# ═════════════════════════════════════════════════════════════════════════
COL_1, COL_2 = 3.50, 7.16
REF_METHOD = "temporal_pc_N5"
# Vertical size of the per-segment figure (fig_persegment), in inches.
# This is the single knob to squeeze that figure vertically: lower = tighter.
# (Width is fixed at COL_2; the top/bottom margins below are fractions that
#  rescale automatically, so you normally only touch this number.)
PERSEG_FIG_H = 3.6
# Vertical size of the merged 2x2 closed-loop figure (fig_closed_loop_grid), in
# inches. Single knob to squeeze it vertically: lower = tighter.
GRID_FIG_H = 6.4
# Vertical size of the actuation+tracking timeseries grid
# (fig_actuation_tracking_grid), in inches. Single knob to squeeze it: lower =
# tighter. Width is fixed at COL_2.
TRACK_FIG_H = 2.8
# Where that figure's single shared legend goes.
#   "inside" -- in the emptiest region of one panel. Costs NO page space at all,
#               which is why it is the default; the trade is that it sits on top
#               of the plot area.
#   "right"  -- a column to the right of the axes. Costs width instead of
#               height, and the axes shrink by roughly the legend's width.
#   "below"  -- the classic strip under the figure. Costs ~0.3 in of height.
TRACK_LEGEND_LOC = "inside"
# Panel that hosts the "inside" legend, as (row, col). Default is the top-right
# panel, whose top-right corner is freed by folding the condition name into the
# panel tag on the left.
TRACK_LEGEND_AX = (0, 1)
# Corner of that panel the legend is pinned to. "best" lets matplotlib score the
# corners, which is only worth it if no corner is reserved -- here one is, and a
# fixed corner keeps the legend in the same place for every trajectory.
TRACK_LEGEND_INSIDE_LOC = "upper right"
# Clearance kept between the bottom of the top-row overlays (the legend and the
# panel tags) and the highest trace, as a fraction of the panel height. The band
# itself is not a fixed number any more -- it is MEASURED from the rendered
# overlays (_fit_row_headroom) and shrunk to the least that still clears them,
# so the panels stay as zoomed-in as the annotation allows. This is the only
# hand-set part: how much air to leave in that gap.
TRACK_LEGEND_HEADROOM_PAD = 0.03
# Starting point size of the "inside" legend, and the floor it may shrink to.
# The legend shares the top band with its panel's tag, so whether it fits is a
# function of the METHOD NAMES -- rename one and a hand-tuned size is wrong
# again. _fit_legend_beside_tag steps down from the first to the second until
# the two clear each other, so the labels can change freely.
TRACK_LEGEND_FONTSIZE = 6.0
TRACK_LEGEND_FONTSIZE_MIN = 5.0
# Legend columns. For "below" one row fits until the labels get long -- too many
# columns push the legend wider than the axes and bbox_inches="tight" then
# widens the whole figure past COL_2. For "inside" a squarer block reads better.
TRACK_LEGEND_NCOL = 7
# Columns in the "inside" legend. It has to fit within ONE panel, so this is a
# shape trade, not a width one: too few columns make a tall block that has to
# sit over the traces, and one row per entry (0) overflows the panel entirely.
# Three keeps it two rows deep and narrow enough that loc="best" can drop it in
# the empty band a timeseries leaves above or below its envelope.
TRACK_LEGEND_NCOL_INSIDE = 3
# Fraction of the figure width the "right" legend column gets. The axes are
# laid out in the remainder, so the figure stays COL_2 wide and the legend
# costs width instead of height.
TRACK_LEGEND_RIGHT_FRAC = 0.17
# Horizontal gap between the two columns of the 2x2 figure: lower = bigger plots.
# Bounded below by panel (a)'s colorbar+label (top gap) and panel (d)'s method
# tick labels (bottom gap), so do not go too tight.
GRID_WSPACE = 0.38


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "savefig.pad_inches": 0.02,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _short(method):
    d = DISPLAY_NAMES.get(method, method)
    return _SHORT.get(d, d).replace("\n", " ")


def _leg_short(method):
    """Compact legend label: PC methods -> 'PC (N=x)' (no full name, no '(ours)')."""
    if method.startswith("temporal_pc"):
        return "PC (N=" + method.split("_N")[-1] + ")"
    return _short(method)


def _holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns (reject_bool, p_adjusted).
    Uses statsmodels if available, else a clean reimplementation.
    Controls family-wise error over the supplied family of p-values."""
    p = np.asarray(pvals, float)
    if _HAVE_SM:
        rej, p_adj, _, _ = multipletests(p, alpha=alpha, method="holm")
        return rej, p_adj
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    # step-down: adjusted p_(i) = max over j<=i of (m-j)*p_(j), capped at 1
    p_adj_sorted = np.empty(m)
    running = 0.0
    for i in range(m):
        running = max(running, (m - i) * p_sorted[i])
        p_adj_sorted[i] = min(running, 1.0)
    p_adj = np.empty(m)
    p_adj[order] = p_adj_sorted
    return p_adj <= alpha, p_adj


def _rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def _boot_rb_ci(d, B=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval on the rank-biserial effect
    size, resampling the REAL paired differences. (Resamples the recorded data.)"""
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"))
    rs = np.empty(B)
    for b in range(B):
        rs[b] = _rank_biserial(d[rng.integers(0, n, n)])
    return (float(np.percentile(rs, 100 * alpha / 2)),
            float(np.percentile(rs, 100 * (1 - alpha / 2))))


def _stars(p, sig):
    if not sig:
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── F1 ───────────────────────────────────────────────────────────────────
# Method families: marker shape encodes family (info that text can't), colour
# encodes prediction RMSE (the heatmap). For the PC family, marker SIZE grows
# with N so the 4 points stay distinguishable without text in the legend-only style.
FAMILY = {
    "rnn-uoro": "recurrent",
    "lwpr": "nonparam",
    "log_gp": "nonparam",
    "cerebnet": "echostate",
    "temporal_pc_N1": "pc",
    "temporal_pc_N2": "pc",
    "temporal_pc_N5": "pc",
    "temporal_pc_N10": "pc",
}
FAMILY_MARKER = {"recurrent": "s", "nonparam": "^", "echostate": "v", "pc": "o"}
FAMILY_LABEL = {
    "nonparam": "Non-parametric (LWPR, LoG-GP)",
    "recurrent": "Recurrent (UORO)",
    "echostate": "Echo-state (CerebNet)",
    "pc": "PC-RNN-I (ours), N=1..10",
}
# one distinct marker per METHOD (used when F1_STYLE == "families")
METHOD_MARKER = {
    "rnn-uoro": "s",
    "lwpr": "^",
    "log_gp": "D",
    "cerebnet": "v",
    "temporal_pc_N1": "o",
    "temporal_pc_N2": "P",
    "temporal_pc_N5": "*",
    "temporal_pc_N10": "X",
}


def _pred_metric_cols(df=None):
    """Return (y_col, color_col, y_label, cbar_label) for the chosen F1 metric.
    Falls back to the residual metric if the magnitude columns are not present
    in the dataframe (e.g. an old cache built before they were added)."""
    if F1_PRED_METRIC == "magnitude":
        if df is not None and "Pred_Mag_Std" not in df.columns:
            print(
                "  [F1] WARNING: 'Pred_Mag_*' columns not in this dataframe "
                "(stale/external cache). Falling back to the residual metric. "
                "Re-run with force_recompute=True (or delete the cache) to get "
                "the predicted-error-magnitude plot."
            )
        else:
            return (
                "Pred_Mag_Std",
                "Pred_Mag_RMSE",
                "Predicted-error magnitude std (mm)",
                "Predicted-error magnitude (mm)",
            )
    return (
        "Smoothness_Pred",
        "Pred_RMSE",
        "Open-loop prediction-error std (mm)",
        "Prediction RMSE (mm)",
    )




# ── F2 ───────────────────────────────────────────────────────────────────


# ── F3 ───────────────────────────────────────────────────────────────────


# ── Combined main-results figure (F1 families + F3) ───────────────────────


# ── Merged 2x2 closed-loop figure (fig_main_pair + fig_persegment) ────────
def fig_closed_loop_grid(df, path="F_closed_loop_2x2.pdf", tau=10.0, tau_max_q=0.99):
    """Single 2x2 figure merging the two closed-loop figures (does NOT replace
    them -- fig_main_pair and fig_persegment are left intact):
      (a) prediction-error variance vs. gain   (b) cumulative error distribution
      (c) mean tracking error along the body    (d) per-segment significance map
    All panel/stat logic mirrors the two source functions. Height knob: GRID_FIG_H.
    The (d) star/CI key is a one-line footnote here; you may instead move it to
    the LaTeX caption (RAL-idiomatic) and delete the fig.text line below."""
    from matplotlib.lines import Line2D

    ycol, ccol, ylab, clab = _pred_metric_cols(df)

    fig = plt.figure(figsize=(COL_2, GRID_FIG_H))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=GRID_WSPACE, hspace=0.42
    )
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # ===== (a) trade-off landscape ==========================================
    learn = df[~df["Method"].isin(["baseline", "open_loop"])].dropna(subset=[ycol])
    agg = learn.groupby("Method")[["Gain", ycol, ccol]].mean().dropna()
    if not agg.empty:
        norm = mpl.colors.Normalize(agg[ccol].min(), agg[ccol].max())
        cmap = plt.cm.plasma
        x = agg["Gain"].values
        y = agg[ycol].values
        order = [m for m in METHODS_ORDER if m in agg.index]
        sc = None
        for m in order:
            sc = axa.scatter(
                agg.loc[m, "Gain"],
                agg.loc[m, ycol],
                c=[agg.loc[m, ccol]],
                cmap=cmap,
                norm=norm,
                marker=METHOD_MARKER.get(m, "o"),
                s=70 if m == REF_METHOD else 48,
                edgecolor="k",
                linewidth=0.4,
                zorder=4,
            )
        handles = [
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKER.get(m, "o"),
                color="none",
                markerfacecolor="0.5",
                markeredgecolor="k",
                markersize=6.5,
                label=_leg_short(m),
            )
            for m in order
        ]
        legA = axa.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.9,
            fontsize=6.2,
            handletextpad=0.4,
            labelspacing=0.25,
            borderpad=0.4,
        )
        for txt, m in zip(legA.get_texts(), order):
            if m.startswith("temporal_pc"):
                txt.set_color("#c81e1e")
                txt.set_fontweight("bold")
        if sc is not None:
            cax = axa.inset_axes([1.035, 0.0, 0.032, 1.0])
            cb = fig.colorbar(sc, cax=cax)
            cb.ax.tick_params(labelsize=6.0, pad=1)
            lo, hi = float(agg[ccol].min()), float(agg[ccol].max())
            cb.set_ticks([lo, hi])
            cb.formatter = mpl.ticker.FuncFormatter(
                lambda v, _: f"{np.floor(v * 1000) / 1000:.3f}"
            )
            cb.update_ticks()
            cb.set_label(clab, fontsize=6.3, labelpad=0, rotation=90)
            cb.ax.yaxis.set_label_coords(2.4, 0.5)
    axa.set_xlabel("Closed-loop error-reduction gain (%)", fontsize=8.5)
    axa.set_ylabel(ylab, fontsize=8.5)
    axa.tick_params(axis="both", labelsize=8)
    axa.set_title(
        "(a)  Prediction error variance vs. gain", fontsize=9, fontweight="bold", pad=6
    )

    # ===== (b) cumulative error distribution ================================
    cons = [
        m
        for m in ["baseline", "rnn-uoro", "lwpr", "log_gp", "cerebnet", REF_METHOD]
        if m in df["Method"].unique()
    ]
    base_rows = df[df["Method"] == "baseline"]
    tau_max = (
        float(
            np.percentile(
                np.concatenate(base_rows["err_per_step_array"].values), tau_max_q * 100
            )
        )
        if not base_rows.empty
        else 90.0
    )
    hl = []
    for m in cons:
        r = df[df["Method"] == m]
        if r.empty:
            continue
        allerr = np.concatenate(r["err_per_step_array"].values)
        s = np.sort(allerr)
        yy = np.arange(1, len(s) + 1) / len(s)
        ls = "--" if m == "baseline" else "-"
        lw = 2.2 if m == REF_METHOD else 1.4
        (line,) = axb.plot(
            s,
            yy,
            ls,
            color=COLOR_MAP.get(m, "grey"),
            lw=lw,
            zorder=5 if m == REF_METHOD else 3,
        )
        cx = float(np.mean(allerr < tau))
        axb.plot(
            tau,
            cx,
            "o",
            color=COLOR_MAP.get(m, "grey"),
            ms=4.5,
            mec="k",
            mew=0.5,
            zorder=6,
        )
        hl.append((line, f"{_leg_short(m)} \u2014 {cx*100:.0f}%"))
    axb.axvline(tau, color="0.15", lw=1.3, ls=":")
    axb.text(
        tau + 1.5,
        0.985,
        f"{tau:.0f} mm",
        fontsize=8,
        color="0.15",
        fontweight="bold",
        va="top",
        ha="left",
    )
    axb.set_xlim(0, tau_max)
    axb.set_ylim(0, 1.0)
    axb.set_xlabel("Euclidean tracking error (mm)", fontsize=8.5)
    axb.set_ylabel("Fraction of timesteps", fontsize=8.5)
    axb.tick_params(axis="both", labelsize=8)
    axb.set_title(
        "(b)  Cumulative error distribution", fontsize=9, fontweight="bold", pad=6
    )
    if hl:
        hs, ls_ = zip(*hl)
        axb.legend(
            hs,
            ls_,
            loc="lower right",
            fontsize=6.3,
            framealpha=0.9,
            title=f"Method \u2014 % under {tau:.0f} mm",
            title_fontsize=6.3,
        )

    # ===== per-segment computations (for (c) and (d)) =======================
    rows_d = [
        m
        for m in [
            "open_loop",
            "baseline",
            "rnn-uoro",
            "lwpr",
            "log_gp",
            "cerebnet",
            "temporal_pc_N1",
            "temporal_pc_N2",
            "temporal_pc_N10",
        ]
        if m in df["Method"].unique()
    ]
    plotC = [
        m
        for m in ["baseline", "rnn-uoro", "lwpr", "log_gp", "cerebnet", REF_METHOD]
        if m in df["Method"].unique()
    ]
    work = df.copy()
    for i in range(4):
        work[f"m{i}"] = work["per_marker_mean_error"].apply(
            lambda v: float(v[i]) if (v is not None and len(v) > i) else np.nan
        )
    seg_piv = [
        work.pivot_table(index=["Traj", "Cond"], columns="Method", values=f"m{i}")
        for i in range(4)
    ]
    body_piv = work.pivot_table(
        index=["Traj", "Cond"], columns="Method", values="Mean_Euclidean"
    )

    P = np.full((len(rows_d), 5), np.nan)
    ES = np.full_like(P, np.nan)
    CIL = np.full_like(P, np.nan)
    CIH = np.full_like(P, np.nan)
    for r, m in enumerate(rows_d):
        for c in range(5):
            piv = seg_piv[c] if c < 4 else body_piv
            if m not in piv.columns or REF_METHOD not in piv.columns:
                continue
            j = piv[[m, REF_METHOD]].dropna()
            d = j[m].values - j[REF_METHOD].values
            if len(d) < 1 or np.ptp(d) == 0:
                P[r, c], ES[r, c] = 1.0, 0.0
                CIL[r, c], CIH[r, c] = 0.0, 0.0
                continue
            try:
                P[r, c] = scipy_stats.wilcoxon(
                    d, alternative="two-sided", zero_method="pratt", mode="exact"
                ).pvalue
            except (ValueError, TypeError):
                try:
                    P[r, c] = scipy_stats.wilcoxon(
                        d, alternative="two-sided", zero_method="pratt"
                    ).pvalue
                except ValueError:
                    P[r, c] = 1.0
            ES[r, c] = _rank_biserial(d)
            CIL[r, c], CIH[r, c] = _boot_rb_ci(d)
    SIG = np.zeros_like(P, dtype=bool)
    Padj = np.full_like(P, np.nan)
    for c in range(5):
        col = P[:, c]
        valid = np.isfinite(col)
        if valid.sum() == 0:
            continue
        rej, padj = _holm(col[valid], alpha=0.05)
        SIG[valid, c] = rej
        Padj[valid, c] = padj
    n_blocks = int(body_piv.dropna().shape[0])
    seg_cols = ["Seg 1\n(base)", "Seg 2", "Seg 3", "Seg 4\n(tip)", "Body\navg."]

    # ===== (c) mean tracking error along the body ===========================
    xseg = np.arange(4)
    for m in plotC:
        sub = work[work["Method"] == m]
        if sub.empty:
            continue
        mu = [sub[f"m{i}"].mean() for i in range(4)]
        ls = "--" if m == "baseline" else "-"
        lw = 2.0 if m == REF_METHOD else 1.4
        axc.plot(
            xseg,
            mu,
            ls,
            color=COLOR_MAP.get(m, "grey"),
            lw=lw,
            marker="o",
            ms=3.5,
            label=_leg_short(m),
            zorder=5 if m == REF_METHOD else 3,
        )
    axc.set_xticks(xseg)
    axc.set_xticklabels(seg_cols[:4], fontsize=8)
    axc.set_ylabel("Mean Euclidean error (mm)", fontsize=8.5)
    axc.tick_params(axis="both", labelsize=8)
    mean_tops = [
        max(work[work["Method"] == m][f"m{i}"].mean() for i in range(4))
        for m in plotC
        if not work[work["Method"] == m].empty
    ]
    axc.set_ylim(0, (max(mean_tops) * 1.08) if mean_tops else None)
    axc.legend(
        loc="upper left",
        fontsize=6.3,
        framealpha=0.9,
        handlelength=1.5,
        labelspacing=0.25,
        borderpad=0.4,
    )
    axc.set_title(
        "(c)  Mean tracking error along the body", fontsize=9, fontweight="bold", pad=6
    )

    # ===== (d) per-segment significance map =================================
    disp = np.where(SIG, ES, np.nan)
    sig_vals = disp[np.isfinite(disp)]
    if sig_vals.size:
        vlo = max(0.0, np.floor((sig_vals.min() - 0.05) * 20) / 20)
        vhi = min(1.0, np.ceil((sig_vals.max() + 0.02) * 20) / 20)
        if vhi - vlo < 0.15:
            vlo, vhi = max(0.0, vhi - 0.20), min(1.0, vlo + 0.20)
    else:
        vlo, vhi = 0.0, 1.0
    im = axd.imshow(disp, cmap=plt.cm.viridis, vmin=vlo, vmax=vhi, aspect="auto")

    def _txtcol(v, sig):
        if not sig:
            return "#222222"
        return "white" if (v - vlo) / (vhi - vlo + 1e-9) < 0.55 else "#111111"

    for r in range(len(rows_d)):
        for c in range(5):
            if not SIG[r, c]:
                axd.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor="#ededed",
                        edgecolor="white",
                        lw=1,
                        zorder=1,
                    )
                )
    axd.set_xticks(range(5))
    axd.set_xticklabels(seg_cols, fontsize=7.5)
    axd.set_yticks(range(len(rows_d)))
    axd.set_yticklabels([_leg_short(m) for m in rows_d], fontsize=6.8)
    axd.axvline(3.5, color="k", lw=1.0)
    for r in range(len(rows_d)):
        for c in range(5):
            sig = SIG[r, c]
            tc = _txtcol(ES[r, c], sig)
            p_show = Padj[r, c] if np.isfinite(Padj[r, c]) else P[r, c]
            axd.text(
                c,
                r - 0.14,
                _stars(p_show, sig),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=tc,
            )
            axd.text(
                c,
                r + 0.28,
                f"[{CIL[r,c]:.2f}, {CIH[r,c]:.2f}]",
                ha="center",
                va="center",
                fontsize=5.0,
                color=tc,
            )
    axd.set_xticks(np.arange(-0.5, 5), minor=True)
    axd.set_yticks(np.arange(-0.5, len(rows_d)), minor=True)
    axd.grid(which="major", visible=False)
    axd.grid(which="minor", color="white", lw=1.2)
    axd.tick_params(which="minor", length=0)
    caxd = axd.inset_axes([1.04, 0.0, 0.038, 1.0])
    cbd = fig.colorbar(im, cax=caxd)
    cbd.set_label("effect size $r$", fontsize=6.6, labelpad=1)
    cbd.ax.tick_params(labelsize=6)
    # panel (d) title + the stats key as a 2-line subtitle of THIS panel
    # (was a figure-level footnote). To drop it instead, delete the axd.text below.
    axd.set_title(
        "(d)  Per-segment significance vs. PC-RNN-I (N=5)",
        fontsize=9,
        fontweight="bold",
        pad=20,
    )
    axd.text(
        0.5,
        1.015,
        rf"Wilcoxon signed-rank, Holm-corrected ($n{{=}}{n_blocks}$ blocks)" + "\n"
        r"$^{***}p<0.001$  $^{**}p<0.01$  $^{*}p<0.05$   $\cdot$   "
        r"$\mathrm{ns}$ = n.s.   $\cdot$   [ ] = 95\% CI on $r$",
        transform=axd.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#333333",
        style="italic",
    )

    # ===== figure title =====================================================
    fig.suptitle(
        "Closed-loop performance analysis", fontsize=11.5, fontweight="bold", y=0.985
    )
    # left widened so the long left-column y-labels are not clipped
    fig.subplots_adjust(left=0.092, right=0.90, top=0.92, bottom=0.075)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")






# ── Per-cell significance numbers (effect size r, p, power) ──────────────


# ── Per-step computation-time table (Avg / Max of the PU step) ────────────
# Machine the timings were taken on, and what they do/don't cover. NOT caption
# padding: absolute milliseconds cannot be read without the CPU they were
# measured on, and without the scope note a reader takes `Total` for the whole
# control cycle rather than computation alone. Kept here because it is a fact
# about the rig, not about the data -- edit it when the rig changes. Set either
# to "" to drop that clause.
TIMING_HARDWARE = (
    # Kept terse: this caption sits in a single column of an 8-page RA-L
    # submission with zero slack, and the longer wording cost a whole page.
    r"an Intel Core i7-11800H, non-real-time Ubuntu kernel, single Python process"
)
# R8.9 asked Table IV to report numeric precision and memory alongside the
# hardware. Both are configuration facts, not run outputs, so they live here with
# the rest of the caption text. Precision: no torch.set_default_dtype anywhere in
# predictive_dyn_models, so PC-RNN-I and the LSTM policy run at torch's float32
# default; every NumPy-based learner (LWPR, LoG-GP, CerebNet, the [21] family) is
# float64. Memory: summed nbytes/element_size over each learner's arrays and
# tensors right after OnlineLearningModel(model_type=m, input_dim=3,
# output_dim=12) -- the closed-loop constructor call, which passes no kwargs, so
# these are the sizes actually used. Measured: LWPR 0.09, CerebNet 0.18, UORO
# 0.18, PC-RNN-I 0.85, LoG-GP 98.3 MB. LoG-GP is the outlier because its
# constructor pre-allocates K as [pts*outs, pts*N] = [1200, 10000] float64 = 96 MB
# regardless of how many points arrive; the others grow with the data. Hence
# "at start-up": LWPR in particular allocates receptive fields online.
TIMING_ENVIRONMENT = (
    r"float32 (PC-RNN-I, policy), float64 elsewhere; start-up model state "
    r"$<$1\,MB except LoG-GP (98\,MB, pre-allocated)"
)
TIMING_SCOPE = (
    r"computation only, excluding motion-capture reception and motor "
    r"communication"
)


def make_timing_table(df, path="timing_table.tex", budget_ms=50.0):
    """Per-step forward-model computation time = predict + online update ('PU').
    Emits a booktabs LaTeX table with Avg PU, and the mean / p99 / in-budget
    share of the per-step TOTAL, per method, pooled over all runs.
    PU time is the same computation regardless of control frequency, so pooling
    across the 10/20 Hz conditions is fine. The real-time budget is 1000/f ms per
    control step (100 ms @ 10 Hz, 50 ms @ 20 Hz); `budget_ms` bolds any Max PU
    above it (set None to disable). OL-C and NCL-C run no adaptive forward model,
    so any PU time logged for them is not meaningful and is excluded.
    NOTE: PU is the right quantity to COMPARE learners; to back the real-time
    *claim*, also cite total cycle time / PctOvershoot in the text."""
    t = df.dropna(subset=["AvgPU_ms"])
    # OL-C (open_loop) and NCL-C (baseline) run no adaptive forward model, so any
    # PU time logged for them is not meaningful -- drop those rows explicitly.
    t = t[~t["Method"].isin(["open_loop", "baseline"])]
    if t.empty:
        print("Timing: no adaptive-method PU times in df; table skipped.")
        return
    # Total = per-step COMPUTATION = forward model (PU) + fixed LSTM-policy
    # inference. We report the AVERAGE and the 99th PERCENTILE of the per-step
    # total, plus the fraction of steps within the control period. p99 (not max)
    # is the honest tail: max is dominated by one-off OS/scheduler outliers. The
    # pooled p99 is the percentile of the per-step totals CONCATENATED across a
    # method's runs -- not a percentile of per-run summaries.
    t = t.copy()
    t["AvgTotal_ms"] = t["AvgPU_ms"] + t["AvgFixed_ms"]
    bm = budget_ms if budget_ms is not None else 50.0

    def _pooled_p99(sub):
        arrs = [a for a in sub["Total_PerStep_ms"] if a is not None and len(a)]
        if not arrs:
            return np.nan, np.nan
        allsteps = np.concatenate(arrs)
        p99 = float(np.percentile(allsteps, 99))
        pct = float(np.mean(allsteps <= bm) * 100)
        return p99, pct

    rows = {}
    for m, sub in t.groupby("Method"):
        p99, pct = _pooled_p99(sub)
        rows[m] = {
            "AvgPU": float(sub["AvgPU_ms"].mean()),
            "AvgTot": float(sub["AvgTotal_ms"].mean()),
            "P99Tot": p99,
            "PctUnder": pct,
        }
    g = pd.DataFrame(rows).T
    order = g.sort_values("AvgPU").index.tolist()  # rows by increasing Avg PU

    # Whether `bm` is the harshest budget in the study, checked against the
    # rates actually present rather than asserted: the caption should only claim
    # the table is scored against the strictest deadline if it is.
    _b = sorted({float(v) for v in df.get("Budget_ms", pd.Series(dtype=float)).dropna()})
    _strictest = ""  # dropped: costs a line in the 8-page RA-L layout

    L = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Per-step computation time. "
        rf"\emph{{PU}}: adaptive forward model (predict + online update). "
        rf"\emph{{Total}}: PU + fixed LSTM policy inference"
        + (rf"; {TIMING_SCOPE}" if TIMING_SCOPE else "")
        + rf". {bm:.0f}\,ms is the {1000.0 / bm:g}\,Hz control period{_strictest}."
        + (rf" Measured on {TIMING_HARDWARE}." if TIMING_HARDWARE else "")
        + (rf" {TIMING_ENVIRONMENT}." if TIMING_ENVIRONMENT else "")
        + r"}",
        r"\label{tab:timing}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& Fwd. (PU) & \multicolumn{3}{c}{Total computation} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}",
        rf"Method & Avg [ms] & Avg [ms] & p99 [ms] & \% $\le${bm:.0f} ms \\",
        r"\midrule",
    ]
    for m in order:
        apu = float(g.loc[m, "AvgPU"])
        at = float(g.loc[m, "AvgTot"])
        p99 = float(g.loc[m, "P99Tot"])
        pu = float(g.loc[m, "PctUnder"])
        L.append(
            f"{_leg_short(m)} & {apu:.2f} & {at:.2f} & {p99:.2f} & {pu:.2f} " + r"\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(L)
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(tex)
    print(f"\nSaved: {path}")


# ═════════════════════════════════════════════════════════════════════════
# Actuation & tracking timeseries (ported from make_figures.py)
# TRANSPOSED variant of the original 2x2 plot_actuation_and_tracking: rows are
# the signal (motor command, configuration-space coordinate) and columns the
# condition, instead of the other way round. Logic, options and values are
# identical to the source -- only the axes layout differs.
#
# The transpose is what buys the page space. With rows = condition no axis can
# be shared (a row spans rad AND mm), so every panel needs its own ticks, title
# and legend. With rows = signal, each row is ONE quantity in ONE unit across
# the conditions, so sharey="row" is meaningful and sharex is exact (all panels
# are the same trajectory at the same rate). Dropping the duplicated tick
# labels, the four titles and three of the four legends takes the figure from
# 7.1x6.0 in to COL_2 x TRACK_FIG_H with nothing removed -- and a shared y makes
# the Nominal/Fault comparison quantitative rather than eyeballed.
# ═════════════════════════════════════════════════════════════════════════
def _normalise_methods(methods):
    if methods is None:
        return {DISPLAY_NAMES[m]: [m] for m in METHODS_ORDER}
    if isinstance(methods, list):
        return {DISPLAY_NAMES[m]: [m] for m in methods}
    return methods


def _load_pos_for_method(m, d_name, cfg, folder):
    trials = _method_trial_indices(m) if m.startswith("temporal_pc") else [0]
    i = trials[0]
    if m == "baseline":
        fname = f"{d_name}_closed_loop_seq{i}{cfg['cond']}.npz"
    elif m == "open_loop":
        fname = f"{d_name}_open_loop_seq{i}{cfg['cond']}.npz"
    else:
        fname = f"{d_name}_seq{_method_stem(m)}_adaptive{i}{cfg['cond']}.npz"
    p = os.path.join(folder, fname)
    return transform(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def _find_best_component_idx(
    df, dataset_id, cond_name, ref_method="cerebnet", cmp_method="temporal_pc_N5"
):
    sub = df[(df["Traj"] == dataset_id) & (df["Cond"] == cond_name)]
    r_ref = sub[sub["Method"] == ref_method]
    r_cmp = sub[sub["Method"] == cmp_method]
    if r_ref.empty or r_cmp.empty:
        return -1
    disc = np.abs(
        np.asarray(r_ref["rms_per_component"].iloc[0])
        - np.asarray(r_cmp["rms_per_component"].iloc[0])
    )
    return int(np.argmax(disc))


def _motor_label(motor_indices):
    """Y-label for the actuation row -- "Motor 1 cmd (rad)".

    Numbering is 1-based to match the "Ref M{i+1}" series labels this figure
    already uses. `None` means every motor is drawn, and then there is no single
    number to name."""
    if motor_indices is None:
        return "Motor cmd (rad)"
    ids = [int(i) + 1 for i in motor_indices]
    if len(ids) == 1:
        return f"Motor {ids[0]} cmd (rad)"
    return f"Motors {', '.join(map(str, ids))} cmd (rad)"


def _component_label(idx):
    """Name of configuration-space component `idx` of the 24-vector `transform`
    returns -- 'Tip $y$', 'Frame 5 $z$'.

    Derived rather than written down because _find_best_component_idx CHOOSES
    the component from the data: a hardcoded axis label would silently disagree
    the moment the most discriminative component moves."""
    frame, axis = divmod(int(idx), 3)
    name = "Tip" if frame == INDEX_OBS_FRAMES[-1] else f"Frame {frame}"
    return f"{name} ${'xyz'[axis]}$"


def _fit_legend_beside_tag(ax, H, L, fig, **kw):
    """Draw the shared legend in `ax`, shrunk until it clears that panel's tag.

    Both are pinned to the top of the same panel -- tag left, legend right -- so
    they collide exactly when the method names get long enough, which is a thing
    that changes whenever a series is renamed. Rather than hand-tune the point
    size against the current names, step down until the boxes no longer touch
    (typically zero or one step), and give up at the floor rather than shrink
    into illegibility."""
    tags = list(ax.texts)
    fs = TRACK_LEGEND_FONTSIZE
    while True:
        leg = ax.legend(H, L, fontsize=fs, **kw)
        if fs <= TRACK_LEGEND_FONTSIZE_MIN:
            return leg
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lb = leg.get_window_extent(r)
        if not any(lb.overlaps(t.get_window_extent(r)) for t in tags):
            return leg
        leg.remove()
        fs -= 0.25


def _fit_row_headroom(fig, row_axes, pad=0.03, cap=0.02):
    """Open exactly enough blank band above `row_axes` for their overlays.

    Two things make this worth measuring instead of guessing a multiple of the
    data range. The overlays (legend, panel tags) are pinned to the AXES box, so
    the room they need is a fixed number of INCHES, not a fraction of the range.
    And each one only covers PART of the panel: the legend sits over the right
    of the top-right panel, the tags over the top-left of each. So the trace
    that has to stay below an overlay is the tallest one UNDER THAT OVERLAY's x
    span -- not the tallest in the row, which is usually somewhere else entirely
    and would push the limits up for no reason.

    `cap` is the clearance kept above the row's global peak, which has to stay
    inside the axes wherever it falls. Assumes sharey across the row: setting
    the limits on the first panel sets them for all of them.

    Only Line2D data is considered; the fault axvspan spans the full height by
    construction and would make every constraint unsatisfiable."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = row_axes[0].get_ylim()[0]
    tops = []  # (highest value that must clear it, fraction it must clear by)

    for ax in row_axes:
        inv_ax, inv_dat = ax.transAxes.inverted(), ax.transData.inverted()
        series = [
            (np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float))
            for l in ax.get_lines()
            if len(l.get_ydata())  # proxy handles: ax.plot([], []) for the legend
        ]
        if not series:
            continue
        tops.append((max(np.nanmax(y) for _, y in series), 1.0 - cap))

        overlays = list(ax.texts)
        if ax.get_legend() is not None:
            overlays.append(ax.get_legend())
        for a in overlays:
            bb = a.get_window_extent(r)
            frac = float(inv_ax.transform((bb.x0, bb.y0))[1]) - pad
            x_lo = float(inv_dat.transform((bb.x0, bb.y0))[0])
            x_hi = float(inv_dat.transform((bb.x1, bb.y0))[0])
            under = [
                y[(x >= x_lo) & (x <= x_hi)] for x, y in series
            ]
            under = [u for u in under if len(u)]
            if under and 0.0 < frac < 1.0:
                tops.append((max(np.nanmax(u) for u in under), frac))

    need = [y0 + (top - y0) / frac for top, frac in tops if top > y0]
    if need:
        row_axes[0].set_ylim(y0, max(need))


def fig_actuation_tracking_grid(
    df,
    dataset_id="v21",
    base_freq="10Hz",
    component_idx=None,
    methods=None,
    motor_indices=None,
    motor_rows=False,
    conds=("Nominal", "Fault"),
    path=None,
):
    """Transposed, shared-axis version of plot_actuation_and_tracking.

    Rows are the SIGNAL -- motor command (top) and one configuration-space
    coordinate (bottom); columns are the conditions in `conds`. Every drawing
    decision (method grouping, median-over-trials, min-max band, motor scaling,
    fault window, component choice) is the original's; only the layout differs.
    See the section header above for why the transpose is what saves the page.

    `motor_rows` gives each motor in `motor_indices` its OWN row instead of
    stacking them in one. Stacking them was tried and is unreadable: the motors
    of a method share its colour, so N motors x M methods in one panel is N*M
    traces the reader cannot attribute -- and the frozen actuator, which is the
    point of the fault column, vanishes into the pile. One row per motor keeps
    colour meaning "method" and position meaning "motor", at the cost of the
    page height the transpose was meant to save."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, "..", "data")
    conds = list(conds)
    cond_names = [f"{base_freq} {c}" for c in conds]

    # Chosen once from the first condition and reused across columns, as in the
    # original: the point of the row is to compare conditions on the SAME
    # coordinate, which a per-column choice would quietly break.
    if component_idx is None:
        component_idx = _find_best_component_idx(df, dataset_id, cond_names[0])

    groups = _normalise_methods(methods)
    folder = os.path.join(base_data_dir, "closed_loop")
    dt = 1.0 / float(base_freq.replace("Hz", ""))
    d_name = (
        f"spiraling_babbling_{base_freq.replace('Hz','_Hz')}"
        f"_test_trajectory_{dataset_id}_mvel0.5"
    )

    # The reference recording depends on neither the condition nor the fault
    # suffix, so it is loaded ONCE instead of per column. That is what makes the
    # shared x-axis correct by construction: every panel is the same trajectory
    # on the same time base, not two that merely happen to line up.
    ref_p = os.path.join(folder, f"{d_name}.npz")
    if not os.path.exists(ref_p):
        print(f"[tracking-grid] reference not found: {ref_p}; skipped.")
        return
    pos_ref = transform(np.load(ref_p, allow_pickle=True))
    time = np.arange(len(pos_ref)) * dt

    # One actuation row per motor, or all motors stacked in a single row.
    if motor_rows and motor_indices is None:
        sample = df[df["Traj"] == dataset_id]["motor_pos"].dropna()
        motor_indices = list(range(sample.iloc[0].shape[1])) if len(sample) else [0]
    n_act = len(motor_indices) if (motor_rows and motor_indices) else 1
    # Row-for-row: the two-row layout's height is what one signal row plus one
    # tracking row costs, so every extra motor row costs the same again.
    fig, axes = plt.subplots(
        n_act + 1,
        len(conds),
        figsize=(COL_2, TRACK_FIG_H * (n_act + 1) / 2.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    # Which actuation row a motor index draws into. Without motor_rows they all
    # share row 0, which keeps the single-row layout on exactly the old path.
    act_row = (
        {int(m): k for k, m in enumerate(motor_indices)} if motor_rows else {}
    )

    col_tags = []
    for col, cond_name in enumerate(cond_names):
        act_axes, ax2 = axes[:n_act, col], axes[n_act, col]
        ax1 = act_axes[0]
        cfg = {
            "dir": "closed_loop",
            "cond": "_fault" if "Fault" in cond_name else "",
            "freq": base_freq,
            "state": cond_name.split()[1],
        }
        cond_rows = df[(df["Traj"] == dataset_id) & (df["Cond"] == cond_name)]

        if "Fault" in cond_name:
            for ax in list(act_axes) + [ax2]:
                ax.axvspan(
                    FAULT_START_STEP * dt,
                    FAULT_END_STEP * dt,
                    color="red",
                    alpha=0.08,
                    label="Fault",
                    zorder=1,
                )

        # ---- row 0: actuation ------------------------------------------------
        ref_act = cond_rows["ref_act_array"].iloc[0] if not cond_rows.empty else None
        if ref_act is not None:
            nr = min(len(time), len(ref_act))
            midx = (
                motor_indices if motor_indices is not None else range(ref_act.shape[1])
            )
            for mi in midx:
                act_axes[act_row.get(int(mi), 0)].plot(
                    time[:nr],
                    MOTOR_REVOLUTION_PER_STEP_RAD * ref_act[:nr, mi],
                    "k--",
                    linewidth=1.5,
                    label=f"Ref M{mi+1}",
                    zorder=10,
                )

        for label, grp in groups.items():
            color = COLOR_MAP.get(grp[0], "grey")
            acts = [
                cond_rows[cond_rows["Method"] == m]["motor_pos"].iloc[0]
                for m in grp
                if not cond_rows[cond_rows["Method"] == m].empty
                and cond_rows[cond_rows["Method"] == m]["motor_pos"].iloc[0] is not None
            ]
            if not acts:
                continue
            nt = min(len(time), min(len(a) for a in acts))
            midx = (
                motor_indices if motor_indices is not None else range(acts[0].shape[1])
            )
            if len(acts) == 1:
                for mi in midx:
                    act_axes[act_row.get(int(mi), 0)].plot(
                        time[:nt],
                        MOTOR_REVOLUTION_PER_STEP_RAD * acts[0][:nt, mi],
                        color=color,
                        linewidth=1.2,
                        alpha=0.7,
                    )
                ax1.plot([], [], color=color, label=f"{label}")
            else:
                st = np.stack([a[:nt] for a in acts], axis=0)
                for mi in midx:
                    act_axes[act_row.get(int(mi), 0)].plot(
                        time[:nt],
                        MOTOR_REVOLUTION_PER_STEP_RAD * np.median(st[:, :, mi], 0),
                        color=color,
                        linewidth=1.0,
                        alpha=0.7,
                    )
                ax1.plot([], [], color=color, label=f"{label} (Median)")

        # ---- row 1: configuration-space coordinate --------------------------
        ax2.plot(
            time,
            pos_ref[:, component_idx],
            "k--",
            linewidth=1.5,
            label="Target",
            zorder=10,
        )
        for label, grp in groups.items():
            color = COLOR_MAP.get(grp[0], "grey")
            posns = [
                p
                for p in (_load_pos_for_method(m, d_name, cfg, folder) for m in grp)
                if p is not None
            ]
            if not posns:
                continue
            nt = min(len(time), min(len(p) for p in posns))
            if len(posns) == 1:
                ax2.plot(
                    time[:nt],
                    posns[0][:nt, component_idx],
                    color=color,
                    label=label,
                    linewidth=1.2,
                )
            else:
                st = np.stack([p[:nt, component_idx] for p in posns], axis=0)
                ax2.fill_between(
                    time[:nt], st.min(0), st.max(0), color=color, alpha=0.2
                )
                ax2.plot(
                    time[:nt], np.median(st, 0), color=color, linewidth=1.2, label=label
                )

        # The condition used to be named in the TOP-RIGHT corner of the top
        # panel; it is now folded into that panel's letter tag (see below).
        # The corner it occupied is the one place a legend fits without
        # sitting on a trace, and only one of the two could have it.
        # RANK_COND_LABEL is still the wording, so this figure and the ranking
        # figures cannot drift apart.
        # Only the condition NAME, not the full "10 Hz . Fault": every column
        # here is the same rate, so the frequency is a property of the figure
        # (caption) rather than of the column, and dropping it is what leaves
        # the legend enough of the band to sit in.
        col_tags.append(cond_name.split(maxsplit=1)[1])
        ax2.set_xlabel("Time (s)")

    # Units named once per row: sharey="row" means the inner columns carry no
    # tick labels, so a second copy of the label would have nothing to label.
    #
    # The row plots ONE motor of several by default (motor_indices), so the
    # label names which -- "Motor cmd" alone leaves the reader to guess whether
    # they are looking at one actuator or all four superimposed. Same reason
    # _component_label is derived rather than hardcoded for the bottom row.
    if act_row:
        for mi, k in act_row.items():
            axes[k, 0].set_ylabel(_motor_label([mi]))
    else:
        axes[0, 0].set_ylabel(_motor_label(motor_indices))
    axes[n_act, 0].set_ylabel(f"{_component_label(component_idx)} (mm)")

    # Panel letters INSIDE the axes. Outside (the original's (-0.10, 1.05)) they
    # reserve vertical space above every row, which is exactly what this layout
    # is trying not to spend.
    #
    # Parenthesised lowercase -- (a), (b), ... -- and NOT bare capitals: the
    # columns are labelled with condition names, and a bare "A" next to them
    # reads as a third condition rather than as a panel index.
    #
    # The TOP row's tag carries the condition too ("(b) Fault"): the
    # condition has to be named somewhere, and folding it into the tag on the
    # left frees the top-right corner for the legend without costing a title
    # band. The bottom row inherits its column's condition from the panel
    # above it, so repeating it there would only be noise.
    tags = [f"({c})" for c in "abcdefgh"]
    for i, t in enumerate(col_tags):
        tags[i] = f"{tags[i]}  {t}"
    for ax, lbl in zip(axes.flat, tags):
        ax.text(
            0.015,
            0.95,
            lbl,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
            zorder=20,
        )

    # sharex means the top row carries no tick labels and no x-label, so the
    # gap between rows only has to separate two frames -- the default pad is
    # sized for a row that has both.
    fig.tight_layout(h_pad=0.15)

    # ONE legend for the figure instead of one per panel. Deduped by label (the
    # fault span and the references are drawn on several panels) and ordered
    # reference -> methods -> annotation, so it reads the way the panels do
    # rather than in axes-traversal order.
    #
    # Every reference series collapses to a single "Reference" entry: the
    # commanded motor reference and the target position are drawn in the SAME
    # black dashed style, so listing them separately would put two entries in
    # the legend that a reader cannot tell apart. Which quantity each row's
    # dashed line is, is already settled by that row's y-label.
    seen, H, L = set(), [], []
    for ax in axes.T.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            key = "Reference" if (l.startswith("Ref ") or l == "Target") else l
            if key not in seen:
                seen.add(key)
                H.append(h)
                L.append(key)

    def _leg_rank(lbl):
        if lbl == "Fault":
            return 2
        return 0 if lbl == "Reference" else 1

    order = sorted(range(len(L)), key=lambda i: (_leg_rank(L[i]), i))
    H, L = [H[i] for i in order], [L[i] for i in order]
    if H:
        if TRACK_LEGEND_LOC == "inside":
            _fit_legend_beside_tag(
                axes[TRACK_LEGEND_AX],
                H,
                L,
                fig,
                loc=TRACK_LEGEND_INSIDE_LOC,
                ncol=len(L) if not TRACK_LEGEND_NCOL_INSIDE else min(
                    len(L), TRACK_LEGEND_NCOL_INSIDE
                ),
                # Tighter than the defaults on every axis: the legend shares
                # the headroom band with the panel tag on the left, and a
                # 3-column block at the default padding runs into it.
                framealpha=0.0,
                borderpad=0.2,
                handlelength=1.1,
                handletextpad=0.4,
                columnspacing=0.8,
                labelspacing=0.25,
            )
            # Band opened AFTER the legend exists: its height is what decides
            # how much room the traces have to give up, and it can only be
            # measured once it is drawn.
            _fit_row_headroom(fig, list(axes[0]), pad=TRACK_LEGEND_HEADROOM_PAD)
        elif TRACK_LEGEND_LOC == "right":
            # Re-lay the axes into the left part of the SAME canvas and give the
            # legend the strip that frees up. Anchoring it outside the canvas
            # instead would keep the axes full width and simply make the saved
            # figure wider than COL_2, which LaTeX then scales back down --
            # shrinking the panels anyway, and the fonts with them.
            fig.tight_layout(h_pad=0.15, rect=[0, 0, 1 - TRACK_LEGEND_RIGHT_FRAC, 1])
            fig.legend(
                H,
                L,
                loc="center right",
                ncol=1,
                frameon=False,
                handlelength=1.6,
                labelspacing=0.6,
            )
        else:
            fig.legend(
                H,
                L,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=min(len(L), TRACK_LEGEND_NCOL),
                frameon=False,
            )

    out = path or f"F_tracking_timeseries_{dataset_id}_{base_freq}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
# Per-trajectory ranking views  ──  ADDITIONS (existing code untouched).
# All read ONLY from the real `df`. `Gain` is the % error reduction vs the
# NCL-C baseline computed WITHIN each (Traj, Cond) block (avg_base is per-block
# in collect_comprehensive_data), so trajectory difficulty is divided out and
# summaries of `Gain` isolate the method effect a global mean would wash away.
# ═════════════════════════════════════════════════════════════════════════
# Closed-loop learners shown by default; PC is represented by REF_METHOD.
RANK_DEFAULT_METHODS = ["rnn-uoro", "lwpr", "log_gp", "cerebnet", REF_METHOD]
# Canonical PC-RNN-I red -- the SAME colour used for PC everywhere else (the
# REF_METHOD entry of COLOR_MAP), so the open-loop bars/labels match the
# closed-loop PC line and the rest of the paper.
PC_RED = COLOR_MAP.get(REF_METHOD, "#f80909")
# Display order -> compact names: index in these lists defines the Tx / Cx
# numbers. Defaults follow the paper's canonical order.
RANK_TRAJ_ORDER = list(TARGET_DATASETS)  # -> T0..T<n-1>
RANK_COND_ORDER = list(TARGET_CONDITIONS)  # -> C0..C<n-1>
# Human-readable condition labels shown high above each condition band.
RANK_COND_LABEL = {
    "10Hz Nominal": "10 Hz · Nominal",
    "10Hz Fault": "10 Hz · Fault",
    "20Hz Nominal": "20 Hz · Nominal",
    "20Hz Fault": "20 Hz · Fault",
}
# Very light background tints, one per condition (must NOT overshadow the lines).
_COND_BAND_COLORS = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
    "#8c8c8c",
]


def _gain_pivot(df, methods):
    """(Traj, Cond) x Method table of per-block `Gain`, restricted to `methods`
    and to blocks where every method has a value."""
    methods = [m for m in methods if m in df["Method"].unique()]
    piv = df.pivot_table(index=["Traj", "Cond"], columns="Method", values="Gain")
    piv = piv[methods].dropna(how="any")
    return piv, methods


def _draw_gain_by_traj(
    ax,
    df,
    methods=None,
    traj_order=None,
    cond_order=None,
    legend_loc="top",
    cond_loc="top",
    cond_key=False,
    tick_fs=8,
    label_fs=9,
    cond_fs=8.5,
    legend_fs=8,
):
    """Core renderer for the per-trajectory gain view, drawn into a given `ax`.
    Returns the ordered method list.
      legend_loc : "top" | "bottom" | "inside" | "none"
                   "inside" = one compact row inside the top of the axes.
      cond_key   : if True, append a 'Cx = <condition>' key to the legend so the
                   short Cx labels are self-documenting.
      cond_loc   : "top" | "bottom" | "none"
                   "top"    = two-line "Cx / <name>" above the lines (needs top
                              headroom); "bottom" = short "Cx" in a thin strip at
                              the bottom (cheap on vertical space).
    Font sizes are explicit so the composite renders at paper-text size."""
    methods = list(methods) if methods is not None else list(RANK_DEFAULT_METHODS)
    traj_order = list(traj_order) if traj_order is not None else list(RANK_TRAJ_ORDER)
    cond_order = list(cond_order) if cond_order is not None else list(RANK_COND_ORDER)
    piv, methods = _gain_pivot(df, methods)
    if piv.empty or len(methods) < 1:
        print("[gain_by_traj] not enough complete blocks; skipped.")
        return methods

    t_idx = {t: i for i, t in enumerate(traj_order)}
    c_idx = {c: i for i, c in enumerate(cond_order)}
    ordered = sorted(
        piv.index, key=lambda b: (c_idx.get(b[1], 99), t_idx.get(b[0], 99))
    )
    piv = piv.loc[ordered]
    x = np.arange(len(ordered))
    labels = [f"T{t_idx.get(t, '?')}-C{c_idx.get(c, '?')}" for (t, c) in ordered]

    # Consecutive condition groups (for bands + top labels): [cond, x0, x1].
    groups = []
    prev = object()
    for i, (_t, c) in enumerate(ordered):
        if c != prev:
            groups.append([c, i, i])
            prev = c
        else:
            groups[-1][2] = i

    for cond, x0, x1 in groups:
        ci = c_idx.get(cond, 0)
        ax.axvspan(
            x0 - 0.5,
            x1 + 0.5,
            lw=0,
            zorder=0,
            alpha=0.07,
            color=_COND_BAND_COLORS[ci % len(_COND_BAND_COLORS)],
        )
        if x0 > 0:
            ax.axvline(x0 - 0.5, color="0.8", lw=0.6, zorder=0)

    for m in methods:
        # OL-C is a NEGATIVE-gain reference (open loop is worse than NCL-C) -- draw
        # it dashed and slightly thicker so it reads as a baseline, not a learner.
        is_ref = m == "open_loop"
        ax.plot(
            x,
            piv[m].values.astype(float),
            ("--o" if is_ref else "-o"),
            color=COLOR_MAP.get(m, "black"),
            lw=(1.8 if is_ref else 1.6),
            ms=4.0,
            alpha=0.95,
            label=_leg_short(m),
            zorder=(3 if is_ref else 4),
        )

    # Headroom: reserve only what the chosen label/legend placements need.
    _inside_pad = (0.30 if cond_key else 0.18) if legend_loc == "inside" else 0.0
    top_pad = (0.24 if cond_loc == "top" else 0.0) + _inside_pad
    bot_pad = 0.10 if cond_loc == "bottom" else 0.0
    ymin, ymax = ax.get_ylim()
    rng = ymax - ymin
    ax.set_ylim(ymin - bot_pad * rng, ymax + top_pad * rng)

    for cond, x0, x1 in groups:
        ci = c_idx.get(cond, 0)
        xc = 0.5 * (x0 + x1)
        col = _COND_BAND_COLORS[ci % len(_COND_BAND_COLORS)]
        if cond_loc == "top":
            ax.annotate(
                f"C{ci}\n{RANK_COND_LABEL.get(cond, cond)}",
                xy=(xc, 0.99),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=cond_fs,
                fontweight="bold",
                color=col,
            )
        elif cond_loc == "bottom":
            ax.annotate(
                f"C{ci}",
                xy=(xc, 0.025),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=cond_fs,
                fontweight="bold",
                color=col,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=90)
    ax.set_xlabel(
        "Trajectory–condition block  (grouped by condition)", fontsize=label_fs
    )
    ax.set_ylabel(
        "Error reduction\nvs baseline controller (NCL-C) (%)", fontsize=label_fs
    )
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.margins(x=0.01)

    # Method handles, plus an optional 'Cx = <condition>' key (band-colour swatches).
    m_handles, m_labels = ax.get_legend_handles_labels()
    extra_h, extra_l = [], []
    if cond_key:
        import matplotlib.patches as mpatches

        present = sorted({c for (c, _0, _1) in groups}, key=lambda c: c_idx.get(c, 99))
        for c in present:
            ci = c_idx.get(c, 0)
            extra_h.append(
                mpatches.Patch(
                    facecolor=_COND_BAND_COLORS[ci % len(_COND_BAND_COLORS)],
                    alpha=0.55,
                    edgecolor="none",
                )
            )
            # compact form for the legend (full names stay in the table caption)
            _cl = (
                RANK_COND_LABEL.get(c, c)
                .replace(" · ", " ")
                .replace("Nominal", "nom.")
                .replace("Fault", "fault")
            )
            extra_l.append(f"C{ci}: {_cl}")
    handles, leglabels = m_handles + extra_h, m_labels + extra_l

    if legend_loc == "top":
        ax.legend(
            handles,
            leglabels,
            fontsize=legend_fs,
            framealpha=0.85,
            ncol=len(methods),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.10),
        )
    elif legend_loc == "bottom":
        ax.legend(
            handles,
            leglabels,
            fontsize=legend_fs,
            framealpha=0.85,
            ncol=len(methods),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
        )
    elif legend_loc == "inside":
        ax.legend(
            handles,
            leglabels,
            fontsize=legend_fs,
            framealpha=0.92,
            ncol=len(methods),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            borderaxespad=0.2,
            borderpad=0.3,
            columnspacing=0.9,
            handletextpad=0.3,
            handlelength=1.4,
            labelspacing=0.3,
        )
    return methods




# ── Per-trajectory RAW Euclidean tracking error (mm) ──────────────────────
# Same x layout/philosophy as the gain plot ((Traj, Cond) blocks grouped by
# condition, Tx-Cx ticks, faint condition bands + Cx labels), but the y is the
# absolute mean Euclidean tracking error in mm -- so the plot shows the actual
# error each method incurs per test, not its relative gain. NCL-C and OL-C
# (the references) are included by default as dashed lines (they are the
# baseline floor and the no-learning envelope), and can be dropped via the
# `methods` argument. `log_y=True` switches to a log y if OL-C compresses the
# learners' band; linear is the paper default.
RAW_ERR_DEFAULT_METHODS = [
    "open_loop",
    "baseline",
    "rnn-uoro",
    "lwpr",
    "log_gp",
    "cerebnet",
    REF_METHOD,
]


def _draw_error_by_traj(
    ax,
    df,
    methods=None,
    traj_order=None,
    cond_order=None,
    legend_loc="top",
    cond_loc="top",
    cond_key=False,
    log_y=False,
    tick_fs=8,
    label_fs=9,
    cond_fs=8.5,
    legend_fs=8,
):
    """Per-block mean Euclidean tracking error (mm), in the same layout as
    `_draw_gain_by_traj`. NCL-C ('baseline') and OL-C ('open_loop') are drawn
    dashed when present in `methods`; every other method is a solid coloured
    line in COLOR_MAP. See `_draw_gain_by_traj` for the layout-knob semantics
    (legend_loc / cond_loc / cond_key)."""
    methods = list(methods) if methods is not None else list(RAW_ERR_DEFAULT_METHODS)
    traj_order = list(traj_order) if traj_order is not None else list(RANK_TRAJ_ORDER)
    cond_order = list(cond_order) if cond_order is not None else list(RANK_COND_ORDER)
    methods = [m for m in methods if m in df["Method"].unique()]
    piv = df.pivot_table(
        index=["Traj", "Cond"], columns="Method", values="Mean_Euclidean"
    )
    piv = piv[methods].dropna(how="any")
    if piv.empty or len(methods) < 1:
        print("[error_by_traj] not enough complete blocks; skipped.")
        return methods

    t_idx = {t: i for i, t in enumerate(traj_order)}
    c_idx = {c: i for i, c in enumerate(cond_order)}
    ordered = sorted(
        piv.index, key=lambda b: (c_idx.get(b[1], 99), t_idx.get(b[0], 99))
    )
    piv = piv.loc[ordered]
    x = np.arange(len(ordered))
    labels = [f"T{t_idx.get(t, '?')}-C{c_idx.get(c, '?')}" for (t, c) in ordered]

    # Consecutive condition groups (for bands + labels): [cond, x0, x1].
    groups, prev = [], object()
    for i, (_t, c) in enumerate(ordered):
        if c != prev:
            groups.append([c, i, i])
            prev = c
        else:
            groups[-1][2] = i

    for cond, x0, x1 in groups:
        ci = c_idx.get(cond, 0)
        ax.axvspan(
            x0 - 0.5,
            x1 + 0.5,
            lw=0,
            zorder=0,
            alpha=0.07,
            color=_COND_BAND_COLORS[ci % len(_COND_BAND_COLORS)],
        )
        if x0 > 0:
            ax.axvline(x0 - 0.5, color="0.8", lw=0.6, zorder=0)

    for m in methods:
        # NCL-C / OL-C are references -> dashed and slightly thicker, so the
        # reader can tell at a glance which lines are the floor/no-learning.
        is_ref = m in ("baseline", "open_loop")
        ax.plot(
            x,
            piv[m].values.astype(float),
            ("--o" if is_ref else "-o"),
            color=COLOR_MAP.get(m, "black"),
            lw=(1.8 if is_ref else 1.6),
            ms=4.0,
            alpha=0.95,
            label=_leg_short(m),
            zorder=(3 if is_ref else 4),
        )

    if log_y:
        ax.set_yscale("log")

    # Headroom for top labels / inside legend; bottom strip for short Cx labels.
    _inside_pad = (0.30 if cond_key else 0.18) if legend_loc == "inside" else 0.0
    top_pad = (0.24 if cond_loc == "top" else 0.0) + _inside_pad
    bot_pad = 0.10 if cond_loc == "bottom" else 0.0
    ymin, ymax = ax.get_ylim()
    if log_y:
        # On log y, expand multiplicatively (additive shifts can pull the floor below 0).
        ax.set_ylim(ymin * (1.0 - 0.7 * bot_pad), ymax * (1.0 + 0.7 * top_pad))
    else:
        rng = ymax - ymin
        ax.set_ylim(ymin - bot_pad * rng, ymax + top_pad * rng)

    for cond, x0, x1 in groups:
        ci = c_idx.get(cond, 0)
        xc = 0.5 * (x0 + x1)
        col = _COND_BAND_COLORS[ci % len(_COND_BAND_COLORS)]
        if cond_loc == "top":
            ax.annotate(
                f"C{ci}\n{RANK_COND_LABEL.get(cond, cond)}",
                xy=(xc, 0.99),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=cond_fs,
                fontweight="bold",
                color=col,
            )
        elif cond_loc == "bottom":
            ax.annotate(
                f"C{ci}",
                xy=(xc, 0.025),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=cond_fs,
                fontweight="bold",
                color=col,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=90)
    ax.set_xlabel(
        "Trajectory–condition block  (grouped by condition)", fontsize=label_fs
    )
    ax.set_ylabel(
        (
            "Mean Euclidean error\n[mm] (log)"
            if log_y
            else "Mean Euclidean\ntracking error (mm)"
        ),
        fontsize=label_fs,
    )
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.margins(x=0.01)

    # Method handles + optional Cx-condition key (band-colour swatches).
    m_handles, m_labels = ax.get_legend_handles_labels()
    extra_h, extra_l = [], []
    if cond_key:
        import matplotlib.patches as mpatches

        present = sorted({c for (c, _0, _1) in groups}, key=lambda c: c_idx.get(c, 99))
        for c in present:
            ci = c_idx.get(c, 0)
            extra_h.append(
                mpatches.Patch(
                    facecolor=_COND_BAND_COLORS[ci % len(_COND_BAND_COLORS)],
                    alpha=0.55,
                    edgecolor="none",
                )
            )
            _cl = (
                RANK_COND_LABEL.get(c, c)
                .replace(" · ", " ")
                .replace("Nominal", "nom.")
                .replace("Fault", "fault")
            )
            extra_l.append(f"C{ci}: {_cl}")
    handles, leglabels = m_handles + extra_h, m_labels + extra_l

    if legend_loc == "top":
        ax.legend(
            handles,
            leglabels,
            fontsize=legend_fs,
            framealpha=0.85,
            ncol=max(1, len(methods)),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.10),
        )
    elif legend_loc == "bottom":
        ax.legend(
            handles,
            leglabels,
            fontsize=legend_fs,
            framealpha=0.85,
            ncol=max(1, len(methods)),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
        )
    elif legend_loc == "inside":
        ax.legend(
            handles,
            leglabels,
            fontsize=legend_fs,
            framealpha=0.92,
            ncol=max(1, len(methods)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            borderaxespad=0.2,
            borderpad=0.3,
            columnspacing=0.9,
            handletextpad=0.3,
            handlelength=1.4,
            labelspacing=0.3,
        )
    return methods




# ── Open-loop performance panel (multi-seed-aware) ─────────────────────────
# Path(s) to the pickle(s) produced by run_openloop_sweep.py. The open-loop
# row of the composite reads the MULTI-SEED result (mean +/- across-seed std)
# from here. This MUST be set (or openloop_data passed explicitly): if it is
# None or the file is missing, fig_open_closed_2row RAISES rather than silently
# falling back to the OPENLOOP_RMSE frequency sweep.
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# One pickle per open-loop dataset: the same five test trajectories as the
# closed-loop study (TARGET_DATASETS), at both control frequencies. Built from
# TARGET_DATASETS so the two halves cannot drift apart -- add a trajectory there
# and it is picked up here, provided run_openloop_sweep.py has produced it.
OPENLOOP_FREQS_HZ = [10, 20]
_OPENLOOP_DATA_DIR = os.path.join(
    _ROOT, "data", "open_loop"
)


def _openloop_recordings():
    """[(traj, npz, pkl)] -- one recording per TEST TRAJECTORY, the first listed.

    The (trajectory x frequency) grid does not give equal evidence per
    trajectory: load_robot_data ignores `freq` for spiraling_robot, and the
    10 Hz and 20 Hz source files are byte-identical for v5, v19 and v23 while
    genuinely differing for v21 and v35. Taking every distinct file would weight
    v21 and v35 twice as heavily as the rest. One recording per trajectory makes
    the trajectory the unit of analysis and weights all five equally. Mirrors
    run_multiseed_openloop._one_per_trajectory.

    The source .npz is carried alongside the results pickle so that anything
    describing the recording itself (its length, see _openloop_samples) is read
    from the same file the multi-seed run was given, not re-derived."""
    seen, out = set(), []
    for hz in OPENLOOP_FREQS_HZ:
        for traj in TARGET_DATASETS:
            if traj in seen:
                continue
            stem = f"spiraling_babbling_{hz}_Hz_test_trajectory_{traj}_mvel0.5"
            npz = os.path.join(_OPENLOOP_DATA_DIR, stem + ".npz")
            if not os.path.exists(npz):
                continue
            seen.add(traj)
            out.append(
                (
                    traj,
                    npz,
                    os.path.join(
                        _OPENLOOP_DATA_DIR,
                        f"results_multiseed_spiraling_robot_{stem}.pkl",
                    ),
                )
            )
    return out


def _openloop_pickles():
    """The results pickles of _openloop_recordings, in the same order."""
    return [pkl for _traj, _npz, pkl in _openloop_recordings()]


def _openloop_samples():
    """(n_min, n_max, n_total) samples the open-loop column is scored over, or
    None if a recording cannot be read.

    run_method predicts and updates once per sample of the recording it is
    given, so the recording length IS the number of scored one-step
    predictions. Read from the source .npz rather than the results pickle: the
    two agree by construction and the .npz is ~20 MB cheaper to open."""
    ns = []
    for _traj, npz, _pkl in _openloop_recordings():
        try:
            with np.load(npz, allow_pickle=True) as z:
                key = "total_time_cycle" if "total_time_cycle" in z.files else "pos_array"
                ns.append(int(np.asarray(z[key]).shape[0]))
        except Exception as _e:  # missing/corrupt recording -> caption omits it
            print(f"[warn] open-loop sample count unavailable ({npz}: {_e}).")
            return None
    if not ns:
        return None
    return min(ns), max(ns), int(sum(ns))


OPENLOOP_MULTISEED_PKL = _openloop_pickles()


def _openloop_provenance():
    """(n_recordings, n_trajectories) actually behind the open-loop column.

    These are NOT trajectories x frequencies: duplicate source files are dropped
    (see _openloop_pickles), so most trajectories contribute one recording and
    only those with genuinely distinct 10/20 Hz data contribute two."""
    import re

    trajs = set()
    for f in OPENLOOP_MULTISEED_PKL:
        m = re.search(r"trajectory_(v\d+)_", os.path.basename(f))
        if m:
            trajs.add(m.group(1))
    return len(OPENLOOP_MULTISEED_PKL), len(trajs)



# Total height (inches) of the consolidated open+closed 2-row figure. Single
# knob to trim vertical space: lower = tighter. With bbox_inches="tight" on
# save and the adaptive open-loop y-limit, the figure keeps only the room
# needed for labels and the legend.
OPEN_CLOSED_FIG_H = 5.3


def load_openloop_multiseed(paths):
    """Read run_multiseed_openloop pickle(s) and return
    {display_method: {"mean": mm, "std": mm}} from the across-seed aggregate.
    RMSE in the pickle is in metres; converted to mm here. Multiple datasets
    are pooled by averaging their per-method means/stds.

    Strict: raises FileNotFoundError if any path is missing and RuntimeError if
    no across-seed results are found. Never returns a silent/empty fallback."""
    import pickle

    if isinstance(paths, str):
        paths = [paths]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Open-loop multi-seed pickle(s) not found: "
            + repr(missing)
            + ". Run run_openloop_sweep.py first, or fix OPENLOOP_MULTISEED_PKL."
        )
    acc = {}
    runs = {}  # raw label -> {"per_ds_ok": [int,...], "failed": int, "unknown": bool}
    for p in paths:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        for label, a in blob.get("aggregate", {}).items():
            # --- run audit (count BEFORE the across-seed filter, so an all-failed
            #     method with no mean is still reported as 0 ok / N failed) -------
            ok = a.get("n_seeds_ok")
            if ok is None and a.get("mean_rmse_per_seed") is not None:
                ok = int(np.atleast_1d(np.asarray(a["mean_rmse_per_seed"])).size)
            rec = runs.setdefault(
                label, {"per_ds_ok": [], "failed": 0, "unknown": False}
            )
            rec["failed"] += int(a.get("n_failed", 0) or 0)
            if ok is None:
                rec["unknown"] = True
            else:
                rec["per_ds_ok"].append(int(ok))
            # --- data accumulation (only methods with a usable across-seed mean) -
            if "mean_rmse_across_seeds" not in a:
                continue
            disp = label  # keep the raw label; _draw_openloop_panel renders it via _disp_name
            acc.setdefault(disp, {"mean": [], "std": []})
            acc[disp]["mean"].append(
                a["mean_rmse_across_seeds"] * METERS_TO_MILLIMETERS
            )
            acc[disp]["std"].append(a["std_rmse_across_seeds"] * METERS_TO_MILLIMETERS)

    # Audit: number of runs (seeds) actually found per method, from the data above.
    print(
        f"\n[open-loop] multi-seed runs found per method "
        f"(across {len(paths)} pickle(s)):"
    )
    for label in sorted(runs, key=_disp_name):
        rec = runs[label]
        name = _disp_name(label)
        if rec["unknown"]:
            print(f"    {name:18s}: run count not stored in aggregate")
            continue
        per, nds = rec["per_ds_ok"], len(rec["per_ds_ok"])
        tot = sum(per)
        spread = f" ({'+'.join(map(str, per))} over {nds} datasets)" if nds > 1 else ""
        tail = f", {rec['failed']} failed" if rec["failed"] else ""
        print(f"    {name:18s}: {tot} run(s){spread}{tail}")

    if not acc:
        raise RuntimeError(
            "No across-seed open-loop results found in "
            + repr(paths)
            + " (expected 'aggregate' entries with 'mean_rmse_across_seeds')."
        )
    return {
        m: {"mean": float(np.mean(v["mean"])), "std": float(np.mean(v["std"]))}
        for m, v in acc.items()
    }


def _disp_name(label):
    """Professional figure label for ANY implementation label. A trailing
    '_N<digits>' (or an embedded '(N=<digits>)') becomes a ' (N=x)' suffix; the
    base is mapped to a clean name -- the PC family to 'PC', every other base to
    its DISPLAY_NAMES name (first line, without the '(ours)' tag). Examples:
    'temporal_pc-full-pi10_N5' -> 'PC (N=5)', 'cerebnet_N5' -> 'CerebNet (N=5)',
    'cerebnet' -> 'CerebNet', 'rnn-rtrl' -> 'RTRL', 'lwpr' -> 'LWPR'."""
    s = str(label)
    base, n = s, None
    if "_N" in s and s.rsplit("_N", 1)[-1].isdigit():  # 'base_N5'
        base, n = s.rsplit("_N", 1)
    elif "N=" in s:  # 'PC-RNN-I (N=5)'
        digits = "".join(c for c in s.rsplit("N=", 1)[-1] if c.isdigit())
        n = digits or None
        base = s.split("(", 1)[0].strip()
    if base.startswith("temporal_pc") or base.startswith("PC"):
        name = "PC"
    elif base in DISPLAY_NAMES:
        name = DISPLAY_NAMES[base].split("\n")[0].replace(" (ours)", "").strip()
    else:
        name = _leg_short(base)
    return f"{name} (N={n})" if n else name


def _draw_openloop_panel(ax, data, clip=40.0, tick_fs=8, label_fs=9, annot_fs=8.0):
    """Open-loop prediction-RMSE bars (mean) with a spread whisker, log-y, PC
    highlighted. `data` is {method: [values]} (whisker = min..max) or
    {method: {"mean":, "std":}} (whisker = mean +/- std). `data` is REQUIRED --
    there is no silent default (pass OPENLOOP_RMSE explicitly to use the
    frequency sweep)."""
    import matplotlib.ticker as mt

    if not data:
        raise ValueError(
            "_draw_openloop_panel: no open-loop data provided. Pass the multi-seed "
            "result (or OPENLOOP_RMSE explicitly). No silent fallback."
        )

    def _stat(v):
        if isinstance(v, dict):
            m, s = float(v["mean"]), float(v.get("std", 0.0))
            return m, max(m - s, 1e-6), m + s
        arr = np.atleast_1d(np.asarray(v, float))
        return float(arr.mean()), float(arr.min()), float(arr.max())

    # Drop any method whose value is NaN/Inf (or unparseable) -- do not plot it.
    stats, skipped = {}, []
    for m, v in data.items():
        try:
            triple = _stat(v)
        except (TypeError, ValueError, KeyError):
            skipped.append(m)
            continue
        if not all(np.isfinite(x) for x in triple):
            skipped.append(m)
            continue
        stats[m] = triple
    if skipped:
        print(
            "  [open-loop] skipped (NaN/Inf): "
            + ", ".join(_disp_name(s) for s in skipped)
        )
    if not stats:
        raise ValueError(
            "_draw_openloop_panel: no finite open-loop data to plot (all methods "
            "were NaN/Inf)."
        )
    # Order by the DISPLAYED value (rounded to the printed 1 decimal), not the
    # raw float, so methods that print the same number are treated as tied.
    # Within a display-tie, UORO is placed first (it stands in for the RNN family,
    # which all cluster at the same open-loop RMSE); the rest keep their input
    # (harness) order. Distinct printed values are unaffected.
    _idx = {m: i for i, m in enumerate(stats)}  # input (harness) order
    order = sorted(
        stats,
        key=lambda m: (
            float(f"{stats[m][0]:.1f}"),  # 1) printed value, lowest first
            0 if _disp_name(m) == "UORO" else 1,  # 2) UORO leads any display-tie
            _idx[m],
        ),
    )  # 3) else keep input order
    is_pc = lambda m: (
        str(m).startswith("PC-RNN-I")
        or str(m).startswith("PC ")
        or str(m).startswith("PC (")
        or "temporal_pc" in str(m)
    )
    x = np.arange(len(order))
    for xi, m in zip(x, order):
        mean, lo, hi = stats[m]
        col = PC_RED if is_pc(m) else "0.6"
        ax.bar(
            xi, min(mean, clip), color=col, edgecolor="k", lw=0.4, width=0.7, zorder=2
        )
        ax.plot([xi, xi], [min(lo, clip), min(hi, clip)], color="k", lw=0.9, zorder=3)
        if hi > clip:
            ax.scatter(
                xi, clip, marker="^", s=20, color=col, edgecolor="k", lw=0.3, zorder=4
            )
        import matplotlib.patheffects as pe

        lblcol = PC_RED if is_pc(m) else "#222222"
        ax.annotate(
            f"{mean:.1f}",
            (xi, min(hi, clip)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annot_fs,
            fontweight="bold",
            color=lblcol,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )
    ax.set_yscale("log")
    # Tight y-limits: only as tall as the drawn bars/whiskers (plus headroom for
    # the value labels), instead of a fixed range that leaves the top half empty.
    lows = [stats[m][1] for m in order]
    highs = [min(stats[m][2], clip) for m in order]
    if any(stats[m][2] > clip for m in order):
        highs.append(clip)  # room for the off-scale arrow at clip
    ymin = max(0.5, (min(lows) if lows else 1.0) * 0.85)
    ymax = (
        max(highs) if highs else clip
    ) * 2.0  # 2.0x (log) headroom for the larger value labels
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_disp_name(m) for m in order], rotation=40, ha="right", fontsize=tick_fs
    )
    for tl, m in zip(ax.get_xticklabels(), order):
        if is_pc(m):
            tl.set_color(PC_RED)
            tl.set_fontweight("bold")
    ax.set_ylabel("Prediction RMSE\n[mm] (log)", fontsize=label_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", which="both", color="#eee", lw=0.5)
    ax.margins(x=0.01)




def _ranking_rows(df, methods, ref):
    """Per-method summary for the ranking table: returns ordered rows (best
    median gain first), the block count, the win audit, and the (Traj, Cond)
    index of the blocks actually used -- the last so callers can describe the
    design (how many trajectories, how many conditions) from the real data
    instead of hardcoding it into a caption."""
    piv, methods = _gain_pivot(df, methods)
    n_blocks = len(piv)
    winners = piv[methods].idxmax(axis=1) if n_blocks else pd.Series([], dtype=object)
    # Per-block margin of the winner over the runner-up, at FULL precision.
    # This is what lets us tell a real win from a rounding-scale near-tie.
    if n_blocks and len(methods) >= 2:
        _sv = np.sort(piv[methods].values.astype(float), axis=1)  # ascending per row
        _margins = _sv[:, -1] - _sv[:, -2]
    else:
        _margins = np.full(n_blocks, np.inf)
    win_audit = (
        [
            {"block": blk, "winner": winners.loc[blk], "margin": float(mg)}
            for blk, mg in zip(piv.index, _margins)
        ]
        if n_blocks
        else []
    )
    others = [m for m in methods if m != ref]
    pvals = {}
    for m in others:
        d = (piv[m].values - piv[ref].values) if ref in methods else np.array([])
        if len(d) < 1 or np.ptp(d) == 0:
            pvals[m] = 1.0
            continue
        try:
            pvals[m] = scipy_stats.wilcoxon(
                d, alternative="two-sided", zero_method="pratt", mode="exact"
            ).pvalue
        except (ValueError, TypeError):
            try:
                pvals[m] = scipy_stats.wilcoxon(
                    d, alternative="two-sided", zero_method="pratt"
                ).pvalue
            except ValueError:
                pvals[m] = 1.0
    padj = {}
    if others:
        _, adj = _holm([pvals[m] for m in others])
        padj = dict(zip(others, adj))
    rows = []

    # Region mean Euclidean error (mm), per method, across the SAME blocks as the
    # gain pivot so every number lines up row-to-row. `per_marker_mean_error` is
    # [seg1..seg4] base->tip in mm. region="body" -> mean of all 4 segments;
    # region="tip" -> segment 4 (index 3). Returns a {block: value} dict.
    def _region_block_err(meth, region):
        sub = df[df["Method"] == meth]
        if sub.empty or "per_marker_mean_error" not in sub.columns:
            return {}
        sub = sub.set_index(["Traj", "Cond"])
        out = {}
        for blk in piv.index:  # same (Traj, Cond) blocks as the pivot
            if blk not in sub.index:
                continue
            a = sub.loc[blk, "per_marker_mean_error"]
            if hasattr(a, "iloc"):  # de-dup if multiple rows per block
                a = a.iloc[0]
            if a is None or len(a) < 4:
                continue
            a = np.asarray(a, float)
            v = float(np.mean(a)) if region == "body" else float(a[3])
            if np.isfinite(v):
                out[blk] = v
        return out

    def _region_mean(meth, region):
        d = _region_block_err(meth, region)
        return float(np.mean(list(d.values()))) if d else np.nan

    # Per-region gain vs NCL-C, computed block-by-block with a region-RESTRICTED
    # baseline (so the % is self-consistent with that region's error), then
    # averaged over blocks: gain_blk = (base_blk - method_blk)/base_blk * 100.
    base_body = _region_block_err("baseline", "body")
    base_tip = _region_block_err("baseline", "tip")

    def _region_gain(meth, region):
        if meth == "baseline":
            return 0.0
        base = base_body if region == "body" else base_tip
        meth_e = _region_block_err(meth, region)
        gains = []
        for blk, be in base.items():
            if be and be > 0 and blk in meth_e:
                gains.append((be - meth_e[blk]) / be * 100.0)
        return float(np.mean(gains)) if gains else np.nan

    # Fraction of time-steps below TAU_MM, computed exactly as the CDF panel (b):
    # concatenate every per-step error across this method's blocks and take the
    # mean of (err < tau). err_per_step_array is metric A per-step (mean of the 4
    # marker distances), so this column is consistent with the rest of the table.
    TAU_MM = 10.0

    def _pct_under(meth):
        sub = df[df["Method"] == meth]
        if sub.empty or "err_per_step_array" not in sub.columns:
            return np.nan
        arrs = [
            np.asarray(a, float)
            for a in sub["err_per_step_array"].values
            if a is not None and np.ndim(a) > 0 and len(a)
        ]
        if not arrs:
            return np.nan
        allerr = np.concatenate(arrs)
        allerr = allerr[np.isfinite(allerr)]
        return float(np.mean(allerr < TAU_MM) * 100.0) if len(allerr) else np.nan

    for m in methods:
        g = piv[m].values.astype(float)
        delta = (
            np.nan if m == ref else float(np.median(piv[ref].values - piv[m].values))
        )
        rows.append(
            {
                "method": m,
                "mean": float(np.mean(g)),
                "median": float(np.median(g)),
                "min": float(np.min(g)),
                "max": float(np.max(g)),
                "wins": int((winners == m).sum()) if n_blocks else 0,
                "delta_vs_ref": delta,
                "p_holm": (np.nan if m == ref else padj.get(m, np.nan)),
                "body_gain": _region_gain(m, "body"),
                "tip_gain": _region_gain(m, "tip"),
                "body_mm": _region_mean(m, "body"),
                "tip_mm": _region_mean(m, "tip"),
                "pct_under": _pct_under(m),
            }
        )
    # Order by number of per-test wins (Best), highest first; ties broken by the
    # DISPLAYED whole-body gain (body_gain), so the ordering matches the column the
    # reader sees rather than the canonical Mean_Euclidean gain.
    rows.sort(
        key=lambda r: (
            r["wins"] if r["wins"] is not None else -1,
            r["body_gain"] if np.isfinite(r["body_gain"]) else -1e9,
        ),
        reverse=True,
    )
    # NCL-C reference row: it is the gain baseline, so its gain is 0% by
    # definition and the dispersion / win columns are not meaningful (dashed).
    # Its tip error is a real measured value (same blocks as the pivot). Pinned
    # at the very bottom as a reference, regardless of sort.
    if "baseline" in df["Method"].unique():
        rows.append(
            {
                "method": "baseline",
                "mean": 0.0,
                "median": 0.0,
                "min": np.nan,
                "max": np.nan,
                "wins": None,
                "delta_vs_ref": np.nan,
                "p_holm": np.nan,
                "body_gain": 0.0,
                "tip_gain": 0.0,
                "body_mm": _region_mean("baseline", "body"),
                "tip_mm": _region_mean("baseline", "tip"),
                "pct_under": _pct_under("baseline"),
                "is_ref_row": True,
            }
        )
    return rows, n_blocks, win_audit, piv.index


def _design_phrase(blocks):
    """'5 trajectories $\\times$ 4 conditions' derived from the (Traj, Cond)
    blocks ACTUALLY used, so a caption can never disagree with the data behind
    it. Returns '' when the design is ragged (some trajectory missing some
    condition), because then no single 'A x B' product describes it and
    printing one would be a new inaccuracy -- callers fall back to stating the
    block count alone."""
    n_t = len(blocks.get_level_values("Traj").unique())
    n_c = len(blocks.get_level_values("Cond").unique())
    if n_t * n_c != len(blocks):
        return ""
    t_word = "trajectory" if n_t == 1 else "trajectories"
    c_word = "condition" if n_c == 1 else "conditions"
    return rf"{n_t} {t_word} $\times$ {n_c} {c_word}"


def _cond_key_phrase(blocks):
    """'C0 = 10 Hz . Nominal, ...' for the conditions actually present. The Cx
    number stays the RANK_COND_ORDER position, matching the figures' labels."""
    present = set(blocks.get_level_values("Cond").unique())
    return ", ".join(
        f"C{i} = {RANK_COND_LABEL.get(c, c)}"
        for i, c in enumerate(RANK_COND_ORDER)
        if c in present
    )


def make_ranking_table(df, path="ranking_table.tex", methods=None, ref=REF_METHOD):
    """Compact, intuitive ranking table (booktabs). Rows = methods (best median
    gain first). Columns are plain-language quantities only:
      - Median gain (%)   : typical tracking-error reduction vs NCL-C per test
      - [min, max] (%)    : best- and worst-case gain across the tests
      - Best (k/N)        : on how many tests the method had the highest gain
    (The paired Delta-vs-PC and Wilcoxon p are intentionally omitted; report
    those in the text/appendix if a reviewer wants the significance.) By default
    the table includes the non-PC baselines plus EVERY PC-RNN-I variant in `df`."""
    if methods is None:

        def _pc_n(m):
            tail = "".join(c for c in str(m).split("_N")[-1] if c.isdigit())
            return int(tail) if tail else 0

        pcs = sorted(
            (m for m in df["Method"].unique() if str(m).startswith("temporal_pc")),
            key=_pc_n,
        )
        baselines = [
            m for m in RANK_DEFAULT_METHODS if not str(m).startswith("temporal_pc")
        ]
        # OL-C is the no-learning reference: include it if present so the table
        # shows the floor the learners improve from. Its Gain is negative
        # (open loop is worse than NCL-C); it will sort to the bottom by mean.
        extra = ["open_loop"] if "open_loop" in df["Method"].unique() else []
        methods = extra + baselines + pcs
    else:
        methods = list(methods)
    rows, n_blocks, win_audit, blocks = _ranking_rows(df, methods, ref)
    if n_blocks == 0:
        print("[ranking_table] no complete blocks; table skipped.")
        return

    print(f"\nPer-trajectory closed-loop ranking  (n = {n_blocks} tests)")
    print(
        "Gain = % error reduction vs NCL-C. 'Tip' columns = segment 4 only; "
        "the others are whole-body (all 4 segments)."
    )
    hdr = (
        f"{'Method':14s} {'Best':>7s} {'Gain%':>7s} {'TipG%':>7s} "
        f"{'Err(mm)':>9s} {'Tip(mm)':>9s} {'<10mm%':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    def _f1(v, dash="   —"):
        return dash if not np.isfinite(v) else f"{v:.1f}"

    for r in rows:
        best = "—" if r.get("is_ref_row") else f"{r['wins']:>2d}/{n_blocks:<2d}"
        print(
            f"{_leg_short(r['method']):14s} {best:>7s} "
            f"{_f1(r['body_gain']):>7s} {_f1(r['tip_gain']):>7s} "
            f"{_f1(r['body_mm']):>9s} {_f1(r['tip_mm']):>9s} "
            f"{_f1(r['pct_under']):>8s}"
        )

    # Win-margin audit: each per-block winner is decided on FULL-PRECISION gain,
    # so 0.1%-display rounding cannot create a win. The only way a win could be
    # "not really there" is a near-tie that idxmax breaks by column order; flag
    # any block where the winner's lead over the runner-up is below the display
    # resolution so it can be checked on the real data.
    TIE_EPS = 0.10  # percentage points (the table's display resolution)
    close = sorted(
        (a for a in win_audit if a["margin"] < TIE_EPS), key=lambda a: a["margin"]
    )
    print(
        "\nWin-margin audit (per-block winner from full-precision gain; "
        "display rounding does not affect it):"
    )
    if not close:
        print(
            f"  all {n_blocks} per-block wins lead the runner-up by "
            f">= {TIE_EPS:.2f} pp -> none are rounding-scale."
        )
    else:
        for a in close:
            t, c = a["block"]
            print(
                f"  T={t}, C={c}: winner {_leg_short(a['winner'])}, "
                f"lead over 2nd = {a['margin']:.3f} pp  <-- near-tie"
            )
        print(
            f"  {len(close)} of {n_blocks} wins are within {TIE_EPS:.2f} pp "
            f"(rounding-scale) -- verify these are real, not column-order ties."
        )

    cond_key_txt = _cond_key_phrase(blocks)
    # Design phrase is derived from the blocks actually used, never hardcoded.
    design = _design_phrase(blocks)
    # Design only. `cond_key_txt` (C0 = 10 Hz . Nominal, ...) decodes tick
    # labels the FIGURES use; this table has no Cx column, so it is not part of
    # reading it.
    design_txt = f" ({design})" if design else ""

    _tex1 = _fmt1  # one truncated decimal, shared with the other tables

    L = [
        r"\begin{table}[t]",
        r"\centering",
        # Column meanings and nothing else. The old caption also spelled out the
        # C0..C3 condition key, which this table has no Cx column to decode --
        # caption text explaining columns that are not there.
        rf"\caption{{Whole-body (all four markers) and tip (most distal) tracking "
        rf"over {n_blocks} blocks{design_txt}. \emph{{Gain}}: \% reduction in mean "
        rf"Euclidean error vs.\ NCL-C; \emph{{Error}}: that error in mm. "
        rf"$<${int(10)}\,mm: share of time-steps below that whole-body error. "
        rf"\emph{{Best}}: blocks won, and the ordering.}}",
        r"\label{tab:ranking}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Whole body} & \multicolumn{2}{c}{Tip} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        rf"Method & Best & Gain & Error & Gain & Error & $<$10\,mm \\",
        rf" & (/{n_blocks}) & (\%) & (mm) & (\%) & (mm) & (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        name = _leg_short(r["method"])
        if r["method"] == ref:
            name = r"\textbf{" + name + "}"
        if r.get("is_ref_row"):
            L.append(r"\midrule")
            L.append(
                f"{name} & $-$ & {_tex1(r['body_gain'])} & "
                f"{_tex1(r['body_mm'])} & {_tex1(r['tip_gain'])} & "
                f"{_tex1(r['tip_mm'])} & {_tex1(r['pct_under'])} " + r"\\"
            )
        else:
            L.append(
                f"{name} & {r['wins']}/{n_blocks} & {_tex1(r['body_gain'])} & "
                f"{_tex1(r['body_mm'])} & {_tex1(r['tip_gain'])} & "
                f"{_tex1(r['tip_mm'])} & {_tex1(r['pct_under'])} " + r"\\"
            )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nSaved: {path}")


# ═════════════════════════════════════════════════════════════════════════
# TABLE ALTERNATIVE TO fig_open_closed_2row
# ═════════════════════════════════════════════════════════════════════════
# Directory for .tex deliverables. Kept next to this script so the figures and
# the tables sit side by side in the same tree.
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")

# Open-loop labels whose RMSE is statistically indistinguishable are collapsed
# into ONE row, since spending eight rows (or eight bars) on a single result is
# what makes the figure large without making it more informative. UORO is
# excluded from the collapse because it is also a closed-loop method and needs
# its own row. Set group_rnn=False to list every member separately.
_RNN_FAMILY_PREFIX = "rnn-"

# What to do with methods that have NO closed-loop measurement. Each carries a
# single number (Pred. RMSE) and dashes every other column, so as full rows they
# cost a line apiece while leaving ~90% of the table width blank -- eight of
# them on the current data.
#   "inline" -- one wrapped line spanning the full width at the foot of the
#               table. Keeps the fact that the closed-loop pick (UORO) is one of
#               a large tied family, which is the reason they are shown at all,
#               and spends the width the table already has instead of height.
#   "rows"   -- one full row each (the earlier behaviour).
#   "drop"   -- omit them; make_openloop_full_table lists every one of them with
#               its seed spread, so nothing is lost, only relocated.
OPEN_CLOSED_OL_ONLY = "inline"

def _tabcolsep_for(wide, ncol):
    """Column padding that keeps the table body inside its float.

    Measured by compiling, not chosen by eye. Natural width grows by 2*ncol per
    point of \\tabcolsep, so the widest setting that fits depends on how many
    columns there are. Against \\columnwidth (252pt in IEEEtran conference):
    the 10-column body (gains + tracking error) fits at 2pt and overflows by
    1.5pt at 2.5pt; the 7-column body (gains only) still fits at 7pt and
    overflows only at 8pt. Page-wide floats have room to spare and keep 4pt.

    Shrinking the FONT is not an alternative: IEEEtran already sets table
    bodies at \\footnotesize, so \\footnotesize changes the width not at all and
    \\small makes it worse."""
    if wide:
        return "4pt"
    return "2pt" if ncol >= 9 else "5pt"

# Display names whose N-sweep members are left out of the summary table: the
# sweep spends rows without adding a result, so only the base method stays.
# Members are matched as "<base> (N=k)".
DROP_N_SWEEP = ("CerebNet",)


def _is_sweep_member(name, bases):
    return any(str(name).startswith(f"{b} (N=") for b in bases)


def _trunc(v, nd=1):
    """Truncate towards zero to `nd` decimals -- never round up. Reporting only
    the digits that are actually there keeps 13.1988 from reading as 13.2, and
    lets methods that agree to the displayed precision read as the ties they
    are (the online-RNN learners all land on 13.1). The pre-round at 9 decimals
    guards against a binary representation like 2.9999999996 truncating to 2.9."""
    v = float(v)
    if not np.isfinite(v):
        return v
    f = 10.0 ** nd
    return float(np.trunc(round(v * f, 9))) / f


def _fmt1(v, dash="$-$"):
    """One truncated decimal, or `dash` when the value is missing. Shared by
    every table so the same quantity never prints two different ways."""
    v = float(v) if v is not None else np.nan
    return dash if not np.isfinite(v) else f"{_trunc(v, 1):.1f}"


def _tex_rmse(mean, std):
    """'mean $\\pm$ std' in mm for the .tex tables. Mean at 2 dp. The seed spread
    is shown at whatever precision makes it visible (3 dp for the sub-0.01 mm
    spreads of the PC variants), so a small spread is never silently rendered as
    no spread -- a bare mean means the method really is deterministic. Shared by
    both table builders so the same method reads identically in both."""
    if not np.isfinite(mean):
        return "$-$"
    m = _fmt1(mean)
    if not np.isfinite(std) or std <= 0:
        return m
    s = f"{std:.2f}" if std >= 0.005 else f"{std:.3f}"
    return f"{m}\\,$\\pm$\\,{s}"


def _openloop_by_display(openloop_data):
    """{display name: {"mean", "std", "label"}} keyed the same way the
    closed-loop side is keyed, so the two halves can be joined on the name."""
    out = {}
    for label, v in openloop_data.items():
        out[_disp_name(label)] = {
            "mean": float(v["mean"]),
            "std": float(v.get("std", 0.0)),
            "label": label,
        }
    return out


def _cond_freqs_hz(conds):
    """Distinct control frequencies (Hz) named by the condition labels, in
    ascending order. Parsed from the labels themselves ("10Hz Fault" -> 10.0)
    so a caption quoting durations tracks TARGET_CONDITIONS instead of
    hardcoding the two rates the current study happens to use."""
    import re

    out = []
    for c in conds:
        m = re.match(r"\s*([0-9.]+)\s*Hz", str(c))
        if m:
            f = float(m.group(1))
            if f not in out:
                out.append(f)
    return sorted(out)


def _secs_at(n_steps, hz):
    """Duration (s, rounded to the nearest second) of `n_steps` control steps at
    `hz`. Nominal, not measured: the control period is what the reader converts
    with, and the measured cycle time tracks it to <0.5% (see the timing
    table)."""
    return int(np.floor(float(n_steps) / float(hz) + 0.5))


def _traj_length_stats(df, piv):
    """({Traj: n_steps}, n_min, n_max, n_mean, n_per_condition) over the blocks
    the table actually uses.

    Length comes from `err_per_step_array` -- the per-step error series every
    tracking number in the table is averaged over -- so it reports the steps
    that entered the statistics, not the nominal length of the reference
    recording (collect_comprehensive_data aligns run and reference with min(),
    which costs one sample per block).

    The same reference trajectories are replayed at every condition, so the step
    count does not depend on frequency or fault state; if a trajectory ever
    disagreed across its blocks the shortest is reported and a warning printed,
    rather than silently averaging two different designs."""
    trajs = list(dict.fromkeys(piv.index.get_level_values("Traj")))
    conds = set(piv.index.get_level_values("Cond"))
    out = {}
    for t in trajs:
        sub = df[(df["Traj"] == t) & (df["Cond"].isin(conds))]
        ns = sorted(
            {int(len(a)) for a in sub["err_per_step_array"] if a is not None}
        )
        if not ns:
            continue
        if len(ns) > 1:
            print(
                f"  [open_closed_table] trajectory {t} has uneven block lengths "
                f"{ns}; reporting the shortest."
            )
        out[t] = ns[0]
    if not out:
        return {}, None, None, None, None
    vals = list(out.values())
    # Mean over TRAJECTORIES, not over blocks: every trajectory is run once per
    # condition, so the two coincide, but the trajectory is the unit the caption
    # talks about. It sits well above the median (three of the five recordings
    # are the same length and one is ~40% longer), which is why the caption
    # keeps the range alongside it rather than quoting the mean alone.
    return out, min(vals), max(vals), float(np.mean(vals)), int(sum(vals))


def make_open_closed_table(
    df,
    path=None,
    openloop_data=None,
    methods=None,
    both_only=True,
    order_by="gain",
    drop_sweep=DROP_N_SWEEP,
    ranking_cols=True,
    include_baseline=True,
    group_rnn=True,
    wide=True,
    ol_only_mode=None,
    openloop_col=True,
    tabcolsep=None,
):
    """Table replacement for fig_open_closed_2row AND make_ranking_table: ONE
    table carrying both rows of that figure plus the ranking columns. Rows are
    methods; columns are grouped left to right as

      Overall              mean gain (%), blocks won, open-loop pred. RMSE (mm)
      Gain by condition    mean gain (%) within each condition
      Tracking error       whole-body (mm), tip gain (%), tip (mm), % < 10 mm

    with a vertical rule between groups. The two headline quantities sit first,
    immediately after the method name.

    ranking_cols=False drops the Tracking-error group (and the NCL-C row),
    leaving the earlier open/closed-only form. include_baseline=False drops just
    the NCL-C reference row.

    The ranking columns are taken from _ranking_rows on the same method list, so
    they are the SAME numbers make_ranking_table prints. Note that the ranking
    table's whole-body `Gain` is not repeated here: it is equal to the mean of
    the per-block `Gain` (the 'Overall' column) to machine precision.

    both_only=True (default) restricts the table to methods measured in BOTH
    regimes, so every row is complete and the table is a single uninterrupted
    block ordered by mean gain. The methods this leaves out (the CerebNet
    N-sweep and the online-RNN learners other than UORO) are all listed in
    make_openloop_full_table, so no measurement is lost -- only relocated.

    both_only=False restores the two-block form: methods with both regimes
    first, then open-loop-only methods with '$-$' in the closed-loop columns.

    Provenance is the same as the figure and equally strict: the open-loop half
    comes from `openloop_data` or the OPENLOOP_MULTISEED_PKL pickle(s) and
    RAISES if absent; the closed-loop half is the same `Gain` pivot the figure's
    lower row draws, restricted to blocks where every method has a value."""
    if openloop_data is None:
        if not OPENLOOP_MULTISEED_PKL:
            raise RuntimeError(
                "Open-loop multi-seed data is not configured. Set "
                "OPENLOOP_MULTISEED_PKL to the run_multiseed_openloop pickle(s), "
                "or pass openloop_data explicitly."
            )
        openloop_data = load_openloop_multiseed(OPENLOOP_MULTISEED_PKL)

    # Ordering by a quantity the table does not print would leave the caption
    # asserting a sort the reader cannot verify, so the open-loop ordering is
    # only available while the open-loop column is.
    if not openloop_col and order_by == "openloop":
        order_by = "gain"

    ol = _openloop_by_display(openloop_data)
    _n_ol_rec, _n_ol_traj = _openloop_provenance()
    # Methods reported without an across-seed spread. The runner treats these as
    # deterministic and runs them once (DETERMINISTIC_METHODS), so their missing
    # '$\pm$' means "not measured", NOT "verified constant" -- LoG-GP in fact
    # partitions its tree with random.random() and does vary with the seed.
    single_seed = sorted(
        n
        for n, v in ol.items()
        if np.isfinite(v["mean"]) and not (np.isfinite(v["std"]) and v["std"] > 0)
    )
    dropped_sweep = sorted(n for n in ol if _is_sweep_member(n, drop_sweep or ()))
    # Keep their values: the caption justifies the omission from the numbers,
    # so the justification has to be re-derived whenever the data changes.
    dropped_vals = {n: dict(ol[n]) for n in dropped_sweep}
    if dropped_sweep:
        ol = {n: v for n, v in ol.items() if n not in dropped_sweep}

    # Default method set. With both_only (the default) the table carries ONLY
    # methods measured in both regimes, so every row is full and the table needs
    # no second section; OL-C is kept because it is the no-learning closed-loop
    # reference the gain is read against, and it has no predictor by
    # construction (one '$-$' cell, not a section). Methods measured open-loop
    # only are listed in make_openloop_full_table instead, so nothing is lost.
    if methods is None:
        cl_keys = list(df["Method"].unique())
        if both_only:
            cl_by_name = {_leg_short(m): m for m in cl_keys}
            methods = [cl_by_name[n] for n in sorted(ol) if n in cl_by_name]
            if "open_loop" in cl_keys:
                methods.append("open_loop")
        else:
            methods = ["open_loop"] + list(RANK_DEFAULT_METHODS)
    else:
        methods = list(methods)

    piv, methods = _gain_pivot(df, methods)
    n_blocks = len(piv)
    if n_blocks == 0:
        print("[open_closed_table] no complete blocks; table skipped.")
        return
    winners = piv[methods].idxmax(axis=1)

    conds = [c for c in RANK_COND_ORDER if c in piv.index.get_level_values("Cond")]
    # Counted from the blocks actually used, so the caption cannot drift from
    # the data (e.g. if TARGET_DATASETS changes, or a block is dropped).
    n_traj = len(piv.index.get_level_values("Traj").unique())

    # How long a test trajectory is. Reported so any per-step quantity in the
    # table can be read as wall-clock time: the SAME reference trajectories are
    # replayed at every condition, so the step count is shared and only the
    # duration changes with the control rate.
    len_by_traj, n_min, n_max, n_mean, n_steps_cond = _traj_length_stats(df, piv)
    freqs_hz = _cond_freqs_hz(conds)
    ol_len = _openloop_samples()

    # Ranking metrics (whole-body / tip error, %<10mm) come from _ranking_rows on
    # the SAME method list, so its `_gain_pivot` sees the same blocks and every
    # number lines up with the per-condition means computed here. Its `body_gain`
    # is the same quantity as `all` below -- verified equal to machine precision
    # -- so it is NOT emitted as a second column.
    rank_by_key, ref_row = {}, None
    if ranking_cols:
        rrows, _, _, _ = _ranking_rows(df, methods, REF_METHOD)
        for rr in rrows:
            if rr.get("is_ref_row"):
                ref_row = rr  # NCL-C: the gain reference, pinned to the bottom
            else:
                rank_by_key[rr["method"]] = rr

    # ---- block 1: methods present in BOTH halves -------------------------
    joint = []
    for m in methods:
        name = _leg_short(m)
        g = piv[m].values.astype(float)
        o = ol.get(name)
        rk = rank_by_key.get(m, {})
        joint.append(
            {
                "name": name,
                "key": m,
                "rmse": (o["mean"] if o else np.nan),
                "std": (o["std"] if o else np.nan),
                "per_cond": [
                    float(np.mean(piv.xs(c, level="Cond")[m].values)) for c in conds
                ],
                "all": float(np.mean(g)),
                "wins": int((winners == m).sum()),
                "body_mm": rk.get("body_mm", np.nan),
                "tip_gain": rk.get("tip_gain", np.nan),
                "tip_mm": rk.get("tip_mm", np.nan),
                "pct_under": rk.get("pct_under", np.nan),
            }
        )
    joint.sort(key=lambda r: r["all"], reverse=True)
    claimed = {r["name"] for r in joint}

    # NCL-C reference row: gain is 0 by definition (it IS the baseline), it has
    # no predictor, and it wins nothing -- but its error columns are real
    # measurements and give the absolute scale the percentages are relative to.
    baseline_row = None
    if ranking_cols and include_baseline and ref_row is not None:
        baseline_row = {
            "name": _leg_short(ref_row["method"]),
            "key": ref_row["method"],
            "rmse": np.nan,
            "std": np.nan,
            "per_cond": [0.0] * len(conds),
            "all": 0.0,
            "wins": None,
            "body_mm": ref_row.get("body_mm", np.nan),
            "tip_gain": ref_row.get("tip_gain", np.nan),
            "tip_mm": ref_row.get("tip_mm", np.nan),
            "pct_under": ref_row.get("pct_under", np.nan),
        }

    # order_by="openloop": ONE block holding every open-loop method, ordered by
    # prediction RMSE (best first), with '$-$' wherever a method was not carried
    # into the closed-loop study. Ordering by the open-loop metric is what makes
    # the selection visible -- the reader can see how far down the open-loop
    # ranking the closed-loop columns stay populated. Controllers with no
    # predictor at all (OL-C, NCL-C) cannot be placed in that ordering, so they
    # drop to a reference block at the bottom.
    refs_extra = []
    # OL-C is a controller baseline, not a learned method: it has no forward
    # model and nothing about it is being ranked. Ranking it among the methods
    # (which sorting by Gain alone would do, since its 42.7% lands mid-table)
    # gives one row two different meanings, so it joins NCL-C below the rule in
    # every ordering -- the open-loop ordering already puts it there for its own
    # reason, that a method with no predictor has no place in an RMSE sort.
    if order_by != "openloop":
        refs_extra = [r for r in joint if r["key"] == "open_loop"]
        joint = [r for r in joint if r["key"] != "open_loop"]
    if order_by == "openloop":
        by_name = {r["name"]: r for r in joint}
        unified = []
        for name, v in ol.items():
            base = by_name.get(name)
            row = dict(base) if base is not None else {
                "name": name,
                "key": None,
                "per_cond": [np.nan] * len(conds),
                "all": np.nan,
                "wins": None,
                "body_mm": np.nan,
                "tip_gain": np.nan,
                "tip_mm": np.nan,
                "pct_under": np.nan,
            }
            row["rmse"], row["std"] = v["mean"], v["std"]
            row["diverged"] = not np.isfinite(v["mean"])
            unified.append(row)
        refs_extra = [r for r in joint if r["name"] not in ol]
        # Order by the value the reader can SEE (the truncated one), not by
        # invisible digits. KERNL 6.26719 vs UORO 6.28669 both print 6.2, and
        # the 0.02 mm between them is ~1/20 of their own seed spread -- ordering
        # on that asserts a ranking the table cannot show and the data does not
        # support. Within a display tie the method carried into the closed-loop
        # study leads, as the family's representative there (the same rule
        # _draw_openloop_panel uses to let UORO stand for the online-RNN
        # family); raw RMSE only breaks what is still tied after that.
        unified.sort(
            key=lambda r: (
                r["diverged"],
                0.0 if r["diverged"] else _trunc(r["rmse"], 1),
                0 if np.isfinite(r["all"]) else 1,
                0.0 if r["diverged"] else r["rmse"],
            )
        )
        joint = unified

    # ---- methods with no closed-loop measurement --------------------------
    # They dash every column after Pred. RMSE, so a full row each spends height
    # on blank width. See OPEN_CLOSED_OL_ONLY for the three ways to place them.
    ol_only = []
    if not openloop_col:
        # No column to print their one number in, so they cannot appear here at
        # all; make_openloop_full_table is where they live.
        joint = [r for r in joint if np.isfinite(r["all"])]
    elif (ol_only_mode or OPEN_CLOSED_OL_ONLY) != "rows":
        ol_only = [r for r in joint if not np.isfinite(r["all"])]
        joint = [r for r in joint if np.isfinite(r["all"])]
        if (ol_only_mode or OPEN_CLOSED_OL_ONLY) == "drop":
            ol_only = []

    # ---- block 2: open-loop-only methods (suppressed when both_only) ------
    rest = (
        []
        if order_by == "openloop"
        else [(n, v) for n, v in ol.items() if n not in claimed]
    )
    omitted = sorted(n for n, _ in rest) if both_only else []
    if both_only:
        rest = []
    if not openloop_col:
        rest = []  # block 2 is open-loop-only by definition
    diverged = [n for n, v in rest if not np.isfinite(v["mean"])]
    finite = [(n, v) for n, v in rest if np.isfinite(v["mean"])]

    grouped = None
    if group_rnn:
        fam = [
            (n, v)
            for n, v in finite
            if str(v["label"]).startswith(_RNN_FAMILY_PREFIX)
        ]
        if len(fam) > 1:
            lo = min(v["mean"] for _, v in fam)
            hi = max(v["mean"] for _, v in fam)
            grouped = {
                "names": sorted(n for n, _ in fam),
                "lo": lo,
                "hi": hi,
            }
            finite = [
                (n, v)
                for n, v in finite
                if not str(v["label"]).startswith(_RNN_FAMILY_PREFIX)
            ]
    finite.sort(key=lambda nv: nv[1]["mean"])

    # ---- console echo (same habit as the other table builders) -----------
    cond_hdr = "".join(f"{_cond_short(c, tex=False):>11s}" for c in conds)
    rank_hdr = (
        f"{'Body(mm)':>9s}{'TipG%':>7s}{'Tip(mm)':>8s}"
        if ranking_cols
        else ""
    )
    print(f"\nOpen-loop + closed-loop summary  (n = {n_blocks} closed-loop tests)")
    print(
        f"{'Method':14s}{'RMSE(mm)':>12s}{'Gain%':>7s}{'Best':>8s}"
        f"{cond_hdr}{rank_hdr}"
    )

    def _echo(r):
        rm = (
            "div."
            if r.get("diverged")
            else ("—" if not np.isfinite(r["rmse"]) else f"{_trunc(r['rmse']):.1f}")
        )
        best = "—" if r["wins"] is None else f"{r['wins']}/{n_blocks}"
        def _n(v, w, d=1):
            if not np.isfinite(v):
                return "—".rjust(w)
            return f"{_trunc(v, d):{w}.{d}f}"

        line = f"{r['name']:14s}{rm:>12s}{_n(r['all'], 7)}{best:>8s}" + "".join(
            _n(v, 10) for v in r["per_cond"]
        )
        if ranking_cols:
            line += (
                f"{_n(r['body_mm'], 9)}{_n(r['tip_gain'], 7)}"
                f"{_n(r['tip_mm'], 8)}"
            )
        print(line)

    for r in joint:
        _echo(r)
    if refs_extra or baseline_row is not None:
        print("-" * 14 + "  (reference)")
    for r in refs_extra:
        _echo(r)
    if baseline_row is not None:
        _echo(baseline_row)
    for n, v in finite:
        print(f"{n:14s}{v['mean']:12.2f}" + "—".rjust(11) * (len(conds) + 2))
    if grouped:
        print(
            f"{'RNN family':14s}{grouped['lo']:6.1f}–{grouped['hi']:<5.1f}"
            + "—".rjust(11) * (len(conds) + 2)
            + f"   ({len(grouped['names'])} methods, tied)"
        )
    for n in diverged:
        print(f"{n:14s}{'div.':>12s}")
    if len_by_traj:
        print(
            f"  [open_closed_table] test-trajectory length: {n_mean:.1f} control "
            f"steps on average ({n_min}-{n_max}), {n_steps_cond} per condition"
        )
        for t in sorted(len_by_traj):
            durs = ", ".join(
                f"{_secs_at(len_by_traj[t], f)} s @ {f:g} Hz" for f in freqs_hz
            )
            print(f"      {t}: {len_by_traj[t]} steps" + (f"  ({durs})" if durs else ""))
    if ol_len:
        print(
            f"  [open_closed_table] open-loop evaluated over {ol_len[2]} samples "
            f"({ol_len[0]}-{ol_len[1]} per recording)"
        )
    if ol_only:
        print(
            f"  [open_closed_table] no closed-loop measurement, inlined as one "
            f"spanned line ({len(ol_only)}): "
            + ", ".join(r["name"] for r in ol_only)
        )
    if dropped_sweep:
        print(
            f"  [open_closed_table] N-sweep members excluded from the table "
            f"({len(dropped_sweep)}): " + ", ".join(dropped_sweep)
        )
    if omitted:
        print(
            f"  [open_closed_table] open-loop only, not shown here "
            f"({len(omitted)}): " + ", ".join(omitted)
        )
        print("  -> all of them appear in tables/openloop_full_table.tex")

    # ---- LaTeX ------------------------------------------------------------
    _tex1 = _fmt1  # one truncated decimal, shared with the other tables

    # Method + (Pred. RMSE) + (Gain, Best) + one per condition + error columns.
    # c0 is the 1-based index of the first CLOSED-LOOP column, so every rule and
    # span below is expressed relative to it and shifts by itself when the
    # open-loop column comes or goes.
    n_rank = 3 if ranking_cols else 0
    c0 = 3 if openloop_col else 2
    ncol = c0 + 1 + len(conds) + n_rank
    env = "table*" if wide else "table"
    # Caption states what each column is and how to read it -- nothing more.
    # Every quantity is derived, so it cannot drift from the numbers above.
    # Caption is written for a reader who has only the paper: it defines each
    # column and nothing else. Implementation details (display precision, which
    # ablation members were left out) belong in the text or the code, not here.
    # Caption carries ONLY what a reader needs to decode the columns in front
    # of them -- what each one measures, and what bold means. What the numbers
    # imply, why these methods, how the study was designed: that is the main
    # text's job, and repeating it here just makes the caption long enough to
    # skip. Every quantity is still derived, so it cannot drift from the table.
    design = _design_phrase(piv.index)
    caption = rf"Closed-loop tracking over {n_blocks} blocks"
    caption += rf" ({design}). " if design else ". "
    if openloop_col:
        _ol_n = rf", {ol_len[2]} samples" if ol_len else ""
        caption += (
            rf"\emph{{Pred.\ RMSE}}: open-loop one-step prediction error "
            rf"({_n_ol_traj} trajectories{_ol_n}). "
        )
    caption += (
        rf"\emph{{Gain}}: \% reduction in mean Euclidean error vs.\ NCL-C. "
        rf"\emph{{Best}}: blocks won. "
    )
    if ranking_cols:
        caption += (
            rf"\emph{{Body}}: all four markers; \emph{{Tip}}: most distal. "
        )
    # Block length: the one piece of design a reader cannot get from the table
    # itself, and it is needed to read any per-step quantity as wall-clock.
    if len_by_traj and freqs_hz:
        _durs = ", ".join(
            rf"{_secs_at(n_mean, f)}\,s at {f:g}\,Hz" for f in freqs_hz
        )
        caption += (
            rf"Blocks average {_secs_at(n_mean, 1)} steps ({n_min}--{n_max}): "
            rf"{_durs}. "
        )
    caption += r"Bold: best in column."
    if order_by == "openloop" and ol_only:
        caption += (
            rf" Below the rule: {len(ol_only)} methods measured open-loop only."
        )

    # Column spec: vertical rules ONLY at group boundaries, so they read as
    # separators rather than a grid. The first split is the regime --
    #   Method | Open-loop | Closed-loop( overall | by condition | error )
    spec = "l" + ("|r" if openloop_col else "") + "|rr|" + "r" * len(conds)
    if ranking_cols:
        spec += "|" + "r" * n_rank

    def _cond_hdr_tex(c):
        """Two-line condition header. Spelled out on one line these are far
        wider than the numbers beneath them and overflow the double column."""
        s = str(c)
        freq = r"20\,Hz" if s.startswith("20") else r"10\,Hz"
        kind = "Fault" if "Fault" in s else "Nom."
        return rf"\shortstack{{{freq}\\{kind}}}"

    def _fmt_mm(key):
        return lambda r: _fmt1(r[key])

    # (value getter, display formatter, direction, decimals shown). Direction
    # suits the quantity: lowest wins for the error columns, highest for gains.
    col_defs = []
    if openloop_col:
        col_defs.append(
            (
                lambda r: r["rmse"],
                lambda r: (
                    "div." if r.get("diverged") else _tex_rmse(r["rmse"], r["std"])
                ),
                "min",
                1,
            )
        )
    col_defs += [
        (lambda r: r["all"], lambda r: _tex1(r["all"]), "max", 1),
        (
            lambda r: (np.nan if r["wins"] is None else float(r["wins"])),
            lambda r: ("$-$" if r["wins"] is None else f"{r['wins']}/{n_blocks}"),
            "max",
            0,
        ),
    ]
    col_defs += [
        (
            lambda r, i=i: r["per_cond"][i],
            lambda r, i=i: _tex1(r["per_cond"][i]),
            "max",
            1,
        )
        for i in range(len(conds))
    ]
    if ranking_cols:
        col_defs += [
            (lambda r: r["body_mm"], _fmt_mm("body_mm"), "min", 1),
            (lambda r: r["tip_gain"], lambda r: _tex1(r["tip_gain"]), "max", 1),
            (lambda r: r["tip_mm"], _fmt_mm("tip_mm"), "min", 1),
        ]

    def _cmp(getter, nd, r):
        """Comparison value at the precision the reader actually sees -- the
        TRUNCATED value, so a bold cell always prints as the best one."""
        try:
            v = float(getter(r))
        except (TypeError, ValueError):
            return np.nan
        return _trunc(v, nd) if np.isfinite(v) else np.nan

    # Winners decided on the ROUNDED-DOWN value, so two cells printing the same
    # number are either both bold or neither.
    col_best = []
    for getter, _f, direction, nd in col_defs:
        # joint + ol_only: a value moved into the inline note is still SHOWN, so
        # it still competes for the bold. Scoring only `joint` would let the
        # inline strip quietly hold the best prediction error.
        vals = [
            v
            for v in (_cmp(getter, nd, r) for r in joint + ol_only)
            if np.isfinite(v)
        ]
        col_best.append(
            None if not vals else (min(vals) if direction == "min" else max(vals))
        )

    # Three header tiers: regime, then closed-loop sub-groups, then columns.
    n_cl = 2 + len(conds) + n_rank
    # The regime tier exists only to separate open-loop from closed-loop. With
    # no open-loop column the whole table is closed-loop, so it would be a
    # banner over everything -- a row of height carrying no distinction.
    tier1 = (
        [
            "",
            r"\multicolumn{1}{c|}{Open-loop}",
            r"\multicolumn{" + str(n_cl) + r"}{c}{Closed-loop tracking}",
        ]
        if openloop_col
        else None
    )
    tier2 = ["", ""][: 2 if openloop_col else 1] + [
        r"\multicolumn{2}{c|}{Overall}",
        r"\multicolumn{" + str(len(conds)) + r"}{c" + ("|" if ranking_cols else "")
        + r"}{Gain by condition (\%)}",
    ]
    if ranking_cols:
        tier2.append(r"\multicolumn{" + str(n_rank) + r"}{c}{Tracking error}")
    tier3 = (
        ["Method"]
        + ([r"\shortstack{Pred. RMSE\\(mm)}"] if openloop_col else [])
        + [r"\shortstack{Gain\\(\%)}", "Best"]
        + [_cond_hdr_tex(c) for c in conds]
    )
    if ranking_cols:
        tier3 += [
            r"\shortstack{Body\\(mm)}",
            r"\shortstack{Tip gain\\(\%)}",
            r"\shortstack{Tip\\(mm)}",
        ]
    cmids1 = (
        r"\cmidrule(lr){2-2}" + rf"\cmidrule(lr){{{c0}-{ncol}}}"
        if openloop_col
        else None
    )
    cmids2 = [
        rf"\cmidrule(lr){{{c0}-{c0 + 1}}}",
        rf"\cmidrule(lr){{{c0 + 2}-{c0 + 1 + len(conds)}}}",
    ]
    if ranking_cols:
        cmids2.append(rf"\cmidrule(lr){{{c0 + 2 + len(conds)}-{ncol}}}")

    L = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:open_closed}",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep or _tabcolsep_for(wide, ncol)}}}",
        # With a note underneath, the tabular is typeset into scratch box 0 so
        # the note can be set to \wd0 -- the tabular's OWN width. Spanning the
        # note across the columns instead (\multicolumn with a p{} of
        # \textwidth) forces the tabular to that width, and the slack lands in
        # the last column: the Tip column drifts away from the rest of its
        # group. Box 0 is scratch and nothing else touches it in between.
        *([r"\sbox0{%"] if ol_only else []),
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        *([" & ".join(tier1) + r" \\", cmids1] if tier1 else []),
        " & ".join(tier2) + r" \\",
        "".join(cmids2),
        " & ".join(tier3) + r" \\",
        r"\midrule",
    ]

    def _row(r, reference=False):
        # No emphasis in the Method column: bold means "best value in this
        # column", and a method name is not a value. Marking the proposed
        # variant here would give the same weight two different meanings.
        cells = [r["name"]]
        for (getter, fmt, _d, nd), best in zip(col_defs, col_best):
            txt = fmt(r)
            v = _cmp(getter, nd, r)
            if not reference and best is not None and np.isfinite(v) and v == best:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        return " & ".join(cells) + r" \\"

    for r in joint:
        L.append(_row(r))
    if refs_extra or baseline_row is not None:
        L.append(r"\midrule")
    for r in refs_extra:
        L.append(_row(r, reference=True))
    if baseline_row is not None:
        L.append(_row(baseline_row, reference=True))
    # Third block only exists when open-loop-only rows are being shown; with
    # both_only there is nothing below, so no separating rule either.
    if finite or grouped or diverged:
        L.append(r"\midrule")
    # Pred. RMSE is column 2 now; every column after it is closed-loop, so an
    # open-loop-only method dashes all of them.
    n_dash_pre, n_dash_post = 0, 2 + len(conds) + n_rank
    for n, v in finite:
        L.append(
            " & ".join(
                [n] + ["$-$"] * n_dash_pre + [_tex_rmse(v["mean"], v["std"])]
                + ["$-$"] * n_dash_post
            )
            + r" \\"
        )
    if grouped:
        L.append(
            " & ".join(
                [rf"RNN family ({len(grouped['names'])})"]
                + ["$-$"] * n_dash_pre
                + [rf"{grouped['lo']:.1f}--{grouped['hi']:.1f}"]
                + ["$-$"] * n_dash_post
            )
            + r" \\"
        )
    for n in diverged:
        L.append(
            " & ".join(
                [n] + ["$-$"] * n_dash_pre + ["div."] + ["$-$"] * n_dash_post
            )
            + r" \\"
        )
    L += [r"\bottomrule", r"\end{tabular}"]
    if ol_only:
        # One wrapping note instead of one row per method: these carry a single
        # number each, so listing them along the width the table already has
        # costs two lines where full rows cost eight.
        items = ", ".join(
            f"{r['name']} {'div.' if r.get('diverged') else _tex1(r['rmse'])}"
            for r in ol_only
        )
        L += [
            r"}",  # closes \sbox0
            r"\usebox0\\[3pt]",
            r"\parbox{\wd0}{\footnotesize\raggedright%",
            rf"Open-loop only, no closed-loop measurement "
            rf"(\emph{{Pred.\ RMSE}}, mm): {items}.}}",
        ]
    L += [rf"\end{{{env}}}"]

    os.makedirs(TABLES_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(TABLES_DIR, "open_closed_table.tex")
    elif not os.path.isabs(path):
        path = os.path.join(TABLES_DIR, path)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nSaved: {path}")
    return path


def _cond_short(cond, tex=True):
    """'10Hz Nominal' -> '10\\,Hz Nom.' (tex) or '10Hz Nom.' (console), for a
    narrow column header."""
    s = str(cond)
    freq = "20" if s.startswith("20") else "10"
    kind = "Fault" if "Fault" in s else "Nom."
    return rf"{freq}\,Hz {kind}" if tex else f"{freq}Hz {kind}"


def make_openloop_full_table(df=None, path=None, openloop_data=None):
    """Companion to make_open_closed_table: EVERY open-loop method listed
    individually (nothing collapsed), for an appendix or the supplement. Two
    side-by-side panels so the 19 rows cost half the vertical space."""
    if openloop_data is None:
        openloop_data = load_openloop_multiseed(OPENLOOP_MULTISEED_PKL)
    ol = _openloop_by_display(openloop_data)
    # CerebNet is reported at its published single update only. Its inner loop is
    # NOT an inference depth: _update_cerebnet computes the error once, outside
    # the loop, and the reservoir state is frozen, so N applications sum exactly
    # to one step at N*eta. Kalidindi et al. (2019) define one update per
    # simulation step and justify the rule as covariance-based plasticity, not
    # gradient descent, so there is no multi-step version of the method to
    # report. Printed beside "PC (N=...)" the label would invite the reading that
    # N is the same knob for both. The sweep stays in the pickle; only its
    # publication stops here.
    ol = {n: v for n, v in ol.items() if not n.startswith("CerebNet (N=")}
    # Provenance lives here now: this is the only table reporting the open-loop
    # regime, so the evidence behind it has to be stated here rather than in the
    # closed-loop table's caption.
    _n_ol_rec, _n_ol_traj = _openloop_provenance()
    ol_len = _openloop_samples()
    finite = sorted(
        ((n, v) for n, v in ol.items() if np.isfinite(v["mean"])),
        key=lambda nv: nv[1]["mean"],
    )
    diverged = [n for n, v in ol.items() if not np.isfinite(v["mean"])]

    def _cell(n, v):
        return n, _tex_rmse(v["mean"], v["std"])

    rows = finite + [(n, {"mean": np.nan, "std": np.nan}) for n in diverged]
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]

    L = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Open-loop one-step prediction RMSE over {_n_ol_traj} test "
        rf"trajectories"
        + (rf" ({ol_len[2]} samples)" if ol_len else "")
        + rf", mean $\pm$ s.d.\ across seeds, best first."
        # Each notation is explained only when it actually appears in the body.
        # With the CerebNet N rows gone every remaining method is stochastic and
        # none diverges, so both sentences would otherwise be dead text.
        + (
            r" No spread: deterministic method."
            if any(v["std"] == 0 for _, v in finite)
            else ""
        )
        + (
            r" \emph{div.}: non-finite RMSE on at least one seed."
            if diverged
            else ""
        )
        + r"}",
        r"\label{tab:openloop_full}",
        r"\begin{tabular}{lr@{\hskip 1.2em}lr}",
        r"\toprule",
        r"Method & RMSE (mm) & Method & RMSE (mm) \\",
        r"\midrule",
    ]
    for i in range(half):
        ln, lv = left[i]
        lname, lval = _cell(ln, lv)
        if not np.isfinite(lv["mean"]):
            lval = "div."
        if i < len(right):
            rn, rv = right[i]
            rname, rval = _cell(rn, rv)
            if not np.isfinite(rv["mean"]):
                rval = "div."
        else:
            rname, rval = "", ""
        L.append(f"{lname} & {lval} & {rname} & {rval} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(TABLES_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(TABLES_DIR, "openloop_full_table.tex")
    elif not os.path.isabs(path):
        path = os.path.join(TABLES_DIR, path)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_style()
    df = load_dataframe()
    if df is None or df.empty:
        raise SystemExit("No data loaded. Set CACHE_PKL or run where the data lives.")
    fig_closed_loop_grid(df)
    make_timing_table(df)
    make_ranking_table(df)
    # Actuation + configuration-space tracking timeseries, transposed onto
    # shared axes so both conditions fit in half the height.
    # OL-C is deliberately NOT here. It was tried and dropped: a sixth trace in
    # panels this dense costs more legibility than the no-feedback floor buys,
    # and that floor is already quantified in the tables. The PC entry carries
    # its N, since N is a setting of the method and the other figures report
    # several values of it -- an unqualified "PC-RNN-I" would not say which one
    # this is.
    methods_tracking = {
        "NCL-C": ["baseline"],
        "LoG-GP": ["log_gp"],
        "PC-RNN-I (N=5, ours)": ["temporal_pc_N5"],
        "CerebNet": ["cerebnet"],
    }
    fig_actuation_tracking_grid(
        df,
        dataset_id="v21",
        base_freq="10Hz",
        component_idx=None,
        methods=methods_tracking,
        motor_indices=[0],
    )
    # The two regimes are reported as two tables rather than one: the
    # closed-loop table is 10 columns and needs the full page width, while the
    # open-loop result is a single number per method and fits a text column
    # two-up. Merging them made the wide table carry eight rows whose only
    # content was one number and nine dashes.
    # wide=False: with the open-loop column gone the body fits one text column
    # (see OPEN_CLOSED_TABCOLSEP), so it no longer needs a page-wide float.
    # ranking_cols=False: the whole-body/tip error breakdown is make_ranking_table's
    # subject (tab:ranking), so this table keeps only the gains -- the overall
    # pair and the per-condition split -- and the two stop duplicating columns.
    make_open_closed_table(df, openloop_col=False, wide=False, ranking_cols=False)
    make_openloop_full_table(df)
    print(
        "Done: F_closed_loop_2x2.pdf, F_tracking_timeseries_v21_10Hz.pdf, "
        "ranking_table.tex, timing_table.tex, "
        "tables/open_closed_table.tex, tables/openloop_full_table.tex"
    )
