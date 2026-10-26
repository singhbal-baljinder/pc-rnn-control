# Online Predictive Coding for Adaptive Shape Control of Soft Robots

Code and data for the paper (`docs/paper.pdf`). All experiments ran on a physical
cable-driven soft arm; the recordings are included, so every figure and table can
be regenerated without hardware.

## Install

With [pixi](https://pixi.sh):

```bash
pixi install
pixi run build-lwpr   # optional: compiles the LWPR baseline
```

Or with pip (Python 3.11–3.14, LWPR baseline skipped): `pip install -e .`

## Reproduce the figures and tables

```bash
pixi run figures      # or: python scripts/make_figures.py
```

| Paper | Output |
|---|---|
| Fig. 3 | `F_closed_loop_2x2.pdf` |
| Fig. 4 | `F_tracking_timeseries_v21_10Hz.pdf` |
| Table I | `ranking_table.tex` |
| Table II | `tables/openloop_full_table.tex` |
| Table III | `tables/open_closed_table.tex` |
| Table IV | `timing_table.tex` |

Takes about a minute. Closed-loop metrics are read from
`data/closed_loop_cache.pkl`; delete it to recompute them from the raw
recordings. Prebuilt outputs are attached to each
[release](https://github.com/singhbal-baljinder/pc-rnn-control/releases).

## Re-run the experiments

**Open loop** (no hardware; several hours):

```bash
pixi run openloop     # writes data/open_loop/
```

**Closed loop** needs three Dynamixel motors on `/dev/ttyUSB0`, an OptiTrack
system (set its addresses at the top of `scripts/run_closed_loop.py`) and
`pip install -e ".[robot]"`.

```bash
python scripts/run_babbling.py      # collect training data
python scripts/train_policy.py      # train the policy
python scripts/run_closed_loop.py   # one run
```

Each run is set by the globals at the top of `run_closed_loop.py`:

| Global | Values |
|---|---|
| `version` | test trajectory: `5, 19, 21, 23, 35` |
| `desired_freq` | `[10]` or `[20]` Hz |
| `condition` | `""` nominal, `"_fault"` motor 0 frozen |
| `USE_ADAPTIVE_MODEL` | `True` for a learner, `False` for NCL-C / OL-C |
| `TEST_CLOSED_LOOP_WITHOUT_ADAPTATION` | with `USE_ADAPTIVE_MODEL=False`: `True` NCL-C, `False` OL-C |
| `ADAPTIVE_METHOD` | `temporal_pc`, `cerebnet`, `log_gp`, `lwpr`, `rnn-uoro` |
| `idx_trial` | for `temporal_pc`, selects *N*: `40`→1, `0`→2, `20`→5, `60`→10 |

## Data

- `data/closed_loop/`: closed-loop recordings, the policy
  (`lstm_inverse_dynamics.pth`) and its training data.
- `data/open_loop/`: test trajectories and open-loop results.

Closed-loop files are named
`spiraling_babbling_{10|20}_Hz_test_trajectory_v{5|19|21|23|35}_mvel0.5[_suffix][_fault].npz`,
where the suffix is empty for the reference, `_closed_loop_seq0` for NCL-C,
`_open_loop_seq0` for OL-C, and `_seq{method}_adaptive{idx_trial}` for a learner.

## License

BSD-3-Clause (`LICENSE`). Third-party components keep their own licenses: see
`NOTICE`. To cite, see `CITATION.cff`.
