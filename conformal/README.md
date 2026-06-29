# Conformal Prediction

This module runs **post-hoc split conformal prediction** on **MeshGraphNet rollout `.pkl` files** (DeepMind `meshgraphnets.run_model --mode=eval --rollout_path=...`).

## What target do we conformalize?

- **CylinderFlow (CFD)**: conformal prediction on the model's **velocity** output  
  Uses rollout keys: `pred_velocity`, `gt_velocity` (vector dim $D=2$).

- **Flag (Cloth)**: conformal prediction on the model's **position** output (**by design**)  
  Uses rollout keys: `pred_pos`, `gt_pos` (vector dim $D=3$).

Why position for Flag: the released MeshGraphNet cloth model rollouts are in position space (`pred_pos/gt_pos`). Converting to velocity via finite differences would amplify noise and changes the error distribution; we keep CP on the native predicted quantity.

## Methods implemented

Given residual vectors $r = y^{true} - y^{pred}$:

- **Standard CP ($\ell_2$)**: score $s = \|r\|\_2$ → radius $Q_2$
- **Standard CP ($\ell_\infty$)**: score $s = \|r\|\_\infty$ → half-width $Q_\infty$
- **Mahalanobis CP**: fit covariance $\Sigma$ on **aux**, score $s=\sqrt{r^\top \Sigma^{-1} r}$ → whitened radius $Q_{\mathrm{Mah}}$
- **Spatially-adaptive CP**: learn $\sigma(x)$ on **aux**, score $s=\|r\|\_2/\sigma(x)$ on **cal** → adaptive radius $Q_{\mathrm{adapt}}\cdot\sigma(x)$
- **CW-Adaptive CP**: learn per-component $\sigma_j(x)$ on **aux**, score $s_j=|r_j|/\sigma_j(x)$ on **cal** → per-component adaptive half-width $Q_{\mathrm{cw}}\cdot\sigma_j(x)$

### Efficiency summary

| Dataset | Best Method | Notes |
|---------|-------------|-------|
| **CylinderFlow** | CW-Adaptive | Smallest sets (up to ~55% vs L2) via per-component, spatially-varying scales; Mahalanobis gives only ~14% from global anisotropy |
| **Flag** | CW-Adaptive | Per-axis scaling captures position heterogeneity; Mahalanobis can inflate sets (no exploitable global structure) |

## Inputs (what you need to generate first)

Three rollout `.pkl` files for a given dataset:

- auxiliary rollout for $\Sigma$ and $\sigma(x)$
- calibration rollout (for quantiles)
- evaluation rollout (held-out; for reporting only)

Each `.pkl` is a `list[dict]` (one dict per trajectory). Required keys:

- Cylinder: `mesh_pos`, `pred_velocity`, `gt_velocity`
- Flag: `mesh_pos`, `pred_pos`, `gt_pos`

## Exchangeability and practical validity

Split conformal prediction guarantees marginal coverage under the assumption that calibration samples are exchangeable with future samples from the target distribution. In mesh-based autoregressive simulations, this assumption is violated due to strong temporal dependence within rollouts and spatial coupling across mesh nodes.

In this work, we therefore interpret conformal coverage as approximately valid, following prior conformal analyses of dependent spatiotemporal data. Our data partitioning is designed to prevent information leakage and align calibration with evaluation regimes, rather than to enforce independence:

- all splits are disjoint in trajectory identity (distinct simulation runs), and
- auxiliary, calibration, and evaluation splits use the same timestep horizon as training to ensure comparable error growth under autoregressive rollout dynamics.

This protocol avoids leakage between training, score learning, calibration, and evaluation, but does not render the data i.i.d. Accordingly, reported coverage results should be interpreted as empirical coverage under controlled dependence, rather than exact finite-sample guarantees.

## Run (CLI)

```bash
python3 conformal/run_conformal.py \
  --aux_pkl /path/to/rollout_*_auxiliary_*.pkl \
  --cal_pkl /path/to/rollout_*_calibration_*.pkl \
  --eval_pkl /path/to/rollout_*_test_*.pkl \
  --outdir conformal_out/my_run \
  --alphas 0.1 0.05
```

### Reproduce the paper (exact canonical runs)

These are the **exact** commands behind the paper's tables and figures. They use the
full (uncapped) `rollouts_200k_big` rollouts and `--seed 42`. The non-adaptive methods
(ℓ₂, ℓ∞, Mahalanobis) are bit-exact; with `n_jobs=1` (the default in `models.py`) the
XGBoost-based adaptive methods are reproducible as well.

```bash
# CylinderFlow (velocity) — canonical
python3 conformal/run_conformal.py \
  --aux_pkl  meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl \
  --cal_pkl  meshgraphnet/rollouts_200k_big/rollout_cylinder_calibration_200k.pkl \
  --eval_pkl meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl \
  --outdir   conformal/_out_cylinder_200k_big_inregime_xgbq_physfull \
  --alphas 0.3 0.2 0.1 0.05 \
  --sigma_model xgboost \
  --feature_set physics_full \
  --seed 42

# Flag (position) — canonical (sigma capped at the 98th percentile)
python3 conformal/run_conformal.py \
  --aux_pkl  meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl \
  --cal_pkl  meshgraphnet/rollouts_200k_big/rollout_flag_calibration_200k.pkl \
  --eval_pkl meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl \
  --outdir   conformal/_out_flag_200k_big_inregime_xgboost_sigcap098 \
  --alphas 0.3 0.2 0.1 0.05 \
  --sigma_model xgboost \
  --feature_set physics_full \
  --sigma_cap_quantile 0.98 \
  --seed 42
```

Both runs are uncapped (≈74.7M cylinder / ≈31.3M flag node×timestep samples) and fit on
~120 GB RAM. The `--max_*` flags below are an **optional** memory-control variant that
subsamples and therefore does **not** reproduce the canonical numbers exactly.

Generate the paper tables from these outputs with:

```bash
python3 plot/tables.py \
  --cyl_conformal_out  conformal/_out_cylinder_200k_big_inregime_xgbq_physfull \
  --flag_conformal_out conformal/_out_flag_200k_big_inregime_xgboost_sigcap098 \
  --out_dir paper_tables --alphas 0.30 0.20 0.10 0.05
```

### Large rollouts (recommended caps)

This pipeline flattens rollouts to node×timestep rows. Use `--max_aux_cov/--max_aux_sigma/--max_cal/--max_eval` to cap rows and control memory on very large rollouts (while still using millions of samples).

### Adaptive CP robustness knob: `--sigma_cap_quantile`

For **adaptive** CP on heavy-tailed rollouts (notably Flag), the mean volume metric can be dominated by a tiny fraction of samples with very large predicted $\sigma(x)$ (since size scales as $(q\,\sigma)^D$). To keep results stable with minimal changes, you can cap $\sigma(x)$ at a high quantile of auxiliary $\sigma$-predictions and recalibrate as usual. The canonical Flag run already uses `--sigma_cap_quantile 0.98` (see "Reproduce the paper" above).

The example below is an **optional, memory-capped variant** for machines with limited RAM: the `--max_*` flags subsample to a few million rows, so it does **not** reproduce the canonical numbers exactly (it is a separate output directory by design).

```bash
python3 conformal/run_conformal.py \
  --aux_pkl  meshgraphnet/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl \
  --cal_pkl  meshgraphnet/rollouts_200k_big/rollout_flag_calibration_200k.pkl \
  --eval_pkl meshgraphnet/rollouts_200k_big/rollout_flag_test_200k.pkl \
  --outdir conformal/_out_flag_sigcap098_capped_variant \
  --alphas 0.3 0.2 0.1 0.05 \
  --sigma_model xgboost \
  --feature_set physics_full \
  --sigma_cap_quantile 0.98 \
  --max_aux_cov 6000000 --max_aux_sigma 2000000 --max_cal 6000000 --max_eval 6000000 \
  --seed 42
```

Empirically in the Flag setting, `0.98` dramatically reduced adaptive size while keeping coverage near nominal; `0.995` was too weak and `0.99` was intermediate.

### Recommended setting (Flag)

On the current Flag rollouts, the best overall trade-off we observed was:
- `--sigma_model xgboost`
- `--sigma_cap_quantile 0.98`

This kept coverage near nominal at $\alpha=0.10$ and $\alpha=0.05$ while substantially reducing adaptive size compared to the uncapped baseline.

## Outputs (written to `--outdir`)

- `Sigma.npy`, `Sigma_inv.npy`, `Sigma_meta.json`
- `sigma_model.pkl`
- `thresholds.json` (per method/per alpha)
- `summary.json` (coverage + average radii summaries)

## Notes about time dimension

It is normal for rollout arrays to have $T = \text{trajectory length} - 2$ due to DeepMind's preprocessing (`val[1:-1]` and targets `val[2:]`). The conformal pipeline simply uses the timesteps present in the rollout files.


