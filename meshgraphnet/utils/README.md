# MeshGraphNet Utility Scripts

Everything here operates on existing MeshGraphNet rollout `.pkl` files (no training, no inference).

## Inputs
- Base rollouts: `meshgraphnet/rollouts_200k_big/*.pkl`
- Sensitivity rollouts: `meshgraphnet/rollouts_sensitivity_big/*.pkl`

## Outputs
All outputs are written to `meshgraphnet/_artifacts/` and then copied to `assets/` for documentation.

See [../RESULTS.md](../RESULTS.md) for rendered figures and tables.

---

### Error accumulation (RMSE vs time)
```bash
python -m meshgraphnet.utils.error_accumulation \
  --rollout_pkl meshgraphnet/rollouts_200k_big/rollout_cylinder_test_200k.pkl \
  --traj_idx -1 \
  --out_dir meshgraphnet/_artifacts/error_accumulation/cylinder_rich
```

### Split sensitivity summary (across seeds)
```bash
python -m meshgraphnet.utils.split_sensitivity \
  --rollouts_dir meshgraphnet/rollouts_sensitivity_big \
  --traj_idx -1 \
  --out_dir meshgraphnet/_artifacts/split_sensitivity_dense
```

### Exchangeability diagnostics (temporal + spatial)
```bash
python -m meshgraphnet.utils.exchangeability \
  --rollout_pkl meshgraphnet/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl \
  --traj_idx 0 \
  --out_dir meshgraphnet/_artifacts/exchangeability/cylinder_aux_rich
```

### Exchangeability batch (all rollouts)
```bash
python -m meshgraphnet.utils.exchangeability_batch \
  --out_dir meshgraphnet/_artifacts/exchangeability/batch_all \
  --rollouts_dirs meshgraphnet/rollouts_200k_big meshgraphnet/rollouts_sensitivity_big
```
