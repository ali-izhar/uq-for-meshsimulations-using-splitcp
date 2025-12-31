Final **TensorFlow 1.x** checkpoints at step **200,000** (you can use these to generate rollouts).

- `checkpoints/cylinder_ts/`: CylinderFlow checkpoint
- `checkpoints/flag_ts/`: Flag checkpoint

Each contains `model.ckpt-200000.{data-00000-of-00001,index,meta}`. Run the TF1 MeshGraphNet evaluator and point `--checkpoint_dir` to one of the directories above (see `meshgraphnet/README.md` for the full environment setup). Example:

```bash
python -m meshgraphnets.run_model \
  --mode=eval \
  --model=cfd \
  --checkpoint_dir=/workspace/checkpoints/cylinder_ts \
  --dataset_dir=/workspace/data/cylinder_flow_filtered/test \
  --rollout_path=/workspace/rollouts_200k/rollout_cylinder_test_200k.pkl \
  --num_rollouts=3
```
