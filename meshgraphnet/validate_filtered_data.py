#!/usr/bin/env python3
"""Validates filtered MeshGraphNet TFRecord datasets against split JSON + meta.json."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


SPLITS = ("train", "auxiliary", "calibration", "test")


def _tfrecord_iterator(path: Path):
    # TF1: tf.python_io.tf_record_iterator
    # TF2: tf.compat.v1.io.tf_record_iterator
    if hasattr(tf, "python_io") and hasattr(tf.python_io, "tf_record_iterator"):
        return tf.python_io.tf_record_iterator(str(path))
    return tf.compat.v1.io.tf_record_iterator(str(path))


def _count_records(path: Path) -> int:
    n = 0
    for _ in _tfrecord_iterator(path):
        n += 1
    return n


def _decode_one_example(record_bytes: bytes, meta: dict) -> dict:
    ex = tf.train.Example.FromString(record_bytes)
    out = {}
    for key, field in meta["features"].items():
        if key not in ex.features.feature:
            raise KeyError(f"Missing feature '{key}' in TFRecord example")
        raw = ex.features.feature[key].bytes_list.value
        if not raw:
            raise ValueError(f"Feature '{key}' has empty bytes_list")
        dtype = np.dtype(field["dtype"])
        shape = [int(s) for s in field["shape"]]
        arr = np.frombuffer(raw[0], dtype=dtype).reshape(shape)
        out[key] = arr
        if field["type"] == "dynamic_varlen":
            lk = "length_" + key
            if lk not in ex.features.feature:
                raise KeyError(f"Missing '{lk}' for dynamic_varlen feature '{key}'")
    return out


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def validate(filtered_root: Path, splits_file: Path, max_decode: int = 1) -> None:
    _assert(filtered_root.exists(), f"filtered_root not found: {filtered_root}")
    _assert(splits_file.exists(), f"splits_file not found: {splits_file}")

    with open(splits_file, "r") as f:
        splits = json.load(f)

    print(f"Filtered root: {filtered_root}")
    print(f"Splits file:   {splits_file}")
    print(f"Dataset:       {splits.get('metadata', {}).get('dataset')}")

    for split in SPLITS:
        split_dir = filtered_root / split
        _assert(split_dir.exists(), f"Missing split dir: {split_dir}")

        train_tf = split_dir / "train.tfrecord"
        _assert(train_tf.exists(), f"Missing {train_tf}")
        _assert(train_tf.stat().st_size > 0, f"Empty {train_tf}")

        # DeepMind compatibility copies (as produced by filter script)
        if split in ("auxiliary", "calibration", "test"):
            valid_tf = split_dir / "valid.tfrecord"
            _assert(valid_tf.exists(), f"Missing {valid_tf}")
            _assert(
                valid_tf.stat().st_size == train_tf.stat().st_size,
                f"{valid_tf} size != train.tfrecord",
            )
        if split == "test":
            test_tf = split_dir / "test.tfrecord"
            _assert(test_tf.exists(), f"Missing {test_tf}")
            _assert(
                test_tf.stat().st_size == train_tf.stat().st_size,
                f"{test_tf} size != train.tfrecord",
            )

        meta_path = split_dir / "meta.json"
        _assert(meta_path.exists(), f"Missing {meta_path}")
        meta = json.loads(meta_path.read_text())

        t0, t1 = splits[split]["timestep_range"]
        expected_len = int(t1 - t0)
        _assert(
            int(meta["trajectory_length"]) == expected_len,
            f"{split}/meta.json trajectory_length={meta['trajectory_length']} != expected {expected_len}",
        )

        # Validate meta shapes are consistent with DeepMind loader assumptions
        for key, field in meta["features"].items():
            shape0 = int(field["shape"][0])
            ftype = field["type"]
            if ftype == "static":
                _assert(
                    shape0 == 1,
                    f"{split}/meta.json static feature '{key}' shape[0] must be 1, got {shape0}",
                )
            elif ftype == "dynamic":
                _assert(
                    shape0 == expected_len,
                    f"{split}/meta.json dynamic feature '{key}' shape[0]={shape0} != {expected_len}",
                )

        # Count trajectories (records) match trajectory_ids count
        expected_traj = len(splits[split]["trajectory_ids"])
        actual_traj = _count_records(train_tf)
        _assert(
            actual_traj == expected_traj,
            f"{split}/train.tfrecord records={actual_traj} != expected {expected_traj}",
        )

        # Decode a couple examples to ensure bytes reshape correctly
        it = _tfrecord_iterator(train_tf)
        for i in range(max_decode):
            try:
                record = next(it)
            except StopIteration:
                break
            decoded = _decode_one_example(record, meta)
            # Minimal sanity checks
            for key, field in meta["features"].items():
                arr = decoded[key]
                if field["type"] == "dynamic":
                    _assert(
                        arr.shape[0] == expected_len,
                        f"{split} example: '{key}' time dim {arr.shape[0]} != {expected_len}",
                    )
                if field["type"] == "static":
                    _assert(
                        arr.shape[0] == 1,
                        f"{split} example: '{key}' time dim {arr.shape[0]} != 1",
                    )

        print(
            f"OK: {split:11s} records={actual_traj:2d} trajectory_length={expected_len}"
        )

    print("OK: all checks passed")


def main():
    ap = argparse.ArgumentParser(description="Validate filtered TFRecord datasets")
    ap.add_argument(
        "--filtered_root",
        type=str,
        required=True,
        help="Filtered dataset root (contains train/auxiliary/calibration/test subdirs)",
    )
    ap.add_argument(
        "--splits_file", type=str, required=True, help="Split JSON file used to filter"
    )
    ap.add_argument(
        "--max_decode",
        type=int,
        default=1,
        help="How many examples per split to decode (default: 1)",
    )
    args = ap.parse_args()

    try:
        validate(
            Path(args.filtered_root), Path(args.splits_file), max_decode=args.max_decode
        )
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
