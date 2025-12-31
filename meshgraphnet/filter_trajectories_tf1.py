#!/usr/bin/env python3
"""Creates filtered datasets with train/aux/cal/test splits for downstream conformal prediction."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf


def filter_tfrecord_by_index_and_timesteps(
    input_tfrecord: Path,
    output_tfrecord: Path,
    trajectory_indices: set,
    timestep_range: tuple,
    meta: dict,
) -> int:
    """
    Filter TFRecord by trajectory index and slice timesteps (TensorFlow 1.x compatible).

    Args:
        input_tfrecord: Path to input TFRecord file
        output_tfrecord: Path to output filtered TFRecord file
        trajectory_indices: Set of trajectory indices to keep (0-indexed)
        timestep_range: (start, end) timestep range to keep (inclusive start, exclusive end)
        meta: Metadata dictionary from meta.json

    Returns:
        Number of trajectories kept
    """
    kept = 0
    total = 0
    t_start, t_end = timestep_range

    print(f"Filtering {input_tfrecord}")
    print(f"Keeping trajectory indices: {sorted(trajectory_indices)}")
    print(f"Timestep range: [{t_start}, {t_end})")

    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    writer = tf.python_io.TFRecordWriter(str(output_tfrecord))

    def _tfrecord_iterator(path: Path):
        # TF1: tf.python_io.tf_record_iterator
        # TF2: tf.compat.v1.io.tf_record_iterator
        if hasattr(tf, "python_io") and hasattr(tf.python_io, "tf_record_iterator"):
            return tf.python_io.tf_record_iterator(str(path))
        return tf.compat.v1.io.tf_record_iterator(str(path))

    try:
        for record in _tfrecord_iterator(input_tfrecord):
            if total in trajectory_indices:
                example_in = tf.train.Example.FromString(record)

                out_features = {}
                for key, field in meta["features"].items():
                    if key not in example_in.features.feature:
                        raise KeyError(
                            f"Feature '{key}' not found in TFRecord example. "
                            f"Available keys: {sorted(example_in.features.feature.keys())}"
                        )

                    raw_values = example_in.features.feature[key].bytes_list.value
                    if not raw_values:
                        raise ValueError(f"Feature '{key}' has empty bytes_list")

                    dtype = np.dtype(field["dtype"])
                    shape = [int(s) for s in field["shape"]]

                    data = np.frombuffer(raw_values[0], dtype=dtype).reshape(shape)

                    ftype = field["type"]
                    if ftype == "static":
                        # IMPORTANT: DeepMind loader reshapes using meta['features'][k]['shape']
                        # and tiles static to meta['trajectory_length'] at load time.
                        # So we must keep static tensors at length 1 (do NOT tile here).
                        data_sliced = data
                    elif ftype == "dynamic":
                        if t_end > data.shape[0]:
                            raise ValueError(
                                f"Requested timestep_range {timestep_range} exceeds "
                                f"available timesteps ({data.shape[0]}) for feature '{key}'."
                            )
                        data_sliced = data[t_start:t_end]
                    elif ftype == "dynamic_varlen":
                        # Optional support (not used by cylinder_flow / flag_simple meta.json).
                        length_key = "length_" + key
                        if length_key not in example_in.features.feature:
                            raise KeyError(
                                f"Missing '{length_key}' needed for dynamic_varlen feature '{key}'."
                            )
                        length_raw = example_in.features.feature[
                            length_key
                        ].bytes_list.value
                        if not length_raw:
                            raise ValueError(
                                f"Feature '{length_key}' has empty bytes_list"
                            )
                        length = np.frombuffer(length_raw[0], dtype=np.int32).reshape(
                            -1
                        )
                        if t_end > length.shape[0]:
                            raise ValueError(
                                f"Requested timestep_range {timestep_range} exceeds "
                                f"available row_lengths ({length.shape[0]}) for '{length_key}'."
                            )

                        length_sliced = length[t_start:t_end]
                        cumsum = np.concatenate([[0], np.cumsum(length)])
                        start_idx = int(cumsum[t_start])
                        end_idx = int(cumsum[t_end])
                        data_sliced = data[start_idx:end_idx]

                        out_features[length_key] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[length_sliced.tobytes()]
                            )
                        )
                    else:
                        raise ValueError(f"Unknown field type: {ftype}")

                    out_features[key] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data_sliced.tobytes()])
                    )

                example_out = tf.train.Example(
                    features=tf.train.Features(feature=out_features)
                )
                writer.write(example_out.SerializeToString())
                kept += 1

            total += 1
            if total % 100 == 0:
                print(f"  Processed {total} trajectories, kept {kept}")
    finally:
        writer.close()

    print(
        f"\nFiltering complete: Total={total}, Kept={kept}, Output={output_tfrecord}\n"
    )
    return kept


def filter_tfrecord_by_index(
    input_tfrecord: Path, output_tfrecord: Path, trajectory_indices: set
) -> int:
    """
    Filter TFRecord by trajectory index only (no timestep slicing).

    Args:
        input_tfrecord: Path to input TFRecord file
        output_tfrecord: Path to output filtered TFRecord file
        trajectory_indices: Set of trajectory indices to keep (0-indexed)

    Returns:
        Number of trajectories kept
    """
    kept = 0
    total = 0

    print(f"Filtering {input_tfrecord}")
    print(f"Keeping trajectory indices: {sorted(trajectory_indices)}")

    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    writer = tf.python_io.TFRecordWriter(str(output_tfrecord))

    try:
        dataset = tf.data.TFRecordDataset([str(input_tfrecord)])
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            while True:
                try:
                    record = sess.run(next_element)
                    if total in trajectory_indices:
                        writer.write(record)
                        kept += 1
                    total += 1

                    if total % 100 == 0:
                        print(f"  Processed {total} trajectories, kept {kept}")
                except tf.errors.OutOfRangeError:
                    break
    finally:
        writer.close()

    print(
        f"\nFiltering complete: Total={total}, Kept={kept}, Output={output_tfrecord}\n"
    )
    return kept


def create_filtered_dataset(
    splits_file: Path, input_base_dir: Path, output_base_dir: Path, dataset_name: str
) -> None:
    """
    Create filtered dataset with train/aux/cal/test splits.

    Creates correct TFRecord file structure for DeepMind:
    - train.tfrecord (primary file)
    - valid.tfrecord (copy for eval splits)
    - test.tfrecord (copy for test split)
    - meta.json (copied to root and each split)

    Args:
        splits_file: Path to splits JSON file
        input_base_dir: Base directory with original dataset
        output_base_dir: Base directory for filtered dataset
        dataset_name: 'cylinder_flow' or 'flag_simple'
    """
    with open(splits_file, "r") as f:
        splits = json.load(f)

    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    source_file = input_base / "train.tfrecord"
    if not source_file.exists():
        alternatives = list(input_base.glob("*.tfrecord"))
        if alternatives:
            source_file = alternatives[0]
            print(f"Using {source_file.name} as source")
        else:
            raise FileNotFoundError(f"No TFRecord file found in {input_base}")

    print("\n" + "=" * 70)
    print(f"Creating filtered dataset: {dataset_name}")
    print(f"Source: {source_file}")
    print(f"Output: {output_base}")
    print("=" * 70 + "\n")

    split_file_config = {
        "train": ["train.tfrecord"],
        "auxiliary": ["train.tfrecord", "valid.tfrecord"],
        "calibration": ["train.tfrecord", "valid.tfrecord"],
        "test": ["train.tfrecord", "valid.tfrecord", "test.tfrecord"],
    }

    # Load metadata
    metadata_file = input_base / "meta.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with open(metadata_file, "r") as f:
        meta = json.load(f)

    for split_name in ["train", "auxiliary", "calibration", "test"]:
        trajectory_ids = set(splits[split_name]["trajectory_ids"])
        timestep_range = tuple(splits[split_name]["timestep_range"])
        output_dir = output_base / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        primary_file = output_dir / "train.tfrecord"

        print(f"Creating {split_name} split:")
        print(f"  Trajectories: {sorted(trajectory_ids)}")
        print(f"  Timestep range: [{timestep_range[0]}, {timestep_range[1]})")

        filter_tfrecord_by_index_and_timesteps(
            source_file, primary_file, trajectory_ids, timestep_range, meta
        )

        # Update metadata for this split:
        # - DeepMind loader reshapes tensors using meta['features'][k]['shape']
        # - For static features, it tiles to meta['trajectory_length']
        # Therefore we must update BOTH trajectory_length and dynamic feature shapes[0].
        split_meta = json.loads(json.dumps(meta))  # deep copy (JSON-friendly)
        new_len = int(timestep_range[1] - timestep_range[0])
        split_meta["trajectory_length"] = new_len
        for _, field in split_meta.get("features", {}).items():
            if field.get("type") == "dynamic":
                shape = list(field.get("shape", []))
                if shape:
                    shape[0] = new_len
                    field["shape"] = shape
            elif field.get("type") == "dynamic_varlen":
                # For varlen, the stored data is ragged; DeepMind still reshapes to field['shape'].
                # If the first dimension is time, update it; otherwise leave as-is.
                shape = list(field.get("shape", []))
                if shape and shape[0] not in (-1, 0):
                    shape[0] = new_len
                    field["shape"] = shape
        split_meta_file = output_dir / "meta.json"
        with open(split_meta_file, "w") as f:
            json.dump(split_meta, f, indent=2)

        for expected_file in split_file_config[split_name][1:]:
            shutil.copy(primary_file, output_dir / expected_file)
            print(f"  Created {expected_file}")

    # Root metadata uses original trajectory_length (for reference)
    if metadata_file.exists():
        shutil.copy(metadata_file, output_base / "meta.json")
        print(f"\nCopied metadata to root (original trajectory_length)")

    print("\n" + "=" * 70)
    print(f"Filtered dataset created at: {output_base}")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter TFRecord datasets for conformal prediction (TensorFlow 1.x)"
    )
    parser.add_argument(
        "--splits_file", type=str, required=True, help="Path to splits JSON file"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for filtered dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cylinder", "flag"],
        help="Dataset name (auto-detected from splits if not provided)",
    )

    args = parser.parse_args()

    if args.dataset is None:
        with open(args.splits_file, "r") as f:
            splits = json.load(f)
        dataset_name = splits["metadata"]["dataset"]
    else:
        dataset_name = "cylinder_flow" if args.dataset == "cylinder" else "flag_simple"

    create_filtered_dataset(
        Path(args.splits_file),
        Path(args.input_dir),
        Path(args.output_dir),
        dataset_name,
    )


if __name__ == "__main__":
    main()
