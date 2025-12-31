# Raw run logs (splits + rollouts)

This file preserves the full contents of `meshgraphnet/splits.log` and `meshgraphnet/rollouts.log` (verbatim) so the logs can be deleted later without losing the evidence trail for:
- split construction / filtering / validation, and
- how rollout error grows with horizon (reported by `meshgraphnets.run_model` as `mse_{k}_steps`).

## `meshgraphnet/splits.log` (verbatim)

```text
root@4277158b7b77:/workspace# python3 - << 'PY'
> import tensorflow as tf
> from pathlib import Path
> 
> def count(path):
>     n = 0
>     for _ in tf.compat.v1.io.tf_record_iterator(str(path)):
>         n += 1
>     return n
> 
> print("cylinder train.tfrecord trajectories:", count("data/cylinder_flow/cylinder_flow/train.tfrecord"))
> print("flag     train.tfrecord trajectories:", count("data/flag_simple/flag_simple/train.tfrecord"))
> PY
2025-12-28 03:02:18.907779: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
cylinder train.tfrecord trajectories: 1000
flag     train.tfrecord trajectories: 1000

root@4277158b7b77:/workspace# python create_splits.py \
>   --dataset cylinder \
>   --num_trajectories 1000 \
>   --seed 42 \
>   --keep_train_from data/splits/cylinder_splits.json \
>   --timesteps_per_traj 600 \
>   --train_timesteps 400 \
>   --nontrain_timestep_range 0 400 \
>   --aux_n 100 \
>   --cal_n 100 \
>   --test_n 100 \
>   --output data/splits/cylinder_splits_big_inregime.json

============================================================
DATA SPLIT SUMMARY
============================================================

Dataset: cylinder_flow
Total trajectories: 1000

TRAIN:
  Trajectories: [0, 1, 2, 4, 5, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 25]
  Timestep range: (0, 400)
  Total samples: 6,800

AUXILIARY:
  Trajectories: [40, 47, 56, 76, 80, 83, 84, 87, 93, 103, 105, 113, 124, 137, 156, 175, 185, 191, 201, 215, 226, 227, 232, 238, 248, 252, 254, 264, 277, 278, 283, 292, 294, 297, 311, 315, 324, 329, 331, 349, 361, 374, 378, 379, 425, 428, 437, 441, 459, 465, 484, 495, 510, 512, 514, 530, 558, 565, 574, 587, 598, 616, 643, 652, 653, 679, 686, 694, 696, 697, 704, 721, 734, 738, 756, 762, 800, 814, 818, 827, 830, 839, 867, 869, 872, 875, 881, 885, 897, 902, 912, 924, 936, 939, 947, 949, 974, 979, 985, 991]
  Timestep range: (0, 400)
  Total samples: 40,000

CALIBRATION:
  Trajectories: [46, 61, 66, 72, 77, 82, 89, 95, 127, 153, 154, 158, 182, 216, 230, 235, 261, 271, 298, 307, 313, 316, 322, 326, 336, 348, 350, 351, 363, 367, 369, 397, 399, 405, 411, 415, 442, 449, 450, 457, 462, 464, 467, 468, 481, 502, 507, 516, 518, 531, 533, 535, 536, 538, 549, 551, 553, 564, 566, 576, 592, 601, 604, 610, 621, 632, 634, 647, 650, 651, 660, 667, 672, 685, 712, 723, 728, 731, 733, 747, 754, 768, 774, 782, 820, 823, 828, 852, 894, 900, 905, 926, 942, 948, 950, 951, 989, 993, 995, 998]
  Timestep range: (0, 400)
  Total samples: 40,000

TEST:
  Trajectories: [7, 18, 20, 27, 42, 48, 50, 71, 94, 98, 99, 101, 114, 118, 126, 135, 172, 209, 213, 221, 225, 228, 244, 256, 267, 276, 282, 303, 309, 323, 335, 340, 345, 359, 368, 376, 380, 382, 387, 398, 446, 447, 453, 456, 466, 473, 479, 496, 498, 511, 517, 539, 543, 545, 548, 550, 561, 568, 575, 584, 585, 605, 607, 608, 612, 613, 620, 637, 648, 702, 724, 752, 758, 766, 771, 772, 777, 779, 794, 801, 803, 805, 815, 819, 834, 840, 858, 870, 895, 903, 907, 918, 923, 935, 943, 953, 960, 978, 994, 997]
  Timestep range: (0, 400)
  Total samples: 40,000

OK Auxiliary split: 40000 samples (need >= 480)
============================================================

Saved splits to data/splits/cylinder_splits_big_inregime.json
root@4277158b7b77:/workspace# python create_splits.py \
>   --dataset flag \
>   --num_trajectories 1000 \
>   --seed 42 \
>   --keep_train_from data/splits/flag_splits.json \
>   --timesteps_per_traj 401 \
>   --train_timesteps 200 \
>   --nontrain_timestep_range 0 200 \
>   --aux_n 100 \
>   --cal_n 100 \
>   --test_n 100 \
>   --output data/splits/flag_splits_big_inregime.json

============================================================
DATA SPLIT SUMMARY
============================================================

Dataset: flag_simple
Total trajectories: 1000

TRAIN:
  Trajectories: [0, 1, 2, 3, 5, 8, 9, 11, 13, 15, 16, 17]
  Timestep range: (0, 200)
  Total samples: 2,400

AUXILIARY:
  Trajectories: [22, 35, 42, 51, 71, 75, 78, 79, 82, 88, 98, 100, 108, 119, 151, 170, 186, 196, 210, 221, 225, 256, 266, 273, 277, 292, 293, 298, 304, 321, 330, 343, 364, 371, 374, 375, 392, 394, 400, 435, 436, 454, 457, 461, 469, 493, 497, 509, 511, 512, 531, 533, 534, 546, 547, 556, 559, 590, 593, 594, 602, 607, 608, 614, 623, 627, 637, 645, 650, 655, 656, 684, 694, 700, 732, 772, 790, 792, 800, 801, 813, 821, 832, 842, 845, 846, 867, 876, 880, 884, 885, 892, 900, 906, 924, 935, 948, 949, 974, 985]
  Timestep range: (0, 200)
  Total samples: 20,000

CALIBRATION:
  Trajectories: [37, 41, 56, 67, 72, 77, 84, 90, 122, 132, 148, 149, 153, 177, 180, 211, 222, 227, 230, 233, 243, 249, 259, 272, 278, 301, 302, 311, 317, 322, 331, 333, 340, 354, 356, 362, 406, 432, 444, 445, 451, 460, 462, 463, 479, 490, 494, 528, 553, 555, 561, 596, 597, 599, 603, 606, 618, 629, 630, 632, 641, 648, 662, 665, 669, 685, 706, 708, 726, 742, 744, 752, 764, 766, 769, 777, 782, 802, 812, 818, 828, 863, 873, 875, 878, 893, 896, 899, 901, 902, 907, 914, 925, 938, 941, 942, 943, 961, 986, 995]
  Timestep range: (0, 200)
  Total samples: 20,000

TEST:
  Trajectories: [7, 14, 19, 21, 43, 45, 61, 66, 89, 93, 94, 96, 109, 113, 121, 130, 167, 204, 208, 216, 220, 223, 239, 247, 251, 262, 271, 287, 289, 306, 308, 310, 318, 319, 324, 326, 335, 346, 358, 363, 367, 369, 377, 382, 383, 393, 434, 440, 465, 468, 480, 489, 498, 513, 526, 530, 535, 541, 543, 548, 551, 563, 564, 580, 582, 605, 610, 611, 625, 642, 646, 696, 718, 722, 729, 734, 751, 767, 786, 799, 836, 837, 838, 840, 857, 860, 869, 891, 904, 910, 916, 918, 946, 959, 975, 977, 978, 989, 994, 997]
  Timestep range: (0, 200)
  Total samples: 20,000

OK Auxiliary split: 20000 samples (need >= 180)
============================================================

Saved splits to data/splits/flag_splits_big_inregime.json

root@4277158b7b77:/workspace# python filter_trajectories_tf1.py \
>   --splits_file data/splits/cylinder_splits_big_inregime.json \
>   --input_dir data/cylinder_flow/cylinder_flow \
>   --output_dir data/cylinder_flow_filtered_big \
>   --dataset cylinder
2025-12-28 03:05:56.485562: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.

======================================================================
Creating filtered dataset: cylinder_flow
Source: data/cylinder_flow/cylinder_flow/train.tfrecord
Output: data/cylinder_flow_filtered_big
======================================================================

Creating train split:
  Trajectories: [0, 1, 2, 4, 5, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 25]
  Timestep range: [0, 400)
Filtering data/cylinder_flow/cylinder_flow/train.tfrecord
Keeping trajectory indices: [0, 1, 2, 4, 5, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 25]
Timestep range: [0, 400)
WARNING:tensorflow:From filter_trajectories_tf1.py:42: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.

  Processed 100 trajectories, kept 17
  Processed 200 trajectories, kept 17
  Processed 300 trajectories, kept 17
  Processed 400 trajectories, kept 17
  Processed 500 trajectories, kept 17
  Processed 600 trajectories, kept 17
  Processed 700 trajectories, kept 17
  Processed 800 trajectories, kept 17
  Processed 900 trajectories, kept 17
  Processed 1000 trajectories, kept 17

Filtering complete: Total=1000, Kept=17, Output=data/cylinder_flow_filtered_big/train/train.tfrecord

Creating auxiliary split:
  Trajectories: [40, 47, 56, 76, 80, 83, 84, 87, 93, 103, 105, 113, 124, 137, 156, 175, 185, 191, 201, 215, 226, 227, 232, 238, 248, 252, 254, 264, 277, 278, 283, 292, 294, 297, 311, 315, 324, 329, 331, 349, 361, 374, 378, 379, 425, 428, 437, 441, 459, 465, 484, 495, 510, 512, 514, 530, 558, 565, 574, 587, 598, 616, 643, 652, 653, 679, 686, 694, 696, 697, 704, 721, 734, 738, 756, 762, 800, 814, 818, 827, 830, 839, 867, 869, 872, 875, 881, 885, 897, 902, 912, 924, 936, 939, 947, 949, 974, 979, 985, 991]
  Timestep range: [0, 400)
Filtering data/cylinder_flow/cylinder_flow/train.tfrecord
Keeping trajectory indices: [40, 47, 56, 76, 80, 83, 84, 87, 93, 103, 105, 113, 124, 137, 156, 175, 185, 191, 201, 215, 226, 227, 232, 238, 248, 252, 254, 264, 277, 278, 283, 292, 294, 297, 311, 315, 324, 329, 331, 349, 361, 374, 378, 379, 425, 428, 437, 441, 459, 465, 484, 495, 510, 512, 514, 530, 558, 565, 574, 587, 598, 616, 643, 652, 653, 679, 686, 694, 696, 697, 704, 721, 734, 738, 756, 762, 800, 814, 818, 827, 830, 839, 867, 869, 872, 875, 881, 885, 897, 902, 912, 924, 936, 939, 947, 949, 974, 979, 985, 991]
Timestep range: [0, 400)
  Processed 100 trajectories, kept 9
  Processed 200 trajectories, kept 18
  Processed 300 trajectories, kept 34
  Processed 400 trajectories, kept 44
  Processed 500 trajectories, kept 52
  Processed 600 trajectories, kept 61
  Processed 700 trajectories, kept 70
  Processed 800 trajectories, kept 76
  Processed 900 trajectories, kept 89
  Processed 1000 trajectories, kept 100

Filtering complete: Total=1000, Kept=100, Output=data/cylinder_flow_filtered_big/auxiliary/train.tfrecord

  Created valid.tfrecord
Creating calibration split:
  Trajectories: [46, 61, 66, 72, 77, 82, 89, 95, 127, 153, 154, 158, 182, 216, 230, 235, 261, 271, 298, 307, 313, 316, 322, 326, 336, 348, 350, 351, 363, 367, 369, 397, 399, 405, 411, 415, 442, 449, 450, 457, 462, 464, 467, 468, 481, 502, 507, 516, 518, 531, 533, 535, 536, 538, 549, 551, 553, 564, 566, 576, 592, 601, 604, 610, 621, 632, 634, 647, 650, 651, 660, 667, 672, 685, 712, 723, 728, 731, 733, 747, 754, 768, 774, 782, 820, 823, 828, 852, 894, 900, 905, 926, 942, 948, 950, 951, 989, 993, 995, 998]
  Timestep range: [0, 400)
Filtering data/cylinder_flow/cylinder_flow/train.tfrecord
Keeping trajectory indices: [46, 61, 66, 72, 77, 82, 89, 95, 127, 153, 154, 158, 182, 216, 230, 235, 261, 271, 298, 307, 313, 316, 322, 326, 336, 348, 350, 351, 363, 367, 369, 397, 399, 405, 411, 415, 442, 449, 450, 457, 462, 464, 467, 468, 481, 502, 507, 516, 518, 531, 533, 535, 536, 538, 549, 551, 553, 564, 566, 576, 592, 601, 604, 610, 621, 632, 634, 647, 650, 651, 660, 667, 672, 685, 712, 723, 728, 731, 733, 747, 754, 768, 774, 782, 820, 823, 828, 852, 894, 900, 905, 926, 942, 948, 950, 951, 989, 993, 995, 998]
Timestep range: [0, 400)
  Processed 100 trajectories, kept 8
  Processed 200 trajectories, kept 13
  Processed 300 trajectories, kept 19
  Processed 400 trajectories, kept 33
  Processed 500 trajectories, kept 45
  Processed 600 trajectories, kept 61
  Processed 700 trajectories, kept 74
  Processed 800 trajectories, kept 84
  Processed 900 trajectories, kept 89
  Processed 1000 trajectories, kept 100

Filtering complete: Total=1000, Kept=100, Output=data/cylinder_flow_filtered_big/calibration/train.tfrecord

  Created valid.tfrecord
Creating test split:
  Trajectories: [7, 18, 20, 27, 42, 48, 50, 71, 94, 98, 99, 101, 114, 118, 126, 135, 172, 209, 213, 221, 225, 228, 244, 256, 267, 276, 282, 303, 309, 323, 335, 340, 345, 359, 368, 376, 380, 382, 387, 398, 446, 447, 453, 456, 466, 473, 479, 496, 498, 511, 517, 539, 543, 545, 548, 550, 561, 568, 575, 584, 585, 605, 607, 608, 612, 613, 620, 637, 648, 702, 724, 752, 758, 766, 771, 772, 777, 779, 794, 801, 803, 805, 815, 819, 834, 840, 858, 870, 895, 903, 907, 918, 923, 935, 943, 953, 960, 978, 994, 997]
  Timestep range: [0, 400)
Filtering data/cylinder_flow/cylinder_flow/train.tfrecord
Keeping trajectory indices: [7, 18, 20, 27, 42, 48, 50, 71, 94, 98, 99, 101, 114, 118, 126, 135, 172, 209, 213, 221, 225, 228, 244, 256, 267, 276, 282, 303, 309, 323, 335, 340, 345, 359, 368, 376, 380, 382, 387, 398, 446, 447, 453, 456, 466, 473, 479, 496, 498, 511, 517, 539, 543, 545, 548, 550, 561, 568, 575, 584, 585, 605, 607, 608, 612, 613, 620, 637, 648, 702, 724, 752, 758, 766, 771, 772, 777, 779, 794, 801, 803, 805, 815, 819, 834, 840, 858, 870, 895, 903, 907, 918, 923, 935, 943, 953, 960, 978, 994, 997]
Timestep range: [0, 400)
  Processed 100 trajectories, kept 11
  Processed 200 trajectories, kept 17
  Processed 300 trajectories, kept 27
  Processed 400 trajectories, kept 40
  Processed 500 trajectories, kept 49
  Processed 600 trajectories, kept 61
  Processed 700 trajectories, kept 69
  Processed 800 trajectories, kept 79
  Processed 900 trajectories, kept 89
  Processed 1000 trajectories, kept 100

Filtering complete: Total=1000, Kept=100, Output=data/cylinder_flow_filtered_big/test/train.tfrecord

  Created valid.tfrecord
  Created test.tfrecord

Copied metadata to root (original trajectory_length)

======================================================================
Filtered dataset created at: data/cylinder_flow_filtered_big
======================================================================

root@4277158b7b77:/workspace# python filter_trajectories_tf1.py \
>   --splits_file data/splits/flag_splits_big_inregime.json \
>   --input_dir data/flag_simple/flag_simple \
>   --output_dir data/flag_simple_filtered_big \
>   --dataset flag
2025-12-28 03:07:59.023885: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.

======================================================================
Creating filtered dataset: flag_simple
Source: data/flag_simple/flag_simple/train.tfrecord
Output: data/flag_simple_filtered_big
======================================================================

Creating train split:
  Trajectories: [0, 1, 2, 3, 5, 8, 9, 11, 13, 15, 16, 17]
  Timestep range: [0, 200)
Filtering data/flag_simple/flag_simple/train.tfrecord
Keeping trajectory indices: [0, 1, 2, 3, 5, 8, 9, 11, 13, 15, 16, 17]
Timestep range: [0, 200)
WARNING:tensorflow:From filter_trajectories_tf1.py:42: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.

  Processed 100 trajectories, kept 12
  Processed 200 trajectories, kept 12
  Processed 300 trajectories, kept 12
  Processed 400 trajectories, kept 12
  Processed 500 trajectories, kept 12
  Processed 600 trajectories, kept 12
  Processed 700 trajectories, kept 12
  Processed 800 trajectories, kept 12
  Processed 900 trajectories, kept 12
  Processed 1000 trajectories, kept 12

Filtering complete: Total=1000, Kept=12, Output=data/flag_simple_filtered_big/train/train.tfrecord

Creating auxiliary split:
  Trajectories: [22, 35, 42, 51, 71, 75, 78, 79, 82, 88, 98, 100, 108, 119, 151, 170, 186, 196, 210, 221, 225, 256, 266, 273, 277, 292, 293, 298, 304, 321, 330, 343, 364, 371, 374, 375, 392, 394, 400, 435, 436, 454, 457, 461, 469, 493, 497, 509, 511, 512, 531, 533, 534, 546, 547, 556, 559, 590, 593, 594, 602, 607, 608, 614, 623, 627, 637, 645, 650, 655, 656, 684, 694, 700, 732, 772, 790, 792, 800, 801, 813, 821, 832, 842, 845, 846, 867, 876, 880, 884, 885, 892, 900, 906, 924, 935, 948, 949, 974, 985]
  Timestep range: [0, 200)
Filtering data/flag_simple/flag_simple/train.tfrecord
Keeping trajectory indices: [22, 35, 42, 51, 71, 75, 78, 79, 82, 88, 98, 100, 108, 119, 151, 170, 186, 196, 210, 221, 225, 256, 266, 273, 277, 292, 293, 298, 304, 321, 330, 343, 364, 371, 374, 375, 392, 394, 400, 435, 436, 454, 457, 461, 469, 493, 497, 509, 511, 512, 531, 533, 534, 546, 547, 556, 559, 590, 593, 594, 602, 607, 608, 614, 623, 627, 637, 645, 650, 655, 656, 684, 694, 700, 732, 772, 790, 792, 800, 801, 813, 821, 832, 842, 845, 846, 867, 876, 880, 884, 885, 892, 900, 906, 924, 935, 948, 949, 974, 985]
Timestep range: [0, 200)
  Processed 100 trajectories, kept 11
  Processed 200 trajectories, kept 18
  Processed 300 trajectories, kept 28
  Processed 400 trajectories, kept 38
  Processed 500 trajectories, kept 47
  Processed 600 trajectories, kept 60
  Processed 700 trajectories, kept 73
  Processed 800 trajectories, kept 78
  Processed 900 trajectories, kept 92
  Processed 1000 trajectories, kept 100

Filtering complete: Total=1000, Kept=100, Output=data/flag_simple_filtered_big/auxiliary/train.tfrecord

  Created valid.tfrecord
Creating calibration split:
  Trajectories: [37, 41, 56, 67, 72, 77, 84, 90, 122, 132, 148, 149, 153, 177, 180, 211, 222, 227, 230, 233, 243, 249, 259, 272, 278, 301, 302, 311, 317, 322, 331, 333, 340, 354, 356, 362, 406, 432, 444, 445, 451, 460, 462, 463, 479, 490, 494, 528, 553, 555, 561, 596, 597, 599, 603, 606, 618, 629, 630, 632, 641, 648, 662, 665, 669, 685, 706, 708, 726, 742, 744, 752, 764, 766, 769, 777, 782, 802, 812, 818, 828, 863, 873, 875, 878, 893, 896, 899, 901, 902, 907, 914, 925, 938, 941, 942, 943, 961, 986, 995]
  Timestep range: [0, 200)
Filtering data/flag_simple/flag_simple/train.tfrecord
Keeping trajectory indices: [37, 41, 56, 67, 72, 77, 84, 90, 122, 132, 148, 149, 153, 177, 180, 211, 222, 227, 230, 233, 243, 249, 259, 272, 278, 301, 302, 311, 317, 322, 331, 333, 340, 354, 356, 362, 406, 432, 444, 445, 451, 460, 462, 463, 479, 490, 494, 528, 553, 555, 561, 596, 597, 599, 603, 606, 618, 629, 630, 632, 641, 648, 662, 665, 669, 685, 706, 708, 726, 742, 744, 752, 764, 766, 769, 777, 782, 802, 812, 818, 828, 863, 873, 875, 878, 893, 896, 899, 901, 902, 907, 914, 925, 938, 941, 942, 943, 961, 986, 995]
Timestep range: [0, 200)
  Processed 100 trajectories, kept 8
  Processed 200 trajectories, kept 15
  Processed 300 trajectories, kept 25
  Processed 400 trajectories, kept 36
  Processed 500 trajectories, kept 47
  Processed 600 trajectories, kept 54
  Processed 700 trajectories, kept 66
  Processed 800 trajectories, kept 77
  Processed 900 trajectories, kept 88
  Processed 1000 trajectories, kept 100

Filtering complete: Total=1000, Kept=100, Output=data/flag_simple_filtered_big/calibration/train.tfrecord

  Created valid.tfrecord
Creating test split:
  Trajectories: [7, 14, 19, 21, 43, 45, 61, 66, 89, 93, 94, 96, 109, 113, 121, 130, 167, 204, 208, 216, 220, 223, 239, 247, 251, 262, 271, 287, 289, 306, 308, 310, 318, 319, 324, 326, 335, 346, 358, 363, 367, 369, 377, 382, 383, 393, 434, 440, 465, 468, 480, 489, 498, 513, 526, 530, 535, 541, 543, 548, 551, 563, 564, 580, 582, 605, 610, 611, 625, 642, 646, 696, 718, 722, 729, 734, 751, 767, 786, 799, 836, 837, 838, 840, 857, 860, 869, 891, 904, 910, 916, 918, 946, 959, 975, 977, 978, 989, 994, 997]
  Timestep range: [0, 200)
Filtering data/flag_simple/flag_simple/train.tfrecord
Keeping trajectory indices: [7, 14, 19, 21, 43, 45, 61, 66, 89, 93, 94, 96, 109, 113, 121, 130, 167, 204, 208, 216, 220, 223, 239, 247, 251, 262, 271, 287, 289, 306, 308, 310, 318, 319, 324, 326, 335, 346, 358, 363, 367, 369, 377, 382, 383, 393, 434, 440, 465, 468, 480, 489, 498, 513, 526, 530, 535, 541, 543, 548, 551, 563, 564, 580, 582, 605, 610, 611, 625, 642, 646, 696, 718, 722, 729, 734, 751, 767, 786, 799, 836, 837, 838, 840, 857, 860, 869, 891, 904, 910, 916, 918, 946, 959, 975, 977, 978, 989, 994, 997]
Timestep range: [0, 200)
  Processed 100 trajectories, kept 12
  Processed 200 trajectories, kept 17
  Processed 300 trajectories, kept 29
  Processed 400 trajectories, kept 46
  Processed 500 trajectories, kept 53
  Processed 600 trajectories, kept 65
  Processed 700 trajectories, kept 72
  Processed 800 trajectories, kept 80
  Processed 900 trajectories, kept 88
  Processed 1000 trajectories, kept 100

Filtering complete: Total=1000, Kept=100, Output=data/flag_simple_filtered_big/test/train.tfrecord

  Created valid.tfrecord
  Created test.tfrecord

Copied metadata to root (original trajectory_length)

======================================================================
Filtered dataset created at: data/flag_simple_filtered_big
======================================================================

root@4277158b7b77:/workspace# python validate_filtered_data.py --filtered_root data/cylinder_flow_filtered_big --splits_file data/splits/cylinder_splits_big_inregime.json
2025-12-28 03:13:04.646856: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
Filtered root: data/cylinder_flow_filtered_big
Splits file:   data/splits/cylinder_splits_big_inregime.json
Dataset:       cylinder_flow
OK: train       records=17 trajectory_length=400
OK: auxiliary   records=100 trajectory_length=400
OK: calibration records=100 trajectory_length=400
OK: test        records=100 trajectory_length=400
OK: all checks passed
root@4277158b7b77:/workspace# python validate_filtered_data.py --filtered_root data/flag_simple_filtered_big     --splits_file data/splits/flag_splits_big_inregime.json
2025-12-28 03:13:12.038462: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
Filtered root: data/flag_simple_filtered_big
Splits file:   data/splits/flag_splits_big_inregime.json
Dataset:       flag_simple
OK: train       records=12 trajectory_length=200
OK: auxiliary   records=100 trajectory_length=200
OK: calibration records=100 trajectory_length=200
OK: test        records=100 trajectory_length=200
OK: all checks passed
root@4277158b7b77:/workspace#
```

## `meshgraphnet/rollouts.log` (verbatim)

```text
python -m meshgraphnets.run_model --mode=eval --model=cfd \
--checkpoint_dir=/workspace/checkpoints_cylinder_ts \
--dataset_dir=/workspace/data/cylinder_flow_filtered_big/auxiliary \
--rollout_path=/workspace/rollouts_200k_big/rollout_cylinder_auxiliary_200k.pkl \
--num_rollouts=100
I1228 03:26:03.340367 129576913110848 run_model.py:108] mse_1_steps: 4.89168e-05
I1228 03:26:03.340635 129576913110848 run_model.py:108] mse_10_steps: 0.00025628
I1228 03:26:03.340759 129576913110848 run_model.py:108] mse_20_steps: 0.000496697
I1228 03:26:03.340871 129576913110848 run_model.py:108] mse_50_steps: 0.00143578
I1228 03:26:03.341012 129576913110848 run_model.py:108] mse_100_steps: 0.0029496
I1228 03:26:03.341130 129576913110848 run_model.py:108] mse_200_steps: 0.00508053

python -m meshgraphnets.run_model --mode=eval --model=cfd \
--checkpoint_dir=/workspace/checkpoints_cylinder_ts \
--dataset_dir=/workspace/data/cylinder_flow_filtered_big/calibration \
--rollout_path=/workspace/rollouts_200k_big/rollout_cylinder_calibration_200k.pkl \
--num_rollouts=100
I1228 03:31:38.347477 133983561582400 run_model.py:108] mse_1_steps: 6.7407e-05
I1228 03:31:38.347704 133983561582400 run_model.py:108] mse_10_steps: 0.000491677
I1228 03:31:38.347809 133983561582400 run_model.py:108] mse_20_steps: 0.000824743
I1228 03:31:38.347929 133983561582400 run_model.py:108] mse_50_steps: 0.00215337
I1228 03:31:38.348039 133983561582400 run_model.py:108] mse_100_steps: 0.00424973
I1228 03:31:38.348134 133983561582400 run_model.py:108] mse_200_steps: 0.00686158

python -m meshgraphnets.run_model --mode=eval --model=cfd \
--checkpoint_dir=/workspace/checkpoints_cylinder_ts \
--dataset_dir=/workspace/data/cylinder_flow_filtered_big/test \
--rollout_path=/workspace/rollouts_200k_big/rollout_cylinder_test_200k.pkl \
--num_rollouts=100
I1228 03:37:37.676693 128166499178304 run_model.py:108] mse_1_steps: 6.92228e-05
I1228 03:37:37.676957 128166499178304 run_model.py:108] mse_10_steps: 0.000456855
I1228 03:37:37.677088 128166499178304 run_model.py:108] mse_20_steps: 0.000768666
I1228 03:37:37.677205 128166499178304 run_model.py:108] mse_50_steps: 0.00213939
I1228 03:37:37.677309 128166499178304 run_model.py:108] mse_100_steps: 0.00443646
I1228 03:37:37.677402 128166499178304 run_model.py:108] mse_200_steps: 0.00742983

python -m meshgraphnets.run_model --mode=eval --model=cloth \
--checkpoint_dir=/workspace/checkpoints_flag_ts \
--dataset_dir=/workspace/data/flag_simple_filtered_big/auxiliary \
--rollout_path=/workspace/rollouts_200k_big/rollout_flag_auxiliary_200k.pkl \
--num_rollouts=100
I1228 03:41:22.741400 139777214125888 run_model.py:108] mse_1_steps: 8.11162e-06
I1228 03:41:22.741680 139777214125888 run_model.py:108] mse_10_steps: 0.00169201
I1228 03:41:22.741826 139777214125888 run_model.py:108] mse_20_steps: 0.0147205
I1228 03:41:22.741970 139777214125888 run_model.py:108] mse_50_steps: 0.105052
I1228 03:41:22.742102 139777214125888 run_model.py:108] mse_100_steps: 0.255706
I1228 03:41:22.742219 139777214125888 run_model.py:108] mse_200_steps: 18.4769

python -m meshgraphnets.run_model --mode=eval --model=cloth \
--checkpoint_dir=/workspace/checkpoints_flag_ts \
--dataset_dir=/workspace/data/flag_simple_filtered_big/calibration \
--rollout_path=/workspace/rollouts_200k_big/rollout_flag_calibration_200k.pkl \
--num_rollouts=100
I1228 03:46:04.016163 128501664495424 run_model.py:108] mse_1_steps: 9.13314e-06
I1228 03:46:04.016480 128501664495424 run_model.py:108] mse_10_steps: 0.00145692
I1228 03:46:04.016661 128501664495424 run_model.py:108] mse_20_steps: 0.013609
I1228 03:46:04.016812 128501664495424 run_model.py:108] mse_50_steps: 0.0925646
I1228 03:46:04.016973 128501664495424 run_model.py:108] mse_100_steps: 0.272882
I1228 03:46:04.017119 128501664495424 run_model.py:108] mse_200_steps: 15.8142

python -m meshgraphnets.run_model --mode=eval --model=cloth \
--checkpoint_dir=/workspace/checkpoints_flag_ts \
--dataset_dir=/workspace/data/flag_simple_filtered_big/test \
--rollout_path=/workspace/rollouts_200k_big/rollout_flag_test_200k.pkl \
--num_rollouts=100
I1228 03:49:03.486294 128617067259712 run_model.py:108] mse_1_steps: 6.59612e-06
I1228 03:49:03.486533 128617067259712 run_model.py:108] mse_10_steps: 0.00100763
I1228 03:49:03.486653 128617067259712 run_model.py:108] mse_20_steps: 0.00952753
I1228 03:49:03.486748 128617067259712 run_model.py:108] mse_50_steps: 0.0762604
I1228 03:49:03.486836 128617067259712 run_model.py:108] mse_100_steps: 0.169146
I1228 03:49:03.486948 128617067259712 run_model.py:108] mse_200_steps: 0.280274
```

