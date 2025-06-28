import argparse
import numpy as np
import os

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram

def main():
    args = parse_args()

    TRAIN_DIR = args.train_dir
    VALID_DIR = args.valid_dir
    TEST_DIR = args.test_dir
    rng = np.random.RandomState(args.seed)
    reference = args.reference
    tractogram_name = os.path.basename(args.in_file)

    assert args.train_ratio + args.valid_ratio + args.test_ratio == 1

    sft = load_tractogram(args.in_file, reference, bbox_valid_check=False)
    
    random_indices = np.arange(len(sft))
    rng.shuffle(random_indices)

    train_idx = random_indices[:int(len(sft) * args.train_ratio)]
    valid_idx = random_indices[int(len(sft) * args.train_ratio):int(len(sft) * (args.train_ratio + args.valid_ratio))]
    test_idx = random_indices[int(len(sft) * (args.train_ratio + args.valid_ratio)):]

    train_sft = sft[train_idx]
    valid_sft = sft[valid_idx]
    test_sft = sft[test_idx]

    print("Processing tractogram {}. Train: {}, Valid: {}, Test: {}".format(tractogram_name, len(train_sft), len(valid_sft), len(test_sft)))

    save_tractogram(train_sft, os.path.join(TRAIN_DIR, "tr-" + tractogram_name), bbox_valid_check=False)
    save_tractogram(valid_sft, os.path.join(VALID_DIR, "v-" + tractogram_name), bbox_valid_check=False)
    save_tractogram(test_sft, os.path.join(TEST_DIR, "te-" + tractogram_name), bbox_valid_check=False)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("in_file", type=str, help="in tractogram file to separate into train/valid/test")
    parser.add_argument("train_dir", type=str, help="output directory for train tractogram")
    parser.add_argument("valid_dir", type=str, help="output directory for valid tractogram")
    parser.add_argument("test_dir", type=str, help="output directory for test tractogram")
    parser.add_argument("--reference", type=str, default="same", help="reference file to split the tractogram")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="ratio of train tractogram")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="ratio of valid tractogram")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="ratio of test tractogram")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()


if __name__ == '__main__':
    main()