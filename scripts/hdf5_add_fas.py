import nibabel as nib
import numpy as np
import os
import h5py
import argparse
from tqdm import tqdm

from tractoracle_irt.datasets.create_dataset_tracking import add_volume_to_hdf5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_to_append', type=str)
    parser.add_argument('fa_folder', type=str, help='Folder with FA images.')
    return parser.parse_args()

def main():
    args = parse_args()

    assert os.path.exists(args.hdf5_to_append), f"{args.hdf5_to_append} does not exist."
    assert os.path.exists(args.fa_folder), f"{args.fa_folder} does not exist."

    with h5py.File(args.hdf5_to_append, 'a') as hdf:
        fa_folder = args.fa_folder

        print(">>>>>>>> TRAINING")
        for subject in tqdm(hdf['training'].keys(), desc='Training'):
            print(f"Subject: {subject}")

            if 'fa_volume' in hdf['training'][subject]:
                print(f"({subject}) FA volume already exists. Skipping...")
                continue

            fa_file = os.path.join(fa_folder, "trainset", f"{subject}__fa.nii.gz")
            fa_img = nib.load(fa_file)
            add_volume_to_hdf5(hdf['training'][subject], fa_img, 'fa_volume')

        print(">>>>>>>> VALIDATION")
        for subject in tqdm(hdf['validation'].keys(), desc='Validation'):
            print(f"Subject: {subject}")

            if 'fa_volume' in hdf['validation'][subject]:
                print(f"({subject}) FA volume already exists. Skipping...")
                continue

            fa_file = os.path.join(fa_folder, "validset", f"{subject}__fa.nii.gz")
            fa_img = nib.load(fa_file)
            add_volume_to_hdf5(hdf['validation'][subject], fa_img, 'fa_volume')

        print(">>>>>>>> TESTING")
        for subject in tqdm(hdf['testing'].keys(), desc='Testing'):
            print(f"Subject: {subject}")

            if 'fa_volume' in hdf['testing'][subject]:
                print(f"({subject}) FA volume already exists. Skipping...")
                continue

            fa_file = os.path.join(fa_folder, "testset", f"{subject}__fa.nii.gz")
            fa_img = nib.load(fa_file)
            add_volume_to_hdf5(hdf['testing'][subject], fa_img, 'fa_volume')

        print("Done!")

if __name__ == '__main__':
    main()
