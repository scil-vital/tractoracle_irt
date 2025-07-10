import h5py
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Check if the dataset in the HDF5 file is shuffled."
)

parser.add_argument('dataset_file', type=str, help='Path to the HDF5 dataset file.')
parser.add_argument('--datagroup', type=str, default='streamlines',
                    help='Name of the group in the HDF5 file containing the dataset.')
args = parser.parse_args()

def is_dataset_shuffled(scores: np.ndarray) -> bool:
    """ Checks to make sure that the scores are somewhat shuffled in the dataset.
    Essentially, it should split the dataset into 8 different parts and count the
    number of scores==1 and scores==0 in each part. If the dataset is shuffled, the 
    number of scores==1 and scores==0 should be roughly equal in each part.
    """

    if len(scores) == 0:
        return False
    # Split the dataset into 8 parts
    num_parts = 8
    part_size = len(scores) // num_parts
    threshold = 0.05*part_size
    if part_size == 0:
        return False

    for i in range(num_parts):
        part = scores[i * part_size:(i + 1) * part_size]
        if len(part) == 0:
            continue
        nb_ones = np.sum(part == 1)
        nb_zeros = np.sum(part == 0)
        if nb_ones == 0 or nb_zeros == 0:
            return False
        if nb_ones < threshold or nb_zeros < threshold:
            print(f"Part {i} is too small: {nb_ones} ones, {nb_zeros} zeros")
            return False

        print(f"Part {i}: {np.sum(part == 1)} ({nb_ones / part_size * 100:.2f}%) ones, {np.sum(part == 0)} ({nb_zeros / part_size * 100:.2f}%) zeros")


    return True

if __name__ == '__main__':
    with h5py.File(args.dataset_file, 'r') as f:
        if args.datagroup not in f:
            raise ValueError(f"Group '{args.datagroup}' not found in the HDF5 file.")
        
        scores = f[args.datagroup]['scores'][:]
        
        if len(scores) == 0:
            print("The dataset is empty.")
        else:
            shuffled = is_dataset_shuffled(scores)
            if shuffled:
                print("The dataset is shuffled.")
            else:
                print("The dataset is NOT shuffled.")




