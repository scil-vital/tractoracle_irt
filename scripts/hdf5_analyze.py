import h5py
import numpy as np
from tqdm import tqdm

# This script will load an HDF5 file containing a dataset of streamlines and their associated scores.
# The data is organized as follows:
# streamlines: {datagroup}/data
# scores: {datagroup}/scores
# The script will print the number of streamlines available in the dataset.
# It will also make statistics on the number of positives and negative samples in the dataset with their respective percentages and counts.
# {datagroup} = {'streamlines', 'train', 'test'}
def check_for_nulls(streamlines: np.ndarray, scores: np.ndarray):
    nb_streamlines = len(streamlines)

    # First check how many streamlines have a coordinate of (0, 0, 0) anywhere in their path
    have_zero = np.any(streamlines == 0, axis=(1, 2))
    nb_have_zero = np.sum(have_zero)
    print("Number of streamlines with a zero coordinate ({:.2f}%): {}/{}".format((nb_have_zero/nb_streamlines)*100, nb_have_zero, nb_streamlines))

    start_by_zero = np.all(streamlines[:, 0, :] == 0, axis=1)
    nb_start_by_zero = np.sum(start_by_zero)

    print("Number of streamlines starting by zero ({:.2f}%): {}/{}".format((nb_start_by_zero/nb_streamlines)*100, nb_start_by_zero, nb_streamlines))

    start_by_zero_idxs = np.arange(len(streamlines))
    start_by_zero_idxs = start_by_zero_idxs[start_by_zero] # Indexes of streamlines that start by zero shape: (N,)

    # Count the number of zeros in the streamlines.
    num_zero_points_per_streamline = np.sum(np.all(streamlines[start_by_zero_idxs] == 0, axis=2), axis=1)
    print("Number of zero points per streamline:", num_zero_points_per_streamline)

    # 
    streamlines_idx = np.arange(128)
    streamlines_idx = np.tile(streamlines_idx, nb_start_by_zero).reshape(-1, 128)
    
    # Find the indices of each non-null point in the streamlines starting by zero.
    non_null_idx = streamlines_idx[np.all((streamlines[start_by_zero_idxs, :, :] != 0), axis=2)]
    print("Indices of non-null points in streamlines starting by zero:", non_null_idx)

    # Some streamlines are all zeros, print their associated scores (N, 128, 3)
    full_zero_streamlines = np.all(streamlines == 0, axis=(1, 2))
    
    full_zero_indices = np.arange(nb_streamlines)[full_zero_streamlines]
    print("Indices where all points in the streamlines are zeros:", full_zero_indices)
    
    full_zero_streamlines_scores = scores[full_zero_indices]
    print(f"Scores of streamlines where all points are zeros ({full_zero_streamlines_scores.sum()} positive, {len(full_zero_streamlines_scores) - full_zero_streamlines_scores.sum()} negative): {full_zero_streamlines_scores}")


def print_balance(scores: np.ndarray):
    # Print the number of streamlines
    print(f"Number of streamlines: {len(scores)}")

    # Compute the number of positive and negative samples
    num_positives = np.sum(scores)
    num_negatives = len(scores) - num_positives

    # Compute the percentage of positive and negative samples
    percentage_positives = (num_positives / len(scores)) * 100
    percentage_negatives = (num_negatives / len(scores)) * 100

    # Print the statistics
    print(f"Number of positive samples: {num_positives} ({percentage_positives:.2f}%)")
    print(f"Number of negative samples: {num_negatives} ({percentage_negatives:.2f}%)")

def check_scores(scores: np.ndarray):
    # Create an histogram of the scores.
    # The scores are expected to be distributed between 0 and 1.
    # The histogram will bin the scores in 10 bins.
    import matplotlib.pyplot as plt

    # Create the histogram
    plt.hist(scores, bins=10, edgecolor="black")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Histogram of scores")

    # Save the figure as check_scores.png
    plt.savefig("check_scores.png")
    
def count_zeros(streamlines: h5py.Dataset):
    total_count = 0
    
    # Count the number of zeros in the streamlines
    # by processing in batches
    batch_size = 1000
    nb_batches = len(streamlines) // batch_size
    for i in tqdm(range(nb_batches)):
        start = i*batch_size
        end = min((i+1)*batch_size, len(streamlines))
        batch = streamlines[start: end]

        # Count the number of streamlines with zeros
        count = np.sum(np.all(batch == 0, axis=(1, 2)))
        total_count += count

    print("Number of streamlines with zeros:", total_count)
    

def main(file, operation, datagroup):
    with h5py.File(file, "r") as hdf5_file:
        streamlines = hdf5_file[f"{datagroup}/data"]
        scores = hdf5_file[f"{datagroup}/scores"]

        if operation == "balance":
            print_balance(np.asarray(scores))
        elif operation == "scores":
            check_scores(np.asarray(scores))
        elif operation == "zeros":
            check_for_nulls(np.asarray(streamlines), np.asarray(scores))
        elif operation == "count_zeros":
            count_zeros(streamlines)
        else:
            print("Operation not supported. Please use 'balance'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze the content of an HDF5 file containing streamlines and scores.")
    parser.add_argument("file", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--datagroup", type=str, default="train", choices=['streamlines', 'train', 'valid', 'test'], help="Name of the group containing the streamlines and scores. Default is 'streamlines'.")
    parser.add_argument("--operation", type=str, default="balance", help="Operation to perform on the HDF5 file. Default is balance.")
    args = parser.parse_args()
    main(args.file, args.operation, args.datagroup)

