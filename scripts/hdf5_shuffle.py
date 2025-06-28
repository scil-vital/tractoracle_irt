import h5py
import numpy as np
import sys
import os
from tqdm import tqdm

def shuffle_hdf5_streamlines_on_disk(input_file, chunk_size=1024):
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_shuffled{ext}"

    with h5py.File(input_file, 'r') as f_in:
        data_ds = f_in['streamlines/data']
        scores_ds = f_in['streamlines/scores']

        N = data_ds.shape[0]
        assert scores_ds.shape[0] == N, "Data and scores must have the same length"

        # Create the shuffle permutation
        perm = np.random.permutation(N)

        with h5py.File(output_file, 'w') as f_out:
            grp = f_out.create_group('streamlines')
            shuffled_data = grp.create_dataset('data', shape=data_ds.shape, dtype=data_ds.dtype)
            shuffled_scores = grp.create_dataset('scores', shape=scores_ds.shape, dtype=scores_ds.dtype)

            # Process in chunks with progress bar
            with tqdm(total=N, desc="Shuffling", unit="samples") as pbar:
                for i in range(0, N, chunk_size):
                    end = min(i + chunk_size, N)
                    data_chunk = data_ds[i:end]
                    scores_chunk = scores_ds[i:end]
                    target_indices = perm[i:end]

                    # Sort target indices so we can write in bulk
                    sorted_idx = np.argsort(target_indices)
                    sorted_targets = target_indices[sorted_idx]
                    sorted_data = data_chunk[sorted_idx]
                    sorted_scores = scores_chunk[sorted_idx]

                    # Write sorted chunk to sorted target locations
                    shuffled_data[sorted_targets] = sorted_data
                    shuffled_scores[sorted_targets] = sorted_scores

                    pbar.update(end - i)

    print(f"\nShuffled file written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python shuffle_hdf5_large.py <input_file.h5>")
        sys.exit(1)

    shuffle_hdf5_streamlines_on_disk(sys.argv[1])
