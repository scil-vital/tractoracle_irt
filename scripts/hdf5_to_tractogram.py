import h5py
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
import numpy as np
from scilpy.viz.color import get_lookup_table

# This script takes in a HDF5 file that holds streamline data and scores.
# 'streamline/data' holds the streamline data and 'streamline/scores' holds the scores.
# The script should gather all streamlines within the HDF5 file and put them into a
# stateful tractogram object so that we can save it as a .trk file.
# The script should also gather all scores and assign each corresponding streamline to a color
# based on the score. The color should be red for a score of 0, green for a score of 1.
# The script should then save the tractogram as a .trk file.

def old_hdf5_to_tractogram(hdf5_path: str, reference_path: str, trk_path: str):
    # Load the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Get the streamline data and scores
        streamlines = np.array(hdf_file['streamlines/data'])
        scores = np.array(hdf_file['streamlines/scores'])
        
        number_scores_zero = np.sum(scores == 0)
        number_scores_one = np.sum(scores == 1)
        ratio_of_ones = number_scores_one / len(scores)
        print("Ratio of scores of 1:", ratio_of_ones)

        # Create a stateful tractogram object
        reference = nib.load(reference_path)
        sft = StatefulTractogram(streamlines, reference, Space.VOX)
        
        # Assign colors to streamlines based on scores
        cmap = get_lookup_table("jet")
        color = cmap(scores)[:, 0:3] * 255
       
        # Add the colors to the stateful tractogram
        sft.data_per_point['color'] = streamlines
        sft.data_per_point['color']._data = color
        
        # Save the tractogram as a .trk file
        save_tractogram(sft, trk_path)

def hdf5_to_tractogram(hdf5_path: str, reference_path: str, trk_path: str):
    # Load the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf_file:
        # Get the streamline data and scores
        train_streamlines = np.array(hdf_file['train/data'])
        train_scores = np.array(hdf_file['train/scores'])
        train_save_file = trk_path.replace('.trk', '_train.trk')

        packup_streamlines_and_scores(train_streamlines, train_scores, reference_path, train_save_file)
        del train_streamlines
        del train_scores

        test_streamlines = np.array(hdf_file['test/data'])
        test_scores = np.array(hdf_file['test/scores'])
        test_save_file = trk_path.replace('.trk', '_test.trk')

        packup_streamlines_and_scores(test_streamlines, test_scores, reference_path, test_save_file)
        del test_streamlines
        del test_scores

def packup_streamlines_and_scores(streamlines: np.ndarray, scores: np.ndarray, reference_path: str, save_path: str):
    # Create a stateful tractogram object
    print("Loading reference")
    reference = nib.load(reference_path)
    print("Buildling stateful tractogram")
    sft = StatefulTractogram(streamlines, reference, Space.VOX)
    
    # Assign colors to streamlines based on scores
    cmap = get_lookup_table("jet")
    color = (cmap(scores)[:, 0:3] * 255).astype(np.uint8)

    # Add the colors to the stateful tractogram
    print("Tiling colors")
    tmp = [np.tile([color[i]], (len(streamlines[i]), 1)) for i in range(len(streamlines))]
    del color
    print("Adding color to points")
    sft.data_per_point['color'] = tmp
    # sft.data_per_point['color'] = sft.streamlines
    # sft.data_per_point['color']._data = color

    # Save the tractogram as a .trk file
    print("Saving tractogram")
    breakpoint()
    save_tractogram(sft, save_path)
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert HDF5 file to Tractogram')
    parser.add_argument('hdf5_path', type=str, help='Path to the HDF5 file')
    parser.add_argument('reference_path', type=str, help='Path to the reference.')
    parser.add_argument('trk_path', type=str, help='Path to save the Tractogram')
    parser.add_argument('--old_format', action='store_true', help='Use old format for HDF5 file')
    args = parser.parse_args()

    if args.old_format:
        old_hdf5_to_tractogram(args.hdf5_path, args.reference_path, args.trk_path)
    else:
        hdf5_to_tractogram(args.hdf5_path, args.reference_path, args.trk_path)

    print("Tractogram saved to:", args.trk_path)

if __name__ == '__main__':
    main()
