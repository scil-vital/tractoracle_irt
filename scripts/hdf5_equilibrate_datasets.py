import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm

"""
This script allows to equilibrate by using stratified sampling to balance one dataset
to the same proportion of the other dataset.

For example, if the first dataset has a split of 40%/60% and the second dataset has a split
of 15%/85%, this script will balance the second dataset to have a 40%/60% split.

Each dataset is a HDF5 file with the following structure:
- 'train'
    - 'data' (N, 128, 3)
    - 'scores' (N,)
- 'test'
    - 'data' (N, 128, 3)
    - 'scores' (N,)

We want to change the train and test splits for the second dataset.
"""

unbalanceness_threshold = 0.5


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Equilibrate dataset')
    parser.add_argument('dataset1', type=str,
                        help='Path to the first dataset (an HDF5 file)')
    parser.add_argument(
        'dataset2', type=str, help='Path to the second dataset which is to be modified (an HDF5 file)')
    parser.add_argument('output', type=str,
                        help='Path to the output dataset (an HDF5 file)')
    parser.add_argument('--nb_splits', type=int, default=1,
                        help='Number of random splits that match the input data.')
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    with h5py.File(args.dataset1, 'r') as f1:
        print('Loading train data from dataset 1...')
        train_scores1 = np.asarray(f1['train']['scores'])
        print('Loading test data from dataset 1...')
        test_scores1 = np.asarray(f1['test']['scores'])

    with h5py.File(args.dataset2, 'r') as f2:
        print('Loading train data from dataset 2...')
        train_data2 = np.asarray(f2['train']['data'])
        train_scores2 = np.asarray(f2['train']['scores'])
        print('Loading test data from dataset 2...')
        test_data2 = np.asarray(f2['test']['data'])
        test_scores2 = np.asarray(f2['test']['scores'])
        f2_attrs = dict(f2.attrs)

    print("Done.")

    nb_splits = args.nb_splits
    assert nb_splits > 0 and nb_splits is not None, "nb_splits must be an int greater than 0."

    print('Matching balance of the train set. Generating {} splits...'.format(nb_splits))
    train_indices_to_keep_splits = match_balance(
        train_scores1, train_data2, train_scores2, nb_splits=nb_splits)

    del train_scores1

    print('Matching balance of the test set. Generating {} splits...'.format(nb_splits))
    test_indices_to_keep_splits = match_balance(
        test_scores1, test_data2, test_scores2, nb_splits=nb_splits)

    del test_scores1

    out_dirpath = os.path.dirname(args.output)
    for i, (train_indices_to_keep, test_indices_to_keep) in enumerate(zip(train_indices_to_keep_splits, test_indices_to_keep_splits)):
        # Craft split file name
        split_filepath = os.path.join(
            out_dirpath, "split_{}_{}".format(i, os.path.basename(args.output)))

        # Save into the new HDF5 file
        out = h5py.File(split_filepath, 'w')
        out.attrs['version'] = f2_attrs['version']
        out.attrs['nb_points'] = f2_attrs['nb_points']

        train_group = out.create_group('train')
        test_group = out.create_group('test')

        # Match the balance of the second train set to the first dataset's train set

        new_train_data, new_train_scores = train_data2[
            train_indices_to_keep], train_scores2[train_indices_to_keep]
        train_group.create_dataset('data',
                                   shape=new_train_data.shape,
                                   maxshape=(None, new_train_data.shape[1], new_train_data.shape[2]))
        train_group.create_dataset('scores',
                                   shape=new_train_scores.shape,
                                   maxshape=(None,))

        add_data_to_hdf5(train_group, new_train_data, new_train_scores)

        # Match the balance of the second test set tot eh first dataset's test set
        print('Matching balance of the test set')
        new_test_data, new_test_scores = test_data2[test_indices_to_keep], test_scores2[test_indices_to_keep]
        test_group.create_dataset('data',
                                  shape=new_test_data.shape,
                                  maxshape=(None, new_test_data.shape[1], new_test_data.shape[2]))
        test_group.create_dataset('scores',
                                  shape=new_test_scores.shape,
                                  maxshape=(None,))

        add_data_to_hdf5(test_group, new_test_data, new_test_scores)

        out.close()
        print('Dataset split {} was saved as {}'.format(i, split_filepath))


def add_data_to_hdf5(group, data, scores, sub_pbar_desc="", batch_size=1000):
    data_group = group['data']
    scores_group = group['scores']
    idx = np.arange(len(scores))

    num_batches = (len(idx) // batch_size) + (len(idx) % batch_size != 0)

    for batch_start in tqdm(range(0, len(idx), batch_size), desc=sub_pbar_desc, total=num_batches, leave=False):
        batch_end = min(batch_start + batch_size, len(idx))
        batch_idx = idx[batch_start:batch_end]
        batch_streamlines = data[batch_start:batch_end]
        batch_scores = scores[batch_start:batch_end]

        data_group[batch_idx] = batch_streamlines
        scores_group[batch_idx] = batch_scores


def match_balance(scores1: np.ndarray, data2: np.ndarray, scores2: np.ndarray, nb_splits: int = 1):
    perc_pos_1, perc_neg_1 = np.mean(scores1), 1 - np.mean(scores1)
    perc_pos_2, perc_neg_2 = np.mean(scores2), 1 - np.mean(scores2)

    len_dataset2 = len(scores2)
    nb_neg_dataset2 = len(scores2[scores2 == 0])
    nb_pos_dataset2 = len(scores2[scores2 == 1])

    unbalanceness_1 = np.abs(perc_pos_1 - perc_neg_1)
    unbalanceness_2 = np.abs(perc_pos_2 - perc_neg_2)
    unbalanceness_diff = unbalanceness_1 - unbalanceness_2

    # If the datasets are balanced equivalently, we don't need to do anything
    if np.abs(unbalanceness_diff) < unbalanceness_threshold:
        return data2, scores2
    # If the first dataset is more balanced than the second, we need to resample the majority class in the second dataset
    elif unbalanceness_diff < 0:
        class_to_resample = 'pos' if perc_pos_2 > perc_neg_2 else 'neg'
    # If the first dataset is less balanced than the second, we need to resample the minority class in the second dataset
    # (to basically unbalance the second dataset even more)
    elif unbalanceness_diff > 0:
        class_to_resample = 'neg' if perc_pos_2 > perc_neg_2 else 'pos'

    if class_to_resample == 'pos':
        new_dataset_total_size = int(nb_neg_dataset2 / perc_neg_1)
        num_samples_to_resample = new_dataset_total_size - nb_neg_dataset2
        initial_num_samples = int(len_dataset2 * perc_pos_2)
    else:
        new_dataset_total_size = int(nb_pos_dataset2 / perc_pos_1)
        num_samples_to_resample = new_dataset_total_size - nb_neg_dataset2
        initial_num_samples = int(len_dataset2 * perc_neg_2)

    print(
        f"Will resample the class \"{class_to_resample}\" from {initial_num_samples} to {num_samples_to_resample} num samples.")

    # Get the indices of the class to resample
    score_of_class_to_resample = 1 if class_to_resample == 'pos' else 0
    indices_of_other_class = np.arange(
        len(scores2))[scores2 != score_of_class_to_resample]
    indices_to_resample = np.arange(
        len_dataset2)[scores2 == score_of_class_to_resample]

    # Resample the indices
    resampled_indices_splits = []
    for _ in range(nb_splits):
        resampled_indices = np.random.choice(
            indices_to_resample, num_samples_to_resample, replace=False)
        resampled_indices_splits.append(resampled_indices)

    # Concatenate the resampled indices with the rest of the indices from the other class.
    indices_to_keep_splits = [np.concatenate(
        [indices_of_other_class, ri]) for ri in resampled_indices_splits]

    # Evaluate the new balance of the dataset
    perc_pos_new = np.mean(scores2[indices_to_keep_splits[0]])
    perc_neg_new = 1 - perc_pos_new
    print('New balance of the set: {:.2f}% pos, {:.2f}% neg'.format(
        perc_pos_new * 100, perc_neg_new * 100))
    print('Difference in unbalanceness: {:.2f}'.format(
        (perc_pos_new - perc_neg_new) - unbalanceness_1))
    assert np.abs((perc_pos_new - perc_neg_new) -
                  unbalanceness_1) < unbalanceness_threshold, "Balance between the two datasets is not the same."

    return indices_to_keep_splits


if __name__ == '__main__':
    main()
