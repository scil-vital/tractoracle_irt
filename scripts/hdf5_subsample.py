import h5py
import numpy as np
from tqdm import tqdm

hdf5_file = "antoine-pft.hdf5"
out_file = "antoine-pft-eq-small-old.hdf5"

f = h5py.File(hdf5_file, 'r')

# Create and init the new hdf5 file
f2 = h5py.File(out_file, 'w')

f2.attrs['version'] = f.attrs['version']
f2.attrs['nb_points'] = f.attrs['nb_points']

train_group = f2.create_group('streamlines')
# test_group = f2.create_group('test')

#################
# Train dataset #
#################
# Subsample the training data
train_data, train_scores = np.asarray(f['train/data'], dtype=np.float32), np.asarray(f['train/scores'], dtype=np.float32)

indices = np.arange(len(train_scores))
pos_indices, neg_indices = indices[train_scores == 1], indices[train_scores == 0]

np.random.shuffle(pos_indices); np.random.shuffle(neg_indices)

new_pos_data, new_pos_scores = train_data[pos_indices[:10000]], train_scores[pos_indices[:10000]]
new_neg_data, new_neg_scores = train_data[neg_indices[:10000]], train_scores[neg_indices[:10000]]

new_train_data, new_train_scores = np.concatenate([new_pos_data, new_neg_data]), np.concatenate([new_pos_scores, new_neg_scores]) # Only adding the positive samples into the dataset

_perm = np.random.choice(len(new_train_scores), len(new_train_scores), replace=False)
new_train_data, new_train_scores = new_train_data[_perm], new_train_scores[_perm]

# Add the training data
train_data_ds = train_group.create_dataset('data', shape=new_train_data.shape)
train_scores_ds = train_group.create_dataset('scores', shape=new_train_scores.shape)
for i, (st, sc) in tqdm(enumerate(zip(new_train_data, new_train_scores)), total=len(new_train_data), desc="Adding train data"):
    train_data_ds[i] = st
    train_scores_ds[i] = sc

################
# Test dataset #
################

# Subsample the test data
# test_data, test_scores = np.asarray(f['test/data'], dtype=np.float32), np.asarray(f['test/scores'], dtype=np.float32)

# indices = np.arange(len(test_scores))
# pos_indices, neg_indices = indices[test_scores == 1], indices[test_scores == 0]

# np.random.shuffle(pos_indices); np.random.shuffle(neg_indices)

# new_pos_data, new_pos_scores = test_data[pos_indices[:1000]], test_scores[pos_indices[:1000]]
# new_neg_data, new_neg_scores = test_data[neg_indices[:1000]], test_scores[neg_indices[:1000]]

# new_test_data, new_test_scores = np.concatenate([new_pos_data, new_neg_data]), np.concatenate([new_pos_scores, new_neg_scores]) # Only adding the positive samples into the dataset
# _perm = np.random.choice(len(new_test_scores), len(new_test_scores), replace=False)
# new_test_data, new_test_scores = new_test_data[_perm], new_test_scores[_perm]


# # Add the testing data
# test_data_ds = test_group.create_dataset('data', shape=new_test_data.shape)
# test_scores_ds = test_group.create_dataset('scores', shape=new_test_scores.shape)
# # With tqdm
# for i, (st, sc) in tqdm(enumerate(zip(new_test_data, new_test_scores)), total=len(new_test_data), desc="Adding test data"):
#     test_data_ds[i] = st
#     test_scores_ds[i] = sc  

f.close()
f2.close()

print("Done. New dataset saved at {}".format(out_file))
