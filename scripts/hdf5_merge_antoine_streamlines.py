import h5py
import argparse
import os

# I have 3 files that I want to merge. Each of those HDF5 files have the following structure:
# - train.hdf5
#     - streamlines
#         - data
#         - scores
#     - metadata
#
# - test.hdf5
#     - streamlines
#         - data
#         - scores
#     - metadata
#
# - validation.hdf5
#     - streamlines
#         - data
#         - scores
#     - metadata
#
# I want to get the following single HDF5 file:
# - merged.hdf5
#     - train
#         - data
#         - scores
#     - test
#         - data
#         - scores
#     - valid
#         - data
#         - scores
#

def create_out_file(in_train_file, out_file):
    with h5py.File(out_file, 'w') as out_handle:

        out_handle.create_group('train')
        out_handle.create_group('test')
        out_handle.create_group('valid')

        with h5py.File(in_train_file, 'r') as in_handle:
            out_handle.attrs['nb_points'] = in_handle.attrs['nb_points']
            out_handle.attrs['version'] = 1

def transfer_dataset(in_file, out_file, name):
    print(f'Transferring dataset {name}...')
    with h5py.File(in_file, 'r') as in_handle:
        with h5py.File(out_file, 'a') as out_handle:
            in_group = in_handle['streamlines']
            out_group = out_handle[name]

            out_group.create_dataset('data', data=in_group['data'])
            out_group.create_dataset('scores', data=in_group['scores'])



def parse_args():
    parser = argparse.ArgumentParser(description="Merge datasets in HDF5 format")
    parser.add_argument('--train_file', required=True, type=str, help='Train file')
    parser.add_argument('--valid_file', required=True, type=str, help='Validation file')
    parser.add_argument('--test_file', required=True, type=str, help='Test file')
    parser.add_argument('--out_file', required=True, type=str, help='Output file')
    
    return parser.parse_args()

def main():
    args = parse_args()
    train_file = args.train_file
    valid_file = args.valid_file
    test_file = args.test_file
    out_file = args.out_file

    # Assert all the files exist
    assert os.path.exists(train_file), f'Train file {train_file} does not exist'
    assert os.path.exists(valid_file), f'Validation file {valid_file} does not exist'
    assert os.path.exists(test_file), f'Test file {test_file} does not exist'

    create_out_file(train_file, out_file)

    transfer_dataset(train_file, out_file, 'train')
    transfer_dataset(valid_file, out_file, 'valid')
    transfer_dataset(test_file, out_file, 'test')

if '__main__' == __name__:
    main()