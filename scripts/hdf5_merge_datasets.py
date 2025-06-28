import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm


def _build_args():
    parser = argparse.ArgumentParser(
        description='Merge datasets in HDF5 format')
    parser.add_argument('input_files', type=str, nargs='+', help='Input files')
    parser.add_argument('output_file', type=str, help='Output file')
    return parser.parse_args()


def main():
    args = _build_args()
    input_files = args.input_files
    output_file = args.output_file

    assert len(input_files) > 0, 'No input files provided'

    inputs_info = []
    output_handle = h5py.File(output_file, 'w')
    datashape = None
    try:
        # Prepare the data
        for input_file in input_files:
            assert os.path.exists(
                input_file), f'Input file {input_file} does not exist'
            input_handle = h5py.File(input_file, 'r')
            train_length = len(input_handle['train/scores'])
            test_length = len(input_handle['test/scores'])

            if datashape is None:
                datashape = (
                    input_handle['test/data'].shape[1], input_handle['test/data'].shape[2])

            input_info = {
                'handle': input_handle,
                'train_length': train_length,
                'test_length': test_length
            }
            inputs_info.append(input_info)

        # Create new datasets
        train_length = sum([input_info['train_length']
                            for input_info in inputs_info])
        test_length = sum([input_info['test_length']
                           for input_info in inputs_info])

        print(f"resulting train length: {train_length}")
        print(f"resulting test length: {test_length}")

        output_handle['version'] = 1
        output_handle['nb_points'] = inputs_info[0]['handle'].attrs['nb_points']

        train_group = output_handle.create_group('train')
        train_group.create_dataset('data', shape=(
            train_length, datashape[0], datashape[1]), dtype=np.float32)
        train_group.create_dataset('scores', (train_length,))

        test_group = output_handle.create_group('test')
        test_group.create_dataset('data', shape=(
            test_length, datashape[0], datashape[1]), dtype=np.float32)
        test_group.create_dataset('scores', (test_length,))

        # Prepare the indices where to add the data
        train_indices = np.arange(train_length)
        test_indices = np.arange(test_length)

        for input_info in tqdm(inputs_info):
            # Add training data
            train_data = input_info['handle']['train/data']
            train_scores = input_info['handle']['train/scores']
            train_length = input_info['train_length']

            train_group['data'][train_indices[:train_length]] = train_data
            train_group['scores'][train_indices[:train_length]] = train_scores

            # Add test data
            test_data = input_info['handle']['test/data']
            test_scores = input_info['handle']['test/scores']
            test_length = input_info['test_length']

            test_group['data'][test_indices[:test_length]] = test_data
            test_group['scores'][test_indices[:test_length]] = test_scores

            # Update indices for next documents
            train_indices = train_indices[train_length:]
            test_indices = test_indices[test_length:]

            # Close the input file
            input_info['handle'].close()

    finally:
        output_handle.close()
        for input_info in inputs_info:
            input_info['handle'].close()

    print(f"Data merged in {output_file}")


if '__main__' == __name__:
    main()
