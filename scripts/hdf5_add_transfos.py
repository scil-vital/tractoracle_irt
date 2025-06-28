import h5py
import json
import numpy as np
import os
import argparse

from scilpy.io.utils import load_matrix_in_any_format
from tractoracle_irt.utils.utils import Timer
import nibabel as nib

DATASETS = ['training', 'validation', 'testing']
TRANSFORMATION = 'transformation'
DEFORMATION = 'deformation'


def parse_args():
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__)
    parser.add_argument('hdf5_file', type=str,
                        help='Path to the hdf5 file.')
    parser.add_argument('config_file', type=str,
                        help='Path to the json file.')
    return parser.parse_args()

def resolve_path(path):
    return os.path.abspath(os.path.expanduser(path))

def verify_all_paths(config):
    for dataset in DATASETS:
        for subject_id in config[dataset]:
            subject_config = config[dataset][subject_id]
            found_transformation = False
            found_deformation = False
            for key in subject_config:
                if key == TRANSFORMATION or key == DEFORMATION:
                    resolved_path = resolve_path(subject_config[key])
                    assert os.path.exists(resolved_path), "Path for subject {} and key {} doesn't exist (path: {})".format(subject_id, key, resolved_path)

                if key == TRANSFORMATION:
                    found_transformation = True
                if key == DEFORMATION:
                    found_deformation = True
            
            assert found_transformation, "No transformation found for subject {}".format(subject_id)
            assert found_deformation, "No deformation found for subject {}".format(subject_id)

def add_transfos_to_hdf5(ds_config, ds_hdf5):
    for subject_id in ds_config:
        with Timer("Processing subject {}".format(subject_id), newline=True, color='blue'):
            subject_config = ds_config[subject_id]
            hdf_subject = ds_hdf5[subject_id]

            found_transformation = False
            found_deformation = False
            
            print(f"config keys for subject {subject_config.keys()}")
            for key in subject_config:
                if key == TRANSFORMATION or key == DEFORMATION:
                    if hdf_subject.get('transformation_volume') is not None and key == TRANSFORMATION:
                        print(f"Subject {subject_id} already has a transformation_volume key in the hdf5 file. Skipping.")
                        found_transformation = True
                        continue
                    if hdf_subject.get('deformation_volume') is not None and key == DEFORMATION:
                        print(f"Subject {subject_id} already has a deformation_volume key in the hdf5 file. Skipping.")
                        found_deformation = True
                        continue

                    resolved_path = resolve_path(subject_config[key])

                    if key == TRANSFORMATION:
                        transfo = load_matrix_in_any_format(resolved_path)
                        transfo_volume_group = hdf_subject.create_group('transformation_volume')
                        transfo_volume_group.create_dataset('data', data=transfo)
                        found_transformation = True
                    elif key == DEFORMATION:
                        deformation_data = nib.load(resolved_path).get_fdata(dtype=np.float32)
                        deformation_data = np.squeeze(deformation_data)
                        deform_volume_group = hdf_subject.create_group('deformation_volume')
                        deform_volume_group.create_dataset('data', data=deformation_data)
                        found_deformation = True
                    else:
                        raise ValueError("Unknown key: {}".format(key))
                    
            assert found_transformation, "No transformation found for subject {}".format(subject_id)
            assert found_deformation, "No deformation found for subject {}".format(subject_id)

def main():
    args = parse_args()
    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)
        verify_all_paths(config)

        with h5py.File(args.hdf5_file, 'a') as f:

            assert DATASETS[0] in config
            assert DATASETS[1] in config
            assert DATASETS[2] in config

            for dataset in DATASETS:
                add_transfos_to_hdf5(config[dataset], f[dataset])


            

if __name__ == '__main__':
    main()