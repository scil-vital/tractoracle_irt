import h5py
import numpy as np
import argparse

def cp_attrs(src, out):
    out.attrs['nb_points'] = src.attrs['nb_points']
    out.attrs['version'] = src.attrs['version']

def main():
    args = parse_args()

    with h5py.File(args.in_file, "r") as f:
        print("Creating train file ", args.train_out)
        with h5py.File(args.train_out, "w") as o1:
            o1.create_group("streamlines")

            o1["streamlines"].create_dataset("data", data=f["train/data"])
            o1["streamlines"].create_dataset("scores", data=f["train/scores"])
            cp_attrs(f, o1)

        print("Creating valid file ", args.valid_out)
        with h5py.File(args.valid_out, "w") as o2:
            o2.create_group("streamlines")

            o2["streamlines"].create_dataset("data", data=f["valid/data"])
            o2["streamlines"].create_dataset("scores", data=f["valid/scores"])
            cp_attrs(f, o2)

        print("Creating test file ", args.test_out)
        with h5py.File(args.test_out, "w") as o3:
            o3.create_group("streamlines")

            o3["streamlines"].create_dataset("data", data=f["test/data"])
            o3["streamlines"].create_dataset("scores", data=f["test/scores"])
            cp_attrs(f, o3)

    print("done")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str, help="in hdf5 file to separate into train/valid/test")
    parser.add_argument("train_out", type=str)
    parser.add_argument("valid_out", type=str)
    parser.add_argument("test_out", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    main()