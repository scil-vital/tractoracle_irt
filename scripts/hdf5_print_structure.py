import h5py
import sys

def print_hdf5_structure(file_name):
    def print_group(group, indent=0):
        """Recursive function to print the group structure with indentation."""
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                print("  " * indent + f"[Group] {key}/")
                print_group(item, indent + 1)
            elif isinstance(item, h5py.Dataset):
                print("  " * indent + f"[Dataset] {key} - Shape: {item.shape}, Dtype: {item.dtype}")
            else:
                print("  " * indent + f"[Unknown] {key}")

    try:
        with h5py.File(file_name, 'r') as hdf_file:
            print(f"HDF5 File: {file_name}")
            print_group(hdf_file)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_hdf5_structure.py <hdf5_file>")
        sys.exit(1)
    
    file_name = sys.argv[1]
    print_hdf5_structure(file_name)