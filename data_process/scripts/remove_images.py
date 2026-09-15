import h5py
import numpy as np
import os
import glob
from tqdm import tqdm
import shutil
import tempfile


def copy_group_without_images(source_group, dest_group, exclude_keys=None):
    """
    Recursively copy HDF5 group contents, excluding specified keys
    """
    if exclude_keys is None:
        exclude_keys = ['agentview_image', 'depth_image']

    for key in source_group.keys():
        if key in exclude_keys:
            print(f"  Skipping {key}")
            continue

        if isinstance(source_group[key], h5py.Group):
            # Create subgroup and recursively copy
            dest_subgroup = dest_group.create_group(key)
            copy_group_without_images(
                source_group[key], dest_subgroup, exclude_keys)
        else:
            # Copy dataset
            dest_group.create_dataset(key, data=source_group[key][()])
            print(f"  Copied {key}")


def remove_images_from_hdf5(input_dir="data", output_dir=None):
    """
    Remove image data from HDF5 files

    Args:
        input_dir: Directory containing input HDF5 files
        output_dir: Directory to save processed files (if None, overwrites input files)
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all episode files matching the pattern
    episode_files = glob.glob(os.path.join(input_dir, "leap_action_*.hdf5"))
    print(f"Found {len(episode_files)} episode files in {input_dir}")
    episode_files.sort()  # Sort to ensure consistent ordering

    for file_path in tqdm(episode_files, desc="Processing files", unit="file"):
        print(f"\nProcessing file: {file_path}")

        # Determine output file path
        if output_dir:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
        else:
            output_path = file_path  # Overwrite original

        # Create a temporary file for the output
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5')
        temp_file.close()

        try:
            with h5py.File(file_path, 'r') as input_f:
                with h5py.File(temp_file.name, 'w') as output_f:
                    # Copy all groups and datasets except image data
                    copy_group_without_images(input_f, output_f)

            # Move temp file to final destination
            shutil.move(temp_file.name, output_path)
            print(f"Successfully saved to {output_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Clean up temp file if something went wrong
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    print(f"\nCompleted processing {len(episode_files)} files.")


if __name__ == '__main__':
    # Define input and output directories
    input_dir = "/media/yaxun/B197/teleop_data/success/data"
    output_dir = "data_wo_images"  # Set to None to overwrite original files

    remove_images_from_hdf5(input_dir, output_dir)
