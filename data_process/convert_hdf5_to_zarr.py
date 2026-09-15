"""
Convert HDF5 dataset to Zarr format compatible with UMI training pipeline.

Usage:
    python convert_hdf5_to_zarr.py <input.hdf5> <output.zarr.zip> [--split]

Example:
    # Write one 224x448 image as camera0_rgb
    python convert_hdf5_to_zarr.py dataset.hdf5 dataset.zarr.zip

    # Split a 224x448 image into two 224x224 images
    python convert_hdf5_to_zarr.py dataset.hdf5 dataset.zarr.zip --split

Input HDF5 structure:
    /data/demo_0/
        - action: (N, D) [will be ignored]
        - agentview_image: (N, H, W, 3)
        - eef_pos: (N, 3)
        - eef_quat: (N, 4) [wxyz order]
        - gripper_qpos: (N, 16)
    /data/demo_1/
        ...

Output Zarr structure:
    /data/
        - camera0_rgb: (Total_N, 224, 448, 3), without --split
          or
        - camera0_rgb_left: (Total_N, 224, 224, 3), with --split
        - camera0_rgb_right: (Total_N, 224, 224, 3), with --split
        - robot0_eef_pos: (Total_N, 3)
        - robot0_eef_rot_axis_angle: (Total_N, 3) [axis-angle]
        - robot0_gripper_width: (Total_N, 16)
        - robot0_demo_start_pose: (Total_N, 6)
        - robot0_demo_end_pose: (Total_N, 6)
    /meta/
        - episode_ends: (num_episodes,)

Note: The 'action' key will NOT be included in the output Zarr file.
      Actions will be constructed dynamically during training from:
      robot0_eef_pos + robot0_eef_rot_axis_angle + robot0_gripper_width
"""

import sys
import os
import argparse
import h5py
import zarr
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer

register_codecs()


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 448
SPLIT_IMAGE_WIDTH = IMAGE_WIDTH // 2


def quat_wxyz_to_axis_angle(quat_wxyz):
    """
    Convert quaternion (wxyz order) to axis-angle representation
    
    Args:
        quat_wxyz: (..., 4) - [w, x, y, z] format
    
    Returns:
        axis_angle: (..., 3) - axis-angle representation
    """
    quat_wxyz = np.asarray(quat_wxyz)
    original_shape = quat_wxyz.shape[:-1]
    quat_flat = quat_wxyz.reshape(-1, 4)
    
    # Convert from wxyz to xyzw (scipy uses xyzw format)
    quat_xyzw = np.concatenate([quat_flat[:, 1:4], quat_flat[:, 0:1]], axis=-1)
    
    rot = Rotation.from_quat(quat_xyzw)
    axis_angle = rot.as_rotvec()
    
    return axis_angle.reshape(original_shape + (3,))


def prepare_camera_images(images, split):
    """Resize HWC RGB frames to 224x448 and optionally split them in half."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "agentview_image must have shape (N, H, W, 3); "
            f"got {images.shape}"
        )

    if images.shape[1:3] == (IMAGE_HEIGHT, IMAGE_WIDTH):
        wide_images = images
    else:
        wide_images = np.stack([
            cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            for image in images
        ])

    wide_images = np.ascontiguousarray(wide_images)
    if not split:
        return {'camera0_rgb': wide_images}

    return {
        'camera0_rgb_left': np.ascontiguousarray(
            wide_images[:, :, :SPLIT_IMAGE_WIDTH, :]
        ),
        'camera0_rgb_right': np.ascontiguousarray(
            wide_images[:, :, SPLIT_IMAGE_WIDTH:, :]
        ),
    }


def load_hdf5_dataset(hdf5_path, split_images=False):
    """
    Load HDF5 dataset and organize it into episodes
    
    Args:
        hdf5_path: Path to input HDF5 file
    
    Returns:
        episodes: List of episode dictionaries
        total_frames: Total number of frames
    """
    print(f"Loading HDF5 dataset from: {hdf5_path}")
    
    episodes = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get all demo keys
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        print(f"Found {len(demo_keys)} demos")
        
        # Check first demo structure
        first_demo = data_group[demo_keys[0]]
        if 'obs' in first_demo:
            print("Detected structure: /data/demo_X/obs/...")
            has_obs_group = True
        else:
            print("Detected structure: /data/demo_X/...")
            has_obs_group = False
        
        for demo_key in tqdm(demo_keys, desc="Loading demos"):
            demo_group = data_group[demo_key]
            
            # Navigate to correct location based on structure
            if has_obs_group:
                obs_group = demo_group['obs']
            else:
                obs_group = demo_group
            
            # Read data (action is NOT included)
            agentview_image = obs_group['agentview_image'][:]
            eef_pos = obs_group['eef_pos'][:]
            eef_quat = obs_group['eef_quat'][:]  # Assuming wxyz order
            gripper_qpos = obs_group['gripper_qpos'][:]
            
            # Check quaternion order from first sample
            if len(episodes) == 0:
                sample_quat = eef_quat[0]
                quat_norm = np.linalg.norm(sample_quat)
                print(f"  First quaternion sample: {sample_quat}")
                print(f"  Quaternion norm: {quat_norm:.6f}")
                if abs(quat_norm - 1.0) > 0.01:
                    print(f"  ⚠️  Warning: Quaternion not normalized!")
            
            # Convert quaternion to axis-angle
            eef_rot_axis_angle = quat_wxyz_to_axis_angle(eef_quat)
            
            episode = {
                'robot0_eef_pos': eef_pos,
                'robot0_eef_rot_axis_angle': eef_rot_axis_angle,
                'robot0_gripper_width': gripper_qpos,
            }
            episode.update(prepare_camera_images(agentview_image, split_images))
            
            # Add demo start and end pose
            # Start pose: first frame's [pos(3), rot(3)] = 6D
            start_pose = np.concatenate([eef_pos[0], eef_rot_axis_angle[0]])
            # End pose: last frame's [pos(3), rot(3)] = 6D
            end_pose = np.concatenate([eef_pos[-1], eef_rot_axis_angle[-1]])
            
            # Tile to match episode length
            episode_length = len(eef_pos)
            episode['robot0_demo_start_pose'] = np.tile(start_pose, (episode_length, 1))
            episode['robot0_demo_end_pose'] = np.tile(end_pose, (episode_length, 1))
            
            episodes.append(episode)
    
    # Calculate total frames
    total_frames = sum(len(ep['robot0_eef_pos']) for ep in episodes)
    
    print(f"Loaded {len(episodes)} episodes with {total_frames} total frames")
    print(f"Note: 'action' key is NOT included (will be constructed during training)")
    
    return episodes, total_frames


def create_zarr_dataset(episodes, output_path, split_images=False):
    """
    Create Zarr dataset from episodes
    
    Args:
        episodes: List of episode dictionaries
        output_path: Path to output .zarr.zip file
    """
    print(f"\nCreating Zarr dataset: {output_path}")
    
    # Create temporary directory for Zarr
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix='zarr_temp_')
    temp_zarr_path = os.path.join(temp_dir, 'dataset.zarr')
    
    try:
        # Create ReplayBuffer in temporary directory
        print(f"  Creating temporary Zarr at: {temp_zarr_path}")
        replay_buffer = ReplayBuffer.create_empty_zarr(storage=temp_zarr_path)
        
        # Add episodes
        for i, episode in enumerate(tqdm(episodes, desc="Writing episodes")):
            # Convert to float32 for consistency
            # Note: 'action' is NOT included
            episode_data = {
                'robot0_eef_pos': episode['robot0_eef_pos'].astype(np.float32),
                'robot0_eef_rot_axis_angle': episode['robot0_eef_rot_axis_angle'].astype(np.float32),
                'robot0_gripper_width': episode['robot0_gripper_width'].astype(np.float32),
                'robot0_demo_start_pose': episode['robot0_demo_start_pose'].astype(np.float32),
                'robot0_demo_end_pose': episode['robot0_demo_end_pose'].astype(np.float32),
            }
            if split_images:
                episode_data['camera0_rgb_left'] = episode['camera0_rgb_left']
                episode_data['camera0_rgb_right'] = episode['camera0_rgb_right']
            else:
                episode_data['camera0_rgb'] = episode['camera0_rgb']
            
            # Use default compressors (JPEG-XL for images)
            replay_buffer.add_episode(data=episode_data, compressors='disk')
        
        # Compress to zip
        print(f"  Compressing to: {output_path}")
        
        # Use zarr.copy_store to copy from directory to ZipStore
        import zarr
        source_store = zarr.DirectoryStore(temp_zarr_path)
        with zarr.ZipStore(output_path, mode='w') as zip_store:
            zarr.copy_store(source_store, zip_store)
        
        print(f"✓ Zarr dataset created successfully!")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"  Cleaned up temporary files")


def verify_zarr_dataset(zarr_path):
    """
    Verify the created Zarr dataset
    
    Args:
        zarr_path: Path to .zarr.zip file
    """
    print(f"\nVerifying Zarr dataset: {zarr_path}")
    
    with zarr.ZipStore(zarr_path, mode='r') as zip_store:
        root = zarr.group(store=zip_store)
        
        print("\n【Data Structure】")
        print("  📁 /")
        print("  ├── 📁 data/")
        
        data_group = root['data']
        for key in sorted(data_group.array_keys()):
            arr = data_group[key]
            print(f"  │   ├── 📊 {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        print("  └── 📁 meta/")
        meta_group = root['meta']
        episode_ends = meta_group['episode_ends'][:]
        print(f"      └── 📊 episode_ends: shape={episode_ends.shape}, dtype={episode_ends.dtype}")
        
        # Calculate statistics
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        episode_lengths = episode_ends - episode_starts
        
        print(f"\n【Statistics】")
        print(f"  Total Episodes: {len(episode_ends)}")
        print(f"  Total Frames: {episode_ends[-1]}")
        print(f"  Episode Length: min={episode_lengths.min()}, max={episode_lengths.max()}, avg={episode_lengths.mean():.1f}")
        
        # Verify 'action' key is NOT present
        if 'action' in data_group.array_keys():
            print(f"\n⚠️  Warning: 'action' key found in dataset (should not be present)")
        else:
            print(f"\n✓ Confirmed: 'action' key is NOT present (will be constructed during training)")
        
        # Check first episode
        print(f"\n【First Episode Sample】")
        for image_key in ('camera0_rgb', 'camera0_rgb_left', 'camera0_rgb_right'):
            if image_key in data_group:
                print(f"  {image_key}[0]: shape={data_group[image_key][0].shape}")
        print(f"  robot0_eef_pos[0]: {data_group['robot0_eef_pos'][0]}")
        print(f"  robot0_eef_rot_axis_angle[0]: {data_group['robot0_eef_rot_axis_angle'][0]}")
        print(f"  robot0_gripper_width[0]: shape={data_group['robot0_gripper_width'][0].shape}")
        print(f"  robot0_demo_start_pose[0]: {data_group['robot0_demo_start_pose'][0]}")
        print(f"  robot0_demo_end_pose[0]: {data_group['robot0_demo_end_pose'][0]}")
        
    print("\n✓ Verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 dataset to Zarr format for UMI training"
    )
    parser.add_argument(
        'input_hdf5',
        type=str,
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        'output_zarr',
        type=str,
        help='Path to output .zarr.zip file'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification after conversion'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help=(
            'Resize frames to 224x448 and write camera0_rgb_left and '
            'camera0_rgb_right (224x224 each) instead of camera0_rgb'
        )
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_hdf5):
        print(f"Error: Input file not found: {args.input_hdf5}")
        sys.exit(1)
    
    # Check output file doesn't exist (to avoid overwriting)
    if os.path.exists(args.output_zarr):
        response = input(f"Output file already exists: {args.output_zarr}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            sys.exit(0)
        os.remove(args.output_zarr)
    
    # Load HDF5 dataset
    episodes, total_frames = load_hdf5_dataset(args.input_hdf5, split_images=args.split)
    
    # Create Zarr dataset
    create_zarr_dataset(episodes, args.output_zarr, split_images=args.split)
    
    # Verify dataset
    if not args.no_verify:
        verify_zarr_dataset(args.output_zarr)
    
    print(f"\n{'='*60}")
    print(f"Conversion completed successfully!")
    print(f"  Input:  {args.input_hdf5}")
    print(f"  Output: {args.output_zarr}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total Frames: {total_frames}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nUsage: python convert_hdf5_to_zarr.py <input.hdf5> <output.zarr.zip>")
        sys.exit(1)
    
    main()
