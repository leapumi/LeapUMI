import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
from scipy.spatial.transform import Rotation

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()


def quat_wxyz_to_axis_angle(quat_wxyz):
    """
    Convert quaternion (wxyz order) to axis-angle representation
    quat_wxyz: (..., 4) - [w, x, y, z] format
    Returns: (..., 3) - axis-angle representation
    """
    quat_wxyz = np.asarray(quat_wxyz)
    original_shape = quat_wxyz.shape[:-1]
    quat_flat = quat_wxyz.reshape(-1, 4)
    
    # Convert from wxyz to xyzw (scipy uses xyzw format)
    quat_xyzw = np.concatenate([quat_flat[:, 1:4], quat_flat[:, 0:1]], axis=-1)
    
    rot = Rotation.from_quat(quat_xyzw)
    axis_angle = rot.as_rotvec()
    
    return axis_angle.reshape(original_shape + (3,))


def rot6d_to_axis_angle(rot6d):
    """
    Convert 6D rotation representation to axis-angle representation
    rot6d: (..., 6) - 6D rotation representation (first two column vectors)
    Returns: (..., 3) - axis-angle representation
    """
    rot6d = np.asarray(rot6d)
    original_shape = rot6d.shape[:-1]
    rot6d_flat = rot6d.reshape(-1, 6)
    
    # Recover rotation matrix from 6D representation
    # 6D representation is concatenation of first two columns of rotation matrix
    a1 = rot6d_flat[:, 0:3]
    a2 = rot6d_flat[:, 3:6]
    
    # Orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    # Construct rotation matrix
    rot_mat = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)
    
    # Convert to axis-angle
    rot = Rotation.from_matrix(rot_mat)
    axis_angle = rot.as_rotvec()
    
    return axis_angle.reshape(original_shape + (3,))


class CustomUmiDataset(BaseDataset):
    """
    Custom UMI Dataset Loader
    
    Adapted for the following dataset format:
    - agentview_image: (N, H, W, 3) -> camera0_rgb
    - eef_pos: (N, 3) -> robot0_eef_pos
    - eef_quat: (N, 4) [wxyz order] -> robot0_eef_rot_axis_angle (needs conversion)
    - gripper_qpos: (N, 16) -> robot0_gripper_width
    - action: (N, 25) -> pos:3 + rot6d:6 + gripper:16
    
    Output action format: pos:3 + rot6d:6 + gripper:16 = 25 dims
    
    Purpose of demo_start_pose/demo_end_pose:
    - demo_start_pose: Used to calculate pose features relative to episode start (wrt_start), provides task progress information
    - demo_end_pose: Records episode end pose, currently not directly used in code
    - These keys are optional, only used when wrt_start features are needed in config
    """
    
    # Configuration: gripper dimension
    GRIPPER_DIM = 16
    
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        # Load custom dataset
        print(f"Loading custom dataset from: {dataset_path}")
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            root = zarr.group(store=zip_store)
            
            # Read original data
            data_group = root['data']
            meta_group = root['meta']
            
            # Read all data into memory
            agentview_image = data_group['agentview_image'][:]
            eef_pos = data_group['eef_pos'][:]
            eef_quat = data_group['eef_quat'][:]  # wxyz order
            gripper_qpos = data_group['gripper_qpos'][:]  # (N, 16)
            action_raw = data_group['action'][:]  # (N, 25): pos:3 + rot6d:6 + gripper:16
            episode_ends = meta_group['episode_ends'][:]
        
        print(f"Dataset loaded: {len(episode_ends)} episodes, {len(eef_pos)} steps")
        print(f"  - agentview_image: {agentview_image.shape}")
        print(f"  - eef_pos: {eef_pos.shape}")
        print(f"  - eef_quat: {eef_quat.shape} (wxyz order)")
        print(f"  - gripper_qpos: {gripper_qpos.shape}")
        print(f"  - action: {action_raw.shape} (pos:3 + rot6d:6 + gripper:16)")
        
        # Convert data format
        # 1. Convert quaternion (wxyz) to axis-angle
        eef_rot_axis_angle = quat_wxyz_to_axis_angle(eef_quat)
        print(f"  - eef_rot_axis_angle (converted): {eef_rot_axis_angle.shape}")
        
        # 2. Gripper state (keep all 16 dimensions)
        gripper_width = gripper_qpos  # shape: (N, 16)
        
        # 3. Process action: pos:3 + rot6d:6 + gripper:16 = 25
        action_pos = action_raw[:, 0:3]           # Position (N, 3)
        action_rot6d = action_raw[:, 3:9]         # 6D rotation (N, 6)
        action_gripper = action_raw[:, 9:]        # Gripper (all 16 dims) (N, 16)
        
        # Convert 6D rotation to axis-angle
        action_rot_axis_angle = rot6d_to_axis_angle(action_rot6d)
        
        # Combine into action: [pos(3) + rot(3) + gripper(16)] = 22 dims
        action = np.concatenate([action_pos, action_rot_axis_angle, action_gripper], axis=-1)
        print(f"  - action (converted): {action.shape} (pos:3 + rot:3 + gripper:16)")
        
        # Create in-memory ReplayBuffer
        replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
        
        # Add data by episode
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            episode_data = {
                'camera0_rgb': agentview_image[start:end],
                'robot0_eef_pos': eef_pos[start:end].astype(np.float32),
                'robot0_eef_rot_axis_angle': eef_rot_axis_angle[start:end].astype(np.float32),
                'robot0_gripper_width': gripper_width[start:end].astype(np.float32),
                'action': action[start:end].astype(np.float32),
            }
            
            # Add demo_start_pose and demo_end_pose (for wrt_start feature, optional)
            start_pose = np.concatenate([eef_pos[start], eef_rot_axis_angle[start]])
            end_pose = np.concatenate([eef_pos[end-1], eef_rot_axis_angle[end-1]])
            episode_length = end - start
            episode_data['robot0_demo_start_pose'] = np.tile(start_pose, (episode_length, 1)).astype(np.float32)
            episode_data['robot0_demo_end_pose'] = np.tile(end_pose, (episode_length, 1)).astype(np.float32)
            
            replay_buffer.add_episode(data=episode_data, compressors=None)
        
        print(f"ReplayBuffer created with {replay_buffer.n_episodes} episodes")
        
        # Parse shape_meta
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            # Parse obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # Parse horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # Parse latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # Parse down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # Parse action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        # Validation set split
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        # Set sampler lowdim keys
        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                if query_key in shape_meta['obs']:
                    key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                    key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                    key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        # Create sampler
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        
        # Save attributes
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Iterate dataset to get normalization statistics
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=8,
        )
        for batch in tqdm(dataloader, desc='Computing normalization statistics'):
            for key in self.lowdim_keys:
                if key in batch['obs']:
                    data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            if len(data_cache[key]) == 0:
                continue
            data_cache[key] = np.concatenate(data_cache[key])
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # Action normalization: pos(3) + rot6d(6) + gripper(16) = 25
        if len(data_cache['action']) > 0:
            action_normalizers = list()
            for i in range(self.num_robot):
                # Each robot's action dimension: 9 (pos+rot6d) + 16 (gripper) = 25
                robot_action_start = i * 25
                # pos: 3 dims
                action_normalizers.append(get_range_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., robot_action_start: robot_action_start + 3])))
                # rot6d: 6 dims
                action_normalizers.append(get_identity_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., robot_action_start + 3: robot_action_start + 9])))
                # gripper: 16 dims
                action_normalizers.append(get_range_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., robot_action_start + 9: robot_action_start + 25])))
            normalizer['action'] = concatenate_normalizer(action_normalizers)

        # Obs normalization
        for key in self.lowdim_keys:
            if key not in data_cache or len(data_cache[key]) == 0:
                continue
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                this_normalizer = get_range_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        # Image normalization
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
            
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        
        # Process RGB images
        for key in self.rgb_keys:
            if key not in data:
                continue
            # Move channel from last to second dimension
            # T,H,W,C -> T,C,H,W
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            del data[key]
            
        # Process low-dimensional data
        for key in self.sampler_lowdim_keys:
            if key in data:
                obs_dict[key] = data[key].astype(np.float32)
                del data[key]

        # Process each robot's pose
        # Original action format: pos(3) + rot(3) + gripper(16) = 22
        # Output action format: pos(3) + rot6d(6) + gripper(16) = 25
        actions = list()
        for robot_id in range(self.num_robot):
            pos_key = f'robot{robot_id}_eef_pos'
            rot_key = f'robot{robot_id}_eef_rot_axis_angle'
            
            if pos_key not in obs_dict or rot_key not in obs_dict:
                continue
            
            # Original action: each robot occupies 22 dims: pos(3) + rot(3) + gripper(16)
            robot_action_start = robot_id * 22
            
            # Convert pose to matrix
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[pos_key],
                obs_dict[rot_key]
            ], axis=-1))
            # Take pos(3) + rot(3) = 6 dims to build action matrix
            action_mat = pose_to_mat(data['action'][..., robot_action_start: robot_action_start + 6])

            # Convert to relative pose
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)

            # Convert to 10D pose representation (pos:3 + rot6d:6 + ...), take first 9 dims
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)

            # gripper: 16 dims
            action_gripper = data['action'][..., robot_action_start + 6: robot_action_start + 22]
            
            # Concatenate: pos(3) + rot6d(6) + gripper(16) = 25 dims
            actions.append(np.concatenate([action_pose[:, :9], action_gripper], axis=-1))

            # Update obs_dict
            obs_dict[pos_key] = obs_pose[:, :3]
            obs_dict[rot_key] = obs_pose[:, 3:9]

        if len(actions) > 0:
            data['action'] = np.concatenate(actions, axis=-1)

        # Delete unnecessary demo_pose
        del_keys = [key for key in obs_dict if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose')]
        for key in del_keys:
            del obs_dict[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
