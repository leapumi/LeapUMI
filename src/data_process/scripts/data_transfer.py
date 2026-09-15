import h5py
from pathlib import Path
import sys
import torch
import numpy as np
import glob
# sys.path.append(Path('/work/yyang1/diffusion_policy/').resolve().as_posix())
from rotation_transformer import RotationTransformer

dataset = '/media/yaxun/B197/teleop_data/success/data'

obs = ['actions', 'obs/agentview_image', 'obs/arm_qpos',
       'obs/eef_pos', 'obs/eef_rot', 'obs/gripper_qpos', 'obs/time']
quat_2_ax = RotationTransformer('quaternion', 'axis_angle')
mat_2_quat = RotationTransformer('matrix', 'quaternion')

# Find all leap_action_*.hdf5 files and sort by demo index
file_pattern = f'{dataset}/leap_action_*.hdf5'
hdf5_files = sorted(glob.glob(file_pattern),
                    key=lambda x: int(Path(x).stem.split('_')[-1]))

with h5py.File('./leap_action_fixed.hdf5', 'w') as new_f:
    # Create the 'data' group in the new file
    for demo_idx, filepath in enumerate(hdf5_files):
        with h5py.File(filepath, 'r') as original_f:
            # Assume each file has one demo under 'data/demo_0' or similar structure
            demo_keys = list(original_f['data'].keys())
            source_demo = demo_keys[0]  # Take the first (and likely only) demo

            for o in obs:
                if o == 'actions':
                    pos, quat, gripper = original_f[f'data/{source_demo}/actions'][:][:, :3].copy(
                    ), original_f[f'data/{source_demo}/actions'][:][:, 3:7].copy(), original_f[f'data/{source_demo}/actions'][:][:, 7:].copy()
                    ax = quat_2_ax.forward(torch.tensor(quat)).numpy()
                    action = np.concatenate((pos, ax, gripper), axis=1)

                    new_f[f'data/demo_{demo_idx}/actions'] = action
                elif o == 'obs/eef_rot':
                    quat = []
                    for j in range(len(original_f[f'data/{source_demo}/obs/eef_rot'])):
                        quat += [mat_2_quat.forward(torch.tensor(
                            original_f[f'data/{source_demo}/obs/eef_rot'][:].copy()[j].reshape(3, 3))).numpy()]
                    new_f[f'data/demo_{demo_idx}/obs/eef_quat'] = np.array(
                        quat)

                elif o == 'obs/gripper_qpos':
                    gripper_qpos = original_f[f'data/{source_demo}/obs/gripper_qpos'][:][..., None].copy(
                    )
                    new_f[f'data/demo_{demo_idx}/obs/gripper_qpos'] = gripper_qpos
                else:
                    new_f[f'data/demo_{demo_idx}/{o}'] = original_f[f'data/{source_demo}/{o}'][:]
