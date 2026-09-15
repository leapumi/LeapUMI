import h5py
import numpy as np
import sys
import os

import glob
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm


def leap_hand_diffusion_data():

    # Define the base path where episode data is stored
    base_path = "data"
    output_file = './leap_hand_15cases.hdf5'

    # Find all episode directories matching the pattern
    episode_files = glob.glob(os.path.join(base_path, "leap_action_*.hdf5"))
    print(f"Found {len(episode_files)} episode directories.")
    episode_files.sort()  # Sort to ensure consistent ordering

    # Output file path

    with h5py.File(output_file, 'w') as output_f:
        global_demo_id = 0

        for file_path in tqdm(episode_files, desc="Processing demos", unit="demo"):
            with h5py.File(file_path, 'r') as f:
                print(f"Processing file: {file_path}")

                # 检查文件结构
                if 'data' in f and 'demo' in f['data']:
                    demo_group = f['data/demo']

                    # 检查是否直接包含数据
                    if 'actions' in demo_group:
                        # 单个episode的情况
                        actions_data = demo_group['actions'][()]
                        obs_gripper = demo_group['obs/gripper_qpos'][()]

                        print(
                            f"demo_{global_demo_id} - actions shape: {actions_data.shape}, obs_gripper shape: {obs_gripper.shape}")

                        for i in range(len(actions_data)):
                            if i == 0:
                                # 第一个action不变
                                pass
                            else:
                                # 其余action的后16维改为前一个时刻的gripper状态
                                actions_data[i][-16:] = obs_gripper[i-1]

                        # Create dataset
                        demo_key = f'demo_{global_demo_id}'
                        output_f.create_dataset(
                            f'data/{demo_key}/actions', data=actions_data)
                        output_f.create_dataset(
                            f'data/{demo_key}/obs/time', data=demo_group['obs/time'][()])
                        output_f.create_dataset(
                            f'data/{demo_key}/obs/arm_qpos', data=demo_group['obs/arm_qpos'][()])
                        output_f.create_dataset(
                            f'data/{demo_key}/obs/eef_pos', data=demo_group['obs/eef_pos'][()])
                        output_f.create_dataset(
                            f'data/{demo_key}/obs/eef_quat', data=demo_group['obs/eef_rot'][()])
                        output_f.create_dataset(
                            f'data/{demo_key}/obs/gripper_qpos', data=obs_gripper)

                        rgb = demo_group['obs/agentview_image'][()]
                        # 确保图片为 uint8
                        print('[DEBUG]', rgb.dtype, rgb.shape)
                        rgb = np.array([cv2.resize(img, (92, 92))
                                       for img in rgb])
                        print('[DEBUG]', rgb.dtype, rgb.shape)
                        output_f.create_dataset(
                            f'data/{demo_key}/obs/agentview_image', data=rgb)

                        # depth = demo_group['obs/depth_image'][()]
                        # depth = np.array([cv2.resize(img, (92, 92))
                        #                  for img in depth])
                        # output_f.create_dataset(
                        #     f'data/{demo_key}/obs/depth_image', data=depth)

                        global_demo_id += 1

                    # else:
                    #     # 多个episode的情况（如果有的话）
                    #     print("Multiple demos structure detected")
                    #     # 这里可以添加处理多个demo的逻辑

                else:
                    print(f"Unexpected file structure in {file_path}")

        print(f"Total demos processed: {global_demo_id}")


if __name__ == '__main__':
    leap_hand_diffusion_data()
