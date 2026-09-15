import h5py
from pathlib import Path
import torch
import numpy as np
import glob
import cv2
from scripts.rotation_transformer import RotationTransformer

# === 配置项 ===
input_dir = "~/leapUMI-dataset/rotation_gt_100demo"
input_pattern = "leap_action_*.hdf5"
output_file = "~/leapUMI-dataset/training_dataset/rotation_gt.hdf5"
crop_range_txt = "crop_ranges.txt"  # 示例配置路径
norm_gripper = False # false 
skip = False  # 如果为True则跳过不在范围内的文件，如果为False则保留所有文件

obs = [
    "actions",
    "obs/agentview_image",
    "obs/arm_qpos",
    "obs/eef_rot",
    "obs/eef_pos",
    "obs/gripper_qpos",
    "obs/time",
    "obs/traj_pos",
    "obs/traj_quat"
]

quat_2_ax = RotationTransformer("quaternion", "axis_angle")
mat_2_quat = RotationTransformer("matrix", "quaternion")


def normalize_pose(pose):
    INIT_POSE = np.full(16, 3.1415926)
    END_POSE = np.array(
        [
            3.1538644,
            3.6002529,
            5.0513988,
            3.7337093,
            3.0664277,
            3.7643888,
            4.5850687,
            4.092661,
            2.9329712,
            3.9177868,
            4.144816,
            4.2874765,
            4.862719,
            3.19068,
            3.6063888,
            4.5221753,
        ]
    )
    pose = np.asarray(pose)
    normed = (pose - INIT_POSE) / (END_POSE - INIT_POSE)
    return np.mean(normed)


def process_actions(actions_arr):
    if actions_arr.shape[1] == 23:
        pos = actions_arr[:, :3]
        quat = actions_arr[:, 3:7]
        # quat[:,3:7] is in xyzw order, change to wxyz
        quat = quat[:, [3, 0, 1, 2]]
        gripper = actions_arr[:, 7:]
        if norm_gripper:
            gripper = np.array([normalize_pose(g) for g in gripper])[:, None]
        ax = quat_2_ax.forward(torch.tensor(quat)).numpy()
        return np.concatenate((pos, ax, gripper), axis=1)
    else:
        return actions_arr.copy()


def process_eef_rot(eef_rot_arr):
    if eef_rot_arr.ndim == 3 and eef_rot_arr.shape[1:] == (3, 3):
        quat = [mat_2_quat.forward(torch.tensor(m)).numpy() for m in eef_rot_arr]
        return np.array(quat)
    elif eef_rot_arr.ndim == 2 and eef_rot_arr.shape[1] == 4:
        eef_quat = eef_rot_arr[:, [3, 0, 1, 2]]  # change to wxyz
        return eef_quat.copy()


def parse_demo_ranges(txt_path):
    selected = dict()
    with open(txt_path, "r") as f:
        for line in f:
            if line.strip():
                parts = list(map(int, line.strip().split()))
                demo_id = parts[0]
                if len(parts) == 1:
                    continue
                elif len(parts) == 2:
                    selected[demo_id] = (0, parts[1] + 1)
                elif len(parts) == 3:
                    selected[demo_id] = (parts[1], parts[2] + 1)
    return selected


demo_ranges = parse_demo_ranges(crop_range_txt)

with h5py.File(output_file, "w") as out_f:
    demo_counter = 0

    input_files = sorted(glob.glob(str(Path(input_dir) / input_pattern)))

    for input_file in input_files:
        # 从文件名提取实际的文件索引
        filename = Path(input_file).stem  # 获取不带扩展名的文件名
        try:
            # 提取 leapdata_ 后面的数字部分
            file_idx = int(filename.split("leap_action_")[1])
        except (IndexError, ValueError):
            print(f"[警告] 无法解析文件索引: {input_file}")
            continue

        if skip and file_idx not in demo_ranges:
            print(f"[跳过] demo {file_idx} 不在范围配置中")
            continue

        if file_idx in demo_ranges:
            start_idx, end_idx = demo_ranges[file_idx]
            print(f"[处理] {input_file} 选取帧区间: [{start_idx}, {end_idx})")
        else:
            # 当skip=False时，保留整个文件
            start_idx, end_idx = 0, None
            print(f"[处理] {input_file} 保留全部帧")

        with h5py.File(input_file, "r") as original_f:
            for demo_key in original_f["data"].keys():
                group_prefix = f"data/demo_{demo_counter}"

                for o in obs:
                    src_path = f"data/{demo_key}/{o}"
                    dst_path = f'{group_prefix}/{o.replace("eef_rot", "eef_quat")}'

                    arr = original_f[src_path][()]
                    if end_idx is not None:
                        arr = arr[start_idx:end_idx]
                    else:
                        arr = arr[start_idx:]

                    if o == "actions":
                        out_f[f"{group_prefix}/actions"] = process_actions(arr)

                    elif o == "obs/agentview_image":
                        resized = np.array([cv2.resize(img, (92, 92)) for img in arr])
                        out_f[f"{group_prefix}/obs/agentview_image"] = resized.astype(np.uint8)
		
                    elif o == "obs/eef_rot":
                        quat_arr = process_eef_rot(arr)
                        out_f[f"{group_prefix}/obs/eef_quat"] = quat_arr

                    elif o == "obs/gripper_qpos":
                        if norm_gripper:
                            gripper_qpos = np.array([normalize_pose(g) for g in arr])[
                                :, None
                            ]
                        else:
                            gripper_qpos = arr
                        out_f[f"{group_prefix}/obs/gripper_qpos"] = gripper_qpos
                    
                    elif o == "obs/traj_quat":
                        traj_quat_arr = arr[:, [3, 0, 1, 2]]  # change to wxyz
                        out_f[f"{group_prefix}/obs/traj_quat"] = traj_quat_arr

                    else:
                        out_f[dst_path] = arr


                demo_counter += 1

print(f"[完成] 数据已保存至: {output_file}")
