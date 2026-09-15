# LeapUMI Workflow

This directory contains three workflows: LeapUMI data collection, UMI diffusion-policy training, and real-robot deployment with a Kinova J2N6S300 arm and a LEAP Hand.

## 0. Shared Setup

Run the following commands from the workspace root:

```bash
export WS=/home/donglzh/leapumi_workspace
cd "$WS"
```

If the environments have not yet been created, use the environment definitions exported by this repository:

```bash
conda env create -f "$WS/src/teleop_env.yaml"
conda env create -f "$WS/src/umi_env.yaml"
```

These YAML files contain only Conda/Python dependencies. ROS Noetic, `cv_bridge`, camera drivers, the Kinova driver, and device udev rules must still be installed separately on the system.

Before using ROS packages for the first time, build and source the workspace:

```bash
cd "$WS"
source /opt/ros/noetic/setup.bash
catkin_make
source "$WS/devel/setup.bash"
```

Every terminal that uses ROS must subsequently source both `/opt/ros/noetic/setup.bash` and `$WS/devel/setup.bash`.

## 1. Data Collection

This workflow uses two independent scripts:

- `telekinesis/leapumi_system_control.py`: obtains hand-joint transforms from a Quest/Oculus headset and controls the LEAP Hand through PyBullet IK.
- `telekinesis/leapumi_dataset_recording.py`: reads LEAP Hand states, subscribes to Insta360 images and T265 poses, and writes HDF5 files.

### 1.1 Pre-collection Checklist

Prepare the following:

- A powered LEAP Hand. The scripts try `/dev/ttyUSB0`, `/dev/ttyUSB1`, and `/dev/ttyUSB2` in sequence at 4 Mbps.
- A Quest/Oculus headset with USB debugging enabled and a working ADB connection. It should appear as `device` after running `adb devices`.
- A running ROS master.
- Camera drivers that publish the following topics. This repository only subscribes to these topics; it does not include their publisher nodes.
  - `/insta360/image_raw` of type `sensor_msgs/Image`. Its encoding must be convertible to `rgb8` by `CvBridge`.
  - `/camera/odom/sample` of type `nav_msgs/Odometry`, published by a T265 or equivalent tracking node.

Verify that sensor data is available:

```bash
rostopic type /insta360/image_raw
rostopic type /camera/odom/sample
rostopic hz /insta360/image_raw
rostopic echo -n 1 /camera/odom/sample
```

### 1.2 Start ROS and Sensors

Start ROS in terminal A:

```bash
conda activate bidex_manus_teleop
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
roscore
```

Start the installed Insta360 and T265 drivers in terminals B and C. Launch-file names differ by driver, so use the commands for your installed drivers.

For example, run the following in two terminals:

```bash
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
roslaunch jsk_perception sample_insta360_air.launch
```

```bash
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
roslaunch realsense2_camera rs_t265.launch
```

Complete the `rostopic` checks from the previous section before continuing.

### 1.3 LeapUMI Control

Run the following in terminal D:

```bash
conda activate bidex_manus_teleop
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
cd "$WS"
adb devices
python src/telekinesis/leapumi_system_control.py
```

The script starts headless PyBullet IK, connects to the Quest headset, and enables the LEAP Hand to hold position. Press `q` to quit. Pressing `a` sends a hard-coded hand target pose; do not press it before completing safety validation.

### 1.4 LeapUMI Data Recording

The recording directory is hard-coded in the script as:

```text
/media/Common/leapUMI-dataset/watering_task_100demo
```

To use another disk or location, change `self.dataset_path` in `leapumi_dataset_recording.py` first. The directory is created automatically when it does not exist.

Run the following in terminal E:

```bash
conda activate bidex_manus_teleop
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
cd "$WS"
python src/telekinesis/leapumi_dataset_recording.py
```

Recording controls:

| Key | Action |
| --- | --- |
| `r` | Start buffering frames for the current episode. |
| `p` | Stop buffering and discard the current episode. |
| `s` | Mark the episode as successful and save it. |
| `q` | Exit the program. |

Each episode is saved as `leap_action_<id>.hdf5` and contains `actions`, T265 poses, 16-DoF LEAP Hand states, and `agentview_image`.

### 1.5 Recorded Data and Training Format

The recorder creates one HDF5 file per episode. Before training, inspect, crop/preprocess, and then convert the data to a Zarr Zip file. All scripts below are in `data_process`:

1. Use `hdf5_image_view.py` to inspect raw recordings. Its GUI lets you switch between demos and frames and view images, arm state, end-effector pose, LEAP Hand state, and actions.

   ```bash
   cd "$WS/src/data_process"
   conda activate umi
   python hdf5_image_view.py /absolute/path/leap_action_0.hdf5
   ```

2. Use `data_transfer_crop.py` to select data, crop episodes, and perform training-time preprocessing. Configure `input_dir`, `output_file`, `crop_range_txt`, `skip`, and `norm_gripper` at the top of the file. Each line in `crop_ranges.txt` follows `demo_id [start_frame] end_frame`; the end frame is included. The script merges selected episodes and performs the following operations: resizes images to `92×92`, renames/converts `eef_rot` to `eef_quat` in `wxyz` order, and converts 23-dimensional actions (position 3 + `xyzw` quaternion 4 + hand 16) to 22-dimensional actions (position 3 + axis-angle 3 + hand 16).

   ```bash
   cd "$WS/src/data_process"
   conda activate umi
   python data_transfer_crop.py
   ```

   `data_transfer_crop.py` imports `scripts.rotation_transformer` from the current directory, so it must be run from `data_process`. After processing, open the merged HDF5 file with the viewer from step 1 and verify the crop ranges, images, and numeric values.

3. Use `convert_hdf5_to_zarr.py` to convert the preprocessed HDF5 file into a UMI-readable `.zarr.zip` file. The converter changes `eef_quat` from `wxyz` to axis-angle and writes images, end-effector position/rotation, hand state, the start/end pose of each trajectory, and `episode_ends`. Actions are constructed dynamically from these fields during training.

   ```bash
   cd "$WS/src/data_process"
   conda activate umi
   python convert_hdf5_to_zarr.py \
     /absolute/path/training_dataset.hdf5 \
     /absolute/path/training_dataset.zarr.zip
   ```

   By default, the converter prints the dataset structure and statistics after conversion. Add `--no-verify` to skip this verification step.

4. Use `zarr_dataset_viewer.py` to inspect the final training dataset. By default, it prints the Zarr tree and episode statistics and opens a GUI. Use `--info-only` to print only the structure and statistics, which is suitable for a machine without a graphical desktop.

   ```bash
   cd "$WS/src/data_process"
   conda activate umi
   python zarr_dataset_viewer.py /absolute/path/training_dataset.zarr.zip
   # On a machine without a graphical desktop:
   python zarr_dataset_viewer.py /absolute/path/training_dataset.zarr.zip --info-only
   ```

The training configuration must match the fields and dimensions in the final Zarr file. This LEAP-UMI workflow uses a 16-dimensional hand state and converts the 22-dimensional action representation (position 3 + axis-angle 3 + hand 16) into a 25-dimensional representation (position 3 + rotation-6d 6 + hand 16) during training.

## 2. Training

Training code is located in `universal_manipulation_interface`. The example below assumes a Zarr Zip dataset whose fields match the selected task.

### 2.1 Select a Compatible Data Configuration

| Task configuration | Image input | Raw hand action |
| --- | --- | --- |
| `umi` | `camera0_rgb`, 224×224 | Single-DoF gripper; not compatible with 16-DoF LEAP Hand data. |
| `leapumi` | `camera0_rgb`, 160×320 | 16-DoF LEAP Hand; training outputs 25-dimensional actions. |
| `leapumi_split` | `camera0_rgb_left` and `camera0_rgb_right`, 224×224 each | 16-DoF LEAP Hand; training outputs 25-dimensional actions. |
| `leap_umi_abs` / `leap_umi_rel` | `camera0_rgb_left` and `camera0_rgb_right`, 224×224 each | 16-DoF LEAP Hand; training outputs 25-dimensional actions. |

Training and deployment in this project use `train_diffusion_unet_timm_leapumi_workspace.yaml`. It selects `task: leapumi_split` by default, and the corresponding task configuration file is present. Leave this setting unchanged.

`leapumi_split` requires two `224×224` image keys in the Zarr file: `camera0_rgb_left` and `camera0_rgb_right`. The current `data_process/convert_hdf5_to_zarr.py` writes only `camera0_rgb`, so its output cannot be used with this training configuration directly. Before training, split the wide image into equal left and right halves using the same convention as real deployment, and write both image keys.

### 2.2 Training

```bash
conda activate umi
cd "$WS/src/universal_manipulation_interface"

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=train_diffusion_unet_timm_leapumi_workspace \
  task.dataset_path=/absolute/path/to/leapumi_dataset.zarr.zip \
  logging.mode=offline \
  hydra.run.dir="$WS/data/outputs/leapumi_$(date +%Y%m%d_%H%M%S)"
```

When training completes, or whenever the checkpoint interval is reached, the model is saved as `checkpoints/latest.ckpt` in the output directory. Retain this checkpoint: deployment reads its serialized Hydra configuration, so training and deployment must use matching image keys, action dimensions, and network architecture.

## 3. Deployment

The deployment entry point is `universal_manipulation_interface/deployment/real.py`. It subscribes to Insta360 images, reads Kinova and LEAP Hand states, generates actions with the diffusion policy, and sends arm-joint and hand-position commands directly.

### 3.1 Deployment Path Check

`real.py`, `real_system_config.py`, and `simulator.py` now infer the workspace root from their own locations, so no absolute paths need to be edited. They use `$WS/src/kinova-ros/kinova_description/urdf/robot.urdf` and import LEAP Hand utilities from `$WS/src/leap_hand_utils`.

### 3.2 Hardware and ROS Preflight

Deployment requires:

- An available NVIDIA GPU; the default device is `cuda:0`.
- An active X/desktop session. The code calls `p.connect(p.GUI)` and cannot run directly in a headless environment.
- A connected, released Kinova J2N6S300 arm with the correct udev rules.
- A powered LEAP Hand with no other process holding its serial port.
- An Insta360 node publishing `/insta360/image_raw`.
- A Kinova driver providing joint-state feedback and the action server.

Start ROS in terminal A:

```bash
conda activate umi
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
roscore
```

Start the Kinova driver in terminal B:

```bash
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2n6s300
```

After starting the Insta360 ROS driver in terminal C, run these checks:

```bash
rostopic echo -n 1 /j2n6s300_driver/out/joint_state
rostopic type /insta360/image_raw
rostopic hz /insta360/image_raw
rostopic info /j2n6s300_driver/joints_action/joint_angles
```

The last command should show the Kinova action server. If it does not, `real.py` waits at `wait_for_server()`.

### 3.3 Checkpoint and Observation Mode

`real.py` currently splits one wide image into left and right halves and supplies `camera0_rgb_left` and `camera0_rgb_right` to the policy. Therefore, the checkpoint must come from the project-specified `train_diffusion_unet_timm_leapumi_workspace.yaml` configuration (that is, `leapumi_split`), trained with dual images and a 16-DoF LEAP Hand.

Do not use a single-image `leapumi` checkpoint containing only `camera0_rgb` with the current `real.py`; the observation keys do not match. The deployment script supports predicted actions of 10 or 25 dimensions. The 25-dimensional representation is position 3 + rotation-6d 6 + LEAP Hand 16.

### 3.4 Run in Stages

`real.py` has no hardware-free dry-run mode. Even with `--max-steps 0`, initialization connects to Kinova, enables LEAP Hand torque, loads the URDF, and opens the PyBullet GUI; it only avoids executing policy actions.

First start with zero action steps to verify the model, camera, ROS topics, URDF, and GUI:

```bash
conda activate umi
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
cd "$WS/src/universal_manipulation_interface/deployment"

python real.py \
  --checkpoint /absolute/path/to/checkpoints/latest.ckpt \
  --device cuda:0 \
  --max-steps 0
```

After confirming that initialization is correct, run the full deployment:

```bash
python real.py \
  --checkpoint /absolute/path/to/checkpoints/latest.ckpt \
  --device cuda:0 \
  --steps-per-inference 8
```

Press `Ctrl+C` to stop. The action log is overwritten at the path defined by `self.action_log_path` (line 193).
