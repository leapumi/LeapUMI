#!/usr/bin/env python3
"""
PyBullet Simulator deployment of diffusion policy
Reads observations from zarr dataset and executes actions in simulation
Uses the local diffusion_policy module from this project
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import hydra
import dill
import time
import cv2
import zarr
import pybullet as p
import pybullet_data
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep

# Register codecs for zarr
register_codecs()

# Register eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


class SimulatorDiffusionPolicy:
    def __init__(self, checkpoint_path, dataset_path, device='cuda:0', mode='pose', demo_idx=0):
        """
        Initialize simulator diffusion policy
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            dataset_path: Path to zarr dataset (.zarr.zip file)
            device: Device to run the model on
            mode: 'pose' or 'qpos'
            demo_idx: Which demo/episode to use from dataset
        """
        self.device = torch.device(device)
        self.mode = mode
        self.demo_idx = demo_idx
        
        # Load diffusion policy model first to get config
        self.load_policy(checkpoint_path)
        
        # Load dataset (before PyBullet so we can get initial pose)
        self.load_dataset(dataset_path)
        
        # Get initial pose from dataset first frame
        self.get_initial_pose_from_dataset()
        
        # Initialize PyBullet simulation with initial pose
        self.init_pybullet()

        # State variables
        self.past_obs = None
        self.current_obs = None
        self.current_timestep = 0
        
        # Rotation transformers
        self.rot_transformer = RotationTransformer('rotation_6d', 'quaternion')
        
        # Detect image mode from config (single image vs split images)
        self._detect_image_mode()
        
        print("Simulator Diffusion Policy initialized")
    
    def _detect_image_mode(self):
        """Detect image mode from policy's obs_encoder keys.
        
        Uses the actual rgb_keys and low_dim_keys from the loaded policy
        to determine image processing mode and required observation keys.
        """
        obs_shape_meta = self.shape_meta['obs']
        
        # Determine image mode based on policy's rgb_keys
        self.image_mode = 'split'
        self.left_shape = list(obs_shape_meta['camera0_rgb_left']['shape'])
        self.right_shape = list(obs_shape_meta['camera0_rgb_right']['shape'])
        print(f"Image mode: split (left: {self.left_shape}, right: {self.right_shape})")
    
    def split_image_left_right(self, image):
        """Split a wide image into left and right halves
        
        Args:
            image: Input image in HWC format
            
        Returns:
            left_img: Left half of the image
            right_img: Right half of the image
        """
        if image is None:
            return None, None
        
        h, w, c = image.shape
        mid = w // 2
        left_img = image[:, :mid, :]
        right_img = image[:, mid:, :]
        return left_img, right_img

    def get_initial_pose_from_dataset(self):
        """Get initial end-effector pose from the dataset's demo_start_pose"""
        from scipy.spatial.transform import Rotation as R
        
        # Get first frame data
        first_frame_idx = self.episode_start
        
        # Try to get demo start pose first (preferred)
        if 'robot0_demo_start_pose' in self.replay_buffer.keys():
            demo_start_pose = self.replay_buffer['robot0_demo_start_pose'][first_frame_idx].astype(np.float32)
            print(f"Using robot0_demo_start_pose: {demo_start_pose}")
            
            # demo_start_pose format: [x, y, z, ax, ay, az] (position + axis-angle)
            self.initial_eef_pos = demo_start_pose[:3]
            self.initial_eef_rot_axis_angle = demo_start_pose[3:6]
        else:
            # Fallback to individual position and rotation keys
            print("robot0_demo_start_pose not found, using robot0_eef_pos and robot0_eef_rot_axis_angle")
            if 'robot0_eef_pos' in self.replay_buffer.keys():
                self.initial_eef_pos = self.replay_buffer['robot0_eef_pos'][first_frame_idx].astype(np.float32)
            else:
                self.initial_eef_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
            
            if 'robot0_eef_rot_axis_angle' in self.replay_buffer.keys():
                self.initial_eef_rot_axis_angle = self.replay_buffer['robot0_eef_rot_axis_angle'][first_frame_idx].astype(np.float32)
            else:
                self.initial_eef_rot_axis_angle = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Convert to quaternion for PyBullet [x, y, z, w]
        rot = R.from_rotvec(self.initial_eef_rot_axis_angle)
        self.initial_eef_quat = rot.as_quat()  # scipy returns [x, y, z, w]
        
        # Get gripper/hand positions if available
        if 'robot0_gripper_width' in self.replay_buffer.keys():
            self.initial_gripper_qpos = self.replay_buffer['robot0_gripper_width'][first_frame_idx].astype(np.float32)
        else:
            self.initial_gripper_qpos = None
        
        # Store as episode start pose for relative pose computation
        self.episode_start_pose = np.concatenate([self.initial_eef_pos, self.initial_eef_rot_axis_angle])
        self.start_pose_mat = pose_to_mat(self.episode_start_pose)

        print(f"Initial pose from dataset:")
        print(f"  Position: {self.initial_eef_pos}")
        print(f"  Rotation (axis-angle): {self.initial_eef_rot_axis_angle}")
        print(f"  Quaternion [x,y,z,w]: {self.initial_eef_quat}")
        if self.initial_gripper_qpos is not None:
            print(f"  Gripper qpos: {self.initial_gripper_qpos[:4]}... (shape: {self.initial_gripper_qpos.shape})")
    
    def load_policy(self, checkpoint_path):
        """Load the trained diffusion policy model using this project's interface"""
        print(f"Loading policy from {checkpoint_path}")
        
        # Load checkpoint
        if not checkpoint_path.endswith('.ckpt'):
            checkpoint_path = os.path.join(checkpoint_path, 'checkpoints', 'latest.ckpt')
        
        payload = torch.load(open(checkpoint_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        
        # Create workspace and load model
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get policy model
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model
        
        # Set inference steps for DDIM
        self.policy.num_inference_steps = 16
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Get pose representation config
        self.obs_pose_repr = self.cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print(f"Obs pose repr: {self.obs_pose_repr}")
        print(f"Action pose repr: {self.action_pose_repr}")
        
        # Get shape meta for observation processing
        self.shape_meta = self.cfg.task.shape_meta
        
        print(f"Policy loaded successfully")
    
    def load_dataset(self, dataset_path):
        """Load zarr.zip dataset"""
        print(f"Loading dataset from {dataset_path}")
        
        # Open zarr.zip file
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            self.replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store,
                store=zarr.MemoryStore()
            )
        
        # Get episode info
        n_episodes = self.replay_buffer.n_episodes
        print(f"Dataset has {n_episodes} episodes")
        
        if self.demo_idx >= n_episodes:
            raise ValueError(f"Demo index {self.demo_idx} out of range (max: {n_episodes-1})")
        
        # Get episode slice using ReplayBuffer's episode_ends property
        episode_ends = self.replay_buffer.episode_ends[:]
        if self.demo_idx == 0:
            episode_start = 0
        else:
            episode_start = episode_ends[self.demo_idx - 1]
        episode_end = episode_ends[self.demo_idx]
        
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.num_timesteps = episode_end - episode_start
        
        print(f"Episode {self.demo_idx}: timesteps {episode_start} to {episode_end} ({self.num_timesteps} frames)")
        
        # Print available keys
        data_keys = list(self.replay_buffer.keys())
        print(f"Available data keys: {data_keys}")
    
    def init_pybullet(self):
        """Initialize PyBullet simulation with initial pose from dataset"""
        # Connect to PyBullet in GUI mode for visualization
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up simulation
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        # p.loadURDF("plane.urdf")
        
        # Load robot - use a default robot path or make it configurable
        self.kinova_base_pos = [0.0, 0.0, 0.0]
        
        robot_urdf = os.path.join(
            SRC_DIR, "kinova-ros", "kinova_description", "urdf", "robot.urdf"
        )

        self.kinova_id = p.loadURDF(
            robot_urdf,
            self.kinova_base_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # Get number of joints
        self.num_joints = p.getNumJoints(self.kinova_id)
        print(f"Robot has {self.num_joints} joints")
        
        # Print joint info
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.kinova_id, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}, type={joint_info[2]}")
        
        # Find end-effector link index (assembly_to_hand joint)
        self.ee_link_index = 9
        
        # Set robot to initial pose from dataset using IK
        # self.set_robot_to_initial_pose()
        self.set_default_pose()
        # Configure camera to look at the robot
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=self.initial_eef_pos
        )
        
        print(f"PyBullet simulation initialized with initial pose from dataset")
    
    def set_robot_to_initial_pose(self):
        """Set robot joints to achieve the initial end-effector pose from dataset"""
        print("Setting robot to initial pose from dataset...")
        
        # Calculate IK for initial pose
        joint_positions = p.calculateInverseKinematics(
            self.kinova_id,
            self.ee_link_index,
            self.initial_eef_pos,
            self.initial_eef_quat,
            maxNumIterations=500,
            residualThreshold=0.00001
        )
        
        # Kinova arm joints are at indices 2-7 (6 DOF)
        arm_joint_indices = [2, 3, 4, 5, 6, 7]
        
        # Set arm joint positions
        for i, joint_idx in enumerate(arm_joint_indices):
            p.resetJointState(self.kinova_id, joint_idx, joint_positions[i])
        
        # Set LeapHand joints if available (joints 10-28, excluding fixed joints)
        if self.initial_gripper_qpos is not None:
            # LeapHand joint mapping (from dataset order to URDF order)
            # Dataset: 16 joints
            # URDF: joints at indices 10,11,12,13, 15,16,17,18, 20,21,22,23, 25,26,27,28
            leaphand_joint_indices = [11, 10, 12, 13, 16, 15, 17, 18, 21, 20, 22, 23, 25, 26, 27, 28]
            
            for i, joint_idx in enumerate(leaphand_joint_indices):
                if i < len(self.initial_gripper_qpos):
                    # Apply -π offset for LeapHand
                    joint_pos = self.initial_gripper_qpos[i] - np.pi
                    p.resetJointState(self.kinova_id, joint_idx, joint_pos)
        
        # Step simulation to apply changes
        for _ in range(10):
            p.stepSimulation()
        
        # Verify the pose
        link_state = p.getLinkState(self.kinova_id, self.ee_link_index)
        actual_pos = np.array(link_state[4])
        actual_quat = np.array(link_state[5])
        
        pos_error = np.linalg.norm(actual_pos - self.initial_eef_pos)
        print(f"Initial pose set:")
        print(f"  Target pos: {self.initial_eef_pos}")
        print(f"  Actual pos: {actual_pos}")
        print(f"  Target quat: {self.initial_eef_quat}")
        print(f"  Actual quat: {actual_quat}")
        print(f"  Position error: {pos_error:.6f} m")

        time.sleep(3)

    def set_default_pose(self):
        from scipy.spatial.transform import Rotation as R

        """Set the robot to a default pose with random perturbation"""
        default_qpos = [1.590, 2.836, 4.898, 4.683, -0.108, 9.549]
        np.random.seed(int(time.time() * 1000000) % (2**32))
        # First set PyBullet to default pose to get the default EE position
        for i in range(2, 8):
            p.resetJointState(self.kinova_id, i, default_qpos[i - 2])
        
        # Get the default end-effector pose
        link_state = p.getLinkState(self.kinova_id, self.ee_link_index)
        default_ee_pos = np.array(link_state[4])
        default_ee_quat_xyzw = np.array(link_state[5])  # PyBullet: (x, y, z, w)
        
        # Add random perturbation to position (0 to 5 cm in each direction)
        pos_perturbation = np.random.uniform(-0.05, 0.05, size=3)
        target_pos = default_ee_pos + pos_perturbation
        
        # Add random perturbation to orientation (0 to 15 degrees in each axis)
        angle_perturbation = np.random.uniform(-np.deg2rad(15), np.deg2rad(15), size=3)
        
        # Convert to rotation matrix and apply perturbation
        default_rot_mat = R.from_quat(default_ee_quat_xyzw).as_matrix()
        
        # Build euler rotation matrices
        cx, cy, cz = np.cos(angle_perturbation)
        sx, sy, sz = np.sin(angle_perturbation)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        
        target_rot_mat = default_rot_mat @ (Rz @ Ry @ Rx)
        
        # Convert back to quaternion (wxyz -> xyzw for PyBullet)
        target_quat = R.from_matrix(target_rot_mat).as_quat()
        
        print(f"Position perturbation: {pos_perturbation * 100} cm")
        print(f"Angle perturbation: {np.rad2deg(angle_perturbation)} degrees")
        
        # Calculate IK and send to robot
        self.move_to_ee_pose(target_pos, target_quat, default_qpos)
    
    def move_to_ee_pose(self, target_pos, target_quat, fallback_qpos=None):
        """
        Move robot to target end-effector pose using IK
        
        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [x, y, z, w] (PyBullet format)
            fallback_qpos: Fallback joint positions if IK fails
        """
        # Reset PyBullet to current real robot position
        # current_qpos = list(self.kinova_node.current_qpos.values())
        # for i in range(2, 8):
        #     p.resetJointState(self.kinova_id, i, current_qpos[i - 2])
        
        # Calculate IK
        joint_poses = p.calculateInverseKinematics(
            self.kinova_id,
            self.ee_link_index,
            target_pos,
            target_quat,
            solver=p.IK_DLS,
            maxNumIterations=500,
            residualThreshold=0.0001,
        )

        for i in range(2, 8):
            p.resetJointState(self.kinova_id, i, fallback_qpos[i - 2])
        print("set initial pose")

    
    def preprocess_image(self, image, target_shape):
        """Preprocess camera image for the model
        
        Args:
            image: Input image in HWC format
            target_shape: Target shape [C, H, W]
        """
        if image is None:
            return None
        
        c, h, w = target_shape
        
        # Resize image if needed
        if image.shape[0] != h or image.shape[1] != w:
            resized = cv2.resize(image, (w, h))
        else:
            resized = image
        
        # Normalize to [0,1] range and convert to CHW
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.moveaxis(normalized, -1, 0)  # HWC to CHW
        
        return normalized
    
    def get_observation_from_dataset(self, timestep):
        """Get observation from zarr dataset at given timestep"""
        # Convert local timestep to global index
        global_idx = self.episode_start + timestep
        
        if timestep >= self.num_timesteps:
            return None
        
        obs = {}
        
        # Handle different image modes
        if self.image_mode == 'split':
            # Split image mode: check if dataset has separate left/right or needs splitting
            if 'camera0_rgb_left' in self.replay_buffer.keys() and 'camera0_rgb_right' in self.replay_buffer.keys():
                # Dataset already has separate left/right images
                img_left = self.replay_buffer['camera0_rgb_left'][global_idx]
                img_right = self.replay_buffer['camera0_rgb_right'][global_idx]
            elif 'camera0_rgb' in self.replay_buffer.keys():
                # Need to split the single image
                img = self.replay_buffer['camera0_rgb'][global_idx]
                img_left, img_right = self.split_image_left_right(img)
            else:
                raise ValueError("No camera image found in dataset")
            
            obs['camera0_rgb_left'] = self.preprocess_image(img_left, self.left_shape)
            obs['camera0_rgb_right'] = self.preprocess_image(img_right, self.right_shape)
        else:
            # Single image mode: camera0_rgb
            if 'camera0_rgb' in self.replay_buffer.keys():
                img = self.replay_buffer['camera0_rgb'][global_idx]
                obs['camera0_rgb'] = self.preprocess_image(img, self.single_shape)
            else:
                raise ValueError("No camera image found in dataset")
        
        # End-effector position
        if 'robot0_eef_pos' in self.replay_buffer.keys():
            obs['robot0_eef_pos'] = self.replay_buffer['robot0_eef_pos'][global_idx].astype(np.float32)
        
        # End-effector rotation (axis-angle)
        if 'robot0_eef_rot_axis_angle' in self.replay_buffer.keys():
            obs['robot0_eef_rot_axis_angle'] = self.replay_buffer['robot0_eef_rot_axis_angle'][global_idx].astype(np.float32)
        
        # Gripper width
        if 'robot0_gripper_width' in self.replay_buffer.keys():
            obs['robot0_gripper_width'] = self.replay_buffer['robot0_gripper_width'][global_idx].astype(np.float32)
        
        return obs
    
    def prepare_obs_dict(self, obs_history):
        """
        Prepare observation dictionary for policy inference
        Following the format used in get_real_umi_obs_dict
        
        Args:
            obs_history: list of observations [past_obs, current_obs]
        """
        from scipy.spatial.transform import Rotation as R
        
        obs_dict = {}
        
        # Stack observations along time dimension
        T = len(obs_history)
        
        # Handle different image modes
        if self.image_mode == 'split':
            # Split image mode: camera0_rgb_left and camera0_rgb_right
            if 'camera0_rgb_left' in obs_history[0]:
                imgs_left = np.stack([obs['camera0_rgb_left'] for obs in obs_history], axis=0)
                obs_dict['camera0_rgb_left'] = imgs_left
            if 'camera0_rgb_right' in obs_history[0]:
                imgs_right = np.stack([obs['camera0_rgb_right'] for obs in obs_history], axis=0)
                obs_dict['camera0_rgb_right'] = imgs_right
        else:
            # Single image mode: camera0_rgb
            if 'camera0_rgb' in obs_history[0]:
                imgs = np.stack([obs['camera0_rgb'] for obs in obs_history], axis=0)
                obs_dict['camera0_rgb'] = imgs
        
        # Process pose observations
        if 'robot0_eef_pos' in obs_history[0] and 'robot0_eef_rot_axis_angle' in obs_history[0]:
            # Stack raw poses
            poses = []
            for obs in obs_history:
                pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']])
                poses.append(pose)
            poses = np.stack(poses, axis=0)  # [T, 6]
            
            # Convert to pose matrix
            pose_mat = pose_to_mat(poses)
            
            # Convert to observation pose representation
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False
            )
            
            # Convert to pose10d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            
            obs_dict['robot0_eef_pos'] = obs_pose[..., :3].astype(np.float32)
            obs_dict['robot0_eef_rot_axis_angle'] = obs_pose[..., 3:].astype(np.float32)
        
        # Gripper width: [T, gripper_dim]
        if 'robot0_gripper_width' in obs_history[0]:
            gripper = np.stack([obs['robot0_gripper_width'] for obs in obs_history], axis=0)
            obs_dict['robot0_gripper_width'] = gripper.astype(np.float32)
        
        return obs_dict
    
    def predict_action(self):
        """Predict action using diffusion policy"""
        # Get current and past observation
        current_obs = self.get_observation_from_dataset(self.current_timestep)
        if current_obs is None:
            return None
        
        # Build observation history (need obs_horizon frames)
        # Get obs_horizon from the first available camera key
        if self.image_mode == 'split':
            obs_horizon = self.shape_meta['obs']['camera0_rgb_left'].get('horizon', 2)
        else:
            obs_horizon = self.shape_meta['obs']['camera0_rgb'].get('horizon', 2)
        
        obs_history = []
        
        for i in range(obs_horizon):
            t = max(0, self.current_timestep - (obs_horizon - 1 - i))
            obs = self.get_observation_from_dataset(t)
            if obs is not None:
                obs_history.append(obs)
            else:
                obs_history.append(obs_history[-1] if obs_history else current_obs)
        
        # Prepare observation dict
        obs_dict_np = self.prepare_obs_dict(obs_history)
        
        # Convert to tensor and add batch dimension
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        # Run inference
        with torch.no_grad():
            start_time = time.perf_counter()
            result = self.policy.predict_action(obs_dict)
            inference_time = time.perf_counter() - start_time
            print(f"Inference time: {inference_time:.4f}s")
        
        # Get action prediction
        action_chunk = result['action_pred'][0].detach().cpu().numpy()
        
        # Store current obs as past obs for next iteration
        self.past_obs = current_obs
        
        return action_chunk
    
    def convert_action_chunk_to_pose(self, action_chunk, current_obs):
        """
        Convert entire action chunk (pose10d format) to world poses
        Following get_real_umi_action logic - convert all actions based on current observation
        
        Args:
            action_chunk: [T, action_dim] array of actions
            current_obs: current observation dict
            
        Returns:
            action_poses: [T, 6] array of poses (pos + axis_angle)
            action_grippers: [T, gripper_dim] array of gripper positions
        """
        from scipy.spatial.transform import Rotation as R

        # Get current pose matrix (base for all action conversions)
        link_state = p.getLinkState(self.kinova_id, self.ee_link_index)
        actual_pos = np.array(link_state[4])
        actual_quat = np.array(link_state[5])
        actual_ax = R.from_quat(actual_quat).as_rotvec()
        
        current_pose = np.concatenate([
            actual_pos,
            actual_ax
        ])
        current_pose_mat = pose_to_mat(current_pose)
        
        # Handle different action dimensions
        action_dim = action_chunk.shape[-1]
        if action_dim == 10:
            # Standard UMI format: 9 pose + 1 gripper
            action_pose10d = action_chunk[..., :9]
            action_gripper = action_chunk[..., 9:10]
        elif action_dim == 25:
            # LeapHand format: 9 pose + 16 gripper
            action_pose10d = action_chunk[..., :9]
            action_gripper = action_chunk[..., 9:25]
        else:
            raise ValueError(f"Unsupported action dimension: {action_dim}")

        # # replace action chunk by dataset
        # for i in range(0, 8):
        #     timestep = self.current_timestep + i + 1
        #     first_obs = self.get_observation_from_dataset(self.current_timestep)
        #     action_obs = self.get_observation_from_dataset(timestep)

        #     first_pos = first_obs['robot0_eef_pos']
        #     first_rot = first_obs['robot0_eef_rot_axis_angle']
        #     first_pose = np.concatenate([first_pos, first_rot])

        #     action_pos = action_obs['robot0_eef_pos']
        #     action_rot = action_obs['robot0_eef_rot_axis_angle']
        #     action_pose = np.concatenate([action_pos, action_rot])

        #     # Calculate the matrix between first_pose and action_pose, then apply it to current_pose
        #     relative_mat = np.linalg.inv(pose_to_mat(first_pose)) @ pose_to_mat(action_pose)
        #     action_pose10d[i] = mat_to_pose10d(relative_mat)


        # Convert pose10d to matrix for all actions
        action_pose_mat = pose10d_to_mat(action_pose10d)
        
        # Convert from relative to absolute for all actions using the same base pose
        action_mat = convert_pose_mat_rep(
            action_pose_mat,
            base_pose_mat=current_pose_mat,
            pose_rep=self.action_pose_repr,
            backward=True
        )
        
        # Convert to pose [pos, axis_angle] for all actions
        action_poses = mat_to_pose(action_mat)
        
        return action_poses, action_gripper
    
    def update_robot_in_sim(self, target_pos, target_rot_axis_angle, gripper_qpos):
        """Update robot pose in PyBullet simulation using IK"""
        from scipy.spatial.transform import Rotation as R
        
        # Convert axis-angle to quaternion for PyBullet
        rot = R.from_rotvec(target_rot_axis_angle)
        quat_scipy = rot.as_quat()  # [x, y, z, w]
        target_quat = quat_scipy  # PyBullet uses [x, y, z, w]
        
        # Get current joint states as restPoses (IK seed) for ALL movable joints
        # PyBullet requires restPoses length == number of movable joints, otherwise it is silently ignored
        # rest_poses = []
        # for i in range(p.getNumJoints(self.kinova_id)):
        #     joint_info = p.getJointInfo(self.kinova_id, i)
        #     rest_poses.append(p.getJointState(self.kinova_id, i)[0])

        # Calculate IK with restPoses to guide the solver from current configuration
        joint_positions = p.calculateInverseKinematics(
            self.kinova_id,
            self.ee_link_index,
            target_pos,
            target_quat,
            # restPoses=rest_poses,
            maxNumIterations=500,
            residualThreshold=0.00001
        )

        # Kinova arm joints are at indices 2-7 (6 DOF)
        arm_joint_indices = [2, 3, 4, 5, 6, 7]
        for i, joint_idx in enumerate(arm_joint_indices):
            p.resetJointState(self.kinova_id, joint_idx, joint_positions[i])

        # Update LeapHand joints if available
        # Mapping: dataset index -> URDF joint index
        # Format: (urdf_joint_idx, dataset_idx)
        leaphand_mapping = [
            (10, 1), (11, 0), (12, 2), (13, 3),    # Finger 1
            (15, 5), (16, 4), (17, 6), (18, 7),    # Finger 2
            (20, 9), (21, 8), (22, 10), (23, 11),  # Finger 3
            (25, 12), (26, 13), (27, 14), (28, 15) # Thumb
        ]
        
        for urdf_idx, data_idx in leaphand_mapping:
            joint_pos = gripper_qpos[data_idx] - np.pi
            p.resetJointState(self.kinova_id, urdf_idx, joint_pos)
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            time.sleep(1./240.)
        # Verify the pose
        link_state = p.getLinkState(self.kinova_id, self.ee_link_index)
        actual_pos = np.array(link_state[4])
        actual_quat = np.array(link_state[5])
        print("current actual pose:", actual_pos, actual_quat)
    
    def execute_action_chunk(self, action_chunk, current_obs, steps_per_inference=8):
        """
        Execute a chunk of actions
        
        Args:
            action_chunk: [T, action_dim] array of actions from policy
            current_obs: current observation at the time of inference
            steps_per_inference: number of steps to execute
        """
        from scipy.spatial.transform import Rotation as R

        num_actions = min(len(action_chunk), steps_per_inference)
        
        # Convert all actions to world poses based on current observation
        # This is done ONCE before execution, not per-action
        target_poses, gripper_qpos_all = self.convert_action_chunk_to_pose(action_chunk, current_obs)
        
        print(f"Converted {num_actions} actions to world poses based on current observation")
        
        for i in range(num_actions):
            target_pos = target_poses[i, :3]
            target_rot = target_poses[i, 3:6]
            gripper_qpos = gripper_qpos_all[i]

            # Execute in simulation
            self.update_robot_in_sim(target_pos, target_rot, gripper_qpos)
            
            print(f"  Action {i}: pos={target_pos}, rot={target_rot}, hand={gripper_qpos}...")
            
            # Update timestep
            self.current_timestep += 1
            
            if self.current_timestep >= self.num_timesteps:
                break
            time.sleep(0.5)
        return True
    
    def run(self, max_steps=None, steps_per_inference=8):
        """Main execution loop"""
        print("Starting simulator diffusion policy execution...")
        
        if max_steps is None:
            max_steps = self.num_timesteps
        
        inference_count = 0
        
        # Reset policy
        self.policy.reset()
        
        try:
            while self.current_timestep < self.num_timesteps and self.current_timestep < max_steps:
                print(f"\n=== Inference {inference_count}, Timestep {self.current_timestep}/{self.num_timesteps} ===")
                
                # Predict action chunk
                action_chunk = self.predict_action()
                
                if action_chunk is not None:
                    print(f"Action chunk shape: {action_chunk.shape}")
                    # Execute action chunk
                    current_obs = self.get_observation_from_dataset(self.current_timestep)
                    self.execute_action_chunk(action_chunk, current_obs, steps_per_inference)
                    inference_count += 1
                else:
                    print("No action predicted")
                    break
                
                # Small delay for visualization
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        finally:
            self.cleanup()
        
        print(f"Simulator execution completed. Total inferences: {inference_count}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Disconnect PyBullet
            p.disconnect()
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy diffusion policy in PyBullet simulator')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to diffusion policy checkpoint (.ckpt file or directory)')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Path to zarr dataset (.zarr.zip file)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--mode', type=str, default='pose',
                        choices=['pose', 'qpos'],
                        help='Control mode: pose or qpos')
    parser.add_argument('--demo', type=int, default=0,
                        help='Demo/episode index to use from dataset')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum number of timesteps to run')
    parser.add_argument('--steps-per-inference', type=int, default=8,
                        help='Number of action steps to execute per inference')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the simulator
        system = SimulatorDiffusionPolicy(
            args.checkpoint, 
            args.dataset,
            args.device,
            args.mode,
            args.demo
        )
        system.run(args.max_steps, args.steps_per_inference)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()