#!/usr/bin/env python3
"""
Real robot deployment of diffusion policy
Uses ROS for camera and robot control
Robot: Kinova arm (6DOF) + LeapHand (16DOF)
"""

import os
import sys


# Resolve all project paths from this file. This keeps deployment portable when
# the workspace directory is renamed or moved.
DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEPLOYMENT_DIR)
SRC_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, DEPLOYMENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from kinova_node import KinovaNode
import real_system_config as config

import numpy as np
import torch
import hydra
import dill
import time
import cv2
import pybullet as p
import pybullet_data
from omegaconf import OmegaConf
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep


from scipy.spatial.transform import Rotation as R

# Add the workspace src directory so the shared LeapHand utilities are importable.
# PROJECT_ROOT is <workspace>/src/universal_manipulation_interface.
LEAPHAND_UTILS_PATH = SRC_DIR
sys.path.insert(0, LEAPHAND_UTILS_PATH)
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu

# Register codecs for zarr
register_codecs()

# Register eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


class LeapNode:
    """LeapHand control node - copied from leap_kinova_ik_real_recording.py"""
    def __init__(self):
        # Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(
            np.zeros(16))

        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, "/dev/ttyUSB0", 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(
                    motors, "/dev/ttyUSB1", 4000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(
                        motors, "/dev/ttyUSB2", 4000000)
                    self.dxl_client.connect()
                except Exception:
                    self.dxl_client = DynamixelClient(motors, "COM13", 4000000)
                    self.dxl_client.connect()
        # Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kP, 84, 2
        )  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2
        )  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kI, 82, 2
        )  # Igain
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kD, 80, 2
        )  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2
        )  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(
            len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print("Leap Node initialized")

    # Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()  # 16dof

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


class RealDiffusionPolicy:
    def __init__(self, checkpoint_path, device='cuda:0'):
        """
        Initialize real robot diffusion policy
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            device: Device to run the model on
        """
        self.device = torch.device(device)
        
        # Initialize ROS node
        rospy.init_node('diffusion_policy_real', anonymous=True)
        
        # Initialize ROS Image handling for Insta360 Air camera
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/insta360/image_raw", Image, self.image_callback)
        self.current_camera_image = None
        
        # Load diffusion policy model
        self.load_policy(checkpoint_path)
        
        # Initialize Kinova node
        self.kinova_node = KinovaNode()
        
        # Initialize LeapHand node
        self.leap_node = LeapNode()
        
        # Initialize PyBullet for IK
        self.init_pybullet()

        # State variables
        self.current_joint_states = None
        self.current_leaphand_states = None
        self.past_obs = None
        self.current_obs = None
        self.current_timestep = 0
        
        # Rotation transformers
        self.rot_transformer = RotationTransformer('rotation_6d', 'quaternion')
        self.mat_to_quat = RotationTransformer('matrix', 'quaternion')
        self.quat_to_mat = RotationTransformer('quaternion', 'matrix')
        self.rot6d_to_axis_angle = RotationTransformer('rotation_6d', 'axis_angle')
        
        # Action log file
        self.action_log_path = "/home/donglzh/deployment_result/toy/toy_resnet_trial_2.txt"
        os.makedirs(os.path.dirname(self.action_log_path), exist_ok=True)
        # Clear the file at the start of each run
        with open(self.action_log_path, 'w') as f:
            pass
        print(f"Action log file: {self.action_log_path}")
        
        # Episode start pose (set from first observation)
        self.episode_start_pose = None
        
        # Wait for initial data
        rospy.sleep(2.0)
        
        # Detect image mode from config (single image vs split images)
        self._detect_image_mode()
        
        # self.set_default_pose()
        # self.set_fixed_qpos()
        # Get initial pose from real robot
        self.get_initial_pose_from_robot()
        
        print("Real Diffusion Policy initialized")
    
    def set_fixed_qpos(self):
        """Set the robot to a fixed default pose (no random perturbation)"""
        default_qpos = [1.77808147,2.79352735,4.61347811,4.65257918,5.68799544,3.53582385]
        # First set PyBullet to default pose to get the default EE position
        for i in range(2, 8):
            p.resetJointState(self.kinova_id, i, default_qpos[i - 2])
        
        # Get the default end-effector pose
        link_state = p.getLinkState(self.kinova_id, self.ee_link_index)
        target_pos = np.array(link_state[4])
        target_quat = np.array(link_state[5])  # PyBullet: (x, y, z, w)
        
        # Calculate IK and send to robot
        self.move_to_ee_pose(target_pos, target_quat, default_qpos)

    def _detect_image_mode(self):
        """Detect image mode from task config name
        
        - leap_umi_abs: split mode, need to split camera0_rgb into left/right 224x224 images
        - leapumi: single mode, use camera0_rgb as 160x320 image
        """
        # Get task name from config
        task_name = self.cfg.task.name
        obs_shape_meta = self.shape_meta['obs']
        
        print(f"Task name: {task_name}")
        
        # rgb_keys = self.policy.obs_encoder.rgb_keys
        # rgb_keys = [key for key in rgb_keys] 
        # print("rgb_keys list: ", rgb_keys)
        # # Determine image mode based on task name
        # if len(rgb_keys) == 2:
        #     # Split mode: need to split single camera image into left and right
        #     self.image_mode = 'split'
        #     # Target shapes for left and right images (from config)
        #     if 'camera0_rgb_left' in obs_shape_meta:
        #         self.left_shape = list(obs_shape_meta['camera0_rgb_left']['shape'])
        #     else:
        #         self.left_shape = [3, 224, 224]  # Default
        #     if 'camera0_rgb_right' in obs_shape_meta:
        #         self.right_shape = list(obs_shape_meta['camera0_rgb_right']['shape'])
        #     else:
        #         self.right_shape = [3, 224, 224]  # Default
        #     print(f"Image mode: split (left: {self.left_shape}, right: {self.right_shape})")
        # else:
        #     # Single mode: use camera0_rgb directly
        #     self.image_mode = 'single'
        #     if 'camera0_rgb' in obs_shape_meta:
        #         self.single_shape = list(obs_shape_meta['camera0_rgb']['shape'])
        #     else:
        #         self.single_shape = [3, 160, 320]  # Default
        #     print(f"Image mode: single ({self.single_shape})")

        # Single mode: use camera0_rgb directly
        # self.image_mode = 'single'
        # if 'camera0_rgb' in obs_shape_meta:
        #     self.single_shape = list(obs_shape_meta['camera0_rgb']['shape'])
        # else:
        #     self.single_shape = [3, 160, 320]  # Default
        # print(f"Image mode: single ({self.single_shape})")
        
        # Split mode: need to split single camera image into left and right
        self.image_mode = 'split'
        # Target shapes for left and right images (from config)
        if 'camera0_rgb_left' in obs_shape_meta:
            self.left_shape = list(obs_shape_meta['camera0_rgb_left']['shape'])
        else:
            self.left_shape = [3, 224, 224]  # Default
        if 'camera0_rgb_right' in obs_shape_meta:
            self.right_shape = list(obs_shape_meta['camera0_rgb_right']['shape'])
        else:
            self.right_shape = [3, 224, 224]  # Default
        print(f"Image mode: split (left: {self.left_shape}, right: {self.right_shape})")

    def image_callback(self, msg):
        """Callback for ROS Image messages from Insta360 Air camera"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.current_camera_image = cv_image.copy()
        except Exception as e:
            rospy.logerr(f"Failed to convert ROS Image: {e}")
    
    def get_camera_image(self):
        """Get the latest image from the camera"""
        if self.current_camera_image is None:
            return None
        return self.current_camera_image.copy()
    
    def preprocess_image(self, image, target_shape):
        """Preprocess camera image for the model
        
        Args:
            image: Input image in HWC format
            target_shape: Target shape [C, H, W]
        """
        if image is None:
            return None
        
        c, h, w = target_shape
        
        if image.shape[0] != h or image.shape[1] != w:
            resized = cv2.resize(image, (w, h))
        else:
            resized = image
        
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.moveaxis(normalized, -1, 0)
        return normalized
    
    def split_image_left_right(self, image):
        """Split a wide image into left and right halves
        
        Args:
            image: Input image in HWC format (e.g., 224x448x3)
            
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
    
    def get_initial_pose_from_robot(self):
        """Get initial end-effector pose from real robot"""
        from scipy.spatial.transform import Rotation as R
        
        # Update joint states
        current_joints = self.kinova_node.get_current_joint_angles()
        if current_joints is not None:
            self.current_joint_states = current_joints
            self.update_pybullet_arm_qpos(current_joints)
        
        # Get EE pose from PyBullet (synced with real robot)
        ee_pos, ee_rot = self.get_ee_pose_from_pybullet()

        self.initial_eef_pos = ee_pos
        self.initial_eef_rot_axis_angle = ee_rot
        self.episode_start_pose = np.concatenate([ee_pos, ee_rot])
        self.start_pose_mat = pose_to_mat(self.episode_start_pose)
        # Get gripper state
        try:
            self.current_leaphand_states = self.leap_node.read_pos()
            self.initial_gripper_qpos = self.current_leaphand_states
            self.update_pybullet_leaphand_qpos(self.initial_gripper_qpos)
        except:
            self.initial_gripper_qpos = None
        
        print(f"Initial qpos: {current_joints}")
        print(f"Initial pose from robot:")
        print(f"  Position: {self.initial_eef_pos}")
        print(f"  Rotation (axis-angle): {self.initial_eef_rot_axis_angle}")
    
    def load_policy(self, checkpoint_path):
        """Load the trained diffusion policy model"""
        print(f"Loading policy from {checkpoint_path}")
        
        if not checkpoint_path.endswith('.ckpt'):
            checkpoint_path = os.path.join(checkpoint_path, 'checkpoints', 'latest.ckpt')
        
        payload = torch.load(open(checkpoint_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        
        # Avoid timm online download during construction; checkpoint already has model weights.
        # Keep BN/GN topology consistent with how the checkpoint was trained.
        train_pretrained = bool(self.cfg.policy.obs_encoder.pretrained)
        self.cfg.policy.obs_encoder.pretrained = False
        if train_pretrained and bool(getattr(self.cfg.policy.obs_encoder, 'use_group_norm', False)):
            # TimmObsEncoder only replaces BN->GN when pretrained is False.
            # If training used pretrained=True, keep BatchNorm layout at deployment.
            self.cfg.policy.obs_encoder.use_group_norm = False
            print("Use batch norm")
        
        # print(f"Model name: {self.cfg.policy.obs_encoder.model_name}")
        
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        try:
            # Prefer strict loading to catch real architecture mismatches.
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        except RuntimeError as e:
            err_msg = str(e)
            bn_buffer_tokens = ('running_mean', 'running_var', 'num_batches_tracked')
            if 'Unexpected key(s) in state_dict' in err_msg and any(tok in err_msg for tok in bn_buffer_tokens):
                print('Strict checkpoint loading failed due to BatchNorm buffer mismatch. '
                      'Retrying with strict=False for deployment compatibility.')
                workspace.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)
            else:
                raise
        
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model
        
        self.policy.num_inference_steps = 16
        self.policy.to(self.device)
        self.policy.eval()
        
        self.obs_pose_repr = self.cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print(f"Obs pose repr: {self.obs_pose_repr}")
        print(f"Action pose repr: {self.action_pose_repr}")
        
        self.shape_meta = self.cfg.task.shape_meta
        print(f"Policy loaded successfully")
    
    def init_pybullet(self):
        """Initialize PyBullet simulation for IK"""
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        self.kinova_base_pos = [0.0, 0.0, 0.0]
        KINOVA_URDF_PATH = config.KINOVA_URDF_PATH
        
        self.kinova_id = p.loadURDF(
            KINOVA_URDF_PATH,
            self.kinova_base_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        
        self.num_joints = p.getNumJoints(self.kinova_id)
        self.ee_link_index = 9
        
        # Initialize PyBullet with real robot joint positions
        current_joints = self.kinova_node.get_current_joint_angles()
        try:
            # Get current joint positions from real robot
            current_joint_angles = self.kinova_node.get_current_joint_angles()
            
            if current_joint_angles is not None and len(current_joint_angles) == 6:
                print("Initializing PyBullet with real robot joint positions:")
                for i, joint_angle in enumerate(current_joint_angles):
                    p.resetJointState(self.kinova_id, i + 2, joint_angle)  # joints 2-7 for 6DOF arm
                    print(f"  Joint {i+1}: {joint_angle:.4f} rad")
            else:
                print("Could not get real robot joint positions, using default configuration")
                # Initialize joints to a reasonable default configuration
                for i in range(2, 8):
                    p.resetJointState(self.kinova_id, i, 0.0)
        except Exception as e:
            print(f"Error getting real robot joint positions: {e}")
            print("Using default joint configuration")
            # Initialize joints to a reasonable default configuration
            for i in range(2, 8):
                p.resetJointState(self.kinova_id, i, 0.0)
        
        print("PyBullet initialized for IK")
    

    def set_default_pose(self):
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
        current_qpos = list(self.kinova_node.current_qpos.values())
        for i in range(2, 8):
            p.resetJointState(self.kinova_id, i, current_qpos[i - 2])
        
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
        
        if joint_poses is not None and not any(math.isnan(pose) for pose in joint_poses[:6]):
            self.execute_arm_joint_command(joint_poses[:6])
            for i in range(2, 8):
                p.resetJointState(self.kinova_id, i, joint_poses[i - 2])
            print("Moved to target pose successfully")
        elif fallback_qpos is not None:
            print("IK failed, using fallback pose")
            self.execute_arm_joint_command(fallback_qpos)
            for i in range(2, 8):
                p.resetJointState(self.kinova_id, i, fallback_qpos[i - 2])
        else:
            print("IK failed and no fallback provided")


    def update_pybullet_arm_qpos(self, joint_positions):
        """Update PyBullet with real robot arm joint positions"""
        if joint_positions is None or len(joint_positions) != 6:
            return False
        for i, joint_pos in enumerate(joint_positions):
            p.resetJointState(self.kinova_id, i + 2, joint_pos)
        p.stepSimulation()
        return True
    
    def update_pybullet_leaphand_qpos(self, leaphand_positions):
        """Update PyBullet with real robot LeapHand joint positions"""
        if leaphand_positions is None or len(leaphand_positions) != 16:
            return False
        
        leaphand_joint_indices = [11, 10, 12, 13, 16, 15, 17, 18, 21, 20, 22, 23, 25, 26, 27, 28]    
        for i, joint_idx in enumerate(leaphand_joint_indices):
            if i < len(leaphand_positions):
                # Apply -π offset for LeapHand
                joint_pos = leaphand_positions[i] - np.pi
                p.resetJointState(self.kinova_id, joint_idx, joint_pos)
        return True

    def get_ee_pose_from_pybullet(self):
        """Get end-effector pose from PyBullet simulation"""
        link_state = p.getLinkState(self.kinova_id, self.ee_link_index)
        ee_pos = np.array(link_state[4])
        ee_quat= np.array(link_state[5])
        ee_rot = R.from_quat(ee_quat).as_rotvec()
        return ee_pos, ee_rot
    
    def get_observation_from_robot(self):
        """Get observation from real robot"""
        # Get camera image
        image = self.get_camera_image()
        if image is None:
            return None
        
        # Update joint states
        current_joints = self.kinova_node.get_current_joint_angles()
        if current_joints is not None:
            self.current_joint_states = current_joints
            self.update_pybullet_arm_qpos(current_joints)
        
        # Get EE pose from PyBullet
        ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
        
        # Get gripper state
        try:
            self.current_leaphand_states = self.leap_node.read_pos()
        except:
            pass
        
        obs = {}
        
        # Handle different image modes
        if self.image_mode == 'split':
            # Split image mode: split the wide image into left and right
            left_img, right_img = self.split_image_left_right(image)
            obs['camera0_rgb_left'] = self.preprocess_image(left_img, self.left_shape)
            obs['camera0_rgb_right'] = self.preprocess_image(right_img, self.right_shape)
        else:
            # Single image mode
            obs['camera0_rgb'] = self.preprocess_image(image, self.single_shape)
        
        obs['robot0_eef_pos'] = ee_pos.astype(np.float32)
        obs['robot0_eef_rot_axis_angle'] = ee_rot.astype(np.float32)
        obs['robot0_gripper_width'] = self.current_leaphand_states.astype(np.float32)
        
        return obs
    
    def prepare_obs_dict(self, obs_history):
        """Prepare observation dictionary for policy inference"""
        from scipy.spatial.transform import Rotation as R
        
        obs_dict = {}
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
            poses = []
            for obs in obs_history:
                pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_rot_axis_angle']])
                poses.append(pose)
            poses = np.stack(poses, axis=0)
            
            pose_mat = pose_to_mat(poses)
            
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False
            )
            
            # sensor abs
            # obs_pose_mat = convert_pose_mat_rep(
            #     pose_mat,
            #     base_pose_mat=self.start_pose_mat,
            #     pose_rep='relative',
            #     backward=False
            # )

            obs_pose = mat_to_pose10d(obs_pose_mat)
            
            obs_dict['robot0_eef_pos'] = obs_pose[..., :3].astype(np.float32)
            obs_dict['robot0_eef_rot_axis_angle'] = obs_pose[..., 3:].astype(np.float32)
            
            # wrt_keys = self.policy.obs_encoder.low_dim_keys
            # wrt_keys = [key for key in wrt_keys if 'wrt_start' in key]

            # if len(wrt_keys) > 0:
            # Compute rotation relative to start
            rel_obs_pose_mat = convert_pose_mat_rep( 
                pose_mat, 
                base_pose_mat=self.start_pose_mat, 
                pose_rep='relative', 
                backward=False) 
            obs_pose_wrt_start = mat_to_pose10d(rel_obs_pose_mat) 
            # obs_dict['robot0_eef_pos_wrt_start'] = obs_pose_wrt_start[..., :3].astype(np.float32) 
            # obs_dict['robot0_eef_rot_axis_angle_wrt_start'] = obs_pose_wrt_start[..., 3:].astype(np.float32)
            # print("robot0_eef_pos_wrt_start: ", obs_dict['robot0_eef_pos_wrt_start'])
            # print("robot0_eef_rot_axis_angle_wrt_start: ", obs_dict['robot0_eef_rot_axis_angle_wrt_start'])
        
        # Gripper width: [T, gripper_dim]
        if 'robot0_gripper_width' in obs_history[0]:
            gripper = np.stack([obs['robot0_gripper_width'] for obs in obs_history], axis=0)
            obs_dict['robot0_gripper_width'] = gripper.astype(np.float32)
        print("obs_dict_pos: ", obs_dict['robot0_eef_pos'])
        print("obs_dict_rot: ", obs_dict['robot0_eef_rot_axis_angle'])
        return obs_dict
    
    def predict_action(self):
        """Predict action using diffusion policy"""
        current_obs = self.get_observation_from_robot()
        if current_obs is None:
            return None, None
        
        # Get obs_horizon from the first available camera key
        if self.image_mode == 'split':
            obs_horizon = self.shape_meta['obs']['camera0_rgb_left'].get('horizon', 2)
        else:
            obs_horizon = self.shape_meta['obs']['camera0_rgb'].get('horizon', 2)
        
        # Build observation history
        if self.past_obs is None:
            obs_history = [current_obs] * obs_horizon
        else:
            obs_history = [self.past_obs, current_obs]
        
        obs_dict_np = self.prepare_obs_dict(obs_history)
        
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            start_time = time.perf_counter()
            result = self.policy.predict_action(obs_dict)
            inference_time = time.perf_counter() - start_time
            print(f"Inference time: {inference_time:.4f}s")
        
        action_chunk = result['action_pred'][0].detach().cpu().numpy()
        
        return action_chunk, current_obs
    
    def convert_action_chunk_to_pose(self, action_chunk, current_obs):
        """Convert action chunk to world poses"""
        current_pose = np.concatenate([
            current_obs['robot0_eef_pos'],
            current_obs['robot0_eef_rot_axis_angle']
        ])
        current_pose_mat = pose_to_mat(current_pose)
        
        action_dim = action_chunk.shape[-1]
        if action_dim == 10:
            action_pose10d = action_chunk[..., :9]
            action_gripper = action_chunk[..., 9:10]
        elif action_dim == 25:
            action_pose10d = action_chunk[..., :9]
            action_gripper = action_chunk[..., 9:25]
        else:
            raise ValueError(f"Unsupported action dimension: {action_dim}")

        action_pose_mat = pose10d_to_mat(action_pose10d)
        
        action_mat = convert_pose_mat_rep(
            action_pose_mat,
            base_pose_mat=current_pose_mat,
            pose_rep=self.action_pose_repr,
            backward=True
        )

        # sensor abs
        # action_mat = convert_pose_mat_rep(
        #     action_pose_mat, 
        #     base_pose_mat=self.start_pose_mat, 
        #     pose_rep='relative',
        #     backward=True
        # )
        
        action_poses = mat_to_pose(action_mat)
        
        return action_poses, action_gripper
    
    def convert_action_25d_to_22d(self, action_25d):
        """Convert a 25-DOF raw action to 22-DOF by converting rotation_6d to axis_angle.
        
        Input:  action_25d[0:3]  = position (3)
                action_25d[3:9]  = rotation 6D (6)
                action_25d[9:25] = gripper (16)
        Output: action_22d[0:3]  = position (3)
                action_22d[3:6]  = rotation axis-angle (3)
                action_22d[6:22] = gripper (16)
        """
        pos = action_25d[0:3]
        rot_6d = action_25d[3:9]
        gripper = action_25d[9:25]
        rot_axis_angle = self.rot6d_to_axis_angle.forward(rot_6d)
        return np.concatenate([pos, rot_axis_angle, gripper])
    
    def log_action(self, action_25d):
        """Log converted action and append 6 real Kinova joint values per line."""
        action_22d = self.convert_action_25d_to_22d(action_25d)
        arm_joints = self.kinova_node.get_current_joint_angles()

        line_values = np.concatenate([np.asarray(action_22d), np.asarray(arm_joints)])
        line = ' '.join(f'{v:.6f}' for v in line_values)
        with open(self.action_log_path, 'a') as f:
            f.write(line + '\n')

    def execute_arm_joint_command(self, joint_positions):
        """Execute joint command on Kinova arm"""
        try:
            success = self.kinova_node.move_to_joint_angles(joint_positions, blocking=True)
            return success
        except Exception as e:
            rospy.logerr(f"Failed to execute arm command: {e}")
            return False
    
    def execute_action_chunk(self, action_chunk, current_obs, steps_per_inference=8):
        """Execute a chunk of actions on real robot"""
        from scipy.spatial.transform import Rotation as R
        print("action chunk length: ",len(action_chunk))
        num_actions = min(len(action_chunk), steps_per_inference)
        
        target_poses, gripper_qpos_all = self.convert_action_chunk_to_pose(action_chunk, current_obs)
        print("target_poses: ", target_poses)

        print(f"Executing {num_actions} actions")
        
        
        for i in range(num_actions):
            # if i == 0:
                # continue
            target_pos = target_poses[i, :3]
            target_rot_axis_angle = target_poses[i, 3:6]
            gripper_qpos = gripper_qpos_all[i]
            
            # Convert axis-angle to quaternion for PyBullet IK
            rot = R.from_rotvec(target_rot_axis_angle)
            target_quat_xyzw = rot.as_quat()
            
            # Calculate IK
            joint_positions = p.calculateInverseKinematics(
                self.kinova_id,
                self.ee_link_index,
                target_pos,
                target_quat_xyzw,
                maxNumIterations=500,
                residualThreshold=0.0001
            )
            
            # Execute arm command
            arm_joints = np.array(joint_positions[:6])
            self.execute_arm_joint_command(arm_joints)
            self.update_pybullet_arm_qpos(arm_joints)
            
            # Execute LeapHand command
            self.leap_node.set_leap(gripper_qpos)
            self.update_pybullet_leaphand_qpos(gripper_qpos)
                    
            print(f"  Action {i}: pos={target_pos}, rot={target_rot_axis_angle}, gripper={gripper_qpos}")
            print("Simulation pose after action:")
            sim_ee_pos, sim_ee_rot = self.get_ee_pose_from_pybullet()
            print(f"  Sim EE pos: {sim_ee_pos}, Sim EE rot (axis-angle): {sim_ee_rot}")
            print("--------------------------------")

            # Log the raw 25-DOF action to file as 22-DOF
            self.log_action(action_chunk[i])
            
            # Wait for execution
            rospy.sleep(config.ACTION_EXECUTION_DELAY)
            
            self.current_timestep += 1
            # time.sleep(1)
            # 在倒数第二帧和倒数第一帧时获取观测
            if i == num_actions - 2:
                self.past_obs = self.get_observation_from_robot()
            elif i == num_actions - 1:
                self.current_obs = self.get_observation_from_robot()
        
        # time.sleep(1)
        return True
    
    def run(self, max_steps=None, steps_per_inference=8):
        """Main execution loop"""
        rospy.loginfo("Starting real robot diffusion policy execution...")
        
        inference_count = 0
        self.policy.reset()
        
        rate = rospy.Rate(config.CONTROL_FREQUENCY)
        t0 = 0
        t1 = 0
        try:
            while not rospy.is_shutdown():
                if max_steps is not None and self.current_timestep >= max_steps:
                    break
                
                print(f"\n=== Inference {inference_count}, Timestep {self.current_timestep} ===")
                
                # Predict action
                action_chunk, current_obs = self.predict_action()
                
                t1 = time.time()
                print("Inference time: ", t1-t0)
                print(f"Time since last action execution: {t1 - t0:.4f}s")
                if action_chunk is not None:
                    print(f"Action chunk shape: {action_chunk.shape}")
                    self.execute_action_chunk(action_chunk, current_obs, steps_per_inference)
                    inference_count += 1
                else:
                    rospy.logwarn("No action predicted, waiting for sensor data...")
                    rate.sleep()
                    continue

                t0 = time.time()
                
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        finally:
            self.cleanup()
        
        print(f"Execution completed. Total inferences: {inference_count}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'kinova_node'):
                self.kinova_node.cancel_all_goals()
            p.disconnect()
            rospy.loginfo("Cleanup completed")
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy diffusion policy on real robot')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to diffusion policy checkpoint (.ckpt file or directory)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum number of timesteps to run')
    parser.add_argument('--steps-per-inference', type=int, default=8,
                        help='Number of action steps to execute per inference')
    
    args = parser.parse_args()
    
    try:
        system = RealDiffusionPolicy(
            args.checkpoint,
            args.device
        )
        system.run(args.max_steps, args.steps_per_inference)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
