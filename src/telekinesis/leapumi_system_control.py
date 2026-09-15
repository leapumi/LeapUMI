#!/usr/bin/env python3
from datetime import datetime
import pybullet as p
import numpy as np
import os
import time

from oculus_reader.scripts import *
from oculus_reader.scripts.reader import OculusReader

from geometry_msgs.msg import PoseArray
import sys
from pathlib import Path
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R

# import keyboard
from pynput.keyboard import Key, Listener
import rospy
import threading
import math

from collections import defaultdict

# from teleop.visualizer import RawScene
import h5py
import cv2
import termios
import tty

"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""
# visualizer = RawScene()

space_pressed = False
record_pressed = False

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

camera_base_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

leap2human_rot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

correction_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

reflection_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

Rx_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


# Convert 3*3 matrix to quaternion
def mat2quat(mat):
    q = np.zeros([4])
    q[3] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[0] = (mat[2][1] - mat[1][2]) / (4 * q[3])
    q[1] = (mat[0][2] - mat[2][0]) / (4 * q[3])
    q[2] = (mat[1][0] - mat[0][1]) / (4 * q[3])
    return q


# VR ==> MJ mapping when teleOp user is standing infront of the robot
def vrfront2mj(pose):
    pos = np.zeros([3])
    pos[0] = -1.0 * pose[2][3]
    pos[1] = -1.0 * pose[0][3]
    pos[2] = +1.0 * pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.0 * pose[2][:3]
    mat[1][:] = +1.0 * pose[0][:3]
    mat[2][:] = -1.0 * pose[1][:3]

    return pos, mat2quat(mat)

A = np.array([
    [1, 0 ,0],
    [0, 0, -1],
    [0, -1, 0]
])

B = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, -1, 0]
])

angle = - np.pi / 6  # 30 degrees in radians
Rz_30 = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
])

angle = - np.pi/6
Rx_30 = np.array([
    [1, 0, 0],
    [0, np.cos(angle), -np.sin(angle)],
    [0, np.sin(angle), np.cos(angle)]
])

angle = np.pi/6
Ry30 = np.array([
    [np.cos(angle), 0, np.sin(angle)],
    [0, 1, 0],
    [-np.sin(angle), 0, np.cos(angle)]
])

angle = -np.pi/6
Ry_30 = np.array([
    [np.cos(angle), 0, np.sin(angle)],
    [0, 1, 0],
    [-np.sin(angle), 0, np.cos(angle)]
])

# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = camera_base_rot @ pose[:3, 3]
    mat = A @ pose[:3, :3] @ B    
    # mat = Rx_30 @ Ry30 @ mat
    q = mat2quat(mat)

    return pos, q


def negQuat(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def mulQuat(qa, qb):
    res = np.zeros(4)
    res[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
    res[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
    res[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
    res[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]
    return res


def diffQuat(quat1, quat2):
    neg = negQuat(quat1)
    diff = mulQuat(quat2, neg)
    return diff


def quat_xyzw_to_rot_matrix(quat_xyzw):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix using scipy"""
    return R.from_quat(quat_xyzw).as_matrix()


def rot_matrix_to_quat_xyzw(rot_mat):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w) using scipy"""
    return R.from_matrix(rot_mat).as_quat()


class LeapNode:
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


class SystemPybulletIK:
    def __init__(self):
        rospy.init_node("teleoperation_node", anonymous=True)

        # start pybullet
        p.connect(p.DIRECT)
        # load right leap hand
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [4, 9, 14, 19]
        self.hand_q = None

        leap_path_src = os.path.join(
            path_src, "leap_hand_mesh_right/robot_pybullet.urdf"
        )
        self.leapId = p.loadURDF(
            leap_path_src,
            [0.0, 0.038, 0.098],
            p.getQuaternionFromEuler([0, -1.57, 0]),
            useFixedBase=True,
        )

        self.leapnumJoints = p.getNumJoints(self.leapId)

        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)

        print("=" * 20)
        self.operator2mano = OPERATOR2MANO_RIGHT
        self.leap_node = LeapNode()
        self.oculus_reader = OculusReader()
        self.joint_names = [
            "Index1",
            "Index2",
            "Index3",
            "IndexTip",
            "Middle1",
            "Middle2",
            "Middle3",
            "MiddleTip",
            "Ring1",
            "Ring2",
            "Ring3",
            "RingTip",
            "Thumb1",
            "Thumb2",
            "Thumb3",
            "ThumbTip",
        ]

        # Keyboard input thread variables
        self.should_quit = False
        self.keyboard_lock = threading.Lock()
        
        # Start the keyboard thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        print("OculusReader success.")
        print("+" * 20)

    def keyboard_loop(self):
        """Keyboard thread loop - continuously monitors keyboard input"""
        import select
        import termios
        import tty
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setcbreak(fd)
            while not self.should_quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    with self.keyboard_lock:
                        if key == 'q' or key == 'Q':
                            self.should_quit = True
                            print("Quit signal received")
                        if key == 'a' or key == 'A':
                            target = [3.2060199,3.2366996,3.186078,3.1354568,3.3057287,3.1431267,3.3778257,3.0909712,3.2366996,3.3333402,3.3333402,3.126253,4.1662917,3.4560587,3.7183695,3.1615343]
                            target = np.array(target)
                            self.leap_node.set_leap(target)
                            
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def cleanup(self):
        """Cleanup resources when shutting down"""
        print("Cleaning up keyboard thread...")
        self.should_quit = True
        if hasattr(self, 'keyboard_thread'):
            self.keyboard_thread.join(timeout=1.0)
        print("Cleanup completed")

    def compute_IK(self, hand_pos, rot):
        p.stepSimulation()

        index_mcp_pos = hand_pos[0]
        index_pip_pos = hand_pos[1]
        index_dip_pos = hand_pos[2]
        index_tip_pos = hand_pos[3]
        middle_mcp_pos = hand_pos[4]
        middle_pip_pos = hand_pos[5]
        middle_dip_pos = hand_pos[6]
        middle_tip_pos = hand_pos[7]
        ring_mcp_pos = hand_pos[8]
        ring_pip_pos = hand_pos[9]
        ring_dip_pos = hand_pos[10]
        ring_tip_pos = hand_pos[11]
        thumb_mcp_pos = hand_pos[12]
        thumb_pip_pos = hand_pos[13]
        thumb_dip_pos = hand_pos[14]
        thumb_tip_pos = hand_pos[15]

        index_mcp_rot = rot[0]
        index_pip_rot = rot[1]
        index_dip_rot = rot[2]
        index_tip_rot = rot[3]
        middle_mcp_rot = rot[4]
        middle_pip_rot = rot[5]
        middle_dip_rot = rot[6]
        middle_tip_rot = rot[7]
        ring_mcp_rot = rot[8]
        ring_pip_rot = rot[9]
        ring_dip_rot = rot[10]
        ring_tip_rot = rot[11]
        thumb_mcp_rot = rot[12]
        thumb_pip_rot = rot[13]
        thumb_dip_rot = rot[14]
        thumb_tip_rot = rot[15]

        leapEndEffectorPos = [
            index_tip_pos,
            middle_tip_pos,
            ring_tip_pos,
            thumb_tip_pos,
        ]

        leapEndEffectorRot = [
            index_tip_rot,
            middle_tip_rot,
            ring_tip_rot,
            thumb_tip_rot,
        ]

        # Compute the joint angles for the leap hand and send qpos
        leap_jointPoses = p.calculateInverseKinematics2(
            self.leapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            leapEndEffectorRot,
            solver=p.IK_DLS,
            maxNumIterations=500,
            residualThreshold=0.0001,
        )

        leap_whole_jointPoses = (
            leap_jointPoses[0:4]
            + (0.0,)
            + leap_jointPoses[4:8]
            + (0.0,)
            + leap_jointPoses[8:12]
            + (0.0,)
            + leap_jointPoses[12:16]
            + (0.0,)
        )
        leap_whole_jointPoses = list(leap_whole_jointPoses)
        
        for i in range(self.leapnumJoints):
            p.setJointMotorControl2(
                bodyIndex=self.leapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=leap_whole_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        real_robot_hand_q[0:4] = leap_jointPoses[0:4]
        real_robot_hand_q[4:8] = leap_jointPoses[4:8]
        real_robot_hand_q[8:12] = leap_jointPoses[8:12]
        real_robot_hand_q[12:16] = leap_jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        self.leap_node.set_allegro(real_robot_hand_q)
        self.hand_q = real_robot_hand_q
        
        return 

    def operation(self):
        # first loop: for each trajectory
        while True:
            # Check quit signal from keyboard thread
            with self.keyboard_lock:
                if self.should_quit:
                    break
                

            
            joint_pos = self.oculus_reader.get_joint_transformations()[1]

            # poll input device
            transformations, buttons = (
                self.oculus_reader.get_transformations_and_buttons()
            )
            print("----------------")
            # receive info from VR
            if transformations and "r" in transformations:
                # leaphand
                pos = []
                final_pos = []
                rot = []
                if joint_pos == {}:
                    print("not receive joint pos")
                    continue
                else:
                    mediapipe_wrist_rot = joint_pos["WristRoot"][:3, :3]
                    wrist_position = joint_pos["WristRoot"][:3, 3]

                    for joint_name in self.joint_names:
                        joint_transformation = joint_pos[joint_name]

                        pos.append(joint_transformation[:3, 3] - wrist_position)
                        if joint_name == "MiddleTip":
                            pos[-1][0] += 0.009375
                        if joint_name == "RingTip":
                            pos[-1][0] -= 0.004375
                        if joint_name == "IndexTip":
                            pos[-1][0] -= 0.00125
                        if joint_name == "ThumbTip":
                            pos[-1][0] += 0.00375
                        pos[-1] = pos[-1] @ mediapipe_wrist_rot @ self.operator2mano

                        # Turn the rotation matrix into quaternion (pure numpy)
                        rotation = (
                            joint_transformation[:3, :3]
                            @ mediapipe_wrist_rot
                            @ self.operator2mano
                        )
                        # Convert matrix to quaternion (xyzw format for PyBullet)
                        quaternion = rot_matrix_to_quat_xyzw(rotation)
                        rot.append(quaternion)

                        final_pos.append([
                            pos[-1][0] * self.glove_to_leap_mapping_scale * 1.15,
                            pos[-1][1] * self.glove_to_leap_mapping_scale,
                            pos[-1][2] * self.glove_to_leap_mapping_scale,
                        ])
                        final_pos[-1][2] -= 0.05

                self.compute_IK(final_pos, rot)
        
        # Cleanup when exiting
        self.cleanup()

def main(args=None):

    leappybulletik = SystemPybulletIK()

    leappybulletik.operation()


if __name__ == "__main__":
    main()
