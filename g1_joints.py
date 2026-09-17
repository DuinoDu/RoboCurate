"""Unitree G1 29-DOF joint layout as it appears in raw v1.4 MCAP topics.

This is the *native* ordering used by `unitree_hg/msg/LowState.motor_state[i]`,
`unitree_hg/msg/LowCmd.motor_cmd[i]`, the 29 floats of `/action/humanoid`, and
`LocalRetargetFrame.reference_qpos[7:36]`.

It is deliberately NOT the `camera_first` dataset ordering. `g1_layout.py`
re-packs this native order into the 33-wide HoloBrain layout via
`pack_camera_first`, whose slicing (`body[:, 15:22]` = left arm,
`body[:, 22:29]` = right arm, `body[:, 12:15]` = waist, `body[:, 0:6]` = left
leg, `body[:, 6:12]` = right leg) is the authority for the group boundaries
encoded here.
"""

from __future__ import annotations

# Kinematic chains in native lowstate order.
LEG_JOINTS = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
)
WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

BODY_JOINT_NAMES: tuple[str, ...] = (
    tuple(f"left_{name}" for name in LEG_JOINTS)
    + tuple(f"right_{name}" for name in LEG_JOINTS)
    + WAIST_JOINTS
    + tuple(f"left_{name}" for name in ARM_JOINTS)
    + tuple(f"right_{name}" for name in ARM_JOINTS)
)

assert len(BODY_JOINT_NAMES) == 29, len(BODY_JOINT_NAMES)

# Contiguous [start, stop) spans matching pack_camera_first's slicing.
JOINT_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("left_leg", 0, 6),
    ("right_leg", 6, 12),
    ("waist", 12, 15),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
)

GROUP_OF_INDEX: tuple[str, ...] = tuple(
    next(name for name, start, stop in JOINT_GROUPS if start <= i < stop)
    for i in range(29)
)

# unitree_hg LowState/LowCmd arrays are a fixed 35 entries. The G1 29-DOF
# configuration only drives 0..28; 29..34 are unused padding in this build and
# are dropped rather than shown as phantom joints.
MOTOR_ARRAY_WIDTH = 35
BODY_DOF = 29

# BrainCo hands report 6 motors per side in SDK order. `q` is normalised to
# [0, 1] by the driver (`positions[i] / 1000`), where 0 is fully open and 1 is
# fully closed under the default BRAINCO_OPEN_IS_ONE=false.
#
# The command stream is not a free 6-vector: `target_q_for_grip` drives indices
# 2..5 with the raw grip amount, scales the thumb by BRAINCO_THUMB_CLOSE_SCALE
# (0.55), and confines thumb_aux to [0.8, 1.0] — so thumb_aux never
# approaches 0
# even with the hand fully open. `convert_mcap_to_holobrain.py` medians
# `positions[2:6]` for exactly this reason, and the viewer highlights
# those four
# as the grip-carrying fingers.
HAND_JOINT_NAMES: tuple[str, ...] = (
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
)
HAND_DOF = len(HAND_JOINT_NAMES)

# Indices whose q tracks the grip amount directly.
HAND_GRIP_SPAN = (2, 6)

# reference_qpos[36] = root_pos[3] + root_quat_wxyz[4] + dof_pos[29]
REF_ROOT_POS = ("root_x", "root_y", "root_z")
REF_ROOT_QUAT = ("qw", "qx", "qy", "qz")
REF_QPOS_WIDTH = len(REF_ROOT_POS) + len(REF_ROOT_QUAT) + BODY_DOF

assert REF_QPOS_WIDTH == 36, REF_QPOS_WIDTH
