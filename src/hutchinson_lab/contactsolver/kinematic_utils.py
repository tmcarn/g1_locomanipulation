from dataclasses import dataclass

import pinocchio as pin
import mujoco
import numpy as np
from constants import *

@dataclass
class CDRef:
    x_ref:   np.ndarray | None = None
    xd_ref:  np.ndarray | None = None
    xdd_ref: np.ndarray | None = None

@dataclass
class FRef:
    f_hand: np.ndarray | None = None
    f_foot: np.ndarray | None = None


class SceneModel:
    """
    Owns the robot/box Pinocchio models, frame ids, and mass/geometry
    constants — built once, shared by CentroidalSolver and WBC so neither
    reloads the MJCF nor risks drifting out of sync with the other.
    """

    def __init__(self, contact_height_ratio=0.75):
        mj_model = mujoco.MjModel.from_xml_path(ROBOT_PATH)
        self.tau_max = mj_model.actuator_forcerange[:, 1]   # now real, since forcerange is set in the XML

        self.robot_model = pin.buildModelFromMJCF(ROBOT_PATH)
        self.robot_data = self.robot_model.createData()

        self.nq_robot = self.robot_model.nq

        self.box_model = pin.buildModelFromMJCF(BOX_PATH)
        self.box_data = self.box_model.createData()
        self.box_fid = self.box_model.getFrameId("box")

        self.frame_idxs = {name: self.robot_model.getFrameId(name) for name in SITE_NAMES}

        self.robot_mass = sum(self.robot_model.inertias[i].mass
                               for i in range(1, self.robot_model.njoints))

        self.g = 9.81

        # Friction Coefficients
        self.mu_ground_foot = 0.6
        self.mu_box_hand = 1.0

        self._set_box_config(contact_height_ratio)

    def _set_box_config(self, contact_height_ratio):
        box_jid = self.box_model.getJointId("box_x")
        self.box_mass = self.box_model.inertias[box_jid].mass

        mj_model = mujoco.MjModel.from_xml_path(BOX_PATH)
        box_geom_id = mj_model.geom("box_geom").id
        self.box_half_extents = mj_model.geom_size[box_geom_id]

        # viscous box-ground friction, read from the sim so the two can't drift
        self.box_damping = float(mj_model.dof_damping[mj_model.joint("box_x").dofadr[0]])

        self.b_contact_pos = np.array([-self.box_half_extents[0], contact_height_ratio * 2 * self.box_half_extents[2]])
        self.b_grf_pos = np.array([0.0, -self.box_half_extents[2]])
        self.w_box_pos = np.array([0.0, self.box_half_extents[2]])

    def get_joint_states(self, q, dq):
        q_robot, dq_robot = q[:self.nq_robot], dq[:self.nq_robot]
        q_box, dq_box = q[self.nq_robot:], dq[self.nq_robot:]
        return q_robot, dq_robot, q_box, dq_box

    def get_centroidal_state(self, q_robot, dq_robot):
        pin.centerOfMass(self.robot_model, self.robot_data, q_robot, dq_robot)
        p_com_full = self.robot_data.com[0]

        h = pin.computeCentroidalMomentum(self.robot_model, self.robot_data, q_robot, dq_robot)

        p_com = np.array([p_com_full[0], p_com_full[2]])
        l = np.array([h.linear[0], h.linear[2]])
        L = h.angular[1]

        return p_com, l, L

    def get_box_state(self, q_box, dq_box):
        pin.forwardKinematics(self.box_model, self.box_data, q_box, dq_box)
        pin.updateFramePlacements(self.box_model, self.box_data)

        box_pose = self.box_data.oMf[self.box_fid]
        box_vel = pin.getFrameVelocity(self.box_model, self.box_data, self.box_fid,
                                        pin.LOCAL_WORLD_ALIGNED)
        return box_pose.translation[0], box_vel.linear[0]

    def get_contact_position(self, q_robot, site_id):
        pin.forwardKinematics(self.robot_model, self.robot_data, q_robot)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        pos = self.robot_data.oMf[site_id].translation
        return np.array([pos[0], pos[2]])  # planar x, z

    def get_contact_locations(self, q_robot):
        pin.forwardKinematics(self.robot_model, self.robot_data, q_robot)
        pin.updateFramePlacements(self.robot_model, self.robot_data)

        hand = self.robot_data.oMf[self.frame_idxs["right_hand"]].translation
        foot = self.robot_data.oMf[self.frame_idxs["right_foot"]].translation

        return np.array(hand)[[0, 2]], np.array(foot)[[0, 2]]

    def get_com_jacobian(self, q_robot):
        J = pin.jacobianCenterOfMass(self.robot_model, self.robot_data, q_robot)
        return J[[0, 2], :]

    def get_contact_jacobian(self, q_robot, frame_name):
        pin.computeJointJacobians(self.robot_model, self.robot_data, q_robot)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        J = pin.getFrameJacobian(self.robot_model, self.robot_data,
                                self.frame_idxs[frame_name], pin.LOCAL_WORLD_ALIGNED)
        return J[[0, 2], :]   # planar: x, z rows only

    def get_terms(self, q_rob, dq_rob):
        pin.computeAllTerms(self.robot_model, self.robot_data, q_rob, dq_rob)
        M = self.robot_data.M
        h = pin.nonLinearEffects(self.robot_model, self.robot_data, q_rob, dq_rob)
        return M, h


def get_interpolated_ref(node0: CDRef, node1: CDRef, frac: float) -> CDRef:
    return CDRef(
        x_ref=node0.x_ref + frac * (node1.x_ref - node0.x_ref),
        xd_ref=node0.xd_ref + frac * (node1.xd_ref - node0.xd_ref),
        xdd_ref=node1.xdd_ref,   # held constant across the node
    )