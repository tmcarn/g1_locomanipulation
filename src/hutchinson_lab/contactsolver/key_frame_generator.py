import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np
from constants import *

from kinematic_utils import SceneModel


class Solver:
    def __init__(self, scene:SceneModel):
        self.scene = scene

        self.cmodel = cpin.Model(self.scene.robot_model)
        self.cdata = self.cmodel.createData()

        # Get frame indices for available sites
        self.frame_idxs = {}
        for site in SITE_NAMES:
            try:
                self.frame_idxs[site] = self.cmodel.getFrameId(site)
            except ValueError:
                print(f"Warning: Site '{site}' not found in model")

        self.foot_offset = 0.75 # meters
        self.foot_z = 0.1  # Ground level

    def frame_pos(self, qt, site_name):
        cpin.forwardKinematics(self.cmodel, self.cdata, qt)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        return self.cdata.oMf[int(self.frame_idxs[site_name])].translation

    def solve_ik(self, q_guess=None):
        rfoot_x = self.scene.w_box_pos[0] - self.scene.box_half_extents[0] - self.foot_offset
        target_hand_xz = self.scene.b_contact_pos

        q_sym = ca.SX.sym("q", self.cmodel.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata, q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        rhand_pos = self.cdata.oMf[int(self.frame_idxs['right_hand'])].translation
        rfoot_pos = self.cdata.oMf[int(self.frame_idxs['right_foot'])].translation

        cpin.centerOfMass(self.cmodel, self.cdata, q_sym)
        com_pos = self.cdata.com[0]

        fk_fn = ca.Function("fk", [q_sym], [rhand_pos, rfoot_pos, com_pos])

        q_min = np.array(self.scene.robot_model.lowerPositionLimit).flatten()
        q_max = np.array(self.scene.robot_model.upperPositionLimit).flatten()

        opti = ca.Opti()
        q = opti.variable(self.cmodel.nq)
        rhand_p, rfoot_p, com_p = fk_fn(q)

        # Hand position constraints
        opti.subject_to(rhand_p[0] == target_hand_xz[0])
        opti.subject_to(rhand_p[2] == target_hand_xz[1])

        # Foot position constraints
        opti.subject_to(rfoot_p[2] == self.foot_z)
        opti.subject_to(rfoot_p[0] == rfoot_x)

        # Torso angle (free joint index 2)
        opti.subject_to(q[2] == 0.2)

        # CoM over foot
        delta = 0.05
        opti.subject_to(com_p[0] >= rfoot_p[0] - delta)
        opti.subject_to(com_p[0] <= rfoot_p[0] + delta)

        # Joint limits
        opti.subject_to(opti.bounded(q_min, q, q_max))

        # Minimize deviation from comfortable configuration
        q0 = np.zeros(self.cmodel.nq)
        q0_actuated = q0[3:]

        opti.minimize(ca.sumsqr(q[3:] - q0_actuated))
        opti.set_initial(q, q0)

        opti.solver('ipopt')
        sol = opti.solve()
        return sol.value(q)


if __name__ == "__main__":
    scene = SceneModel()
    
    solver = Solver(scene)  # pass a SceneModel instance to scale standoff with box size
    q0 = solver.solve_ik()
