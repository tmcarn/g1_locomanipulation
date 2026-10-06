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

        self.frame_idxs = np.zeros(4, dtype=int)
        for i, site in enumerate(SITE_NAMES):
            self.frame_idxs[i] = self.cmodel.getFrameId(site)

        self.lfoot_offset = 0.75 #meters
        self.rfoot_offset = 1.25 #meters

        self.foot_z = 0.0

    def frame_pos(self, qt, site_idx):
        cpin.forwardKinematics(self.cmodel, self.cdata, qt)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        return self.cdata.oMf[int(self.frame_idxs[site_idx])].translation  # 3-vector; take [0],[2] for planar x,z

    def solve_ik(self, q_guess=None):
        
        lfoot_x = self.scene.w_box_pos[0] - self.scene.box_half_extents[0] - self.lfoot_offset
        rfoot_x = self.scene.w_box_pos[0] - self.scene.box_half_extents[0] - self.rfoot_offset

        target_hand_xz = self.scene.b_contact_pos

        print(lfoot_x, rfoot_x, target_hand_xz)

        q_sym = ca.SX.sym("q", self.cmodel.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata, q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        rhand_pos  = self.cdata.oMf[int(self.frame_idxs[0])].translation
        lhand_pos  = self.cdata.oMf[int(self.frame_idxs[1])].translation

        rfoot_pos = self.cdata.oMf[int(self.frame_idxs[2])].translation
        lfoot_pos = self.cdata.oMf[int(self.frame_idxs[3])].translation

        cpin.centerOfMass(self.cmodel, self.cdata, q_sym)   # computed on the SX symbol
        com_pos = self.cdata.com[0]

        fk_fn = ca.Function("fk", [q_sym], [rhand_pos, lhand_pos, rfoot_pos, lfoot_pos, com_pos])

        q_min = np.array(self.scene.robot_model.lowerPositionLimit).flatten()
        q_max = np.array(self.scene.robot_model.upperPositionLimit).flatten()

        opti = ca.Opti()
        q = opti.variable(self.cmodel.nq)
        rhand_p, lhand_p, rfoot_p, lfoot_p, com_p = fk_fn(q)

        opti.subject_to(rhand_p[0] == target_hand_xz[0])
        opti.subject_to(rhand_p[2] == target_hand_xz[1])
        opti.subject_to(lhand_p[0] == target_hand_xz[0])
        opti.subject_to(lhand_p[2] == target_hand_xz[1])

        # Foot Position Contraints
        opti.subject_to(rfoot_p[2] == self.foot_z)
        opti.subject_to(lfoot_p[2] == self.foot_z)

        opti.subject_to(rfoot_p[0] == rfoot_x)
        opti.subject_to(lfoot_p[0] == lfoot_x)

        # Torso Angle
        opti.subject_to(q[2] == 0.2)  # Torso Upright

        # CoM over Feet
        delta = 0.05
        opti.subject_to(com_p[0] >= ca.fmin(rfoot_p[0], lfoot_p[0]) - delta)
        opti.subject_to(com_p[0] <= ca.fmax(rfoot_p[0], lfoot_p[0]) + delta)

        # Joint Limits Satisfied
        opti.subject_to(opti.bounded(q_min, q, q_max))

        # Comfortable Configuration
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
