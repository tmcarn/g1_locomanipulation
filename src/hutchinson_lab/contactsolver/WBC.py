from kinematic_utils import SceneModel, CDRef, FRef
from logger import WBCLogger

import casadi as ca
import numpy as np


class WBC:
    ACTIVE_CONTACTS = ['right_hand', 'right_foot']
    STATIC_CONTACTS = ['right_foot'] # Stay planted on the floor
    DYNAMIC_CONTACTS = ['right_hand'] # Move with the Box

    def __init__(self, scene: SceneModel):
        self.scene = scene
        self.gains = {'com': (10, 2),
                      'box': (10, 2)}

        self.nv = self.scene.robot_model.nv
        self.na = self.nv - 3  # planar floating base: x, z, pitch unactuated

        self._prev_sol = None
        self._build_qp()

    def xdd_des(self, cd_ref: CDRef, cd_meas: CDRef, K):
        ''' Returns desired task space acceleration, based on CDMPC reference, tracked with a PD Controller'''
        Kp, Kd = K
        xdd_des = (cd_ref.xdd_ref
                   + Kp * (cd_ref.x_ref - cd_meas.x_ref)
                   + Kd * (cd_ref.xd_ref - cd_meas.xd_ref))
        return xdd_des

    def get_meas(self, q, dq):
        q_rob, dq_rob, q_box, dq_box = self.scene.get_joint_states(q, dq)
        p_com, l, L = self.scene.get_centroidal_state(q_rob, dq_rob)
        p_box, v_box = self.scene.get_box_state(q_box, dq_box)

        p_com_meas = CDRef(x_ref=p_com, xd_ref=l / self.scene.robot_mass)
        p_box_meas = CDRef(x_ref=p_box, xd_ref=v_box)
        return p_com_meas, p_box_meas

    def _build_qp(self):
        """Build the QP structure once: variables, parameters, constraints,
        cost, and solver. Numeric values (M, h, Jacobians, desired accel/force)
        are filled in per-tick via set_value in solve()."""
        self.opti = ca.Opti()

        nv, na = self.nv, self.na
        contacts = self.ACTIVE_CONTACTS

        # --- decision variables ---
        self.qdd = self.opti.variable(nv)
        self.tau = self.opti.variable(na)
        self.fc = {name: self.opti.variable(2) for name in contacts}
        fc_stack = ca.vertcat(*[self.fc[n] for n in contacts])

        # --- parameters: everything that's numeric-per-tick ---
        self.M_param = self.opti.parameter(nv, nv)
        self.h_param = self.opti.parameter(nv)
        self.Jcom_param = self.opti.parameter(2, nv)
        self.Jc_param = {name: self.opti.parameter(2, nv) for name in contacts}
        self.pdd_com_des_param = self.opti.parameter(2)
        self.pdd_hand_target_param = {name: self.opti.parameter(2) for name in self.DYNAMIC_CONTACTS}
        self.f_des_param = {name: self.opti.parameter(2) for name in contacts}

        Jc_stack_param = ca.vertcat(*[self.Jc_param[n] for n in contacts])

        # --- dynamics: unactuated rows constrain, actuated rows define tau ---
        self.opti.subject_to(
            self.M_param[0:3, :] @ self.qdd + self.h_param[0:3]
            == Jc_stack_param[:, 0:3].T @ fc_stack
        )
        self.opti.subject_to(
            self.M_param[3:, :] @ self.qdd + self.h_param[3:]
            == self.tau + Jc_stack_param[:, 3:].T @ fc_stack
        )

        # --- rigid-contact no-slip for the feet ---
        for name in self.STATIC_CONTACTS:
            self.opti.subject_to(self.Jc_param[name] @ self.qdd == 0)

        # # --- rigid-contact for the hands (must accelerate with the box) ---
        # for name in self.DYNAMIC_CONTACTS:
        #     opti.subject_to(self.Jc_param[name][0, :] @ self.qdd == self.pdd_box_des_param)
        #     opti.subject_to(self.Jc_param[name][1, :] @ self.qdd == 0)

        self.opti.subject_to(self.opti.bounded(-self.scene.tau_max, self.tau, self.scene.tau_max))

        # --- friction cones ---
        mu = 1.0
        for name in self.DYNAMIC_CONTACTS:
            self.opti.subject_to(self.fc[name][1] <=  mu * (-self.fc[name][0]))
            self.opti.subject_to(self.fc[name][1] >= -mu * (-self.fc[name][0]))
            self.opti.subject_to(self.fc[name][0] <= 0)

        for name in self.STATIC_CONTACTS:
            self.opti.subject_to(self.fc[name][0] <=  mu * self.fc[name][1])
            self.opti.subject_to(self.fc[name][0] >= -mu * self.fc[name][1])
            self.opti.subject_to(self.fc[name][1] >= 0)

        # --- cost ---
        w_task, w_force, w_reg = 10.0, 5.0, 1e-4
        cost = w_task * ca.sumsqr(self.Jcom_param @ self.qdd - self.pdd_com_des_param)
        for name in contacts:
            cost += w_force * ca.sumsqr(self.fc[name] - self.f_des_param[name])

        w_hand_accel = 1000.0  # tune: high enough to matter, not so high it re-creates the rigidity problem
        for name in self.DYNAMIC_CONTACTS:
            cost += w_hand_accel * ca.sumsqr(self.Jc_param[name] @ self.qdd - self.pdd_hand_target_param[name])

        cost += w_reg * (ca.sumsqr(self.tau))
        self.opti.minimize(cost)

        self.opti.solver('ipopt', {'print_time': False}, {
            'acceptable_tol': 1e-3,
            'acceptable_iter': 10,
            'max_iter': 500,
            'print_level': 0,
            'sb': 'yes',
        })

    def solve(self, q, dq, p_com_ref:CDRef, box_ref:CDRef, f_ref: FRef, logger: WBCLogger=None, tick: int=None):

        print("Force Ref:", f_ref.f_hand, f_ref.f_foot)
        p_com_meas, p_box_meas = self.get_meas(q, dq)
        q_rob, dq_rob, q_box, dq_box = self.scene.get_joint_states(q, dq)

        pdd_com_des = self.xdd_des(p_com_ref, p_com_meas, self.gains['com'])
        # pdd_box_des computed for parity with earlier version; not yet wired
        # into the cost (box tracking currently enters only via f_ref)
        pdd_box_des = self.xdd_des(box_ref, p_box_meas, self.gains['box'])

        
        # print("xdd_ref:", box_ref.xdd_ref)
        print(f"BOX REF POS:{box_ref.x_ref}, BOX MEAS POS:{p_box_meas.x_ref}")
        print(f"BOX REF VEL:{box_ref.xd_ref}, BOX MEAS VEL:{p_box_meas.xd_ref}")
        print(f"BOX XDD DES:", pdd_box_des)

        M, h = self.scene.get_terms(q_rob, dq_rob)
        J_com = self.scene.get_com_jacobian(q_rob)
        J_c = {name: self.scene.get_contact_jacobian(q_rob, name) for name in self.ACTIVE_CONTACTS}

        # --- per-hand target: full 2D position/velocity PD toward the box contact point ---
        Kp_gap, Kd_gap = 200, 30
        pdd_hand_target = {}
        for name in self.DYNAMIC_CONTACTS:
            hand_pos = self.scene.get_contact_position(q_rob, name)
            hand_vel = J_c[name] @ dq_rob

            target_pos = np.array([
                p_box_meas.x_ref + self.scene.b_contact_pos[0],
                self.scene.w_box_pos[1] + self.scene.b_contact_pos[1],
            ])
            target_vel = np.array([p_box_meas.xd_ref, 0.0])

            gap_pos_err = target_pos - hand_pos
            gap_vel_err = target_vel - hand_vel

            pdd_hand_target[name] = (np.array([box_ref.xdd_ref, 0.0])
                                    + Kp_gap * gap_pos_err + Kd_gap * gap_vel_err)

            f_des = {'right_hand': f_ref.f_hand, 'right_foot': f_ref.f_foot}

            hand_pos = self.scene.get_contact_position(q_rob, name) # (x, z) via FK — need this method if not present
            hand_vel = J_c[name] @ dq_rob  # (2,) — both x and z rows of the Jacobian

            target_pos = np.array([
                p_box_meas.x_ref + self.scene.b_contact_pos[0],
                self.scene.w_box_pos[1] + self.scene.b_contact_pos[1],
            ])

            target_vel = np.array([p_box_meas.xd_ref, 0.0])  # box_z_world is constant, so z-velocity target is 0

            gap_pos_err = target_pos - hand_pos
            gap_vel_err = target_vel - hand_vel

            Kp_gap, Kd_gap = 200, 30
            pdd_hand_target = np.array([box_ref.xdd_ref, 0.0]) + Kp_gap * gap_pos_err + Kd_gap * gap_vel_err
            
        self.opti.set_value(self.M_param, M)
        self.opti.set_value(self.h_param, h)
        self.opti.set_value(self.Jcom_param, J_com)
        self.opti.set_value(self.pdd_com_des_param, pdd_com_des)
        self.opti.set_value(self.pdd_box_des_param, pdd_box_des)


        for name in self.DYNAMIC_CONTACTS:
            self.opti.set_value(self.pdd_hand_target_param[name], pdd_hand_target[name])

        for name in self.ACTIVE_CONTACTS:
            self.opti.set_value(self.Jc_param[name], J_c[name])
            self.opti.set_value(self.f_des_param[name], f_des[name])

        if self._prev_sol is not None:
            prev = self._prev_sol
            self.opti.set_initial(self.qdd, prev.value(self.qdd))
            self.opti.set_initial(self.tau, prev.value(self.tau))
            for name in self.ACTIVE_CONTACTS:
                self.opti.set_initial(self.fc[name], prev.value(self.fc[name]))

        try:
            sol = self.opti.solve()
            self._prev_sol = sol

            fc_sol = {name: sol.value(self.fc[name]) for name in self.ACTIVE_CONTACTS}
            tau_sol = sol.value(self.tau)
            qdd_sol = sol.value(self.qdd)

            print("Jc @ qdd (hand x accel):", (J_c['right_hand'][0,:] @ qdd_sol))
            print("pdd_box_des:", pdd_box_des)

            if logger is not None:
                logger.record(tick, f_des, fc_sol, tau_sol, qdd_sol,
                            pdd_com_des, J_com)

            return tau_sol
        
        except RuntimeError:
            print("qdd:", self.opti.debug.value(self.qdd))
            print("tau:", self.opti.debug.value(self.tau))
            for name in self.ACTIVE_CONTACTS:
                print(f"fc[{name}]:", self.opti.debug.value(self.fc[name]))
            raise
        