from kinematic_utils import SceneModel, CDRef, FRef

import casadi as ca
import pinocchio as pin
import mujoco
import pinocchio.casadi as cpin
import numpy as np
from constants import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CentroidalSolver:
    def __init__(self, scene: SceneModel):
        self.scene = scene

        self.nq_robot = 11

        self.robot_mass = self.scene.robot_mass
        self.z_nom = 1.2
        self.z_min = 1.0

        self.a_max = 1.0

        # Friction Coefficients
        self.mu_ground_box = 1.0
        self.mu_ground_foot = 0.6
        self.mu_box_hand = 1.0

        # MPC Parameters 
        self.T = MPC_T # Planning Horizon
        self.dt = MPC_DT

        self._build_nlp()
        self._prev_sol = None

    def box_dynamics(self, f_rhand_x, f_lhand_x, f_ground_x):
        f_net_x = -f_rhand_x - f_lhand_x + f_ground_x
        return f_net_x / self.scene.box_mass

    def _build_nlp(self):
        self.opti = ca.Opti()

        # parameters — set fresh values each solve, don't rebuild the graph
        self.p_com_0_param = self.opti.parameter(2)
        self.l_0_param     = self.opti.parameter(2)
        self.k_0_param     = self.opti.parameter(1)
        self.p_box_0_param = self.opti.parameter(1)
        self.v_box_0_param = self.opti.parameter(1)
        self.V_target_param = self.opti.parameter(1)
        self.rfoot_param = self.opti.parameter(2)   # contact locations also vary per solve
        self.lfoot_param = self.opti.parameter(2)

        self.p_box_prev_param = [self.opti.parameter(1) for _ in range(self.T)]
        self.p_com_prev_param = [self.opti.parameter(2) for _ in range(self.T)]

        '''Decision Variables'''
        # Centroidal Dynamics
        self.p_com = [self.opti.variable(2) for _ in range(self.T+1)]
        self.l     = [self.opti.variable(2) for _ in range(self.T+1)]
        self.k     = [self.opti.variable(1) for _ in range(self.T+1)]

        # Box Dynamics
        self.p_box = [self.opti.variable(1) for _ in range(self.T+1)]
        self.v_box = [self.opti.variable(1) for _ in range(self.T+1)]

        # Contact Forces (acting on the Robot)
        self.f_rhand  = [self.opti.variable(2) for _ in range(self.T)]
        self.f_lhand  = [self.opti.variable(2) for _ in range(self.T)]
        self.f_rfoot  = [self.opti.variable(2) for _ in range(self.T)]
        self.f_lfoot  = [self.opti.variable(2) for _ in range(self.T)]

        self.f_ground_x = [self.opti.variable(1) for _ in range(self.T)]

        '''Constraints'''
        # x[0] = x0
        self.opti.subject_to(self.p_com[0] == self.p_com_0_param)
        self.opti.subject_to(self.l[0] == self.l_0_param)
        self.opti.subject_to(self.k[0] == self.k_0_param)
        self.opti.subject_to(self.p_box[0] == self.p_box_0_param)
        self.opti.subject_to(self.v_box[0] == self.v_box_0_param)

        def cross2d(a, b):
            return a[0]*b[1] - a[1]*b[0]

        for t in range(self.T):
            ''' Centroidal Dynamics '''
            # Center of Mass Trajectory
            self.opti.subject_to(self.p_com[t+1] == self.p_com[t] + self.dt * (self.l[t] / self.robot_mass))

            # Linear Momentum 
            gravity_force = ca.vertcat(0, -self.robot_mass * self.scene.g)
            f_tot_t = self.f_rhand[t] + self.f_lhand[t] + self.f_rfoot[t] + self.f_lfoot[t] # Hand Forces Negated
            self.opti.subject_to(self.l[t+1] == self.l[t] + self.dt * (gravity_force + f_tot_t))

            # Angular Momentum
            p_hand_t = ca.vertcat(self.p_box_prev_param[t] + self.scene.b_contact_pos[0],
                       self.scene.w_box_pos[1] + self.scene.b_contact_pos[1])
            arm_rhand = p_hand_t - self.p_com_prev_param[t]
            arm_lhand = p_hand_t - self.p_com_prev_param[t]
            arm_rfoot = self.rfoot_param - self.p_com[t]
            arm_lfoot = self.lfoot_param - self.p_com[t]

            torque_tot_t = (cross2d(arm_rhand, self.f_rhand[t])
                        + cross2d(arm_lhand, self.f_lhand[t])
                        + cross2d(arm_rfoot, self.f_rfoot[t])
                        + cross2d(arm_lfoot, self.f_lfoot[t]))

            self.opti.subject_to(self.k[t+1] == self.k[t] + self.dt * torque_tot_t)

            ''' Box Dynamics '''
            f_ground_z_t = self.scene.box_mass * self.scene.g + self.f_rhand[t][1] + self.f_lhand[t][1]
            f_net_x_t = -self.f_rhand[t][0] - self.f_lhand[t][0] + self.f_ground_x[t]

            # Position
            self.opti.subject_to(self.p_box[t+1] == self.p_box[t] + self.dt * self.v_box[t]) 

            # Velocity
            self.opti.subject_to(self.v_box[t+1] == self.v_box[t] + self.dt * (f_net_x_t / self.scene.box_mass))

            # Acceleration Cap
            self.opti.subject_to(f_net_x_t <=  self.scene.box_mass * self.a_max)
            self.opti.subject_to(f_net_x_t >= -self.scene.box_mass * self.a_max)

            # Box Friction Cone
            self.opti.subject_to(self.f_ground_x[t] <=  self.mu_ground_box * f_ground_z_t)
            self.opti.subject_to(self.f_ground_x[t] >= -self.mu_ground_box * f_ground_z_t)
            self.opti.subject_to(f_ground_z_t >= 0)

            # Dissapation Constraint
            self.opti.subject_to(self.f_ground_x[t] <= 0) 

            # Hand Friction Cones
            for f in [self.f_rhand[t], self.f_lhand[t]]:
                self.opti.subject_to(f[1] <=  self.mu_box_hand * (-f[0]))
                self.opti.subject_to(f[1] >= -self.mu_box_hand * (-f[0]))
                self.opti.subject_to(f[0] <= 0)

            # Foot Friction Cones
            for f in [self.f_rfoot[t], self.f_lfoot[t]]:
                self.opti.subject_to(f[0] <=  self.mu_ground_foot * f[1])
                self.opti.subject_to(f[0] >= -self.mu_ground_foot * f[1])
                self.opti.subject_to(f[1] >= 0)

        cost = ca.sumsqr(self.v_box[self.T] - self.V_target_param)
        for t in range(self.T):
            # Achieve Desired Velocity
            cost += 1 * ca.sumsqr(self.v_box[t] - self.V_target_param)
            # Maintain CoM Height
            cost += 10 * ca.sumsqr(self.p_com[t][1] - self.z_nom)
            # Angular Momentum Normalization
            cost += ca.sumsqr(self.k[t])
            # Force Normalization
            cost += 1e-5 * (ca.sumsqr(self.f_rhand[t]) + ca.sumsqr(self.f_lhand[t])
                        + ca.sumsqr(self.f_rfoot[t]) + ca.sumsqr(self.f_lfoot[t]))


        self.opti.minimize(cost)

        self.opti.solver('ipopt', {'print_time': False},
                          {'acceptable_tol': 1e-3, 'acceptable_iter': 15,
                           'max_iter': 2000, 'print_level': 0, 'sb': 'yes'})


    def solve_mpc(self, q0, dq0, V_target):
        q_robot_0, dq_robot_0, q_box_0, dq_box_0 = self.scene.get_joint_states(q0, dq0)

        rhand_pos, lhand_pos, rfoot_pos, lfoot_pos = self.scene.get_contact_locations(q_robot_0)

        p_com_0, l_0, k_0 = self.scene.get_centroidal_state(q_robot_0, dq_robot_0)
        p_box_0, v_box_0 = self.scene.get_box_state(q_box_0, dq_box_0)

        self.opti.set_value(self.p_com_0_param, p_com_0)
        self.opti.set_value(self.l_0_param, l_0)
        self.opti.set_value(self.k_0_param, k_0)
        self.opti.set_value(self.p_box_0_param, p_box_0)
        self.opti.set_value(self.v_box_0_param, v_box_0)
        self.opti.set_value(self.V_target_param, V_target)
        self.opti.set_value(self.rfoot_param, rfoot_pos)
        self.opti.set_value(self.lfoot_param, lfoot_pos)

        if self._prev_sol is not None:
            for t in range(self.T):
                self.opti.set_value(self.p_com_prev_param[t], self._prev_sol.value(self.p_com[t+1]))
                self.opti.set_value(self.p_box_prev_param[t], self._prev_sol.value(self.p_box[t+1]))

            # warm start: shift previous trajectory forward by 1 node,
            # hold the last value for the newly-exposed tail node
            # states (length T+1): shift forward by one node, hold the last
            for var, name in [(self.p_com, 'p_com'), (self.l, 'l'), (self.k, 'k'),
                            (self.p_box, 'p_box'), (self.v_box, 'v_box')]:
                for t in range(self.T):
                    self.opti.set_initial(var[t], self._prev_sol.value(var[t+1]))
                self.opti.set_initial(var[self.T], self._prev_sol.value(var[self.T]))

            # forces (length T): same shift, but the last valid index is T-1, not T —
            # hold prev's final force since there's no T-th one to shift in
            for var in [self.f_rhand, self.f_lhand, self.f_rfoot, self.f_lfoot, self.f_ground_x]:
                for t in range(self.T - 1):
                    self.opti.set_initial(var[t], self._prev_sol.value(var[t+1]))
                self.opti.set_initial(var[self.T - 1], self._prev_sol.value(var[self.T - 1]))

        else:
            # first solve, no previous trajectory yet — best available guess is the current state
            for t in range(self.T):
                self.opti.set_value(self.p_box_prev_param[t], p_box_0 + (t + 1) * self.dt * V_target)
                self.opti.set_value(self.p_com_prev_param[t], p_com_0)  # CoM x/z assumed to move much less over 0.5s

        sol = self.opti.solve()
        self._prev_sol = sol

        self.plot_solution(sol, self.p_box, self.v_box, self.f_rhand, self.f_lhand, self.f_rfoot, self.f_lfoot, self.f_ground_x, self.p_com, V_target)


        return self.get_full_trajectory(sol)

        # # Centroidal Trajectory
        # p_com_1 = CDRef()
        # p_com_1.x_ref = sol.value(self.p_com[1])
        # p_com_1.xd_ref = sol.value(self.l[1]) / self.robot_mass

        # f_tot_1 = sol.value(self.f_rhand[0] + self.f_lhand[0] + self.f_rfoot[0] + self.f_lfoot[0])
        # gravity_force = np.array([0, -self.robot_mass * self.scene.g])

        # p_com_1.xdd_ref = (gravity_force + f_tot_1) / self.robot_mass

        # p_com_0_ref = CDRef(x_ref=p_com_0, xd_ref=l_0/self.robot_mass, xdd_ref=p_com_1.xdd_ref)

        # # Box Trajectory
        # box_ref = CDRef()
        # box_ref.x_ref = sol.value(self.p_box[1])
        # box_ref.xd_ref = sol.value(self.v_box[1])
        # box_ref.xdd_ref = self.box_dynamics(sol.value(self.f_rhand[0][0]), sol.value(self.f_lhand[0][0]), sol.value(self.f_ground_x[0]))

        # box_0_ref   = CDRef(x_ref=p_box_0, xd_ref=v_box_0, xdd_ref=box_ref.xdd_ref)

        # # Forces
        # f_ref = FRef()
        # f_ref.f_rhand = sol.value(self.f_rhand[0])
        # f_ref.f_lhand = sol.value(self.f_lhand[0])
        # f_ref.f_rfoot = sol.value(self.f_rfoot[0])
        # f_ref.f_lfoot = sol.value(self.f_lfoot[0])


        # print("FRICTION FORCE:",sol.value(self.f_ground_x[0]))
        # # in solve_mpc, after solve
        # print("k trajectory:", [float(sol.value(self.k[t])) for t in range(0, self.T, 10)])
        # print("l trajectory:", [sol.value(self.l[t])/self.robot_mass for t in range(0, self.T, 10)])
        # return p_com_0_ref, p_com_1, box_0_ref, box_ref, f_ref

    def get_full_trajectory(self, sol):
        """Extracts the entire solved horizon as per-node CDRef/FRef sequences,
        for open-loop validation (track one full MPC solve with no replanning).
        Uses the same feedforward derivation as solve_mpc's single-node extraction,
        just repeated across all T nodes instead of only node 0/1.
        """
        com_refs, box_refs, f_refs = [], [], []
        gravity_force = np.array([0, -self.robot_mass * self.scene.g])

        for t in range(self.T):
            f_rhand_t = sol.value(self.f_rhand[t])
            f_lhand_t = sol.value(self.f_lhand[t])
            f_rfoot_t = sol.value(self.f_rfoot[t])
            f_lfoot_t = sol.value(self.f_lfoot[t])
            f_ground_x_t = sol.value(self.f_ground_x[t])

            f_tot_t = f_rhand_t + f_lhand_t + f_rfoot_t + f_lfoot_t
            com_xdd_t = (gravity_force + f_tot_t) / self.robot_mass

            com_refs.append(CDRef(
                x_ref=sol.value(self.p_com[t]),
                xd_ref=sol.value(self.l[t]) / self.robot_mass,
                xdd_ref=com_xdd_t,
            ))

            box_refs.append(CDRef(
                x_ref=sol.value(self.p_box[t]),
                xd_ref=sol.value(self.v_box[t]),
                xdd_ref=self.box_dynamics(f_rhand_t[0], f_lhand_t[0], f_ground_x_t),
            ))

            f_refs.append(FRef(
                f_rhand=f_rhand_t,
                f_lhand=f_lhand_t,
                f_rfoot=f_rfoot_t,
                f_lfoot=f_lfoot_t,
            ))

        # append the terminal node (T) for com_refs/box_refs so interpolation has
        # a valid endpoint for the final window (t = T-1 -> T); no matching force
        # exists at T (forces are per-interval, length T), so f_refs stays length T
        # and the last interval reuses f_refs[T-1] as its held value.
        com_refs.append(CDRef(
            x_ref=sol.value(self.p_com[self.T]),
            xd_ref=sol.value(self.l[self.T]) / self.robot_mass,
            xdd_ref=com_refs[-1].xdd_ref,  # hold last known feedforward — no forces beyond node T-1
        ))
        box_refs.append(CDRef(
            x_ref=sol.value(self.p_box[self.T]),
            xd_ref=sol.value(self.v_box[self.T]),
            xdd_ref=box_refs[-1].xdd_ref,
        ))

        return com_refs, box_refs, f_refs

    def plot_solution(self, sol, p_box, v_box, f_rhand, f_lhand, f_rfoot, f_lfoot, f_ground_x, r_com, V_target):
        t_axis = np.arange(self.T+1) * self.dt
        t_axis_f = np.arange(self.T) * self.dt  # forces are per-interval, one shorter

        box_x  = [float(sol.value(p_box[t])) for t in range(self.T+1)]
        box_vx = [float(sol.value(v_box[t])) for t in range(self.T+1)]
        com_x  = [float(sol.value(r_com[t])[0]) for t in range(self.T+1)]
        com_z  = [float(sol.value(r_com[t])[1]) for t in range(self.T+1)]

        frh = np.array([sol.value(f_rhand[t]) for t in range(self.T)])
        flh = np.array([sol.value(f_lhand[t]) for t in range(self.T)])
        frf = np.array([sol.value(f_rfoot[t]) for t in range(self.T)])
        flf = np.array([sol.value(f_lfoot[t]) for t in range(self.T)])
        fgx = [float(sol.value(f_ground_x[t])) for t in range(self.T)]
        fgz = [self.scene.box_mass * self.scene.g - frh[t, 1] - flh[t, 1] for t in range(self.T)]

        fig, axes = plt.subplots(3, 2, figsize=(11, 9))

        axes[0,0].plot(t_axis, box_x, marker='o', ms=3)
        axes[0,0].set_title("Box position (x)")
        axes[0,0].set_ylabel("m")

        axes[0,1].plot(t_axis, box_vx, marker='o', ms=3, label="v_box")
        axes[0,1].axhline(V_target, color='r', linestyle='--', label="target")
        axes[0,1].set_title("Box velocity (vx)")
        axes[0,1].legend()

        axes[1,0].plot(t_axis, com_x, label="x")
        axes[1,0].plot(t_axis, com_z, label="z")
        axes[1,0].set_title("Robot CoM position")
        axes[1,0].legend()

        axes[1,1].plot(t_axis_f, fgx, label="Fnx")
        axes[1,1].plot(t_axis_f, fgz, label="Fnz")
        axes[1,1].set_title("Ground normal force (Fnx, Fnz)")
        axes[1,1].legend()

        axes[2,0].plot(t_axis_f, frh[:,0], label="rhand fx")
        axes[2,0].plot(t_axis_f, flh[:,0], label="lhand fx")
        axes[2,0].plot(t_axis_f, frh[:,1], label="rhand fz")
        axes[2,0].plot(t_axis_f, flh[:,1], label="lhand fz")
        axes[2,0].set_title("Hand forces (fx + fz)")
        axes[2,0].legend()

        axes[2,1].plot(t_axis_f, frf[:,0], label="rfoot fx")
        axes[2,1].plot(t_axis_f, flf[:,0], label="lfoot fx")
        axes[2,1].plot(t_axis_f, frf[:,1], label="rfoot fz")
        axes[2,1].plot(t_axis_f, flf[:,1], label="lfoot fz")
        axes[2,1].set_title("Foot forces (fx + fz)")
        axes[2,1].legend()

        for ax in axes.flat:
            ax.set_xlabel("time (s)")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/mpc results.png")