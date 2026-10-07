import pinocchio as pin
import numpy as np
from qpsolvers import solve_qp

from kinematic_utils import SceneModel

class WholeBodyController:

    def __init__(self, scene:SceneModel, hand_name="hand_site"):
        self.scene = scene

        self.model = scene.robot_model
        self.data = scene.robot_data

        self.hand_name = hand_name
        self.hand_id = scene.robot_model.getFrameId(self.hand_name)

        # Viscous Damping Coefficient
        self.b = 5

        # PD Controller gains
        self.Kp = 100.0 
        self.Kd = 20.0

    def compute_jacobians(self, q, dq):
        # Robot Configuration
        q = q[:2]
        dq = dq[:2]

        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)

        # Hand Jacobian
        Jc = pin.getFrameJacobian(self.model, self.data, self.hand_id, pin.LOCAL_WORLD_ALIGNED)
        Jc_dot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.hand_id, pin.LOCAL_WORLD_ALIGNED)

        # Planar Selection (x and z rows ONLY)
        Jc = Jc[[0, 2], :]
        Jc_dot = Jc_dot[[0, 2], :]
       
        return  Jc, Jc_dot
    
    def pd_control(self, q, dq, Jc):
        x_box = q[2]
        x_dot_box = dq[2]

        des_contact_pos_x = x_box - self.scene.box_half_extents[0]
        des_contact_pos_z = self.scene.box_half_extents[1]
        des_contact_pos = [des_contact_pos_x, des_contact_pos_z]

        hand_pos = self.scene.get_contact_position(q[:2], site_id=self.hand_id)

        hand_pos_error = hand_pos - des_contact_pos

        hand_vel = Jc @ dq[:2]
        box_vel = [x_dot_box, 0]

        hand_vel_error = hand_vel - box_vel

        xdd_hand_des =  -self.Kp * hand_pos_error - self.Kd * hand_vel_error
        
        return xdd_hand_des

    def build_cost(self, lamda_des, w_fx=1e2, w_fz=1e1, w_r=1e-6):
        w = np.full(7, w_r)
        w[5] += w_fx
        w[6] += w_fz
        P = np.diag(w)

        q = np.zeros(7)
        q[5:7] = -np.array([w_fx, w_fz]) * lamda_des
        return P, q
    
    def compute_control(self, q, dq, lambda_des):
        Jc, Jc_dot = self.compute_jacobians(q, dq)
        xdd_hand_des = self.pd_control(q, dq, Jc)

        # Formulate and solve QP to get q̈ and contact forces λ
        pin.crba(self.model, self.data, q[:2])               
        M = np.triu(self.data.M) + np.triu(self.data.M, 1).T  
        h = pin.nonLinearEffects(self.model, self.data, q[:2], dq[:2])

        print(M, h)

        ''' Build Equality Constraints'''
        A_eq = np.zeros((5, 7))
        b_eq = np.zeros(5)

        iq, ixb, itau, ilam = slice(0,2), 2, slice(3,5), slice(5,7)

        # Robot Dynamics
        A_eq[:2, iq] = M
        A_eq[:2, itau] = -np.eye(2)
        A_eq[:2, ilam] = -Jc.T

        # Box Dynamics
        A_eq[2, ixb] = self.scene.box_mass
        A_eq[2, ilam] = np.array([1, 0]) # only the horizontal force is effecting box motion


        # Hand Tracking
        A_eq[3:, iq] = Jc
        A_eq[3:, ixb] = np.array([-1, 0])

        # RHS
        b_eq[:2] = -h
        b_eq[2] = -self.b * dq[2]
        b_eq[3:] = xdd_hand_des - Jc_dot @ dq[:2]

        ''' Build Inequality Constraints '''
        G_ineq = np.zeros((3, 7))
        h_ineq = np.zeros(3)

        # Unilateral Constraint (lambda_x <= 0)
        G_ineq[0, 5] = 1

        # Friction Cone Constraints
        G_ineq[1, 5] = self.scene.mu_box_hand
        G_ineq[1, 6] = 1
        G_ineq[2, 5] = self.scene.mu_box_hand
        G_ineq[2, 6] = -1

        # Torque Limits
        pass
        # Joint Limits
        pass

        P, q = self.build_cost(lambda_des)

        x = solve_qp(P, q, A=A_eq, b=b_eq, G=G_ineq, h=h_ineq, solver="quadprog")

        tau = x[itau]
        lamb = x[ilam]

        print("CMD TAU: ", tau)
        print("CMD LAMBDA: ", lamb)
        
        return tau

if __name__ == "__main__":
    wbc = WholeBodyController()

    for i, name in enumerate(wbc.model.names):
        print(f"Pin joint {i}: {name}")