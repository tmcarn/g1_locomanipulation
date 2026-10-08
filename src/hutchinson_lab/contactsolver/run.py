from env import Env
from CDMPC import CentroidalSolver
# from WBC import WBC
from whole_body_control import WholeBodyController
from kinematic_utils import SceneModel, get_interpolated_ref
from logger import WBCLogger

from vizualization import DebugVisualizer
from constants import *

import numpy as np
import time

V_cmd = 0.3  # m/s (desired box velocity)


def run():
    env = Env()
    q, dq = env.reset()

    scene = SceneModel()
    cdmpc = CentroidalSolver(scene)
    wbc = WholeBodyController(scene)

    ticks_per_node = int(MPC_DT / SIM_DT)
    total_ticks = MPC_T * ticks_per_node
    logger = WBCLogger(n_ticks=300)

    viz = DebugVisualizer(env.viewer)

    # --- single MPC solve, no replanning for the rest of this run ---
    q0, dq0 = env.get_joint_state()
    cdmpc.solve_mpc(q0, dq0, V_cmd)
    com_refs, box_refs, f_refs = cdmpc.get_full_trajectory(cdmpc._prev_sol)

    tot_ticks = 0

    while env.viewer.is_running() and tot_ticks < total_ticks:
        q, dq = env.get_joint_state()

        node_idx = min(tot_ticks // ticks_per_node, MPC_T - 1)
        frac = (tot_ticks % ticks_per_node) / ticks_per_node

        p_com_ref_now = get_interpolated_ref(com_refs[node_idx], com_refs[node_idx + 1], frac)
        box_ref_now = get_interpolated_ref(box_refs[node_idx], box_refs[node_idx + 1], frac)
        f_ref_now = f_refs[node_idx]  # forces are ZOH within a node, no interpolation

        tau = wbc.compute_control(q, dq, p_com_ref_now, box_ref_now, f_ref_now)

        env.step(tau)

        env.render()
        time.sleep(env.dt * 1)

        if logger.full():
            logger.plot()  # writes wbc_log.png over the full open-loop trajectory
            break

    

    env.close()


if __name__ == "__main__":
    run()