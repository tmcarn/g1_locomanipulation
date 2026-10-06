import mujoco
print(mujoco.__version__)
import mujoco.viewer
import numpy as np
import time
import os
from PIL import Image

from constants import *
from debug_viz import DebugVisualizer

from kinematic_utils import SceneModel
from key_frame_generator import Solver


class Env:
    '''
    MuJoCo environment wrapper for the biped model. This class handles:
        - MuJoCo model and data management
        - State retrieval (joint states)
        - Stepping the simulation with given control inputs
        - Rendering (using MuJoCo's built-in viewer)
    '''

    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.dt = SIM_DT  # control timestep

        print([a for a in dir(self.mj_model) if 'cam' in a.lower()])

        self.nq_robot = 11

        self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "hands_on_box")
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)

        mujoco.mj_step(self.mj_model, self.mj_data)
    
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.renderer = mujoco.Renderer(self.mj_model, height=1080, width=1920)
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "poster_view")
        self.mj_model.cam_projection[cam_id] = 1
        self.mj_model.cam_fovy[cam_id] = 3.0          # orthographic extent, same meaning as fovy="2"

        self.site_ids = {
            "right_hand": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES[0]),
            "left_hand": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES[1]),
            "right_foot": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES[2]),
            "left_foot":  mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES[3]),
        }

        mujoco.mj_forward(self.mj_model, self.mj_data)   # or mj_step — populates xpos/site_xpos from qpos


    def reset(self, q=None):
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)
        if q is not None:
            self.mj_data.qpos[:] = q

        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def get_joint_state(self):
        q = self.mj_data.qpos.copy()
        dq = self.mj_data.qvel.copy()

        return q, dq

    def step(self, tau):
        # tau shape (10,) — actuated joints only
        self.mj_data.ctrl[:] = tau
        mujoco.mj_step(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def render(self):
        '''Sync the viewer to the current sim state.'''
        self.viewer.sync()

    def capture_frame(self, camera="planar_view", save_path="frame.png"):
        self.renderer.update_scene(self.mj_data, camera=camera)
        pixels = self.renderer.render()
        Image.fromarray(pixels).save(os.path.join("images", save_path))
        return pixels

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def run(q=None):
    env = Env()
    env.reset(q)

    viz = DebugVisualizer(env.viewer)

    while env.viewer.is_running():
        q, dq = env.get_joint_state()
        tau = np.zeros(8)

        # env.step(tau)

        # viz.draw(
        #     env.mj_data,
        #     env.site_ids,
        #     draw_world=False
        # )
        
        env.render()
        time.sleep(env.dt*10)

        

    env.close()


if __name__ == "__main__":
    scene = SceneModel()
    print("mass", scene.robot_mass)

    env = Env()
        
    solver = Solver(scene)  # pass a SceneModel instance to scale standoff with box size
    q0 = solver.solve_ik()
    q = np.zeros(12)
    q[:-1] = q0

    env.reset(q)
    env.capture_frame(save_path="planar_view.png")
    env.close()