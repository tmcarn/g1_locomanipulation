import mujoco
print(mujoco.__version__)
import mujoco.viewer
import numpy as np
import time
import os
from PIL import Image

from constants import *
from vizualization import DebugVisualizer, VideoRecorder

from kinematic_utils import SceneModel
from key_frame_generator import Solver
from whole_body_control import WholeBodyController


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

        # self.nq_robot = 11

        # self.key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "hands_on_box")
        # mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)

        mujoco.mj_step(self.mj_model, self.mj_data)
    
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.renderer = mujoco.Renderer(self.mj_model, height=1080, width=1920)
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "side_track")
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
        # mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.key_id)
        if q is not None:
            self.mj_data.qpos[:] = q

        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def get_joint_state(self):
        q = self.mj_data.qpos.copy()
        dq = self.mj_data.qvel.copy()

        return q, dq

    def step(self, tau=None):
        # tau shape (10,) — actuated joints only
        if tau is not None:
            self.mj_data.ctrl[:] = tau

        mujoco.mj_step(self.mj_model, self.mj_data)
        return self.get_joint_state()

    def render(self):
        '''Sync the viewer to the current sim state.'''
        self.viewer.sync()

    def capture_frame(self, camera="side_track", save_path="frame.png"):
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

    fps = 30
    record_every = int(round(1 / (fps * env.dt)))   # e.g. 33 steps at dt = 1 ms
    rec = VideoRecorder("videos/push.mp4", fps=fps)

    step = 0
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
        time.sleep(env.dt*100)

        

    env.close()


if __name__ == "__main__":
    F_des = np.array([-10, 0]) # 100N force
    scene = SceneModel()

    env = Env()

    q0 = [np.pi/6, np.pi/3, 0]

    q, dq = env.reset(q0)

    wbc = WholeBodyController(scene)

    fps = 30
    slowdown = 10

    T_record = 1.0  # 1s of simulation time
    record_every = int(round(1 / (slowdown * fps * env.dt)))   # e.g. 33 steps at dt = 1 ms
    rec = VideoRecorder("videos/push.mp4", fps=fps)

    step = 0
    while env.viewer.is_running():
        q, dq = env.get_joint_state()
        tau = wbc.compute_control(q, dq, F_des)
        env.step(tau)
        env.render()
        time.sleep(env.dt*1)
        if rec is not None:
            if env.mj_data.time >= T_record:
                rec.close()
                rec = None                     # stop recording, keep simulating
                print("Recording saved")
            elif step % record_every == 0:
                rec.add(env.capture_frame())
        step += 1

    if rec is not None:
        rec.close()                            # in case the viewer was closed early
    env.close()