import os

SCENE_PATH = os.path.join(os.path.dirname(__file__), "xmls/scene.xml")
# ROBOT_PATH = os.path.join(os.path.dirname(__file__), "xmls/humanoid_2D.xml")
ROBOT_PATH = os.path.join(os.path.dirname(__file__), "xmls/two_link_arm.xml")
BOX_PATH = os.path.join(os.path.dirname(__file__), "xmls/box.xml")


SITE_NAMES = ['right_hand', 'left_hand', 'right_foot', 'left_foot']

SIM_DT = 0.001 # 1kHz
MPC_DT = 0.010 # 100Hz 
MPC_T = 50 # 0.5 second horizon

DECIMATION = 10 # MPC_DT / SIM_DT # 10 WBC steps per MPC replanning

