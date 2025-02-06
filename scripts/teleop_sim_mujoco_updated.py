#TODO: 1. Add data recording, FPS
#TODO: 2. Add data replay? 
#TODO: 3. Add task env
import faulthandler
faulthandler.enable()

# from isaacgym import gymtorch, gymapi, gymutil
import mujoco as mj
import mujoco.viewer as mjv

import numpy as np

from opentv import TeleVision

from sim_mujoco import MujocoSim

# import globals

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
import datetime

import h5py

import time
import os
import json
import atexit


TABLE_SIZE = [0.33, 0.33, 0.1]
TABLE_POS = [-0.3, 0, 1.2]
ROBOT_POS = [-0.8, 0, 1.1]

toggle_value = False
artificial_gest_check = False
def on_press(key):
    try:
        # WARNING: Changed from 't' to 'r'
        # Toggle the boolean value when 'r' is pressed
        if key.char == 'r':
            print("Recording toggled")
            global toggle_value
            toggle_value = not toggle_value
            # global artificial_gest_check
            # artificial_gest_check = not artificial_gest_check

        # Exit the program when 'q' is pressed
        elif key.char == 'q':
            print("Exiting...")
            return False  # Stop listener to exit the program
    except AttributeError:
        # Handle other special keys here if needed
        pass

class GestureCheck:
    def __init__(self, path, freq, threshold=0.15) -> None:
        # add gesture check
        self.threshold = threshold
        self.freq = freq
        with open(path, 'rb') as f:
            self.gesture = np.load(f)
        self.init_buf()

        self.gesture_list = []
    
    def init_buf(self):
        self.gesture_window = np.zeros((self.freq*3, 1))
        self.gesture_cnt = 0

    def check(self, gesture):
        # post action gesture check
        verified = np.linalg.norm(gesture - self.gesture) <= self.threshold
        self.gesture_window = np.concatenate((np.array(verified).reshape((1,1)), self.gesture_window[:-1,:]))
        if self.gesture_window.sum() >= self.gesture_window.shape[0] * (9/10):
            self.init_buf()
            return True
        return False

    def add_gesture_to_list(self, new_gesture):
        # Append the gesture to the list
        self.gesture_list.append(new_gesture)
        # print("Gesture added to list.")
    
    def calculate_average_gesture(self):
        if not self.gesture_list:
            print("No gestures in list to calculate average.")
            return None
        # Calculate the average of all gestures in the list
        average_gesture = np.mean(self.gesture_list[100:-100], axis=0)
        print("Average gesture calculated.")
        return average_gesture
    
    def save_average_gesture(self, save_path="./data/ref_gestures/average_gesture.npy"):
        average_gesture = self.calculate_average_gesture()
        if average_gesture is not None:
            with open(save_path, 'wb') as f:
                np.save(f, average_gesture)
            print(f"Average gesture saved to {save_path}")

class Dataset:
    def __init__(self, path,) -> None:
        # add gesture check
        self.path = path
        self.data_dict = {'/obs/timestamp': [],
                          '/obs/qpos': [],  # 14 +12
                          '/obs/qvel': [],  # 14
                          '/action/joint_pos': [],
                          '/action/cmd/head_mat': [],
                          '/action/cmd/rel_left_wrist_mat': [],
                          '/action/cmd/rel_right_wrist_mat': [],
                          '/action/cmd/rel_left_hand_keypoints': [],
                          '/action/cmd/rel_right_hand_keypoints': []}   # 4*4 + 4*4 + 4*4 + 25*3 + 25*3

    def insert(self, 
               timestamp, 
               qpos, 
               qvel, 
               actions, 
               head_mat, 
               rel_left_wrist_mat, 
               rel_right_wrist_mat, 
               rel_left_hand_keypoints, 
               rel_right_hand_keypoints):
        self.data_dict['/obs/timestamp'].append(timestamp)
        self.data_dict['/obs/qpos'].append(qpos)
        self.data_dict['/obs/qvel'].append(qvel)
        self.data_dict['/action/joint_pos'].append(actions)
        self.data_dict['/action/cmd/head_mat'].append(head_mat)
        self.data_dict['/action/cmd/rel_left_wrist_mat'].append(rel_left_wrist_mat)
        self.data_dict['/action/cmd/rel_right_wrist_mat'].append(rel_right_wrist_mat)
        self.data_dict['/action/cmd/rel_left_hand_keypoints'].append(rel_left_hand_keypoints)
        self.data_dict['/action/cmd/rel_right_hand_keypoints'].append(rel_right_hand_keypoints)
    
    def save_to_hdf5(self, description, embodiment):
        print(f"Saving data to {self.path}.hdf5")
        with h5py.File(self.path + ".hdf5", 'w') as file:
            for key, value in self.data_dict.items():
                file.create_dataset(key, data=value)
        with open(self.path + "_meta.json", 'w') as f:
            json.dump({"description": description, 
                       "embodiment": embodiment,
                       "num_prop_frames": len(self.data_dict['/obs/timestamp']),
                       "time_start": self.data_dict['/obs/timestamp'][0],
                       "time_end": self.data_dict['/obs/timestamp'][-1],
                       "total_time": self.data_dict['/obs/timestamp'][-1] - self.data_dict['/obs/timestamp'][0]}, 
                       f)

class VuerTeleop:
    def __init__(
            self, 
            freq,
            config_files, 
            print_freq=False, 
            task_id=0, 
            tasktype=0,
            path = "../../data/recordings",
            is_viewer=False,
            stream_mode="image",
            scene_description="",
            is_debug=False,
            ):
        
        self.resolution = (720, 1280)
        # self.resolution = (1080, 1920)
        self.crop_size_w = 0 # (resolution[1] - resolution[0]) // 2
        self.crop_size_h = 0
        self.resolution_cropped = (int((self.resolution[0]-self.crop_size_h)/1.5), (self.resolution[1]-2*self.crop_size_w)//2)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3) # 540, 720 * 2, 3
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(name="sim_image", create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)

        self.freq = freq
        self.dt = 1.0 / self.freq
        self.timestep = 0
        self.warmup_steps = 300

        self.is_recording = False
        # image_queue = Queue()
        # self.toggle_task = Event()

        self.if_record = not args['no_record']
        self.manager = Manager()
        self.control_dict = self.manager.dict()
        self.control_dict['is_recording'] = False
        self.control_dict['path'] = ""
        self.toggle_recording = Event()
        self.toggle_streaming = Event()

        self.scene_description = scene_description
        
        cfgs=[]
        for config_file_name in config_files:
            config_file_path = Path(__file__).resolve().parent.parent / "configs" / config_file_name
            with Path(config_file_path).open('r') as f:
                cfg = yaml.safe_load(f)['robot_cfg']
            cfgs.append(cfg)
            
        if cfgs[task_id]['name'] == 'h1_inspire':
            self.valid_q_indices = [*range(13, 20), 20, 22, 24, 26, 28, 29, *range(32, 39), 39, 41, 43, 45, 47, 48]
        elif cfgs[task_id]['name'] == 'gr1_inspire':
            self.valid_q_indices = [*range(18, 25), 25, 27, 29, 31, 33, 34, *range(37, 44), 44, 46, 48, 50, 52, 54]
        else:
            raise NotImplementedError
            
        self.sim = MujocoSim(config_files, 
                            print_freq=print_freq, 
                            task_id=task_id, 
                            tasktype=tasktype,
                            path=path, 
                            shm_name = self.shm.name, 
                            # image_queue=image_queue, 
                            img_shape=self.img_shape, 
                            is_viewer=is_viewer,
                            control_dict = self.control_dict,
                            toggle_recording = self.toggle_recording,
                            cfgs=cfgs)
        
        self.tv = TeleVision(self.resolution_cropped, 
                             self.shm.name, 
                             rtc_offer_addr="https://127.0.0.1:8080/offer",
                             stream_mode=stream_mode, 
                             cert_file="./cert.pem", 
                             key_file="./key.pem", 
                             ngrok=args['http'], 
                             ctrl_config=config_files[task_id],
                             asset_dir="../assets",
                             config_dir="../configs")

        self.episode = 0
        self.path = path

        self.start_check = GestureCheck('../data/ref_gestures/start_gesture.npy', self.freq)
        self.record_check = GestureCheck('../data/ref_gestures/record_gesture.npy', self.freq)

        # self.last_description = ""

        # Register cleanup function
        atexit.register(self.cleanup_shared_memory)

    def cleanup_shared_memory(self):
        print("Cleaning up shared memory...")
        self.shm.close()
        self.shm.unlink()

    def step(self, viewer):
        processed_mat, cmds, head_rot_mat = self.tv.step()
        actions = cmds[self.valid_q_indices]  # 8 + 6 + 12
        #! pose and key points
        self.arm_motor_indices = np.array(self.sim.cfgs[self.sim.cur_env].get('arm_motor_indices', [])) # 26

        # get qpos and qvel from self.sim
        pos = np.array(self.sim.data.qpos[self.arm_motor_indices])
        vel = np.array(self.sim.data.qvel[self.arm_motor_indices])

        if (self.control_dict['is_recording'] and self.if_record):
            action_time = time.time()
            self.dataset.insert(action_time,
                                pos,
                                vel,
                                np.concatenate((actions, self.tv.controller.ypr[:2])),
                                *processed_mat)    
            print(f"Inserted data at {action_time}")
        
        self.sim.step(cmds, head_rot_mat, viewer)
        # time.sleep(1)

        self.post_step_callback()

    def post_step_callback(self):

        if self.timestep < self.warmup_steps:
            self.timestep += 1 

        new_gesture = self.tv.processor.get_hand_gesture(self.tv)
        # if self.start_check.check(new_gesture):
            # print("Exit gesture detected, exit")
            # exit()

        global toggle_value
        #! here we start and end the dataset recording, seems do not need to change at all
        
        # assert arg.debug
        # if (self.record_check.check(new_gesture) and self.if_record) or toggle_value:
        if (self.record_check.check(new_gesture) and self.if_record):
            print("Record gesture detected, recording toggled")
            # from IPython import embed; embed()
            self.toggle_recording.set()
            if self.control_dict['is_recording']: # stop recording
                self.episode += 1
                # 
                # description = input(f"Enter episode description just finished, Press Enter to use last description {self.last_description}:\n")
                # description = description if description != "" else self.last_description
                self.dataset.save_to_hdf5(self.scene_description, args["embodiment"])
                # self.last_description = description
            os.makedirs(self.path, exist_ok=True)  # start recording
            self.control_dict["path"] = os.path.join(self.path, f"episode_{self.episode}")#f"../data/{time.time()}.svo"
            self.dataset = Dataset(self.control_dict["path"])
            toggle_value = not toggle_value


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", type=str, required=True, choices=["gr1_inspire_sim", "h1_2_inspire_sim"])
    parser.add_argument("--http", action="store_true", default=False)
    parser.add_argument("--taskid", type=int, default=0, help="Task ID")
    parser.add_argument("--tasktype", type=str, required=True, help="Choose from given task types: microwave, pickplace, wiping, pour, tap, pepsi, picking, stacking")
    parser.add_argument("--expid", type=str, help="experiment id", required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--stream_mode", choices=["image", "webrtc"], default="image")
    parser.add_argument("--realworld", action="store_true", default=False)
    parser.add_argument('--no_record', default=False, action='store_true')
    parser.add_argument("--scene_description", type=str, help="scene description for the current demos", required=True)
    # parser.add_argument("--scene_config_path", type=str, help="path for the json config for scene profile ", required=True)

    FREQ = 60
    STEP_TIME = 1/FREQ

    # args = parser.parse_args()
    args = vars(parser.parse_args())
    cur_env = args["taskid"]
    tasktype = args["tasktype"]
    task_type_list = ["microwave", "pickplace", "wiping", "pour", "tap", "pepsi", "picking", "stacking"]
    if tasktype.lower() not in task_type_list:
        raise ValueError("Invalid task type. Please choose from: microwave, pickplace, wiping, pour, tap, pepsi, picking, stacking")

    time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base_path = str(Path(__file__).resolve().parent.parent / "data" / "recordings")
    folder_name = os.path.join(base_path, args["expid"] + "-" + time_str)
    
    task_config = yaml.safe_load(open(str(Path(__file__).resolve().parent.parent / "configs" / "all_tasks.yml"), "r"))["tasks"]
    print(task_config)
    config_files = [task_config[key]["file"] for key in task_config.keys()]

    print("[Sim] Start task: ", cur_env)
    teleoperator = VuerTeleop(FREQ,
                              config_files, 
                              print_freq=False, 
                              task_id=cur_env, 
                              tasktype=tasktype,
                              path=folder_name, 
                              is_viewer=args["viewer"],
                              stream_mode=args["stream_mode"],
                              scene_description=args["scene_description"])
    # stablizer = OpticalFlowStablizer()

    if args["debug"]:
        from pynput import keyboard
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    try:
        with mj.viewer.launch_passive(teleoperator.sim.model, teleoperator.sim.data) as viewer:
            # Setup viewer for mujoco
            teleoperator.sim.setup_viewer(viewer)
            while viewer.is_running:
  
                #! here we change the color to show the toggle value
                # table_color = gymapi.Vec3(0, 1, 0) if globals.toggle_value else gymapi.Vec3(1, 0, 0)  
                teleoperator.sim.color = (0, 255, 0) if teleoperator.control_dict['is_recording'] else (255, 0, 0)
                # teleoperator.sim.gesture_color = (0, 0, 255) if globals.testing else (0, 0, 0)
                teleoperator.step(viewer)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
    except Exception as e:
        print(e)
        print("Exiting...")
        exit(0)
    