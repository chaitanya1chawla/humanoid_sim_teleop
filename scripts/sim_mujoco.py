# TODO: Change indexing for cube_handles and other handles if multiple environments
# TODO: Edit all step_check according to information required in final dataset. 

import os
import cv2
import json
import time
import yaml
import glfw
import numpy as np
from pathlib import Path

from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value

import torch
import mujoco as mj

# import globals
from scipy.spatial.transform import Rotation as R


ROBOT_POS = [0, 0, 1.6]
ROBOT_POS_OFFSET = [0, 0, 0]  # For table-top manipulation
ROBOT_POS = [ROBOT_POS[i] + ROBOT_POS_OFFSET[i] for i in range(3)]

# Table limits => x: [-0.465, -0.135], y: [-0.165, 0.165], z: [1.2]
# TABLE_SIZE = [0.33, 0.33, 0.1]
# TABLE_POS = [-0.3, 0, 1.35]
# TABLE_POS = [TABLE_POS[i] + ROBOT_POS_OFFSET[i] for i in range(3)]
# TABLE_POS_LIM = {'x': [TABLE_POS[0] - TABLE_SIZE[0]/2, TABLE_POS[0] + TABLE_SIZE[0]/2], 
#                  'y': [TABLE_POS[1] - TABLE_SIZE[1]/2, TABLE_POS[1] + TABLE_SIZE[1]/2], 
#                  'z': [1.2]}

PLATFORM_SIZE = np.array([0.15, 0.15, 0.05])

# NutInsertion
POLE_SIZE = np.array([0.015, 0.015, 0.15])
NUT_SIZE = np.array([0.1, 0.1, 0.02]) # Bbox rough estimate. Look at the MJCF (.xml) file for the actual dimensions

#  Uncomment and update these constants for table and cube
TABLE_SIZE = [0.33, 0.33, 0.1]
TABLE_POS = [-0.3, 0, 1.35]
TABLE_POS = [TABLE_POS[i] + ROBOT_POS_OFFSET[i] for i in range(3)]
TABLE_POS_LIM = {'x': [TABLE_POS[0] - TABLE_SIZE[0]/2, TABLE_POS[0] + TABLE_SIZE[0]/2], 
                 'y': [TABLE_POS[1] - TABLE_SIZE[1]/2, TABLE_POS[1] + TABLE_SIZE[1]/2], 
                 'z': [1.2]}

CUBE_SIZE = [0.05, 0.05, 0.05]  # Add this constant for the cube size


class MujocoSim:
    def __init__(
                self,
                config_file_names,
                print_freq=True,
                task_id=0,
                tasktype=None,
                
                path = "../data/recordings",
                shm_name = None,
                # image_queue = None,
                img_shape = (720, 1280, 3),
                is_viewer=False,
                control_dict = None,
                toggle_recording = None,
                crop_size_w = 0,
                crop_size_h = 0,
                cfgs=None
                ):
        
        self.control_dict = control_dict
        self.toggle_recording = toggle_recording
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

        self.print_freq = print_freq
        self.is_viewer = is_viewer
        self.img_shape = img_shape
        # self.img_shape = (2*self.img_shape[0], 2*self.img_shape[1])

        #!!!
        self.frame_rate = 100
        self.last_frame_time = time.time()
        self.start_time = time.time()
        self.is_recording = False
        self.episode = 0
        self.path = path
        self.tasktype = tasktype.lower()

        # Initialize video writers for left and right cameras
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        
        # self.image_queue = image_queue
        self.existing_shm = shared_memory.SharedMemory(name=shm_name)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.existing_shm.buf)

        # state
        indicator_pos_y = self.img_shape[0] // 10
        indicator_pos_x = self.img_shape[1] // 4
        self.color = (0, 255, 0)
        self.record_position = (indicator_pos_x, indicator_pos_y)
        self.img_width = self.img_array.shape[1] // 2
        self.img_height = self.img_array.shape[0]

        self.radius = 10  # Radius of the dot
        self.thickness = -1  # Filled circle

        # gesture check
        self.gesture_color = (0, 0, 0)
        self.gesture_position = (indicator_pos_x, indicator_pos_y)
        self.gesture_thickness = 5

        cfgs = []
        for config_file_name in config_file_names:
            config_file_path = Path(__file__).resolve().parent.parent / "configs" / config_file_name
            with Path(config_file_path).open('r') as f:
                cfg = yaml.safe_load(f)['robot_cfg']
            cfgs.append(cfg)
        self.cfgs = cfgs

        # num
        self.episode_position = (indicator_pos_x + 30, indicator_pos_y+ 10)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.txt_color = (255, 255, 255)
        self.txt_thickness = 2
        self.num_envs = len(cfgs)
        self.envs = []
        self.robot_handles = []

        self.cur_env = task_id
        self.robot_asset_files = []
        self.dof_names = []
        self.num_dof = []
        self.initial_object_qpos = []

        robot_asset_root = str(Path(__file__).resolve().parent.parent / "assets")
        xml_path = self.cfgs[task_id]['xml_path']
        xml_path = os.path.join(robot_asset_root, xml_path, self.tasktype + "_scene.xml")
        self.robot_asset_files.append(xml_path)

        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        self.model.opt.timestep = 1.0 / 70.0  # Increase simulation frequency from 40Hz to 100Hz   ## of =100
        self.model.opt.gravity = np.array([0.0, 0.0, -9.81])
        self.model.opt.iterations = 100  # Increase solver iterations           ## og = 100
        self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON  # Use Newton solver
        # self.model.opt.contactpairs = 1000000
        self.model.opt.tolerance = 1e-6  # Increase tolerance slightly
        self.model.opt.noslip_tolerance = 1e-6  # Increase no-slip tolerance

        # Add integrator settings
        self.model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICIT  # Use implicit integrator
        self.model.opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL  # Use pyramidal friction cone

        # Add more numerical stability settings
        self.model.opt.impratio = 3  # Ratio of implicit to explicit integration

        # Increase joint damping for stability
        for i in range(self.model.nv):
            self.model.dof_damping[i] *= 2.0  # Double the damping

        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        self.hidden_window = glfw.create_window(1, 1, "hidden", None, None)
        glfw.make_context_current(self.hidden_window)

        # Common viewport for both cameras
        self.viewport = mj.MjrRect(0, 0, self.img_width, self.img_height)
        # self.viewport = mj.MjrRect(0, 0, 1280, 720)

        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)        
        self.scene = mj.MjvScene(self.model, maxgeom=10000)

        self.create_envs()
    
    def create_envs(self):

        for env_idx in range(self.num_envs):
            cfg = self.cfgs[env_idx]

            self.robot_joint_ids = np.array(cfg.get('robot_indices', []))
            self.body_ids = np.array(cfg.get('body_indices', []))
            self.initial_object_qpos.append(self.data.qpos[self.robot_joint_ids.shape[0]:].copy())

            dof = len(self.robot_joint_ids)
            robot_dof_props = self.get_dof_properties()
            self.hand_indices = cfg.get('left_hand_indices', []) + cfg.get('right_hand_indices', [])
            self.torso_indices = cfg.get('torso_indices', [])
            self.left_arm_indices = cfg.get('left_arm_indices', [])
            self.right_arm_indices = cfg.get('right_arm_indices', [])
            self.right_hand_indices = cfg.get('right_hand_indices', [])
            self.left_hand_indices = cfg.get('left_hand_indices', [])
            self.lower_torque_limits = cfg.get('lower_torque_limits', [])
            self.upper_torque_limits = cfg.get('upper_torque_limits', [])
            pd = cfg.get('pd', None)
            if pd is not None: # Defined PD gains for H1_inspire and GR1_inspire
                robot_dof_props['stiffness'][cfg.get('body_indices', [])] = pd['body_kp']
                robot_dof_props['damping'][cfg.get('body_indices', [])] = pd['body_kd']
                for side in ['left', 'right']:
                    side_arm_indices = cfg.get(f'{side}_arm_indices', [])
                    side_hand_indices = cfg.get(f'{side}_hand_indices', [])
                    robot_dof_props['stiffness'][side_arm_indices] = pd['arm_kp']
                    robot_dof_props['damping'][side_arm_indices] = pd['arm_kd']
                    robot_dof_props['stiffness'][side_hand_indices] = pd['hand_kp']
                    robot_dof_props['damping'][side_hand_indices] = pd['hand_kd']
            else:
                for i in range(dof):
                    if i in self.hand_indices:
                        robot_dof_props['stiffness'][i] = 500.0
                        robot_dof_props['damping'][i] = 10.0
                        # robot_dof_props['effort'][i] = 39 # effort only for actuators
                    else:
                        robot_dof_props['stiffness'][i] = 60.0
                        robot_dof_props['damping'][i] = 20.0
            
            self.model.geom_friction = 0.3

            joint_names = [self.model.joint(i).name for i in range(self.model.njnt)][:self.robot_joint_ids.shape[0]]
            self.dof_names.append(joint_names)   
            self.num_dof.append(self.robot_joint_ids.shape[0])      

        print('Done creating envs')
       
    def get_dof_properties(self):
        """Get properties only for robot joints, excluding table and cube."""
        dof_props = {
            'lower': self.model.jnt_range[self.robot_joint_ids, 0],     # Joint lower limits
            'upper': self.model.jnt_range[self.robot_joint_ids, 1],     # Joint upper limits
            'velocity': self.model.actuator_ctrlrange[:, 1] if len(self.model.actuator_ctrlrange) > 0 else np.array([]),  # Max velocity limits
            'effort': self.model.actuator_forcerange[:, 1] if len(self.model.actuator_forcerange) > 0 else np.array([]),   # Max torque/force limits
            'stiffness': self.model.jnt_stiffness[self.robot_joint_ids],  # Joint stiffness
            'damping': self.model.dof_damping[self.robot_joint_ids],      # Joint damping
            'friction': self.model.dof_frictionloss[self.robot_joint_ids], # Joint friction
            'armature': self.model.dof_armature[self.robot_joint_ids],    # Joint armature (rotor inertia)
        }
        return dof_props
 
    def setup_viewer(self, viewer):
        self.viewer = viewer
        if self.cfgs[self.cur_env]['name'] == 'gr1' or self.cfgs[self.cur_env]['name'] == 'gr1_inspire':
            self.viewer_pos = np.array([0.16, 0, 1.65]) # TODO: set as head_pos in yml
            self.viewer_target = np.array([0.45, 0, 1.45])
        elif self.cfgs[self.cur_env]['name'] == 'g1_dex3':
            self.viewer_pos = np.array([0.15, 0, 1.3]) # TODO: set as head_pos in yml
            self.viewer_target = np.array([0.45, 0, 1.0])
        else:
            self.viewer_pos = np.array([0.15, 0, 1.65]) # TODO: set as head_pos in yml
        self.viewer_target = np.array([0.45, 0, 1.45])
        # self.viewer_pos = np.array([0.1, 0, 1.65]) # TODO: set as head_pos in yml
        # self.viewer_target = np.array([0.6, 0, 0.955])
        # self.viewer_target = np.array([0.45, 0, 1.45])

        # Calculate distance and angles for MuJoCo camera
        direction = self.viewer_target - self.viewer_pos
        distance = np.linalg.norm(direction)  # Distance from pos to target
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        elevation = np.degrees(np.arcsin(direction[2] / distance))

        # Set camera parameters
        self.viewer.cam.distance = distance
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation
        self.viewer.cam.lookat = self.viewer_target

        # Setup recording cameras
        # self.cam_lookat_offset = np.array([1, 0, -1])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        #! here we hardcode?
        # self.cam_pos = np.array(ROBOT_POS) + np.array([0.1, 0, 1.65])
        # self.cam_pos = np.array(ROBOT_POS) + np.array(self.cfgs[self.cur_env]['head_pos'])

        self.left_cam = mj.MjvCamera()
        self.left_cam.type = mj.mjtCamera.mjCAMERA_FREE
        l_rec_cam_pos = self.viewer_pos + self.left_cam_offset
        l_rec_cam_target = self.viewer_target + self.left_cam_offset
        
        self.right_cam = mj.MjvCamera()
        self.right_cam.type = mj.mjtCamera.mjCAMERA_FREE
        r_rec_cam_pos = self.viewer_pos + self.right_cam_offset
        r_rec_cam_target = self.viewer_target + self.right_cam_offset

        direction = self.viewer_target - self.viewer_pos
        distance = np.linalg.norm(direction)  # Distance from target to pos
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        elevation = np.degrees(np.arcsin(direction[2] / distance))

        self.left_cam.lookat = l_rec_cam_target.tolist()
        self.right_cam.lookat = r_rec_cam_target.tolist()

        # Apart from the lookat, all the other parameters [distance, azimuth, elevation] are same for the cameras. 
        for cam in [self.left_cam, self.right_cam]:
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation

        print('Done setting up viewer')
    
    def sim_config(self):
        return {"num_dof": self.num_dof[self.cur_env], 
                  "dof_names": self.dof_names[self.cur_env], 
                  "urdf_path": self.robot_asset_files[self.cur_env]}
    
    # This is used by open-television
    @property
    def all_sim_configs(self):
        return [{"num_dof": self.num_dof[i], 
                  "dof_names": self.dof_names[i], 
                  "urdf_path": self.robot_asset_files[i]} for i in range(self.num_envs)]

    def step(self, cmd, head_rmat, viewer):

        self.step_camera(head_rmat)

        if self.print_freq:
            start = time.time()

        # For GR1, GR1_JAW, GR1_ACE, force the waist joint's roll, pitch, yaw to 0
        if self.cfgs[self.cur_env]['name'] == 'gr1' or \
            self.cfgs[self.cur_env]['name'] == 'gr1_inspire':

            #TODO: Map this to follow camera instead. 
            # cmd = np.insert(cmd, [53,54,55], 0) # Insert extra indices for joint_head_yaw, joint_head_roll, joint_head_pitch (for GR1)  

            cmd[self.body_ids] = 0
            kp = np.ones(self.robot_joint_ids.shape[0])*200
            kd = np.ones(self.robot_joint_ids.shape[0])*20
            
            kp[self.hand_indices] = 80
            kd[self.hand_indices] = 5

            kp[self.torso_indices] = 600
            kd[self.torso_indices] = 40

            self.set_torque_servo(np.arange(self.model.nu), 1)
            self.data.ctrl[self.robot_joint_ids] = -kp*(self.data.qpos[self.robot_joint_ids] - cmd) - kd*self.data.qvel[self.robot_joint_ids]
            
            
        elif self.cfgs[self.cur_env]['name'] == 'g1_dex3':
            # kp = np.ones(self.robot_joint_ids.shape[0])*100
            # kd = np.ones(self.robot_joint_ids.shape[0])*10
            
            # from IPython import embed; embed()
            
            kp = np.array([100, 100, 100, 200, 20, 20, 
                           100, 100, 100, 200, 20, 20, 
                           300, 300, 300, 
                           90, 60, 20, 60, 4, 4, 4, 
                           1, 1, 1, 1, 1, 1, 1,
                           90, 60, 20, 60, 4, 4, 4,
                           1, 1, 1, 1, 1, 1, 1])
            
            kd = np.array([2.5, 2.5, 2.5, 5, 0.2, 0.1, 
                           2.5, 2.5, 2.5, 5, 0.2, 0.1, 
                           5.0, 5.0, 5.0, 
                           2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2, 
                           0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                           2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2,
                           0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
            
            # self.model.jnt_stiffness[self.hand_indices] = 0
            self.set_torque_servo(np.arange(self.model.nu), 1)
            
            kp[:] = 0
            kd[:] = 0
            
            kp[self.torso_indices] = 600
            kd[self.torso_indices] = 8
            
            kp[self.left_arm_indices] = np.array([90, 60, 20, 60, 4, 4, 4])
            kd[self.left_arm_indices] = np.array([2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2])

            kp[self.right_arm_indices] = np.array([90, 60, 20, 60, 4, 4, 4])
            kd[self.right_arm_indices] = np.array([2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2])
            
            kp[self.right_hand_indices] = np.array([1.0]*7)
            kd[self.right_hand_indices] = np.array([0.2]*7)

            cmd[:] = 0
            cmd[16:20] = 0.5
            cmd[30:34] = -0.5
            
            torques = -kp*(self.data.qpos[self.robot_joint_ids] - cmd) - kd*self.data.qvel[self.robot_joint_ids]
            # torques = np.clip(torques, -10.0, 10.0)
            
            if len(self.lower_torque_limits) > 0:
                torques = np.clip(torques, self.lower_torque_limits, self.upper_torque_limits)
            
            torques = np.clip(torques, -10.0, 10.0)
            
            self.data.ctrl[self.robot_joint_ids] = torques
        
        else:
            kp = np.ones(self.robot_joint_ids.shape[0])*200
            kd = np.ones(self.robot_joint_ids.shape[0])*10
            
            kp[self.hand_indices] = 20
            kd[self.hand_indices] = 0.5
            self.model.jnt_stiffness[self.hand_indices] = 0
            # self.model.jnt_damping[self.hand_indices] = 1
            self.set_torque_servo(np.arange(self.model.nu), 1)
            self.data.ctrl[self.robot_joint_ids] = -kp*(self.data.qpos[self.robot_joint_ids] - cmd) - kd*self.data.qvel[self.robot_joint_ids]

        mj.mj_step(self.model, self.data)
        viewer.sync()

        if self.toggle_recording.is_set():
            if self.control_dict['is_recording']:
                print("Vid recording stopped")
                self.video_writer.release() 
                timestamps_dict = {'video_frame_timestamps': self.timestamps}
                with open(self.timestamp_path, 'w') as f:
                    json.dump(timestamps_dict, f)
                print(f"Data saved to {self.timestamp_path}")
                self.color = (0, 255, 0)
                self.control_dict['is_recording'] = False
            else:
                os.makedirs(self.path, exist_ok=True)
                print('start recording')
                self.video_path = f"{self.path}/episode_{self.episode}_stereo.mp4"
                self.video_writer = cv2.VideoWriter(self.video_path, self.fourcc, self.frame_rate, (int(self.img_shape[1]), self.img_shape[0]))
                # self.video_writer = cv2.VideoWriter(self.video_path, self.fourcc, self.frame_rate, (2560, 720))
                self.timestamp_path = f"{self.path}/episode_{self.episode}_vid_timestamps.json"
                self.timestamps = []
                self.episode += 1
                self.control_dict['is_recording'] = True
                self.color = (255, 0, 0)
                self.reset_env_randomize()
            self.toggle_recording.clear()  


        if self.print_freq:
            print(f"Time taken for step: {time.time() - start}")

    def step_camera(self, head_rmat):

        curr_viewer_target = self.viewer_target @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        l_rec_cam_pos = self.viewer_pos + curr_left_offset
        l_rec_cam_target = curr_viewer_target + curr_left_offset

        r_rec_cam_pos = self.viewer_pos + curr_right_offset
        r_rec_cam_target = curr_viewer_target + curr_right_offset

        direction = curr_viewer_target - self.viewer_pos
        distance = np.linalg.norm(direction)  # Distance from target to pos
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        elevation = np.degrees(np.arcsin(direction[2] / distance))

        self.left_cam.lookat = l_rec_cam_target.tolist()
        self.right_cam.lookat = r_rec_cam_target.tolist()

        # Apart from the lookat, all the other parameters [distance, azimuth, elevation] are same for the cameras. 
        for cam in [self.left_cam, self.right_cam]:
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation

        # Get camera images
        left_image = self.get_camera_image(cam_id=0)  # Use cam_id instead of fixedcamid
        right_image = self.get_camera_image(cam_id=1)

        # Convert images to contiguous arrays that OpenCV can modify
        left_image = np.ascontiguousarray(left_image)
        right_image = np.ascontiguousarray(right_image)

        left_image_ = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image_ = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        combined_frame = np.hstack((left_image_, right_image_))

        # Write images to video files
        if self.control_dict['is_recording']:
            try:
                current_time = time.time()
                # if current_time - self.last_frame_time > 1.0 / self.frame_rate:
                    # print('video shape: ', combined_frame.shape)
                self.video_writer.write(combined_frame)
                self.last_frame_time = current_time
                self.timestamps.append(current_time)
            except Exception as e:
                print(f"Error during video writing: {e}")


        cv2.circle(left_image, self.record_position, self.radius, self.color, self.thickness)
        cv2.circle(right_image, self.record_position, self.radius, self.color, self.thickness)
        cv2.circle(left_image, self.gesture_position, self.radius, self.gesture_color, self.gesture_thickness)
        cv2.circle(right_image, self.gesture_position, self.radius, self.gesture_color, self.gesture_thickness)
        cv2.putText(left_image, str(self.episode), self.episode_position, self.font, self.font_scale, self.txt_color, self.txt_thickness)
        cv2.putText(right_image, str(self.episode), self.episode_position, self.font, self.font_scale, self.txt_color, self.txt_thickness)

        # here we crop the image 
        # left_rgb = left_image[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]
        # right_rgb = right_image[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]

        rgb = np.hstack((left_image, right_image))
        np.copyto(self.img_array, rgb)


        if self.is_viewer:
            cv2.imshow('Left Camera', left_image_)
            cv2.imshow('Right Camera', right_image_)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.end()  
                exit(0)                
 
    def get_camera_image(self, cam_id):
        
        # Update scene with the camera
        mj.mjv_updateScene(
            self.model, 
            self.data, 
            mj.MjvOption(), 
            None,
            self.left_cam if cam_id == 0 else self.right_cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene
        )

            
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)

        rgb_img = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
        mj.mjr_render(self.viewport, self.scene, self.context)
        mj.mjr_readPixels(rgb_img, None, self.viewport, self.context)
            
        return np.flipud(rgb_img)
    
    def set_torque_servo(self, actuator_indices, flag): 
        if (flag==0):
            self.model.actuator_gainprm[actuator_indices, 0] = 0
        else:
            self.model.actuator_gainprm[actuator_indices, 0] = 1
        
    def fetch_pos(self, body_name):
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        return self.model.body(body_id).pos

    def end(self):
        if self.is_viewer:
            glfw.destroy_window(self.hidden_window)
        glfw.terminate()

    def reset_env_randomize(self):

        # # Randomize lighting
        # for i in range(self.model.nlight):
        #     # Store initial light positions if not already stored
        #     if not hasattr(self, '_init_light_pos'):
        #         self._init_light_pos = self.model.light_pos.copy()
            
        #     # Randomize light position relative to initial position
        #     pos_noise = np.random.uniform(-0.5, 0.5, 3)
        #     self.model.light_pos[i] = self._init_light_pos[i] + pos_noise
            
        #     # Randomize light color/intensity (RGB values between 0.5 and 1.0)
        #     self.model.light_ambient[i] = np.random.uniform(0.1, 0.3, 3)
        #     self.model.light_diffuse[i] = np.random.uniform(0.5, 1.0, 3)
        #     self.model.light_specular[i] = np.random.uniform(0.5, 1.0, 3)

        # Hardcoded for tap -- 
        if self.tasktype == 'tap':
            tap_body_pos = self.fetch_pos('tap_object')
            if not hasattr(self, '_init_tap_pos'):
                self._init_tap_pos = tap_body_pos.copy()
            # tap_body.pos = np.array([0.8, 0, -0.3])
            x_offset = np.random.normal(0, 0.03)  # Gaussian noise for the second element
            y_offset = np.random.normal(0, 0.03)  # Gaussian noise for the third element
            tap_body_pos[0] = self._init_tap_pos[0] + x_offset
            tap_body_pos[1] = self._init_tap_pos[1] + y_offset
            mj.mj_forward(self.model, self.data)
            return
        
        # Create a 7-digit array
        number_of_obj_dof = (self.data.qpos.shape[0] - self.robot_joint_ids.shape[0])
        number_of_objects = number_of_obj_dof // 7
        new_array = np.zeros(7 * number_of_objects)  # Initialize a 7-digit array with zeros

        if self.tasktype == 'microwave':
            microwave_body_pos = self.fetch_pos('microwave_object')
            if not hasattr(self, '_init_microwave_pos'):
                self._init_microwave_pos = microwave_body_pos.copy()
            # microwave_body.pos = np.array([0.79, 0.01, 1.33])
            x_offset = np.random.normal(0, 0.01)  # Gaussian noise for the second element
            y_offset = np.random.normal(0, 0.02)  # Gaussian noise for the third element
            microwave_body_pos[0] = self._init_microwave_pos[0] + x_offset
            microwave_body_pos[1] = self._init_microwave_pos[1] + y_offset
            new_array = np.zeros(8) # 1dof for microwave hinge, 7dof for object

        def scalar_first_quat(quat):
            return np.array([quat[3], quat[0], quat[1], quat[2]])
        
        def scalar_last_quat(quat):
            return np.array([quat[1], quat[2], quat[3], quat[0]])

        # Fill the second and third elements with Gaussian noise
        for i in range(number_of_objects):
            new_array[7*i+0] = np.random.normal(0, 0.02)  # Gaussian noise for the second element
            new_array[7*i+1] = np.random.normal(0, 0.02)  # Gaussian noise for the third element

            # Get initial quaternion for this object
            initial_quat = self.initial_object_qpos[self.cur_env][7*i+3:7*i+7]

            # add noise to orientation about z axis
            z = np.random.normal(0, np.pi)
            quat = scalar_first_quat(R.from_rotvec(np.array([0, 0, z])).as_quat())
            quat_rot = R.from_quat(scalar_last_quat(quat))
            quat_init = R.from_quat(scalar_last_quat(initial_quat))

            if self.tasktype == 'microwave':
                quat_init = R.from_quat(scalar_last_quat(self.initial_object_qpos[self.cur_env][7*i+4:7*i+8]))
                self.initial_object_qpos[self.cur_env][7*i+4:7*i+8] = scalar_first_quat((quat_rot * quat_init).as_quat())
            else:
                self.initial_object_qpos[self.cur_env][7*i+3:7*i+7] = scalar_first_quat((quat_rot * quat_init).as_quat())

        # Sample from Gaussian distribution only for 1st and 2nd dof
        self.data.qpos[-number_of_obj_dof:] = self.initial_object_qpos[self.cur_env] + new_array
        self.data.qvel[-number_of_obj_dof:] = 0
        mj.mj_forward(self.model, self.data)