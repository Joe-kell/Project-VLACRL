import os
import numpy as np
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        # Optional: per-run camera overrides (used by RLinf to create suites with
        # modified camera poses without touching LIBERO assets).
        camera_overrides = kwargs.pop("camera_overrides", None)
        # Optional: per-run light overrides (used by RLinf to create suites with
        # modified lighting conditions without touching LIBERO assets).
        light_overrides = kwargs.pop("light_overrides", None)
        # Optional: per-run robot base position override (used by RLinf to create suites with
        # modified robot initial positions without touching LIBERO assets).
        robot_base_pos_override = kwargs.pop("robot_base_pos_override", None)

        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

        # Store camera_overrides for re-application after reset/set_init_state
        self._camera_overrides = camera_overrides
        # Store light_overrides for re-application after reset/set_init_state
        self._light_overrides = light_overrides
        # Store robot_base_pos_override for re-application after reset/set_init_state
        self._robot_base_pos_override = robot_base_pos_override

        # Apply camera overrides initially if provided
        if camera_overrides is not None:
            self._apply_camera_overrides(is_initial=True)
        # Apply light overrides initially if provided
        if light_overrides is not None:
            self._apply_light_overrides(is_initial=True)
        # Apply robot base position override initially if provided
        if robot_base_pos_override is not None:
            self._apply_robot_base_pos_override(is_initial=True)

    def _apply_camera_overrides(self, is_initial=False):
        """
        Apply stored camera overrides to the MuJoCo model.
        
        This method must be called after reset() and set_state() because:
        1. reset() may reset camera parameters to defaults
        2. set_state() restores the entire MuJoCo state (including camera params) 
           from a saved state, overwriting our overrides
        
        Args:
            is_initial: If True, this is the initial application (during __init__).
                       If False, this is a re-application after reset/set_init_state.
        """
        if self._camera_overrides is None:
            return

        model = self.env.sim.model
        for cam_name, cam_cfg in self._camera_overrides.items():
            try:
                cam_id = model.camera_name2id(cam_name)
            except Exception:
                continue

            if "pos" in cam_cfg:
                model.cam_pos[cam_id] = np.array(cam_cfg["pos"])
            if "quat" in cam_cfg:
                model.cam_quat[cam_id] = np.array(cam_cfg["quat"])

        self.env.sim.forward()

    def _apply_light_overrides(self, is_initial=False):
        """
        Apply stored light overrides to the MuJoCo model.
        
        This method must be called after reset() and set_state() because:
        1. reset() may reset light parameters to defaults
        2. set_state() restores the entire MuJoCo state (including light params) 
           from a saved state, overwriting our overrides
        
        Args:
            is_initial: If True, this is the initial application (during __init__).
                       If False, this is a re-application after reset/set_init_state.
        """
        if self._light_overrides is None:
            return

        model = self.env.sim.model
        for light_name, light_cfg in self._light_overrides.items():
            try:
                light_id = model.light_name2id(light_name)
            except Exception:
                continue

            if "pos" in light_cfg:
                model.light_pos[light_id] = np.array(light_cfg["pos"])
            if "dir" in light_cfg:
                model.light_dir[light_id] = np.array(light_cfg["dir"])
            if "diffuse" in light_cfg:
                model.light_diffuse[light_id] = np.array(light_cfg["diffuse"])
            if "specular" in light_cfg:
                model.light_specular[light_id] = np.array(light_cfg["specular"])
            if "directional" in light_cfg:
                model.light_directional[light_id] = light_cfg["directional"]

        self.env.sim.forward()

    def _apply_robot_base_pos_override(self, is_initial=False):
        """
        Apply stored robot base position override to the MuJoCo model.
        
        This method must be called after reset() and set_state() because:
        1. reset() may reset robot position to defaults
        2. set_state() restores the entire MuJoCo state (including robot position) 
           from a saved state, overwriting our overrides
        
        Args:
            is_initial: If True, this is the initial application (during __init__).
                       If False, this is a re-application after reset/set_init_state.
        """
        if self._robot_base_pos_override is None:
            return

        model = self.env.sim.model
        
        # Get robot base body from robot model
        try:
            robot_base_body_name = self.env.robots[0].robot_model.root_body
        except Exception:
            # Fallback: try to find by name pattern
            robot_base_body_name = None
            for body_id in range(model.nbody):
                body_name = model.id2name(body_id, "body")
                if body_name and "robot0" in body_name and "base" in body_name:
                    robot_base_body_name = body_name
                    break
        
        if robot_base_body_name is None:
            return
        
        try:
            body_id = model.body_name2id(robot_base_body_name)
        except Exception:
            return

        # Apply position override (relative to default position)
        # robot_base_pos_override should be [x, y, z] offset
        if isinstance(self._robot_base_pos_override, (list, tuple, np.ndarray)):
            offset = np.array(self._robot_base_pos_override)
            # Get current position and add offset
            current_pos = model.body_pos[body_id].copy()
            new_pos = current_pos + offset
            model.body_pos[body_id] = new_pos
        elif isinstance(self._robot_base_pos_override, dict):
            # If it's a dict with absolute position
            if "pos" in self._robot_base_pos_override:
                model.body_pos[body_id] = np.array(self._robot_base_pos_override["pos"])
            elif "offset" in self._robot_base_pos_override:
                offset = np.array(self._robot_base_pos_override["offset"])
                current_pos = model.body_pos[body_id].copy()
                new_pos = current_pos + offset
                model.body_pos[body_id] = new_pos

        self.env.sim.forward()

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        # Re-apply camera overrides after reset
        self._apply_camera_overrides(is_initial=False)
        # Re-apply light overrides after reset
        self._apply_light_overrides(is_initial=False)
        # Re-apply robot base position override after reset
        self._apply_robot_base_pos_override(is_initial=False)

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        
        # Re-apply camera overrides before observables are updated
        self._apply_camera_overrides(is_initial=False)
        # Re-apply light overrides before observables are updated
        self._apply_light_overrides(is_initial=False)
        # Re-apply robot base position override before observables are updated
        self._apply_robot_base_pos_override(is_initial=False)
        
        self._post_process()

        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        # ret_seg[segmentation_image == self.segmentation_robot_id+1] = -1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
