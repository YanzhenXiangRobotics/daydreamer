import argparse
import os, sys
sys.path.append("/home/xyz/PRL/Orbit")

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_delta_m", type=float, default=0.04)
parser.add_argument("--control_rate_hz", type=float, default=20)
args_cli = parser.parse_args()

config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch the simulator
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""

import dataclasses
import enum
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import dcargs
import numpy as np
import pyrealsense2 as rs

import embodied

from omni.isaac.orbit_envs.utils import parse_env_cfg
from omni.isaac.orbit_envs.manipulation.reach.reach_env import ReachEnv, ReachEnvCfg
import torch

class FrankaRobotSimWrapper:

    X_RESET_LOW = -0.1
    X_RESET_HIGH = 0.1
    Y_RESET_LOW = -0.1
    Y_RESET_HIGH = 0.1
    Z_RESET = 0.3 # Need to set to a value
    Z_MOVING = 0.3
    Z_TABLE = 0.18

    def __init__(self):
        env_cfg = parse_env_cfg(task_name="Isaac-Reach-Franka-v0", use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
        self.env = ReachEnv(cfg=env_cfg)

    def get_robot_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        servo_angle = self.env._observation_manager.arm_dof_pos_normalized(self.env)
        cart_pos = self.env._observation_manager.ee_position(self.env)
        return (
            np.array(servo_angle, np.float32),
            np.array(cart_pos, np.float32),
        )

    def set_position(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
    ) -> None:
        if z is None:
            z = self.env._observation_manager.ee_position()[-1]
        assert z is not None

        self.env._step_impl(actions=torch.tensor((x, y, z, 0, 0, 0))) # desired rpy?

    def set_z(self, z: float) -> None:
        p = self.env._observation_manager.ee_position(self.env)
        p[-1] = z
        self.env._step_impl(actions=torch.from_numpy(p))


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.001)
        self.last = time.time()

class BaseTask:
    def __init__(self):

        self._arm = FrankaRobotSimWrapper()
        self.rate = Rate(args_cli.control_rate_hz)

    @property
    def obs_space(self) -> Dict[str, embodied.Space]:
        return {
            "image": embodied.Space(np.uint8, (64, 64, 3)), # TODO: add
            "depth": embodied.Space(np.uint8, (64, 64, 1)),
            "cartesian_position": embodied.Space(np.float32, (6,)),
            "joint_positions": embodied.Space(np.float32, (7,)),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    def get_reward(self, curr_obs: Dict[str, Any]) -> float:
        """Provide the enviroment reward.

        A subclass should implement this method.

        Args:
            curr_obs (Dict[str, Any]): Observation dict.

        Returns:
            float: reward
        """
        raise NotImplementedError

    def get_obs(
        self,
        is_first: bool = False,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        color_image, depth_image = np.zeros((64, 64, 3)), np.zeros((64, 64, 3)) #TODO: add visual

        # change observations to be within reasonable values
        servo_angle, cart_pos = self._arm.get_robot_state()
        obs = dict(
            image=color_image,
            depth=depth_image,
            cartesian_position=cart_pos,
            joint_positions=servo_angle,
            is_last=False,
            is_terminal=False,
        )

        if reward is None:
            obs["reward"] = float(self.get_reward(obs))
        else:
            obs["reward"] = reward

        obs["is_first"] = is_first
        return obs

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _reset(self) -> Dict[str, Any]:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        """TODO."""
        raise NotImplementedError

    def close(self) -> None:
        """Nothing to close."""

class ReachTask(BaseTask):  # GraspRewardEnv
    COUNTDOWN_STEPS = 3

    def __init__(self) -> None:
        super().__init__()
        self._arm.set_z(self._arm.Z_RESET)
        x, y = np.random.uniform(self._arm.X_RESET_LOW, self._arm.X_RESET_HIGH), \
                np.random.uniform(self._arm.Y_RESET_LOW, self._arm.Y_RESET_HIGH)
        self._arm.set_position(x, y, self._arm.Z_RESET)

    @property
    def act_space(self) -> Dict[str, embodied.Space]:
        return {"action": embodied.Space(np.int64, (), 0, 4)}


    def compute_arm_position(self, control_action: np.ndarray) -> np.ndarray:
        """Convert control action to TCP homogeneous transform.

        Args:
            env_config (EnvConfig): The environment configuration.
            control_action (np.ndarray, shape=self.control_shape()): control_action
            (should be values between -1 and 1, following the dm_control convention)
            curr_pose (np.ndarray, shape=(6, )): the current robot pose

        Returns:
            np.ndarray, shape=(6, ): The target pose.
        """
        control_action = np.clip(control_action, -1, 1) * self.cfg.max_delta_m
        assert control_action.shape == (2,), control_action

        _, _, cart_pos = self._arm.get_robot_state()
        target_pose = np.array(cart_pos)
        target_pose[:2] = target_pose[:2] + control_action

        target_pose[:2] = (
            np.round(target_pose[:2] / self.cfg.max_delta_m)
        ) * self.cfg.max_delta_m
        xy_min, xy_max = -0.5, 0.5 # Need to further investigate
        target_pose[:2] = np.clip(target_pose[:2], xy_min, xy_max)

        target_pose[2] = self._arm.Z_MOVING
        if control_action[0] == 0:
            target_pose[0] = cart_pos[0]
        if control_action[1] == 0:
            target_pose[1] = cart_pos[1]
        return target_pose

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if action["reset"]:
            if action.get("manual_resume", False):
                return self.get_obs(is_first=True)
            else:
                return self._reset()

        if action["action"] < 4: # move in x, y positions
            pos_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))[action["action"]]
            xy = self.compute_arm_position(np.array(pos_delta))
            self._arm.set_position(xy[0], xy[1])

        elif action["action"] == 4: # try to reach
            self._arm.set_z(self._arm.Z_TABLE)

        self.rate.sleep()

        obs = self.get_obs(is_first=False)
        if obs["reward"] != 0:
            obs = self.get_obs(is_first=False, reward=obs["reward"]
            )

        return obs

    def _reset(self) -> Dict[str, Any]:
        obs = self.get_obs(is_first=True)
        return obs

    def get_reward(self, curr_obs: Dict[str, Any]) -> float:
        if curr_obs["gripper_pos"] >= 0.4 and curr_obs["gripper_pos"] <= 0.6:
            return 10
        else:
            return 0

# def main(env_config: EnvConfig) -> None:
#     pass

def main():
    env = ReachTask()
    obs = env.get_obs(is_first=True)
    print(obs)

if __name__ == "__main__":
    # main(dcargs.parse(EnvConfig))
    main()
