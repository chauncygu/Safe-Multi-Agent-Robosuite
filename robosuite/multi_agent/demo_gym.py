"""
This script shows how to adapt an environment to be compatible
with the OpenAI Gym-style API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI
gym documentation.

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
from robosuite import load_controller_config

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper

    # load OSC controller to use for all environments
    controller = load_controller_config(default_controller="OSC_POSE")

    # these arguments are the same for all envs
    config = {
        "controller_configs": controller,
        # "horizon": 500,
        "control_freq": 20,
        "reward_shaping": True,
        "has_renderer": True,
        "reward_scale": 1.0,
        "use_camera_obs": False,
        "ignore_done": True,
        "hard_reset": False,
    }

    # this should be used during training to speed up training
    # A renderer should be used if you're visualizing rollouts!
    config["has_offscreen_renderer"] = False

    env = GymWrapper(
        suite.make(
            "Stack", # "TwoArmPegInHole", #"Lift", #  "TwoArmPegInHole"
            robots= "Panda",      # ["Panda", "Panda"], #"Panda",      #  ["Panda", "Panda"], #         # use Sawyer robot
            use_camera_obs=False,           # do not use pixel observations
            has_offscreen_renderer=False,   # not needed since not using pixel obs
            has_renderer=True,              # make sure we can render to the screen
            reward_shaping=True,            # use dense rewards
            control_freq=20,                # control should happen fast enough so that simulation looks smooth
            # **config,
        )
    )

    # record_actions = np.load('record_actions1.npy')

    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):

            # a_record_actions = record_actions[t][0][0]
            # b_record_actions = record_actions[t][0][1]
            # get_record_actions = np.hstack((b_record_actions, a_record_actions))
            # right_record_action = get_record_actions.tolist()
            # action = right_record_action

            env.render()
            action = env.action_space.sample()
            # print("action:", action)
            observation, reward, done, info = env.step(action)
            # print("----reward----:", reward)
            # print("----info----:", info)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
