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

import imageio
import numpy as np

import robosuite.utils.macros as macros
from robosuite import make

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

options={'env_name': 'Lift', 'robots': 'Panda', 'controller_configs': {'type': 'OSC_POSITION', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05], 'output_min': [-0.05, -0.05, -0.05], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2}}

#options={'env_name': 'Lift', 'robots': 'Panda', 'controller_configs': {'type': 'OSC_POSITION'}}


if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            #**options,
            "TwoArmPegInHole", #"TwoArmHandover", #Lift
            robots=["Panda", "Panda"], # "Panda",                # use Sawyer robot
            #controller_configs = {'type': 'OSC_POSITION'},
            # use_camera_obs=False,           # do not use pixel observations
            # has_offscreen_renderer=False,   # not needed since not using pixel obs
            # has_renderer=True,              # make sure we can render to the screen
            reward_shaping=True,            # use dense rewards
            control_freq=20,                # control should happen fast enough so that simulation looks smooth

            # for recording video
            has_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names="agentview",
            camera_heights=512,
            camera_widths=512,
        )
    )
    # print("self.env._get_observations():",env._get_observations())

    # env = suite.make(
    #     # **options,
    #     "Lift",  # "TwoArmHandover", #Lift
    #     robots="Panda",  # ["Panda", "Panda"], # "Panda",                # use Sawyer robot
    #     # controller_configs = {'type': 'OSC_POSITION'},
    #     # use_camera_obs=False,           # do not use pixel observations
    #     # has_offscreen_renderer=False,   # not needed since not using pixel obs
    #     # has_renderer=True,              # make sure we can render to the screen
    #     reward_shaping=True,  # use dense rewards
    #     control_freq=20,  # control should happen fast enough so that simulation looks smooth
    #
    #     # for recording video
    #     has_renderer=False,
    #     ignore_done=True,
    #     use_camera_obs=True,
    #     use_object_obs=False,
    #     camera_names="agentview",
    #     camera_heights=512,
    #     camera_widths=512,
    # )

    # for recording video
    obs = env.reset()
    ndim = env.action_dim
    # create a video writer with imageio
    writer = imageio.get_writer("video_gsd2022_agentview.mp4", fps=20)
    frames = []

    for i_episode in range(2):
        observation = env.reset()
        for t in range(100):
            # env.render()
            action = env.action_space.sample()
            # action = 0.5 * np.random.randn(ndim)
            print("action---demo_gym_functionality.py:", action)
            observation, reward, done, info = env.step(action)
            # print("observation:", observation)
            # print("----info----:", info)
            print("observation---demo_gym_functionality.py:", observation)

            # dump a frame from every K frames
            if t % 1 == 0:
                frame = observation["agentview" + "_image"]
                writer.append_data(frame)
                print("Saving frame #{}".format(t))

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    writer.close()
