
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np

from .manyrobot_env import MultiAgentEnv
# from .manyagent_swimmer import ManyAgentSwimmerEnv
from .multi_ropbot_obsk import get_joints_at_kdist, get_parts_and_edges, build_obs

# from gym import utils
# from gym.envs.mujoco import mujoco_env

import robosuite as suite
from robosuite.wrappers import GymWrapper


# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class MujocoEnv_():
    def __init__(self, scenario=None, smarobosuite_robots=None, **kwargs):
        # mujoco_env.MujocoEnv.__init__(self, '/envs/robosuite/robosuite/models/assets/robots/panda/robot.xml', 5)
        # utils.EzPickle.__init__(self)
        # Notice how the environment is wrapped by the wrapper

        # options = {}
        # options["env_name"] = "TwoArmHandover"
        self.scenario = scenario["env_args"]["scenario"]  # e.g. Lift

        self.smarobosuite_robots = scenario["env_args"]["smarobosuite_robots"]  # e.g. Panda #['Panda', 'Panda'],


        # print("++++++++++++++++self.scenario:", self.scenario)

        self.env = GymWrapper(
            suite.make(
                self.scenario, #"Lift",
                robots=self.smarobosuite_robots, # robots= ["Panda"],  #['Panda', 'Panda'], #"Panda", # # "Panda",  # use Sawyer robot "Panda", #
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=False,  # True # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
            )
        )
        self.action_space = self.env.action_space
        self.observation_space = self._get_obs()
        # print("-------------self.env._setup_observables()", self.env._setup_observables())
        # print("-------------self._get_obs()", self._get_obs())
        # self.reward_range = self.env.reward()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # self.env.render()







        # for i_episode in range(20):
        #     observation = env.reset()
        #     for t in range(500):
        #         env.render()
        #         action = env.action_space.sample()
        #         print("action:", action)
        #         observation, reward, done, info = env.step(action)
        #         if done:
        #             print("Episode finished after {} timesteps".format(t + 1))
        #             break
        return observation, reward, done, info
    def reset(self):
        observation = self.env.reset()
        return observation
    def _get_obs(self):
        observation_space = self.env._get_observations()  # self.env._setup_observables()
        observation_space_ = []

        if 'robot0_proprio-state' in observation_space.keys():
            observation_space_.append(observation_space['robot0_proprio-state'])
        if 'robot1_proprio-state' in observation_space.keys():
            observation_space_.append(observation_space['robot1_proprio-state'])
        if 'object-state' in observation_space.keys():
            observation_space_.append(observation_space['object-state'])

        # for i in observation_space.keys():
        #     if i == "robot0_proprio-state" or "robot1_proprio-state" or "object-state":
        #         observation_space_.append(observation_space[i])

        _observation_space_ = []

        for j in range(len(observation_space_)):
            for k in range(len(observation_space_[j])):
                _observation_space_.append(observation_space_[j][k])

        # print("observation_space_------------:", observation_space_)
        # print("-----------------_observation_space_------------:", _observation_space_)
        return _observation_space_









class MujocoMulti(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]  # e.g. Lift
        self.smarobosuite_robots = kwargs["env_args"]["smarobosuite_robots"]  # e.g. panda
        self.agent_conf = kwargs["env_args"]["agent_conf"]  # e.g. '2x3'

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = get_parts_and_edges(self.scenario,
                                                                                            self.agent_conf)
        # self.steps = 0

        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        # print("+++++++++++++++++kwargs[env_args]++++++++++++=", kwargs["env_args"])

        self.agent_obsk = kwargs["env_args"].get("agent_obsk",
                                                 None)  # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get("agent_obsk_agents",
                                                        False)  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            # print("this is agent_obsk")
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                self.k_categories_label = "qpos,qvel|qpos"



            k_split = self.k_categories_label.split("|")
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(",") for k in range(self.agent_obsk + 1)]

            self.global_categories_label = kwargs["env_args"].get("global_categories")
            self.global_categories = self.global_categories_label.split(
                ",") if self.global_categories_label is not None else []

        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(agent_id,
                                                self.agent_partitions,
                                                self.mujoco_edges,
                                                k=self.agent_obsk,
                                                kagents=False, ) for agent_id in range(self.n_agents)]

        # load scenario from script
        self.episode_limit = self.args.episode_limit

        self.env_version = kwargs["env_args"].get("env_version", 2)
        if self.env_version == 2:
            if self.scenario in ["Door", "Lift", "NutAssembly", "NutAssemblyRound", "NutAssemblySingle",
                   "NutAssemblySquare", "PickPlace", "PickPlaceBread", "PickPlaceCan", "PickPlaceCereal",
                   "PickPlaceMilk", "PickPlaceSingle", "Stack"]:
                this_env = MujocoEnv_(kwargs)
            elif self.scenario in ["TwoArmPegInHole","TwoArmHandover", "TwoArmLift"]:
                this_env = MujocoEnv_(kwargs)

            else:
                raise NotImplementedError('Custom env not implemented!')

            # self.wrapped_env = NormalizedActions(
            #     TimeLimit(this_env(**kwargs["env_args"]), max_episode_steps=self.episode_limit))

        else:
            assert False, "not implemented!"
        # self.timelimit_env = self.wrapped_env.env
        # self.timelimit_env._max_episode_steps = self.episode_limit
        # self.env = self.timelimit_env.env
        self.env = this_env
        self.env.reset()
        self.obs_size = self.get_obs_size()
        self.share_obs_size = self.get_state_size()

        # COMPATIBILITY
        self.n = self.n_agents

        # print("---------------self.n_agents:", self.n_agents)


        self.observation_space = [Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in
                                        range(self.n_agents)]

        # self.observation_space = self.env._get_obs()
        # self.share_observation_space= self.env._get_obs()
        self.action_space = self.env.action_space

        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple([Box(self.env.action_space.low[sum(acdims[:a]):sum(acdims[:a + 1])],
                                       self.env.action_space.high[sum(acdims[:a]):sum(acdims[:a + 1])]) for a in
                                   range(self.n_agents)])

        pass

    def step(self, actions):

        # need to remove dummy actions that arise due to unequal action vector sizes across agents
        flat_actions = np.concatenate([actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
        # obs_n, reward_n, done_n, info_n = self.wrapped_env.step(flat_actions)
        obs_n, reward_n, done_n, info_n = self.env.step(flat_actions)
        self.steps += 1
        # print("-----------------------flat_actions---------------------:", flat_actions)

        info = {}
        info.update(info_n)

        # if done_n:
        #     if self.steps < self.episode_limit:
        #         info["episode_limit"] = False   # the next state will be masked out
        #     else:
        #         info["episode_limit"] = True    # the next state will not be masked out

        if done_n:
            if self.steps < self.episode_limit:
                info["bad_transition"] = False  # the next state will be masked out
            else:
                info["bad_transition"] = True  # the next state will not be masked out


        # if self.steps > self.episode_limit:
        #     done_n = True
        # else:
        #     done_n = False


        # return reward_n, done_n, info
        rewards = [[reward_n]] * self.n_agents
        # print("self.n_agents", self.n_agents)
        info["cost"] = [[info["cost"]]] * self.n_agents
        dones = [done_n] * self.n_agents
        infos = [info for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def get_obs(self):
        """ Returns all agent observat3ions in a list """
        state = self.env._get_obs()
        # print("self.env._get_obs()", self.env._get_obs())
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            # obs_n.append(self.get_obs_agent(a))
            # obs_n.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
            # obs_n.append(np.concatenate([self.get_obs_agent(a), agent_id_feats]))
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_n.append(obs_i)
        return obs_n

    def get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env._get_obs()
        else:
            # return build_obs(self.env,
            #                       self.k_dicts[agent_id],
            #                       self.k_categories,
            #                       self.mujoco_globals,
            #                       self.global_categories,
            #                       vec_len=getattr(self, "obs_size", None))
            return build_obs(self.env,
                             self.k_dicts[agent_id],
                             self.k_categories,
                             self.mujoco_globals,
                             self.global_categories)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return len(self.get_obs()[0])
            # return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])

    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        state = self.env._get_obs()
        share_obs = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            # share_obs.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
            state_i = np.concatenate([state, agent_id_feats])
            state_i = (state_i - np.mean(state_i)) / np.std(state_i)
            share_obs.append(state_i)
        return share_obs

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state()[0])

    def get_avail_actions(self):  # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions  # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        self.steps = 0
        # self.timelimit_env.reset()
        self.env.reset()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info



"""
[0] Door (OK)
[1] Lift (OK)
[2] NutAssembly
[3] NutAssemblyRound
[4] NutAssemblySingle
[5] NutAssemblySquare
[6] PickPlace
[7] PickPlaceBread
[8] PickPlaceCan
[9] PickPlaceCereal
[10] PickPlaceMilk
[11] PickPlaceSingle
[12] Stack (OK)
[13] TwoArmHandover
[14] TwoArmLift
[15] TwoArmPegInHole
[16] Wipe



options: {'env_name': 'TwoArmHandover', 'env_configuration': 'single-arm-opposed', 'robots': ['Panda', 'Panda'], 'controller_configs': {'type': 'JOINT_POSITION', 'input_max': 1, 'input_min': -1, 'output_max': 0.05, 'output_min': -0.05, 'kp': 50, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'qpos_limits': None, 'interpolation': None, 'ramp_ratio': 0.2}}

"""
