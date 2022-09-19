# Safe Multi-Agent Robosuite Benchmark

Safe Multi-Agent Robosuite is an extension of [Robosuite](https://github.com/ARISE-Initiative/robosuite) . We split the control over a robot across multiple controllers of its joints---one or more per controller. For example, the Lift task comes with 3 variants: 2 four-dimensional agents (2x4 Lift), 4 two-dimensional agents (4x2 Lift), and 8 one-dimensional agents (8x1 Lift). (This repository is under actively development. We appreciate any constructive comments and suggestions)

Safe MARobosuite tasks are fully cooperative, partialy observable, continuous, and safety-aware. Its multi-agency makes it a compatible framework for training modular robots which are built of multiple, robust parts, refer to the [work](https://mediatum.ub.tum.de/doc/1506779/1506779.pdf). We adopt the reward setting from Robosuite.

 <div align=center>
 <img src="https://github.com/chauncygu/Safe-Multi-Agent-Robosuite/blob/main/docs/safe-multi-agent-robosuite.png" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Figure.1 Example tasks in Safe MARobosuite Environment. (a): Safe 2x4-Lift, (b): Safe 4x2-Lift,  (c): Safe 8x1-Lift, (d): Safe 2x4-Stack, (e): Safe 4x2-Stack,  (f): Safe 8x1-Stack, (g): Safe 14x1-TwoArmPegInHole, (h): Safe 2x7-TwoArmPegInHole. Body parts of different colours of robots are controlled by different agents. Agents jointly learn to manipulate the robot, while avoiding crashing into unsafe red areas.  </center>
 </div>
 
 

## Installation

- Install Robosuite accoring to [Robosuite](https://github.com/ARISE-Initiative/robosuite) and [MuJoCo website](https://www.roboti.us/license.html).
- clone safety multi-agent mujoco to the env path.
&nbsp;

``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

``` Bash
pip install -e .
```


## Quick Start

``` Bash
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", 
    robots="Panda",  
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, cost, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
```

## Tasks
``` Bash
    Lift
    env_args = {"scenario": "Lift",
                  "smarobosuite_robots": "panda"
                  "agent_conf": "2x4",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
    env_args = {"scenario": "Lift",
                  "smarobosuite_robots": "panda"
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
    env_args = {"scenario": "Lift",
                  "smarobosuite_robots": "panda"
                  "agent_conf": "8x1",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
    Stack
    env_args = {"scenario": "Stack",
                  "smarobosuite_robots": "panda"
                  "agent_conf": "2x4",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
    env_args = {"scenario": "Stack",
                  "smarobosuite_robots": "panda"
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
    env_args = {"scenario": "Stack",
                  "smarobosuite_robots": "panda"
                  "agent_conf": "8x1",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
    TwoArmPegInHole
                  
    env_args = {"scenario": "TwoArmPegInHole",
                  "smarobosuite_robots": ["panda", "panda"],
                  "agent_conf": "2x7",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
                  
  

    
```





 



