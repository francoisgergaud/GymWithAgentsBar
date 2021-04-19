# Overview
I recently caught up with the deep learning and [Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) 
concepts. 

The basic idea is to implement a continuous loop for training an agent:
1. An agent observes the environment. The *state_size* tell the agent the size of the environment.
1. Once the agent has processed the observation, it will choose a possible action from all the actions possible. This 
   size of the possible actions is the *action_size*.
1. This action will then be applied to the environment. The environment will update, and then a new state of the 
   environment is observable by the agent. Agent also receives a reward for getting into that new state.
1. At this point, the previous observation, the new one, the action taken, and the reward can be communicated to the 
   agent, so that is can learn from them. The reward can be thought as a score of how good was the last action taken.
1. Then we loop back to the beginning until the simulation has terminated. The CartPole simulation will terminate once 
   the stick falls down.

A friend of mine just created a service called [Agents-Bar](https://agents.bar/) which provides agents as service. The 
nice thing is that these agents learn and improve continuously, and I don't have to install any machine learning 
framework locally as the agents are provided as services.

I want to share my experience on using the [Agents-Bar](https://agents.bar/) services with the [Gym AI](https://gym.openai.com/) 
[CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment.

This tutorial's final code is available in this same repository.

# Requirements
The pre-requirement for this tutorial is to have an account in [Agents-Bar](https://agents.bar/) and Python installed locally. The final code 
is provided in this repository.

1. Install the [GymAI](https://gym.openai.com/) and run the CartPole simulation

> Gym is a toolkit for developing and comparing reinforcement learning algorithms.

Gym AI provide several simulations for reinforcement learning exercise. The goal is to provide an agent which can learn 
and improve over time.
```console
pip3 install gym==0.17.3
```
Once installed, I just followed the [Gym official documentation](https://gym.openai.com/docs/) and copy/pasted the following file content into my python script:
```python
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```
and run it. It tooks a few seconds to run on my laptop, seeing the animation of the CartPole failing as it is driven by 
a random decision.

# Creating an Agent as a service with [Agents-Bar](https://agents.bar/)

The basic CartPole implementation has no learning mechanism implemented. It only takes a random action from the list of 
available actions.
Let's try to make it better using an agent which can learn and improve over time.

1. In order to create my agent, I accessed my account on the [Agents-Bar console](https://docs.agents.bar/console/agents.html#), 
   and then selected the *Agents* menu.
1. I selected a model, pick *DQN* which is great for discrete actions like "move left" and "move right". I then have 
   been asked to provide an agent configuration:
```
{
	"state_size": 3,
	"action_size": 1
}
```
What am I supposed to put here?

The Gym AI environment actually provides these pieces of information: after a (really quick) research, I could find 
[the CartPole environment definition](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L26-L38). 
The observation will have a size of 4, and only 2 actions can be taken. So I configured my agent with the following:
```
{
	"state_size": 4,
	"action_size": 2
}
```
I gave a name *CartPoleAgent* to my agent. I saved it and my agent was ready to be used.
**Note**: it is also possible to get the *state_size* and *action_size* from [the Gym AI environment](https://gym.openai.com/docs/#spaces).


## Integrating the agent with the CartPole environment.

From the [Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) page, I am missing the last 
important element: the reward function which tells you how good was the last action the agent performed on the 
environment. The Gym AI environment provides an easy way to get the reward of an action: the *step* function of the 
environment returns this reward aside the new state of the environment:
```python
import gym
env = gym.make('CartPole-v0')
env.reset()
done = False
while not done:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
env.close()
```

In order to train a model, the agent must be able to run a several simulations. We can decide to stop the learning 
process of an agent once it reaches the reward threshold, or the number of simulations run exceed a threshold (just in 
case we were too optimistic, and the reward is unreachable).
So let's define a function that will run the whole simulation and return the score (or cumulated reward) from the final 
state when the simulation is done. This function takes the Gym AI environment, the agent to use, and an *action_randomness*.

**The action randomness** (also called **epsilon**):

If an agent is not given any randomness, it will always execute the same action. But we need the agent to explore 
different possibility so that it can improve. The epsilone is used to define this randomness. When it's 0.5 then we half 
the time do (action) what agent says, and the other half do something completely random. 
Common approach is to change it over time. At the beginning, we want the agent to "explore" as many possibilities as 
possible, by providing a high randomness. Then, as the agent becomes better and better by learning, the want this 
randomness to decrease so that it uses the past learning to perform in better way. In other words: we first need to 
explore the "problem" space  and then, once we've seen enough, we can start exploiting by reducing the exploration. Such 
progression is used to mitigate the "exploration/exploitation" problem.
There is a whole science about [*epsilon greedy policy*](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/). 

Here, we start with a high value, e.g. 0.99, and gradually declining to something small, e.g. 0.01. We simply assign: 
```noise = 0.99 ** episode_number```
so that the randomness consecutive values will go decreasing from 0.99 (0.99, 0.9801, 0.970299, 0.96059601 ...).

```python
def play_simulation_until_done(environment, agent, action_randomness):
    observation = environment.reset()
    done = False
    score = 0.0
    while not done:
        # agentBar acts and returns a float. CartPole environment requires an integer
        action = int(agent.act(observation, action_randomness))
        # we update the environment
        next_observation, reward, done, info = environment.step(action)
        score += reward
        # agentBar learn from it previous action
        agent.step(observation, action, reward, next_observation, done)
        observation = next_observation
    return score
```
Note: the content of the *AgentsBarRemoteAgent* is a library provided by [Agents-Bar](https://agents.bar/) (which encapsulate the calls to the REST API). Aside the authentication features, it provides 2 functions on an agent:
* *act*: the agent receives environment observation, also called *state*, and returns an action
* *step*: the agent receives the previous environment observation, the last action taken, the new environment observation, and the reward to quantify how good was the last taken action. The agent uses these data to learn and improve.

Let's now use this play simulation function over and over again so that the model can train and learn:
```python

import gym

from AgentsBarRemoteAgent import RemoteAgent

env = gym.make('CartPole-v0')
max_number_of_simulation = 100
score_threshold = 50
agent_name = "<your_agent_name>"
login = "<your_agent_bar_login>"
password = "<your_agent_bar_password>"
agentBar = RemoteAgent(login, password, agent_name)  # This should have (state_size, action_size, name=, login=, password=)

# the main loop triggering simulations to train the model
for simulation_number in range(max_number_of_simulation):
    action_randomness = 0.99 ** simulation_number
    score = play_simulation_until_done(env, agentBar, action_randomness)
    print("score: " + str(score))
    if score >= score_threshold:
        break

env.close()
```

Let's run the agent until it reaches my reward threshold, or the number of maximum simulations is reached. Here is the successive simulations' scores:
```console
score: 23.0
score: 10.0
score: 9.0
score: 14.0
...
score: 90.0
```

We can also display a last simulation using the current learning from the agent, to get an idea about it better performs compared to a random agent.

```python
# the main loop triggering simulations to train the model
for simulation_number in range(max_number_of_simulation):
    action_randomness = 0.99 ** simulation_number
    score = play_simulation_until_done(env, agentBar, action_randomness)
    print("score: " + str(score))
    if score >= score_threshold:
        observation = env.reset()
        done = False
        while not done:
            # agentBar acts and returns a float. CartPole environment requires an integer
            action = int(agentBar.act(observation, action_randomness))
            # we update the environment
            next_observation, reward, done, info = env.step(action)
            env.render()
            observation = next_observation
        break
env.close()
```

After a given number of simulations, I reach my reward goal.
It was slow compared of having to run an agent locally (I guess it is mainly due to network latencies). But it also brings some advantages:
* I could actually stop my agent training, and continue it later: [Agents-Bar](https://agents.bar/) manage my agent as the service: the agent is always available, for both training and acting.
* I don't have to install any deep-learning frameworks locally or any other ML tooling. Only my environment runs locally. [Agents-Bar](https://agents.bar/) provides agents with default configurations which are versatile enough for my environment.
* I can scale it: as [Agents-Bar](https://agents.bar/) provides agents as service, it is possible to train much more agents in parallel, or agents with much denser layers. I can use the computation power from the service without having the required this computation power locally.


