import gym

from AgentsBarRemoteAgent import RemoteAgent

env = gym.make('CartPole-v0')
max_number_of_simulation = 100
score_threshold = 50
agent_name = "CartPoleAgent"
login = "<your_login>"
password = "<your_password>"
agentBar = RemoteAgent(login, password, agent_name)


def play_simulation_until_done(environment, agent, action_randomness):
    '''
    Play a simulation (also called episode) until the environment says it's done.
    :param environment: The Gym AI environment
    :param agent: The Agents-Bar agent which act on the environment, and learns/improves
    :param action_randomness: the randomness of the agent's action for the simulation.
    :return: The score performed by the agent on the environment during this simulation.
    '''
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


for episode_number in range(max_number_of_simulation):
    action_randomness = 0.99 ** episode_number
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
