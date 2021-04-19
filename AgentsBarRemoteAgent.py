import json
import urllib

import requests as requests


class RemoteAgent:
    agent_name = None
    authentication_token = None
    API_BASE_URL = 'https://agents.bar/api/v1/'
    VERIFY_SSL = True

    def __init__(self, username, password, agent_name):
        self.agent_name = agent_name
        self._login(username, password)

    def _login(self, username, password):
        """
        authenticate the user from the input parameters
        :param username: the user's name
        :param password: the user's password
        :return: the 'Authentication' header to be used for further API calls
        """
        data = "grant_type=&username=" + urllib.parse.quote(username) + "&password=" + urllib.parse.quote(
            password) + "&scope=&client_id=&client_secret="
        response = requests.post(
            RemoteAgent.API_BASE_URL + "login/access-token",
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded', 'accept': 'application/json'},
            verify=RemoteAgent.VERIFY_SSL
        )
        if response.status_code != 200:
            raise Exception("could not not connect with user " + username)
        else:
            response_json = response.json()
            self.authentication_token = response_json["token_type"] + " " + response_json["access_token"]
            if self._exist():
                return
            else:
                raise Exception("agent " + self.agent_name + " does not exist")

    def _exist(self):
        """
        check if an agent exist for the user
        :return: True if the agent with the input name exist, False otherwise
        """
        response = requests.get(
            RemoteAgent.API_BASE_URL + "agents/" + self.agent_name,
            headers={'Authorization': self.authentication_token, 'accept': 'application/json'},
            verify=RemoteAgent.VERIFY_SSL
        )
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            raise Exception("could verify if agent exists: " + str(response))

    def act(self, state, noise):
        data = state.tolist()
        response = requests.post(
            RemoteAgent.API_BASE_URL + "agents/" + self.agent_name + "/act?noise="+str(noise),
            data=json.dumps(data),
            headers={'Authorization': self.authentication_token, 'accept': 'application/json'},
            verify=RemoteAgent.VERIFY_SSL
        )
        if response.status_code != 200:
            raise Exception("could not  act on agent: " + str(response))
        else:
            actions = json.loads(response.content)['action']
            if len(actions) == 1:
                return actions[0]
            else:
                raise Exception("action return is invalid: " + str(actions))

    def step(self, state, action, reward, next_state, done):
        data = json.dumps({
                'step_data': {
                    'state': state.tolist(),
                    'action': [action],
                    'reward': reward,
                    'next_state': next_state.tolist(),
                    'done': done
                }
            })
        response = requests.post(
            RemoteAgent.API_BASE_URL + "agents/" + self.agent_name + "/step",
            data=data,
            headers={'Authorization': self.authentication_token, 'accept': 'application/json'},
            verify=RemoteAgent.VERIFY_SSL
        )
        if response.status_code != 202:
            raise Exception("could not step on agent: " + str(response))
