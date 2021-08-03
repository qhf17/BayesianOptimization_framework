from typing import Dict, List
import numpy as np

NUM_RUNS = 400
NUM_USER = 3
NUM_CREATOR = 3


class Agent(object):
    def __init__(
            self,
            seg_i: list = True,
            seg_j: list = True,
            lambda_i: list = True,
            lambda_j: list = True,
            rho_ij: list = True,

    ):
        """constructor.

        Args:
            seg_i:  a list proportion of user type i (i=1,2,3)
            seg_j:  a list proportion of creator type j (j=1,2,3)
            lambda_i: a list of user type i arrival rate (i=1,2,3)
            lambda_j: a list of creator type j arrival rate (j=1,2,3)
            rho_ij: a list proportion of creator type j recommended to user type i (list=[rho_11, rho_12, rho_13, rho_21,..., rho_32, rho_33])


        """
        self.seg_i = seg_i
        self.seg_j = seg_j
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.rho_ij = rho_ij

    # cacluate the new arrival rate of user type i and creator type j
    def user_arrival_rate(self, i, theta_i):
        List_i = np.array(self.rho_ij)
        List_i = (List_i.reshape(3, 3))[:, i - 1]
        f_i = np.sum(theta_i * np.array(self.seg_i) + np.dot(List_i, np.array(self.lambda_j)))
        return f_i

    def creator_arrival_rate(self, j, theta_j):
        List_j = np.array(self.rho_ij)
        List_j = (List_j.reshape(3, 3))[:, j - 1]
        f_j = np.sum(theta_j * np.array(self.seg_j)+ np.dot(List_j, np.array(self.lambda_i)))
        return f_j


class ContextualRunner:
    def __init__(self, num_contexts, context_list, return_context_reward=True):
    # context_list = [seg_i, seg_j, lambda_i, lambda_j]

        self.context_list = context_list
        self.return_context_reward = return_context_reward
        self.Seg_i=self.context_list [0]
        self.Seg_j = self.context_list[1]
        self.Lambda_i = self.context_list[2]
        self.Lambda_j = self.context_list[3]

        self.num_contexts = num_contexts
        self._contextual_parameters = []
        for i in range(NUM_USER+1):
           self._contextual_parameters.extend(
               [
                   {
                    "name": f"Rho_{i}{j}",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                    "log_scale": False,
                  }
                  for j in range(NUM_CREATOR+1)
              ]
           )


    def f(self, Rho_ij, Theta_i, Theta_j):
        agent_sample = Agent(
            seg_i=self.Seg_i,
            seg_j=self.Seg_j,
            lambda_i=self.Lambda_i,
            lambda_j=self.Lambda_j,
            rho_ij=Rho_ij,
        )
        context_rewards = arrival_aggregate_reward(
            agent=agent_sample, i=NUM_USER, j=NUM_CREATOR, theta_i=Theta_i, theta_j=Theta_j
        )
        return context_rewards  # reward maximization

    def contextual_parameters(self) -> List[Dict]:
        return self._contextual_parameters

def arrival_aggregate_reward (agent, i, j, theta_i, theta_j):
    reward = 0
    for k in range(i+1):
        reward += agent.user_arrival_rate(k, theta_i[k-1])
    for k in range(j+1):
        reward += agent.creator_arrival_rate(k, theta_j[k - 1])
    return reward