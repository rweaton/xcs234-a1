import copy
import random
import numpy as np

class Inventory:
    def __init__(self, init_state=3, seed=1234):
        self.num_states = 11
        self.num_actions = 2  # O <=> SELL, 1 <=> BUY

        # Encode actions to their corresponding
        # position-encoding index values
        sell, buy = 0, 1

        # Configure reward function
        R = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Add +1 rewards for selling when in states 1 through 9
        for s in range(1, 10):
            R[s, sell, (s - 1)] = 1

        # Add +100 reward for buying when in state 9
        R[9, buy, 10] = 100


        # Configure transition function
        T = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Encode initial and rewarding state transitions
        for t in range(0, 10):
            # 100% chance of going to state s+1 when agent in 
            # state s in {0, ..., 9} and performing the "buy" action
            T[t, buy, t + 1] = 1.

        for t in range(1, 10):
            # 100% chance of going to state s-1 when in
            # state s in {1, ..., 9} and performing the "sell" action
            T[t, sell, t - 1] = 1.
        
        # 100% chance of staying in state 10 when agent 
        # performs the "buy" or the "sell" action
        T[10, buy, 10] = 1.
        T[10, sell, 10] = 1.

        # TODO: Check that all other values in the array are zero...

        self.R = np.array(R)
        self.T = np.array(T)

        # Agent always starts with an inventory of 3
        # at the beginning of each episode
        self.init_state = init_state
        self.curr_state = self.init_state

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_model(self):
        return copy.deepcopy(self.R), copy.deepcopy(self.T)

    def reset(self):
        return self.init_state

    def step(self, action):
        # This needs to be modified to account for reward function
        # of form R[s, a, s']
        # reward = self.R[self.curr_state, action]
        reward = np.inner(
            self.T[self.curr_state, action, :],
            self.R[self.curr_state, action, :]
        )
        next_state = np.random.choice(range(self.num_states), p=self.T[self.curr_state, action])
        self.curr_state = next_state
        return reward, next_state