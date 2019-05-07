import numpy as np


class ExperienceReplay(object):

    def __init__(self, max_memory=100, discount=.9):
        """ Initialize the memory of the agent. """
        self.max_memory = max_memory  # number of inputs that can be saved
        self.memory = list()
        self.discount = discount  # the discount factor aka gamma (shortsighted / longsighted agent)

    def remember(self, states, done):
        """ Renew memory pool if it gets overflown. """
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], done?]
        self.memory.append([states, done])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        """ Calculate the current target values based on previous states/actions."""
        len_memory = len(self.memory)
        # num_actions = model.output_shape[-1]

        inputs = np.zeros((min(len_memory, batch_size), 64, 64, 3))

        # targets = np.zeros((inputs.shape[0], num_actions))

        # get random samples from the memory pool
        it = np.random.randint(0, len_memory, size=inputs.shape[0])
        it = np.sort(it)
        states_ts = []
        states_tps = []

        for i, idx in enumerate(it):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            inputs[i:i+1] = state_t
            # get the previous state_t and the current state_tp1
            states_ts.append(state_t)
            states_tps.append(state_tp1)

        states_ts = np.array(states_ts)
        states_ts = states_ts.reshape((-1, 64, 64, 3))

        targets = model.predict(states_ts)
        states_tps = np.array(states_tps).reshape((-1, 64, 64, 3))
        q_sa = np.max(model.predict(states_tps), axis=1)
        # maybe dont update q at every step but after each ~50

        for i, idx in enumerate(it):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            done = self.memory[idx][1]

            if done:  # if done is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * q_sa[i]

        return inputs, targets
