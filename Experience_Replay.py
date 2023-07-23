import numpy as np

class ExperienceReplyBuffer:
    def __init__(self, config):
        self.current_state = np.zeros(
            (
                config['experience_replay_buffer_size'],
                config['NetworkArchitecture']['n_channels'],
                config['NetworkArchitecture']['height'],
                config['NetworkArchitecture']['width']
            )
        )
        self.current_action = np.zeros(
            (
                config['experience_replay_buffer_size'],
                1
            )
        )
        self.next_state = np.zeros(
            (
                config['experience_replay_buffer_size'],
                config['NetworkArchitecture']['n_channels'],
                config['NetworkArchitecture']['height'],
                config['NetworkArchitecture']['width']
            )
        )
        self.reward = np.zeros(
            (
                config['experience_replay_buffer_size'],
                1
            )
        )
        self.current_idx = 0
        self.buffer_size = config['experience_replay_buffer_size']
    def store(self, current_state, current_action, next_state, reward):
        self.current_state[self.current_idx] = current_state
        self.current_action[self.current_idx] = current_action
        self.next_state[self.current_idx] = next_state
        self.reward[self.current_idx] = reward
        self.current_idx = (self.current_idx + 1) % self.buffer_size
    def __len__(self):
        return self.current_idx
    def sample(self, N):
        assert N <= self.buffer_size, "The size of sample cannot exceed size of buffer!"
        idx = np.random.choice([i for i in range(self.buffer_size)], N, replace = False)
        return self.current_state[idx], self.current_action[idx], self.next_state[idx], self.reward[idx]