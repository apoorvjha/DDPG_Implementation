import numpy as np

class OrnsteinUlembeckNoise:
    """
    To implement the better exploration by actor network, This function creates noisy
    perturbations. It samples noise from normal correlated distribution.

    Code(Ref) : https://keras.io/examples/rl/ddpg_pendulum/  
    """
    def __init__(self, mean, std, theta = 0.15, dt = 1e-2, x_initial = None):
        self.theta = theta
        self.mean = mean
        self.std = std
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
    def sample(self):
        x = (
            self.x_previous
            + self.theta * (self.mean - self.x_previous) * self.dt
            + self.std * np.sqrt(self.dt) * np.random.normal(size = self.mean.shape)
        )
        self.x_previous = x
        return x
    def reset(self):
        if self.x_initial is not None:
            x_previous = self.x_initial
        else:
            self.x_previous = np.zeros_like(self.mean)