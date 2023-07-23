import gymnasium as gym

def Environment(config):
    return gym.make(config['environment_name'])