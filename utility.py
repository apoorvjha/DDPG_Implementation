import json
import torch
import numpy as np
import matplotlib.pyplot as plt

def read_configuration():
    try:
        with open("./configuration.json", "r") as fd:
            config = json.load(fd)
    except Exception as e:
        raise(f"[Exception-UTIL001] : Failed to read configuration file due to {e}!")
    return config
def get_device():
    if torch.cuda.is_available():
        print("GPU is available!")
        return torch.device('cuda')
    else:
        print("CPU is used as Fallback!")
        return torch.device('cpu')
def zero_pad_state(state, config):
    canvas = np.zeros(
        (
            config['NetworkArchitecture']['height'],
            config['NetworkArchitecture']['width'],
            config['NetworkArchitecture']['n_channels']
        )
    )
    padding_height = (config['NetworkArchitecture']['height'] - state.shape[0]) // 2 
    padding_width = (config['NetworkArchitecture']['width'] - state.shape[1]) // 2

    canvas[padding_height : state.shape[0] + padding_height,padding_width : state.shape[1] + padding_width] = state
    return canvas.reshape(
        1,
        config['NetworkArchitecture']['n_channels'], 
        config['NetworkArchitecture']['height'],
        config['NetworkArchitecture']['width']
    )
def plot(data, x_label, y_label, title, config):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(data)
    plt.savefig(f"{config['artefact_folder']}/{title}.png")
    plt.clf()
