import json
import torch

def read_configuration():
    try:
        with open("./configuration.json", "r") as fd:
            config = json.load(fd)
    except Exception as e:
        raise(f"[Exception-UTIL001] : Failed to read configuration file due to {e}!")
    return config
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
