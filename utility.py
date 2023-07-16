import json

def read_configuration():
    try:
        with open("./configuration.json", "r") as fd:
            config = json.load(fd)
    except Exception as e:
        raise(f"[Exception-UTIL001] : Failed to read configuration file due to {e}!")
    return config

