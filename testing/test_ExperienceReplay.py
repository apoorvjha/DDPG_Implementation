import sys
import os
sys.path.append("../")
import Experience_Replay
sys.path.pop()
import json
import numpy as np

with open("../configuration.json", "r") as fd:
    config = json.load(fd)

def test_init():
    try:
        Experience_Replay.ExperienceReplyBuffer(config)
        assert True
    except Exception as e:
        assert False

def test_store():
    buffer = Experience_Replay.ExperienceReplyBuffer(config)
    try:
        buffer.store(
            np.random.rand(3,128,128),
            np.random.randint(0,2),
            np.random.rand(3,128,128),
            np.random.randint(0,2)
        )
        assert True
    except Exception as e:
        assert False

def test_sample():
    buffer = Experience_Replay.ExperienceReplyBuffer(config)
    for i in range(10):
        buffer.store(
            np.random.rand(3,128,128),
            np.random.randint(0,2),
            np.random.rand(3,128,128),
            np.random.randint(0,2)
        )  
    try:
        k = buffer.sample(5)
        for i in k:
            if i.shape[0] != 5:
                assert False
        assert True
    except Exception as e:
        assert False  