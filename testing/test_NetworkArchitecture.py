import sys
import os
sys.path.append("../")
import Network_Architecture
sys.path.pop()
import numpy as np
import torch 
import json

with open("../configuration.json", "r") as fd:
    config = json.load(fd)

def test_init():
    try:
        net = Network_Architecture.Network(config)
        assert True
    except Exception as e:
        assert False
def test_feedforward():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128).astype(float))
    try:
        net.forward(arr.float())
        assert True
    except Exception as e:
        print(e)
        assert False
def test_updateMethod1():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128).astype(float))
    pred = net(arr.float())
    try:
        net.update(pred, torch.ones_like(pred))
        assert True
    except Exception as e:
        assert False
def test_updateMethod2():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128).astype(float))
    pred = net(arr.float())
    try:
        net.update(torch.ones_like(pred))
        assert True
    except Exception as e:
        print(e)
        assert False
def test_updateMethod3():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128).astype(float))
    pred = net(arr.float())
    try:
        net.update(pred, torch.ones_like(pred), pred)
        assert False
    except Exception as e:
        assert True
def test_updateParameters():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128).astype(float))
    pred = net(arr.float())
    try:
        net.update_parameters(0.2, net.state_dict())
        assert True
    except Exception as e:
        assert False