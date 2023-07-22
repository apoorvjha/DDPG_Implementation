import sys
sys.path.insert(1, "../")
import Network_Architecture
import utility
config = utility.read_configuration()
sys.path.pop(1)
import numpy as np
import torch

def test_init():
    try:
        net = Network_Architecture.Network(config)
        assert True
    except Exception as e:
        assert False
def test_feedforward():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128))
    try:
        net(arr)
        assert True
    except Exception as e:
        assert False
def test_updateMethod1():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128))
    pred = net(arr)
    try:
        net.update(pred, np.ones_like(pred))
        assert True
    except Exception as e:
        assert False
def test_updateMethod2():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128))
    pred = net(arr)
    try:
        net.update(np.ones_like(pred))
        assert True
    except Exception as e:
        assert False
def test_updateMethod3():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128))
    pred = net(arr)
    try:
        net.update(pred, np.ones_like(pred), pred)
        assert True
    except Exception as e:
        assert False
def test_updateParameters():
    net = Network_Architecture.Network(config)
    arr = torch.from_numpy(np.random.rand(100, 3, 128, 128))
    pred = net(arr)
    try:
        net.update_parameters(0.2, net.state_dict())
        assert True
    except Exception as e:
        assert False