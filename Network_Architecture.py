from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Tanh, MSELoss, Dropout
from torch.optim import Adam 
import torch

class Network(Module):
    def __init__(self, config, mode = 'actor'):
        super(Network, self).__init__()
        self.config = config
        if self.config['NetworkArchitecture']['type'] == 'CNN':
            # Assuming input is C-channel image of the environment state.
            # Expected Image dimensions -> C X 128 X 128 
            self.cnn_1 = Conv2d(
                in_channels = self.config['NetworkArchitecture']['n_channels'], 
                out_channels = 32, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.cnn_2 = Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            ) 
            self.max_pool = MaxPool2d(
                kernel_size = (2,2)
            )
            self.cnn_3 = Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.cnn_4 = Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.cnn_5 = Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.cnn_6 = Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.cnn_7 = Conv2d(
                in_channels = 128, 
                out_channels = 256, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.cnn_8 = Conv2d(
                in_channels = 256, 
                out_channels = 256, 
                kernel_size = (3,3),
                padding = 0,
                dilation = 1 
            )
            self.flat = Flatten()
            if mode != 'actor':
                self.dense_1 = Linear(64 * 29 * 29 + self.config['n_actions'], 64)
            else:    
                self.dense_1 = Linear(64 * 29 * 29, 64)
            self.dense_2 = Linear(64, 16)
            self.dense_6 = Linear(16, 8)
            if mode == 'actor':
                self.dense_7 = Linear(8, self.config['n_actions'])
            else:    
                self.dense_7 = Linear(8, 1)
        else:
            raise Exception(f"[Exception-NA001] - {self.config['NetworkArchitecture']['type']} Architecture type is not supported!")
        self.optimizer = Adam(self.parameters(), lr = self.config['NetworkArchitecture']['learning_rate'])
    def forward(self, x, x_action = None):
        # assert x.shape[0] == self.config['NetworkArchitecture']['n_channels'], "Input data contains different number of channels!" 
        # assert (x.shape[1] == x.shape[2]) and (x.shape[2] == 128), "Height and width of the array is not 128!"
        
        x = ReLU()(self.cnn_1(x)) # (C, 128, 128) -> (32, 126, 126)
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x) 
        x = ReLU()(self.cnn_2(x)) # (32, 126, 126) -> (32, 124, 124)
        x = self.max_pool(x) # (32, 124, 124) -> (32, 62, 62)
        x = ReLU()(self.cnn_3(x)) # (32, 62, 62) -> (64, 60, 60)
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.cnn_4(x)) # (64, 60, 60) -> (64, 58, 58)
        x = self.max_pool(x) # (64, 58, 58) -> (64, 29, 29)
        x = self.flat(x) # (64, 29, 29) -> (64 * 29 * 29)
        if x_action is not None:
            # x = torch.cat((ReLU()(self.dense_1(x)), x_action), dim = 0) # (64 * 29 * 29 + 1) -> 256
            x = ReLU()(self.dense_1(torch.cat((x, x_action), dim = 1)))
        else:
            x = ReLU()(self.dense_1(x)) # (64 * 29 * 29) -> 256
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_2(x)) # 256 -> 128
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_6(x)) # 16 -> 8
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = self.dense_7(x) # 8 -> 1
        if self.config['NetworkArchitecture']['output_activation'] == 'relu':
            x = ReLU()(x)
        else:
            x = Tanh()(x)
        return x
    def update(self, *args):
        if len(args) == 2:
            # use squard mean loss to update the network
            loss = MSELoss()(args[0], args[1])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif len(args) == 1:
            # Mean of supplied argument as loss
            policy_gradient = -1 * torch.mean(args[-1])
            loss = policy_gradient.clone().detach().requires_grad_(True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            raise Exception("[Exception-NA002] - The supplied number of arguments is incorrect!")
    def update_parameters(self, TAU, parameters):
        for param in self.state_dict().keys():
            self.state_dict()[param] = self.state_dict()[param] * TAU + (1 - TAU) * parameters[param]