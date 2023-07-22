from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Tanh, MSELoss
from torch.optim import Adam 

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
                padding = 1,
                dialation = 1 
            )
            self.cnn_2 = Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = (3,3),
                padding = 1,
                dialation = 1 
            ) 
            self.max_pool = MaxPool2d(
                kernel_size = (2,2)
            )
            self.cnn_3 = Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = (3,3),
                padding = 1,
                dialation = 1 
            )
            self.cnn_4 = Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = (3,3),
                padding = 1,
                dialation = 1 
            )
            self.cnn_5 = Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = (3,3),
                padding = 0,
                dialation = 1 
            )
            self.cnn_6 = Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = (3,3),
                padding = 0,
                dialation = 1 
            )
            self.cnn_7 = Conv2d(
                in_channels = 128, 
                out_channels = 256, 
                kernel_size = (3,3),
                padding = 0,
                dialation = 1 
            )
            self.cnn_8 = Conv2d(
                in_channels = 256, 
                out_channels = 256, 
                kernel_size = (3,3),
                padding = 0,
                dialation = 1 
            )
            self.flat = Flatten()
            if mode != 'actor':
                self.dense_1 = Linear(256 * 5 * 5 + self.config['n_actions'], 256)
            else:    
                self.dense_1 = Linear(256 * 5 * 5, 256)
            self.dense_2 = Linear(256, 128)
            self.dense_3 = Linear(128, 64)
            self.dense_4 = Linear(64, 32)
            self.dense_5 = Liner(32, 16)
            self.dense_6 = Linear(16, 8)
            self.dense_7 = Linear(8, 1)
        else:
            raise Exception(f"[Exception-NA001] - {self.config['NetworkArchitecture']['type']} Architecture type is not supported!")
    def forward(self, x, x_action = None):
        # assert x.shape[0] == self.config['NetworkArchitecture']['n_channels'], "Input data contains different number of channels!" 
        # assert (x.shape[1] == x.shape[2]) and (x.shape[2] == 128), "Height and width of the array is not 128!"
        
        x = ReLU()(self.cnn_1(x)) # (C, 128, 128) -> (32, 128, 128)
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x) 
        x = ReLU()(self.cnn_2(x)) # (32, 128, 128) -> (32, 128, 128)
        x = self.max_pool() # (32, 128, 128) -> (32, 64, 64)
        x = ReLU()(self.cnn_3(x)) # (32, 64, 64) -> (64, 64, 64)
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.cnn_4(x)) # (64, 64, 64) -> (64, 64, 64)
        x = self.max_pool() # (64, 64, 64) -> (64, 32, 32)
        x = ReLU()(self.cnn_5(x)) # (64, 32, 32) -> (128, 30, 30)
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.cnn_6(x)) # (128, 30, 30) -> (128, 28, 28)
        x = self.max_pool() # (128, 28, 28) -> (128, 14, 14)
        x = ReLU()(self.cnn_7(x)) # (128, 14, 14) -> (256, 7, 7)
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.cnn_8(x)) # (256, 7, 7) -> (256, 5, 5)
        x = self.flat(x) # (256, 5, 5) -> (256 * 5 * 5)
        if x_action is not None:
            x = torch.cat((ReLU()(self.dense_1(x)), x_action), dim = 0) # (256 * 5 * 5 + 1) -> 256
        else:
            x = ReLU()(self.dense_1(x)) # (256 * 5 * 5) -> 256
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_2(x)) # 256 -> 128
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_3(x)) # 128 -> 64
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_4(x)) # 64 -> 32
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_5(x)) # 32 -> 16
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = ReLU()(self.dense_6(x)) # 16 -> 8
        x = Dropout(p = self.config['NetworkArchitecture']['p_dropout'])(x)
        x = self.dense_7(x) # 8 -> 1
        if self.config['NetworkArchitecture']['output_activation'] == 'relu':
            x = ReLU()(x)
        else:
            x = Tanh(x)
        return x
    def update(self, *args):
        optimizer = Adam(self.parameters(), lr = self.config['NetworkArchitecture']['learning_rate'])
        if len(args) == 2:
            # use squard mean loss to update the network
            loss = MSELoss()(args[0], args[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif len(args) == 1:
            # Mean of supplied argument as loss
            loss = -1 * torch.mean(args[-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            raise Exception("[Exception-NA002] - The supplied number of arguments is incorrect!")
    def update_parameters(self, TAU, parameters):
        for param in self.state_dict().keys():
            self.state_dict()[param][0] = self.state_dict()[param] * TAU + (1 - TAU) * parameters[param][0]