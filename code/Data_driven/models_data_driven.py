import torch
import torch.nn as nn
import numpy as np

# A function for loading list data containing irregular arrays
# Writing the function for loading txt data
def read_matrix_data(file_path):
    matrices = []
    current_matrix = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[['):
                current_matrix = []
                data = line[2:-1].split()
                current_matrix.append([float(x) for x in data])
            elif line.endswith(']]'):
                data = line[1:-2].split()
                current_matrix.append([float(x) for x in data])
                matrices.append(np.array(current_matrix))
            elif line.startswith('['):
                data = line[1:-1].split()
                current_matrix.append([float(x) for x in data])
    
    return matrices

# Build a neural network model
## Build a residual network
### Fully connected residual block
class FCResidualBlock(nn.Module):
    def __init__(self, dim):
        super(FCResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 2), 
            nn.Tanh(), 
            nn.Linear(dim * 2, dim),
        )
        
        self.tanh = nn.Tanh()
        
        # Initialization
        for m in self.block:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.tanh(self.block(x) + x)

### Fully connected residual network
class FCResNet(torch.nn.Module):
    def __init__(self, 
                 input_dim            : int         = 3,
                 hidden_dim           : int         = 50,
                 output_dim           : int         = 1,
                 num_blocks_long      : int         = 16,
                 num_blocks_middle    : int         = 8,
                 num_blocks_short     : int         = 1,
                 block                              = FCResidualBlock):
        super(FCResNet, self).__init__()

        self.block = block
        self.hidden_dim = hidden_dim

        # input mapping layer
        self.input_mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh())
      
        # output layer
        self.output_layer_pcM = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 1),
            nn.Tanh()
        )
        
        # residual connection module
        self.net1 = self.add_blocks(num_blocks_long)
        self.net2 = self.add_blocks(num_blocks_middle)
        self.net3 = self.add_blocks(num_blocks_short)

        # Initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
        self.input_mapping.apply(init_weights)
        self.output_layer_pcM.apply(init_weights)
        self.fusion.apply(init_weights)
        
        print('Network initialization completed！')
    
    def add_blocks(self, length):
        blocks = []
        for i in range(length):
            blocks.append(self.block(self.hidden_dim))

        return torch.nn.Sequential(*blocks)

    def forward(self, x):
        x = self.input_mapping(x)

        x1 = x
        x2 = x
        x3 = x
        
        for block in self.net1:
            x1 = block(x1)
        for block in self.net2:
            x2 = block(x2)
        for block in self.net3:
            x3 = block(x3)
            
        combined = torch.cat([x1, x2, x3], dim=-1)
        fused = self.fusion(combined)

        return self.output_layer_pcM(fused)