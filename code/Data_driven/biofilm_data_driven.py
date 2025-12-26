import numpy as np
import torch
import torch.nn as nn
import sys
import os
import time
import math
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import uniform, norm
from torchvision import transforms 
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DataParallel
from models_data_driven import FCResNet,read_matrix_data
from save_method import save_training_artifacts

random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float)

# cuda support？
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available !')
else:
    device = torch.device('cpu')


# Neural network structure configuration
input_dim = 3
hidden_dim = 50
output_dim = 1
num_blocks_long = 16
num_blocks_middle = 8
num_blocks_short = 1

# batch size
batch_size_label = int(12800 / 2)

# Optimizer settings
lr = 8e-4
weight_decay = 0.00002
step_size = 20
gamma = 0.9

# Number of iterations
epoch_number = 100

# Define the threshold for gradient clipping
max_grad_norm = 1 

# Data division
cut_time = 16    # The time point at which the training data and the extrapolation data (testing data) are divided
rate_train_cover_fracture = 0.1  # Sampling ratio within the sampling space that includes free channels
rate_train_without_fracture = 0.04  # Sampling ratio within the sampling space that does not include free channels

# The address of data
# Biomass data (Corresponding to biofilm regions; Used for sampling of labeled points)
file_path = "/data2/yq2/biofilm_zgf/data/biofilm_data_scale2.txt"
# Biomass data (Corresponding to free channel regions; Used for sampling of labeled points)
file_path2 = "/data2/yq2/biofilm_zgf/data/fracture_data_scale2.txt"

# The storage address of the training and extrapolation data
datasave_path = '/data2/yq2/biofilm_zgf2/data/save_dataset/'
os.makedirs(datasave_path, exist_ok=True)

config = {
    "input_size": input_dim,
    "hidden_size": hidden_dim,
    "output_size": output_dim,
    "number of long ResNet's blocks": num_blocks_long,
    "number of middle ResNet's blocks": num_blocks_middle,
    "number of short ResNet's blocks": num_blocks_short,
    "The moment of division between the training set and the extrapolated data": cut_time,
    "The percentage of training data in the training set (cover fracture)": rate_train_cover_fracture,
    "The percentage of training data in the training set (without fracture)": rate_train_without_fracture,
    "batch_size": batch_size_label,
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "The learning rate changes the stride": step_size,
    "Learning rate change ratio": gamma,
    "num_epochs": epoch_number,
    "notes": "The input and the input normalization range are [-1,1]. The optimizer is Adam. The loss function is HuberLoss. The activation function is tanh().",
    "Main code name": "biofilm_data_driven.py",
    "Model code name": "models_data_driven.py",
    "The code used for saving is": "save_method.py",
    "The code path is": "/data2/yq2/biofilm_zgf2/code/data_driven/"
}

# Data extraction
biofilm_data = read_matrix_data(file_path)
fracture_data = read_matrix_data(file_path2)

number_img1 = len(biofilm_data)
number_img2 = len(fracture_data)
if number_img1 == number_img2:
    number_img = number_img1
    print('The number of experimental images is:',number_img)
else:
    print('Incorrect matching of data quantity !')
    sys.exit()

# Merge data
biomass_data_set = []
for i in range(number_img):
    biofilm_mass = biofilm_data[i]
    fracture_mass = fracture_data[i]
    biomass_data_one = np.concatenate((biofilm_mass,fracture_mass),axis = 0)
    biomass_data_set.append(biomass_data_one)

N = biomass_data_set[0].shape[0]
print('The number of spatial points in a single moment is：', N)

# Total data
data_all = np.array(biomass_data_set).reshape(-1,4)

max_data_all = np.max(data_all, axis=0)
min_data_all = np.min(data_all, axis=0)
print('The maximum value in the total data:',max_data_all)
print('The minimum value in the total data:',min_data_all)

# Partitioning the training data (including the training set and the validation set) and the extrapolation/testing data
# the testing data
data_timeout = data_all[cut_time*N:,:] 
print('The testing data includes following time points：',np.unique(data_timeout[:,2:3]))
print('The size of the testing data is：',data_timeout.shape)

print('----------------------------')
# the training data
data_train = data_all[0*N:cut_time*N,:]
print('The training data includes following time points：',np.unique(data_train[:,2:3]))
print('The size of the training data is：',data_train.shape)

np.save(datasave_path + 'less_train_data.npy', data_train)
np.save(datasave_path + 'less_timeout_data.npy', data_timeout)

# Data preprocessing
xmax, xmin = max_data_all[0], 0.0 * 0.001
ymax, ymin = max_data_all[1], 0.0 * 0.001
tmax, tmin = max_data_all[2], 0.0 * 3 * 60
Mmax, Mmin = max_data_all[3], 0.0

normalize_index_max_with_M = np.array([xmax, ymax, tmax, Mmax])
normalize_index_min_with_M = np.array([xmin, ymin, tmin, Mmin])

normalize_index_max = np.array([xmax, ymax, tmax])
normalize_index_min = np.array([xmin, ymin, tmin])

# Normalization
data_train = (data_train - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0
data_timeout = (data_all[23*N:24*N,:] - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0

# The training data is divided into a training set and a validation set
mask = (data_train[:,1:2]> ((0.0017 - ymin) * 2.0 / (ymax - ymin) - 1.0)).flatten()
data_train_cover_facture = data_train[mask]
data_train_without_facture = data_train[~mask]

idx_train_cover_facture = np.random.choice(data_train_cover_facture.shape[0], int(146743 * 10 / 14), replace=False)
idx_train_without_facture = np.random.choice(data_train_without_facture.shape[0], int(146743 * 4 / 14), replace=False)

idx_validate_cover_facture = np.array(list(set(np.arange(data_train_cover_facture.shape[0])).difference(set(idx_train_cover_facture))))
idx_validate_without_facture = np.array(list(set(np.arange(data_train_without_facture.shape[0])).difference(set(idx_train_without_facture))))

data_training_cover_facture = data_train_cover_facture[idx_train_cover_facture]
data_training_without_facture = data_train_without_facture[idx_train_without_facture]

data_validating_cover_fracture = data_train_cover_facture[idx_validate_cover_facture]
data_validating_without_fracture  = data_train_without_facture[idx_validate_without_facture]

data_training = np.concatenate((data_training_cover_facture, data_training_without_facture), axis = 0)
data_validating = np.concatenate((data_validating_cover_fracture, data_validating_without_fracture), axis = 0)

np.random.shuffle(data_training)
np.random.shuffle(data_validating)

print('The size of the training set is:', data_training.shape)
print('The size of the validation set is:', data_validating.shape)

# Convert to tensor
data_train_t = torch.from_numpy(data_training).float()
data_validate_t = torch.from_numpy(data_validating).float()
data_timeout_t = torch.from_numpy(data_timeout).float()

# Batch settings
dataset_train_label = TensorDataset(data_train_t)
dataloader_label = DataLoader(dataset_train_label, batch_size=batch_size_label, shuffle=False)

# Define the main model
model = FCResNet(input_dim, hidden_dim, output_dim, num_blocks_long, num_blocks_middle, num_blocks_short).to(device)
model = DataParallel(model)


# Define the optimizer 
params = [
            {'params': model.parameters()}
]
optimizer = torch.optim.Adam(params, lr=lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# The loss function
class HuberLoss(nn.Module):
    """
    Huber loss function.

    Parameters:
    delta --  the switching point that controls the transition between squared loss and linear loss (default value 1.0)
    reduction -- aggregation method: 'none'|'mean'|'sum' (Default:'mean')
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, targets, inputs):
        """
        Calculate the Huber loss.
        
        Parameters:
        inputs -- Predicted value (Tensor)
        targets -- True value (Tensor)
        
        Return:
        Loss value (scalar Tensor or Tensor of the same shape as the input)
        """
        if inputs.shape != targets.shape:
            raise ValueError(f"The input shape: {inputs.shape} does not match the target shape: {targets.shape}.")
        
        error = targets - inputs
        abs_error = torch.abs(error)
        
        delta_tensor = torch.tensor(self.delta, device=error.device, dtype=error.dtype)
        
        loss = torch.where(
            abs_error <= delta_tensor,
            0.5 * error**2,
            delta_tensor * (abs_error - 0.5 * delta_tensor)
        )
        
        # aggregation
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("The reduction parameter must be 'none', 'mean' or 'sum' !")

    def extra_repr(self):
        return f'delta={self.delta}, reduction={self.reduction}'

huber_loss = HuberLoss(delta=1.5, reduction='mean')

# Model training
loss_total = []
start_time = time.time()

R2_best = 0.0
for epoch in range(epoch_number):
    epoch += 1
    ###################
    # train the model #
    ###################
    for (batch_idx1, data_label) in tqdm(
        enumerate(dataloader_label),
        total=len(dataloader_label),
        desc=f'Epoch {epoch}/{epoch_number}',
        leave=True):
        
        # Biomass
        data_label = data_label[0].to(device).requires_grad_(True)
        out_label = model(data_label[:,0:3])
        loss_label_M_1 = huber_loss(data_label[:,3:4], out_label)

        loss_label_M = loss_label_M_1

        # Total loss
        loss =  loss_label_M

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    scheduler.step()
    
    if epoch % 1 == 0:
        time_elapsed = time.time() - start_time
        print('-------------------------------------------------------------------------------')
        print(f'''Epoch={epoch}
Training loss (all) ={loss.item():.5f}
loss label(M)={loss_label_M.item():.5f}
''')
        # The validation set
        data_validate = data_validate_t.to(device).requires_grad_(True)
        out_validate = model(data_validate[:,0:3])
        loss_validate_M = huber_loss(data_validate[:,3:4], out_validate)

        # The extrapolation set
        data_timeout = data_timeout_t.to(device).requires_grad_(True)
        out_timeout = model(data_timeout[:,0:3])
        loss_timeout_M = huber_loss(data_timeout[:,3:4], out_timeout)

        R2_validate = r2_score(data_validate[:,3:4].cpu().detach().numpy(), out_validate.cpu().detach().numpy())
        R2_timeout = r2_score(data_timeout[:,3:4].cpu().detach().numpy(), out_timeout.cpu().detach().numpy())
        print('The accuracy rate on the validation set is：：',R2_validate)
        print('The accuracy rate on the extrapolation set is：',R2_timeout)
        print('-------------------------------------------------------------------------------')
        
        epoch_loss = [epoch,\
                      time_elapsed,\
                      loss.item(),\
                      loss_label_M.item()
                        ]
        
        loss_total.append(epoch_loss)
        if R2_timeout > R2_best:
            R2_best = R2_timeout
            loss_Total = np.array(loss_total)
            save_training_artifacts(model, loss_Total, config, model_name="data_driven", best = True)
    # cache
    torch.cuda.empty_cache()

loss_Total = np.array(loss_total)
save_training_artifacts(model, loss_Total, config, model_name="data_driven", best = False)