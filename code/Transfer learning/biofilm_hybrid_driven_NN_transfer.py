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
from models_hybrid_driven_NN_transfer import FCResNet, NN_c, read_matrix_data, HuberLoss
from save_method_hybrid_NN_transfer import save_training_artifacts

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
output_dim = 5
num_blocks_long = 16
num_blocks_middle = 8
num_blocks_short = 1

# batch size
batch_size_label = int(12800 / 5) 

# Define the permeability conversion parameter
alpha = torch.tensor(2.748).to(device) 
alpha.requires_grad = False
if alpha.requires_grad:
    alpha_requires_grad = 1.0
else:
    alpha_requires_grad = 0.0

# Define the adjustment parameter b
co1 = torch.tensor(1.08).to(device)
co1.requires_grad = False
if co1.requires_grad:
    co1_requires_grad = 1.0
else:
    co1_requires_grad = 0.0

# Optimizer settings
lr = 2e-4
weight_decay = 0.00002
step_size = 20
gamma = 0.9

# Number of iterations
epoch_number = 100

# Define the threshold for gradient clipping
max_grad_norm = 1

# The iteration threshold for transitioning from pure data-driven training to PINN
epoch_change = 0

# Setting of loss weights
sigma1 = 80.0
sigma2 = 20.0

# Data division
cut_time = 8  # The time point at which the training data and the extrapolation data (testing data) are divided
rate_train_cover_fracture = 0.05  # Sampling ratio within the sampling space that includes free channels
rate_train_without_fracture = 0.025  # Sampling ratio within the sampling space that does not include free channels
rate_initial_M = 0.1  # Sampling ratio of initial biomass
rate_initial_boundary = 0.35  # Boundary condition

# The address of data
# Biomass data (Corresponding to biofilm regions; Used for sampling of labeled points)
file_path = "/data1/yq1/biofilm_zgf/data/biofilm_data_scale_03301.txt"
# Biomass data (Corresponding to free channel regions; Used for sampling of labeled points)
file_path2 = "/data1/yq1/biofilm_zgf/data/fracture_data_scale_03301.txt"

# Boundary
file_initial_boundary = '/data2/yq2/biofilm_zgf/data/data_boundary_less.npy'
# Outlet
file_initial_outlet = '/data2/yq2/biofilm_zgf/data/data_outlet.npy'

# The storage address of the training and extrapolation data
datasave_path = '/data1/yq1/biofilm_zgf/data/save_dataset/'
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
    "The usage ratio of the initial biomass data points": rate_initial_M,
    "The usage ratio of the initial boundary data points": rate_initial_boundary,
    "batch size of label data": batch_size_label,
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "The learning rate changes the stride": step_size,
    "Learning rate change ratio": gamma,
    "num_epochs": epoch_number,
    "Gradient clipping threshold": max_grad_norm,
    "The epoch threshold between pure data-driven transitions and PINNs": epoch_change,
    "Initial alpha value": float(alpha),
    "Initial co1 value": float(co1),
    "Does alpha need to be inverted? (0 presents No, 1 presents Yes)": alpha_requires_grad,
    "Does co1 need to be inverted? (0 presents No, 1 presents Yes)": co1_requires_grad,
    "weight of loss_label_M": sigma1,
    "weight of loss_initial_M": sigma2,
    "notes": "The input and the input normalization range are [-1,1]. The optimizer is Adam. The loss function is HuberLoss. The activation function is tanh().",
    "Main code name": "biofilm_hybrid_driven_NN_transfer.py",
    "Model code name": "models_hybrid_driven_NN_transfer.py",
    "The code used for saving is": "save_method_hybrid_NN_transfer.py",
    "The code path is": "/data1/yq1/biofilm_zgf/code/"
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
    biomass_data_one = np.concatenate((fracture_mass,biofilm_mass),axis = 0)
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
data_timeout = data_all[cut_time*N:13*N,:] 
print('The testing data includes following time points：',np.unique(data_timeout[:,2:3]))
print('The size of the testing data is：',data_timeout.shape)

print('----------------------------')
# the training data
data_train = data_all[0*N:cut_time*N,:]
print('The training data includes following time points：',np.unique(data_train[:,2:3]))
print('The size of the training data is：',data_train.shape)

np.save(datasave_path + 'less_train_data_03301.npy', data_train)
np.save(datasave_path + 'less_timeout_data_03301.npy', data_timeout)

# Data preprocessing
xmax, xmin = float(max_data_all[0]), 0.0 * 0.001
ymax, ymin = float(max_data_all[1]), 0.0 * 0.001
tmax, tmin = float(max_data_all[2]), 0.0 * 3 * 60
Mmax, Mmin = float(max_data_all[3]), 0.0

normalize_index_max_with_M = np.array([xmax, ymax, tmax, Mmax])
normalize_index_min_with_M = np.array([xmin, ymin, tmin, Mmin])

normalize_index_max = np.array([xmax, ymax, tmax])
normalize_index_min = np.array([xmin, ymin, tmin])

# Normalization
data_train = (data_train - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0
data_timeout = (data_all[12*N:13*N,:] - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0

# The training data is divided into a training set and a validation set
mask = (data_train[:,1:2]> ((0.0017 - ymin) * 2.0 / (ymax - ymin) - 1.0)).flatten()
data_train_cover_facture = data_train[mask]
data_train_without_facture = data_train[~mask]

idx_train_cover_facture = np.random.choice(data_train_cover_facture.shape[0], int(rate_train_cover_fracture*data_train_cover_facture.shape[0]), replace=False)
idx_train_without_facture = np.random.choice(data_train_without_facture.shape[0], int(rate_train_without_fracture*data_train_without_facture.shape[0]), replace=False)

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


# Boundary data
data_boundary_array = np.load(file_initial_boundary, allow_pickle=True)
max_boundary = np.max(data_boundary_array, axis = 0)
min_boundary = np.min(data_boundary_array, axis = 0)
print('The maximum value in the boundary data is：', max_boundary)
print('The minimum value in the boundary data is：', min_boundary)
data_boundary_norm = (data_boundary_array - normalize_index_min) * 2.0 / (normalize_index_max - normalize_index_min) - 1.0
np.random.shuffle(data_boundary_norm)
idx_train = np.random.choice(data_boundary_norm.shape[0], int(rate_initial_boundary*data_boundary_norm.shape[0]), replace=False)
data_boundary_norm_choose = data_boundary_norm[idx_train]
data_boundary_t = torch.from_numpy(data_boundary_norm_choose).float()
print('The size of the boundary data is：',data_boundary_t.shape[0])


# Initial biomass data
biofilm_initial = data_all[0*N:1*N]
max_biofilm_initial = np.max(biofilm_initial, axis = 0)
min_biofilm_initial = np.min(biofilm_initial, axis = 0)
print('The maximum value in the initial biomass data is：', max_biofilm_initial)
print('The minimum value in the initial biomass data is：', min_biofilm_initial)
biofilm_initial = (biofilm_initial - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0
idx_biofilm_initial = np.random.choice(biofilm_initial.shape[0], int(rate_initial_M * biofilm_initial.shape[0]), replace=False)
biofilm_initial_choose = biofilm_initial[idx_biofilm_initial]
np.random.shuffle(biofilm_initial_choose)
biofilm_initial_t = torch.from_numpy(biofilm_initial_choose).float()
print('The size of the initial biomass data is:', biofilm_initial_t.shape[0])



# Batch settings
dataset_train_label = TensorDataset(data_train_t)
dataloader_label = DataLoader(dataset_train_label, batch_size=batch_size_label, shuffle=False)

# Define the main model
model = FCResNet(input_dim, hidden_dim, output_dim, num_blocks_long, num_blocks_middle, num_blocks_short).to(device)
model = DataParallel(model)
# Load the pre-trained model
checkpoint = torch.load('/data1/yq1/biofilm_zgf/result/without_s_curve/hybrid_driven_new_parameters_using_yangxiaofan_new_NN_revise2_20251018(2)/parameters_best/parameters_best.pt')
model.load_state_dict(checkpoint['model_dict'])

# Determine the number of parameters (W，b) in the model
N_p = len(list(model.parameters()))
print('The number of (w,b) is :',N_p/2)

# Freezing specific layers of the neural network,">" for freezing of the shallow layer,"<"" for freezing of the deep layer
i = 0
for param in model.parameters():
    i = i + 1
    print(param)
    if (i//2) < 44:   # "44" represents the number of parameters (w, b), "44" can be modified
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the alternative model for the self-inhibition factor
model_D = NN_c().to(device)
model_D = DataParallel(model_D)
# Load the pre-trained model
checkpoint = torch.load('/data1/yq1/biofilm_zgf/result/without_s_curve/hybrid_driven_new_parameters_using_yangxiaofan_new_NN_revise2_20251018(2)/parameters_best/D.pt')
model_D.load_state_dict(checkpoint['model_dict'])

# Define the optimizer
params = [
            {'params': model.parameters()}
]
optimizer = torch.optim.Adam(params, lr=lr, weight_decay= weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# The loss function
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
    total_batches = len(dataloader_label)

    for (batch_idx1, data_label) in tqdm(
        enumerate(dataloader_label),
        total=total_batches,
        desc=f'Epoch {epoch}/{epoch_number}',
        leave=True):
        
        # Biomass
        data_label = data_label[0].to(device).requires_grad_(True)
        out_label = model(data_label[:,0:3])
        loss_label_M = huber_loss(data_label[:,3:4], out_label[:,4:5])
        
        # Initial biomass
        biofilm_initial = biofilm_initial_t.to(device).requires_grad_(True)
        out_initial = model(biofilm_initial[:,0:3])
        loss_initial_M = huber_loss(biofilm_initial[:,3:4], out_initial[:,4:5])

        # Total loss
        loss = sigma1 * loss_label_M + sigma2 * loss_initial_M
        optimizer.zero_grad()
        loss.backward()

        # Perform gradient clipping
        if epoch > epoch_change:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(model_D.parameters(), max_grad_norm)
        optimizer.step()

    scheduler.step()
    
    if epoch % 1 == 0 and epoch > epoch_change:
        time_elapsed = time.time() - start_time
        print('-------------------------------------------------------------------------------')
        print(f'''Epoch={epoch}
Time={time_elapsed}
Training loss (all) ={loss.item():.5f}
loss label(M)={loss_label_M.item():.5f}
loss initial(M)={loss_initial_M.item():.5f}
alpha = {alpha.item():.5f}
co1 = {co1.item():.5f}
''')

        # The validation set
        data_validate = data_validate_t.to(device).requires_grad_(True)
        out_validate = model(data_validate[:,0:3])
        loss_validate_M = huber_loss(data_validate[:,3:4], out_validate[:,4:5])

        # The extrapolation set
        data_timeout = data_timeout_t.to(device).requires_grad_(True)
        out_timeout = model(data_timeout[:,0:3])
        loss_timeout_M = huber_loss(data_timeout[:,3:4], out_timeout[:,4:5])

        R2_train = r2_score(data_label[:,3:4].cpu().detach().numpy(), out_label[:,4:5].cpu().detach().numpy())
        R2_validate = r2_score(data_validate[:,3:4].cpu().detach().numpy(), out_validate[:,4:5].cpu().detach().numpy())
        R2_timeout = r2_score(data_timeout[:,3:4].cpu().detach().numpy(), out_timeout[:,4:5].cpu().detach().numpy())
        print('The accuracy rate on the training set is：',R2_train)
        print('The accuracy rate on the validation set is：：',R2_validate)
        print('The accuracy rate on the extrapolation set is：',R2_timeout)

        print('-------------------------------------------------------------------------------')
        
        epoch_loss = [epoch,\
                      time_elapsed,\
                      alpha,\
                      loss.item(),\
                      loss_label_M.item(),\
                      loss_initial_M.item(),\
                      co1.item()
                        ]
        
        loss_total.append(epoch_loss)
        if R2_timeout > R2_best:
            R2_best = R2_timeout
            loss_Total = np.array(loss_total)
            save_training_artifacts(model,model_D,loss_Total, config, model_name="hybrid_driven_NN_transfer", best = True)
    else:
        time_elapsed = time.time() - start_time
        print('-------------------------------------------------------------------------------')
        print(f'''Epoch={epoch}
Time={time_elapsed}
Training loss (all) ={loss.item():.5f}
loss label(M)={loss_label_M.item():.5f}
loss initial(M)={loss_initial_M.item():.5f}
alpha = {alpha.item():.5f}
''')
        # The validation set
        data_validate = data_validate_t.to(device).requires_grad_(True)
        out_validate = model(data_validate[:,0:3])
        loss_validate_M = huber_loss(data_validate[:,3:4], out_validate[:,4:5])

        # The extrapolation set
        data_timeout = data_timeout_t.to(device).requires_grad_(True)
        out_timeout = model(data_timeout[:,0:3])
        loss_timeout_M = huber_loss(data_timeout[:,3:4], out_timeout[:,4:5])

        R2_train = r2_score(data_label[:,3:4].cpu().detach().numpy(), out_label[:,4:5].cpu().detach().numpy())
        R2_validate = r2_score(data_validate[:,3:4].cpu().detach().numpy(), out_validate[:,4:5].cpu().detach().numpy())
        R2_timeout = r2_score(data_timeout[:,3:4].cpu().detach().numpy(), out_timeout[:,4:5].cpu().detach().numpy())
        print('The accuracy rate on the training set is：',R2_train)
        print('The accuracy rate on the validation set is：：',R2_validate)
        print('The accuracy rate on the extrapolation set is：',R2_timeout)

        print('-------------------------------------------------------------------------------')
        
        epoch_loss = [epoch,\
                      time_elapsed,\
                      alpha,\
                      loss.item(),\
                      0.0,\
                      loss_label_M.item(),\
                      loss_initial_M.item(),\
                      0.0,\
                      0.0,\
                      0.0,\
                      0.0,\
                      0.0,\
                      0.0
                        ]
        loss_total.append(epoch_loss)
    # cache
    torch.cuda.empty_cache() 

loss_Total = np.array(loss_total)
save_training_artifacts(model,model_D,loss_Total, config, model_name = 'hybrid_driven_NN_transfer', best = False)