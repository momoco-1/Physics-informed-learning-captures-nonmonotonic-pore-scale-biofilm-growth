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
from models_hybrid_driven_origin import FCResNet, PDE_residual, read_matrix_data, HuberLoss
from save_method_hybrid_origin import save_training_artifacts

random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float)

# cuda support?
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
batch_size_label = int(12800 / 4) # Corresponding to labeled points
batch_size_pde = int(12800 / 4)  # Corresponding to collocation points(unlabeled data)

# Define the permeability conversion parameter
alpha = torch.tensor(2.748).to(device)
alpha.requires_grad = False
if alpha.requires_grad:
    alpha_requires_grad = 1.0
else:
    alpha_requires_grad = 0.0

# Define the adjustment parameter b
co1 = torch.tensor(5.0).to(device)
co1.requires_grad = False
if co1.requires_grad:
    co1_requires_grad = 1.0
else:
    co1_requires_grad = 0.0

# Optimizer settings
lr = 8e-4   # Learning rate
weight_decay = 0.00002   # Weight of L2 regularization
step_size = 20
gamma = 0.9  # Attenuation rate

# Number of iterations
epoch_number = 100

# The iteration threshold for transitioning from pure data-driven training to PINN
epoch_change = 0

# Setting of loss weights
## The pure data-driven part
#scale_p = 0.0    # PINN: only M0
sigma1 = 80.0
sigma2 = 20.0
sigma3 = 10.0
sigma4 = 1.0
sigma5 = 1.0
sigma6 = 1.0
sigma7 = 1.0
sigma8 = 1.0
sigma9 = 1.0
## The physics-driven part
sigma10 = 0.001
sigma11 = 1.0
sigma12 = 2.0
sigma13 = 10.0

# Data division
cut_time = 16  # The time point at which the training data and the extrapolation data (testing data) are divided
rate_train_cover_fracture = 0.1  # Sampling ratio within the sampling space that includes free channels
rate_train_without_fracture = 0.04  # Sampling ratio within the sampling space that does not include free channels
rate_initial_M = 0.5  # Sampling ratio of initial biomass
rate_initial_c = 0.5  # Sampling ratio of initial nutrient concentration
rate_initial_velocity = 0.5   
rate_initial_boundary = 0.35  # Boundary condition

num_biofilm_high = 3330  # The number of collocation points within the high-resolution biofilm regions
num_fracture_high = 100  # The number of collocation points within the high-resolution free channel regions

# The address of data
# Biomass data (Corresponding to biofilm regions; Used for sampling of labeled points)
file_path = "/data2/yq2/biofilm_zgf/data/biofilm_data_scale2.txt"
# Biomass data (Corresponding to biofilm regions; Used for sampling of collocation points)
file_biofilm_high_resolution = "/data2/yq2/biofilm_zgf/data/biofilm_data.txt"
# Biomass data (Corresponding to free channel regions; Used for sampling of labeled points)
file_path2 = "/data2/yq2/biofilm_zgf/data/fracture_data_scale2.txt"
# Biomass data (Corresponding to free channel regions; Used for sampling of collocation points)
file_fracture_high_resolution = "/data2/yq2/biofilm_zgf/data/fracture_data.txt"

# Initial velocity
file_initial_velocity = '/data2/yq2/biofilm_zgf2/data/initial_uv.npy'
# Initial nutrient concentration
file_initial_concentration = '/data2/yq2/biofilm_zgf2/data/initial_c.npy'
# Boundary
file_initial_boundary = '/data2/yq2/biofilm_zgf/data/data_boundary_less.npy'
# Inlet
file_initial_inlet = '/data2/yq2/biofilm_zgf/data/data_inlet.npy'
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
    "The usage ratio of the initial concentration data points": rate_initial_c,
    "The usage ratio of the initial velocity data points": rate_initial_velocity,
    "The usage ratio of the initial boundary data points": rate_initial_boundary,
    "batch size of label data": batch_size_label,
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "The learning rate changes the stride": step_size,
    "Learning rate change ratio": gamma,
    "num_epochs": epoch_number,
    "The epoch threshold between pure data-driven transitions and PINNs": epoch_change,
    "Initial alpha value": float(alpha),
    "Initial co1 value": float(co1),
    "Does alpha need to be inverted? (0 presents No, 1 presents Yes)": alpha_requires_grad,
    "Does co1 need to be inverted? (0 presents No, 1 presents Yes)": co1_requires_grad,
    "weight of loss_label_M": sigma1,
    "weight of loss_initial_M": sigma2,
    "weight of loss_initial_c": sigma3,
    "weight of loss_initial_U": sigma4,
    "weight of loss_boundary_U": sigma5,
    "weight of loss_inlet_U": sigma6,
    "weight of loss_inlet_p": sigma7,
    "weight of loss_inlet_c": sigma8,
    "weight of loss_outlet_p": sigma9,
    "weights of dbs_x and dbs_y": sigma10,
    "weight of continue equation": sigma11,
    "weight of CDE": sigma12,
    "weight of biomass growth equation": sigma13,
    "Whether to perform parallel computing": "Yes",
    "notes": "The input and the input normalization range are [-1,1]. The optimizer is Adam. The loss function is HuberLoss. The activation function is tanh().",
    "Main code name": "biofilm_hybrid_driven_origin.py",
    "Model code name": "models_hybrid_driven_origin.py",
    "The code used for saving is": "save_method_hybrid_origin.py",
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
data_timeout = data_all[cut_time*N:24*N,:] 
print('The testing data includes following time points：',np.unique(data_timeout[:,2:3]))
print('The size of the testing data is：',data_timeout.shape)

print('----------------------------')
# the training data
data_train = data_all[0*N:cut_time*N,:]
print('The training data includes following time points:：',np.unique(data_train[:,2:3]))
print('The size of the training data is：',data_train.shape)

np.save(datasave_path + 'training_data.npy', data_train)
np.save(datasave_path + 'testing_data.npy', data_timeout)

# Initial velocity
data_initial_velocity = np.load(file_initial_velocity, allow_pickle=True)

# Data preprocessing
xmax, xmin = float(max_data_all[0]), 0.0 * 0.001
ymax, ymin = float(max_data_all[1]), 0.0 * 0.001
tmax, tmin = float(max_data_all[2]), 0.0 * 3 * 60
Mmax, Mmin = float(max_data_all[3]), 0.0
pmax, pmin = 2005.0, 0.0
umax, umin = float(np.max(data_initial_velocity[:,3:4])), float(np.min(data_initial_velocity[:,3:4]))
vmax, vmin = float(np.max(data_initial_velocity[:,4:5])), float(np.min(data_initial_velocity[:,4:5]))
cmax, cmin = 10.0, 0.0
nmax, nmin = 0.6, 0.2

normalize_index_max_with_M = np.array([xmax, ymax, tmax, Mmax])
normalize_index_min_with_M = np.array([xmin, ymin, tmin, Mmin])

normalize_index_max_with_p = np.array([xmax, ymax, tmax, pmax])
normalize_index_min_with_p = np.array([xmin, ymin, tmin, pmin])

normalize_index_max_with_c = np.array([xmax, ymax, tmax, cmax])
normalize_index_min_with_c = np.array([xmin, ymin, tmin, cmin])

normalize_index_max_with_uv = np.array([xmax, ymax, tmax, umax, vmax])
normalize_index_min_with_uv = np.array([xmin, ymin, tmin, umin, vmin])

normalize_index_max = np.array([xmax, ymax, tmax])
normalize_index_min = np.array([xmin, ymin, tmin])

# Normalization
data_train = (data_train - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0
data_timeout = (data_all[23*N:24*N,:] - normalize_index_min_with_M) * 2.0 / (normalize_index_max_with_M - normalize_index_min_with_M) - 1.0

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


# Randomly sampling the collocation points(unlabeled data)
# In biofilm regions:
biofilm_high_resolution_set = []
biofilm_high_resolution_all = read_matrix_data(file_biofilm_high_resolution)
for i in range(number_img):
    idx_pde_biofilm = np.random.choice(biofilm_high_resolution_all[i].shape[0], num_biofilm_high, replace=False)
    biofilm_high_resolution_choose = biofilm_high_resolution_all[i][idx_pde_biofilm]
    # Set the marker for biofilm regions to 0
    biofilm_high_resolution_set.append(np.concatenate((biofilm_high_resolution_choose[:,0:3],np.zeros_like(biofilm_high_resolution_choose[:,0:1])), axis = 1))

# In free channel regions:
fracture_high_resolution_set = []
fracture_high_resolution_all = read_matrix_data(file_fracture_high_resolution)
for i in range(number_img):
    idx_pde_fracture = np.random.choice(fracture_high_resolution_all[i].shape[0], num_fracture_high, replace=False)
    fracture_high_resolution_choose = fracture_high_resolution_all[i][idx_pde_fracture]
    # Set the marker for free channel regions to 1
    fracture_high_resolution_set.append(np.concatenate((fracture_high_resolution_choose[:,0:3],np.ones_like(fracture_high_resolution_choose[:,0:1])), axis = 1))

high_resolution_set = []
for i in range(number_img):
    biofilm_h_pde = biofilm_high_resolution_set[i]
    fracture_h_pde = fracture_high_resolution_set[i]
    h_pde_one = np.concatenate((biofilm_h_pde,fracture_h_pde),axis = 0)
    high_resolution_set.append(h_pde_one)

print('----------------------------------------------------')
#for i in range(24):
#    print(high_resolution_set[i].shape,i,np.unique(high_resolution_set[i][:,2:3]),np.unique(high_resolution_set[i][:,3:4]))

transformed_samples = np.array(high_resolution_set).reshape(-1,4)

max_pde_choose = np.max(transformed_samples, axis=0)
min_pde_choose = np.min(transformed_samples, axis=0)
print('The maximum value in the collocation points:',max_pde_choose)
print('The minimum value in the collocation points:',min_pde_choose)
print('----------------------------------------------------')

transformed_samples[:,0:3] = (transformed_samples[:,0:3] - normalize_index_min) * 2.0 / (normalize_index_max - normalize_index_min) - 1.0 
np.random.shuffle(transformed_samples)
data_pde_t = torch.from_numpy(transformed_samples).float()
print('The size of the collocation points is：',data_pde_t.shape[0])


# Initial concentration data
data_initial_c = np.load(file_initial_concentration, allow_pickle=True)
max_initial_c = np.max(data_initial_c, axis = 0)
min_initial_c = np.min(data_initial_c, axis = 0)
print('The maximum value in the initial concentration data is：', max_initial_c)
print('The minimum value in the initial concentration data is：', min_initial_c)
data_initial_c_norm = (data_initial_c - normalize_index_min_with_c) * 2.0 / (normalize_index_max_with_c - normalize_index_min_with_c) - 1.0
# Select a portion of the data
idx_data_initial_c_norm = np.random.choice(data_initial_c_norm.shape[0], int(rate_initial_c*data_initial_c_norm.shape[0]), replace=False)
data_initial_c_choose = data_initial_c_norm[idx_data_initial_c_norm]
np.random.shuffle(data_initial_c_choose)
data_initial_c_t = torch.from_numpy(data_initial_c_choose).float()
print('The size of the the initial concentration data：',data_initial_c_t.shape[0])


# Initial velocity data
max_initial_velocity = np.max(data_initial_velocity, axis = 0)
min_initial_velocity = np.min(data_initial_velocity, axis = 0)
print('The maximum value in the initial velocity data is：', max_initial_velocity)
print('The minimum value in the initial velocity data is：', min_initial_velocity)
data_initial_velocity_norm = (data_initial_velocity - normalize_index_min_with_uv) * 2.0 / (normalize_index_max_with_uv - normalize_index_min_with_uv) - 1.0
idx_data_initial_velocity_norm = np.random.choice(data_initial_velocity_norm.shape[0], int(rate_initial_velocity*data_initial_velocity_norm.shape[0]), replace=False)
data_initial_velocity_choose = data_initial_velocity_norm[idx_data_initial_velocity_norm]
np.random.shuffle(data_initial_velocity_choose)
data_initial_velocity_t = torch.from_numpy(data_initial_velocity_choose).float()
print('The size of the initial velocity data is：',data_initial_velocity_t.shape[0])



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



# Inlet/outlet data
data_inlet_array = np.load(file_initial_inlet, allow_pickle=True)
data_outlet_array = np.load(file_initial_outlet, allow_pickle=True)
max_inlet = np.max(data_inlet_array, axis = 0)
min_inlet = np.min(data_inlet_array, axis = 0)
print('The maximum value in the inlet data is：', max_inlet)
print('The minimum value in the inlet data is：', min_inlet)
max_outlet = np.max(data_outlet_array, axis = 0)
min_outlet = np.min(data_outlet_array, axis = 0)
print('The maximum value in the outlet data is：', max_outlet)
print('The minimum value in the outlet data is：', min_outlet)
data_inlet_norm = (data_inlet_array - normalize_index_min_with_p) * 2.0 / (normalize_index_max_with_p - normalize_index_min_with_p) - 1.0
data_outlet_norm = (data_outlet_array - normalize_index_min_with_p) * 2.0 / (normalize_index_max_with_p - normalize_index_min_with_p) - 1.0
np.random.shuffle(data_inlet_norm)
np.random.shuffle(data_outlet_norm)
data_inlet_t = torch.from_numpy(data_inlet_norm).float()
data_outlet_t = torch.from_numpy(data_outlet_norm).float()
print('The size of the inlet data is：',data_inlet_t.shape[0])
print('The size of the outlet data is：',data_outlet_t.shape[0])



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
dataset_train_pde = TensorDataset(data_pde_t)

dataloader_label = DataLoader(dataset_train_label, batch_size=batch_size_label, shuffle=False)
dataloader_pde = DataLoader(dataset_train_pde, batch_size=batch_size_pde, shuffle=False)

# Define the main model
model = FCResNet(input_dim, hidden_dim, output_dim, num_blocks_long, num_blocks_middle, num_blocks_short).to(device)
model = DataParallel(model)


# Define the optimizer
params = [
            {'params': model.parameters()},
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
    total_batches = min(len(dataloader_label), len(dataloader_pde))

    for (batch_idx1, data_label), (batch_idx2, data_pde) in tqdm(
        zip(enumerate(dataloader_label), enumerate(dataloader_pde)),
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

        # Initial Concentration
        data_initial_c = data_initial_c_t.to(device).requires_grad_(True)
        out_initial_c = model(data_initial_c[:,0:3])
        loss_initial_c = huber_loss(data_initial_c[:,3:4], out_initial_c[:,3:4])

        # Initial velocity
        data_initial_velocity = data_initial_velocity_t.to(device).requires_grad_(True)
        [x, y, t, u, v] = torch.split(data_initial_velocity, 1, dim=1)  
        data_xyt_velocity = torch.cat([x, y, t], dim=1)  
        out_initial_velocity = model(data_xyt_velocity)
        out_initial_u = out_initial_velocity[:,0:1]
        out_initial_v = out_initial_velocity[:,1:2]
        loss_initial_u = huber_loss(u, out_initial_u)
        loss_initial_v = huber_loss(v, out_initial_v)
        loss_initial_U = loss_initial_u + loss_initial_v
        
        # Boundary
        data_boundary = data_boundary_t.to(device).requires_grad_(True)
        out_boundary =  model(data_boundary)
        out_boundary_u = out_boundary[:,0:1]
        out_boundary_v = out_boundary[:,1:2]
        loss_boundary_u = huber_loss(((torch.zeros_like(out_boundary_u) - umin) * 2.0 / (umax - umin) - 1.0), out_boundary_u)
        loss_boundary_v = huber_loss(((torch.zeros_like(out_boundary_v) - vmin) * 2.0 / (vmax - vmin) - 1.0), out_boundary_v)
        loss_boundary_U = loss_boundary_u + loss_boundary_v
        
        # Inlet
        data_inlet = data_inlet_t.to(device).requires_grad_(True)
        out_inlet =  model(data_inlet[:,0:3])
        loss_inlet_p = huber_loss(data_inlet[:,3:4], out_inlet[:,2:3])
        loss_inlet_c = huber_loss(torch.ones_like(out_inlet[:,0:1]), out_inlet[:,3:4])
        out_inlet_u = out_inlet[:,0:1]
        out_inlet_v = out_inlet[:,1:2]
        out_inlet_U = torch.sqrt(((out_inlet_u + 1.0) * (umax - umin) / 2.0 + umin) ** 2.0 + ((out_inlet_v + 1.0) * (umax - umin) / 2.0 + umin) ** 2.0)
        loss_inlet_U = huber_loss(torch.ones_like(out_inlet_U) * 3.93e-5, out_inlet_U)

        # Outlet
        data_outlet = data_outlet_t.to(device).requires_grad_(True)
        out_outlet =  model(data_outlet[:,0:3])
        loss_outlet_p = huber_loss(data_outlet[:,3:4], out_outlet[:,2:3])

        # data-driven
        loss_data_driven = sigma1 * loss_label_M + sigma2 * loss_initial_M + sigma3 * loss_initial_c + sigma4 * loss_initial_U + sigma5 * loss_boundary_U + sigma6 * loss_inlet_U + sigma7 * loss_inlet_p + sigma8 * loss_inlet_c  + sigma9 * loss_outlet_p  
        
        # Total loss
        if epoch <= epoch_change:
            loss = loss_data_driven
        else:
            # PDE constraints
            data_pde = data_pde[0].to(device).requires_grad_(True)
            pde_residual = PDE_residual(data_pde, model,
                                        alpha,co1,
                                        xmax, xmin,
                                        ymax, ymin,
                                        tmax, tmin,
                                        umax, umin,
                                        vmax, vmin,
                                        pmax, pmin,
                                        cmax, cmin,
                                        Mmax, Mmin)
    
            eq_dbs = pde_residual.dbs_residual()
            eq_dbs_x = eq_dbs[0]
            eq_dbs_y = eq_dbs[1]
            eq_con = eq_dbs[2]

            eq_cde = pde_residual.cde_residual()

            eq_biofilm, pw, M_real, Db = pde_residual.biofilm_growth()
        
            loss_pde1 = huber_loss(torch.zeros_like(eq_dbs_x), eq_dbs_x)  

            loss_pde2 = huber_loss(torch.zeros_like(eq_dbs_y), eq_dbs_y)

            loss_pde3 = huber_loss(torch.zeros_like(eq_con), eq_con)

            loss_pde4 = huber_loss(torch.zeros_like(eq_cde), eq_cde)

            loss_pde5 = huber_loss(torch.zeros_like(eq_biofilm), eq_biofilm)

            loss_pde = sigma10 * (loss_pde1 + loss_pde2) + sigma11 * loss_pde3 + sigma12 * loss_pde4 + sigma13 * loss_pde5  
            
            #loss = loss_data_driven * scale_p + loss_initial_M + loss_pde  # PINN: only M0
            loss = loss_data_driven + loss_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    
    if epoch % 1 == 0 and epoch > epoch_change:
        time_elapsed = time.time() - start_time
        print('-------------------------------------------------------------------------------')
        print(f'''Epoch={epoch}
Time={time_elapsed}
Training loss (all) ={loss.item():.5f}
Training loss (data_driven) ={loss_data_driven.item():.5f}
Training loss (pde_restraint) ={loss_pde.item():.5f}
loss label(M)={loss_label_M.item():.5f}
loss initial(M)={loss_initial_M.item():.5f}
loss initial(c)={loss_initial_c.item():.5f}
loss initial(U)={loss_initial_U.item():.5f}
loss boundary(U)={loss_boundary_U.item():.5f}
loss inlet(p) ={loss_inlet_p.item():.5f}
loss inlet(c)={loss_inlet_c.item():.5f}
loss inlet(U)={loss_inlet_U.item():.5f}
loss outlet(p)={loss_outlet_p.item():.5f}
loss dbs_x = {loss_pde1.item():.5f}
loss dbs_y = {loss_pde2.item():.5f}
loss con = {loss_pde3.item():.5f}
loss cde = {loss_pde4.item():.5f}
loss biofilm_growth = {loss_pde5.item():.5f}
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
                      loss_data_driven.item(),\
                      loss_pde.item(),\
                      loss_label_M.item(),\
                      loss_initial_M.item(),\
                      #loss_choose_M.item(),\
                      loss_initial_c.item(),\
                      loss_initial_U.item(),\
                      loss_boundary_U.item(),\
                      loss_inlet_p.item(),\
                      loss_inlet_c.item(),\
                      loss_inlet_U.item(),\
                      #loss_inlet_uv.item(),\
                      loss_outlet_p.item(),\
                      loss_pde1.item(),\
                      loss_pde2.item(),\
                      loss_pde3.item(),\
                      loss_pde4.item(),\
                      loss_pde5.item(),\
                      co1.item()
                        ]
        
        loss_total.append(epoch_loss)
        if R2_timeout > R2_best:
            R2_best = R2_timeout
            loss_Total = np.array(loss_total)
            save_training_artifacts(model,loss_Total, config, model_name="hybrid_driven_parameters_origin", best = True)
    else:
        time_elapsed = time.time() - start_time
        print('-------------------------------------------------------------------------------')
        print(f'''Epoch={epoch}
Time={time_elapsed}
Training loss (all) ={loss.item():.5f}
Training loss (data_driven) ={loss_data_driven.item():.5f}
loss label(M)={loss_label_M.item():.5f}
loss initial(M)={loss_initial_M.item():.5f}
loss initial(c)={loss_initial_c.item():.5f}
loss initial(U)={loss_initial_U.item():.5f}
loss boundary(U)={loss_boundary_U.item():.5f}
loss inlet(p) ={loss_inlet_p.item():.5f}
loss inlet(c)={loss_inlet_c.item():.5f}
loss inlet(U)={loss_inlet_U.item():.5f}
loss outlet(p)={loss_outlet_p.item():.5f}
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
                      loss_data_driven.item(),\
                      0.0,\
                      loss_label_M.item(),\
                      loss_initial_M.item(),\
                      #loss_choose_M.item(),\
                      loss_initial_c.item(),\
                      loss_initial_U.item(),\
                      loss_boundary_U.item(),\
                      loss_inlet_p.item(),\
                      loss_inlet_c.item(),\
                      loss_inlet_U.item(),\
                      #loss_inlet_uv.item(),\
                      loss_outlet_p.item(),\
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
save_training_artifacts(model,loss_Total, config, model_name = 'hybrid_driven_parameters_origin', best = False)