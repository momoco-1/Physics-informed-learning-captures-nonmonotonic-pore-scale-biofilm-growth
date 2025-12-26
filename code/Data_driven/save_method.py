import torch
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

def save_training_artifacts(model,
                            loss_Total,
                            config,
                            model_name="data_driven",
                            best = False):
    """
    General Training Information Storage Method
    
    Parameters:
        model: The trained model
        loss_Total: Training loss
        config: Training configuration dictionary
        best: Whether to save the best model during the training process. False indicates saving the model after the final training is completed, while True indicates saving the best model during the training process.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    save_dir = f"/public/home/yq/yq3_data2/biofilm_zgf2/results/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    if best == False:    
        # 1. Save the network parameters of the training model
        save_parameters = save_dir+'/parameters/'
        if not os.path.exists(save_parameters):
            os.makedirs(save_parameters)
        if model_name == "data_driven":
            torch.save({'model_dict':model.state_dict()},save_parameters +'parameters.pt')
        elif model_name == "hybrid_driven":
            torch.save({'model_dict':model.state_dict()},save_parameters +'parameters.pt')

        # 2. Save the loss
        save_loss = save_dir+'/loss/'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)
        np.savetxt(save_loss+'loss.txt',loss_Total)
    
        # 3. Save configuration
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
        print(f"The training information has been saved to: {save_dir}")
    
    elif best == True:
        # 1. Save the best model parameters
        save_parameters = save_dir+'/parameters_best/'
        if not os.path.exists(save_parameters):
            os.makedirs(save_parameters)
        if model_name == "data_driven":
            torch.save({'model_dict':model.state_dict()},save_parameters +'parameters_best.pt')
        elif model_name == "hybrid_driven":
            torch.save({'model_dict':model.state_dict()},save_parameters +'parameters.pt')

        # 2. Save the optimal model loss
        save_loss = save_dir+'/loss_best/'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)
        np.savetxt(save_loss+'loss_best.txt',loss_Total)

        print(f"The training information has been saved to: {save_dir}")