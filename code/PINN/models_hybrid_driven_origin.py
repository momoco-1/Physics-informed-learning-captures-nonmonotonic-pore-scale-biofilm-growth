import torch
import torch.nn as nn
import numpy as np
import math
from pyDOE import lhs
from scipy.stats import uniform, norm

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


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True,only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

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
                 output_dim           : int         = 5,
                 num_blocks_long      : int         = 16,
                 num_blocks_middle    : int         = 8,
                 num_blocks_short     : int         = 1,
                 block                              = FCResidualBlock):
        super(FCResNet, self).__init__()

        self.block = block
        self.hidden_dim = hidden_dim
        self.num_blocks_long = num_blocks_long
        self.num_blocks_middle = num_blocks_middle
        self.num_blocks_short = num_blocks_short

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
        self.net1 = self.add_blocks(self.num_blocks_long)
        self.net2 = self.add_blocks(self.num_blocks_middle)
        self.net3 = self.add_blocks(self.num_blocks_short)

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
            if length == self.num_blocks_long:
                blocks.append(self.block(self.hidden_dim))
            elif length == self.num_blocks_middle:
                blocks.append(self.block(self.hidden_dim))
            elif length == self.num_blocks_short:
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
    
class PDE_residual(nn.Module):
    """PDE Residual Calculation Class
    
    This type is used to calculate the PDE residuals during the biofilm growth, including:
    - Darcy-Brinkman-Stokes (DBS) equations
    - convection-diffusion-reaction (CDR) equation
    - modified diffusion-reaction (DR) equation (biofilm growth)
    
    Args:
        data_pde: Collocation points
        model: Main model
        alpha: Permeability-porosity relationship index
        co1: Adjustment coefficient b
        xmax, xmin: Maximum and minimum values in the x direction
        ymax, ymin: Maximum and minimum values in the y direction
        tmax, tmin: Maximum and minimum values of time
        umax, umin: Maximum and minimum values of the x-direction velocity
        vmax, vmin: Maximum and minimum values of the y-direction velocity
        pmax, pmin: Maximum and minimum pressure values
        cmax, cmin: Maximum and minimum concentrations
        Mmax, Mmin: Maximum and minimum values of biomass
    """
    def __init__(self, data_pde, model, alpha, co1,
                 xmax, xmin, ymax, ymin, tmax, tmin,
                 umax, umin, vmax, vmin, pmax, pmin,
                 cmax, cmin, Mmax, Mmin):
        super(PDE_residual, self).__init__()
        
        # Verify the input parameters
        self._validate_inputs(data_pde, model)
        
        self.data_pde = data_pde
        self.model = model
        self.alpha = alpha
        self.co1 = co1
        
        self.bounds = {
            'x': (xmax, xmin),
            'y': (ymax, ymin),
            't': (tmax, tmin),
            'u': (umax, umin),
            'v': (vmax, vmin),
            'p': (pmax, pmin),
            'c': (cmax, cmin),
            'M': (Mmax, Mmin)
        }
        
        # Set physical parameters
        self._set_physical_params()
        
        # Initialization calculation
        self._initialize_computation()
        
    def _validate_inputs(self, data_pde, model):
        """Verify the validity of the input parameters
        
        Args:
            data_pde: Collocation points
            model: Main model
        """
        if not isinstance(data_pde, torch.Tensor):
            raise TypeError("data_pde must be of torch.Tensor type !")
            
        if not isinstance(model, nn.Module):
            raise TypeError("The model must be of the nn.Module type !")
            
    
    def _set_physical_params(self):
        """Set physical parameters"""
        # The permeability of the free channels
        self.kh = 2.075e-10
        # Fluid density
        self.rho = 993.3
        # Biomembrane thickness (The thickness of the microfluidic chip)
        self.h = 5e-5
        # Fluid viscosity
        self.viscosity = 7e-4
        # Molecular diffusion coefficient
        self.D = 3.3e-9
        # half-saturation constant
        self.ks = 0.285
        # Maximum specific growth rate of bacteria
        self.mium = 1.15e-4
        # cell yield coefficient
        self.Yxs = 0.32
        # maintenance coefficient
        self.ms = 3e-5
    
    def _initialize_computation(self):
        """Initializing the calculation processes"""
        [self.x, self.y, self.t, self.judge] = torch.split(self.data_pde, 1, dim=1)
        
        self.data_inp = torch.cat([self.x, self.y, self.t], dim=1)
        
        out_pde = self.model(self.data_inp)
        
        [self.u, self.v, self.p, self.c, self.M] = torch.split(out_pde, 1, dim=1)

        self.M = torch.clamp(self.M, -1.0, 1.0)
        self.c = torch.clamp(self.c, -1.0, 1.0)
        self.u = torch.clamp(self.u, -1.0, 1.0)
        self.v = torch.clamp(self.v, -1.0, 1.0)
        self.p = torch.clamp(self.p, -1.0, 1.0)

        # Calculate all gradients
        self._compute_gradients()
        
        # Calculate porosity and permeability
        self._compute_porosity_and_permeability()
        
        # Calculate the growth kinetics parameters
        self._compute_growth_kinetics()
        
    def _compute_gradients(self):
        """Calculate the gradients of all field variables and perform gradient clipping to enhance numerical stability."""
        # Velocity field u
        self.u_x = self._safe_gradient(self.u, self.x)
        self.u_y = self._safe_gradient(self.u, self.y)
        self.u_t = self._safe_gradient(self.u, self.t)
        self.u_xx = self._safe_gradient(self.u_x, self.x)
        self.u_yy = self._safe_gradient(self.u_y, self.y)
        
        # Velocity field v
        self.v_x = self._safe_gradient(self.v, self.x)
        self.v_y = self._safe_gradient(self.v, self.y)
        self.v_t = self._safe_gradient(self.v, self.t)
        self.v_xx = self._safe_gradient(self.v_x, self.x)
        self.v_yy = self._safe_gradient(self.v_y, self.y)
        
        # Pressure field
        self.p_x = self._safe_gradient(self.p, self.x)
        self.p_y = self._safe_gradient(self.p, self.y)
        
        # Concentration field
        self.c_x = self._safe_gradient(self.c, self.x)
        self.c_y = self._safe_gradient(self.c, self.y)
        self.c_t = self._safe_gradient(self.c, self.t)
        self.c_xx = self._safe_gradient(self.c_x, self.x)
        self.c_yy = self._safe_gradient(self.c_y, self.y)
        
        # Biomass field
        self.M_x = self._safe_gradient(self.M, self.x)
        self.M_y = self._safe_gradient(self.M, self.y)
        self.M_t = self._safe_gradient(self.M, self.t)
    
    def _safe_gradient(self, y, x, clip_value=1e6):
        """Safe gradient calculation, including gradient clipping
        Args:
            y: Dependent variable
            x: Independent variable
            clip_value: Gradient clipping threshold
        """
        grad = gradients(y, x)
        return torch.clamp(grad, -clip_value, clip_value)
        
    def _denormalize(self, x, bounds_key):
        """Convert the normalized values back to physical values
        Args:
            x: Normalized value
            bounds_key: The key names in the bounds dictionary
        """
        xmax, xmin = self.bounds[bounds_key]
        return (x + 1) * (xmax - xmin) / 2.0 + xmin
    
    def _compute_porosity_and_permeability(self):
        """Calculate porosity and permeability"""
        self.M_real = self._denormalize(self.M,'M')
        
        # Initialize the porosity tensor
        self.n = torch.ones_like(self.M).float()
        
        # Calculate porosity based on the regional markers
        mask0 = (self.judge != 1).squeeze()  # 0 represents the biofilm area
        
        # Calculate porosity
        biomass_porosity = (21.95 * (self.M_real[mask0]) / 5.0) ** (-0.45)

        self.n[mask0] = torch.clamp(biomass_porosity, 0.19, 1.0)  # Ensure that the porosity is within the range of [0, 1]
        
        # Calculate permeability
        self.kf = self.kh * torch.pow(self.n, self.alpha)
        self.kf = torch.clamp(self.kf, min=1e-20)  # Ensure that the permeability is positive
            
    def _compute_growth_kinetics(self):
        """Calculate the growth kinetics parameters"""
        self.c_real = self._denormalize(self.c, 'c')

        # self-inhibition factor
        self.pw = torch.zeros_like(self.M_real).float()
        mask0 = ((self.M_real > 0.296) & (self.M_real < 3.2)).squeeze()
        mask1 = (self.M_real >= 3.2).squeeze()
        
        self.pw[mask0] = (1/(-0.6)) * (1.0 - (self.M_real[mask0] / 3.2)**(-0.6))
        self.pw[mask1] = 0.01

        # Nutrient consumption item
        self.K =  3.894e-4 * self.c_real * self.M_real / (0.285 + self.c_real)

        # Biofilm growth term
        self.K_bio = self.K * self.Yxs * self.pw * 4.0

        # Microbial consumption items
        self.Ds = self.Yxs * self.ms * self.M_real

        # Nutrient diffusion coefficient
        self.D_c_effective = self.D * (self.n ** (4/3))

        # Biofilm diffusion coefficient
        self.Db_biofilm = self.co1 * 1e-13 * self.pw * self.M_real * (self.c_real / self.bounds['c'][0])**6

    def dbs_residual(self):
        """Calculate the residual of the Darcy-Brinkman-Stokes equation"""
        u_real = self._denormalize(self.u, 'u')
        v_real = self._denormalize(self.v, 'v')
        
        # Residual of the x-direction momentum equation
        eq_dbs_x = (
            # Time derivative term
            self.u_t * (self.bounds['u'][0] - self.bounds['u'][1]) / (self.bounds['t'][0] - self.bounds['t'][1]) +
            
            # Convection term
            u_real * self.u_x * (self.bounds['u'][0] - self.bounds['u'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]) / self.n +
            v_real * self.u_y * (self.bounds['u'][0] - self.bounds['u'][1]) / (self.bounds['y'][0] - self.bounds['y'][1]) / self.n +
            
            # Pressure gradient term
            self.p_x * self.n * (self.bounds['p'][0] - self.bounds['p'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]) / self.rho -
            
            # Viscous term
            self.viscosity * (
                self.u_xx * 2.0 * (self.bounds['u'][0] - self.bounds['u'][1]) / ((self.bounds['x'][0] - self.bounds['x'][1])**2) +
                self.u_yy * 2.0 * (self.bounds['u'][0] - self.bounds['u'][1]) / ((self.bounds['y'][0] - self.bounds['y'][1])**2)
            ) / self.rho +
            
            # Darcy resistance term
            self.viscosity * self.n * u_real / self.kf / self.rho
        )
        
        # Residual of the y-direction momentum equation
        eq_dbs_y = (
            self.v_t * (self.bounds['v'][0] - self.bounds['v'][1]) / (self.bounds['t'][0] - self.bounds['t'][1]) +
            
            u_real * self.v_x * (self.bounds['v'][0] - self.bounds['v'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]) / self.n +
            v_real * self.v_y * (self.bounds['v'][0] - self.bounds['v'][1]) / (self.bounds['y'][0] - self.bounds['y'][1]) / self.n +
            
            self.p_y * self.n * (self.bounds['p'][0] - self.bounds['p'][1]) / (self.bounds['y'][0] - self.bounds['y'][1]) / self.rho -
            
            self.viscosity * (
                self.v_xx * 2.0 * (self.bounds['v'][0] - self.bounds['v'][1]) / ((self.bounds['x'][0] - self.bounds['x'][1])**2) +
                self.v_yy * 2.0 * (self.bounds['v'][0] - self.bounds['v'][1]) / ((self.bounds['y'][0] - self.bounds['y'][1])**2)
            ) / self.rho +

            self.viscosity * self.n * v_real / self.kf / self.rho
        )
        
        # Residual of the continuity equation
        eq_con = (
            self.u_x * (self.bounds['u'][0] - self.bounds['u'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]) +
            self.v_y * (self.bounds['v'][0] - self.bounds['v'][1]) / (self.bounds['y'][0] - self.bounds['y'][1])
        )

        return [eq_dbs_x, eq_dbs_y, eq_con]
    
    def cde_residual(self):
        """Calculate the residual of the convection-diffusion-reaction equation"""

        u_real = self._denormalize(self.u, 'u')
        v_real = self._denormalize(self.v, 'v')

        c_xx = self._safe_gradient(
            self.c_x * (self.bounds['c'][0] - self.bounds['c'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]),
            self.x
        )
        c_yy = self._safe_gradient(
            self.c_y * (self.bounds['c'][0] - self.bounds['c'][1]) / (self.bounds['y'][0] - self.bounds['y'][1]),
            self.y
        )
        
        eq_cde = (
            # Time derivative term
            self.n * self.c_t * (self.bounds['c'][0] - self.bounds['c'][1]) / (self.bounds['t'][0] - self.bounds['t'][1]) +
            
            # Convection term
            u_real * self.c_x * (self.bounds['c'][0] - self.bounds['c'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]) +
            v_real * self.c_y * (self.bounds['c'][0] - self.bounds['c'][1]) / (self.bounds['y'][0] - self.bounds['y'][1]) -
            
            # Diffusion term
            (
                c_xx * 2.0 / (self.bounds['x'][0] - self.bounds['x'][1]) +
                c_yy * 2.0 / (self.bounds['y'][0] - self.bounds['y'][1])
            ) * self.D_c_effective +
            
            # Nutrient consumption/reaction item
            self.K
        )

        return 10*eq_cde   
    
    def biofilm_growth(self,):
        """Calculate the residuals of the biofilm growth equation"""

        M_xx = self._safe_gradient(
            self.M_x * (self.bounds['M'][0] - self.bounds['M'][1]) / (self.bounds['x'][0] - self.bounds['x'][1]),
            self.x
        )
        M_yy = self._safe_gradient(
            self.M_y * (self.bounds['M'][0] - self.bounds['M'][1]) / (self.bounds['y'][0] - self.bounds['y'][1]),
            self.y
        )
        
        eq_biofilm = (
            # Time derivative term
            self.M_t * (self.bounds['M'][0] - self.bounds['M'][1]) / (self.bounds['t'][0] - self.bounds['t'][1]) -
            
            # Diffusion term
            (
            M_xx * 2.0 / (self.bounds['x'][0] - self.bounds['x'][1]) +
            M_yy * 2.0 / (self.bounds['y'][0] - self.bounds['y'][1])
            ) * self.Db_biofilm -
            
            # Biofilm growth term
            self.K_bio + 
            
            # Biofilm death term
            self.Ds
        )

        return 100*eq_biofilm, self.pw, self.M_real, self.Db_biofilm