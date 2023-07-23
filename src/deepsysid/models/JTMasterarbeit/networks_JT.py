import copy
import json
import logging
import time
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import functional as F
from .. import utils
from ...networks import loss, rnn
from ...networks.rnn import HiddenStateForwardModule
from ...networks.rnn import LtiRnnConvConstr
from .. import base, utils
from ..base import DynamicIdentificationModelConfig
from ..datasets import RecurrentHybridPredictorDataset,RecurrentInitializerDataset, RecurrentPredictorDataset,FixedWindowDataset
import cvxpy as cp
import torch.jit as jit


logger = logging.getLogger(__name__)

#TODO: make all of them jit
class InputNet(jit.ScriptModule):
# class InputNet(nn.Module):
    def __init__(self, dropout: float):
        super(InputNet, self).__init__()
        self.fc1 = nn.Linear(6, 32)  # 6 input features, 32 output features
        self.fc2 = nn.Linear(32, 64)  # 32 input features, 64 output features
        # self.fc3 = nn.Linear(64, 4)  # 64 input features, 4 output features
        self.fc3 = nn.Linear(64, 128)  # 64 input features, 128 output features
        self.fc4 = nn.Linear(128, 64)  # 128 input features, 64 output features
        self.fc5 = nn.Linear(64, 4)  # 64 input features, 4 output features

        self.relu = nn.ReLU()  # activation function
        self.dropout = nn.Dropout(dropout)  # dropout regularization
   
    @jit.script_method
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x
    
#TODO LayerNorm normalises across all features 
# so it might normalise each point in the sequence
# which would kinda go against the inteded purpose
class InputRNNNet(jit.ScriptModule):
    def __init__(self, dropout: float, control_dim: int):
        super(InputRNNNet, self).__init__()
        self.fc1 = nn.Linear(control_dim, 32)  
        self.fc2 = nn.Linear(32, control_dim)  

        self.relu = nn.ReLU()  # activation function

    @jit.script_method
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MLPInitNet(jit.ScriptModule):
# class MLPInitNet(nn.Module):
    def __init__(self, dropout: float, control_dim, state_dim, seq_len, hidden_dim):
        super(MLPInitNet, self).__init__()
        in_dim = (control_dim+state_dim+state_dim) * seq_len
        self.fc1 = nn.Linear(in_dim, in_dim*3)  
        torch.nn.init.zeros_(self.fc1.weight)
        self.fc2 = nn.Linear(in_dim*3, in_dim)  
        torch.nn.init.zeros_(self.fc2.weight)
        self.fc3 = nn.Linear(in_dim, hidden_dim) 
        torch.nn.init.zeros_(self.fc3.weight)

        self.relu = nn.ReLU()  # activation function
        self.dropout = nn.Dropout(dropout)  # dropout regularization
   
    # @jit.script_method
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class DiskretizedLinear(jit.ScriptModule):
    def __init__(
        self,
        Ad: List[List[np.float64]],
        Bd: List[List[np.float64]],
        Cd: List[List[np.float64]],
        Dd: List[List[np.float64]],
        ssv_input: List[np.float64],
        ssv_states: List[np.float64],
        no_lin : bool,
        no_bias: bool,
    ):
        super().__init__()
        
        self.no_lin = no_lin
        self.no_bias = no_bias

        self.Ad = nn.Parameter(torch.tensor(Ad).squeeze().float())
        self.Ad.requires_grad = False
        self.Bd = nn.Parameter(torch.tensor(Bd).squeeze().float())
        self.Bd.requires_grad = False
        self.Cd = nn.Parameter(torch.tensor(Cd).squeeze().float())
        self.Cd.requires_grad = False
        self.Dd = nn.Parameter(torch.tensor(Dd).squeeze().float())
        self.Dd.requires_grad = False
        self.ssv_input = nn.Parameter(torch.tensor(ssv_input).squeeze().float())
        self.ssv_input.requires_grad = False
        self.ssv_states = nn.Parameter(torch.tensor(ssv_states).squeeze().float())
        self.ssv_states.requires_grad = False
    
    @jit.script_method
    def forward(self, 
                input_forces: torch.Tensor,
                states: torch.Tensor 
                ) -> torch.Tensor:
        #calculates x_(k+1) = Ad*x_k + Bd*u_k + Ad*e_k
        #           with x_corr_k = x_k+e_k the residual error corrected state
        #           y_k = x_k
        #and shifts input, and output to fit with the actual system
        if self.no_lin:
            return torch.zeros_like(states)

        #shift the input to the linearized input
        delta_in = input_forces - self.ssv_input
        if self.no_bias:
            delta_in = input_forces
        # the correction is calculated outside, this is just for clarification
        states_corr = states 
        #also shift the states since the inital state needs to be shifted or if i want to do one step predictions
        delta_states_corr = states_corr - self.ssv_states
        if self.no_bias:
            delta_states_corr = states_corr
        #x_(k+1) = Ad*(x_k+e_k) + Bd*u_k
        #for compatability with torch batches we transpose the equation
        delta_states_next = torch.matmul(delta_states_corr, self.Ad.transpose(0,1)) + torch.matmul(delta_in, self.Bd.transpose(0,1)) 
        #shift the linearized states back to the states
        states_next = delta_states_next + self.ssv_states
        if self.no_bias:
            states_next = delta_states_next
        #dont calculate y here, rather outside since else the calculation order might be wierd
        return states_next
    
    @jit.script_method
    def calc_output(self, 
            states: torch.Tensor,
            input_forces: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """Has no real function yet and just gives out the states but in case of a
        system with direct feedtrough or where C is not Identity it is better
        and cleaner. Represents the general Output of a linear system:
            y=C*x+D*u.

        Args:
            states (torch.Tensor): current state of the diskretized linear model
            input_forces (torch.Tensor, optional): control input forces. Defaults to None.
        
        Returns:
            torch.Tensor: current output of the diskretized linear model
        """
        y = states

        return y
    

class DiskretizedLinearOpt(jit.ScriptModule):
    def __init__(
        self,
        Ad: List[List[np.float64]],
        Bd: List[List[np.float64]],
        Cd: List[List[np.float64]],
        Dd: List[List[np.float64]],
        ssv_input: List[np.float64],
        ssv_states: List[np.float64],
    ):
        super().__init__()

        self.Ad = nn.Parameter(torch.empty(*torch.tensor(Ad).squeeze().shape).float())
        self.Bd = nn.Parameter(torch.empty(*torch.tensor(Bd).squeeze().shape).float())

        # Initialize parameters using Xavier/Glorot initialization
        nn.init.zeros_(self.Ad)
        nn.init.zeros_(self.Bd)


        self.Cd = nn.Parameter(torch.tensor(Cd).squeeze().float())
        self.Cd.requires_grad = False
        self.Dd = nn.Parameter(torch.tensor(Dd).squeeze().float())
        self.Dd.requires_grad = False
        self.ssv_input = nn.Parameter(torch.tensor(ssv_input).squeeze().float())
        self.ssv_input.requires_grad = False
        self.ssv_states = nn.Parameter(torch.tensor(ssv_states).squeeze().float())
        self.ssv_states.requires_grad = False
    
    @jit.script_method
    def forward(self, 
                input_forces: torch.Tensor,
                states: torch.Tensor 
                ) -> torch.Tensor:
        #calculates x_(k+1) = Ad*x_k + Bd*u_k + Ad*e_k
        #           with x_corr_k = x_k+e_k the residual error corrected state
        #           y_k = x_k
        #and shifts input, and output to fit with the actual system

        #shift the input to the linearized input
        delta_in = input_forces #- self.ssv_input
        #add the correction calculated by the RNN to the state
        # can be seen as additional input with Ad matrix as input Matrix
        states_corr = states 
        #also shift the states since the inital state needs to be shifted or if i want to do one step predictions
        delta_states_corr = states_corr #- self.ssv_states
        #x_(k+1) = Ad*(x_k+e_k) + Bd*u_k
        #for compatability with torch batches we transpose the equation
        delta_states_next = torch.matmul(delta_states_corr, self.Ad.transpose(0,1)) + torch.matmul(delta_in, self.Bd.transpose(0,1)) 
        #shift the linearized states back to the states
        states_next = delta_states_next #+ self.ssv_states
        #dont calculate y here, rather outside since else the calculation order might be wierd
        return states_next
    
    @jit.script_method
    def calc_output(self, 
            states: torch.Tensor,
            input_forces: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        """Has no real function yet and just gives out the states but in case of a
        system with direct feedtrough or where C is not Identity it is better
        and cleaner. Represents the general Output of a linear system:
            y=C*x+D*u.

        Args:
            states (torch.Tensor): current state of the diskretized linear model
            input_forces (torch.Tensor, optional): control input forces. Defaults to None.
        
        Returns:
            torch.Tensor: current output of the diskretized linear model
        """
        y = states

        return y
    
    def smaller_gain_penalty(self,weigth):
        return weigth * torch.sum(torch.abs(torch.linalg.eig(self.Ad)[0]))
    

class LtiRnnConvConstr(jit.ScriptModule):
    def __init__(
        self, nx: int, nu: int, ny: int, nw: int, gamma: float, beta: float, bias: bool, ga : float, scal_ex : float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(LtiRnnConvConstr, self).__init__()

        self.nx = nx  # number of (hidden)states
        self.nu = nu  # number of performance (external) input
        self.nw = nw  # number of disturbance input
        self.ny = ny  # number of performance output
        self.nz = nw  # number of disturbance output, always equal to size of w

        self.beta = beta
        self.nl = torch.tanh

        self.Y = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))
        self.A = torch.nn.Linear(self.nx, self.nx, bias=False)
        self.B1 = torch.nn.Linear(self.nu, self.nx, bias=False)
        self.B2 = torch.nn.Linear(self.nw, self.nx, bias=False)        
        self.C1 = torch.nn.Linear(self.nx, self.ny, bias=False)
        self.D11 = torch.nn.Linear(self.nu, self.ny, bias=False)
        self.D12 = torch.nn.Linear(self.nw, self.ny, bias=False)
        self.C2 = torch.nn.Linear(self.nx, self.nz, bias=False)
        self.D21 = torch.nn.Linear(self.nu, self.nz, bias=False)
        self.lambdas = torch.nn.Parameter(torch.zeros((self.nw, 1)))

        self.gamma = gamma
        self.ga = ga
        #scaling factor of the external input that is not part of the feeback-loop
        self.scal_ex = scal_ex*torch.ones(self.nu - self.ny,device=device) 

        scal_fed = (self.gamma) * torch.ones(self.ny,device=device) 
        #the constant input will also be scaled with this (order is important)
        scal_ = torch.cat((self.scal_ex, scal_fed), dim=0)
        self.scal = torch.diag(scal_)


    def initialize_lmi(self) -> None:
        # storage function
        Y = cp.Variable((self.nx, self.nx), 'Y')
        # hidden state
        A_tilde = cp.Variable((self.nx, self.nx), 'A_tilde')
        B1_tilde = cp.Variable((self.nx, self.nu), 'B1_tilde')
        B2_tilde = cp.Variable((self.nx, self.nw), 'B2_tilde')
        # output
        C1 = cp.Variable((self.ny, self.nx), 'C1')
        D11 = cp.Variable((self.ny, self.nu), 'D11')
        D12 = cp.Variable((self.ny, self.nw), 'D12')
        # disturbance
        C2 = np.random.normal(0, 1 / np.sqrt(self.nw), size=(self.nz, self.nx))
        D21 = np.random.normal(0, 1 / np.sqrt(self.nw), size=(self.nz, self.nu))
        # multipliers
        lambdas = cp.Variable((self.nw, 1), 'tau', nonneg=True)
        T = cp.diag(lambdas)

        C2_tilde = T @ C2
        D21_tilde = T @ D21

        if self.ga == 0:
            # lmi that ensures finite l2 gain
            M = cp.bmat(
                [
                    [-Y, self.beta * C2_tilde.T, A_tilde.T],
                    [self.beta * C2_tilde, -2 * T, B2_tilde.T],
                    [A_tilde, B2_tilde, -Y],
                ]
            )
        else:
            # lmi that ensures l2 gain gamma
            M = cp.bmat(
                [
                    [
                        -Y,
                        np.zeros((self.nx, self.nu)),
                        self.beta * C2_tilde.T,
                        A_tilde.T,
                        C1.T,
                    ],
                    [
                        np.zeros((self.nu, self.nx)),
                        -self.ga**2 * np.eye(self.nu),
                        self.beta * D21_tilde.T,
                        B1_tilde.T,
                        D11.T,
                    ],
                    [
                        self.beta * C2_tilde,
                        self.beta * D21_tilde,
                        -2 * T,
                        B2_tilde.T,
                        D12.T,
                    ],
                    [A_tilde, B1_tilde, B2_tilde, -Y, np.zeros((self.nx, self.ny))],
                    [C1, D11, D12, np.zeros((self.ny, self.nx)), -np.eye(self.ny)],
                ]
            )

        # setup optimization problem, objective might change,
        # any feasible solution works as initialization for the parameters
        nM = M.shape[0]
        tol = 1e-4
        # rand_matrix = np.random.normal(0,1/np.sqrt(self.nx), (self.nx,self.nw))
        # objective = cp.Minimize(cp.norm(Y @ rand_matrix - B2_tilde))
        # nu = cp.Variable((1, nM))
        # objective = cp.Minimize(nu @ np.ones((nM, 1)))
        objective = cp.Minimize(None)
        problem = cp.Problem(objective, [M << -tol * np.eye(nM)])

        logger.info(
            'Initialize Parameter by values that satisfy LMI constraints, solve SDP ...'
        )
        problem.solve(solver=cp.SCS)
        # check if t is negative
        # max_eig_lmi = np.max(np.real(np.linalg.eig(M.value)[0]))

        if problem.status == 'optimal':
            logger.info(
                f'Found negative semidefinite LMI, problem status: '
                f'\t {problem.status}'
            )
        else:
            raise Exception(
                "Neural network could not be initialized "
                "since no solution to the SDP problem was found."
            )

        logger.info('Write back Parameters values ...')
        dtype = torch.get_default_dtype()

        self.Y.data = torch.tensor(Y.value, dtype=dtype)
        self.lambdas.data = torch.tensor(lambdas.value, dtype=dtype)

        Y_inv = np.linalg.inv(Y.value)
        T_inv = np.diag(1 / np.squeeze(lambdas.value))

        A_ = Y_inv @ A_tilde.value
        B1_ = Y_inv @ B1_tilde.value
        B2_ = Y_inv @ B2_tilde.value
        C2_ = T_inv @ C2_tilde.value
        D21_ = T_inv @ D21_tilde.value

        self.A.weight.data = torch.tensor(A_, dtype=dtype)

        self.B2.weight.data = torch.tensor(B2_, dtype=dtype)
        if not self.ga == 0:
            self.C1.weight.data = torch.tensor(C1.value, dtype=dtype)
            self.D11.weight.data = torch.tensor(D11.value, dtype=dtype)
            self.D12.weight.data = torch.tensor(D12.value, dtype=dtype)
            self.B1.weight.data = torch.tensor(B1_, dtype=dtype)
            self.D21.weight.data = torch.tensor(D21_, dtype=dtype)
        self.C2.weight.data = torch.tensor(C2_, dtype=dtype)

    @jit.script_method
    def forward(
        self,
        x_pred: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        hx: Optional[torch.Tensor]= None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_sample, _ = x_pred.shape

        # Y_inv = self.Y.inverse()
        # T_inv = torch.diag(1 / torch.squeeze(self.lambdas))
        # initialize output
        y = torch.zeros((n_batch, n_sample, self.ny),device=device)

        if hx is not None:
            x = hx
        else:
            x = torch.zeros((n_batch, self.nx),device=device)

        for k in range(n_sample):
            z = (self.C2(x) + self.D21(x_pred[:, k, :]@self.scal) )#+ self.b_z) @ self.T_inv
            w = self.nl(z)
            y[:, k, :] = self.C1(x) + self.D11(x_pred[:, k, :]@self.scal) + self.D12(w) #+ self.b_y
            x = (
                self.A(x) + self.B1(x_pred[:, k, :]@self.scal) + self.B2(w)
            ) #@ self.Y_inv

        return y, x

    @jit.script_method
    def forward_alt(
        self,
        x_pred: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, n_sample, _ = x_pred.shape

        #Add constant input as alternative Bias term
        # (needs to be considered in initialisation)
        #because of scaling constant input has to be first
        x_pred = torch.cat((torch.ones((n_batch, n_sample, 1), device=device), x_pred), dim=-1)

        # initialize output
        y = torch.zeros((n_batch, n_sample, self.ny),device=device)

        if hx is not None:
            x = hx[0][1]
        else:
            x = torch.zeros((n_batch, self.nx),device=device)

        for k in range(n_sample):
            z = (self.C2(x) + self.D21(x_pred[:, k, :]@self.scal) )#+ self.b_z) @ self.T_inv
            w = self.nl(z)
            y[:, k, :] = self.C1(x) + self.D11(x_pred[:, k, :]@self.scal) + self.D12(w) #+ self.b_y
            x = (
                self.A(x) + self.B1(x_pred[:, k, :]@self.scal) + self.B2(w)
            ) #@ self.Y_inv

        return y, (x, x)
    
    @jit.script_method
    def forward_alt_onestep(
        self,
        x_pred: torch.Tensor,
        hx: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_sample, _ = x_pred.shape

        #Add constant input as alternative Bias term
        # (needs to be considered in initialisation)
        #because of scaling constant input has to be first
        x_pred = torch.cat((torch.ones((n_batch, n_sample,1), device=device), x_pred), dim=-1)

        z = (self.C2(hx) + self.D21(x_pred@self.scal) )
        w = self.nl(z)
        y = self.C1(hx) + self.D11(x_pred@self.scal) + self.D12(w) 
        x = (
            self.A(hx) + self.B1(x_pred@self.scal) + self.B2(w)
        ) 

        return y, x
    
    @jit.script_method
    def forward_onestep(
        self,
        x_pred: torch.Tensor,
        hx: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        z = (self.C2(hx) + self.D21(x_pred@self.scal) )
        w = self.nl(z)
        y = self.C1(hx) + self.D11(x_pred@self.scal) + self.D12(w) 
        x = (
            self.A(hx) + self.B1(x_pred@self.scal) + self.B2(w)
        ) 

        return y, x

    def get_constraints(self) -> torch.Tensor:
        # state sizes
        nx = self.nx
        nu = self.nu
        ny = self.ny

        T = torch.diag(torch.squeeze(self.lambdas))
        ga = self.ga

        beta = self.beta
        # storage function
        Y = self.Y
        device = Y.device

        # state
        A_tilde = Y @ self.A.weight
        B1_tilde = Y @ self.B1.weight
        B2_tilde = Y @ self.B2.weight
        # output
        C1 = self.C1.weight
        D11 = self.D11.weight
        D12 = self.D12.weight
        # disturbance
        D21_tilde = T @ self.D21.weight
        C2_tilde = T @ self.C2.weight

        # M << 0
        if self.ga == 0:
            M = torch.cat(
                [
                    torch.cat(
                        [-Y, beta * C2_tilde.T, A_tilde.T],
                        dim=1,
                    ),
                    torch.cat(
                        [beta * C2_tilde, -2 * T, B2_tilde.T],
                        dim=1,
                    ),
                    torch.cat(
                        [A_tilde, B2_tilde, -Y],
                        dim=1,
                    ),
                ]
            )
        else:

            M = torch.cat(
                [
                    torch.cat(
                        (
                            -Y,
                            torch.zeros((nx, nu), device=device),
                            beta * C2_tilde.T,
                            A_tilde.T,
                            C1.T,
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            torch.zeros((nu, nx), device=device),
                            -(ga**2) * torch.eye(nu, device=device),
                            beta * D21_tilde.T,
                            B1_tilde.T,
                            D11.T,
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (beta * C2_tilde, beta * D21_tilde, -2 * T, B2_tilde.T, D12.T),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            A_tilde,
                            B1_tilde,
                            B2_tilde,
                            -Y,
                            torch.zeros((nx, ny), device=device),
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            C1,
                            D11,
                            D12,
                            torch.zeros((ny, nx), device=device),
                            -torch.eye(ny, device=device),
                        ),
                        dim=1,
                    ),
                ]
            )

        return 0.5 * (M + M.T)

    def get_barrier(self, t: float) -> torch.Tensor:
        M = self.get_constraints()
        barrier = -t * (-M).logdet()

        #this is unnecessary since the last step of each batch calculation
        #is to check constraints and do backtracking until they check
        #and get_barrier will then be calculated before the next optimizer step
        # _, info = torch.linalg.cholesky_ex(-M.cpu())

        # if info > 0:
        #     barrier += torch.tensor(float('inf'),device=barrier.device)

        return barrier

    def check_constr(self) -> bool:
        with torch.no_grad():
            M = self.get_constraints()

            _, info = torch.linalg.cholesky_ex(-M.cpu())

            if info > 0:
                b_satisfied = False
            else:
                b_satisfied = True

        return b_satisfied

    def get_min_max_real_eigenvalues(self) -> Tuple[np.float64, np.float64]:
        M = self.get_constraints()
        return (
            torch.min(torch.real(torch.linalg.eig(M)[0])).cpu().detach().numpy(),
            torch.max(torch.real(torch.linalg.eig(M)[0])).cpu().detach().numpy(),
        )
