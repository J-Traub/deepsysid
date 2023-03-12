import copy
import json
import logging
import time
from typing import Dict, List, Literal, Optional, Tuple

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
from ..datasets import RecurrentInitializerDataset, RecurrentPredictorDataset



logger = logging.getLogger(__name__)


class InputNet(nn.Module):
    def __init__(self):
        super(InputNet, self).__init__()
        self.fc1 = nn.Linear(6, 32)  # 6 input features, 32 output features
        self.fc2 = nn.Linear(32, 64)  # 32 input features, 64 output features
        self.fc3 = nn.Linear(64, 128)  # 64 input features, 128 output features
        self.fc4 = nn.Linear(128, 64)  # 128 input features, 64 output features
        self.fc5 = nn.Linear(64, 4)  # 64 input features, 4 output features

        self.relu = nn.ReLU()  # activation function
        self.dropout = nn.Dropout(0.2)  # dropout regularization

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

class DiskretizedLinear(nn.Module):
    def __init__(
        self,
        Ad: NDArray[np.float64],
        Bd: NDArray[np.float64],
        Cd: NDArray[np.float64],
        Dd: NDArray[np.float64],
        ssv_input: NDArray[np.float64],
        ssv_states: NDArray[np.float64],
    ):
        super().__init__()

        self.Ad = nn.Parameter(torch.tensor(Ad).float())
        self.Ad.requires_grad = False
        self.Bd = nn.Parameter(torch.tensor(Bd).float())
        self.Bd.requires_grad = False
        self.Cd = nn.Parameter(torch.tensor(Cd).float())
        self.Cd.requires_grad = False
        self.Dd = nn.Parameter(torch.tensor(Dd).float())
        self.Dd.requires_grad = False
        self.ssv_input = nn.Parameter(torch.tensor(ssv_input).float())
        self.ssv_input.requires_grad = False
        self.ssv_states = nn.Parameter(torch.tensor(ssv_states).float())
        self.ssv_states.requires_grad = False

    def forward(self, 
                input_forces: torch.Tensor,
                states: torch.Tensor ,
                residual_errors: torch.Tensor
                ) -> torch.Tensor:
        #calculates x_(k+1) = Ad*x_k + Bd*u_k + Ad*e_k
        #           with x_corr_k = x_k+e_k the residual error corrected state
        #           y_k = x_k
        #and shifts input, and output to fit with the actual system

        #shift the input to the linearized input
        delta_in = input_forces - self.ssv_input
        #add the correction calculated by the RNN to the state
        # can be seen as additional input with Ad matrix as input Matrix
        states_corr = states + residual_errors
        #the states dont need to be shifted since they are going to be used only in the linearized system
        #x_(k+1) = Ad*(x_k+e_k) + Bd*u_k
        states_next = torch.mm(self.Ad, states_corr) + torch.mm(self.Bd, delta_in) 
        #shift the linearized output back to the output
        y_linear = states + self.ssv_states
        return y_linear, states_next
    

class HybridLtiRnnConvConstr(LtiRnnConvConstr):
    def __init__(
        self, 
        nx: int, 
        nu: int, 
        ny: int, 
        nw: int, 
        gamma: float, 
        beta: float, 
        bias: bool,
        Ad: NDArray[np.float64],
        Bd: NDArray[np.float64],
        Cd: NDArray[np.float64],
        Dd: NDArray[np.float64],
        ssv_input: NDArray[np.float64],
        ssv_states: NDArray[np.float64],
    ) -> None:
        super().__init__(self,nx=nx, nu=nu, ny=ny, nw=nw, gamma=gamma, beta=beta, bias=bias)
        
        self._diskretized_linear = DiskretizedLinear(
            Ad = Ad,
            Bd = Bd,
            Cd = Cd,
            Dd = Dd,
            ssv_input= ssv_input,
            ssv_states= ssv_states,
        )      

    #TODO: incooperate the linearized system here
    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, n_sample, _ = x_pred.shape

        Y_inv = self.Y.inverse()
        T_inv = torch.diag(1 / torch.squeeze(self.lambdas))
        # initialize output
        y = torch.zeros((n_batch, n_sample, self.ny))

        if hx is not None:
            x = hx[0][1]
        else:
            x = torch.zeros((n_batch, self.nx))

        #TODO: x_pred seems to be just the input 

        for k in range(n_sample):
            z = (self.C2_tilde(x) + self.D21_tilde(x_pred[:, k, :]) + self.b_z) @ T_inv
            w = self.nl(z)
            y[:, k, :] = self.C1(x) + self.D11(x_pred[:, k, :]) + self.D12(w) + self.b_y
            x = (
                self.A_tilde(x) + self.B1_tilde(x_pred[:, k, :]) + self.B2_tilde(w)
            ) @ Y_inv

        return y, (x, x)
            

class HybridConstrainedRnnConfig(DynamicIdentificationModelConfig):
    nx: int
    recurrent_dim: int
    gamma: float
    beta: float
    initial_decay_parameter: float
    decay_rate: float
    epochs_with_const_decay: int
    num_recurrent_layers_init: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    loss: Literal['mse', 'msge']
    bias: bool
    log_min_max_real_eigenvalues: Optional[bool] = False
    ######################################################################################### new
    Ad: NDArray[np.float64]
    Bd: NDArray[np.float64]
    Cd: NDArray[np.float64]
    Dd: NDArray[np.float64]
    ssv_input: NDArray[np.float64]
    ssv_states: NDArray[np.float64]
    #########################################################################



class HybridConstrainedRnn(base.NormalizedHiddenStateInitializerPredictorModel):
    CONFIG = HybridConstrainedRnnConfig

    def __init__(self, config: HybridConstrainedRnnConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.nx = config.nx
        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.recurrent_dim = config.recurrent_dim

        self.initial_decay_parameter = config.initial_decay_parameter
        self.decay_rate = config.decay_rate
        self.epochs_with_const_decay = config.epochs_with_const_decay

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers_init = config.num_recurrent_layers_init
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor = config.epochs_predictor

        ######################################################################################### new
        self.Ad = config.Ad
        self.Bd = config.Bd
        self.Cd = config.Cd
        self.Dd = config.Dd
        self.ssv_input = config.ssv_input
        self.ssv_states = config.ssv_states
        #########################################################################

        self.log_min_max_real_eigenvalues = config.log_min_max_real_eigenvalues

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        ######################################################################################### changed
        # changed nu for the additonal linearized system output as input
        #changed the model to the hybrid version
        self._predictor = HybridLtiRnnConvConstr(
            nx=self.nx,
            nu=self.control_dim + self.state_dim,
            ny=self.state_dim,
            nw=self.recurrent_dim,
            gamma=config.gamma,
            beta=config.beta,
            bias=config.bias,
        ).to(self.device)
        ######################################################################### 
        #TODO: Initialize the feedfoward input net somewhere here

        self._initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.nx,
            num_recurrent_layers=self.num_recurrent_layers_init,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self._initializer.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        us = control_seqs
        ys = state_seqs
        self._predictor.initialize_lmi()
        self._predictor.to(self.device)
        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(us)
        self._state_mean, self._state_std = utils.mean_stddev(ys)

        us = [
            utils.normalize(control, self._control_mean, self._control_std)
            for control in us
        ]
        ys = [utils.normalize(state, self._state_mean, self._state_std) for state in ys]

        #TODO: Train the feedfoward input net somewhere here

        initializer_dataset = RecurrentInitializerDataset(us, ys, self.sequence_length)

        initializer_loss = []
        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self._initializer.zero_grad()
                y, _ = self._initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_init.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {total_loss}'
            )
            initializer_loss.append(total_loss)
        time_end_init = time.time()
        predictor_dataset = RecurrentPredictorDataset(us, ys, self.sequence_length)

        time_start_pred = time.time()
        t = self.initial_decay_parameter
        predictor_loss: List[np.float64] = []
        min_eigenvalue: List[np.float64] = []
        max_eigenvalue: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []

        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
                # Predict and optimize
                y, _ = self._predictor.forward(
                    batch['x'].float().to(self.device), hx=hx
                )
                y = y.to(self.device)
                barrier = self._predictor.get_barrier(t).to(self.device)
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()

                # gradient infos
                grads_norm = [
                    torch.linalg.norm(p.grad)
                    for p in filter(
                        lambda p: p.grad is not None, self._predictor.parameters()
                    )
                ]
                max_grad += max(grads_norm)

                # save old parameter set
                old_pars = [
                    par.clone().detach() for par in self._predictor.parameters()
                ]

                self.optimizer_pred.step()

                # perform backtracking line search if constraints are not satisfied
                max_iter = 100
                alpha = 0.5
                bls_iter = 0
                while not self._predictor.check_constr():
                    for old_par, new_par in zip(old_pars, self._predictor.parameters()):
                        new_par.data = (
                            alpha * old_par.clone() + (1 - alpha) * new_par.data
                        )

                    if bls_iter > max_iter - 1:
                        for old_par, new_par in zip(
                            old_pars, self._predictor.parameters()
                        ):
                            new_par.data = old_par.clone()
                        M = self._predictor.get_constraints()
                        logger.warning(
                            f'Epoch {i+1}/{self.epochs_predictor}\t'
                            f'max real eigenvalue of M: '
                            f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                            f'Backtracking line search exceeded maximum iteration. \t'
                            f'Constraints satisfied? {self._predictor.check_constr()}'
                        )
                        time_end_pred = time.time()
                        time_total_init = time_end_init - time_start_init
                        time_total_pred = time_end_pred - time_start_pred

                        return dict(
                            index=np.asarray(i),
                            epoch_loss_initializer=np.asarray(initializer_loss),
                            epoch_loss_predictor=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            backtracking_iter=np.asarray(backtracking_iter),
                            gradient_norm=np.asarray(gradient_norm),
                            max_eigenvalue=np.asarray(max_eigenvalue),
                            min_eigenvalue=np.asarray(min_eigenvalue),
                            training_time_initializer=np.asarray(time_total_init),
                            training_time_predictor=np.asarray(time_total_pred),
                        )
                    bls_iter += 1

            # decay t following the idea of interior point methods
            if i % self.epochs_with_const_decay == 0 and i != 0:
                t = t * 1 / self.decay_rate
                logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')

            min_ev = np.float64('inf')
            max_ev = np.float64('inf')
            if self.log_min_max_real_eigenvalues:
                min_ev, max_ev = self._predictor.get_min_max_real_eigenvalues()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor}\t'
                f'Total Loss (Predictor): {total_loss:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            predictor_loss.append(np.float64(total_loss))
            barrier_value.append(barrier.cpu().detach().numpy())
            backtracking_iter.append(np.float64(bls_iter))
            gradient_norm.append(np.float64(max_grad))
            max_eigenvalue.append(np.float64(max_ev))
            min_eigenvalue.append(np.float64(min_ev))

        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        return dict(
            index=np.asarray(i),
            epoch_loss_initializer=np.asarray(initializer_loss),
            epoch_loss_predictor=np.asarray(predictor_loss),
            barrier_value=np.asarray(barrier_value),
            backtracking_iter=np.asarray(backtracking_iter),
            gradient_norm=np.asarray(gradient_norm),
            max_eigenvalue=np.asarray(max_eigenvalue),
            min_eigenvalue=np.asarray(min_eigenvalue),
            training_time_initializer=np.asarray(time_total_init),
            training_time_predictor=np.asarray(time_total_pred),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if (
            self._control_mean is None
            or self._control_std is None
            or self._state_mean is None
            or self._state_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._initializer.eval()
        self._predictor.eval()

        initial_u = initial_control
        initial_y = initial_state
        u = control

        initial_u = utils.normalize(initial_u, self._control_mean, self._control_std)
        initial_y = utils.normalize(initial_y, self._state_mean, self._state_std)
        u = utils.normalize(u, self._control_mean, self._control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_u[1:], initial_y[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(u).unsqueeze(0).float().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self._state_mean, self._state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self._initializer.state_dict(), file_path[0])
        torch.save(self._predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'state_mean': self._state_mean.tolist(),
                    'state_std': self._state_std.tolist(),
                    'control_mean': self._control_mean.tolist(),
                    'control_std': self._control_std.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self._initializer.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self._predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_std'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self._initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._initializer)

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._predictor)



























class HybridResidualLSTMModelConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_parallel: int
    epochs_feedback: int
    loss: Literal['mse', 'msge']


class HybridResidualLSTMModel(base.DynamicIdentificationModel, abc.ABC):
    def __init__(
        self,
        config: HybridResidualLSTMModelConfig,
        device_name: str,
    ):
        super().__init__(config)



        self.device_name = device_name
        self.device = torch.device(device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.time_delta = float(config.time_delta)

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers = config.num_recurrent_layers
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_parallel = config.epochs_parallel
        self.epochs_feedback = config.epochs_feedback

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        #TODO initialize linear here

        #TODO change to robust rnn
        self.blackbox = rnn.LinearOutputLSTM(
            input_dim=self.control_dim
            + self.state_dim,  # control input and whitebox estimate
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.state_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            output_dim=[self.state_dim],
            num_recurrent_layers=self.num_recurrent_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_initializer = optim.Adam(
            params=self.initializer.parameters(), lr=self.learning_rate
        )
        self.optimizer_end2end = optim.Adam(
            params=self.blackbox.parameters(), lr=self.learning_rate
        )

        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_std: Optional[NDArray[np.float64]] = None
        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        epoch_losses_initializer = []
        epoch_losses_teacher = []
        epoch_losses_multistep = []

        self.blackbox.train()
        self.initializer.train()
        self.physical.train()
        self.semiphysical.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        un_control_seqs = control_seqs
        un_state_seqs = state_seqs
        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]

        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_std = torch.from_numpy(self.state_std).float().to(self.device)

        def denormalize_state(x: torch.Tensor) -> torch.Tensor:
            return (x * state_std) + state_mean

        def scale_acc_physical(x: torch.Tensor) -> torch.Tensor:
            return x / state_std[self.physical_state_mask]  # type: ignore

        def scale_acc_physical_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = (
                x / self.state_std[self.physical_state_mask]  # type: ignore
            )
            return out

        #train initializer?
        initializer_dataset = RecurrentInitializerDataset(
            control_seqs, state_seqs, self.sequence_length
        )
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y, _ = self.initializer.forward(batch['x'].float().to(self.device))
                # This type error is ignored, since we know that y will not be a tuple.
                batch_loss = mse_loss(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_initializer.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer} '
                f'- Epoch Loss (Initializer): {total_loss}'
            )
            epoch_losses_initializer.append([i, total_loss])

        #TODO here needs to go the first training with the linear network, one step at a time
        dataset = RecurrentHybridPredictorDataset(
            control_seqs=control_seqs,
            state_seqs=state_seqs,
            un_control_seqs=un_control_seqs,
            un_state_seqs=un_state_seqs,
            sequence_length=self.sequence_length,
        )
        for i in range(self.epochs_parallel):
            data_loader = data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                x_control_unnormed = batch['x_control_unnormed'].float().to(self.device)
                x_state_unnormed = batch['x_state_unnormed'].float().to(self.device)
                y_whitebox = (
                    torch.zeros((self.batch_size, self.sequence_length, self.state_dim))
                    .float()
                    .to(self.device)
                )

                for time in range(self.sequence_length):
                    y_semiphysical = self.semiphysical.forward(
                        control=x_control_unnormed[
                            :, time, self.semiphysical_control_mask
                        ],
                        state=x_state_unnormed[:, time, self.semiphysical_state_mask],
                    )
                    ydot_physical = scale_acc_physical(
                        self.physical.forward(
                            control=x_control_unnormed[
                                :, time, self.physical_control_mask
                            ],
                            state=x_state_unnormed[:, time, self.physical_state_mask],
                        )
                    )

                    y_whitebox[:, time, self.physical_state_mask] = (
                        y_whitebox[:, time, self.physical_state_mask]
                        + self.physical.time_delta * ydot_physical
                    )
                    y_whitebox[:, time, self.semiphysical_state_mask] = (
                        y_whitebox[:, time, self.semiphysical_state_mask]
                        + y_semiphysical
                    )

                x_init = batch['x_init'].float().to(self.device)
                x_pred = torch.cat(
                    (batch['x_pred'].float().to(self.device), y_whitebox), dim=2
                )  # serial connection

                _, hx_init = self.initializer.forward(x_init)

                y_blackbox, _ = self.blackbox_forward(x_pred, y_whitebox, hx=hx_init)
                y = y_blackbox + y_whitebox

                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_parallel} '
                f'- Epoch Loss (Parallel): {total_epoch_loss}'
            )
            epoch_losses_teacher.append([i, total_epoch_loss])

        #need to reset the optimizer apparently
        self.optimizer_end2end = optim.Adam(
            params=self.blackbox.parameters(), lr=self.learning_rate
        )
        for i in range(self.epochs_feedback):
            data_loader = data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                current_state = batch['initial_state'].float().to(self.device)
                x_control_unnormed = batch['x_control_unnormed'].float().to(self.device)
                x_pred = batch['x_pred'].float().to(self.device)
                y_est = (
                    torch.zeros((self.batch_size, self.sequence_length, self.state_dim))
                    .float()
                    .to(self.device)
                )

                x_init = batch['x_init'].float().to(self.device)
                _, hx_init = self.initializer.forward(x_init)

                for time in range(self.sequence_length):
                    y_whitebox = (
                        torch.zeros((self.batch_size, self.state_dim))
                        .float()
                        .to(self.device)
                    )
                    y_semiphysical = self.semiphysical.forward(
                        x_control_unnormed[:, time, self.semiphysical_control_mask],
                        current_state[:, self.semiphysical_state_mask],
                    )

                    ydot_physical = scale_acc_physical(
                        self.physical.forward(
                            x_control_unnormed[:, time, self.physical_control_mask],
                            current_state[:, self.physical_state_mask],
                        )
                    )
                    y_whitebox[:, self.physical_state_mask] = (
                        y_whitebox[:, self.physical_state_mask]
                        + self.time_delta * ydot_physical
                    )
                    y_whitebox[:, self.semiphysical_state_mask] = (
                        y_whitebox[:, self.semiphysical_state_mask] + y_semiphysical
                    )

                    y_blackbox, hx_init = self.blackbox_forward(
                        torch.cat(
                            (x_pred[:, time, :].unsqueeze(1), y_whitebox.unsqueeze(1)),
                            dim=2,
                        ),
                        y_whitebox.unsqueeze(1),
                        hx=hx_init,
                    )
                    current_state = y_blackbox.squeeze(1) + y_whitebox
                    y_est[:, time, :] = current_state
                    current_state = denormalize_state(current_state)
                batch_loss = self.loss.forward(
                    y_est, batch['y'].float().to(self.device)
                )
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_feedback} '
                f'- Epoch Loss (Feedback): {total_epoch_loss}'
            )
            epoch_losses_multistep.append([i, total_epoch_loss])

        return dict(
            epoch_loss_initializer=np.array(epoch_losses_initializer, dtype=np.float64),
            epoch_loss_teacher=np.array(epoch_losses_teacher, dtype=np.float64),
            epoch_loss_multistep=np.array(epoch_losses_multistep, dtype=np.float64),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        threshold: float = np.infty,
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y, whitebox, blackbox = self.simulate_hybrid(
            initial_control=initial_control,
            initial_state=initial_state,
            control=control,
        )
        return y, dict(whitebox=whitebox, blackbox=blackbox)

    def simulate_hybrid(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        threshold: float = np.infty,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self.blackbox.eval()
        self.initializer.eval()
        self.semiphysical.eval()
        self.physical.eval()

        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_std = torch.from_numpy(self.state_std).float().to(self.device)

        def denormalize_state(x: torch.Tensor) -> torch.Tensor:
            return (x * state_std) + state_mean

        def scale_acc_physical(x: torch.Tensor) -> torch.Tensor:
            return x / state_std[self.physical_state_mask]  # type: ignore

        un_control = control
        current_state_np = initial_state[-1, :]
        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        y = np.zeros((control.shape[0], self.state_dim), dtype=np.float64)
        whitebox = np.zeros((control.shape[0], self.state_dim), dtype=np.float64)
        blackbox = np.zeros((control.shape[0], self.state_dim), dtype=np.float64)

        with torch.no_grad():
            x_init = (
                torch.from_numpy(np.hstack((initial_control, initial_state)))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            _, hx = self.initializer.forward(
                x_init
            )  # hx is hidden state of predictor LSTM

            x_control_un = (
                torch.from_numpy(un_control).unsqueeze(0).float().to(self.device)
            )
            current_state = (
                torch.from_numpy(current_state_np).unsqueeze(0).float().to(self.device)
            )
            x_pred = torch.from_numpy(control).unsqueeze(0).float().to(self.device)
            for time in range(control.shape[0]):
                y_whitebox = (
                    torch.zeros((current_state.shape[0], self.state_dim))
                    .float()
                    .to(self.device)
                )
                y_semiphysical = self.semiphysical.forward(
                    control=x_control_un[:, time, self.semiphysical_control_mask],
                    state=current_state[:, self.semiphysical_state_mask],
                )
                ydot_physical = scale_acc_physical(
                    self.physical.forward(
                        x_control_un[:, time, self.physical_control_mask],
                        current_state[:, self.physical_state_mask],
                    )
                )
                y_whitebox[:, self.physical_state_mask] = (
                    y_whitebox[:, self.physical_state_mask]
                    + self.time_delta * ydot_physical
                )
                y_whitebox[:, self.semiphysical_state_mask] = (
                    y_whitebox[:, self.semiphysical_state_mask] + y_semiphysical
                )

                x_blackbox = torch.cat(
                    (x_pred[:, time, :], y_whitebox), dim=1
                ).unsqueeze(1)
                y_blackbox, hx = self.blackbox_forward(
                    x_blackbox,
                    None,
                    hx=hx,
                )
                y_blackbox = torch.clamp(y_blackbox, -threshold, threshold)
                y_est = y_blackbox.squeeze(1) + y_whitebox
                current_state = denormalize_state(y_est)
                y[time, :] = current_state.cpu().detach().numpy()
                whitebox[time, :] = y_whitebox.cpu().detach().numpy()
                blackbox[time, :] = y_blackbox.squeeze(1).cpu().detach().numpy()

        return y, whitebox, blackbox

    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.semiphysical.state_dict(), file_path[0])
        torch.save(self.blackbox.state_dict(), file_path[1])
        torch.save(self.initializer.state_dict(), file_path[2])

        semiphysical_params = [
            param.tolist() for param in self.semiphysical.get_parameters_to_save()
        ]

        with open(file_path[3], mode='w') as f:
            json.dump(
                {
                    'state_mean': self.state_mean.tolist(),
                    'state_std': self.state_std.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                    'semiphysical': semiphysical_params,
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self.semiphysical.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self.blackbox.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        self.initializer.load_state_dict(
            torch.load(file_path[2], map_location=self.device_name)
        )
        with open(file_path[3], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self.state_std = np.array(norm['state_std'], dtype=np.float64)
        self.control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self.control_std = np.array(norm['control_std'], dtype=np.float64)
        self.semiphysical.load_parameters(
            [np.array(param, dtype=np.float64) for param in norm['semiphysical']]
        )

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'semi-physical.pth', 'blackbox.pth', 'initializer.pth', 'json'

    def get_parameter_count(self) -> int:
        semiphysical_count = sum(p.numel() for p in self.semiphysical.parameters())
        blackbox_count = sum(
            p.numel() for p in self.blackbox.parameters() if p.requires_grad
        )
        initializer_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        return semiphysical_count + blackbox_count + initializer_count

    def blackbox_forward(
        self,
        x_pred: torch.Tensor,
        y_wb: Optional[torch.Tensor],
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO: x_pred should instead be x_control.
        # TODO: I don't remember the purpose of this function.
        #  Probably to generalize the code in some way?
        return self.blackbox.forward(x_pred, hx=hx)
