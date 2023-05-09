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

logger = logging.getLogger(__name__)


class TimeSeriesDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(self, control_seqs, state_seqs):
        self.control,self.next_state,self.state = self.__load_data(
            control_seqs, state_seqs
        )
        self.subbatchsize = 50
        subbatchnum = int(self.control.shape[0]/self.subbatchsize)
        self.control = np.resize(self.control,(subbatchnum,self.subbatchsize,self.control.shape[1]))
        self.next_state = np.resize(self.next_state,(subbatchnum,self.subbatchsize,self.next_state.shape[1]))
        self.state = np.resize(self.state,(subbatchnum,self.subbatchsize,self.state.shape[1]))

    def __len__(self):
        # compute the total number of samples by summing the lengths of all sequences
        return self.control.shape[0]

    def __load_data(self, control_seqs,state_seqs):
        control = []
        next_state = []
        state = []
        for control_seq, state_seq in zip(control_seqs, state_seqs):
            # create a next state sequence by shifting state_seq
            next_state_seq = np.roll(state_seq, -1,0)
             #put each in one continuos list and drop the last element since it is nonsense now
            next_state.append(next_state_seq[:-1]) 
            control.append(control_seq[:-1])
            state.append(state_seq[:-1])
            
        return np.vstack(control),np.vstack(next_state),np.vstack(state)
    
    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'FNN_input': self.control[idx],
            'next_state': self.next_state[idx], 
            'Lin_input': self.state[idx],
            }

class HybridRecurrentLinearFNNInputDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        dataset = self.__load_data(control_seqs, state_seqs)
        self.x0 = dataset['x0']
        self.y0 = dataset['y0']
        self.control = dataset['cont']
        self.states = dataset['stat']
        self.x0_control = dataset['x0_control']
        self.x0_states = dataset['x0_states']
        self.control_prev = dataset['cont_prev']
        self.states_prev = dataset['stat_prev']

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        x0_seq = list()
        y0_seq = list()
        cont_seq = list()
        stat_seq = list()

        x0_control_seq = list()
        x0_states_seq = list()
        cont_prev_seq = list()
        stat_prev_seq = list()

        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            x0 = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            y0 = np.zeros((n_samples, self.state_dim), dtype=np.float64)
            cont = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            stat = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            cont_prev = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            stat_prev = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )
            x0_control = np.zeros(
                (n_samples, self.sequence_length, self.control_dim),
                dtype=np.float64,
            )
            x0_states = np.zeros(
                (n_samples, self.sequence_length, self.state_dim),
                dtype=np.float64,
            )

            #TODO: sanity check check
            x0_ = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x0[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.sequence_length, :],
                        state[time : time + self.sequence_length, :],
                    )
                )
                x0_control[idx, :, :] = control[time : time + self.sequence_length]
                x0_states[idx, :, :] = state[time : time + self.sequence_length]

                y0[idx, :] = state[time + self.sequence_length - 1, :]
                cont[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                stat[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]

                cont_prev[idx, :, :] = control[
                    time + self.sequence_length -1 : time + 2 * self.sequence_length -1, :
                ]
                stat_prev[idx, :, :] = state[
                    time + self.sequence_length -1: time + 2 * self.sequence_length -1, :
                ]

            x0_seq.append(x0)
            y0_seq.append(y0)
            cont_seq.append(cont)
            stat_seq.append(stat)

            x0_control_seq.append(x0_control)
            x0_states_seq.append(x0_states)
            cont_prev_seq.append(cont_prev)
            stat_prev_seq.append(stat_prev)


        return dict(
            x0 = np.vstack(x0_seq),
            y0 =  np.vstack(y0_seq),
            cont = np.vstack(cont_seq),
            stat =  np.vstack(stat_seq),
            x0_control = np.vstack(x0_control_seq),
            x0_states = np.vstack(x0_states_seq),
            cont_prev = np.vstack(cont_prev_seq),
            stat_prev = np.vstack(stat_prev_seq),

        )

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'x0': self.x0[idx],
            'y0': self.y0[idx],
            'control': self.control[idx],
            'states': self.states[idx],
            'x0_control': self.x0_control[idx], 
            'x0_states': self.x0_states[idx],
            'control_prev': self.control_prev[idx],
            'states_prev': self.states_prev[idx],
        }

class InputNet(nn.Module):
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
class InputRNNNet(nn.Module):
    def __init__(self, dropout: float, control_dim: int):
        super(InputRNNNet, self).__init__()
        self.fc1 = nn.Linear(control_dim, 32)  # 6 input features, 32 output features
        self.fc2 = nn.Linear(32, 64)  # 32 input features, 64 output features
        self.fc3 = nn.Linear(64, control_dim)  # 64 input features, 4 output features

        self.relu = nn.ReLU()  # activation function
        self.dropout = nn.Dropout(dropout)  # dropout regularization

        # LayerNorm
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.norm1(x)  # apply layer normalization
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.norm2(x)  # apply layer normalization
        x = self.fc3(x)
        return x

class DiskretizedLinear(nn.Module):
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

    def forward(self, 
                input_forces: torch.Tensor,
                states: torch.Tensor ,
                residual_errors: torch.Tensor = 0
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
        #also shift the states since the inital state needs to be shifted or if i want to do one step predictions
        delta_states_corr = states_corr - self.ssv_states
        #x_(k+1) = Ad*(x_k+e_k) + Bd*u_k
        #for compatability with torch batches we transpose the equation
        delta_states_next = torch.matmul(delta_states_corr, self.Ad.transpose(0,1)) + torch.matmul(delta_in, self.Bd.transpose(0,1)) 
        #shift the linearized states back to the states
        states_next = delta_states_next + self.ssv_states
        #dont calculate y here, rather outside since else the calculation order might be wierd
        return states_next
    
    def calc_output(self, 
            states: torch.Tensor,
            input_forces: torch.Tensor = None,
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
    


class LinearAndInputFNNConfig(DynamicIdentificationModelConfig):
    # nx: int
    # FF_dim: int
    # num_FF_layers: int
    dropout: float
    # window_size: int
    learning_rate: float
    batch_size: int
    epochs: int
    loss: Literal['mse', 'msge']
    Ad: List[List[np.float64]]
    Bd: List[List[np.float64]]
    Cd: List[List[np.float64]]
    Dd: List[List[np.float64]]
    ssv_input: List[np.float64]
    ssv_states: List[np.float64]
    patience : int
    loss_weights : Optional[List[np.float64]] = None

class LinearAndInputFNN(base.NormalizedControlStateModel):
    CONFIG = LinearAndInputFNNConfig

    def __init__(self, config: LinearAndInputFNNConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.dropout = config.dropout

        # self.window_size = config.window_size

        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs


        self.Ad = config.Ad
        self.Bd = config.Bd
        self.Cd = config.Cd
        self.Dd = config.Dd
        self.ssv_input = config.ssv_input
        self.ssv_states = config.ssv_states

        self.patience = config.patience
        self.loss_weights = config.loss_weights


        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._inputnet = InputNet(dropout = self.dropout).to(self.device)

        # self._diskretized_linear = DiskretizedLinear().to(self.device)   
        self._diskretized_linear = DiskretizedLinear(
            Ad = self.Ad,
            Bd = self.Bd,
            Cd = self.Cd,
            Dd = self.Dd,
            ssv_input= self.ssv_input,
            ssv_states= self.ssv_states,
        ).to(self.device)         

        self.optimizer = optim.Adam(
            self._inputnet.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:

        #not that it would make a difference since all parameters 
        # have requires_grad = False but just to be sure
        self._diskretized_linear.eval()
        self._inputnet.train()
        epoch_losses = []

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)
        us = utils.normalize(control_seqs, self._control_mean, self._control_std)
        ys = utils.normalize(state_seqs, self._state_mean, self._state_std)

        dataset = TimeSeriesDataset(us, ys)
        for i in range(self.epochs):

            data_loader = data.DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=False,
            )
        # ########
        # control = []
        # next_state = []
        # state = []
        # for control_seq, state_seq in zip(us, ys):
        #     # create a next state sequence by shifting state_seq
        #     next_state_seq = np.roll(state_seq, -1, 0)
        #      #put each in one continuos list and drop the last element since it is nonsense now
        #     next_state.append(next_state_seq[:-1]) 
        #     control.append(control_seq[:-1])
        #     state.append(state_seq[:-1])
            

        # control = np.vstack(control)
        # next_state = np.vstack(next_state)
        # state = np.vstack(state)

        # minbatchsize = 50
        # batchnum = int(control.shape[0]/(self.batch_size*minbatchsize))
        # control = np.resize(control,(batchnum,self.batch_size*minbatchsize,control.shape[1]))
        # next_state = np.resize(next_state,(batchnum,self.batch_size*minbatchsize,next_state.shape[1]))
        # state = np.resize(state,(batchnum,self.batch_size*minbatchsize,state.shape[1]))

        # control = torch.from_numpy(control)
        # next_state = torch.from_numpy(next_state)
        # state = torch.from_numpy(state)
        
        # for i in range(self.epochs):
        # ###############

            total_loss = 0.0
            max_batches = 0
            backward_times = []
            run_times = []
            linear_times =[]
            times = []
            time1 = time.time()

            for batch_idx, batch in enumerate(data_loader):
            # #########
            # perm = torch.randperm(control.size()[0])
            # control = control[perm]
            # next_state = next_state[perm]
            # state = state[perm]

            # batch_idx = 0
            # for batch_control, batch_next_state, batch_state in zip(control,next_state,state):
            # ##########
                time0 = time.time()
                self._inputnet.zero_grad()
                # testing = batch['FNN_input'].detach().cpu().numpy()
                # input = batch['FNN_input'].float().to(self.device)  
                # true_states = batch['Lin_input'].float().to(self.device) 
                # true_next_states = batch['next_state'].float().to(self.device) 

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                Lin_input = utils.denormalize(Lin_input, self._state_mean, self._state_std)
                next_state = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])

                # testing1 = batch['next_state'].detach().cpu().numpy()
                # testing = next_state.detach().cpu().numpy()

                input = FNN_input.float().to(self.device)  
                true_states = Lin_input.float().to(self.device) 
                true_next_states = next_state.float().to(self.device) 
                # ###############
                # input = batch_control.float().to(self.device)  
                # batch_state = utils.denormalize(batch_state, self._state_mean, self._state_std)
                # true_states = batch_state.float().to(self.device) 
                # true_next_states = batch_next_state.float().to(self.device)
                # # ###########
                input_forces = self._inputnet.forward(input)
                lin = time.time()
                states_next = self._diskretized_linear.forward(input_forces,true_states)
                states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)
                ear = time.time()

                batch_loss = F.mse_loss(
                    states_next, true_next_states
                )
                total_loss += batch_loss.item()
                back = time.time()
                batch_loss.backward()
                ward = time.time()
                self.optimizer.step()
                max_batches = batch_idx
                

                backward_times.append(ward -back)
                
                linear_times.append(ear-lin)
                # times.append(time2-time1)
                # print(batch_idx)
                timeend = time.time()
                run_times.append(timeend - time0)
                # print(f'Batch {batch_idx + 1} - Batch Loss: {batch_loss}')

                ########
                # batch_idx+=1
                ##########


            # time2 = time.time()

            loss_average = total_loss/(max_batches+1)
            logger.info(f'Epoch {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')
            print(f'Epoch {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')
            # print(
            #     f'backward time {np.mean(backward_times)} - run time mean {np.mean(run_times)}' 
            #     f'\n linear_time {np.mean(linear_times)} - times {time2-time1}'
            #     f'\n dataloader time {time2-time1-np.sum(run_times)} - run time sum {np.sum(run_times)}' 
            #     )
            epoch_losses.append([i, loss_average])

        return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64))
    
    def train_and_val(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        control_seqs_vali: List[NDArray[np.float64]],
        state_seqs_vali: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        
        loss_weights = self.loss_weights
        if loss_weights is not None:
            loss_weights = torch.from_numpy(np.array(loss_weights)).float().to(self.device)

        #not that it would make a difference since all parameters 
        # have requires_grad = False but just to be sure
        # => might actually make a difference since .eval() is not same as no grad
        self._diskretized_linear.eval()
        self._inputnet.train()
        epoch_losses = []

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)
        us = utils.normalize(control_seqs, self._control_mean, self._control_std)
        ys = utils.normalize(state_seqs, self._state_mean, self._state_std)

        #validation
        us_vali = utils.normalize(control_seqs_vali, self._control_mean, self._control_std)
        ys_vali = utils.normalize(state_seqs_vali, self._state_mean, self._state_std)
        dataset_vali = TimeSeriesDataset(us_vali, ys_vali)
        best_val_loss = torch.tensor(float('inf'))
        best_epoch = 0
        validation_losses = []

        dataset = TimeSeriesDataset(us, ys)
        for i in range(self.epochs):

            data_loader = data.DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=False,
            )

            total_loss = 0.0
            max_batches = 0
            backward_times = []
            run_times = []
            linear_times =[]
            times = []
            time1 = time.time()
            self._inputnet.train()

            for batch_idx, batch in enumerate(data_loader):
                time0 = time.time()
                self._inputnet.zero_grad()
                # testing = batch['FNN_input'].detach().cpu().numpy()
                # input = batch['FNN_input'].float().to(self.device)  
                # true_states = batch['Lin_input'].float().to(self.device) 
                # true_next_states = batch['next_state'].float().to(self.device) 

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                Lin_input = utils.denormalize(Lin_input, self._state_mean, self._state_std)
                next_state = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])

                # testing1 = batch['next_state'].detach().cpu().numpy()
                # testing = next_state.detach().cpu().numpy()

                input = FNN_input.float().to(self.device)  
                true_states = Lin_input.float().to(self.device) 
                true_next_states = next_state.float().to(self.device) 

                input_forces = self._inputnet.forward(input)
                lin = time.time()
                states_next = self._diskretized_linear.forward(input_forces,true_states)
                states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)
                ear = time.time()


                #just for sanitiy check
                # batch_loss__ = F.mse_loss(
                #     states_next, true_next_states
                # )
                # batch_loss___ = torch.mean(
                #     ((true_next_states - states_next) ** 2) * loss_weights
                #     )

                #loss calculation, when weights are all 1, they are equivalent    
                if loss_weights is None:
                    batch_loss = F.mse_loss(
                        states_next, true_next_states
                    )
                else:
                    batch_loss = torch.mean(
                        ((true_next_states - states_next) ** 2) * loss_weights
                        )

                total_loss += batch_loss.item()
                back = time.time()
                batch_loss.backward()
                ward = time.time()
                self.optimizer.step()
                max_batches = batch_idx
                

                backward_times.append(ward -back)
                
                linear_times.append(ear-lin)
                # times.append(time2-time1)
                # print(batch_idx)
                timeend = time.time()
                run_times.append(timeend - time0)
                # print(f'Batch {batch_idx + 1} - Batch Loss: {batch_loss}')




            # time2 = time.time()

            #check validation for overfitt prevention
            with torch.no_grad():
                self._inputnet.eval() #this is very important (i think to disable the dropout layers)
                controls_vali = torch.from_numpy(dataset_vali.control)
                states_vali = torch.from_numpy(dataset_vali.state)
                true_next_states_vali = torch.from_numpy(dataset_vali.next_state)

                # for some reason dataloader iteration is very slow otherwise
                #to device needs to be after denormalize else it cant calculate with the numpy std and mean
                controls_vali = controls_vali.reshape(-1,controls_vali.shape[-1]).float().to(self.device)
                states_vali = states_vali.reshape(-1,states_vali.shape[-1])
                states_vali = utils.denormalize(states_vali, self._state_mean, self._state_std).float().to(self.device)
                true_next_states_vali = true_next_states_vali.reshape(-1,true_next_states_vali.shape[-1]).float().to(self.device)

                input_forces_vali = self._inputnet.forward(controls_vali)
                states_next_vali = self._diskretized_linear.forward(input_forces_vali,states_vali)
                states_next_vali = utils.normalize(states_next_vali, _state_mean_torch, _state_std_torch)

                if loss_weights is None:
                    validation_loss = F.mse_loss(
                        states_next_vali, true_next_states_vali
                    )
                else:
                    validation_loss =torch.mean(
                        ((true_next_states_vali - states_next_vali) ** 2) * loss_weights
                        )

                # validation_loss = torch.sqrt(validation_loss) #for debug check, should be the same as the nrmse at the end when test dataset is used
                #sanity check => works
                # #all inputs into simulate_inputforces_onestep have to be unnormalized
                # controls_vali_ = controls_vali.detach().cpu()
                # controls_vali_ = utils.denormalize(controls_vali_, self._control_mean, self._control_std)
                # states_vali_ = states_vali.detach().cpu()
                # input_forces_vali_ = input_forces_vali.detach().cpu()

                # _,states_next_vali_,_ = self.simulate_inputforces_onestep(
                #     controls = controls_vali_.numpy().astype(np.float64),
                #     states = states_vali_.numpy().astype(np.float64),
                #     forces = input_forces_vali_.numpy().astype(np.float64))
                # states_next_vali_ = utils.normalize(states_next_vali_, self._state_mean, self._state_std)

                # #do the nrmse as validation loss (all states are already normalised)
                # nsquared_error = torch.square(torch.from_numpy(states_next_vali_).float().to(self.device) - true_next_states_vali)
                # # nsquared_error = torch.square(states_next_vali - true_next_states_vali)
                # nmse = torch.mean(nsquared_error, dim=0)
                # nrmse = torch.sqrt(nmse)
                # validation_loss__ = torch.sqrt(torch.mean(torch.square(nrmse)))#.item()
                # validation_loss = validation_loss__
                
                # Check if the current validation loss is the best so far
                if validation_loss < best_val_loss:
                    # If it is, save the model parameters
                    torch.save(self._inputnet.state_dict(), 'best_model_params.pth')
                    # Update the best validation loss
                    best_val_loss = validation_loss
                    best_epoch = i
                


            loss_average = total_loss/(max_batches+1)
            logger.info(f'Epoch {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')
            print(f'Epoch {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')
            # print(
            #     f'backward time {np.mean(backward_times)} - run time mean {np.mean(run_times)}' 
            #     f'\n linear_time {np.mean(linear_times)} - times {time2-time1}'
            #     f'\n dataloader time {time2-time1-np.sum(run_times)} - run time sum {np.sum(run_times)}' 
            #     )
            epoch_losses.append([i, loss_average])
            validation_loss_ = validation_loss.item()
            validation_losses.append([i, validation_loss_])

            if  self.patience < (i-best_epoch):
                print("early stopping")
                break

        #load the best parameters with best validation loss (early stopping)
        self._inputnet.load_state_dict(torch.load('best_model_params.pth'))

        return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64), 
                    validation_loss=np.array(validation_losses, dtype=np.float64), 
                    estop_epoch = best_epoch, 
                    best_val_loss = best_val_loss.item())

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')
        
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)
        self._inputnet.eval()
        self._diskretized_linear.eval()
        control_ = utils.normalize(control, self._control_mean, self._control_std)
        init_cont = utils.normalize(initial_control, self._control_mean, self._control_std)
        init_state = utils.normalize(initial_state, self._state_mean, self._state_std)

        control_ = torch.from_numpy(control_).float().to(self.device)
        init_cont = torch.from_numpy(init_cont).float().to(self.device)
        init_state = torch.from_numpy(init_state).float().to(self.device)
        with torch.no_grad():
            last_init_cont = init_cont[-1,:].unsqueeze(0)
            last_init_state = init_state[-1,:].unsqueeze(0)
            #only need the last state/control since i am not utilizing a initializer
            input_lin = self._inputnet.forward(last_init_cont)
            last_init_state = utils.denormalize(last_init_state, _state_mean_torch, _state_std_torch)

            states_next = self._diskretized_linear.forward(input_forces=input_lin,states=last_init_state)
            outputs =[]
            input_lin = self._inputnet.forward(control_)
            for in_lin in input_lin:
                outputs.append(states_next)
                states_next = self._diskretized_linear.forward(input_forces=in_lin.unsqueeze(0),states=states_next)


        outputs = torch.vstack(outputs)
        return outputs.detach().cpu().numpy().astype(np.float64)
    

    def simulate_inputforces_onestep(
        self,
        controls: NDArray[np.float64],
        states: NDArray[np.float64],
        forces: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')  

        self._inputnet.eval()
        self._diskretized_linear.eval()
        controls_ = utils.normalize(controls, self._control_mean, self._control_std)

        controls_ = torch.from_numpy(controls_).float().to(self.device)
        states_ = torch.from_numpy(states).float().to(self.device)
        forces_ = torch.from_numpy(forces).float().to(self.device)

        with torch.no_grad():
            input_forces = self._inputnet.forward(controls_)
            states_next = self._diskretized_linear.forward(input_forces,states_)
            states_next_with_true_input_forces = self._diskretized_linear.forward(forces_,states_)

        states_pred = torch.full(states_.shape, float('nan'))
        states_pred[:,1:,:] = states_next[:,:-1,:]
        states_pred_with_true_input_forces = torch.full(states_.shape, float('nan'))
        states_pred_with_true_input_forces[:,1:,:]  = states_next_with_true_input_forces[:,:-1,:]


        return (input_forces.detach().cpu().numpy().astype(np.float64),
                 states_pred.detach().cpu().numpy().astype(np.float64),
                 states_pred_with_true_input_forces.detach().cpu().numpy().astype(np.float64))


    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')
        torch.save(self._inputnet.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
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
        self._inputnet.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_std'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'pth', 'json'

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self._inputnet.parameters() if p.requires_grad)





class HybridLinearConvRNNConfig(DynamicIdentificationModelConfig):
    nx: int
    # FF_dim: int
    # num_FF_layers: int
    dropout: float
    # window_size: int
    learning_rate: float
    batch_size: int
    loss: Literal['mse', 'msge']
    Ad: List[List[np.float64]]
    Bd: List[List[np.float64]]
    Cd: List[List[np.float64]]
    Dd: List[List[np.float64]]
    ssv_input: List[np.float64]
    ssv_states: List[np.float64]
    recurrent_dim: int
    gamma: float
    beta: float
    initial_decay_parameter: float
    decay_rate: float
    epochs_with_const_decay: int
    num_recurrent_layers_init: int
    sequence_length: int
    epochs_InputFNN: int
    epochs_initializer: int
    epochs_predictor_singlestep: int
    epochs_predictor_multistep: int
    bias: bool
    log_min_max_real_eigenvalues: Optional[bool] = False
    RNNinputnetbool : bool
    forward_alt_bool : bool
    sequence_length_list : List[int]
    patience : int
    loss_weights : Optional[List[np.float64]] = None


#for some reason i called it convRNN but i meant ConstRNN
#as in constrained RNN
class HybridLinearConvRNN(base.NormalizedControlStateModel):
    CONFIG = HybridLinearConvRNNConfig

    def __init__(self, config: HybridLinearConvRNNConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.dropout = config.dropout

        # self.window_size = config.window_size

        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size

        self.Ad = config.Ad
        self.Bd = config.Bd
        self.Cd = config.Cd
        self.Dd = config.Dd
        self.ssv_input = config.ssv_input
        self.ssv_states = config.ssv_states


        self.nx = config.nx
        self.recurrent_dim = config.recurrent_dim

        self.initial_decay_parameter = config.initial_decay_parameter
        self.decay_rate = config.decay_rate
        self.epochs_with_const_decay = config.epochs_with_const_decay

        self.num_recurrent_layers_init = config.num_recurrent_layers_init

        self.sequence_length = config.sequence_length

        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor_singlestep = config.epochs_predictor_singlestep
        self.epochs_predictor_multistep = config.epochs_predictor_multistep
        self.epochs_InputFNN = config.epochs_InputFNN

        self.log_min_max_real_eigenvalues = config.log_min_max_real_eigenvalues
        self.gamma=config.gamma
        self.beta=config.beta
        self.bias=config.bias

        self.RNNinputnetbool = config.RNNinputnetbool
        self.forward_alt_bool = config.forward_alt_bool
        self.sequence_length_list = config.sequence_length_list

        self.patience = config.patience
        self.loss_weights = config.loss_weights

        #TODO:msge should probably never be used
        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')
        
        if self.forward_alt_bool:
            self._predictor = rnn.LtiRnnConvConstr(
                nx=self.nx,
                #in dimesion is state+ output of inputnet dismesion
                #TODO: make it variable when making the inputnet variable
                nu=self.state_dim+self.control_dim+1,################################
                ny=self.state_dim,
                nw=self.recurrent_dim,
                gamma=self.gamma,
                beta=self.beta,
                bias=self.bias,
            ).to(self.device)
        else:
            self._predictor = rnn.LtiRnnConvConstr(
                nx=self.nx,
                #in dimesion is state+ output of inputnet dismesion
                #TODO: make it variable when making the inputnet variable
                nu=self.state_dim+self.control_dim,################################
                ny=self.state_dim,
                nw=self.recurrent_dim,
                gamma=self.gamma,
                beta=self.beta,
                bias=self.bias,
            ).to(self.device)

        self._initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.nx,
            num_recurrent_layers=self.num_recurrent_layers_init,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self._inputnet = InputNet(dropout = self.dropout).to(self.device)

        self._inputRNNnet = InputRNNNet(dropout = self.dropout, control_dim=self.control_dim).to(self.device)

        self._diskretized_linear = DiskretizedLinear(
            Ad = self.Ad,
            Bd = self.Bd,
            Cd = self.Cd,
            Dd = self.Dd,
            ssv_input= self.ssv_input,
            ssv_states= self.ssv_states,
        ).to(self.device)         

        self.optimizer_inputFNN = optim.Adam(
            self._inputnet.parameters(), lr=self.learning_rate
        )
        if self.RNNinputnetbool:
            #learning rate could be made different between the two
            self.optimizer_pred = optim.Adam([
                {'params': self._inputRNNnet.parameters(), 'lr': self.learning_rate},
                {'params': self._predictor.parameters(), 'lr': self.learning_rate}
            ])
        else:
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

        #not that it would make a difference since all parameters 
        # have requires_grad = False but just to be sure
        # => might make a difference since .eval() is not the same as just no grad
        self._diskretized_linear.eval()
        self._inputnet.train()
        self._inputRNNnet.train()
        self._predictor.initialize_lmi()
        self._predictor.to(self.device)
        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)
        us = utils.normalize(control_seqs, self._control_mean, self._control_std)
        ys = utils.normalize(state_seqs, self._state_mean, self._state_std)

        #Initializer training
        #################################
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

            loss_average = total_loss/(len(data_loader))
            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {loss_average}'
            )
            print(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {loss_average}'
            )
            initializer_loss.append([i,np.float64(loss_average)])
        time_end_init = time.time()

        ###########################
        #InputFNN training
        #################################
        inputfnn_losses = []
        dataset = TimeSeriesDataset(us, ys)
        for i in range(self.epochs_InputFNN):

            data_loader = data.DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=False,
            )
            total_loss = 0.0
            max_batches = 0

            for batch_idx, batch in enumerate(data_loader):
                self._inputnet.zero_grad()

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                Lin_input = utils.denormalize(Lin_input, self._state_mean, self._state_std)
                next_state = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])

                input = FNN_input.float().to(self.device)  
                true_states = Lin_input.float().to(self.device) 
                true_next_states = next_state.float().to(self.device) 

                input_forces = self._inputnet.forward(input)
                states_next = self._diskretized_linear.forward(input_forces,true_states)
                states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)

                batch_loss = F.mse_loss(
                    states_next, true_next_states
                )
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_inputFNN.step()
                max_batches = batch_idx

            loss_average = total_loss/(max_batches+1)
            #TODO: sanity check if same
            loss_average_ = total_loss/(len(data_loader))
            logger.info(f'Epoch {i + 1}/{self.epochs_InputFNN} - Epoch average Loss (InputFNN): {loss_average}')
            print(f'Epoch {i + 1}/{self.epochs_InputFNN} - Epoch average Loss (InputFNN): {loss_average}')
            inputfnn_losses.append([i, loss_average])

        ###########################
        #calculate the mean and std of the output of the diskretized linear
        # (input for the RNN) in case you dont have all the states in your training set
        ################################
        #the state mean and std of the training set is not completly equal but very similar
        #so this migth be even better than that mean and std

        full_control_in = torch.from_numpy(us).float().to(self.device)
        full_states = torch.from_numpy(np.asarray(state_seqs)).float().to(self.device)

        full_forces = self._inputnet.forward(full_control_in)
        full_states_next = self._diskretized_linear.forward(full_forces,full_states)
        fsn_ = full_states_next.cpu().detach().numpy().astype(np.float64)
        self._state_mean_RNN_in, self._state_std_RNN_in = utils.mean_stddev(fsn_)
        _state_mean_RNN_in_torch = torch.from_numpy(self._state_mean_RNN_in).float().to(self.device)
        _state_std_RNN_in_torch = torch.from_numpy(self._state_std_RNN_in).float().to(self.device)
    
        ###########################
        #Predictor (ConvRNN) training
        #"one-step"
        #(and continue training of initializer network)
        #(i dont think i want to continue training of the InputFNN)
        #################################
        self._inputnet.eval()

        predictor_dataset = HybridRecurrentLinearFNNInputDataset(us, ys, self.sequence_length)

        time_start_pred = time.time()
        t = self.initial_decay_parameter
        predictor_loss: List[np.float64] = []
        min_eigenvalue: List[np.float64] = []
        max_eigenvalue: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []
        
        for i in range(self.epochs_predictor_singlestep):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
 
                #computationally efficient but not good to read and understand
                #explaination: 
                # we need to initialize the internal state of the _diskretized_linear
                # because else the first state would be x0 from the training data
                # without error and wouldnt train our RNN correctly, and since we
                # use one-step prediction we dont need any loop and can just compute 
                # the following next states in one go with the training data. Thats why
                # we need the _prev values. The lin_states then represent the internal
                # states of the "current" timestep (not the prev) and we can then output those 
                linear_inputs_prev = self._inputnet.forward(batch['control_prev'].float().to(self.device))
                linear_inputs_curr = self._inputnet.forward(batch['control'].float().to(self.device))
                states_prev_ = utils.denormalize(batch['states_prev'], self._state_mean, self._state_std).float().to(self.device)
                lin_states = self._diskretized_linear.forward(
                    input_forces= linear_inputs_prev,
                    states=states_prev_,
                    residual_errors= 0, #since it's onestep prediction, the input state has no error
                )

                #just for me  (represents the timestep such that x(k+1) becomes x(k))
                lin_in_rnn = lin_states
                
                #normalize the out_lin
                out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #with inputnet for the control inputs?
                if self.RNNinputnetbool:
                    control_in = self._inputRNNnet.forward(batch['control'].float().to(self.device))
                else:
                    control_in =batch['control'].float().to(self.device)
                #do the rnn 
                rnn_input = torch.concat((control_in, out_lin_norm),dim=2)
                if self.forward_alt_bool:
                    res_error, _ = self._predictor.forward_alt(x_pred = rnn_input, device=self.device, hx=None)
                else:
                    res_error, _ = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=None)

                res_error = res_error.to(self.device)
                #calculated the corrected output and barrier
                # it is impotant to note that batch['states'] are normalised states (see loss)
                corr_states = out_lin_norm+res_error

                #denormalize to go trough possible linear system output
                corr_states_denorm = utils.denormalize(corr_states, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #this is functionally the same as output = corr_states_denorm but is good for understanding
                #and in case there is direct feedtrough it becomes important but then also needs an extra
                #network to correct the direct feedthrough
                output = self._diskretized_linear.calc_output(
                    states= corr_states_denorm
                )
                output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)

                barrier = self._predictor.get_barrier(t).to(self.device)

                batch_loss = self.loss.forward(output_normed, batch['states'].float().to(self.device))
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()


                #stuff for constraint checking
                ################
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
                ################

                self.optimizer_pred.step()
                ########################### 
                #Constraints Checking
                #################################
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
                            f'Epoch {i+1}/{self.epochs_predictor_singlestep}\t'
                            f'max real eigenvalue of M: '
                            f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                            f'Backtracking line search exceeded maximum iteration. \t'
                            f'Constraints satisfied? {self._predictor.check_constr()}'
                        )
                        time_end_pred = time.time()
                        time_total_init = time_end_init - time_start_init
                        time_total_pred = time_end_pred - time_start_pred

                        raise ValueError(
                            "Error: Model did not complete Teacher Forcing training phase. \n"
                            "Adjust gamma, initial decay, decay rate or singlestep epochs.")

                    bls_iter += 1

            ########################### 
            #Epoch Wrapup
            #################################
            # decay t following the idea of interior point methods
            if i % self.epochs_with_const_decay == 0 and i != 0:
                t = t * 1 / self.decay_rate
                logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')

            min_ev = np.float64('inf')
            max_ev = np.float64('inf')
            if self.log_min_max_real_eigenvalues:
                min_ev, max_ev = self._predictor.get_min_max_real_eigenvalues()

            loss_average = total_loss/(len(data_loader))

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor_singlestep}\t'
                f'Total Loss (Predictor): {loss_average:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            print(
                f'Epoch {i + 1}/{self.epochs_predictor_singlestep}\t'
                f'Total Loss (Predictor): {loss_average:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            predictor_loss.append([i,np.float64(loss_average)])
            barrier_value.append(barrier.cpu().detach().numpy())
            backtracking_iter.append(np.float64(bls_iter))
            gradient_norm.append(np.float64(max_grad))
            max_eigenvalue.append(np.float64(max_ev))
            min_eigenvalue.append(np.float64(min_ev))


        ###########################
        #Predictor (ConvRNN) 
        #multi-step training
        #################################
        self._inputnet.eval()

                
        if self.RNNinputnetbool:
            #learning rate could be made different between the two
            self.optimizer_pred_multistep = optim.Adam([
                {'params': self._inputRNNnet.parameters(), 'lr': self.learning_rate},
                {'params': self._predictor.parameters(), 'lr': self.learning_rate}
            ])
        else:
            self.optimizer_pred_multistep = optim.Adam(
                self._predictor.parameters(), lr=self.learning_rate
            )

        time_start_pred = time.time()
        t = self.initial_decay_parameter
        predictor_loss_multistep: List[np.float64] = []
        min_eigenvalue: List[np.float64] = []
        max_eigenvalue: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []
        
        for i in range(self.epochs_predictor_multistep):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))

                #hopefully more understandable:
                # as we initialize the hiddenstate of the RNN we need to initialize
                # the internal state of the _diskretized_linear the first computation
                # of the error can be omitted for this since we can expect that the
                # initial state as no error 

                #we need only the last point of the x0 sequence for init of the linear
                init_control = batch['x0_control'].float().to(self.device)[:,-1:,:]
                init_state = batch['x0_states'].float().to(self.device)[:,-1:,:]
                init_input = self._inputnet.forward(init_control)
                states_next = self._diskretized_linear.forward(
                    input_forces= init_input,
                    states=init_state,
                    residual_errors= 0)
                
                #get all inputs
                if self.RNNinputnetbool:
                    control_ = self._inputRNNnet.forward(batch['control'].float().to(self.device))
                else:
                    control_ =batch['control'].float().to(self.device)
                    
                control_lin =batch['control'].float().to(self.device)
                input_lin = self._inputnet.forward(control_lin)
                x = torch.zeros((control_.shape[0], 1, self.nx),device=self.device)
                
                outputs =[]
                #get the sequence dimension, sanity check: is sequence length?
                seq_len = control_.size(dim=1)
                for seq_step in range(seq_len):
                    #seq_step:seq_step+1 preserves the original dimensionality
                    in_lin = input_lin[:,seq_step:seq_step+1,:]
                    control_in = control_[:,seq_step:seq_step+1,:]
                    
                    #just for me (represents the timestep such that x(k+1) becomes x(k))
                    lin_in_rnn = states_next

                    #normalize the out_lin
                    out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                    rnn_input = torch.concat([control_in, out_lin_norm],dim=2)

                    if self.forward_alt_bool:

                        res_error, x = self._predictor.forward_alt_onestep(x_pred = rnn_input, device=self.device, hx=x)
                    else:
                        res_error, x = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=x)
                        # hx has a very wierd format and is not the same as the output x
                        x = [[x[0],x[0]]]
                    res_error = res_error.to(self.device)
                    corr_state = out_lin_norm+res_error

                    #denormalize the corrected state and use it as new state for the linear
                    corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
                    
                    #this takes the denormalized corrected state as input
                    states_next = self._diskretized_linear.forward(
                        input_forces=in_lin,
                        states=corr_state_denorm
                        )

                    #this is functionally the same as out_lin = lin_states but is good for understanding
                    #and in case there is direct feedtrough it becomes important but then also needs an extra
                    #network to correct the direct feedthrough
                    output = self._diskretized_linear.calc_output(
                        states= corr_state_denorm
                    )
                    output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                    outputs.append(output_normed)
                    
                outputs_tensor = torch.cat(outputs, dim=1)
                barrier = self._predictor.get_barrier(t).to(self.device)

                true_state = batch['states'].float().to(self.device)
                # test1 = outputs_tensor
                # test2 = true_state
                batch_loss = self.loss.forward(outputs_tensor, true_state)
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()

                #stuff for constraint checking
                ################
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
                ################

                self.optimizer_pred_multistep.step()
                ########################### 
                #Constraints Checking
                #################################
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
                            f'Epoch {i+1}/{self.epochs_predictor_multistep}\t'
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
                            epoch_loss_predictor_multistep=np.asarray(predictor_loss_multistep),
                            inputfnn_losses=np.asarray(inputfnn_losses),
                            barrier_value=np.asarray(barrier_value),
                            backtracking_iter=np.asarray(backtracking_iter),
                            gradient_norm=np.asarray(gradient_norm),
                            max_eigenvalue=np.asarray(max_eigenvalue),
                            min_eigenvalue=np.asarray(min_eigenvalue),
                            training_time_initializer=np.asarray(time_total_init),
                            training_time_predictor=np.asarray(time_total_pred),
                        )
                    bls_iter += 1

            ########################### 
            #Epoch Wrapup
            #################################
            # decay t following the idea of interior point methods
            if i % self.epochs_with_const_decay == 0 and i != 0:
                t = t * 1 / self.decay_rate
                logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')

            min_ev = np.float64('inf')
            max_ev = np.float64('inf')
            if self.log_min_max_real_eigenvalues:
                min_ev, max_ev = self._predictor.get_min_max_real_eigenvalues()

            loss_average = total_loss/(len(data_loader))

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor_multistep}\t'
                f'Total Loss (Predictor Multistep): {loss_average:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            print(
                f'Epoch {i + 1}/{self.epochs_predictor_multistep}\t'
                f'Total Loss (Predictor Multistep): {loss_average:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            predictor_loss_multistep.append([i,np.float64(loss_average)])
            barrier_value.append(barrier.cpu().detach().numpy())
            backtracking_iter.append(np.float64(bls_iter))
            gradient_norm.append(np.float64(max_grad))
            max_eigenvalue.append(np.float64(max_ev))
            min_eigenvalue.append(np.float64(min_ev))

        ########################### 
        #training wrapup
        #################################
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
            epoch_loss_predictor_multistep=np.asarray(predictor_loss_multistep),
            inputfnn_losses=np.asarray(inputfnn_losses),
            barrier_value=np.asarray(barrier_value),
            backtracking_iter=np.asarray(backtracking_iter),
            gradient_norm=np.asarray(gradient_norm),
            max_eigenvalue=np.asarray(max_eigenvalue),
            min_eigenvalue=np.asarray(min_eigenvalue),
            training_time_initializer=np.asarray(time_total_init),
            training_time_predictor=np.asarray(time_total_pred),
        )
    
    def train_and_val(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        control_seqs_vali: List[NDArray[np.float64]],
        state_seqs_vali: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        
        loss_weights = self.loss_weights
        if loss_weights is not None:
            loss_weights = torch.from_numpy(np.array(loss_weights)).float().to(self.device)

        #not that it would make a difference since all parameters 
        # have requires_grad = False but just to be sure
        # => might make a difference since .eval() is not the same as just no grad
        self._diskretized_linear.eval()
        self._inputnet.train()
        self._inputRNNnet.train()
        self._predictor.initialize_lmi()
        self._predictor.to(self.device)
        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)
        us = utils.normalize(control_seqs, self._control_mean, self._control_std)
        ys = utils.normalize(state_seqs, self._state_mean, self._state_std)

        #validation
        us_vali = utils.normalize(control_seqs_vali, self._control_mean, self._control_std)
        ys_vali = utils.normalize(state_seqs_vali, self._state_mean, self._state_std)
        inputfnn_best_val_loss = torch.tensor(float('inf'))
        inputfnn_best_epoch = 0
        inputfnn_validation_losses = []
        predictor_best_val_loss = torch.tensor(float('inf'))
        predictor_best_epoch = 0
        predictor_validation_losses = []
        predictor_multistep_best_val_loss = torch.full((len(self.sequence_length_list),), float('inf'))
        predictor_multistep_best_epoch =  [0] * len(self.sequence_length_list)
        predictor_multistep_validation_losses = []

        #Initializer training
        #################################
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

            loss_average = total_loss/(len(data_loader))

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {loss_average}'
            )
            print(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {loss_average}'
            )
            initializer_loss.append([i,np.float64(loss_average)])
        time_end_init = time.time()

        ###########################
        #InputFNN training
        #################################
        #validation
        inputfnn_dataset_vali = TimeSeriesDataset(us_vali, ys_vali)

        inputfnn_losses = []
        inputfnn_dataset = TimeSeriesDataset(us, ys)
        for i in range(self.epochs_InputFNN):

            data_loader = data.DataLoader(
                inputfnn_dataset, self.batch_size, shuffle=True, drop_last=False,
            )
            total_loss = 0.0
            max_batches = 0

            #because of validation
            self._inputnet.train()

            for batch_idx, batch in enumerate(data_loader):
                self._inputnet.zero_grad()

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                Lin_input = utils.denormalize(Lin_input, self._state_mean, self._state_std)
                next_state = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])

                input = FNN_input.float().to(self.device)  
                true_states = Lin_input.float().to(self.device) 
                true_next_states = next_state.float().to(self.device) 

                input_forces = self._inputnet.forward(input)
                states_next = self._diskretized_linear.forward(input_forces,true_states)
                states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)

                #loss calculation, when weights are all 1, they are equivalent    
                if loss_weights is None:
                    batch_loss = F.mse_loss(
                        states_next, true_next_states
                    )
                else:
                    batch_loss = torch.mean(
                        ((true_next_states - states_next) ** 2) * loss_weights
                        )

                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_inputFNN.step()
                max_batches = batch_idx

            ########################### 
            #main validation check for overfitt prevention
            #################################
            with torch.no_grad():
                self._inputnet.eval() #this is very important (i think to disable the dropout layers)
                #just calculate the whole dataset at once 
                controls_vali = torch.from_numpy(inputfnn_dataset_vali.control)
                states_vali = torch.from_numpy(inputfnn_dataset_vali.state)
                true_next_states_vali = torch.from_numpy(inputfnn_dataset_vali.next_state)

                #to device needs to be after denormalize else it cant calculate with the numpy std and mean
                controls_vali = controls_vali.reshape(-1,controls_vali.shape[-1]).float().to(self.device)
                states_vali = states_vali.reshape(-1,states_vali.shape[-1])
                states_vali = utils.denormalize(states_vali, self._state_mean, self._state_std).float().to(self.device)
                true_next_states_vali = true_next_states_vali.reshape(-1,true_next_states_vali.shape[-1]).float().to(self.device)

                input_forces_vali = self._inputnet.forward(controls_vali)
                states_next_vali = self._diskretized_linear.forward(input_forces_vali,states_vali)
                states_next_vali = utils.normalize(states_next_vali, _state_mean_torch, _state_std_torch)

                if loss_weights is None:
                    validation_loss = F.mse_loss(
                        states_next_vali, true_next_states_vali
                    )
                else:
                    validation_loss =torch.mean(
                        ((true_next_states_vali - states_next_vali) ** 2) * loss_weights
                        )
                
                # Check if the current validation loss is the best so far
                if validation_loss < inputfnn_best_val_loss:
                    # If it is, save the model parameters
                    torch.save(self._inputnet.state_dict(), 'best_model_params.pth')
                    # Update the best validation loss
                    inputfnn_best_val_loss = validation_loss
                    inputfnn_best_epoch = i

            loss_average_ = total_loss/(max_batches+1)
            #TODO:sanity check
            loss_average = total_loss/(len(data_loader))
            logger.info(f'Epoch {i + 1}/{self.epochs_InputFNN} - Epoch average Loss (InputFNN): {loss_average}')
            print(f'Epoch {i + 1}/{self.epochs_InputFNN} - Epoch average Loss (InputFNN): {loss_average}')
            inputfnn_losses.append([i, loss_average])
            #validation
            validation_loss_ = validation_loss.item()
            inputfnn_validation_losses.append([i, validation_loss_])
            if  self.patience < (i-inputfnn_best_epoch):
                print("early stopping")
                break


        #load the best parameters with best validation loss (early stopping)
        self._inputnet.load_state_dict(torch.load('best_model_params.pth'))

        ###########################
        #calculate the mean and std of the output of the diskretized linear
        # (input for the RNN) in case you dont have all the states in your training set
        ################################
        #the state mean and std of the training set is not completly equal but very similar
        #so this migth be even better than that mean and std

        full_control_in = torch.from_numpy(us).float().to(self.device)
        full_states = torch.from_numpy(np.asarray(state_seqs)).float().to(self.device)

        full_forces = self._inputnet.forward(full_control_in)
        full_states_next = self._diskretized_linear.forward(full_forces,full_states)
        fsn_ = full_states_next.cpu().detach().numpy().astype(np.float64)
        self._state_mean_RNN_in, self._state_std_RNN_in = utils.mean_stddev(fsn_)
        _state_mean_RNN_in_torch = torch.from_numpy(self._state_mean_RNN_in).float().to(self.device)
        _state_std_RNN_in_torch = torch.from_numpy(self._state_std_RNN_in).float().to(self.device)
    
        ###########################
        #Predictor (ConvRNN) training
        #"one-step"
        #(and continue training of initializer network)
        #(i dont think i want to continue training of the InputFNN)
        #################################
        self._inputnet.eval()

        #break training loop if constraint is failed 
        # (will then fall back to last validation checkpoint)
        stop_train = False

        #validation
        predictor_dataset_vali = HybridRecurrentLinearFNNInputDataset(
            us_vali,
            ys_vali,
            sequence_length = 900, #validation sequence length should be static i think
            )

        predictor_dataset = HybridRecurrentLinearFNNInputDataset(us, ys, self.sequence_length)

        time_start_pred = time.time()
        t = self.initial_decay_parameter
        predictor_loss: List[np.float64] = []
        min_eigenvalue: List[np.float64] = []
        max_eigenvalue: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []
        
        for i in range(self.epochs_predictor_singlestep):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0

            #because of validation
            self._predictor.train()

            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
 
                #computationally efficient but not good to read and understand
                #explaination: 
                # we need to initialize the internal state of the _diskretized_linear
                # because else the first state would be x0 from the training data
                # without error and wouldnt train our RNN correctly, and since we
                # use one-step prediction we dont need any loop and can just compute 
                # the following next states in one go with the training data. Thats why
                # we need the _prev values. The lin_states then represent the internal
                # states of the "current" timestep (not the prev) and we can then output those 
                linear_inputs_prev = self._inputnet.forward(batch['control_prev'].float().to(self.device))
                linear_inputs_curr = self._inputnet.forward(batch['control'].float().to(self.device))
                states_prev_ = utils.denormalize(batch['states_prev'], self._state_mean, self._state_std).float().to(self.device)
                lin_states = self._diskretized_linear.forward(
                    input_forces= linear_inputs_prev,
                    states=states_prev_,
                    residual_errors= 0, #since it's onestep prediction, the input state has no error
                )
 
                #just for me  (represents the timestep such that x(k+1) becomes x(k))
                lin_in_rnn = lin_states               

                #normalize the out_lin
                out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #with inputnet for the control inputs?
                if self.RNNinputnetbool:
                    control_in = self._inputRNNnet.forward(batch['control'].float().to(self.device))
                else:
                    control_in =batch['control'].float().to(self.device)
                #do the rnn 
                rnn_input = torch.concat((control_in, out_lin_norm),dim=2)
                if self.forward_alt_bool:
                    res_error, _ = self._predictor.forward_alt(x_pred = rnn_input, device=self.device, hx=None)
                else:
                    res_error, _ = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=None)

                res_error = res_error.to(self.device)
                #calculated the corrected output and barrier
                # it is impotant to note that batch['states'] are normalised states (see loss)
                corr_states = out_lin_norm+res_error

                #denormalize to go trough possible linear system output
                corr_states_denorm = utils.denormalize(corr_states, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #this is functionally the same as output = corr_states_denorm but is good for understanding
                #and in case there is direct feedtrough it becomes important but then also needs an extra
                #network to correct the direct feedthrough
                output = self._diskretized_linear.calc_output(
                    states= corr_states_denorm
                )
                output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)


                barrier = self._predictor.get_barrier(t).to(self.device)

                #loss calculation, when weights are all 1, they are equivalent 
                if loss_weights is None:
                    batch_loss = self.loss.forward(
                        output_normed, batch['states'].float().to(self.device)
                        )
                else:
                    batch_loss = torch.mean(
                        ((batch['states'].float().to(self.device) - output_normed) ** 2) * loss_weights
                        )
                
                
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()


                #stuff for constraint checking
                ################
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
                ################

                self.optimizer_pred.step()
                ########################### 
                #Constraints Checking
                #################################
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
                            f'Epoch {i+1}/{self.epochs_predictor_singlestep}\t'
                            f'max real eigenvalue of M: '
                            f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                            f'Backtracking line search exceeded maximum iteration. \t'
                            f'Constraints satisfied? {self._predictor.check_constr()}'
                        )
                        time_end_pred = time.time()
                        time_total_init = time_end_init - time_start_init
                        time_total_pred = time_end_pred - time_start_pred

                        #Trainings that ended here seemed to result in worse models,
                        # also it circumvents the validation so it wont check if an earlier
                        # epoch was better => let it break the training here  
                        stop_train = True
                        break

                    bls_iter += 1

                if stop_train:
                    break
            if stop_train:
                break

            ########################### 
            #main validation check for overfitt prevention
            #################################
            with torch.no_grad():
                self._predictor.eval() #this is very important (i think to disable the dropout layers)

                #do the whole validation dataset at once
                states = torch.from_numpy(predictor_dataset_vali.states)
                control = torch.from_numpy(predictor_dataset_vali.control)
                x0 = torch.from_numpy(predictor_dataset_vali.x0)
                control_prev = torch.from_numpy(predictor_dataset_vali.control_prev)
                states_prev = torch.from_numpy(predictor_dataset_vali.states_prev)

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(x0.float().to(self.device))
                linear_inputs_prev = self._inputnet.forward(control_prev.float().to(self.device))
                linear_inputs_curr = self._inputnet.forward(control.float().to(self.device))
                states_prev_ = utils.denormalize(states_prev, self._state_mean, self._state_std).float().to(self.device)
                lin_states = self._diskretized_linear.forward(
                    input_forces= linear_inputs_prev,
                    states=states_prev_,
                    residual_errors= 0, #since it's onestep prediction, the input state has no error
                )

                #just for me  (represents the timestep such that x(k+1) becomes x(k))
                lin_in_rnn = lin_states
                
                #normalize the out_lin
                out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #with inputnet for the control inputs?
                if self.RNNinputnetbool:
                    control_in = self._inputRNNnet.forward(control.float().to(self.device))
                else:
                    control_in = control.float().to(self.device)
                #do the rnn 
                rnn_input = torch.concat((control_in, out_lin_norm),dim=2)
                if self.forward_alt_bool:
                    res_error, _ = self._predictor.forward_alt(x_pred = rnn_input, device=self.device, hx=None)
                else:
                    res_error, _ = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=None)
                res_error = res_error.to(self.device)
                corr_states = out_lin_norm+res_error

                #denormalize to go trough possible linear system output
                corr_states_denorm = utils.denormalize(corr_states, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #this is functionally the same as output = corr_states_denorm but is good for understanding
                #and in case there is direct feedtrough it becomes important but then also needs an extra
                #network to correct the direct feedthrough
                output = self._diskretized_linear.calc_output(
                    states= corr_states_denorm
                )
                output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)

                barrier = self._predictor.get_barrier(t).to(self.device)

                #loss calculation, when weights are all 1, they are equivalent 
                if loss_weights is None:
                    validation_loss = self.loss.forward(
                        output_normed, states.float().to(self.device)
                        )
                else:
                    #TODO: check if dimesnions are correct
                    validation_loss = torch.mean(
                        ((states.float().to(self.device) - output_normed) ** 2) * loss_weights
                        )

                # Check if the current validation loss is the best so far
                if validation_loss < predictor_best_val_loss:
                    # If it is, save the model parameters
                    torch.save(self._predictor.state_dict(), 'best_model_params.pth')
                    # Update the best validation loss
                    predictor_best_val_loss = validation_loss
                    predictor_best_epoch = i

            ########################### 
            #Epoch Wrapup
            #################################
            # decay t following the idea of interior point methods
            if i % self.epochs_with_const_decay == 0 and i != 0:
                t = t * 1 / self.decay_rate
                logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')

            min_ev = np.float64('inf')
            max_ev = np.float64('inf')
            if self.log_min_max_real_eigenvalues:
                min_ev, max_ev = self._predictor.get_min_max_real_eigenvalues()

            loss_average = total_loss/(len(data_loader))

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor_singlestep}\t'
                f'Total Loss (Predictor): {loss_average:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            print(
                f'Epoch {i + 1}/{self.epochs_predictor_singlestep}\t'
                f'Total Loss (Predictor): {loss_average:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            predictor_loss.append([i,np.float64(loss_average)])
            barrier_value.append(barrier.cpu().detach().numpy())
            backtracking_iter.append(np.float64(bls_iter))
            gradient_norm.append(np.float64(max_grad))
            max_eigenvalue.append(np.float64(max_ev))
            min_eigenvalue.append(np.float64(min_ev))

            #validation
            validation_loss_ = validation_loss.item()
            predictor_validation_losses.append([i, validation_loss_])

            if  self.patience < (i-predictor_best_epoch):
                print("early stopping")
                break

        #load the best parameters with best validation loss (early stopping)
        self._predictor.load_state_dict(torch.load('best_model_params.pth'))

        ###########################
        #Predictor (ConvRNN) 
        #multi-step training
        #################################
        self._inputnet.eval()
        
        #let the sequence length rise slowley and always when
        # validation early stopping triggers or constraints are violated 
        #=> Due to the Validation loss beeing on a fixed sequence it is not affected
        #   by the change of the loss due to different sequence length
        #       => sequence length does not effect loss but the sumation 
        #          over all batches without division over the number of batches does
        for index, seq_len in enumerate(self.sequence_length_list):
            #break training loop if constraint is failed 
            # (will then fall back to last validation checkpoint)
            stop_train = False

            #make dataset with rising sequence length
            predictor_dataset = HybridRecurrentLinearFNNInputDataset(us, ys, seq_len)
                    
            #TODO: does it make sense to reset the optimizer each time 
            #      => at least as long as the sequence length affects loss calculation
            #       => sequence length does not effect loss but the sumation 
            #          over all batches without division over the number of batches does
            #reset optimizer
            if self.RNNinputnetbool:
                #learning rate could be made different between the two
                self.optimizer_pred_multistep = optim.Adam([
                    {'params': self._inputRNNnet.parameters(), 'lr': self.learning_rate},
                    {'params': self._predictor.parameters(), 'lr': self.learning_rate}
                ])
            else:
                self.optimizer_pred_multistep = optim.Adam(
                    self._predictor.parameters(), lr=self.learning_rate
                )


            time_start_pred = time.time()
            t = self.initial_decay_parameter
            predictor_loss_multistep: List[np.float64] = []
            min_eigenvalue: List[np.float64] = []
            max_eigenvalue: List[np.float64] = []
            barrier_value: List[np.float64] = []
            gradient_norm: List[np.float64] = []
            backtracking_iter: List[np.float64] = []
            
            for i in range(self.epochs_predictor_multistep):
                data_loader = data.DataLoader(
                    predictor_dataset, self.batch_size, shuffle=True, drop_last=True
                )
                total_loss = 0
                max_grad = 0

                #because of validation
                self._predictor.train()

                for batch_idx, batch in enumerate(data_loader):
                    self._predictor.zero_grad()

                    # Initialize predictor with state of initializer network
                    _, hx = self._initializer.forward(batch['x0'].float().to(self.device))

                    #hopefully more understandable:
                    # as we initialize the hiddenstate of the RNN we need to initialize
                    # the internal state of the _diskretized_linear the first computation
                    # of the error can be omitted for this since we can expect that the
                    # initial state as no error 

                    #we need only the last point of the x0 sequence for init of the linear
                    init_control = batch['x0_control'].float().to(self.device)[:,-1:,:]
                    init_state = batch['x0_states'].float().to(self.device)[:,-1:,:]
                    init_input = self._inputnet.forward(init_control)
                    states_next = self._diskretized_linear.forward(
                        input_forces= init_input,
                        states=init_state,
                        residual_errors= 0)
                    
                    #get all inputs
                    if self.RNNinputnetbool:
                        control_ = self._inputRNNnet.forward(batch['control'].float().to(self.device))
                    else:
                        control_ =batch['control'].float().to(self.device)
                    control_lin =batch['control'].float().to(self.device)
                    input_lin = self._inputnet.forward(control_lin)
                    x = torch.zeros((control_.shape[0], 1, self.nx),device=self.device)
                    
                    outputs =[]
                    #get the sequence dimension, sanity check: is sequence length?
                    seq_len = control_.size(dim=1)
                    for seq_step in range(seq_len):
                        #seq_step:seq_step+1 preserves the original dimensionality
                        in_lin = input_lin[:,seq_step:seq_step+1,:]
                        control_in = control_[:,seq_step:seq_step+1,:]
                        
                        #just for me (represents the timestep such that x(k+1) becomes x(k))
                        lin_in_rnn = states_next
                        
                        #normalize the out_lin
                        out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                        rnn_input = torch.concat([control_in, out_lin_norm],dim=2)
                        if self.forward_alt_bool:
                            res_error, x = self._predictor.forward_alt_onestep(x_pred = rnn_input, device=self.device, hx=x)
                        else:
                            res_error, x = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=x)
                            # hx has a very wierd format and is not the same as the output x
                            x = [[x[0],x[0]]]
                        res_error = res_error.to(self.device)
                        corr_state = out_lin_norm+res_error

                        #denormalize the corrected state and use it as new state for the linear
                        corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
                        
                        #this takes the denormalized corrected state as input
                        states_next = self._diskretized_linear.forward(
                            input_forces=in_lin,
                            states=corr_state_denorm
                            )
                        
                        #this is functionally the same as out_lin = lin_states but is good for understanding
                        #and in case there is direct feedtrough it becomes important but then also needs an extra
                        #network to correct the direct feedthrough
                        output = self._diskretized_linear.calc_output(
                            states= corr_state_denorm
                        )
                        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                        outputs.append(output_normed)


                    outputs_tensor = torch.cat(outputs, dim=1)
                    barrier = self._predictor.get_barrier(t).to(self.device)

                    true_state = batch['states'].float().to(self.device)
                    # test1 =outputs_tensor
                    # test2 =true_state

                    #loss calculation, when weights are all 1, they are equivalent    
                    if loss_weights is None:
                        batch_loss = self.loss.forward(
                            outputs_tensor, true_state
                        )
                    else:
                        batch_loss = torch.mean(
                            ((true_state - outputs_tensor) ** 2) * loss_weights
                            )

                    total_loss += batch_loss.item()
                    (batch_loss + barrier).backward()

                    #stuff for constraint checking
                    ################
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
                    ################

                    self.optimizer_pred_multistep.step()
                    ########################### 
                    #Constraints Checking
                    #################################
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
                                f'Epoch {i+1}/{self.epochs_predictor_multistep}\t'
                                f'max real eigenvalue of M: '
                                f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                                f'Backtracking line search exceeded maximum iteration. \t'
                                f'Constraints satisfied? {self._predictor.check_constr()}'
                            )
                            time_end_pred = time.time()
                            time_total_init = time_end_init - time_start_init
                            time_total_pred = time_end_pred - time_start_pred

                            #Trainings that ended here seemed to result in worse models,
                            # also it circumvents the validation so it wont check if an earlier
                            # epoch was better => let it break the training here  
                            stop_train = True
                            break

                            return dict(
                                index=np.asarray(i),
                                epoch_loss_initializer=np.asarray(initializer_loss),
                                epoch_loss_predictor=np.asarray(predictor_loss),
                                epoch_loss_predictor_multistep=np.asarray(predictor_loss_multistep),
                                inputfnn_losses=np.asarray(inputfnn_losses),
                                barrier_value=np.asarray(barrier_value),
                                backtracking_iter=np.asarray(backtracking_iter),
                                gradient_norm=np.asarray(gradient_norm),
                                max_eigenvalue=np.asarray(max_eigenvalue),
                                min_eigenvalue=np.asarray(min_eigenvalue),
                                training_time_initializer=np.asarray(time_total_init),
                                training_time_predictor=np.asarray(time_total_pred),
                                #validation
                                inputfnn_val_loss=np.array(inputfnn_validation_losses, dtype=np.float64), 
                                inputfnn_best_epoch = inputfnn_best_epoch, 
                                inputfnn_best_val_loss = inputfnn_best_val_loss.item(),
                                predictor_val_loss=np.array(predictor_validation_losses, dtype=np.float64), 
                                predictor_best_epoch = predictor_best_epoch, 
                                predictor_best_val_loss = predictor_best_val_loss.item(),
                                predictor_multistep_val_loss=np.array(predictor_multistep_validation_losses, dtype=np.float64), 
                                predictor_multistep_best_epoch = predictor_multistep_best_epoch, 
                                predictor_multistep_best_val_loss = predictor_multistep_best_val_loss.item(),
                            )
                        bls_iter += 1

                    if stop_train:
                        break
                if stop_train:
                    break
                ########################### 
                #main validation check for overfitt prevention
                #################################
                with torch.no_grad():
                    self._predictor.eval() #this is very important (i think to disable the dropout layers)

                    #do the whole validation dataset at once
                    states = torch.from_numpy(predictor_dataset_vali.states)
                    control = torch.from_numpy(predictor_dataset_vali.control)
                    x0 = torch.from_numpy(predictor_dataset_vali.x0)
                    control_prev = torch.from_numpy(predictor_dataset_vali.control_prev)
                    states_prev = torch.from_numpy(predictor_dataset_vali.states_prev)
                    x0_control = torch.from_numpy(predictor_dataset_vali.x0_control)
                    x0_states = torch.from_numpy(predictor_dataset_vali.x0_states)

                    # Initialize predictor with state of initializer network
                    _, hx = self._initializer.forward(x0.float().to(self.device))

                    #we need only the last point of the x0 sequence for init of the linear
                    init_control = x0_control.float().to(self.device)[:,-1:,:]
                    init_state = x0_states.float().to(self.device)[:,-1:,:]
                    init_input = self._inputnet.forward(init_control)
                    states_next = self._diskretized_linear.forward(
                        input_forces= init_input,
                        states=init_state,
                        residual_errors= 0)
                    
                    #get all inputs
                    if self.RNNinputnetbool:
                        control_ = self._inputRNNnet.forward(control.float().to(self.device))
                    else:
                        control_ =control.float().to(self.device)

                    control_lin =control.float().to(self.device)
                    input_lin = self._inputnet.forward(control_lin)
                    x = torch.zeros((control_.shape[0], 1, self.nx),device=self.device)
                    
                    outputs =[]
                    #get the sequence dimension, sanity check: is sequence length?
                    seq_len = control_.size(dim=1)
                    for seq_step in range(seq_len):
                        #seq_step:seq_step+1 preserves the original dimensionality
                        in_lin = input_lin[:,seq_step:seq_step+1,:]
                        control_in = control_[:,seq_step:seq_step+1,:]
                        
                        #just for me (represents the timestep such that x(k+1) becomes x(k))
                        lin_in_rnn = states_next
                        
                        #normalize the out_lin
                        out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                        rnn_input = torch.concat([control_in, out_lin_norm],dim=2)
                        if self.forward_alt_bool:
                            res_error, x = self._predictor.forward_alt_onestep(x_pred = rnn_input, device=self.device, hx=x)
                        else:
                            res_error, x = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=x)
                            # hx has a very wierd format and is not the same as the output x
                            x = [[x[0],x[0]]]                           
                        res_error = res_error.to(self.device)
                        corr_state = out_lin_norm+res_error

                        #denormalize the corrected state and use it as new state for the linear
                        corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
                        states_next = self._diskretized_linear.forward(
                            input_forces=in_lin,
                            states=corr_state_denorm
                            )
                        
                        #this is functionally the same as out_lin = lin_states but is good for understanding
                        #and in case there is direct feedtrough it becomes important but then also needs an extra
                        #network to correct the direct feedthrough
                        output = self._diskretized_linear.calc_output(
                            states= corr_state_denorm
                        )
                        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                        outputs.append(output_normed)                    

                    outputs_tensor = torch.cat(outputs, dim=1)
                    barrier = self._predictor.get_barrier(t).to(self.device)

                    true_state = states.float().to(self.device)
                    # test1 =outputs_tensor
                    # test2 =true_state

                    #loss calculation, when weights are all 1, they are equivalent    
                    if loss_weights is None:
                        validation_loss = self.loss.forward(
                            outputs_tensor, true_state
                        )
                    else:
                        validation_loss = torch.mean(
                            ((true_state - outputs_tensor) ** 2) * loss_weights
                            )

                    if validation_loss < predictor_multistep_best_val_loss[index]:
                        # If it is, save the model parameters
                        torch.save(self._predictor.state_dict(), 'best_model_params.pth')
                        # Update the best validation loss
                        predictor_multistep_best_val_loss[index] = validation_loss
                        predictor_multistep_best_epoch[index] = i

                ########################### 
                #Epoch Wrapup
                #################################
                # decay t following the idea of interior point methods
                if i % self.epochs_with_const_decay == 0 and i != 0:
                    t = t * 1 / self.decay_rate
                    logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')

                min_ev = np.float64('inf')
                max_ev = np.float64('inf')
                if self.log_min_max_real_eigenvalues:
                    min_ev, max_ev = self._predictor.get_min_max_real_eigenvalues()

                loss_average = total_loss/(len(data_loader))

                logger.info(
                    f'Epoch {i + 1}/{self.epochs_predictor_multistep}\t'
                    f'Total Loss (Predictor Multistep): {loss_average:1f} \t'
                    f'Barrier: {barrier:1f}\t'
                    f'Backtracking Line Search iteration: {bls_iter}\t'
                    f'Max accumulated gradient norm: {max_grad:1f}'
                )
                print(
                    f'Epoch {i + 1}/{self.epochs_predictor_multistep}\t'
                    f'Total Loss (Predictor Multistep): {loss_average:1f} \t'
                    f'Barrier: {barrier:1f}\t'
                    f'Backtracking Line Search iteration: {bls_iter}\t'
                    f'Max accumulated gradient norm: {max_grad:1f}'
                )
                predictor_loss_multistep.append([i,np.float64(loss_average)])
                barrier_value.append(barrier.cpu().detach().numpy())
                backtracking_iter.append(np.float64(bls_iter))
                gradient_norm.append(np.float64(max_grad))
                max_eigenvalue.append(np.float64(max_ev))
                min_eigenvalue.append(np.float64(min_ev))

                #validation
                validation_loss_ = validation_loss.item()
                predictor_multistep_validation_losses.append([i, validation_loss_])

                if  self.patience < (i-predictor_multistep_best_epoch[index]):
                    print("early stopping")
                    break

        #load the best parameters with best validation loss (early stopping)
        self._predictor.load_state_dict(torch.load('best_model_params.pth'))

        ########################### 
        #training wrapup
        #################################
        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        #with validation
        return dict(
            index=np.asarray(i),
            inputfnn_losses=np.asarray(inputfnn_losses),
            inputfnn_val_loss=np.array(inputfnn_validation_losses, dtype=np.float64), 
            inputfnn_best_epoch = inputfnn_best_epoch, 
            inputfnn_best_val_loss = inputfnn_best_val_loss.item(),
            epoch_loss_initializer=np.asarray(initializer_loss),
            training_time_initializer=np.asarray(time_total_init),
            epoch_loss_predictor=np.asarray(predictor_loss),
            predictor_val_loss=np.array(predictor_validation_losses, dtype=np.float64), 
            predictor_best_epoch = predictor_best_epoch, 
            predictor_best_val_loss = predictor_best_val_loss.item(),
            epoch_loss_predictor_multistep=np.asarray(predictor_loss_multistep),
            predictor_multistep_val_loss=np.array(predictor_multistep_validation_losses, dtype=np.float64),
            predictor_multistep_best_epoch = np.array(predictor_multistep_best_epoch),  
            predictor_multistep_best_val_loss = np.array(predictor_multistep_best_val_loss.cpu().detach().numpy()),
            training_time_predictor=np.asarray(time_total_pred),
            barrier_value=np.asarray(barrier_value),
            backtracking_iter=np.asarray(backtracking_iter),
            gradient_norm=np.asarray(gradient_norm),
            max_eigenvalue=np.asarray(max_eigenvalue),
            min_eigenvalue=np.asarray(min_eigenvalue),
        )


    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> Union[
        NDArray[np.float64], Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]
    ]:
        if (
            self.control_mean is None
            or self.control_std is None
            or self.state_mean is None
            or self.state_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)
        _state_mean_RNN_in_torch = torch.from_numpy(self._state_mean_RNN_in).float().to(self.device)
        _state_std_RNN_in_torch = torch.from_numpy(self._state_std_RNN_in).float().to(self.device)

        self._diskretized_linear.eval()
        self._inputRNNnet.eval()
        self._inputnet.eval()
        self._initializer.eval()
        self._predictor.eval()

        self._inputnet.to(self.device)
        self._inputRNNnet.to(self.device)
        self._diskretized_linear.to(self.device)
        self._predictor.to(self.device)
        self._initializer.to(self.device)

        init_cont = utils.normalize(initial_control, self.control_mean, self.control_std)
        init_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control_ = utils.normalize(control, self.control_mean, self.control_std)


        control_ = torch.from_numpy(control_).float().to(self.device)
        #put it in batch,sequence,state format
        control_ = control_.unsqueeze(0)
        init_cont = torch.from_numpy(init_cont)
        init_state = torch.from_numpy(init_state)

        with torch.no_grad():
            last_init_cont = init_cont[-1,:].unsqueeze(0).float().to(self.device)
            last_init_state = init_state[-1,:].unsqueeze(0).float().to(self.device)
            
            init_input_lin_ = self._inputnet.forward(last_init_cont)
            last_init_state = utils.denormalize(last_init_state, _state_mean_torch, _state_std_torch)

            #init the diskretized_linear internal state
            states_next = self._diskretized_linear.forward(
                input_forces=init_input_lin_,
                states=last_init_state, #initial state has no error
                )
            #needs batch and sequence format
            states_next = states_next.unsqueeze(0)

            init_x = (
                torch.from_numpy(np.hstack((init_cont[1:], init_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            #init the hidden state of our RNN
            _, hx = self._initializer.forward(init_x)
            x = torch.zeros((1, 1, self.nx),device=self.device)

            input_lin = self._inputnet.forward(control_)

            if self.RNNinputnetbool:
                control_ = self._inputRNNnet.forward(control_)

            outputs =[]

            # split the tensors along the second dimension 
            input_lin_list = torch.unbind(input_lin, dim=1)
            control_list = torch.unbind(control_, dim=1)

            # iterate over the resulting tensors
            for in_lin, control_in in zip(input_lin_list, control_list):
                
                #needs batch and sequence format
                in_lin_ = torch.unsqueeze(in_lin, dim=1)
                control_in_ = torch.unsqueeze(control_in, dim=1)

                #just for me  (represents the timestep such that x(k+1) becomes x(k))
                lin_in_rnn = states_next
                
                out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                rnn_input = torch.concat([control_in_, out_lin_norm],dim=2)
                if self.forward_alt_bool:
                    res_error, x = self._predictor.forward_alt_onestep(x_pred = rnn_input, device=self.device, hx=x)
                else:
                    res_error, x = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=x)
                    # hx has a very wierd format and is not the same as the output x
                    x = [[x[0],x[0]]]
                res_error = res_error.to(self.device)
                corr_state = out_lin_norm+res_error
                corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                #this is functionally the same as out_lin = lin_states but is good for understanding
                #and in case there is direct feedtrough it becomes important but then also needs an extra
                #network to correct the direct feedthrough
                output = self._diskretized_linear.calc_output(
                    states= corr_state_denorm
                )
                outputs.append(output)

                #this takes the denormalized corrected state as input
                states_next = self._diskretized_linear.forward(
                    input_forces=in_lin_,
                    states=corr_state_denorm,
                    )

            
            outputs_tensor = torch.cat(outputs, dim=1)
            y_np: NDArray[np.float64] = (
                outputs_tensor.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        return y_np
    
    
    def simulate_inputforces_onestep(
        self,
        controls: NDArray[np.float64],
        states: NDArray[np.float64],
        forces: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be simulated.')   

        self._inputnet.eval()
        self._inputRNNnet.eval()
        self._diskretized_linear.eval()
        self._predictor.eval()
        self._initializer.eval()

        self._inputnet.to(self.device)
        self._inputRNNnet.to(self.device)
        self._diskretized_linear.to(self.device)
        self._predictor.to(self.device)
        self._initializer.to(self.device)

        controls_ = utils.normalize(controls, self._control_mean, self._control_std)
        states_normed_ = utils.normalize(states, self._state_mean, self._state_std)

        _state_mean_RNN_in_torch = torch.from_numpy(self._state_mean_RNN_in).float().to(self.device)
        _state_std_RNN_in_torch = torch.from_numpy(self._state_std_RNN_in).float().to(self.device)
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)

        controls_ = torch.from_numpy(controls_).float().to(self.device)
        states_ = torch.from_numpy(states).float().to(self.device)
        forces_ = torch.from_numpy(forces).float().to(self.device)
        states_normed_ = torch.from_numpy(states_normed_).float().to(self.device)

        x0_control = controls_[:, :50, :]
        x0_states_normed_ = states_normed_[:, :50, :]
        x0_init = torch.cat((x0_control, x0_states_normed_), axis=-1)

        #the 49 incooperates the init of the diskretized linear 
        #and we then drop the last state since, while we can compute
        # the last next state we dont have meassurements there 
        prev_cont_in = controls_[:, 49:, :]
        lin_state_in = states_[:, 49:, :]
        lin_forces_in = forces_[:, 49:, :]

        #for the RNN we now need the values from 50 onward
        curr_cont_in =controls_[:, 50:, :]
        if self.RNNinputnetbool:
            curr_cont_in = self._inputRNNnet.forward(curr_cont_in)

        curr_cont_in_lin =controls_[:, 50:, :]

        with torch.no_grad():
            prev_input_forces = self._inputnet.forward(prev_cont_in)

            #drop last next state to fit with RNN inputs and because see previous comment about 49
            states_next = self._diskretized_linear.forward(prev_input_forces,lin_state_in)[:,:-1,:]
            states_next_with_true_input_forces = self._diskretized_linear.forward(lin_forces_in,lin_state_in)[:,:-1,:]
            
            curr_input_forces_ = self._inputnet.forward(curr_cont_in_lin)

            #just for me  (represents the timestep such that x(k+1) becomes x(k))
            lin_in_rnn = states_next
            states_with_true_input_forces = states_next_with_true_input_forces
            
            #again the problem with the hidden state:
            #   altough the diskretized linear does one step prediction
            #   the RNN has to do a sequence, but since idealy it corrects
            #   the diskretized linears predicted state perfectly, the fact 
            #   that the diskretized linear always gets the true state could be
            #   what the RNN is trained for.
            
            _, hx = self._initializer.forward(x0_init)
            out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
            rnn_input = torch.concat((curr_cont_in,out_lin_norm),dim=2)
            if self.forward_alt_bool:
                res_error, _ = self._predictor.forward_alt(x_pred = rnn_input, device=self.device, hx=None)
            else:
                res_error, _ = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=None)
            res_error = res_error.to(self.device)
            #I DONT denormalise the res_error, I can simply denormalise the corrected states
            pred_states_ = out_lin_norm + res_error
            pred_states_ = utils.denormalize(pred_states_, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

            #this is functionally the same as output = pred_states_ but is good for understanding
            #and in case there is direct feedtrough it becomes important but then also needs an extra
            #network to correct the direct feedthrough
            output = self._diskretized_linear.calc_output(
                states = pred_states_
                )
            true_input_pred_states_ = self._diskretized_linear.calc_output(
                states = states_with_true_input_forces,
                )

            filler_forces_ = self._inputnet.forward(x0_control)

        #fill the start that is used for initialisation with nans
        filler_nans_states = torch.full(x0_states_normed_.shape, float('nan')).to(self.device)
        filler_nans_cont = torch.full(filler_forces_.shape, float('nan')).to(self.device)
        pred_states = torch.concat((filler_nans_states,output),dim=1)
        true_input_pred_states = torch.concat((filler_nans_states,true_input_pred_states_),dim=1)
        curr_input_forces = torch.concat((filler_nans_cont,curr_input_forces_),dim=1)

        return (curr_input_forces.detach().cpu().numpy().astype(np.float64),
                 pred_states.detach().cpu().numpy().astype(np.float64),
                 true_input_pred_states.detach().cpu().numpy().astype(np.float64))


    def simulate_inputforces_multistep(
        self,
        controls: NDArray[np.float64],
        states: NDArray[np.float64],
        forces: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if (
            self.control_mean is None
            or self.control_std is None
            or self.state_mean is None
            or self.state_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')
        self._inputnet.eval()
        self._inputRNNnet.eval()
        self._diskretized_linear.eval()
        self._predictor.eval()
        self._initializer.eval()

        self._inputnet.to(self.device)
        self._inputRNNnet.to(self.device)
        self._diskretized_linear.to(self.device)
        self._predictor.to(self.device)
        self._initializer.to(self.device)

        controls_ = utils.normalize(controls, self._control_mean, self._control_std)
        states_normed_ = utils.normalize(states, self._state_mean, self._state_std)

        _state_mean_RNN_in_torch = torch.from_numpy(self._state_mean_RNN_in).float().to(self.device)
        _state_std_RNN_in_torch = torch.from_numpy(self._state_std_RNN_in).float().to(self.device)
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)

        controls_ = torch.from_numpy(controls_).float().to(self.device)
        states_ = torch.from_numpy(states).float().to(self.device)
        states_normed_ = torch.from_numpy(states_normed_).float().to(self.device)

        x0_control = controls_[:, :50, :]
        x0_states_normed_ = states_normed_[:, :50, :]
        x0_init = torch.cat((x0_control, x0_states_normed_), axis=-1)

        control_ = controls_[:, 50:,:]

        with torch.no_grad():
            #TODO: does it take only one? and the right one?
            last_init_cont = controls_[:, 49:50, :].float().to(self.device)
            #states_ are the not normalised states
            last_init_state = states_[:, 49:50, :].float().to(self.device)
            
            init_input_lin_ = self._inputnet.forward(last_init_cont)

            #init the diskretized_linear internal state
            states_next = self._diskretized_linear.forward(
                input_forces=init_input_lin_,
                states=last_init_state, #initial state has no error
                )

            #TODO: make this correct 
            # init_x = (
            #     torch.from_numpy(np.hstack((init_cont[1:], init_state[:-1])))
            #     .unsqueeze(0)
            #     .float()
            #     .to(self.device)
            # )
            init_x = x0_init

            #init the hidden state of our RNN
            _, hx = self._initializer.forward(init_x)
            x = torch.zeros((control_.shape[0], 1, self.nx),device=self.device)

            input_lin = self._inputnet.forward(control_)

            if self.RNNinputnetbool:
                control_ = self._inputRNNnet.forward(control_)

            outputs =[]
            filler_forces_ = self._inputnet.forward(x0_control)

            # split the tensors along the second dimension
            input_lin_list = torch.unbind(input_lin, dim=1)
            control_list = torch.unbind(control_, dim=1)

            # iterate over the resulting tensors
            for in_lin, control_in in zip(input_lin_list, control_list):
                
                #needs batch and sequence format
                in_lin_ = torch.unsqueeze(in_lin, dim=1)
                control_in_ = torch.unsqueeze(control_in, dim=1)

                #just for me  (represents the timestep such that x(k+1) becomes x(k))
                lin_in_rnn = states_next

                out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

                rnn_input = torch.concat([control_in_, out_lin_norm],dim=2)
                if self.forward_alt_bool:
                    res_error, x = self._predictor.forward_alt_onestep(x_pred = rnn_input, device=self.device, hx=x)
                else:
                    res_error, x = self._predictor.forward(x_pred = rnn_input, device=self.device, hx=x)
                    # hx has a very wierd format and is not the same as the output x
                    x = [[x[0],x[0]]]
                res_error = res_error.to(self.device)
                corr_state = out_lin_norm+res_error
                corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
                
                #this is functionally the same as out_lin = lin_states but is good for understanding
                #and in case there is direct feedtrough it becomes important but then also needs an extra
                #network to correct the direct feedthrough
                output = self._diskretized_linear.calc_output(
                    states = corr_state_denorm,
                )
                outputs.append(output)
                #this takes the denormalized corrected state as input and calculates the next state
                states_next = self._diskretized_linear.forward(
                    input_forces=in_lin_,
                    states=corr_state_denorm,
                    )
            outputs_tensor = torch.cat(outputs, dim=1)

        #fill the start that is used for initialisation with nans
        filler_nans_states = torch.full(x0_states_normed_.shape, float('nan')).to(self.device)
        filler_nans_cont = torch.full(filler_forces_.shape, float('nan')).to(self.device)
        pred_states = torch.concat((filler_nans_states,outputs_tensor),dim=1)
        curr_input_forces = torch.concat((filler_nans_cont,input_lin),dim=1)

        return (curr_input_forces.detach().cpu().numpy().astype(np.float64),
                 pred_states.detach().cpu().numpy().astype(np.float64))

    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')
        torch.save(self._inputnet.state_dict(), file_path[0])
        torch.save(self._initializer.state_dict(), file_path[1])
        torch.save(self._predictor.state_dict(), file_path[2])
        torch.save(self._inputRNNnet.state_dict(), file_path[3])
        with open(file_path[4], mode='w') as f:
            json.dump(
                {
                    'state_mean': self._state_mean.tolist(),
                    'state_std': self._state_std.tolist(),
                    'control_mean': self._control_mean.tolist(),
                    'control_std': self._control_std.tolist(),
                    '_state_mean_RNN_in': self._state_mean_RNN_in.tolist(),
                    '_state_std_RNN_in': self._state_std_RNN_in.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self._inputnet.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self._initializer.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        self._predictor.load_state_dict(
            torch.load(file_path[2], map_location=self.device_name)
        )
        self._inputRNNnet.load_state_dict(
            torch.load(file_path[3], map_location=self.device_name)
        )
        with open(file_path[4], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_std'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_std'], dtype=np.float64)
        self._state_mean_RNN_in = np.array(norm['_state_mean_RNN_in'], dtype=np.float64)
        self._state_std_RNN_in = np.array(norm['_state_std_RNN_in'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'inputfnn.pth','initializer.pth', 'predictor.pth', 'inputRNNnet.pth', 'json'

    def get_parameter_count(self) -> int:
        if self.RNNinputnetbool:
            params = sum([
                sum(p.numel() for p in self._inputnet.parameters() if p.requires_grad),
                sum(p.numel() for p in self._predictor.parameters() if p.requires_grad),
                sum(p.numel() for p in self._initializer.parameters() if p.requires_grad),
                sum(p.numel() for p in self._inputRNNnet.parameters() if p.requires_grad)
            ])
        else:
            params = sum([
                sum(p.numel() for p in self._inputnet.parameters() if p.requires_grad),
                sum(p.numel() for p in self._predictor.parameters() if p.requires_grad),
                sum(p.numel() for p in self._initializer.parameters() if p.requires_grad)
            ])
        return params


#mostly just copy pasted but with my own simulate for a better comparison
# class PlainLtiRnnConfig(DynamicIdentificationModelConfig):
#     nx: int
#     recurrent_dim: int
#     num_recurrent_layers_init: int
#     dropout: float
#     sequence_length: int
#     learning_rate: float
#     batch_size: int
#     epochs_initializer: int
#     epochs_predictor: int
#     clip_gradient_norm: float
#     loss: Literal['mse', 'msge']

#Put this into configuration file

# ,
#       {
#         "model_base_name": "PlainLtiRnn",
#         "model_class": "deepsysid.models.JTMasterarbeit.Single_Parts.PlainLtiRnn",
#         "static_parameters": {
#           "num_recurrent_layers_init": 2,
#           "loss": "mse",
#           "time_delta": 1.0,
#           "dropout": 0.25,
#           "sequence_length": 50,
#           "batch_size": 128,
#           "epochs_initializer": 400,
#           "epochs_predictor": 1000,
#           "nx": 64,
#           "learning_rate": 0.0025,
#           "clip_gradient_norm":  100
#         },
#         "flexible_parameters": {
#           "recurrent_dim": [
#             64,
#             128,
#             192,
#             256
#           ]
#         }
#       }


# class PlainLtiRnn(base.NormalizedHiddenStateInitializerPredictorModel):
#     CONFIG = PlainLtiRnnConfig

#     def __init__(self, config: PlainLtiRnnConfig):
#         super().__init__(config)

#         self.device_name = config.device_name
#         self.device = torch.device(self.device_name)

#         self.nx = config.nx
#         self.control_dim = len(config.control_names)
#         self.state_dim = len(config.state_names)
#         self.recurrent_dim = config.recurrent_dim

#         self.recurrent_dim = config.recurrent_dim
#         self.num_recurrent_layers_init = config.num_recurrent_layers_init
#         self.dropout = config.dropout

#         self.sequence_length = config.sequence_length
#         self.learning_rate = config.learning_rate
#         self.batch_size = config.batch_size
#         self.epochs_initializer = config.epochs_initializer
#         self.epochs_predictor = config.epochs_predictor

#         self.clip_gradient_norm = config.clip_gradient_norm

#         if config.loss == 'mse':
#             self.loss: nn.Module = nn.MSELoss().to(self.device)
#         elif config.loss == 'msge':
#             self.loss = loss.MSGELoss().to(self.device)
#         else:
#             raise ValueError('loss can only be "mse" or "msge"')

#         self._predictor = rnn.LtiRnn(
#             nx=self.nx,
#             nu=self.control_dim,
#             ny=self.state_dim,
#             nw=self.recurrent_dim,
#         ).to(self.device)

#         self._initializer = rnn.BasicLSTM(
#             input_dim=self.control_dim + self.state_dim,
#             recurrent_dim=self.nx,
#             num_recurrent_layers=self.num_recurrent_layers_init,
#             output_dim=[self.state_dim],
#             dropout=self.dropout,
#         ).to(self.device)

#         self.optimizer_pred = optim.Adam(
#             self._predictor.parameters(), lr=self.learning_rate
#         )
#         self.optimizer_init = optim.Adam(
#             self._initializer.parameters(), lr=self.learning_rate
#         )

#     def train(
#         self,
#         control_seqs: List[NDArray[np.float64]],
#         state_seqs: List[NDArray[np.float64]],
#     ) -> Dict[str, NDArray[np.float64]]:
#         us = control_seqs
#         ys = state_seqs

#         self._predictor.train()
#         self._initializer.train()

#         self._control_mean, self._control_std = utils.mean_stddev(us)
#         self._state_mean, self._state_std = utils.mean_stddev(ys)

#         us = [
#             utils.normalize(control, self.control_mean, self.control_std)
#             for control in us
#         ]
#         ys = [utils.normalize(state, self.state_mean, self.state_std) for state in ys]

#         initializer_dataset = RecurrentInitializerDataset(us, ys, self.sequence_length)

#         initializer_loss = []
#         time_start_init = time.time()
#         for i in range(self.epochs_initializer):
#             data_loader = data.DataLoader(
#                 initializer_dataset, self.batch_size, shuffle=True, drop_last=True
#             )
#             total_loss = 0.0
#             for batch_idx, batch in enumerate(data_loader):
#                 self._initializer.zero_grad()
#                 y, _ = self._initializer.forward(batch['x'].float().to(self.device))
#                 batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
#                 total_loss += batch_loss.item()
#                 batch_loss.backward()
#                 self.optimizer_init.step()

#             logger.info(
#                 f'Epoch {i + 1}/{self.epochs_initializer}\t'
#                 f'Epoch Loss (Initializer): {total_loss}'
#             )
#             initializer_loss.append(total_loss)
#         time_end_init = time.time()

#         predictor_dataset = RecurrentPredictorDataset(us, ys, self.sequence_length)

#         time_start_pred = time.time()
#         predictor_loss: List[np.float64] = []
#         gradient_norm: List[np.float64] = []
#         for i in range(self.epochs_predictor):
#             data_loader = data.DataLoader(
#                 predictor_dataset, self.batch_size, shuffle=True, drop_last=True
#             )
#             total_loss = 0
#             max_grad = 0
#             for batch_idx, batch in enumerate(data_loader):
#                 self._predictor.zero_grad()
#                 # Initialize predictor with state of initializer network
#                 _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
#                 # Predict and optimize
#                 y, _ = self._predictor.forward(
#                     batch['x'].float().to(self.device), hx=None
#                 )
#                 y = y.to(self.device)
#                 batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
#                 total_loss += batch_loss.item()
#                 batch_loss.backward()

#                 # gradient infos
#                 grads_norm = [
#                     torch.linalg.norm(p.grad) for p in self._predictor.parameters()
#                 ]
#                 max_grad += max(grads_norm)
#                 torch.nn.utils.clip_grad_norm_(
#                     self._predictor.parameters(), self.clip_gradient_norm
#                 )
#                 self.optimizer_pred.step()

#             logger.info(
#                 f'Epoch {i + 1}/{self.epochs_predictor}\t'
#                 f'Total Loss (Predictor): {total_loss:1f} \t'
#                 f'Max accumulated gradient norm: {max_grad:1f}'
#             )
#             predictor_loss.append(np.float64(total_loss))
#             gradient_norm.append(np.float64(max_grad))

#         time_end_pred = time.time()
#         time_total_init = time_end_init - time_start_init
#         time_total_pred = time_end_pred - time_start_pred
#         logger.info(
#             f'Training time for initializer {time_total_init}s '
#             f'and for predictor {time_total_pred}s'
#         )

#         return dict(
#             index=np.asarray(i),
#             epoch_loss_initializer=np.asarray(initializer_loss),
#             epoch_loss_predictor=np.asarray(predictor_loss),
#             gradient_norm=np.asarray(gradient_norm),
#             training_time_initializer=np.asarray(time_total_init),
#             training_time_predictor=np.asarray(time_total_pred),
#         )

#     def simulate(
#         self,
#         initial_control: NDArray[np.float64],
#         initial_state: NDArray[np.float64],
#         control: NDArray[np.float64],
#     ) -> NDArray[np.float64]:
#         if (
#             self.control_mean is None
#             or self.control_std is None
#             or self.state_mean is None
#             or self.state_std is None
#         ):
#             raise ValueError('Model has not been trained and cannot simulate.')

#         self._initializer.eval()
#         self._predictor.eval()

#         initial_u = initial_control
#         initial_y = initial_state
#         u = control

#         initial_u = utils.normalize(initial_u, self.control_mean, self.control_std)
#         initial_y = utils.normalize(initial_y, self.state_mean, self.state_std)
#         u = utils.normalize(u, self.control_mean, self.control_std)

#         with torch.no_grad():
#             init_x = (
#                 torch.from_numpy(np.hstack((initial_u[1:], initial_y[:-1])))
#                 .unsqueeze(0)
#                 .float()
#                 .to(self.device)
#             )
#             pred_x = torch.from_numpy(u).unsqueeze(0).float().to(self.device)

#             _, hx = self._initializer.forward(init_x)
#             y, _ = self._predictor.forward(pred_x, hx=None)
#             y_np: NDArray[np.float64] = (
#                 y.cpu().detach().squeeze().numpy().astype(np.float64)
#             )

#         y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
#         return y_np

#     def simulate_inputforces_onestep(
#         self,
#         controls: NDArray[np.float64],
#         states: NDArray[np.float64],
#         forces: NDArray[np.float64],
#     ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
#         if (
#             self._state_mean is None
#             or self._state_std is None
#             or self._control_mean is None
#             or self._control_std is None
#         ):
#             raise ValueError('Model has not been trained and cannot be simulated.')   

#         # self._inputnet.eval()
#         # self._diskretized_linear.eval()
#         self._predictor.eval()
#         self._initializer.eval()

#         controls_ = utils.normalize(controls, self._control_mean, self._control_std)
#         states_normed_ = utils.normalize(states, self._state_mean, self._state_std)

#         # _state_mean_RNN_in_torch = torch.from_numpy(self._state_mean_RNN_in).float().to(self.device)
#         # _state_std_RNN_in_torch = torch.from_numpy(self._state_std_RNN_in).float().to(self.device)

#         controls_ = torch.from_numpy(controls_).float().to(self.device)
#         states_ = torch.from_numpy(states).float().to(self.device)
#         forces_ = torch.from_numpy(forces).float().to(self.device)
#         states_normed_ = torch.from_numpy(states_normed_).float().to(self.device)

#         x0_control = controls_[:, :50, :]
#         x0_states_normed_ = states_normed_[:, :50, :]
#         x0_init = torch.cat((x0_control, x0_states_normed_), axis=-1)

#         #the 49 incooperates the init of the diskretized linear 
#         #and we then drop the last state since, while we can compute
#         # the last next state we dont have meassurements there 
#         prev_cont_in = controls_[:, 49:, :]
#         lin_state_in = states_[:, 49:, :]
#         lin_forces_in = forces_[:, 49:, :]

#         #for the RNN we now need the values from 50 onward
#         curr_cont_in =controls_[:, 50:, :]

#         with torch.no_grad():
#             # prev_input_forces = self._inputnet.forward(prev_cont_in)

#             #drop last next state to fit with RNN inputs and because see previous comment
#             # states_next = self._diskretized_linear.forward(prev_input_forces,lin_state_in)[:,:-1,:]
#             # states_next_with_true_input_forces = self._diskretized_linear.forward(lin_forces_in,lin_state_in)[:,:-1,:]
            
#             # curr_input_forces_ = self._inputnet.forward(curr_cont_in)
#             #mostly for better understanding of the timesteps
#             # outlin = self._diskretized_linear.calc_output(
#             #     states = states_next,
#             #     input_forces=curr_input_forces_
#             #     )
#             # true_input_pred_states_ = self._diskretized_linear.calc_output(
#             #     states = states_next_with_true_input_forces,
#             #     input_forces=curr_input_forces_
#             #     )

#             _, hx = self._initializer.forward(x0_init)
#             # out_lin_norm = utils.normalize(outlin, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
#             # rnn_input = torch.concat((curr_cont_in,out_lin_norm),dim=2)
#             res_error, _ = self._predictor.forward(x_pred = curr_cont_in,hx=None)
#             # res_error_denorm = utils.denormalize(res_error.to(self.device), _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
#             pred_states_ = res_error.to(self.device)#res_error_denorm.to(self.device)#outlin + res_error.to(self.device)

#             filler_forces_ = self._inputnet.forward(x0_control)

#         #fill the start that is used for initialisation with nans
#         filler_nans_states = torch.full(x0_states_normed_.shape, float('nan')).to(self.device)
#         filler_nans_cont = torch.full(filler_forces_.shape, float('nan')).to(self.device)
#         pred_states = torch.concat((filler_nans_states,pred_states_),dim=1)
#         # true_input_pred_states = torch.concat((filler_nans_states,true_input_pred_states_),dim=1)
#         # curr_input_forces = torch.concat((filler_nans_cont,curr_input_forces_),dim=1)

#         return (pred_states.detach().cpu().numpy().astype(np.float64),
#                  pred_states.detach().cpu().numpy().astype(np.float64),
#                  pred_states.detach().cpu().numpy().astype(np.float64))


#     def save(self, file_path: Tuple[str, ...]) -> None:
#         if (
#             self.state_mean is None
#             or self.state_std is None
#             or self.control_mean is None
#             or self.control_std is None
#         ):
#             raise ValueError('Model has not been trained and cannot be saved.')

#         torch.save(self._initializer.state_dict(), file_path[0])
#         torch.save(self._predictor.state_dict(), file_path[1])
#         with open(file_path[2], mode='w') as f:
#             json.dump(
#                 {
#                     'state_mean': self.state_mean.tolist(),
#                     'state_std': self.state_std.tolist(),
#                     'control_mean': self.control_mean.tolist(),
#                     'control_std': self.control_std.tolist(),
#                 },
#                 f,
#             )

#     def load(self, file_path: Tuple[str, ...]) -> None:
#         self._initializer.load_state_dict(
#             torch.load(file_path[0], map_location=self.device_name)
#         )
#         self._predictor.load_state_dict(
#             torch.load(file_path[1], map_location=self.device_name)
#         )
#         with open(file_path[2], mode='r') as f:
#             norm = json.load(f)
#         self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
#         self._state_std = np.array(norm['state_std'], dtype=np.float64)
#         self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
#         self._control_std = np.array(norm['control_std'], dtype=np.float64)

#     def get_file_extension(self) -> Tuple[str, ...]:
#         return 'initializer.pth', 'predictor.pth', 'json'

#     def get_parameter_count(self) -> int:
#         # technically parameter counts of both networks are equal
#         init_count = sum(
#             p.numel() for p in self._initializer.parameters() if p.requires_grad
#         )
#         predictor_count = sum(
#             p.numel() for p in self._predictor.parameters() if p.requires_grad
#         )
#         return init_count + predictor_count

#     @property
#     def initializer(self) -> HiddenStateForwardModule:
#         return copy.deepcopy(self._initializer)

#     @property
#     def predictor(self) -> HiddenStateForwardModule:
#         return copy.deepcopy(self._predictor)