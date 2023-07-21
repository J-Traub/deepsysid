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

from .datasets_JT import TimeSeriesDataset, HybridRecurrentLinearFNNInputDataset
from .networks_JT import DiskretizedLinear, InputNet, DiskretizedLinearOpt

logger = logging.getLogger(__name__)

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
    lr_scal : float
    sg_weigth: float
    multistep_train: bool
    linear_train: bool
    smaller_gain_penalty: bool

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

        self.lr_scal = config.lr_scal
        self.sg_weigth = config.sg_weigth
        self.multistep_train = config.multistep_train
        self.linear_train = config.linear_train
        self.smaller_gain_penalty = config.smaller_gain_penalty

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._inputnet = InputNet(dropout = self.dropout).to(self.device)

        if self.linear_train:
            self._diskretized_linear = DiskretizedLinearOpt(
                Ad = self.Ad,
                Bd = self.Bd,
                Cd = self.Cd,
                Dd = self.Dd,
                ssv_input= self.ssv_input,
                ssv_states= self.ssv_states,
            ).to(self.device)
        else:
            self._diskretized_linear = DiskretizedLinear(
                Ad = self.Ad,
                Bd = self.Bd,
                Cd = self.Cd,
                Dd = self.Dd,
                ssv_input= self.ssv_input,
                ssv_states= self.ssv_states,
                no_lin = False,
            ).to(self.device)

        if self.multistep_train:
            self.optimizer = optim.Adam([
                {'params': self._inputnet.parameters(), 'lr': self.learning_rate},
                {'params': self._diskretized_linear.parameters(), 'lr': self.learning_rate*self.lr_scal}
            ])
        else:
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

        dataset = TimeSeriesDataset(us, ys,device=self.device)
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

            for batch_idx, batch in enumerate(data_loader):

                time0 = time.time()
                self._inputnet.zero_grad()

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                if not self.linear_train:
                    Lin_input = utils.denormalize(Lin_input, _state_mean_torch, _state_std_torch)
                next_state = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])


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
                if not self.linear_train:
                    states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)
                ear = time.time()

                batch_loss = F.mse_loss(
                    states_next, true_next_states
                )
                if self.linear_train and self.smaller_gain_penalty:
                    penatly = self._diskretized_linear.smaller_gain_penalty(self.sg_weigth)
                total_loss += batch_loss.item()
                back = time.time()
                if self.linear_train and self.smaller_gain_penalty:
                    (batch_loss+penatly).backward()
                else:
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

        if not self.multistep_train:
            return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64))
        
        #########
        # Multistep
        ###########
        if not self.linear_train:
            self.optimizer_multi = optim.Adam([
                {'params': self._inputnet.parameters(), 'lr': self.learning_rate}
            ])
        else:
            self.optimizer_multi = optim.Adam([
                {'params': self._inputnet.parameters(), 'lr': self.learning_rate},
                {'params': self._diskretized_linear.parameters(), 'lr': self.learning_rate*self.lr_scal}
            ])

        epoch_losses_multistep = []
        scaling = [1,1]
        multistep_dataset = HybridRecurrentLinearFNNInputDataset(
                    us, ys,
                    pred_next_state_seq = ys, #this is just a provisorium since it is not needed anyways here
                    scaling = scaling, 
                    sequence_length = 10, 
                    device=self.device)
        for i in range(self.epochs):

            data_loader = data.DataLoader(
                multistep_dataset, self.batch_size, shuffle=True, drop_last=True,
            )

            total_loss = 0.0
            max_batches = 0

            for batch_idx, batch in enumerate(data_loader):

                self._inputnet.zero_grad()

                control_prev = batch['control_prev']
                states_prev_ = batch['states_prev']
        
                input_lin = self._inputnet.forward(control_prev)

                states_next =  states_prev_[:,0:1,:]
                if not self.linear_train:
                    states_next = utils.denormalize(states_next, _state_mean_torch, _state_std_torch)
                outputs =[]
                seq_len = control_prev.size(dim=1)
                for seq_step in range(seq_len):
                    #seq_step:seq_step+1 preserves the original dimensionality
                    in_lin = input_lin[:,seq_step:seq_step+1,:]


                    #this takes the denormalized corrected state as input
                    states_next = self._diskretized_linear.forward(
                        input_forces=in_lin,
                        states=states_next
                        )
                    
                    output = self._diskretized_linear.calc_output(
                        states= states_next
                    )
                    if not self.linear_train:
                        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                    else:
                        output_normed = output
                    outputs.append(output_normed)
                    
                pred_states = torch.cat(outputs, dim=1)

                true_states = batch['states']

                batch_loss = F.mse_loss(
                    pred_states, true_states
                )
                if self.linear_train and self.smaller_gain_penalty:
                    penatly = self._diskretized_linear.smaller_gain_penalty(self.sg_weigth)
                total_loss += batch_loss.item()

                if self.linear_train and self.smaller_gain_penalty:
                    (batch_loss+penatly).backward()
                else:
                    batch_loss.backward()

                self.optimizer_multi.step()
                max_batches = batch_idx

            loss_average = total_loss/(max_batches+1)
            logger.info(f'Epoch_multistep {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')
            print(f'Epoch_multistep {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')

            epoch_losses_multistep.append([i, loss_average])

        return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64),
                    epoch_losses_multistep=np.array(epoch_losses_multistep, dtype=np.float64),
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
        dataset_vali = TimeSeriesDataset(us_vali, ys_vali,device=self.device)
        best_val_loss = torch.tensor(float('inf'))
        best_epoch = 0
        validation_losses = []

        dataset = TimeSeriesDataset(us, ys,device=self.device)
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
                if not self.linear_train:
                    Lin_input = utils.denormalize(Lin_input, _state_mean_torch, _state_std_torch)
                next_state = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])

                # testing1 = batch['next_state'].detach().cpu().numpy()
                # testing = next_state.detach().cpu().numpy()

                input = FNN_input.float().to(self.device)  
                true_states = Lin_input.float().to(self.device) 
                true_next_states = next_state.float().to(self.device) 

                input_forces = self._inputnet.forward(input)
                lin = time.time()
                states_next = self._diskretized_linear.forward(input_forces,true_states)
                if not self.linear_train:
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

                if self.linear_train and self.smaller_gain_penalty:
                    penatly = self._diskretized_linear.smaller_gain_penalty(self.sg_weigth)
                total_loss += batch_loss.item()
                back = time.time()
                if self.linear_train and self.smaller_gain_penalty:
                    (batch_loss+penatly).backward()
                else:
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
                controls_vali = dataset_vali.control
                states_vali = dataset_vali.state
                true_next_states_vali = dataset_vali.next_state

                # for some reason dataloader iteration is very slow otherwise
                #to device needs to be after denormalize else it cant calculate with the numpy std and mean
                controls_vali = controls_vali.reshape(-1,controls_vali.shape[-1]).float().to(self.device)
                states_vali = states_vali.reshape(-1,states_vali.shape[-1])
                if not self.linear_train:
                    states_vali = utils.denormalize(states_vali, _state_mean_torch, _state_std_torch).float().to(self.device)
                true_next_states_vali = true_next_states_vali.reshape(-1,true_next_states_vali.shape[-1]).float().to(self.device)

                input_forces_vali = self._inputnet.forward(controls_vali)
                states_next_vali = self._diskretized_linear.forward(input_forces_vali,states_vali)
                if not self.linear_train:
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


        if not self.multistep_train:
            return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64), 
                validation_loss=np.array(validation_losses, dtype=np.float64), 
                estop_epoch = best_epoch, 
                best_val_loss = best_val_loss.item(),
                )


        #########
        # Multistep
        ###########

        if not self.linear_train:
            self.optimizer_multi = optim.Adam([
                {'params': self._inputnet.parameters(), 'lr': self.learning_rate}
            ])
        else:
            self.optimizer_multi = optim.Adam([
                {'params': self._inputnet.parameters(), 'lr': self.learning_rate},
                {'params': self._diskretized_linear.parameters(), 'lr': self.learning_rate*self.lr_scal}
            ])

        best_val_loss_multi = torch.tensor(float('inf'))
        best_epoch_multi = 0
        validation_losses_multi = []

        epoch_losses_multistep = []
        scaling = [1,1]
        multistep_dataset = HybridRecurrentLinearFNNInputDataset(
                    us, ys,
                    pred_next_state_seq = ys, #this is just a provisorium since it is not needed anyways here
                    scaling = scaling, 
                    sequence_length = 10, 
                    device=self.device)
        multistep_dataset_vali = HybridRecurrentLinearFNNInputDataset(
                    us_vali, ys_vali,
                    pred_next_state_seq = ys, #this is just a provisorium since it is not needed anyways here
                    scaling = scaling, 
                    sequence_length = 10, 
                    device=self.device)
        for i in range(self.epochs):

            data_loader = data.DataLoader(
                multistep_dataset, self.batch_size, shuffle=True, drop_last=True,
            )

            total_loss = 0.0
            max_batches = 0

            for batch_idx, batch in enumerate(data_loader):

                self._inputnet.zero_grad()

                control_prev = batch['control_prev']
                states_prev_ = batch['states_prev']
        
                input_lin = self._inputnet.forward(control_prev)

                states_next =  states_prev_[:,0:1,:]
                if not self.linear_train:
                    states_next = utils.denormalize(states_next, _state_mean_torch, _state_std_torch)
                outputs =[]
                seq_len = control_prev.size(dim=1)
                for seq_step in range(seq_len):
                    #seq_step:seq_step+1 preserves the original dimensionality
                    in_lin = input_lin[:,seq_step:seq_step+1,:]


                    #this takes the denormalized corrected state as input
                    states_next = self._diskretized_linear.forward(
                        input_forces=in_lin,
                        states=states_next
                        )
                    
                    output = self._diskretized_linear.calc_output(
                        states= states_next
                    )
                    if not self.linear_train:
                        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                    else:
                        output_normed = output
                    outputs.append(output_normed)
                    
                pred_states = torch.cat(outputs, dim=1)

                true_states = batch['states']

                batch_loss = F.mse_loss(
                    pred_states, true_states
                )

                if loss_weights is None:
                    batch_loss = F.mse_loss(
                        pred_states, true_states
                    )
                else:
                    batch_loss =torch.mean(
                        ((true_states - pred_states) ** 2) * loss_weights
                    )
                    
                if self.linear_train and self.smaller_gain_penalty:
                    penatly = self._diskretized_linear.smaller_gain_penalty(self.sg_weigth)
                total_loss += batch_loss.item()

                if self.linear_train and self.smaller_gain_penalty:
                    (batch_loss+penatly).backward()
                else: 
                    batch_loss.backward()

                self.optimizer_multi.step()
                max_batches = batch_idx

            loss_average = total_loss/(max_batches+1)
            logger.info(f'Epoch_multistep {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')
            print(f'Epoch_multistep {i + 1}/{self.epochs} - Epoch average Loss: {loss_average}')

            epoch_losses_multistep.append([i, loss_average])


            #Validation

            #check validation for overfitt prevention
            with torch.no_grad():
                self._inputnet.eval() #this is very important (i think to disable the dropout layers)
                control_prev_vali = multistep_dataset_vali.control_prev
                states_prev_vali = multistep_dataset_vali.states_prev
                states_vali = multistep_dataset_vali.states

                input_lin = self._inputnet.forward(control_prev_vali)

                states_next =  states_prev_vali[:,0:1,:]
                if not self.linear_train:
                    states_next = utils.denormalize(states_next, _state_mean_torch, _state_std_torch)
                outputs =[]
                seq_len = control_prev_vali.size(dim=1)
                for seq_step in range(seq_len):
                    #seq_step:seq_step+1 preserves the original dimensionality
                    in_lin = input_lin[:,seq_step:seq_step+1,:]


                    #this takes the denormalized corrected state as input
                    states_next = self._diskretized_linear.forward(
                        input_forces=in_lin,
                        states=states_next
                        )
                    
                    output = self._diskretized_linear.calc_output(
                        states= states_next
                    )
                    if not self.linear_train:
                        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                    else:
                        output_normed = output
                    outputs.append(output_normed)
                    
                pred_states = torch.cat(outputs, dim=1)

                true_states = states_vali

                if loss_weights is None:
                    validation_loss = F.mse_loss(
                        pred_states, true_states
                    )
                else:
                    validation_loss =torch.mean(
                        ((true_states - pred_states) ** 2) * loss_weights
                    )
                    

               # Check if the current validation loss is the best so far
                if validation_loss < best_val_loss_multi:
                    # If it is, save the model parameters
                    torch.save(self._inputnet.state_dict(), 'best_model_params_mulit.pth')
                    # Update the best validation loss
                    best_val_loss_multi = validation_loss
                    best_epoch_multi = i
                

            validation_loss_ = validation_loss.item()
            validation_losses_multi.append([i, validation_loss_])

            if  self.patience < (i-best_epoch_multi):
                print("early stopping")
                break

        #load the best parameters with best validation loss (early stopping)
        self._inputnet.load_state_dict(torch.load('best_model_params_mulit.pth'))

        return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64), 
                    epoch_loss_multi=np.array(epoch_losses_multistep, dtype=np.float64), 
                    validation_loss=np.array(validation_losses, dtype=np.float64), 
                    validation_loss_multi=np.array(validation_losses_multi, dtype=np.float64), 
                    estop_epoch = best_epoch, 
                    estop_epoch_multi = best_epoch_multi,
                    best_val_loss = best_val_loss.item(),
                    best_val_loss_multi = best_val_loss_multi.item()
                    )

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
            if not self.linear_train:
                last_init_state = utils.denormalize(last_init_state, _state_mean_torch, _state_std_torch)

            states_next = self._diskretized_linear.forward(input_forces=input_lin,states=last_init_state)
            outputs =[]
            input_lin = self._inputnet.forward(control_)
            for in_lin in input_lin:
                outputs.append(states_next)
                states_next = self._diskretized_linear.forward(input_forces=in_lin.unsqueeze(0),states=states_next)


        outputs = torch.vstack(outputs)
        if self.linear_train:
            outputs = utils.denormalize(outputs, _state_mean_torch, _state_std_torch)
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
        _state_mean_torch = torch.from_numpy(self._state_mean).float().to(self.device)
        _state_std_torch = torch.from_numpy(self._state_std).float().to(self.device)

        self._inputnet.eval()
        self._diskretized_linear.eval()
        controls_ = utils.normalize(controls, self._control_mean, self._control_std)
        if self.linear_train:
            states_ = utils.normalize(states_, self._state_mean, self._state_std)

        controls_ = torch.from_numpy(controls_).float().to(self.device)
        states_ = torch.from_numpy(states).float().to(self.device)
        forces_ = torch.from_numpy(forces).float().to(self.device)

        with torch.no_grad():
            input_forces = self._inputnet.forward(controls_)
            states_next = self._diskretized_linear.forward(input_forces,states_)
            if self.linear_train:
                states_next = utils.denormalize(states_next, _state_mean_torch, _state_std_torch)
            states_next_with_true_input_forces = self._diskretized_linear.forward(forces_,states_)
            if self.linear_train:
                states_next_with_true_input_forces = utils.denormalize(states_next_with_true_input_forces, _state_mean_torch, _state_std_torch)

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
        torch.save(self._diskretized_linear.state_dict(), file_path[1])
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
        self._inputnet.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self._diskretized_linear.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_std'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'inputfnn.pth', 'diskretized_linear.pth', 'json'

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self._inputnet.parameters() if p.requires_grad)
