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
# from ...networks.rnn import LtiRnnConvConstr
from .. import base, utils
from ..base import DynamicIdentificationModelConfig
from ..datasets import RecurrentHybridPredictorDataset,RecurrentInitializerDataset, RecurrentPredictorDataset,FixedWindowDataset

from .networks_JT import InputNet, InputRNNNet, DiskretizedLinear, LtiRnnConvConstr
from .datasets_JT import TimeSeriesDataset, HybridRecurrentLinearFNNInputDataset
from .hybrid_model_JT import Hybrid_Model

logger = logging.getLogger(__name__)
    


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
#as in constrained RNN => i think it means convex RNN
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
            self._predictor = LtiRnnConvConstr(
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
            self._predictor =LtiRnnConvConstr(
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


        self._Hybrid_model = Hybrid_Model(
            predictor=self._predictor,
            initializer=self._initializer,
            inputnet=self._inputnet,
            diskretized_linear=self._diskretized_linear,
            inputRNNnet=self._inputRNNnet,
            device=self.device
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
        dataset = TimeSeriesDataset(us, ys ,device=self.device)
        for i in range(self.epochs_InputFNN):

            data_loader = data.DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=False,
            )
            total_loss = 0.0

            for batch_idx, batch in enumerate(data_loader):
                self._inputnet.zero_grad()

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                true_states = utils.denormalize(Lin_input, _state_mean_torch, _state_std_torch)
                true_next_states = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])


                states_next = self._Hybrid_model.forward_inputnet(
                    FNN_input=FNN_input,
                    Lin_input=true_states,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                    )

                batch_loss = F.mse_loss(
                    states_next, true_next_states
                )
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_inputFNN.step()

            loss_average = total_loss/(len(data_loader))
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

        predictor_dataset = HybridRecurrentLinearFNNInputDataset(us, ys, self.sequence_length ,device=self.device)

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
            bls_iter = 0

            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'])

                control_prev = batch['control_prev']
                # linear_inputs_curr = self._inputnet.forward(batch['control'])
                states_prev_ = utils.denormalize(batch['states_prev'], _state_mean_torch, _state_std_torch)
                
                #with inputnet for the control inputs?
                if self.RNNinputnetbool:
                    control_in = self._inputRNNnet.forward(batch['control'])
                else:
                    control_in =batch['control']

                if self.forward_alt_bool:
                    output_normed = self._Hybrid_model.forward_predictor_onestep_alt(
                        control_prev=control_prev, 
                        states_prev_= states_prev_,
                        control_in= control_in,
                        _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                        _state_mean_torch= _state_mean_torch,
                        _state_std_torch= _state_std_torch
                    )
                else:
                    output_normed = self._Hybrid_model.forward_predictor_onestep(
                        control_prev=control_prev,
                        states_prev_= states_prev_,
                        control_in= control_in,
                        _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                        _state_mean_torch= _state_mean_torch,
                        _state_std_torch= _state_std_torch
                    )


                barrier = self._predictor.get_barrier(t)
                batch_loss = self.loss.forward(output_normed, batch['states'])
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()


                #stuff for constraint checking
                ################
                # gradient infos
                old_pars, max_grad = self._Hybrid_model.constr_savings(max_grad=max_grad)

                self.optimizer_pred.step()
                ########################### 
                #Constraints Checking
                #################################
                constr_hold, bls_iter_ = self._Hybrid_model.constraints_checking(
                    old_pars=old_pars,
                    current_epochs = i,
                    max_epochs=self.epochs_predictor_singlestep,
                    logger = logger
                )
                bls_iter += bls_iter_
                if not constr_hold:
                    raise ValueError(
                        "Error: Model did not complete Teacher Forcing training phase. \n"
                        "Adjust gamma, initial decay, decay rate or singlestep epochs.")
               
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

        # time_start_pred = time.time()
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
            bls_iter = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'])

                #hopefully more understandable:
                # as we initialize the hiddenstate of the RNN we need to initialize
                # the internal state of the _diskretized_linear the first computation
                # of the error can be omitted for this since we can expect that the
                # initial state as no error 

                #we need only the last point of the x0 sequence for init of the linear
                init_control = batch['x0_control'][:,-1:,:]
                #inputs into the diskretized model need to be denormalised
                init_state = utils.denormalize(batch['x0_states'][:,-1:,:], _state_mean_torch, _state_std_torch)
                init_input = self._inputnet.forward(init_control)
                states_next = self._diskretized_linear.forward(
                    input_forces= init_input,
                    states=init_state)
                
                #get all inputs
                if self.RNNinputnetbool:
                    control_ = self._inputRNNnet.forward(batch['control'])
                else:
                    control_ =batch['control']
                control_lin =batch['control']
                
                if self.forward_alt_bool:
                    outputs_tensor = self._Hybrid_model.forward_predictor_multistep_alt(
                        control_lin=control_lin,
                        control_=control_,
                        states_next = states_next,
                        nx=self.nx,
                        _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                        _state_mean_torch = _state_mean_torch,
                        _state_std_torch = _state_std_torch,
                    )
                else:
                    outputs_tensor = self._Hybrid_model.forward_predictor_multistep(
                        control_lin=control_lin,
                        control_=control_,
                        states_next = states_next,
                        nx=self.nx,
                        _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                        _state_mean_torch = _state_mean_torch,
                        _state_std_torch = _state_std_torch,
                    )

                barrier = self._predictor.get_barrier(t)

                true_state = batch['states']
                # test1 = outputs_tensor
                # test2 = true_state
                batch_loss = self.loss.forward(outputs_tensor, true_state)
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()

                #stuff for constraint checking
                ################
                old_pars, max_grad = self._Hybrid_model.constr_savings(max_grad=max_grad)

                self.optimizer_pred_multistep.step()
                ########################### 
                #Constraints Checking
                #################################
                constr_hold, bls_iter_ = self._Hybrid_model.constraints_checking(
                    old_pars=old_pars,
                    current_epochs = i,
                    max_epochs=self.epochs_predictor_multistep,
                    logger = logger
                )
                bls_iter += bls_iter_
                if not constr_hold:
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
        state_seqs_vali: List[NDArray[np.float64]]
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
        inputfnn_best_val_loss = torch.tensor(float('inf'), device=self.device)
        inputfnn_best_epoch = 0
        inputfnn_validation_losses = []
        predictor_best_val_loss = torch.tensor(float('inf'), device=self.device)
        predictor_best_epoch = 0
        predictor_validation_losses = []
        predictor_multistep_best_val_loss = torch.full((len(self.sequence_length_list),), float('inf'), device=self.device)
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
        inputfnn_dataset_vali = TimeSeriesDataset(us_vali, ys_vali ,device=self.device)

        inputfnn_losses = []
        inputfnn_dataset = TimeSeriesDataset(us, ys ,device=self.device)
        for i in range(self.epochs_InputFNN):

            data_loader = data.DataLoader(
                inputfnn_dataset, self.batch_size, shuffle=True, drop_last=False,
            )
            total_loss = 0.0

            #because of validation
            self._inputnet.train()

            for batch_idx, batch in enumerate(data_loader):
                self._inputnet.zero_grad()

                # for some reason dataloader iteration is very slow otherwise
                FNN_input = batch['FNN_input'].reshape(-1,batch['FNN_input'].shape[-1])
                Lin_input = batch['Lin_input'].reshape(-1,batch['Lin_input'].shape[-1])
                true_states = utils.denormalize(Lin_input, _state_mean_torch, _state_std_torch)
                true_next_states = batch['next_state'].reshape(-1,batch['next_state'].shape[-1])

                states_next = self._Hybrid_model.forward_inputnet(
                    FNN_input=FNN_input,
                    Lin_input=true_states,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                    )
                
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

            ########################### 
            #main validation check for overfitt prevention
            #################################
            with torch.no_grad():
                self._inputnet.eval() #this is very important (i think to disable the dropout layers)
                #just calculate the whole dataset at once 
                controls_vali = inputfnn_dataset_vali.control
                states_vali = inputfnn_dataset_vali.state
                true_next_states_vali = inputfnn_dataset_vali.next_state

                #to device needs to be after denormalize else it cant calculate with the numpy std and mean
                controls_vali = controls_vali.reshape(-1,controls_vali.shape[-1])
                states_vali = states_vali.reshape(-1,states_vali.shape[-1])
                states_vali = utils.denormalize(states_vali, _state_mean_torch, _state_std_torch)
                true_next_states_vali = true_next_states_vali.reshape(-1,true_next_states_vali.shape[-1])

                states_next_vali = self._Hybrid_model.forward_inputnet(
                    FNN_input=controls_vali,
                    Lin_input=states_vali,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                    )
                
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
                    best_pars = [par.clone().detach() for par in self._inputnet.parameters()]
                    # torch.save(self._inputnet.state_dict(), 'best_model_params.pth')
                    # Update the best validation loss
                    inputfnn_best_val_loss = validation_loss
                    inputfnn_best_epoch = i

            loss_average = total_loss/(len(data_loader))
            logger.info(f'Epoch {i + 1}/{self.epochs_InputFNN} - Epoch average Loss (InputFNN): {loss_average}')
            print(f'Epoch {i + 1}/{self.epochs_InputFNN} - Epoch average Loss (InputFNN): {loss_average}')
            inputfnn_losses.append([i, loss_average])
            #validation
            validation_loss_ = validation_loss.item()
            inputfnn_validation_losses.append([i, validation_loss_])
            if self.patience < (i-inputfnn_best_epoch):
                print("early stopping")
                break


        #load the best parameters with best validation loss (early stopping)
        for best_par, new_par in zip(
                    best_pars, self._inputnet.parameters()
                ):
                    new_par.data = best_par.clone()
        # self._inputnet.load_state_dict(torch.load('best_model_params.pth'))

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
        constr_hold = True

        #validation
        predictor_dataset_vali = HybridRecurrentLinearFNNInputDataset(
            us_vali,
            ys_vali,
            sequence_length = 900, #validation sequence length should be static i think
            device=self.device
            )

        predictor_dataset = HybridRecurrentLinearFNNInputDataset(us, ys, self.sequence_length, device=self.device)

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
            bls_iter = 0

            #because of validation
            self._predictor.train()

            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'])

                control_prev = batch['control_prev']
                # linear_inputs_curr = self._inputnet.forward(batch['control']
                states_prev_ = utils.denormalize(batch['states_prev'], _state_mean_torch, _state_std_torch)
               
                #with inputnet for the control inputs?
                if self.RNNinputnetbool:
                    control_in = self._inputRNNnet.forward(batch['control'])
                else:
                    control_in =batch['control']

                if self.forward_alt_bool:
                    output_normed = self._Hybrid_model.forward_predictor_onestep_alt(
                        control_prev=control_prev, 
                        states_prev_= states_prev_,
                        control_in= control_in,
                        _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                        _state_mean_torch= _state_mean_torch,
                        _state_std_torch= _state_std_torch
                    )
                else:
                    output_normed = self._Hybrid_model.forward_predictor_onestep(
                        control_prev=control_prev,
                        states_prev_= states_prev_,
                        control_in= control_in,
                        _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                        _state_mean_torch= _state_mean_torch,
                        _state_std_torch= _state_std_torch
                    )

                barrier = self._predictor.get_barrier(t)

                #loss calculation, when weights are all 1, they are equivalent 
                if loss_weights is None:
                    batch_loss = self.loss.forward(
                        output_normed, batch['states']
                        )
                else:
                    batch_loss = torch.mean(
                        ((batch['states'] - output_normed) ** 2) * loss_weights
                        )
                
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()


                #stuff for constraint checking
                ################
                # gradient infos
                old_pars, max_grad = self._Hybrid_model.constr_savings(max_grad=max_grad)
                ################

                self.optimizer_pred.step()
                ########################### 
                #Constraints Checking
                #################################
                constr_hold, bls_iter_ = self._Hybrid_model.constraints_checking(
                    old_pars=old_pars,
                    current_epochs = i,
                    max_epochs=self.epochs_predictor_singlestep,
                    logger = logger
                )
                bls_iter += bls_iter_

                if not constr_hold:
                    break
            if not constr_hold:
                break

            ########################### 
            #main validation check for overfitt prevention
            #################################
            with torch.no_grad():
                self._predictor.eval() #this is very important (i think to disable the dropout layers)

                #do the whole validation dataset at once
                states = predictor_dataset_vali.states
                control = predictor_dataset_vali.control
                x0 = predictor_dataset_vali.x0
                control_prev = predictor_dataset_vali.control_prev
                states_prev = predictor_dataset_vali.states_prev

                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(x0)

                control_prev = control_prev
                # linear_inputs_curr = self._inputnet.forward(control)
                states_prev_ = utils.denormalize(states_prev, _state_mean_torch, _state_std_torch)
               
                #with inputnet for the control inputs?
                if self.RNNinputnetbool:
                    control_in = self._inputRNNnet.forward(control)
                else:
                    control_in = control

                if self.forward_alt_bool:
                    output_normed = self._Hybrid_model.forward_predictor_onestep_alt(
                        control_prev=control_prev, 
                        states_prev_= states_prev_,
                        control_in= control_in,
                        _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                        _state_mean_torch= _state_mean_torch,
                        _state_std_torch= _state_std_torch
                    )
                else:
                    output_normed = self._Hybrid_model.forward_predictor_onestep(
                        control_prev=control_prev,
                        states_prev_= states_prev_,
                        control_in= control_in,
                        _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                        _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                        _state_mean_torch= _state_mean_torch,
                        _state_std_torch= _state_std_torch
                    )

                barrier = self._predictor.get_barrier(t)

                #loss calculation, when weights are all 1, they are equivalent 
                if loss_weights is None:
                    validation_loss = self.loss.forward(
                        output_normed, states
                        )
                else:
                    #TODO: check if dimesnions are correct
                    validation_loss = torch.mean(
                        ((states - output_normed) ** 2) * loss_weights
                        )

                # Check if the current validation loss is the best so far
                if validation_loss < predictor_best_val_loss:
                    # If it is, save the model parameters
                    best_pars = [par.clone().detach() for par in self._predictor.parameters()]
                    # torch.save(self._predictor.state_dict(), 'best_model_params.pth')
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

            if self.patience < (i-predictor_best_epoch):
                print("early stopping")
                break

        #load the best parameters with best validation loss (early stopping)
        for best_par, new_par in zip(
                    best_pars, self._predictor.parameters()
                ):
                    new_par.data = best_par.clone()
        # self._predictor.load_state_dict(torch.load('best_model_params.pth'))

        ###########################
        #Predictor (ConvRNN) 
        #multi-step training
        #################################
        self._inputnet.eval()

        predictor_loss_multistep: List[np.float64] = []
        min_eigenvalue: List[np.float64] = []
        max_eigenvalue: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []
        
        #let the sequence length rise slowley and always when
        # validation early stopping triggers or constraints are violated 
        #=> Due to the Validation loss beeing on a fixed sequence it is not affected
        #   by the change of the loss due to different sequence length
        #       => sequence length does not effect loss but the sumation 
        #          over all batches without division over the number of batches does
        for index, seq_len in enumerate(self.sequence_length_list):
            #break training loop if constraint is failed 
            # (will then fall back to last validation checkpoint)
            constr_hold = True

            #make dataset with rising sequence length
            predictor_dataset = HybridRecurrentLinearFNNInputDataset(us, ys, seq_len ,device=self.device)
                    
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


            # time_start_pred = time.time()
            t = self.initial_decay_parameter
            
            for i in range(self.epochs_predictor_multistep):
                data_loader = data.DataLoader(
                    predictor_dataset, self.batch_size, shuffle=True, drop_last=True
                )
                total_loss = 0
                max_grad = 0
                bls_iter = 0

                #because of validation
                self._predictor.train()

                for batch_idx, batch in enumerate(data_loader):
                    self._predictor.zero_grad()

                    # Initialize predictor with state of initializer network
                    _, hx = self._initializer.forward(batch['x0'])

                    #hopefully more understandable:
                    # as we initialize the hiddenstate of the RNN we need to initialize
                    # the internal state of the _diskretized_linear the first computation
                    # of the error can be omitted for this since we can expect that the
                    # initial state as no error 

                    #we need only the last point of the x0 sequence for init of the linear
                    init_control = batch['x0_control'][:,-1:,:]
                    #inputs into the diskretized model need to be denormalised
                    init_state = utils.denormalize(batch['x0_states'][:,-1:,:], _state_mean_torch, _state_std_torch)
                    init_input = self._inputnet.forward(init_control)
                    states_next = self._diskretized_linear.forward(
                        input_forces= init_input,
                        states=init_state)
                    
                    #get all inputs
                    if self.RNNinputnetbool:
                        control_ = self._inputRNNnet.forward(batch['control'])
                    else:
                        control_ =batch['control']
                    control_lin =batch['control']

                    if self.forward_alt_bool:
                        outputs_tensor = self._Hybrid_model.forward_predictor_multistep_alt(
                            control_lin=control_lin,
                            control_=control_,
                            states_next = states_next,
                            nx=self.nx,
                            _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                            _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                            _state_mean_torch = _state_mean_torch,
                            _state_std_torch = _state_std_torch,
                        )
                    else:
                        outputs_tensor = self._Hybrid_model.forward_predictor_multistep(
                            control_lin=control_lin,
                            control_=control_,
                            states_next = states_next,
                            nx=self.nx,
                            _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                            _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                            _state_mean_torch = _state_mean_torch,
                            _state_std_torch = _state_std_torch,
                        )


                    barrier = self._predictor.get_barrier(t)

                    true_state = batch['states']
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
                    old_pars, max_grad = self._Hybrid_model.constr_savings(max_grad=max_grad)

                    self.optimizer_pred_multistep.step()
                    ########################### 
                    #Constraints Checking
                    #################################
                    constr_hold, bls_iter_ = self._Hybrid_model.constraints_checking(
                        old_pars=old_pars,
                        current_epochs = i,
                        max_epochs=self.epochs_predictor_multistep,
                        logger = logger
                    )
                    bls_iter += bls_iter_

                    if not constr_hold:
                        break
                if not constr_hold:
                    break
                ########################### 
                #main validation check for overfitt prevention
                #################################
                with torch.no_grad():
                    self._predictor.eval() #this is very important (i think to disable the dropout layers)

                    #do the whole validation dataset at once
                    states = predictor_dataset_vali.states
                    control = predictor_dataset_vali.control
                    x0 = predictor_dataset_vali.x0
                    control_prev = predictor_dataset_vali.control_prev
                    states_prev = predictor_dataset_vali.states_prev
                    x0_control = predictor_dataset_vali.x0_control
                    x0_states = predictor_dataset_vali.x0_states

                    # Initialize predictor with state of initializer network
                    _, hx = self._initializer.forward(x0)

                    #we need only the last point of the x0 sequence for init of the linear
                    init_control = x0_control[:,-1:,:]
                    #inputs into the diskretized model need to be denormalised
                    init_state = utils.denormalize(x0_states[:,-1:,:], _state_mean_torch, _state_std_torch)
                    init_input = self._inputnet.forward(init_control)
                    states_next = self._diskretized_linear.forward(
                        input_forces= init_input,
                        states=init_state)
                    
                    #get all inputs
                    if self.RNNinputnetbool:
                        control_ = self._inputRNNnet.forward(control)
                    else:
                        control_ =control
                    control_lin =control

                    if self.forward_alt_bool:
                        outputs_tensor = self._Hybrid_model.forward_predictor_multistep_alt(
                            control_lin=control_lin,
                            control_=control_,
                            states_next = states_next,
                            nx=self.nx,
                            _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                            _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                            _state_mean_torch = _state_mean_torch,
                            _state_std_torch = _state_std_torch,
                        )
                    else:
                        outputs_tensor = self._Hybrid_model.forward_predictor_multistep(
                            control_lin=control_lin,
                            control_=control_,
                            states_next = states_next,
                            nx=self.nx,
                            _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                            _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                            _state_mean_torch = _state_mean_torch,
                            _state_std_torch = _state_std_torch,
                        )

                    barrier = self._predictor.get_barrier(t)

                    true_state = states
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
                        
                    #the list is just to know in which iteration the best_val_loss
                    # was updated 
                    if validation_loss < predictor_multistep_best_val_loss.min():
                        # If it is, save the model parameters
                        best_pars = [par.clone().detach() for par in self._predictor.parameters()]
                        # torch.save(self._predictor.state_dict(), 'best_model_params.pth')
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

                if self.patience < (i-predictor_multistep_best_epoch[index]):
                    print("early stopping")
                    break

        #load the best parameters with best validation loss (early stopping)
        for best_par, new_par in zip(
                    best_pars, self._predictor.parameters()
                ):
                    new_par.data = best_par.clone()
        # self._predictor.load_state_dict(torch.load('best_model_params.pth'))

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
        control = utils.normalize(control, self.control_mean, self.control_std)


        control = torch.from_numpy(control).float().to(self.device)
        #put it in batch,sequence,state format
        control = control.unsqueeze(0)
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

            #get all inputs
            if self.RNNinputnetbool:
                control_ = self._inputRNNnet.forward(control)
            else:
                control_ =control
            control_lin =control
            #init the hidden state of our RNN
            _, hx = self._initializer.forward(init_x)

            if self.forward_alt_bool:
                outputs_tensor = self._Hybrid_model.forward_predictor_multistep_alt(
                    control_lin=control_lin,
                    control_=control_,
                    states_next = states_next,
                    nx=self.nx,
                    _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                    _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                )
            else:
                outputs_tensor = self._Hybrid_model.forward_predictor_multistep(
                    control_lin=control_lin,
                    control_=control_,
                    states_next = states_next,
                    nx=self.nx,
                    _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                    _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                )

            outputs_tensor_denorm = utils.denormalize(outputs_tensor, _state_mean_torch, _state_std_torch)

            y_np: NDArray[np.float64] = (
                outputs_tensor_denorm.cpu().detach().squeeze().numpy().astype(np.float64)
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
        #drop last next state to fit with RNN inputs and because see previous comment about 49
        prev_cont_in = controls_[:, 49:-1, :]
        lin_state_in = states_[:, 49:-1, :]
        lin_forces_in = forces_[:, 49:-1, :]

        #for the RNN we now need the values from 50 onward
        if self.RNNinputnetbool:
            curr_cont_in = self._inputRNNnet.forward(curr_cont_in)
        else:
            curr_cont_in =controls_[:, 50:, :]
        curr_cont_in_lin =controls_[:, 50:, :]

        with torch.no_grad():
            _, hx = self._initializer.forward(x0_init)
            true_input_pred_states_ = self._diskretized_linear.forward(lin_forces_in,lin_state_in)
            curr_input_forces_ = self._inputnet.forward(curr_cont_in_lin)
            filler_forces_ = self._inputnet.forward(x0_control)
            
            control_prev = prev_cont_in
            states_prev_ = lin_state_in
            control_in = curr_cont_in

            if self.forward_alt_bool:
                output_normed = self._Hybrid_model.forward_predictor_onestep_alt(
                    control_prev=control_prev, 
                    states_prev_= states_prev_,
                    control_in= control_in,
                    _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                    _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                    _state_mean_torch= _state_mean_torch,
                    _state_std_torch= _state_std_torch
                )
            else:
                output_normed = self._Hybrid_model.forward_predictor_onestep(
                    control_prev=control_prev,
                    states_prev_= states_prev_,
                    control_in= control_in,
                    _state_mean_RNN_in_torch= _state_mean_RNN_in_torch,
                    _state_std_RNN_in_torch= _state_std_RNN_in_torch,
                    _state_mean_torch= _state_mean_torch,
                    _state_std_torch= _state_std_torch
                )

            output_denormed = utils.denormalize(output_normed, _state_mean_torch, _state_std_torch)

        #fill the start that is used for initialisation with nans
        filler_nans_states = torch.full(x0_states_normed_.shape, float('nan')).to(self.device)
        filler_nans_cont = torch.full(filler_forces_.shape, float('nan')).to(self.device)
        pred_states = torch.concat((filler_nans_states,output_denormed),dim=1)
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

        control = controls_[:, 50:,:]

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

            if self.RNNinputnetbool:
                control_ = self._inputRNNnet.forward(control)
            else:
                control_ =control
            control_lin =control
            #init the hidden state of our RNN
            _, hx = self._initializer.forward(init_x)
            filler_forces_ = self._inputnet.forward(x0_control)
            input_lin = self._inputnet.forward(control_lin)


            if self.forward_alt_bool:
                outputs_tensor = self._Hybrid_model.forward_predictor_multistep_alt(
                    control_lin=control_lin,
                    control_=control_,
                    states_next = states_next,
                    nx=self.nx,
                    _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                    _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                )
            else:
                outputs_tensor = self._Hybrid_model.forward_predictor_multistep(
                    control_lin=control_lin,
                    control_=control_,
                    states_next = states_next,
                    nx=self.nx,
                    _state_mean_RNN_in_torch = _state_mean_RNN_in_torch,
                    _state_std_RNN_in_torch = _state_std_RNN_in_torch,
                    _state_mean_torch = _state_mean_torch,
                    _state_std_torch = _state_std_torch,
                )

            outputs_tensor_denorm = utils.denormalize(outputs_tensor, _state_mean_torch, _state_std_torch)

        #fill the start that is used for initialisation with nans
        filler_nans_states = torch.full(x0_states_normed_.shape, float('nan')).to(self.device)
        filler_nans_cont = torch.full(filler_forces_.shape, float('nan')).to(self.device)
        pred_states = torch.concat((filler_nans_states,outputs_tensor_denorm),dim=1)
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
