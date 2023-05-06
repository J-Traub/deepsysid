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
import torch.jit as jit
from .networks_JT import InputNet, InputRNNNet, DiskretizedLinear, LtiRnnConvConstr

# class Hybrid_Model(jit.ScriptModule):
class Hybrid_Model(nn.Module):
    def __init__(
            self, 
            predictor: LtiRnnConvConstr, 
            initializer: rnn.BasicLSTM, 
            inputnet: InputNet, 
            diskretized_linear: DiskretizedLinear, 
            inputRNNnet: InputRNNNet,
            device: torch.device
            ):
        super(Hybrid_Model, self).__init__()
        self.predictor = predictor
        self.initializer = initializer
        self.inputnet = inputnet
        self.diskretized_linear = diskretized_linear
        self.inputRNNnet = inputRNNnet
        self.device = device

    # @jit.script_method
    def forward_inputnet(self, FNN_input, Lin_input,_state_mean_torch,_state_std_torch):
        input_forces = self.inputnet.forward(FNN_input)
        states_next = self.diskretized_linear.forward(input_forces,Lin_input)
        states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)
        return states_next


    # @jit.script_method
    def forward_predictor_onestep_alt(
        self, 
        control_prev, 
        states_prev_,
        control_in,
        _state_mean_RNN_in_torch,
        _state_std_RNN_in_torch,
        _state_mean_torch,
        _state_std_torch
        ):
        linear_inputs_prev = self.inputnet.forward(control_prev)
        #computationally efficient but not good to read and understand
        #explaination: 
        # we need to initialize the internal state of the _diskretized_linear
        # because else the first state would be x0 from the training data
        # without error and wouldnt train our RNN correctly, and since we
        # use one-step prediction we dont need any loop and can just compute 
        # the following next states in one go with the training data. Thats why
        # we need the _prev values. The lin_states then represent the internal
        # states of the "current" timestep (not the prev) and we can then output those 

        lin_states = self.diskretized_linear.forward(
            input_forces= linear_inputs_prev,
            states=states_prev_, #since it's onestep prediction, the input state has no error
        )

        #just for me  (represents the timestep such that x(k+1) becomes x(k))
        lin_in_rnn = lin_states
        
        #normalize the out_lin
        out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

        #do the rnn 
        rnn_input = torch.concat((control_in, out_lin_norm),dim=2)

        res_error, _ = self.predictor.forward_alt(x_pred = rnn_input, device=self.device, hx=None)

        #calculated the corrected output and barrier
        # it is impotant to note that batch['states'] are normalised states (see loss)
        corr_states = out_lin_norm+res_error

        #denormalize to go trough possible linear system output
        corr_states_denorm = utils.denormalize(corr_states, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

        #this is functionally the same as output = corr_states_denorm but is good for understanding
        #and in case there is direct feedtrough it becomes important but then also needs an extra
        #network to correct the direct feedthrough
        output = self.diskretized_linear.calc_output(
            states= corr_states_denorm
        )
        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)

        return output_normed

    # @jit.script_method
    def forward_predictor_onestep(
        self, 
        control_prev, 
        states_prev_,
        control_in,
        _state_mean_RNN_in_torch,
        _state_std_RNN_in_torch,
        _state_mean_torch,
        _state_std_torch
        ):
        linear_inputs_prev = self.inputnet.forward(control_prev)
        #computationally efficient but not good to read and understand
        #explaination: 
        # we need to initialize the internal state of the _diskretized_linear
        # because else the first state would be x0 from the training data
        # without error and wouldnt train our RNN correctly, and since we
        # use one-step prediction we dont need any loop and can just compute 
        # the following next states in one go with the training data. Thats why
        # we need the _prev values. The lin_states then represent the internal
        # states of the "current" timestep (not the prev) and we can then output those 

        lin_states = self.diskretized_linear.forward(
            input_forces= linear_inputs_prev,
            states=states_prev_, #since it's onestep prediction, the input state has no error
        )

        #just for me  (represents the timestep such that x(k+1) becomes x(k))
        lin_in_rnn = lin_states
        
        #normalize the out_lin
        out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)


        #do the rnn 
        rnn_input = torch.concat((control_in, out_lin_norm),dim=2)
 
        res_error, _ = self.predictor.forward(x_pred = rnn_input, device=self.device, hx=None)

        #calculated the corrected output and barrier
        # it is impotant to note that batch['states'] are normalised states (see loss)
        corr_states = out_lin_norm+res_error

        #denormalize to go trough possible linear system output
        corr_states_denorm = utils.denormalize(corr_states, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

        #this is functionally the same as output = corr_states_denorm but is good for understanding
        #and in case there is direct feedtrough it becomes important but then also needs an extra
        #network to correct the direct feedthrough
        output = self.diskretized_linear.calc_output(
            states= corr_states_denorm
        )
        output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)

        return output_normed

    # @jit.script_method
    def forward_predictor_multistep_alt(
        self,
        control_lin,
        control_,
        states_next,
        nx,
        _state_mean_RNN_in_torch,
        _state_std_RNN_in_torch,
        _state_mean_torch,
        _state_std_torch,
    ):
        input_lin = self.inputnet.forward(control_lin)
        x = torch.zeros((control_.shape[0], 1, nx),device=self.device)
        
        outputs =[]
        #get the sequence dimension, sanity check: is sequence length?
        seq_len = control_.size(dim=1)
        for seq_step in range(seq_len):
            #seq_step:seq_step+1 preserves the original dimensionality
            in_lin = input_lin[:,seq_step:seq_step+1,:]
            control_in = control_[:,seq_step:seq_step+1,:]

        #TODO compare speeds if relevant
        # # split the tensors along the second dimension 
        # input_lin_list = torch.unbind(input_lin, dim=1)
        # control_list = torch.unbind(control_, dim=1)

        # # iterate over the resulting tensors
        # for in_lin, control_in in zip(input_lin_list, control_list):
            
        #     #needs batch and sequence format
        #     in_lin_ = torch.unsqueeze(in_lin, dim=1)
        #     control_in_ = torch.unsqueeze(control_in, dim=1)
            
            #just for me (represents the timestep such that x(k+1) becomes x(k))
            lin_in_rnn = states_next

            #normalize the out_lin
            out_lin_norm = utils.normalize(lin_in_rnn, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)

            rnn_input = torch.concat([control_in, out_lin_norm],dim=2)

            res_error, x = self.predictor.forward_alt_onestep(x_pred = rnn_input, device=self.device, hx=x)

            res_error = res_error.to(self.device)
            corr_state = out_lin_norm+res_error

            #denormalize the corrected state and use it as new state for the linear
            corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
            
            #this takes the denormalized corrected state as input
            states_next = self.diskretized_linear.forward(
                input_forces=in_lin,
                states=corr_state_denorm
                )

            #this is functionally the same as out_lin = lin_states but is good for understanding
            #and in case there is direct feedtrough it becomes important but then also needs an extra
            #network to correct the direct feedthrough
            output = self.diskretized_linear.calc_output(
                states= corr_state_denorm
            )
            output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
            outputs.append(output_normed)
            
        return torch.cat(outputs, dim=1)        

    # @jit.script_method
    def forward_predictor_multistep(
        self,
        control_lin,
        control_,
        states_next,
        nx,
        _state_mean_RNN_in_torch,
        _state_std_RNN_in_torch,
        _state_mean_torch,
        _state_std_torch,
    ):
        input_lin = self.inputnet.forward(control_lin)
        x = torch.zeros((control_.shape[0], 1, nx),device=self.device)
        
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

            res_error, x = self.predictor.forward(x_pred = rnn_input, device=self.device, hx=x)
            # hx has a very wierd format and is not the same as the output x
            x = [[x[0],x[0]]]
            res_error = res_error.to(self.device)
            corr_state = out_lin_norm+res_error

            #denormalize the corrected state and use it as new state for the linear
            corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
            
            #this takes the denormalized corrected state as input
            states_next = self.diskretized_linear.forward(
                input_forces=in_lin,
                states=corr_state_denorm
                )

            #this is functionally the same as out_lin = lin_states but is good for understanding
            #and in case there is direct feedtrough it becomes important but then also needs an extra
            #network to correct the direct feedthrough
            output = self.diskretized_linear.calc_output(
                states= corr_state_denorm
            )
            output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
            outputs.append(output_normed)
            
        return torch.cat(outputs, dim=1)
    
    # @jit.script_method
    # def forward_multistep(self, input, state):

    #     inputs = input.unbind(0)
    #     outputs = torch.jit.annotate(List[torch.Tensor], [])
    #     for i in range(len(inputs)):
    #         out, state = self.cell(inputs[i], state)
    #         outputs += [out]
    #     return torch.stack(outputs), state
    
    def constraints_checking(self, old_pars, current_epochs, max_epochs, logger):
    # perform backtracking line search if constraints are not satisfied
        max_iter = 100
        alpha = 0.5
        bls_iter = 0
        while not self.predictor.check_constr():
            for old_par, new_par in zip(old_pars, self.predictor.parameters()):
                new_par.data = (
                    alpha * old_par.clone() + (1 - alpha) * new_par.data
                )

            if bls_iter > max_iter - 1:
                for old_par, new_par in zip(
                    old_pars, self.predictor.parameters()
                ):
                    new_par.data = old_par.clone()
                M = self.predictor.get_constraints()
                logger.warning(
                    f'Epoch {current_epochs+1}/{max_epochs}\t'
                    f'max real eigenvalue of M: '
                    f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                    f'Backtracking line search exceeded maximum iteration. \t'
                    f'Constraints satisfied? {self.predictor.check_constr()}'
                )

                return False, bls_iter

            bls_iter += 1
        return True, bls_iter

    def constr_savings(self,max_grad):
        #stuff for constraint checking
        # gradient infos
        grads_norm = [
            torch.linalg.norm(p.grad)
            for p in filter(
                lambda p: p.grad is not None, self.predictor.parameters()
            )
        ]
        max_grad += max(grads_norm)

        # save old parameter set
        old_pars = [
            par.clone().detach() for par in self.predictor.parameters()
        ]
        return old_pars, max_grad
    

    # def epoch_wrapup(
    #         self,
    #         epoch,
    #         data_loader,
    #         epochs_with_const_decay,
    #         decay_rate,
    #         log_min_max_real_eigenvalues,
    #         total_loss,
    #         epochs_predictor_singlestep,
    #         barrier,
    #         bls_iter,
    #         max_grad,
    #         logger):
    #     i = epoch
    #      # decay t following the idea of interior point methods
    #     if i % epochs_with_const_decay == 0 and i != 0:
    #         t = t * 1 / decay_rate
    #         logger.info(f'Decay t by {decay_rate} \t' f't: {t:1f}')

    #     min_ev = np.float64('inf')
    #     max_ev = np.float64('inf')
    #     if log_min_max_real_eigenvalues:
    #         min_ev, max_ev = self.predictor.get_min_max_real_eigenvalues()

    #     loss_average = total_loss/(len(data_loader))

    #     logger.info(
    #         f'Epoch {i + 1}/{epochs_predictor_singlestep}\t'
    #         f'Total Loss (Predictor): {loss_average:1f} \t'
    #         f'Barrier: {barrier:1f}\t'
    #         f'Backtracking Line Search iteration: {bls_iter}\t'
    #         f'Max accumulated gradient norm: {max_grad:1f}'
    #     )
    #     print(
    #         f'Epoch {i + 1}/{epochs_predictor_singlestep}\t'
    #         f'Total Loss (Predictor): {loss_average:1f} \t'
    #         f'Barrier: {barrier:1f}\t'
    #         f'Backtracking Line Search iteration: {bls_iter}\t'
    #         f'Max accumulated gradient norm: {max_grad:1f}'
    #     )
    #     predictor_loss.append([i,np.float64(loss_average)])
    #     barrier_value.append(barrier.cpu().detach().numpy())
    #     backtracking_iter.append(np.float64(bls_iter))
    #     gradient_norm.append(np.float64(max_grad))
    #     max_eigenvalue.append(np.float64(max_ev))
    #     min_eigenvalue.append(np.float64(min_ev))       