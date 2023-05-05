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
import torch.jit as jit
from .networks_JT import InputNet, InputRNNNet, DiskretizedLinear, LtiRnnConvConstr


logger = logging.getLogger(__name__)

class Hybrid_Model(jit.ScriptModule):
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

    @jit.script_method
    def forward_inputnet(self, FNN_input, Lin_input,_state_mean_torch,_state_std_torch):
        input_forces = self._inputnet.forward(FNN_input)
        states_next = self._diskretized_linear.forward(input_forces,Lin_input)
        states_next = utils.normalize(states_next, _state_mean_torch, _state_std_torch)
        return states_next


    @jit.script_method
    def forward_predictor_onestep_alt(
        self, 
        linear_inputs_prev, 
        states_prev_,
        control_in,
        _state_mean_RNN_in_torch,
        _state_std_RNN_in_torch,
        _state_mean_torch,
        _state_std_torch
        ):
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

    @jit.script_method
    def forward_predictor_onestep(
        self, 
        linear_inputs_prev, 
        states_prev_,
        control_in,
        _state_mean_RNN_in_torch,
        _state_std_RNN_in_torch,
        _state_mean_torch,
        _state_std_torch
        ):
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

    @jit.script_method
    def forward_multistep(self, input, state):

        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state