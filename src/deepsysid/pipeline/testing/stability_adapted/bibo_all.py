import logging
import time
from typing import List, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from ....models import utils
from ....models.base import (
    DynamicIdentificationModel,
    NormalizedControlStateModel,
)
from ..base import TestResult, TestSequenceResult, TestSimulation
from ..io import split_simulations
from .base import BaseStabilityTest, StabilityTestConfig

logger = logging.getLogger(__name__)


class BiboStabilityTest(BaseStabilityTest):
    CONFIG = StabilityTestConfig

    def __init__(self, config: StabilityTestConfig, device_name: str) -> None:
        super().__init__(config, device_name)

        self.control_dim = len(config.control_names)

        self.device_name = device_name
        self.window_size = config.window_size
        self.horizon_size = config.horizon_size
        self.evaluation_sequence = config.evaluation_sequence

        self.initial_mean_delta = config.initial_mean_delta
        self.initial_std_delta = config.initial_std_delta
        self.optimization_lr = config.optimization_lr
        self.optimization_steps = config.optimization_steps
        self.regularization_scale = config.regularization_scale
        self.clip_gradient_norm = config.clip_gradient_norm

    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation]
    ) -> TestResult:

        # if not isinstance(model, NormalizedHiddenStateInitializerPredictorModel):
        #     return TestResult(list(), dict())

        test_sequence_results: List[TestSequenceResult] = []

        time_start_test = time.time()

        if isinstance(self.evaluation_sequence, int):
            logger.info(
                f'Test bibo stability for sequence number {self.evaluation_sequence}'
            )
            print(f'Test bibo stability for sequence number {self.evaluation_sequence}')
            dataset = list(
                split_simulations(self.window_size, self.horizon_size, simulations)
            )
            sim = dataset[self.evaluation_sequence]
            test_sequence_results.append(
                self.evaluate_stability_of_sequence(
                    model=model,
                    device_name=self.device_name,
                    control_dim=self.control_dim,
                    initial_control=sim.initial_control,
                    initial_state=sim.initial_state,
                    true_control=sim.true_control,
                )
            )

        elif self.evaluation_sequence == 'all':
            logger.info(f'Test bibo stability for {self.evaluation_sequence} sequences')
            for idx_data, sim in enumerate(
                split_simulations(self.window_size, self.horizon_size, simulations)
            ):
                logger.info(f'Sequence number: {idx_data}')

                test_sequence_results.append(
                    self.evaluate_stability_of_sequence(
                        model=model,
                        device_name=self.device_name,
                        control_dim=self.control_dim,
                        initial_control=sim.initial_control,
                        initial_state=sim.initial_state,
                        true_control=sim.true_control,
                    )
                )

        time_end_test = time.time()

        test_time = time_end_test - time_start_test

        return TestResult(
            sequences=test_sequence_results, metadata=dict(test_time=[test_time])
        )

    def evaluate_stability_of_sequence(
        self,
        model: NormalizedControlStateModel,
        device_name: str,
        control_dim: int,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        true_control: NDArray[np.float64],
    ) -> TestSequenceResult:
        if (
            model._state_mean is None
            or model._state_std is None
            or model._control_mean is None
            or model._control_std is None
        ):
            raise ValueError(
                'Mean and standard deviation is not initialized in the model'
            )
        _state_mean_torch = torch.from_numpy(model._state_mean).float().to(device_name)
        _state_std_torch = torch.from_numpy(model._state_std).float().to(device_name)
        _state_mean_RNN_in_torch = torch.from_numpy(model._state_mean_RNN_in).float().to(device_name)
        _state_std_RNN_in_torch = torch.from_numpy(model._state_std_RNN_in).float().to(device_name)

        model._diskretized_linear.eval()
        model._inputRNNnet.eval()
        model._inputnet.eval()
        model._initializer.eval()
        model._predictor.eval()

        model._inputnet.to(device_name)
        model._inputRNNnet.to(device_name)
        model._diskretized_linear.to(device_name)
        model._predictor.to(device_name)
        model._initializer.to(device_name)

        control = true_control
        
        init_cont = utils.normalize(initial_control, model.control_mean, model.control_std)
        init_state = utils.normalize(initial_state, model.state_mean, model.state_std)
        control = utils.normalize(control, model.control_mean, model.control_std)

        if model.no_bias:
            init_state = utils.normalize(initial_state, model.ssv_states_np, model.state_std)


        control = torch.from_numpy(control).float().to(device_name)
        #put it in batch,sequence,state format
        control = control.unsqueeze(0)
        init_cont = torch.from_numpy(init_cont).float().to(device_name)
        init_state = torch.from_numpy(init_state).float().to(device_name)
        

        last_init_state = init_state[-1,:].unsqueeze(0).unsqueeze(0)
        
        size_ = control.size()
        if model.bibo == "bibo_sg":
            delta = torch.normal(
                self.initial_mean_delta,
                self.initial_std_delta,
                size=[size_[0],size_[1],10],
                requires_grad=True,
                device=device_name,
            )

            delta_rnn = delta[:,:,:6]
            delta_lin = delta[:,:,6:]

        if model.bibo == "bibo_lin":
            delta = torch.normal(
                self.initial_mean_delta,
                self.initial_std_delta,
                size=[size_[0],size_[1],4],
                requires_grad=True,
                device=device_name,
            )        

        if model.bibo == "bibo_full":
            delta = torch.normal(
                self.initial_mean_delta,
                self.initial_std_delta,
                size=[size_[0],size_[1],6],
                requires_grad=True,
                device=device_name,
            )

        # optimizer
        opt = torch.optim.Adam(  # type: ignore
            [delta], lr=self.optimization_lr*100, maximize=True
        )

        gamma_2: Optional[np.float64] = None
        for step_idx in range(self.optimization_steps):

            if model.bibo == "bibo_full":
                #Delta distrubance on normalized hybrid model input
                control = control + delta

            #get all inputs
            if model.RNNinputnetbool:
                control_ = model._inputRNNnet.forward(control)
            else:
                control_ =control
            control_lin =control

            if model.bibo == "bibo_sg":
                #Delta distrubance on normalized rnn input
                control_ = control_ + delta_rnn

            input_lin = model._inputnet.forward(control_lin)
            if model.bibo == "bibo_sg" or model.bibo == "bibo_lin":
                
                inln = input_lin.cpu().detach().numpy().astype(np.float64)
                input_lin_mean, input_lin_std = utils.mean_stddev(inln)
                if model.no_bias:
                    input_lin_mean = np.zeros_like(input_lin_mean)
                input_lin_mean_torch = torch.from_numpy(input_lin_mean).float().to(device_name)
                input_lin_std_torch = torch.from_numpy(input_lin_std).float().to(device_name)
                input_lin_ = utils.normalize(input_lin, input_lin_mean_torch, input_lin_std_torch)

                if model.bibo == "bibo_sg":
                    #Delta distrubance on normalized linear input (after normalized FNN output)
                    input_lin_ = input_lin_ + delta_lin
                if model.bibo == "bibo_lin":
                    #Delta distrubance on normalized linear input (after normalized FNN output)
                    input_lin_ = input_lin_ + delta

                input_lin = utils.denormalize(input_lin_, input_lin_mean_torch, input_lin_std_torch)


            #need to initialize with 0 to avoid bias terms 
            x = torch.zeros((control_.shape[0], 1, model.nx),device=device_name)
            states_next = torch.zeros_like(last_init_state)

            
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
                if model.normed_linear:
                    out_lin_norm = lin_in_rnn

                #the order is important because the inputs will get scaled individually inside the RNN
                rnn_input = torch.concat([control_in, out_lin_norm],dim=2)

                if model.forward_alt_bool:
                    res_error, hx = model._predictor.forward_alt_onestep(x_pred = rnn_input, device=device_name, hx=x)
                else:
                    res_error, hx = model._predictor.forward_onestep(x_pred = rnn_input, device=device_name, hx=x)
                
                corr_state = out_lin_norm+res_error
                if model.only_lin:
                    corr_state = out_lin_norm

                #denormalize the corrected state and use it as new state for the linear
                corr_state_denorm = utils.denormalize(corr_state, _state_mean_RNN_in_torch, _state_std_RNN_in_torch)
                if model.normed_linear:
                    corr_state_denorm = corr_state

                #this takes the denormalized corrected state as input
                states_next = model._diskretized_linear.forward(
                    input_forces=in_lin,
                    states=corr_state_denorm
                    )

                #this is functionally the same as out_lin = lin_states but is good for understanding
                #and in case there is direct feedtrough it becomes important but then also needs an extra
                #network to correct the direct feedthrough
                output = model._diskretized_linear.calc_output(
                    states= corr_state_denorm
                )
                output_normed = utils.normalize(output, _state_mean_torch, _state_std_torch)
                if model.normed_linear:
                    output_normed = output
                outputs.append(output_normed)
                
            outputs_tensor= torch.cat(outputs, dim=1)     




            # outputs_tensor_denorm = utils.denormalize(outputs_tensor, _state_mean_torch, _state_std_torch)
            y_hat_a = outputs_tensor
            y_hat_a = y_hat_a.squeeze()

            ###############
            # control_squeezed = input_lin_.squeeze()
            if model.bibo == "bibo_sg":
                control__ = torch.cat([control_,input_lin_], dim=2)
                control_squeezed = control__.squeeze()
                input_mean = np.concatenate([model.control_mean,input_lin_mean_torch], axis=-1)
                input_std = np.concatenate([model.control_std,input_lin_std_torch], axis=-1)
                

            if model.bibo == "bibo_lin":
                control_squeezed = input_lin_.squeeze()
                input_mean = input_lin_mean
                input_std = input_lin_std

            if model.bibo == "bibo_full":
                control_squeezed = control.squeeze()
                input_mean = model.control_mean
                input_std = model.control_std
            ###############

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment

            #i think control is the correct input for the norm calc
            #since we are interested in the norm from into linear to out of linear + error correction

            regularization = self.regularization_scale * torch.log(
                utils.sequence_norm(control_squeezed)
            )
            gamma_2_torch = utils.sequence_norm(y_hat_a) / utils.sequence_norm(control_squeezed)
            L = gamma_2_torch + regularization
            L.backward()
            torch.nn.utils.clip_grad_norm_(delta, self.clip_gradient_norm)
            opt.step()

            gamma_2 = gamma_2_torch.cpu().detach().numpy()
            print(
                f'step bibo: {step_idx} \t '
                f'gamma_2: {gamma_2:.3f} \t '
            )
            if step_idx % 99 == 0:
                logger.info(
                    f'step: {step_idx} \t '
                    f'gamma_2: {gamma_2:.3f} \t '
                    f'gradient norm: {torch.norm(delta.grad):.3f} '
                    f'\t -log(norm(denominator)): {regularization:.3f}'
                )
                print(
                    f'step: {step_idx} \t '
                    f'gamma_2: {gamma_2:.3f} \t '
                    f'gradient norm: {torch.norm(delta.grad):.3f} '
                    f'\t -log(norm(denominator)): {regularization:.3f}'
                )

        return TestSequenceResult(
            inputs=dict(
                a=utils.denormalize(
                    control_squeezed.cpu().detach().numpy().squeeze(),
                    input_mean, 
                    input_std,
                )
            ),
            outputs=dict(
                a=utils.denormalize(
                    y_hat_a.cpu().detach().numpy(), model.state_mean, model.state_std
                )
            ),
            metadata=dict(stability_gain=np.array([gamma_2])),
        )
