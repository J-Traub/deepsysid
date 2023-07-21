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


class TimeSeriesDataset(data.Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, control_seqs, state_seqs, device):
        self.control,self.next_state,self.state = self.__load_data(
            control_seqs, state_seqs
        )
        self.subbatchsize = 50
        subbatchnum = int(self.control.shape[0]/self.subbatchsize)
        self.control = np.resize(self.control,(subbatchnum,self.subbatchsize,self.control.shape[1]))
        self.next_state = np.resize(self.next_state,(subbatchnum,self.subbatchsize,self.next_state.shape[1]))
        self.state = np.resize(self.state,(subbatchnum,self.subbatchsize,self.state.shape[1]))

        self.control = torch.from_numpy(self.control).float().to(device)
        self.next_state =  torch.from_numpy(self.next_state).float().to(device)
        self.state =  torch.from_numpy(self.state).float().to(device)


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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'FNN_input': self.control[idx],
            'next_state': self.next_state[idx], 
            'Lin_input': self.state[idx],
            }


class HybridRecurrentLinearInitializerDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]], #actually not a list just array
        state_seqs: List[NDArray[np.float64]], #actually not a list just array
        pred_next_state_seq: List[NDArray[np.float64]], #actually not a list just array
        sequence_length: int,
        scaling : List[float]
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.pred_ns_dim = pred_next_state_seq[0].shape[1]
        self.x, self.y = self.__load_data(control_seqs , state_seqs, pred_next_state_seq, scaling)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        pred_next_state_seq: List[NDArray[np.float64]],
        scaling: List[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state, pred_next_state in zip(control_seqs, state_seqs, pred_next_state_seq):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1) / self.sequence_length
            )

            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim + self.pred_ns_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                #the error correction RNN gets either the current control (t) or the previous control (t-1)
                # doesnt seem to make all to much differnece and both have their merrits but current control (t)
                # makes things clearer. Also gets the predicted current state (t) and 
                #the previous (one-step)error (t-1) with the previous state (t-1) and the predicted previous state (t-1)
                #however the predicted state is inherently one step ahead(t+1) so it would have to be (t-2) and since we
                #cant go negativ the other states have to start at (t+1)
                x[idx, :, :] = np.hstack(
                    (
                        control[time + 2  : time + 2  + self.sequence_length, :]* scaling[0],
                        pred_next_state[time + 1 : time + 1 + self.sequence_length, :]*scaling[1], 
                        state[time + 1 : time + 1 + self.sequence_length, :] - pred_next_state[time : time  + self.sequence_length, :],
                    )
                )

                #since we have to start with time + 1 in the x and we want to predict the next error
                #we have to go time + 2 on state 
                y[idx, :, :] = state[time + 2 : time + 2 + self.sequence_length, :] - pred_next_state[time +1 : time +1 + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {'x': self.x[idx], 'y': self.y[idx]}


class HybridRecurrentLinearFNNInputDataset(data.Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        pred_next_state_seq: List[NDArray[np.float64]],
        scaling : List[float], #needed for small gain
        sequence_length: int,
        device: torch.device,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.pred_ns_dim = pred_next_state_seq[0].shape[1]
        dataset = self.__load_data(control_seqs, state_seqs, pred_next_state_seq, scaling)

        self.x0 = torch.from_numpy(dataset['x0']).float().to(device)
        self.y0 = torch.from_numpy(dataset['y0']).float().to(device)
        self.control = torch.from_numpy(dataset['cont']).float().to(device)
        self.states = torch.from_numpy(dataset['stat']).float().to(device)
        self.x0_control = torch.from_numpy(dataset['x0_control']).float().to(device)
        self.x0_states = torch.from_numpy(dataset['x0_states']).float().to(device)
        self.control_prev = torch.from_numpy(dataset['cont_prev']).float().to(device)
        self.states_prev = torch.from_numpy(dataset['stat_prev']).float().to(device)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        pred_next_state_seq: List[NDArray[np.float64]],
        scaling: List[float],
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

        for control, state, pred_next_state in zip(control_seqs, state_seqs,pred_next_state_seq):
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            x0 = np.zeros(
                (n_samples, self.sequence_length-2, self.control_dim + self.state_dim + self.pred_ns_dim),
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


            for idx in range(n_samples):
                time = idx * self.sequence_length

                x0[idx, :, :] = np.hstack(
                    (
                    #take care if we are working with the previous control or the current control as input
                        control[time + 2  : time  + self.sequence_length, :]* scaling[0],
                        pred_next_state[time + 1 : time -1 + self.sequence_length, :]*scaling[1], 
                        state[time + 1 : time -1 + self.sequence_length, :] - pred_next_state[time : time -2 + self.sequence_length, :],
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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