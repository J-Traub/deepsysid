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