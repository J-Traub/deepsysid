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
from ..datasets import RecurrentInitializerDataset, RecurrentPredictorDataset,FixedWindowDataset

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

class FixedWindowDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        window_size: int,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> None:
        self.window_size = window_size
        self.window_input, self.state_true = self.__load_dataset(
            control_seqs, state_seqs
        )

    def __load_dataset(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        window_input = []
        state_true = []
        for control, state in zip(control_seqs, state_seqs):
            for time in range(
                self.window_size, control.shape[0] - 1, int(self.window_size / 4) + 1
            ):
                window_input.append(
                    np.concatenate(
                        (
                            control[
                                time - self.window_size + 1 : time + 1, :
                            ].flatten(),
                            state[time - self.window_size : time, :].flatten(),
                        )
                    )
                )
                state_true.append(state[time + 1, :])

        return np.vstack(window_input), np.vstack(state_true)

    def __len__(self) -> int:
        return self.window_input.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return dict(
            window_input=self.window_input[idx], state_true=self.state_true[idx]
        )

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
    
#TODO: NDArrays do not work with pydantic i think
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


        # Ad = [
        # [0.95013655722506018541650973929791,                                     0,                                   0,                                   0,                                    0],
        # [                                 0,    0.97301200162687573325115408806596,  0.15010745470066019779942223522085, -0.31718659560396628149803177620925,   0.20920658092307373165930073355412],
        # [                                 0,   0.010989971476874587849592579402724,  0.49397156170454681323178647289751, -0.41836210372089926989858099659614,  -0.70866754485476979308344880337245],
        # [                                 0, -0.0057811139185215635466486006066589, 0.001755356723630133479116532946307,  0.79438406062355182424283839281998, 0.0016112562367335591176353837283841],
        # [                                 0,  0.0058028063111857305922391958574735,  0.80074622709314469126695712475339, -0.24235145964997839573840110460878,   0.60750434207042802725595720403362]
        # ],
        # Bd = [
        # [0.0000025440532028030539262993068444496,                                           0,                                           0,                                           0],
        # [                                      0,    0.00000043484020054323831296305647928224,  -0.000000058044367250997092370129508626456, -0.0000000029457123101043193733947037104334],
        # [                                      0, -0.0000000078192931709820619618461943968642,    0.00000019659355907511883409705707830006, -0.0000000031007639013231448259531736742246],
        # [                                      0,  0.0000000076881039158870533106903728684869, -0.0000000010864635093489020602185588640679,   0.000000011472201145655161557752056949566],
        # [                                      0, -0.0000000043908753737658910011124498422742,    0.00000010884895344888811376612590062218, -0.0000000010957350422826603352287007565489]
        # ],
        # Cd = [
        # [1.0,   0,   0,   0,   0],
        # [  0, 1.0,   0,   0,   0],
        # [  0,   0, 1.0,   0,   0],
        # [  0,   0,   0, 1.0,   0],
        # [  0,   0,   0,   0, 1.0]        
        # ],
        # Dd = [
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0]
        # ],
        # ssv_input = [49000.0, 0, 0, 0],
        # ssv_states = [5.0, 0, 0, 0, 0],

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

    #TODO: include residual error
    def forward(self, 
                input_forces: torch.Tensor,
                states: torch.Tensor ,
                #residual_errors: torch.Tensor
                ) -> torch.Tensor:
        #calculates x_(k+1) = Ad*x_k + Bd*u_k + Ad*e_k
        #           with x_corr_k = x_k+e_k the residual error corrected state
        #           y_k = x_k
        #and shifts input, and output to fit with the actual system

        #shift the input to the linearized input
        delta_in = input_forces - self.ssv_input
        #add the correction calculated by the RNN to the state
        # can be seen as additional input with Ad matrix as input Matrix
        states_corr = states #+ residual_errors
        #also shift the states since the inital state needs to be shifted or if i want to do one step predictions
        delta_states_corr = states_corr - self.ssv_states
        #x_(k+1) = Ad*(x_k+e_k) + Bd*u_k
        #for compatability with torch batches we transpose the equation
        delta_states_next = torch.mm(delta_states_corr, self.Ad.transpose(0,1)) + torch.mm(delta_in, self.Bd.transpose(0,1)) 
        #shift the linearized output back to the output
        y_linear = states + self.ssv_states
        states_next = delta_states_next + self.ssv_states
        return y_linear, states_next
    


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



#TODO: should i normalize the input? does that make sense?
#TODO: make the input net configureable
#Does only simulate the InputFNN output since that is what is trained
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


    #TODO:make it recurrent (not just one step prediciton)
    #u mse zu groß (schauen ob shift richtig) => jetzt alles richtig
    #vergleich mit LSTM aufschreiben
    #enable testing in vscode
    #probably needs overfitting protection else it will just memorize the inputs
    # => for the FNN regularization/overfitting protection will probably be the 
    #    main thing since otherwise it will definitely not learn the correct inputs
    #    but the perfect inputs for the trainingset 
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
                state_pred, states_next = self._diskretized_linear.forward(input_forces,true_states)
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
                state_pred, states_next = self._diskretized_linear.forward(input_forces,true_states)
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




            # time2 = time.time()

            #check validation for overfitt prevention
            #TODO: need to smooth loss somehow to get a good stop criterion
            # => can also just save the best parameters on the validation set and use them at the end
            # => or stop if the best parameters havent changed for a long time 
            # => could use the actual trajectory metric stuff as validation loss if it doesnt take to long
            #TODO: to prevent overfitting on the validation dataset might need to consider splitting the data new each training
            with torch.no_grad():
                controls_vali = torch.from_numpy(dataset_vali.control)
                states_vali = torch.from_numpy(dataset_vali.state)
                true_next_states_vali = torch.from_numpy(dataset_vali.next_state)

                # for some reason dataloader iteration is very slow otherwise
                #to device needs to be after denormalize else it cant calculate with the numpy std and mean
                controls_vali = controls_vali.reshape(-1,controls_vali.shape[-1]).float().to(self.device)
                states_vali = states_vali.reshape(-1,states_vali.shape[-1])
                states_vali = utils.denormalize(states_vali, self._state_mean, self._state_std).float().to(self.device)
                true_next_states_vali = true_next_states_vali.reshape(-1,true_next_states_vali.shape[-1])
                true_next_states_vali = utils.denormalize(true_next_states_vali, self._state_mean, self._state_std).float().to(self.device)

                input_forces_vali = self._inputnet.forward(controls_vali)
                state_pred_vali, states_next_vali = self._diskretized_linear.forward(input_forces_vali,states_vali)
                validation_loss = F.mse_loss(
                    states_next_vali, true_next_states_vali
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
            raise ValueError('Model has not been trained and cannot be saved.')
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
            output, states_next = self._diskretized_linear.forward(input_forces=input_lin,states=last_init_state)
            outputs =[]
            outputs.append(output)
            input_lin = self._inputnet.forward(control_)
            for in_lin in input_lin:
                #unsqueeze for correct shape for the _diskretized_linear
                output, states_next = self._diskretized_linear.forward(input_forces=in_lin.unsqueeze(0),states=states_next)
                outputs.append(output)

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
            raise ValueError('Model has not been trained and cannot be saved.')   

        self._inputnet.eval()
        self._diskretized_linear.eval()
        controls_ = utils.normalize(controls, self._control_mean, self._control_std)

        controls_ = torch.from_numpy(controls_).float().to(self.device)
        states_ = torch.from_numpy(states).float().to(self.device)
        forces_ = torch.from_numpy(forces).float().to(self.device)

        with torch.no_grad():
            input_forces = self._inputnet.forward(controls_)
            state_pred, states_next = self._diskretized_linear.forward(input_forces,states_)
            _, states_next_with_true_input_forces = self._diskretized_linear.forward(forces_,states_)


        return (input_forces.detach().cpu().numpy().astype(np.float64),
                 states_next.detach().cpu().numpy().astype(np.float64),
                 states_next_with_true_input_forces.detach().cpu().numpy().astype(np.float64))



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
