from typing import List, Optional, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from . import utils
from .base import (
    DynamicIdentificationModel,
    DynamicIdentificationModelConfig,
    FixedWindowModel,
)


class LinearModel(DynamicIdentificationModel):
    def __init__(self, config: DynamicIdentificationModelConfig):
        super().__init__(config)

        self.regressor = LinearRegression(fit_intercept=True)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_stddev: Optional[NDArray[np.float64]] = None
        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_stddev: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> None:
        assert len(control_seqs) == len(state_seqs)
        assert control_seqs[0].shape[0] == state_seqs[0].shape[0]
        assert control_seqs[0].shape[1] == self.control_dim
        assert state_seqs[0].shape[1] == self.state_dim

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)

        train_x_list, train_y_list = [], []
        for i in range(len(control_seqs)):
            control = control_seqs[i]
            state = state_seqs[i]

            control = utils.normalize(control, self.control_mean, self.control_stddev)
            state = utils.normalize(state, self.state_mean, self.state_stddev)

            x = np.hstack((control[1:], state[:-1]))
            y = state[1:]

            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.vstack(train_x_list)
        train_x = self._map_input(train_x)
        train_y = np.vstack(train_y_list)

        self.regressor.fit(train_x, train_y)

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert initial_control.shape[1] == self.control_dim
        assert initial_state.shape[1] == self.state_dim
        assert control.shape[1] == self.control_dim

        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        control = utils.normalize(control, self.control_mean, self.control_stddev)
        state = utils.normalize(initial_state, self.state_mean, self.state_stddev)[-1]

        pred_states = np.array(
            [
                np.squeeze(
                    self.regressor.predict(
                        self._map_input(
                            np.concatenate((control[i], state)).reshape(1, -1)
                        )
                    )
                )
                for i in range(control.shape[0])
            ],
            dtype=np.float64,
        )

        pred_states = utils.denormalize(pred_states, self.state_mean, self.state_stddev)

        return pred_states

    def save(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], 'w') as f:
            f.create_dataset('coef_', data=self.regressor.coef_)
            f.create_dataset('intercept_', data=self.regressor.intercept_)

            f.create_dataset('control_mean', data=self.control_mean)
            f.create_dataset('control_stddev', data=self.control_stddev)
            f.create_dataset('state_mean', data=self.state_mean)
            f.create_dataset('state_stddev', data=self.state_stddev)

    def load(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], 'r') as f:
            self.regressor.coef_ = f['coef_'][:].astype(np.float64)
            self.regressor.intercept_ = f['intercept_'][:].astype(np.float64)

            self.control_mean = f['control_mean'][:].astype(np.float64)
            self.control_stddev = f['control_stddev'][:].astype(np.float64)
            self.state_mean = f['state_mean'][:].astype(np.float64)
            self.state_stddev = f['state_stddev'][:].astype(np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return ('hdf5',)

    def get_parameter_count(self) -> int:
        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            return (self.control_dim + self.state_dim) * self.state_dim + self.state_dim

        count = 0
        if len(self.regressor.coef_.shape) == 2:
            count += np.product(self.regressor.coef_.shape)
        else:
            count += self.regressor.coef_.shape[0]
        count += self.regressor.intercept_.shape[0]
        return count

    def _map_input(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x


class LinearLagConfig(DynamicIdentificationModelConfig):
    lag: int


class LinearLag(FixedWindowModel[LinearRegression]):
    """
    Lag applies to control inputs and system states.
    Similar to a NARX model.

    x(t) = P*s(t) with s(t)=[u(t-W) ... u(t) x(t-W) ... x(t-1)]
    """

    CONFIG = LinearLagConfig

    def __init__(self, config: LinearLagConfig) -> None:
        super().__init__(
            window_size=config.lag,
            regressor=LinearRegression(fit_intercept=True),
        )

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

    def save(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], 'w') as f:
            f.attrs['window_size'] = self.window_size
            f.create_dataset('coef_', data=self.regressor.coef_)
            f.create_dataset('intercept_', data=self.regressor.intercept_)

            f.create_dataset('control_mean', data=self.control_mean)
            f.create_dataset('control_stddev', data=self.control_stddev)
            f.create_dataset('state_mean', data=self.state_mean)
            f.create_dataset('state_stddev', data=self.state_stddev)

    def load(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], 'r') as f:
            self.window_size = f.attrs['window_size']
            self.regressor.coef_ = f['coef_'][:]
            self.regressor.intercept_ = f['intercept_'][:]

            self.control_mean = f['control_mean'][:]
            self.control_stddev = f['control_stddev'][:]
            self.state_mean = f['state_mean'][:]
            self.state_stddev = f['state_stddev'][:]

    def get_file_extension(self) -> Tuple[str, ...]:
        return ('hdf5',)

    def get_parameter_count(self) -> int:
        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            return (
                self.state_dim
                * (
                    self.control_dim
                    + self.window_size * (self.control_dim + self.state_dim)
                )
                + self.state_dim
            )

        count = 0
        if len(self.regressor.coef_.shape) == 2:
            count += np.product(self.regressor.coef_.shape)
        else:
            count += self.regressor.coef_.shape[0]
        count += self.regressor.intercept_.shape[0]
        return count


class QuadraticControlLag(LinearLag):
    def _map_regressor_input(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Apply square to control inputs
        cmask = np.ones(self.control_dim, dtype=bool)
        tmask = np.zeros(self.state_dim, dtype=bool)
        mask = np.concatenate(
            (np.tile(np.concatenate((cmask, tmask)), self.window_size), cmask)
        )
        squared_control = np.square(x[:, mask])
        x = np.hstack((x, squared_control))
        return x

    def get_parameter_count(self) -> int:
        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            return (
                self.state_dim
                * (
                    2 * self.control_dim
                    + self.window_size * (2 * self.control_dim + self.state_dim)
                )
                + self.state_dim
            )

        return super().get_parameter_count()
