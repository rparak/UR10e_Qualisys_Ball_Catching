## =========================================================================== ##
# MIT License
# Copyright (c) 2026 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
## =========================================================================== ##
# Author   : Roman Parak
# File Name: ../Learning/Training.py
## =========================================================================== ##

# CSV (CSV file reading and writing)
import csv
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# PyTorch (Tensors and automatic differentiation; the training only) [pip3 install torch]
import torch
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the training configuration
from Configuration.Core import Training_Configuration_Str
#   ../Flight/Core - the one flight model and its state function (the truth a predictor is trained on)
from Flight.Core import Flight_Model_Str, Simulate, CONST_SIMULATION_STEP
#   ../Dataset/Core - a generated dataset
from Dataset.Core import Dataset_Str
#   ../Learning/Core - the inputs, the normalisation, the basis
from Learning.Core import Prefix_Features, Normalisation_Str, Chebyshev_Fit, CONST_HORIZON, CONST_EPOCH_COLUMNS
#   ../Learning/Models - the network and the query
from Learning.Models import Network_Cls, Query

"""
Description:
    Initialization of constants.

    The training of a learned predictor (SPEC §2.13.7, §7.8; F8): the flights of a
    partition prepared with their exact truth (Flight.Core.Simulate on its grid), the normalisation fitted on
    the training prefixes only, the loss on the trajectory, and one model per seed with early stopping on the
    simulated validation flights. A prefix is the first N observed samples of a flight, N drawn from the
    smallest to the largest configured value. The loss is the mean over the prefixes
    of the RMS position error over the remainder of the flight (the shape of the reported metric; Q5), plus
    Loss.Velocity_Weight times the same for the velocity. It reads no file and selects no partition: the
    caller gives the flights. On the CPU a run is reproduced bit for bit (one thread, deterministic
    algorithms; F8 Part A).
"""
# The threads of a training on the CPU: more gave no speed at this size (5.8-6.1 s per epoch at 1, 2, 4 and
# 8 threads, F8 Part A), and the weights are the same at any count.
#   Unit [int]
CONST_THREADS = 1
# The query times drawn per prefix and step. *Assumed* (F8 Part A's pilots).
#   Unit [int]
CONST_QUERIES = 48
# The truth is kept, and the model is trained, this long past the floor (the free flight continued),
# so that the predicted height has a root at the floor. *Assumed*.
#   Unit [float, s]
CONST_MARGIN = 0.1
# The step of the times the validation loss is taken at, from the first multiple at or after t_0 to the floor.
# *Assumed*.
#   Unit [float, s]
CONST_LOSS_STEP = 0.01
# The share of the steps in which the one-cycle learning rate rises, and the largest gradient norm. *Assumed*
# (F8 Part A's pilots).
#   Unit [float, -]
CONST_WARM_UP = 0.1
CONST_GRADIENT_NORM = 1.0
# The training flights the scale of the basis coefficients is fitted on (the first of them). *Assumed*.
#   Unit [int]
CONST_COEFFICIENT_FLIGHTS = 1000
# Inside a square root, so that its gradient stays finite at a zero error.
#   Unit [float, m^2]
CONST_EPSILON = 1e-10
# The number format of epochs.csv (SPEC §9.6; its columns are Learning.Core's).
#   Unit [string]
CONST_EPOCH_FORMAT = '%.9f'


class Training_Error_Cls(ValueError):
    """
    Description:
        A training cannot be made as asked: no flight with enough samples, a device that is not there. The
        message names the value and what to do.
    """


@dataclass
class Flights_Str:
    """
    Description:
        The flights of one partition, prepared for training or early stopping: the observed samples padded to M
        (as observed, never resampled) and the exact truth on Simulate's grid from the release to CONST_MARGIN
        past the floor.
    """

    # The first samples of each flight (at most M), their count and the frame rate.
    #   Unit [Matrix<float> FxM, s], [Tensor<float> FxMx3, m], [Vector<int> 1xF], [Vector<float> 1xF, Hz]
    Time: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))
    Samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The release state and the true state at every grid time k * CONST_SIMULATION_STEP.
    #   Unit [Matrix<float> Fx6, m, m/s], [Tensor<float32> FxKx6, m, m/s]
    Release: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    Truth: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 6), dtype=np.float32))
    # The last grid index at or before each flight's floor time.
    #   Unit [Vector<int> 1xF]
    Floor_Index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def Prepare_Flights(data: Dataset_Str, indices: np.ndarray, samples: int, *, Flight_Model_Str: Flight_Model_Str) -> Flights_Str:
    """
    Description:
        The given flights of a dataset for training: their first `samples` observed samples, and their truth
        from the release state by Simulate on its grid. Only the flights given are read (the caller selects the
        partition).

    Args:
        (1) data [Dataset_Str(object)]: The dataset.
        (2) indices [Vector<int> 1xF]: The flights.
        (3) samples [int]: The samples kept per flight (the largest N).
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the dataset's flight model.

    Returns:
        (1) parameter [Flights_Str(object)]: The flights.
    """

    F = len(indices)
    flights = Flights_Str(Time=np.zeros((F, samples)), Position=np.zeros((F, samples, 3)), Samples=np.zeros(F, dtype=np.int64),
                          Rate=data.Rate[indices].astype(np.float64), Release=data.Release_State[indices].astype(np.float64))
    for row, i in enumerate(indices):
        time, position = data.Observations(int(i))
        n = min(len(time), samples)
        flights.Time[row, :n], flights.Position[row, :n], flights.Samples[row] = time[:n], position[:n], n
    floor = data.Floor_Time[indices]
    h = CONST_SIMULATION_STEP
    K = int(np.ceil((floor.max() + CONST_MARGIN) / h)) + 2 if F else 0
    flights.Truth = np.empty((F, K, 6), dtype=np.float32)
    for a in range(0, F, 500):
        flights.Truth[a:a + 500] = Simulate(flights.Release[a:a + 500], np.arange(K) * h, Flight_Model_Str=Flight_Model_Str)
    flights.Floor_Index = np.floor(floor / h + 1e-9).astype(np.int64)
    return flights


def Merge_Flights(parts: tp.List[Flights_Str]) -> Flights_Str:
    """
    Description:
        The flights of several datasets as one set (the datasets trained on are pooled, SPEC §8.3); a shorter
        truth grid is padded with zeros, which no index reaches (a flight is read to CONST_MARGIN past its own
        floor only).

    Args:
        (1) parts [List<Flights_Str>]: The sets, with the same number of samples per flight.

    Returns:
        (1) parameter [Flights_Str(object)]: The pooled set.
    """

    K = max(part.Truth.shape[1] for part in parts)
    truth = [np.concatenate((part.Truth, np.zeros((len(part.Rate), K - part.Truth.shape[1], 6), dtype=np.float32)), axis=1) for part in parts]
    return Flights_Str(Time=np.concatenate([p.Time for p in parts]), Position=np.concatenate([p.Position for p in parts]),
                       Samples=np.concatenate([p.Samples for p in parts]), Rate=np.concatenate([p.Rate for p in parts]),
                       Release=np.concatenate([p.Release for p in parts]), Truth=np.concatenate(truth),
                       Floor_Index=np.concatenate([p.Floor_Index for p in parts]))


def Fixed_Prefixes(flights: Flights_Str, n_values: tp.Sequence[int]) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Every prefix of the configured values of N: each flight with each N it has the samples for (the set the
        normalisation is fitted on, and the set the validation loss is taken on).

    Args:
        (1) flights [Flights_Str(object)]: The flights.
        (2) n_values [Sequence<int>]: The values of N.

    Returns:
        (1) parameter 1 [Vector<int> 1xP]: The flight of each prefix.
        (2) parameter 2 [Vector<int> 1xP]: Its N.
    """

    rows = [np.where(flights.Samples >= n)[0] for n in n_values]
    return np.concatenate(rows), np.concatenate([np.full(len(r), n, dtype=np.int64) for r, n in zip(rows, n_values)])


def Drawn_Prefixes(flights: Flights_Str, n_values: tp.Sequence[int], rng: np.random.Generator) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The prefixes of one training epoch: each flight as often as there are configured
        values of N, each time with an N drawn uniformly from the smallest configured value to the largest (or to
        the flight's samples, when fewer), in a drawn order. A flight with fewer samples than the smallest N is
        left out.

    Args:
        (1) flights [Flights_Str(object)]: The training flights.
        (2) n_values [Sequence<int>]: The values of N.
        (3) rng [np.random.Generator]: The generator of the epoch.

    Returns:
        (1) parameter 1 [Vector<int> 1xP]: The flight of each prefix.
        (2) parameter 2 [Vector<int> 1xP]: Its N.
    """

    low, high = min(n_values), max(n_values)
    rows = np.tile(np.where(flights.Samples >= low)[0], len(n_values))
    n = rng.integers(low, np.minimum(high, flights.Samples[rows]) + 1)
    order = rng.permutation(len(rows))
    return rows[order], n[order].astype(np.int64)


def _Start(flights: Flights_Str, rows: np.ndarray, n: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The last observed time of each prefix and the first grid index at or after it.

    Args:
        (1) flights [Flights_Str(object)]: The flights.
        (2 - 3) rows, n [Vector<int> 1xP]: The flight and the N of each prefix.

    Returns:
        (1) parameter 1 [Vector<float> 1xP]: t_0 [s].
        (2) parameter 2 [Vector<int> 1xP]: The grid index.
    """

    t_0 = flights.Time[rows, n - 1]
    return t_0, np.ceil(t_0 / CONST_SIMULATION_STEP - 1e-9).astype(np.int64)


def _Loss_Times(flights: Flights_Str, rows: np.ndarray, n: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The grid times every CONST_LOSS_STEP from the first multiple at or after each prefix's t_0 to its floor
        (the first of them always, so every prefix has one): the times of the validation loss.

    Args:
        (1) flights [Flights_Str(object)]: The flights.
        (2 - 3) rows, n [Vector<int> 1xP]: The flight and the N of each prefix.

    Returns:
        (1) parameter 1 [Vector<float> 1xP]: t_0 [s].
        (2) parameter 2 [Matrix<int> PxT]: The grid indices of the times.
        (3) parameter 3 [Matrix<float> PxT]: The times after t_0 [s].
        (4) parameter 4 [Matrix<bool> PxT]: Which of them count.
    """

    h = CONST_SIMULATION_STEP
    stride = int(round(CONST_LOSS_STEP / h))
    t_0 = flights.Time[rows, n - 1]
    first = np.ceil(t_0 / CONST_LOSS_STEP - 1e-9).astype(np.int64)
    last = flights.Floor_Index[rows] // stride
    m = np.arange(int(max((last - first).max(), 0)) + 1)[None, :]
    index = np.minimum(stride * (first[:, None] + m), flights.Truth.shape[1] - 1)
    return t_0, index, index * h - t_0[:, None], ((first[:, None] + m) <= last[:, None]) | (m == 0)


def Fit_Normalisation(flights: Flights_Str, n_values: tp.Sequence[int], *, Flight_Model_Str: Flight_Model_Str) -> Normalisation_Str:
    """
    Description:
        The normalisation of the model (SPEC §7.8), from the training flights given and nothing else: over their
        prefixes of the configured N, the deviation of the samples relative to the last one, the mean and the
        deviation of the true velocity at t_0, and the scale of the outputs: the mean and the deviation of the
        least-squares basis coefficients of the true displacement over CONST_HORIZON, on the first
        CONST_COEFFICIENT_FLIGHTS flights.

    Args:
        (1) flights [Flights_Str(object)]: The training flights.
        (2) n_values [Sequence<int>]: The values of N.
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model (the truth over
                                                         the basis's span).

    Returns:
        (1) parameter [Normalisation_Str(object)]: The normalisation.
    """

    rows, n = Fixed_Prefixes(flights, n_values)
    if not len(rows):
        raise Training_Error_Cls('No training flight has %d samples; a predictor cannot be trained on them. Use a dataset with '
                                 'longer observations or a smaller N.' % min(n_values))
    last = flights.Position[rows, n - 1]
    present = np.arange(flights.Time.shape[1])[None, :] < n[:, None]
    relative = (flights.Position[rows] - last[:, None, :])[present]
    t_0, start = _Start(flights, rows, n)
    velocity = flights.Truth[rows, start, 3:].astype(np.float64)
    normalisation = Normalisation_Str(Relative_Std=relative.std(axis=0), Velocity_Mean=velocity.mean(axis=0), Velocity_Std=velocity.std(axis=0))
    h = CONST_SIMULATION_STEP
    keep = rows < CONST_COEFFICIENT_FLIGHTS
    span = int(round(CONST_HORIZON / h)) + 1
    used = min(CONST_COEFFICIENT_FLIGHTS, len(flights.Rate))
    truth = Simulate(flights.Release[:used], np.arange(int(start[keep].max()) + span) * h, Flight_Model_Str=Flight_Model_Str)[:, :, :3]
    coefficients = np.array([Chebyshev_Fit((first + np.arange(span)) * h - time, truth[row, first:first + span] - origin)
                             for row, first, time, origin in zip(rows[keep], start[keep], t_0[keep], last[keep])])
    normalisation.Output_Mean, normalisation.Output_Scale = coefficients.mean(axis=0), coefficients.std(axis=0)
    return normalisation


def _Batch(flights: Flights_Str, rows: np.ndarray, n: np.ndarray, normalisation: Normalisation_Str, device: torch.device
           ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Description:
        The inputs of a batch of prefixes as tensors.

    Args:
        (1) flights [Flights_Str(object)]: The flights.
        (2 - 3) rows, n [Vector<int> 1xB]: The flight and the N of each prefix.
        (4) normalisation [Normalisation_Str(object)]: The normalisation.
        (5) device [torch.device]: The device.

    Returns:
        (1) parameter 1 [Tensor<float> BxMx9]: The inputs per sample.
        (2) parameter 2 [Matrix<float> Bx1]: The inputs per prefix.
        (3) parameter 3 [Vector<int> 1xB]: N.
        (4) parameter 4 [Matrix<float> Bx3]: The last observed positions [m].
    """

    sequence, context, last = Prefix_Features(flights.Time[rows], flights.Position[rows], n, flights.Rate[rows], Normalisation_Str=normalisation)
    return (torch.tensor(sequence, dtype=torch.float32, device=device), torch.tensor(context, dtype=torch.float32, device=device),
            torch.tensor(n, device=device), last)


def _Errors(network: Network_Cls, flights: Flights_Str, rows: np.ndarray, n: np.ndarray, normalisation: Normalisation_Str,
            queries: tp.Optional[np.ndarray], velocity: bool, device: torch.device) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
    """
    Description:
        The RMS position error and the RMS velocity error of each prefix of a batch over the remainder of its
        flight. With `queries` (the training): at the drawn grid times, from t_0 to CONST_MARGIN past the floor.
        Without: at the times of _Loss_Times (every validation loss).

    Args:
        (1) network [Network_Cls(object)]: The network.
        (2) flights [Flights_Str(object)]: The flights.
        (3 - 4) rows, n [Vector<int> 1xB]: The flight and the N of each prefix.
        (5) normalisation [Normalisation_Str(object)]: The normalisation.
        (6) queries [Matrix<float> BxQ]: Uniform numbers in [0, 1), one per query, or None.
        (7) velocity [bool]: Whether the velocity error is wanted.
        (8) device [torch.device]: The device.

    Returns:
        (1) parameter 1 [Vector<float> 1xB]: The RMS position errors [m].
        (2) parameter 2 [Vector<float> 1xB]: The RMS velocity errors [m/s], or None.
    """

    sequence, context, count, last = _Batch(flights, rows, n, normalisation, device)
    encoding = network.Encode(sequence, context, count)
    mean = torch.tensor(normalisation.Output_Mean, dtype=torch.float32, device=device)
    scale = torch.tensor(normalisation.Output_Scale, dtype=torch.float32, device=device)
    if queries is not None:
        t_0, start = _Start(flights, rows, n)
        span = np.maximum(flights.Floor_Index[rows] - start, 0) + int(round(CONST_MARGIN / CONST_SIMULATION_STEP)) + 1
        index = np.minimum(start[:, None] + (queries * span[:, None]).astype(np.int64), flights.Truth.shape[1] - 1)
        tau, counts = index * CONST_SIMULATION_STEP - t_0[:, None], np.ones(index.shape, dtype=bool)
    else:
        t_0, index, tau, counts = _Loss_Times(flights, rows, n)
    truth = flights.Truth[rows[:, None], index].astype(np.float64)
    truth[:, :, :3] -= last[:, None, :]
    truth = torch.tensor(truth, dtype=torch.float32, device=device)
    weight = torch.tensor(counts, dtype=torch.float32, device=device)
    tau = torch.tensor(tau, dtype=torch.float32, device=device)
    position, speed = Query(network, encoding, tau, mean, scale, velocity)
    total = weight.sum(dim=1)
    position_error = torch.sqrt((((position - truth[:, :, :3]) ** 2).sum(dim=2) * weight).sum(dim=1) / total + CONST_EPSILON)
    if not velocity:
        return position_error, None
    return position_error, torch.sqrt((((speed - truth[:, :, 3:]) ** 2).sum(dim=2) * weight).sum(dim=1) / total + CONST_EPSILON)


def Validation_Loss(network: Network_Cls, flights: Flights_Str, n_values: tp.Sequence[int], normalisation: Normalisation_Str, weight: float,
                    device: tp.Optional[torch.device] = None) -> tp.Tuple[float, float]:
    """
    Description:
        The loss on flights not trained on (the simulated validation partition: early stopping, SPEC §2.13.7):
        over every prefix of the configured N, the mean RMS position error at the times of _Loss_Times, plus
        `weight` times the mean RMS velocity error there.

    Args:
        (1) network [Network_Cls(object)]: The network.
        (2) flights [Flights_Str(object)]: The flights.
        (3) n_values [Sequence<int>]: The values of N.
        (4) normalisation [Normalisation_Str(object)]: The normalisation.
        (5) weight [float]: The weight of the velocity term [s].
        (6) device [torch.device]: The device; None: the CPU.

    Returns:
        (1) parameter 1 [float]: The loss [m].
        (2) parameter 2 [float]: Its position term alone [m].
    """

    device = torch.device('cpu') if device is None else device
    rows, n = Fixed_Prefixes(flights, n_values)
    if not len(rows):
        raise Training_Error_Cls('No validation flight has %d samples; the training cannot be stopped on them. Use a dataset with '
                                 'longer observations or a smaller N.' % min(n_values))
    position_sum, velocity_sum = 0.0, 0.0
    with torch.no_grad():
        for a in range(0, len(rows), 1024):
            position, velocity = _Errors(network, flights, rows[a:a + 1024], n[a:a + 1024], normalisation, None, weight > 0.0, device)
            position_sum += float(position.sum())
            velocity_sum += float(velocity.sum()) if velocity is not None else 0.0
    return (position_sum + weight * velocity_sum) / len(rows), position_sum / len(rows)


def Train(configuration: Training_Configuration_Str, train: Flights_Str, validation: Flights_Str, normalisation: Normalisation_Str, seed: int,
          progress: tp.Optional[tp.Callable[[int, float, float], None]] = None
          ) -> tp.Tuple[tp.Dict[str, np.ndarray], tp.List[tp.Tuple[int, float, float, float]]]:
    """
    Description:
        One model of one seed (SPEC §2.13.7, §7.8). The seed fixes the initial weights and, per epoch, the
        prefixes, their order and the query times; it never changes the data. Adam with a one-cycle learning
        rate over Optimiser.Max_Epochs; after each epoch the validation loss; the weights of the epoch with the
        lowest validation loss are returned, and the training stops after Early_Stopping.Patience epochs without
        an improvement of at least Min_Delta. On the CPU: CONST_THREADS thread(s) and deterministic algorithms,
        so the same call gives the same weights bit for bit.

    Args:
        (1) configuration [Training_Configuration_Str(object)]: The training configuration (a learned predictor's).
        (2) train [Flights_Str(object)]: The training flights.
        (3) validation [Flights_Str(object)]: The validation flights (early stopping only).
        (4) normalisation [Normalisation_Str(object)]: The normalisation (Fit_Normalisation of the training flights).
        (5) seed [int]: The seed.
        (6) progress [Callable[[int, float, float], None]]: Called after each epoch with (epoch, train loss,
                                                            validation loss) [m], or None.

    Returns:
        (1) parameter 1 [Dict<string, Matrix<float>>]: The weights of the best epoch (Network_Cls.Weights).
        (2) parameter 2 [List<Tuple<int, float, float, float>>]: Per epoch: its number, the train loss, the
                                                                 validation loss and its position term [m].
    """

    c = configuration
    if c.Device == 'mps' and not torch.backends.mps.is_available():
        raise Training_Error_Cls('Run.Device is mps, and this machine has no MPS device. Use cpu (SPEC §8.3).')
    device = torch.device(c.Device)
    if c.Device == 'cpu':
        torch.set_num_threads(CONST_THREADS)
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    network = Network_Cls(c.Model_Name, c.Hidden).to(device)
    steps = int(np.ceil(len(Drawn_Prefixes(train, c.Observed_Samples, np.random.default_rng(0))[0]) / c.Batch_Size))
    if steps == 0:
        raise Training_Error_Cls('No training flight has %d samples; a predictor cannot be trained on them. Use a dataset with '
                                 'longer observations or a smaller N.' % min(c.Observed_Samples))
    optimiser = torch.optim.Adam(network.parameters(), lr=c.Learning_Rate, weight_decay=c.Weight_Decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=c.Learning_Rate, total_steps=max(c.Max_Epochs * steps, 2),
                                                   pct_start=CONST_WARM_UP)
    lowest, reference, waited, best, epochs = float('inf'), float('inf'), 0, network.Weights(), []
    for epoch in range(1, c.Max_Epochs + 1):
        rng = np.random.default_rng([seed, epoch])
        rows, n = Drawn_Prefixes(train, c.Observed_Samples, rng)
        network.train()
        total = 0.0
        for a in range(0, len(rows), c.Batch_Size):
            batch_rows, batch_n = rows[a:a + c.Batch_Size], n[a:a + c.Batch_Size]
            queries = rng.random((len(batch_rows), CONST_QUERIES))
            optimiser.zero_grad()
            position, velocity = _Errors(network, train, batch_rows, batch_n, normalisation, queries, c.Velocity_Weight > 0.0, device)
            loss = position.mean() if velocity is None else position.mean() + c.Velocity_Weight * velocity.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), CONST_GRADIENT_NORM)
            optimiser.step()
            schedule.step()
            total += float(loss.detach()) * len(batch_rows)
        network.eval()
        value, position_term = Validation_Loss(network, validation, c.Observed_Samples, normalisation, c.Velocity_Weight, device=device)
        epochs.append((epoch, total / len(rows), value, position_term))
        if progress is not None:
            progress(epoch, total / len(rows), value)
        if value < lowest:
            lowest, best = value, network.Weights()
        if value < reference - c.Min_Delta:
            reference, waited = value, 0
        else:
            waited += 1
            if waited >= c.Patience:
                break
    return best, epochs


def Write_Epochs(path: str, epochs: tp.List[tp.Tuple[int, float, float, float]]) -> None:
    """
    Description:
        epochs.csv (SPEC §9.6): one row per epoch, the train loss, the validation loss and its position term
        [m]; comma-separated, a header line, LF, '%.9f'.

    Args:
        (1) path [string]: The file.
        (2) epochs [List<Tuple<int, float, float, float>>]: The rows (Train).
    """

    with open(path, 'w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_EPOCH_COLUMNS)
        writer.writerows(['%d' % epoch] + [CONST_EPOCH_FORMAT % value for value in values] for epoch, *values in epochs)
