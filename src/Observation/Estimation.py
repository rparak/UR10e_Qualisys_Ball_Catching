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
# File Name: ../Observation/Estimation.py
## =========================================================================== ##

# Itertools (Functions creating iterators; the subsets of the non-negative least squares)
import itertools
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Observation/Core - the model's structure
from Observation.Core import Observation_Model_Str

"""
Description:
    Initialization of constants.

    The estimation of the error model M4 and the statistics it is checked by (SPEC §7.5). The input is the residual of
    flights fitted under the flight model (recorded
    minus simulated positions); this module fits nothing itself and reads no file. The estimator is
    deterministic: the variances by non-negative least squares on the halved squared differences of
    pairs of samples, the time constants on fixed grids, the degrees of freedom by a tail ratio matched
    with fixed seeds.
"""
# The longest lag of a pair of samples inside a run used by the estimator.
#   Unit [float, s]
CONST_MAX_LAG = 0.100
# The grids of the drift's and the offset's time constants (F6 Part A: the residual is correlated over
# 50-80 ms; the jump across a dropout grows with the lost time and saturates).
#   Unit [Tuple<float>, s]
CONST_DRIFT_TIMES = (0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120, 0.150, 0.200, 0.300)
CONST_OFFSET_TIMES = (0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.100, 0.150, 0.200, 0.300, 0.500)
# The grid of the degrees of freedom of the white noise and of the offset's innovations (> 2: the
# variance exists).
#   Unit [Tuple<float>, -]
CONST_DOFS = (2.1, 2.2, 2.3, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 30.0)
# The pairs whose tails set the degrees of freedom: inside a run at a lag of at most this (the white
# noise dominates), and across a dropout with at least this lost time (the offset dominates).
#   Unit [float, s]
CONST_WHITE_TAIL_LAG = 0.0101
CONST_OFFSET_TAIL_LOST = 0.050
# The seed and the sample count of the tail-ratio matching.
#   Unit [int]
CONST_ESTIMATION_SEED = 20261111
CONST_TAIL_SAMPLES = 200000
# The lags of the within-run statistics, and the bins of missing frames of the jump statistics (F6 Part A).
#   Unit [Tuple<float>, s], [Tuple<Tuple<int, int>>]
CONST_LAGS = (0.005, 0.010, 0.020, 0.040, 0.080)
CONST_JUMP_BINS = ((1, 1), (2, 4), (5, 19), (20, 10 ** 9))
# The run lengths [samples] of the offset diagnostics.
#   Unit [Tuple<Tuple<int, int>>]
CONST_RUN_BINS = ((1, 1), (2, 4), (5, 14), (15, 10 ** 9))
# The shuffles of the run lengths per flight of the shuffled offset share (F3 Part A used 50 for 72 %).
#   Unit [int]
CONST_SHUFFLES = 20
# The 19 statistics of the error model's check, and 6 further diagnostics.
#   Unit [Tuple<string>]
CONST_ERROR_STATISTICS = ('rms_median', 'rms_p90', 'max_median', 'offset_share', 'shuffled_offset_share', 'within_5ms', 'within_10ms',
                          'within_20ms', 'within_40ms', 'within_80ms', 'jump_1', 'jump_2_4', 'jump_5_19', 'jump_20', 'edge_rms',
                          'interior_rms', 'within_5ms_mean', 'jump_20_mean', 'jump_5_19_mean')
CONST_ERROR_DIAGNOSTICS = ('offset_run_1', 'offset_run_2_4', 'offset_run_5_14', 'offset_run_15', 'kurtosis_d2_y', 'share_above_20mm')
# The phase bins of the presence profile, from the release.
#   Unit [float, s], [int]
CONST_PHASE_BIN = 0.1
CONST_PHASE_BINS = 12
# The statistics of a dropout pattern (SPEC §2.13.5): the median coverage, the missing and present run
# lengths, the median lag-1 autocorrelation of presence, the share of flights with a dropout longer than
# CONST_LONG_DROPOUT, and the presence per phase bin.
#   Unit [Tuple<string>], [float, s]
CONST_PATTERN_STATISTICS = ('coverage_median', 'missing_p50', 'missing_p90', 'missing_p99', 'present_p50', 'present_p90', 'lag1_median',
                            'long_share') + tuple('phase_%02d' % i for i in range(CONST_PHASE_BINS))
CONST_LONG_DROPOUT = 0.375
# The largest values of a set of residuals from which the error model's bounds come.
#   Unit [Tuple<string>]
CONST_EXTREMES = ('residual_max', 'run_mean_max', 'jump_max', 'step_max')
# The bootstrap: resamples over flights and the level of the bounds.
#   Unit [int], [float, -]
CONST_BOOTSTRAP_RESAMPLES = 1000
CONST_BOUND_LEVEL = 0.95


@dataclass
class Residual_Str:
    """
    Description:
        One flight's residual under the flight model: its frames from 0, its rate and recorded minus
        simulated positions.
    """

    # The frame index of each sample, increasing.
    #   Unit [Vector<int> 1xM]
    Frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The rate.
    #   Unit [float, Hz]
    Rate: float = 0.0
    # The residual per sample.
    #   Unit [Matrix<float> Mx3, m]
    Residual: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def Runs(frame: np.ndarray) -> tp.List[tp.Tuple[int, int]]:
    """
    Description:
        The present runs of a flight: consecutive frames with a sample.

    Args:
        (1) frame [Vector<int> 1xM]: The frames with a sample, increasing.

    Returns:
        (1) parameter [List<Tuple<int, int>>]: (first, last) sample index of each run, inclusive.
    """

    breaks = np.where(np.diff(frame) > 1)[0]
    return list(zip([0] + list(breaks + 1), list(breaks) + [len(frame) - 1])) if len(frame) else []


def _Pairs(flights: tp.List[Residual_Str]) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The pairs of the estimator: every pair inside a run at a lag of at most CONST_MAX_LAG, and every
        adjacent pair across a dropout; per pair the lag, the lost time and half the squared difference
        of the residual per axis. The pairs are returned in a canonical order (within: by lag, then the
        values; across: by lost time, lag, then the values), so that nothing after this depends on the
        order of the flights.

    Args:
        (1) flights [List<Residual_Str>]: The flights.

    Returns:
        (1) parameter 1 [Vector<float>]: The lag of each within-run pair [s].
        (2) parameter 2 [Matrix<float> Px3]: Half the squared difference of each within-run pair [m^2].
        (3) parameter 3 [Vector<float>]: The lag of each pair across a dropout [s].
        (4) parameter 4 [Vector<float>]: The lost time of each pair across a dropout [s].
        (5) parameter 5 [Matrix<float> Qx3]: Half the squared difference of each pair across a dropout [m^2].
    """

    w_tau, w_y, a_tau, a_lost, a_y = [], [], [], [], []
    for flight in flights:
        t, r = flight.Frame / flight.Rate, flight.Residual
        runs = Runs(flight.Frame)
        for first, last in runs:
            for i in range(first, last):
                j = np.arange(i + 1, last + 1)
                j = j[t[j] - t[i] <= CONST_MAX_LAG + 1e-9]
                w_tau.append(t[j] - t[i])
                w_y.append(0.5 * (r[j] - r[i]) ** 2)
        for (_, last), (first, _) in zip(runs[:-1], runs[1:]):
            a_tau.append(t[first] - t[last])
            a_lost.append(t[first] - t[last] - 1.0 / flight.Rate)
            a_y.append(0.5 * (r[first] - r[last]) ** 2)
    w_tau, w_y = np.concatenate(w_tau), np.concatenate(w_y)
    a_tau, a_lost, a_y = np.array(a_tau), np.array(a_lost), np.array(a_y).reshape(-1, 3)
    w = np.lexsort((w_y[:, 2], w_y[:, 1], w_y[:, 0], w_tau))
    a = np.lexsort((a_y[:, 2], a_y[:, 1], a_y[:, 0], a_tau, a_lost))
    return w_tau[w], w_y[w], a_tau[a], a_lost[a], a_y[a]


def _Nonnegative_Least_Squares(G: np.ndarray, c: np.ndarray) -> tp.Tuple[np.ndarray, float]:
    """
    Description:
        The non-negative least squares of three coefficients from the normal equations, by trying every
        subset of free coefficients (exact for three).

    Args:
        (1) G [Matrix<float> 3x3]: X^T X.
        (2) c [Vector<float> 1x3]: X^T y.

    Returns:
        (1) parameter 1 [Vector<float> 1x3]: The coefficients, each >= 0.
        (2) parameter 2 [float]: The sum of squares less y^T y (to compare candidates).
    """

    best, best_value = np.zeros(3), 0.0
    for size in (1, 2, 3):
        for subset in itertools.combinations(range(3), size):
            s = list(subset)
            try:
                beta = np.linalg.solve(G[np.ix_(s, s)], c[s])
            except np.linalg.LinAlgError:
                continue
            if np.all(beta >= 0.0):
                value = float(beta @ G[np.ix_(s, s)] @ beta - 2.0 * beta @ c[s])
                if value < best_value:
                    best, best_value = np.zeros(3), value
                    best[s] = beta
    return best, best_value


def _Tail_Ratio(y: np.ndarray) -> float:
    """
    Description:
        The tail ratio sqrt(mean / median) of halved squared differences: 1.21 for a Gaussian difference,
        larger for heavier tails.

    Args:
        (1) y [Vector<float>]: The values [m^2].

    Returns:
        (1) parameter [float]: The ratio [-].
    """

    return float(np.sqrt(np.mean(y) / np.median(y)))


def _Student_T(rng: np.random.Generator, dof: float, size: int) -> np.ndarray:
    """
    Description:
        Student-t values scaled to unit variance.

    Args:
        (1) rng [np.random.Generator]: The generator.
        (2) dof [float]: The degrees of freedom, > 2.
        (3) size [int]: The count.

    Returns:
        (1) parameter [Vector<float>]: The values [-].
    """

    return rng.standard_t(dof, size=size) * np.sqrt((dof - 2.0) / dof)


def Estimate_Observation_Model(flights: tp.List[Residual_Str]) -> Observation_Model_Str:
    """
    Description:
        The parameters of M4 from residuals. Inside a run, half the squared
        difference of a pair at lag tau has the mean sw^2 + sc^2 (1 - exp(-tau / Drift_Time)); across a
        dropout of lost time L it adds so^2 (1 - exp(-L / Offset_Time)). For every pair of time constants
        of the grids the three variances per axis are the non-negative least squares on all pairs; the
        pair with the smallest sum of squares over the axes is kept (ties: the first of the grids). The
        degrees of freedom are the grid values whose sampled tail ratio, averaged over the axes, is
        nearest the measured one: the white noise from the pairs at a lag of at most
        CONST_WHITE_TAIL_LAG, then the offset from the pairs with a lost time of at least
        CONST_OFFSET_TAIL_LOST.

    Args:
        (1) flights [List<Residual_Str>]: The fitted flights' residuals.

    Returns:
        (1) parameter [Observation_Model_Str(object)]: The estimated model.
    """

    w_tau, w_y, a_tau, a_lost, a_y = _Pairs(flights)
    y = np.vstack((w_y, a_y))
    best = None
    for drift_time in CONST_DRIFT_TIMES:
        drift = np.r_[1.0 - np.exp(-w_tau / drift_time), 1.0 - np.exp(-a_tau / drift_time)]
        for offset_time in CONST_OFFSET_TIMES:
            X = np.c_[np.ones(len(drift)), drift, np.r_[np.zeros(len(w_tau)), 1.0 - np.exp(-a_lost / offset_time)]]
            G = X.T @ X
            fits = [_Nonnegative_Least_Squares(G, X.T @ y[:, axis]) for axis in range(3)]
            total = sum(value for _, value in fits)
            if best is None or total < best[0] - 1e-30:
                best = (total, drift_time, offset_time, np.array([beta for beta, _ in fits]))
    _, drift_time, offset_time, beta = best
    white_std, drift_std, offset_std = np.sqrt(beta[:, 0]), np.sqrt(beta[:, 1]), np.sqrt(beta[:, 2])

    short = w_tau <= CONST_WHITE_TAIL_LAG
    target = np.mean([_Tail_Ratio(w_y[short, axis]) for axis in range(3)])

    def White_Ratio(index: int, dof: float) -> float:
        """
        Description:
            The sampled tail ratio of the short within-run pairs for a white-noise dof.

        Args:
            (1) index [int]: The dof's index in CONST_DOFS (the seed of its stream).
            (2) dof [float]: The degrees of freedom.

        Returns:
            (1) parameter [float]: The ratio averaged over the axes [-].
        """

        ratios = []
        for axis in range(3):
            rng = np.random.default_rng([CONST_ESTIMATION_SEED, 0, index, axis])
            tau = w_tau[short][rng.integers(0, short.sum(), CONST_TAIL_SAMPLES)]
            d = (white_std[axis] * (_Student_T(rng, dof, CONST_TAIL_SAMPLES) - _Student_T(rng, dof, CONST_TAIL_SAMPLES))
                 + drift_std[axis] * np.sqrt(2.0 * (1.0 - np.exp(-tau / drift_time))) * rng.standard_normal(CONST_TAIL_SAMPLES))
            ratios.append(_Tail_Ratio(0.5 * d ** 2))
        return float(np.mean(ratios))

    white_dof = min(enumerate(CONST_DOFS), key=lambda item: (abs(White_Ratio(*item) - target), item[0]))[1]

    long = a_lost >= CONST_OFFSET_TAIL_LOST
    target = np.mean([_Tail_Ratio(a_y[long, axis]) for axis in range(3)])

    def Offset_Ratio(index: int, dof: float) -> float:
        """
        Description:
            The sampled tail ratio of the long across-dropout pairs for an offset dof.

        Args:
            (1) index [int]: The dof's index in CONST_DOFS (the seed of its stream).
            (2) dof [float]: The degrees of freedom.

        Returns:
            (1) parameter [float]: The ratio averaged over the axes [-].
        """

        ratios = []
        for axis in range(3):
            rng = np.random.default_rng([CONST_ESTIMATION_SEED, 1, index, axis])
            k = rng.integers(0, long.sum(), CONST_TAIL_SAMPLES)
            tau, lost = a_tau[long][k], a_lost[long][k]
            d = (white_std[axis] * (_Student_T(rng, white_dof, CONST_TAIL_SAMPLES) - _Student_T(rng, white_dof, CONST_TAIL_SAMPLES))
                 + drift_std[axis] * np.sqrt(2.0 * (1.0 - np.exp(-tau / drift_time))) * rng.standard_normal(CONST_TAIL_SAMPLES)
                 + offset_std[axis] * np.sqrt(2.0 * (1.0 - np.exp(-lost / offset_time))) * _Student_T(rng, dof, CONST_TAIL_SAMPLES))
            ratios.append(_Tail_Ratio(0.5 * d ** 2))
        return float(np.mean(ratios))

    offset_dof = min(enumerate(CONST_DOFS), key=lambda item: (abs(Offset_Ratio(*item) - target), item[0]))[1]
    return Observation_Model_Str(White_Std=white_std, White_Dof=float(white_dof), Drift_Std=drift_std, Drift_Time=float(drift_time),
                                 Offset_Std=offset_std, Offset_Time=float(offset_time), Offset_Dof=float(offset_dof))


def Error_Parts(flights: tp.List[Residual_Str], seed: int = CONST_ESTIMATION_SEED) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The per-flight parts of the error statistics, so that a bootstrap over flights only aggregates.
        The shuffled sums of squares use the run lengths of the flight in CONST_SHUFFLES random orders,
        flight i from the generator [seed, i].

    Args:
        (1) flights [List<Residual_Str>]: The fitted flights' residuals.
        (2) seed [int]: The seed of the shuffles.

    Returns:
        (1) parameter [List<Dict<string, Any>>]: One dictionary of arrays and sums per flight.
    """

    parts = []
    for index, flight in enumerate(flights):
        f, r, rate = flight.Frame, flight.Residual, flight.Rate
        norm2 = np.sum(r ** 2, axis=1)
        runs = Runs(f)
        part: tp.Dict[str, tp.Any] = {'rms': float(np.sqrt(np.mean(norm2))), 'max': float(np.sqrt(np.max(norm2))), 'sse': float(np.sum(norm2))}
        part['sse_offset'] = float(sum(np.sum((r[a:b + 1] - r[a:b + 1].mean(axis=0)) ** 2) for a, b in runs))
        rng = np.random.default_rng([seed, index])
        lengths = [b - a + 1 for a, b in runs]
        shuffled = 0.0
        for _ in range(CONST_SHUFFLES):
            order = [lengths[i] for i in rng.permutation(len(lengths))]
            start = 0
            for n in order:
                segment = r[start:start + n]
                shuffled += float(np.sum((segment - segment.mean(axis=0)) ** 2))
                start += n
        part['sse_shuffled'] = shuffled / CONST_SHUFFLES
        for lag in CONST_LAGS:
            L = lag * rate
            values = []
            if abs(L - round(L)) <= 0.02 and round(L) >= 1:
                L = int(round(L))
                for a, b in runs:
                    if b - a >= L:
                        values.append(np.sum((r[a + L:b + 1] - r[a:b + 1 - L]) ** 2, axis=1))
            part['within_%g' % lag] = np.concatenate(values) if values else np.zeros(0)
        gaps = [(f[first] - f[last] - 1, float(np.sum((r[first] - r[last]) ** 2))) for (_, last), (first, _) in zip(runs[:-1], runs[1:])]
        for low, high in CONST_JUMP_BINS:
            part['jump_%d' % low] = np.array([d2 for g, d2 in gaps if low <= g <= high])
        edge = np.zeros(len(f), dtype=bool)
        step = np.diff(f) > 1
        edge[:-1] |= step
        edge[1:] |= step
        part['edge_sum'], part['edge_count'] = float(np.sum(norm2[edge])), int(edge.sum())
        part['interior_sum'], part['interior_count'] = float(np.sum(norm2[~edge])), int((~edge).sum())
        for low, high in CONST_RUN_BINS:
            part['run_%d' % low] = np.array([np.linalg.norm(r[a:b + 1].mean(axis=0)) for a, b in runs if low <= b - a + 1 <= high])
        triple = (f[2:] - f[1:-1] == 1) & (f[1:-1] - f[:-2] == 1)
        part['d2_y'] = (r[2:, 1] - 2.0 * r[1:-1, 1] + r[:-2, 1])[triple]
        part['above'], part['count'] = int(np.sum(norm2 > 0.020 ** 2)), len(f)
        parts.append(part)
    return parts


def _Pooled(parts: tp.List[tp.Dict[str, tp.Any]], key: str) -> np.ndarray:
    """
    Description:
        One array of the parts, concatenated over the flights.

    Args:
        (1) parts [List<Dict<string, Any>>]: The per-flight parts.
        (2) key [string]: The part's key.

    Returns:
        (1) parameter [Vector<float>]: The pooled values.
    """

    return np.concatenate([part[key] for part in parts])


def Aggregate_Error(parts: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, float]:
    """
    Description:
        The error statistics of a set of flights from their parts (CONST_ERROR_STATISTICS and
        CONST_ERROR_DIAGNOSTICS). A within or jump statistic is sqrt(median / 2) (the '_mean' ones
        sqrt(mean / 2)) of the pooled squared 3D differences: inside runs at a lag of CONST_LAGS, at the
        rates where the lag is whole frames; across a dropout of the missing frames of CONST_JUMP_BINS.
        The offset shares are 1 - (sum of squares about each run's mean) / (sum of squares), in the real
        order and with the run lengths shuffled. The run diagnostics are the median norm of a run's mean
        residual by run length; the kurtosis is the excess kurtosis of the y second differences over
        consecutive triples. Lengths in [m].

    Args:
        (1) parts [List<Dict<string, Any>>]: The per-flight parts (Error_Parts).

    Returns:
        (1) parameter [Dict<string, float>]: Each statistic (NaN without data).
    """

    def Half_Root(values: np.ndarray, mean: bool = False) -> float:
        """
        Description:
            sqrt(median / 2), or sqrt(mean / 2), of squared differences.

        Args:
            (1) values [Vector<float>]: The squared differences [m^2].
            (2) mean [bool]: The mean instead of the median.

        Returns:
            (1) parameter [float]: [m], NaN without values.
        """

        if values.size == 0:
            return float('nan')
        return float(np.sqrt(0.5 * (np.mean(values) if mean else np.median(values))))

    rms = np.array([part['rms'] for part in parts])
    sse = sum(part['sse'] for part in parts)
    out = {'rms_median': float(np.median(rms)), 'rms_p90': float(np.percentile(rms, 90)),
           'max_median': float(np.median([part['max'] for part in parts])),
           'offset_share': 1.0 - sum(part['sse_offset'] for part in parts) / sse,
           'shuffled_offset_share': 1.0 - sum(part['sse_shuffled'] for part in parts) / sse}
    for lag in CONST_LAGS:
        out['within_%dms' % round(lag * 1000)] = Half_Root(_Pooled(parts, 'within_%g' % lag))
    for low, high in CONST_JUMP_BINS:
        name = 'jump_%d' % low + ('' if low == high or high > 10 ** 8 else '_%d' % high)
        out[name] = Half_Root(_Pooled(parts, 'jump_%d' % low))
    out['edge_rms'] = float(np.sqrt(sum(p['edge_sum'] for p in parts) / max(sum(p['edge_count'] for p in parts), 1)))
    out['interior_rms'] = float(np.sqrt(sum(p['interior_sum'] for p in parts) / max(sum(p['interior_count'] for p in parts), 1)))
    out['within_5ms_mean'] = Half_Root(_Pooled(parts, 'within_0.005'), mean=True)
    out['jump_20_mean'] = Half_Root(_Pooled(parts, 'jump_20'), mean=True)
    out['jump_5_19_mean'] = Half_Root(_Pooled(parts, 'jump_5'), mean=True)
    for low, high in CONST_RUN_BINS:
        values = _Pooled(parts, 'run_%d' % low)
        name = 'offset_run_%d' % low + ('' if low == high or high > 10 ** 8 else '_%d' % high)
        out[name] = float(np.median(values)) if values.size else float('nan')
    d2 = _Pooled(parts, 'd2_y')
    d2 = d2 - d2.mean()
    out['kurtosis_d2_y'] = float(np.mean(d2 ** 4) / np.mean(d2 ** 2) ** 2 - 3.0) if d2.size and np.any(d2 != 0.0) else float('nan')
    out['share_above_20mm'] = sum(p['above'] for p in parts) / max(sum(p['count'] for p in parts), 1)
    return out


def Residual_Extremes(flights: tp.List[Residual_Str]) -> tp.Dict[str, float]:
    """
    Description:
        The largest values of the residuals of a set of flights (CONST_EXTREMES), each a 3D norm: the largest
        residual; the largest mean residual of a present run (the offset as it can be measured); the largest
        jump between the last sample of a run and the first of the next (across a dropout); the largest step
        between adjacent samples inside a run (a single sample's noise shows there). The error model's bounds
        are derived from the first and the last (Dataset.Calibration.Derived_Bound).

    Args:
        (1) flights [List<Residual_Str>]: The fitted flights' residuals.

    Returns:
        (1) parameter [Dict<string, float>]: Each value [m]; 0 where no sample, run pair or step exists.
    """

    out = {key: 0.0 for key in CONST_EXTREMES}
    for flight in flights:
        r = flight.Residual
        runs = Runs(flight.Frame)
        if not runs:
            continue
        out['residual_max'] = max(out['residual_max'], float(np.linalg.norm(r, axis=1).max()))
        for a, b in runs:
            out['run_mean_max'] = max(out['run_mean_max'], float(np.linalg.norm(r[a:b + 1].mean(axis=0))))
            if b > a:
                out['step_max'] = max(out['step_max'], float(np.linalg.norm(np.diff(r[a:b + 1], axis=0), axis=1).max()))
        for (_, last), (first, _) in zip(runs[:-1], runs[1:]):
            out['jump_max'] = max(out['jump_max'], float(np.linalg.norm(r[first] - r[last])))
    return out


def Pattern_Parts(frames: tp.List[np.ndarray], rates: tp.List[float]) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The per-flight parts of the pattern statistics: coverage, the missing and present runs [s], the
        lag-1 autocorrelation of presence, the longest dropout [s], and the present and elapsed frames per
        phase bin.

    Args:
        (1) frames [List<Vector<int>>]: Per flight, the frames with a sample, from 0 (the release) to the last.
        (2) rates [List<float>]: Per flight, its rate [Hz].

    Returns:
        (1) parameter [List<Dict<string, Any>>]: One dictionary per flight.
    """

    parts = []
    for frame, rate in zip(frames, rates):
        n = int(frame[-1]) + 1
        present = np.zeros(n, dtype=bool)
        present[frame] = True
        step = np.diff(frame)
        missing = (step[step > 1] - 1) / rate
        runs = Runs(frame)
        x = present - present.mean()
        variance = float(np.sum(x * x))
        phase = np.minimum((np.arange(n) / rate / CONST_PHASE_BIN + 1e-9).astype(np.int64), CONST_PHASE_BINS)
        parts.append({'coverage': float(present.mean()), 'missing': missing, 'present': np.array([(b - a + 1) / rate for a, b in runs]),
                      'lag1': float(np.sum(x[1:] * x[:-1]) / variance) if variance > 0 else float('nan'),
                      'longest': float(missing.max()) if missing.size else 0.0,
                      'phase_present': np.bincount(phase, weights=present, minlength=CONST_PHASE_BINS + 1)[:CONST_PHASE_BINS],
                      'phase_frames': np.bincount(phase, minlength=CONST_PHASE_BINS + 1)[:CONST_PHASE_BINS].astype(float)})
    return parts


def Aggregate_Pattern(parts: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, float]:
    """
    Description:
        The pattern statistics of a set of flights (CONST_PATTERN_STATISTICS): the median coverage; the
        p50, p90, p99 of the pooled missing runs and the p50, p90 of the pooled present runs [s]; the
        median lag-1 autocorrelation; the share of flights whose longest dropout exceeds
        CONST_LONG_DROPOUT; the presence per phase bin of CONST_PHASE_BIN from the release.

    Args:
        (1) parts [List<Dict<string, Any>>]: The per-flight parts (Pattern_Parts).

    Returns:
        (1) parameter [Dict<string, float>]: Each statistic (NaN without data).
    """

    missing, present = _Pooled(parts, 'missing'), _Pooled(parts, 'present')
    out = {'coverage_median': float(np.median([p['coverage'] for p in parts]))}
    for q in (50, 90, 99):
        out['missing_p%d' % q] = float(np.percentile(missing, q)) if missing.size else float('nan')
    for q in (50, 90):
        out['present_p%d' % q] = float(np.percentile(present, q))
    lag1 = np.array([p['lag1'] for p in parts])
    out['lag1_median'] = float(np.median(lag1[np.isfinite(lag1)])) if np.isfinite(lag1).any() else float('nan')
    out['long_share'] = float(np.mean([p['longest'] > CONST_LONG_DROPOUT for p in parts]))
    phase_present = np.sum([p['phase_present'] for p in parts], axis=0)
    phase_frames = np.sum([p['phase_frames'] for p in parts], axis=0)
    for i in range(CONST_PHASE_BINS):
        out['phase_%02d' % i] = float(phase_present[i] / phase_frames[i]) if phase_frames[i] > 0 else float('nan')
    return out


def Bootstrap_Bounds(parts: tp.List[tp.Any], aggregate: tp.Callable[[tp.List[tp.Any]], tp.Dict[str, float]], seed: int,
                     resamples: int = CONST_BOOTSTRAP_RESAMPLES) -> tp.Dict[str, tp.Tuple[float, float]]:
    """
    Description:
        The bootstrap bounds of statistics over flights (SPEC §2.13.5): the flights resampled with
        replacement `resamples` times from the generator `seed`, each statistic's central
        CONST_BOUND_LEVEL interval of the resampled values (NaN resamples ignored). The bound is a
        measurement of the identification flights' own spread, never a tolerance chosen to pass.

    Args:
        (1) parts [List<Any>]: The per-flight parts.
        (2) aggregate [Callable]: Parts -> statistics (Aggregate_Error, Aggregate_Pattern, ...).
        (3) seed [int]: The seed of the resampling.
        (4) resamples [int]: The number of resamples.

    Returns:
        (1) parameter [Dict<string, Tuple<float, float>>]: Per statistic, (low, high).
    """

    rng = np.random.default_rng(seed)
    n = len(parts)
    values: tp.Dict[str, tp.List[float]] = {}
    for _ in range(resamples):
        sample = aggregate([parts[i] for i in rng.integers(0, n, n)])
        for key, value in sample.items():
            values.setdefault(key, []).append(value)
    tail = 100.0 * (1.0 - CONST_BOUND_LEVEL) / 2.0
    bounds = {}
    for key, value in values.items():
        v = np.array(value)
        v = v[np.isfinite(v)]
        bounds[key] = (float(np.percentile(v, tail)), float(np.percentile(v, 100.0 - tail))) if v.size else (float('nan'), float('nan'))
    return bounds
