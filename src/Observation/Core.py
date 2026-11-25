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
# File Name: ../Observation/Core.py
## =========================================================================== ##

# Hashlib (Secure hashes; the donor pools from the names)
import hashlib
# Dataclasses (Data Classes)
from dataclasses import dataclass, field, replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the vocabulary of the dataset configuration's pattern and pool fields
from Configuration.Core import CONST_DROPOUT_PATTERNS, CONST_DONOR_POOLS

"""
Description:
    Initialization of constants.

    The observation model (SPEC §7.5): it turns a simulated flight into motion-capture samples at
    capture times (no latency). Which frames carry a sample comes from the dropout patterns of the 45
    identification flights (the donors), replayed in seconds from the release on the flight's own frame
    grid; the position error is the model M4: white Student-t noise, an
    Ornstein-Uhlenbeck drift continuous in time, and an offset that evolves only while the ball is lost.
    The offset vector and a sample's white-noise vector stay within a bound measured on the identification
    flights: a value beyond it is drawn again. Every random number comes from the
    generator the caller passes (the flight's own stream).
"""
# The mean time between two donor switches of a composed pattern: exponential
# cut times; F6 Part A measured that 0.25 s keeps every dropout statistic inside its bootstrap bound at
# 100 and 200 Hz (0.1 s broke the p99 dropout at 200 Hz). *Assumed*.
#   Unit [float, s]
CONST_CUT_MEAN = 0.25
# The donors a flight draws from: this many of its pool, nearest in duration to the flight (F6 Q3).
#   Unit [int]
CONST_NEAREST_DONORS = 10
# The donor pool of each simulated rate (F6 Q4, Q5): 100, 200 and 300 Hz replay their own donors; 370 Hz,
# which has no recording, replays the 300 Hz donors (an extrapolation, *assumed*). Any other rate has no
# donors and is refused.
#   Unit [Dict<float, float>, Hz]
CONST_DONOR_RATES = {100.0: 100.0, 200.0: 200.0, 300.0: 300.0, 370.0: 300.0}
# The pattern kinds (SPEC §8.2, Observation.Dropout.Pattern): composed replay, the replay of one donor,
# or no dropout (an ablation); defined once, in Configuration.Core.
#   Unit [Tuple<string>]
CONST_PATTERNS = CONST_DROPOUT_PATTERNS
# The donor pools (SPEC §8.2, Observation.Dropout.Donors; F6 Q11): all donors, or one of the two halves
# A and B of each rate, for F8's check that a predictor does not gain from knowing the patterns.
#   Unit [Tuple<string>]
CONST_POOLS = CONST_DONOR_POOLS
# The seed of the split into the pools A and B, from the donor names only (as Recording.Core.Assign_Sets).
#   Unit [int]
CONST_POOL_SEED = 20261111
# A rounding allowance for the comparison of frame and sample times.
#   Unit [float, s]
CONST_TIME_EPSILON = 1e-9
# The redraws of one value before the bound is reported as one that cannot be met.
# With the configured model 0.4 % of the offset draws and 0.002 % of the white-noise draws lie beyond their
# bounds (Part A, 2026-11-15), so a value is redrawn once or twice; 1 000 failures mean a bound far below the
# component's size.
#   Unit [int]
CONST_MAX_REDRAWS = 1000


class Observation_Error_Cls(ValueError):
    """
    Description:
        The observation model cannot be applied: a rate without donors, an unknown pattern or pool, an
        empty pool. The message names the value and what is allowed.
    """


@dataclass
class Donor_Str:
    """
    Description:
        The dropout pattern of one identification flight: its sample times from the first sample at or
        after the 40 ms trim, which stands for the release.
    """

    # The recording, '<folder>/Experiment_<n>.txt'.
    #   Unit [string]
    Name: str = ''
    # The recording's rate.
    #   Unit [float, Hz]
    Rate: float = 0.0
    # The sample times, 0 for the first; whole frames of the rate.
    #   Unit [Vector<float> 1xM, s]
    Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 'A' or 'B' (Assign_Pools).
    #   Unit [string]
    Pool: str = ''

    @property
    def End(self) -> float:
        """
        Description:
            The time of the donor's last sample.

        Returns:
            (1) parameter [float]: [s].
        """

        return float(self.Time[-1])


@dataclass
class Observation_Model_Str:
    """
    Description:
        The parameters of the error model M4 (SPEC §7.5, §8.1 Observation): per axis in the robot frame
        (x, y, z). A Student-t value is scaled to unit variance, so a standard deviation is that of the
        component.
    """

    # The white noise per sample: its standard deviation per axis and its degrees of freedom.
    #   Unit [Vector<float> 1x3, m], [float, -]
    White_Std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    White_Dof: float = 0.0
    # The drift: an Ornstein-Uhlenbeck process continuous in time, its stationary standard deviation per
    # axis and its time constant.
    #   Unit [Vector<float> 1x3, m], [float, s]
    Drift_Std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Drift_Time: float = 0.0
    # The offset: an Ornstein-Uhlenbeck process in the time the ball is lost (constant within a run of
    # samples), its stationary standard deviation per axis, its time constant and the degrees of freedom
    # of its innovations.
    #   Unit [Vector<float> 1x3, m], [float, s], [float, -]
    Offset_Std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Offset_Time: float = 0.0
    Offset_Dof: float = 0.0
    # The bounds: the largest norm of the offset vector and of the white-noise vector of
    # one sample; a value beyond its bound is drawn again (Observation_Error). Infinite: no bound, the
    # components as their Student-t distributions define them (the configuration always gives finite ones).
    #   Unit [float, m]
    Offset_Bound: float = np.inf
    White_Bound: float = np.inf


def Assign_Pools(donors: tp.List[Donor_Str]) -> tp.List[Donor_Str]:
    """
    Description:
        The pools A and B (F6 Q11): per rate, the donors ranked by the SHA-256 hex digest of
        '<CONST_POOL_SEED>/<name>'; the first half (len // 2) is A, the rest B. Only the names are used.

    Args:
        (1) donors [List<Donor_Str>]: The donors.

    Returns:
        (1) parameter [List<Donor_Str>]: Copies with their Pool set, in the given order.
    """

    pools: tp.Dict[str, str] = {}
    for rate in sorted({donor.Rate for donor in donors}):
        names = sorted(donor.Name for donor in donors if donor.Rate == rate)
        ranked = sorted(names, key=lambda name: hashlib.sha256(('%d/%s' % (CONST_POOL_SEED, name)).encode('utf-8')).hexdigest())
        for rank, name in enumerate(ranked):
            pools[name] = 'A' if rank < len(ranked) // 2 else 'B'
    return [replace(donor, Pool=pools[donor.Name]) for donor in donors]


def Frame_Count(duration: float, rate: float) -> int:
    """
    Description:
        The number of frames of a flight from its release (frame 0) to its end: every frame j with
        j / rate <= duration.

    Args:
        (1) duration [float]: The time from the release to the end [s], >= 0.
        (2) rate [float]: The rate [Hz].

    Returns:
        (1) parameter [int]: The number of frames.
    """

    return int(np.floor(duration * rate + CONST_TIME_EPSILON)) + 1


def Present_On_Grid(segments: tp.List[tp.Tuple[Donor_Str, float, float]], rate: float, n: int) -> np.ndarray:
    """
    Description:
        Which frames of a simulated flight carry a sample. Frame j at t = j / rate belongs to the segment
        [start, end) that holds t, and is present when a sample t_k of that segment's donor has t in its
        own frame period [t_k - T/2, t_k + T/2), T the donor's period: a sample stands for its frame, so a
        replay at another rate keeps the coverage. A time after the donor's last sample has no sample.
        Frame 0 (the release) is always present.

    Args:
        (1) segments [List<Tuple<Donor_Str, float, float>>]: (donor, start, end) [s], covering the flight.
        (2) rate [float]: The simulated rate [Hz].
        (3) n [int]: The number of frames.

    Returns:
        (1) parameter [Vector<bool> 1xn]: True for a frame with a sample.
    """

    t = np.arange(n) / rate
    present = np.zeros(n, dtype=bool)
    for donor, start, end in segments:
        inside = (t >= start - CONST_TIME_EPSILON) & (t < end - CONST_TIME_EPSILON)
        if not inside.any():
            continue
        half = 0.5 / donor.Rate
        k = np.searchsorted(donor.Time - half - CONST_TIME_EPSILON, t[inside], side='right') - 1
        offset = t[inside] - donor.Time[np.clip(k, 0, donor.Time.size - 1)]
        present[inside] = (k >= 0) & (offset >= -half - CONST_TIME_EPSILON) & (offset < half - CONST_TIME_EPSILON)
    if n:
        present[0] = True
    return present


def Donor_Pool(donors: tp.List[Donor_Str], rate: float, pool: str) -> tp.List[Donor_Str]:
    """
    Description:
        The donors a flight of a rate may draw from: those of its donor rate (CONST_DONOR_RATES) in the
        pool ('all', 'A', 'B').

    Args:
        (1) donors [List<Donor_Str>]: The donors, their pools assigned (Assign_Pools).
        (2) rate [float]: The simulated rate [Hz].
        (3) pool [string]: One of CONST_POOLS.

    Returns:
        (1) parameter [List<Donor_Str>]: The pool, ordered by name.
    """

    if float(rate) not in CONST_DONOR_RATES:
        raise Observation_Error_Cls('The rate %g Hz has no donors: the observation model replays the identification flights of '
                                    '100, 200 and 300 Hz, and 370 Hz replays the 300 Hz ones. Use one of %s Hz.'
                                    % (rate, ', '.join('%g' % r for r in CONST_DONOR_RATES)))
    if pool not in CONST_POOLS:
        raise Observation_Error_Cls('The donor pool %r is unknown. Use one of %s.' % (pool, ', '.join(CONST_POOLS)))
    selected = sorted((donor for donor in donors if donor.Rate == CONST_DONOR_RATES[float(rate)] and pool in ('all', donor.Pool)),
                      key=lambda donor: donor.Name)
    if not selected:
        raise Observation_Error_Cls('No donor of %g Hz is in the pool %r. Load the identification flights of that rate.'
                                    % (CONST_DONOR_RATES[float(rate)], pool))
    return selected


def Draw_Pattern(donors: tp.List[Donor_Str], rate: float, duration: float, pattern: str, pool: str,
                 rng: np.random.Generator) -> tp.Tuple[np.ndarray, tp.List[str]]:
    """
    Description:
        The frames of a simulated flight that carry a sample (SPEC §7.5). The
        candidates are the CONST_NEAREST_DONORS donors of the pool nearest in duration (their last
        sample) to the flight, ties by name. 'composed': cut times at exponential intervals of mean
        CONST_CUT_MEAN from the release, each segment replaying a candidate drawn at random, at the same
        time since the release; 'recorded': one candidate for the whole flight; 'none': every frame.

    Args:
        (1) donors [List<Donor_Str>]: The donors, their pools assigned (Assign_Pools).
        (2) rate [float]: The simulated rate [Hz].
        (3) duration [float]: The flight's time from the release to the floor [s].
        (4) pattern [string]: One of CONST_PATTERNS.
        (5) pool [string]: One of CONST_POOLS.
        (6) rng [np.random.Generator]: The flight's generator.

    Returns:
        (1) parameter 1 [Vector<bool> 1xn]: True for a frame with a sample, n = Frame_Count(duration, rate).
        (2) parameter 2 [List<string>]: The donors replayed, in the order of their segments.
    """

    if pattern not in CONST_PATTERNS:
        raise Observation_Error_Cls('The dropout pattern %r is unknown. Use one of %s.' % (pattern, ', '.join(CONST_PATTERNS)))
    n = Frame_Count(duration, rate)
    if pattern == 'none':
        return np.ones(n, dtype=bool), []
    candidates = sorted(Donor_Pool(donors, rate, pool), key=lambda donor: (abs(donor.End - duration), donor.Name))[:CONST_NEAREST_DONORS]
    if pattern == 'recorded':
        donor = candidates[int(rng.integers(len(candidates)))]
        return Present_On_Grid([(donor, 0.0, np.inf)], rate, n), [donor.Name]
    segments: tp.List[tp.Tuple[Donor_Str, float, float]] = []
    start = 0.0
    while start <= duration:
        end = start + float(rng.exponential(CONST_CUT_MEAN))
        segments.append((candidates[int(rng.integers(len(candidates)))], start, end))
        start = end
    return Present_On_Grid(segments, rate, n), [donor.Name for donor, _, _ in segments]


def _Student_T(rng: np.random.Generator, dof: float, size: tp.Tuple[int, ...]) -> np.ndarray:
    """
    Description:
        Student-t values scaled to unit variance.

    Args:
        (1) rng [np.random.Generator]: The generator.
        (2) dof [float]: The degrees of freedom, > 2.
        (3) size [Tuple<int>]: The shape.

    Returns:
        (1) parameter [np.ndarray]: The values [-].
    """

    return rng.standard_t(dof, size=size) * np.sqrt((dof - 2.0) / dof)


def _Within_Bound(value: np.ndarray, again: tp.Callable[[], np.ndarray], bound: float, name: str) -> np.ndarray:
    """
    Description:
        A value whose norm is within its bound: the value itself, or the first of at most CONST_MAX_REDRAWS
        new draws that is (a redraw, so that no value sits on the bound). A bound that
        no draw meets is an error naming it, never a silent clamp.

    Args:
        (1) value [Vector<float> 1x3]: The value drawn [m].
        (2) again [Callable[[], Vector<float> 1x3]]: Draws the value again.
        (3) bound [float]: The largest norm [m].
        (4) name [string]: The bound's configuration field, for the message.

    Returns:
        (1) parameter [Vector<float> 1x3]: The value within the bound [m].
    """

    if np.linalg.norm(value) <= bound:
        return value
    for _ in range(CONST_MAX_REDRAWS):
        value = again()
        if np.linalg.norm(value) <= bound:
            return value
    raise Observation_Error_Cls('No draw of the observation error met %s = %g m in %d redraws (the last one %.3g m). The bound is far '
                                'below the size of its component; check it against Observation.Offset_Std and White_Std (SPEC §7.5).'
                                % (name, bound, CONST_MAX_REDRAWS, float(np.linalg.norm(value))))


def Observation_Error(time: np.ndarray, rate: float, *, Observation_Model_Str: Observation_Model_Str, scale: float,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Description:
        The position error of the samples of one flight under M4 (SPEC §7.5): white Student-t noise per
        sample; a drift x_k = a x_{k-1} + sqrt(1 - a^2) s xi_k with a = exp(-dt / Drift_Time), continuous
        across dropouts; an offset that keeps its value within a run and, after a dropout of lost time L
        (the time between the samples less one period), becomes b o + sqrt(1 - b^2) s eta with
        b = exp(-L / Offset_Time) and Student-t eta. Both processes start in their stationary
        distribution. The random numbers are drawn in one fixed order (white, drift, offset; every
        sample), so the error depends on the generator and the times only.

        The bounds: an offset whose norm exceeds Offset_Bound is drawn again (a new
        eta for that step) until it is within, and then a white-noise vector whose norm exceeds White_Bound
        likewise, each in the order of the samples and from the same generator, after the three blocks.
        A flight without a value beyond the bounds has exactly the error of the unbounded model. The
        bounds apply before `scale`, which multiplies the whole error.

    Args:
        (1) time [Vector<float> 1xM]: The sample times [s], increasing, whole frames of the rate.
        (2) rate [float]: The rate [Hz].
        (3) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: the model.
        (4) scale [float]: A multiplier of the whole error (Observation.Noise.Scale), >= 0.
        (5) rng [np.random.Generator]: The flight's generator.

    Returns:
        (1) parameter [Matrix<float> Mx3]: The error to add to the true positions [m].
    """

    model = Observation_Model_Str
    m = len(time)
    white = _Student_T(rng, model.White_Dof, (m, 3)) * model.White_Std
    drift_noise = rng.standard_normal((m, 3)) * model.Drift_Std
    offset_noise = _Student_T(rng, model.Offset_Dof, (m, 3)) * model.Offset_Std
    drift = np.empty((m, 3))
    offset = np.empty((m, 3))

    def Innovation() -> np.ndarray:
        """
        Description:
            A new innovation of the offset.

        Returns:
            (1) parameter [Vector<float> 1x3]: s eta [m].
        """

        return _Student_T(rng, model.Offset_Dof, (3,)) * model.Offset_Std

    if m:
        drift[0] = drift_noise[0]
        offset[0] = _Within_Bound(offset_noise[0], Innovation, model.Offset_Bound, 'Observation.Offset_Bound')
    step = np.diff(time)
    a = np.exp(-step / model.Drift_Time)
    lost = np.maximum(step - 1.0 / rate, 0.0)
    b = np.where(lost > CONST_TIME_EPSILON, np.exp(-lost / model.Offset_Time), 1.0)
    for k in range(1, m):
        drift[k] = a[k - 1] * drift[k - 1] + np.sqrt(1.0 - a[k - 1] ** 2) * drift_noise[k]
        keep, weight = b[k - 1], np.sqrt(1.0 - b[k - 1] ** 2)
        offset[k] = keep * offset[k - 1] + weight * offset_noise[k]
        # Inside a run the offset does not move (weight 0): only a step across a dropout can leave the bound.
        if weight > 0.0:
            offset[k] = _Within_Bound(offset[k], lambda: keep * offset[k - 1] + weight * Innovation(), model.Offset_Bound,
                                      'Observation.Offset_Bound')
    for k in np.flatnonzero(np.linalg.norm(white, axis=1) > model.White_Bound):
        white[k] = _Within_Bound(white[k], lambda: _Student_T(rng, model.White_Dof, (3,)) * model.White_Std, model.White_Bound,
                                 'Observation.White_Bound')
    return scale * (white + drift + offset)


def Error_Covariance(time: np.ndarray, rate: float, *, Observation_Model_Str: Observation_Model_Str) -> np.ndarray:
    """
    Description:
        The covariance of M4's position error (Observation_Error at scale 1) between the samples of one flight,
        per axis (SPEC §7.5): white s_w^2 on the diagonal; the drift s_d^2 exp(-|t_i - t_j| / Drift_Time);
        the offset s_o^2 exp(-L_ij / Offset_Time), L_ij the lost time between the two samples (the sum over
        the steps between them of the step less one period, the steps of lost time within CONST_TIME_EPSILON
        counting none, as in Observation_Error). Student-t values have unit variance, so the degrees of
        freedom do not enter. The weights of the GLS_M4 predictor (SPEC §7.7); the scale of a dataset
        (Observation.Noise.Scale) multiplies every element by its square and is left out: a weighted
        least-squares estimate does not depend on it.

        These are the variances of the unbounded components: Offset_Bound and
        White_Bound do not enter. A bounded offset has 0.60-0.72 of Offset_Std^2 and a bounded white noise
        0.38-0.45 of White_Std^2 per axis at the configured values (Part A, 2026-11-15); GLS_M4 with those
        variances differed by -0.5 to +0.2 mm in the median remainder RMS on the simulated validation
        partition, without a consistent sign, so the weights and the predictor's results on real flights
        stay as they are.

    Args:
        (1) time [Vector<float> 1xM]: The sample times [s], increasing.
        (2) rate [float]: The rate [Hz].
        (3) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: the model.

    Returns:
        (1) parameter [Tensor<float> 3xMxM]: The covariance per axis x, y, z [m^2].
    """

    time = np.asarray(time, dtype=np.float64)
    model = Observation_Model_Str
    lost = np.maximum(np.diff(time) - 1.0 / rate, 0.0)
    lost[lost <= CONST_TIME_EPSILON] = 0.0
    cumulative = np.concatenate(([0.0], np.cumsum(lost)))
    drift = np.exp(-np.abs(time[:, None] - time[None, :]) / model.Drift_Time)
    offset = np.exp(-np.abs(cumulative[:, None] - cumulative[None, :]) / model.Offset_Time)
    identity = np.eye(len(time))
    return np.stack([model.White_Std[a] ** 2 * identity + model.Drift_Std[a] ** 2 * drift + model.Offset_Std[a] ** 2 * offset
                     for a in range(3)])
