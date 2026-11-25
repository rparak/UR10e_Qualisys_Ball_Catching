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
# File Name: ../Recording/Identification.py
## =========================================================================== ##

# Dataclasses (Data Classes)
from dataclasses import dataclass, field, replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Numpy Legendre series (the estimated floor's smooth correction)
from numpy.polynomial import legendre
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Flight/Core - the flight model and the Gauss-Newton fit of an initial state
from Flight.Core import Flight_Model_Str, Fit_Initial_States as Flight_Fit_Initial_States
#   ../Recording/Core - the recordings, the sets and the error
from Recording.Core import Recording_Str, Recording_Error_Cls, CONST_SET_IDENTIFICATION
#   ../Recording/Segmentation - the flights
from Recording.Segmentation import Flight_Str, Segment_Flights, CONST_SEGMENT_TOLERANCE

"""
Description:
    Initialization of constants.

    The identification of the flight model on the real flights (SPEC §2.3, §7.3; F3): quadratic drag a = g - k |v| v
    with one k for all flights, g fixed, each flight's
    initial state fitted by Gauss-Newton (Flight.Core.Fit_Initial_States). k is identified on the
    identification set only (Identify_Drag refuses any other flight); the validation flights are only
    given their initial states under the identified model (Fit_Initial_States fits no model
    parameter). Every value was chosen on the identification flights (F3 Part A, 2026-11-10).
    Pure computation: no file, no configuration, no printing.
"""
# The start of each flight dropped inside the identification only: F2's
# segments may begin in the hand; on the identification flights the first 40 ms lowered the max
# residual from 23.0 to 16.8 mm (median), and k moved 0.00930 -> 0.00939 1/m between 40 and 80 ms
# (F3 Part A). F2's segments are unchanged.
#   Unit [float, s]
CONST_IDENTIFICATION_TRIM = 0.040
# The sensitivity rows (F3): the segmentation's tolerance at the
# identification trim, and the trim at F2's tolerance.
#   Unit [Tuple<float>, m], [Tuple<float>, s]
CONST_SENSITIVITY_TOLERANCES = (0.010, 0.020, 0.050)
CONST_SENSITIVITY_TRIMS = (0.0, 0.040, 0.080)
# The degree of the per-axis Legendre correction of the estimated floor: a
# cubic needs 0.13 m/s^2 RMS (median), degree 5 already 0.52 and 7 1.71 m/s^2, more than any plausible
# force on the ball (the drag itself is about 0.4 m/s^2; F3 Part A). A heuristic, not a bound.
#   Unit [int]
CONST_FLOOR_DEGREE = 3
# The bootstrap over flights of the joint k: its seed (fixed in F3 Part A before any result) and size.
#   Unit [int]
CONST_BOOTSTRAP_SEED = 20261110
CONST_BOOTSTRAP_RESAMPLES = 2000
# The profile of k: the start (the 2022 constants, Cd 0.47, d 92 mm, m 0.12 kg, SPEC §4.2), the spacings
# of the 5 points of the per-flight quadratic in k, coarse first, and the convergence of the centre. The
# sum of squares is not exactly quadratic in k: at 0.5e-3 1/m alone the centre stops 2.7e-7 1/m off on
# noise-free synthetic flights; the bias falls with the spacing squared, so the centre is refined at
# 10 and 100 times finer spacings (test/Recording/test_identification.py). A spacing ends when the centre
# moves less than a hundredth of it, the finest when it moves less than CONST_PROFILE_TOLERANCE.
#   Unit [Tuple<float>, 1/m], [float, 1/m]
CONST_PROFILE_START = 0.0159
CONST_PROFILE_STEPS = (5e-4, 5e-5, 5e-6)
CONST_PROFILE_TOLERANCE = 1e-9


@dataclass
class Fitted_Flight_Str:
    """
    Description:
        One flight fitted under a flight model: its initial state and its residual (recorded minus
        simulated positions), in the QTM frame.
    """

    # The flight (possibly trimmed): its recording, set, rate, frames and positions.
    #   Unit [Flight_Str]
    Flight: Flight_Str = field(default_factory=Flight_Str)
    # The fitted state at the flight's first (kept) sample.
    #   Unit [Vector<float> 1x6, m, m/s]
    X_0: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # The residual per sample.
    #   Unit [Matrix<float> Mx3, m]
    Residual: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def RMS(self) -> float:
        """
        Description:
            The RMS of the residual's norm.

        Returns:
            (1) parameter [float]: [m].
        """

        return float(np.sqrt(np.mean(np.sum(self.Residual ** 2, axis=1))))

    @property
    def Max(self) -> float:
        """
        Description:
            The largest norm of the residual.

        Returns:
            (1) parameter [float]: [m].
        """

        return float(np.max(np.linalg.norm(self.Residual, axis=1)))


@dataclass
class Variant_Str:
    """
    Description:
        One identification of k: the identified row or a sensitivity row (F3).
    """

    # The variant's name, the segmentation's tolerance, the trim, and the flights left out.
    #   Unit [string], [float, m], [float, s], [List<string>]
    Name: str = ''
    Tolerance: float = 0.0
    Trim: float = 0.0
    Without: tp.List[str] = field(default_factory=list)
    # The flights segmented of the 90 recordings, the identification flights fitted, their samples.
    #   Unit [int]
    Flights_Segmented: int = 0
    Flights: int = 0
    Samples: int = 0
    # The joint k.
    #   Unit [float, 1/m]
    Drag: float = 0.0
    # The fitted flights at that k.
    #   Unit [List<Fitted_Flight_Str>]
    Fitted: tp.List[Fitted_Flight_Str] = field(default_factory=list)


@dataclass
class Identification_Str:
    """
    Description:
        The identified flight model and how it was obtained (SPEC §9.8).
    """

    # The identified model: gravity, the joint k and its standard deviation.
    #   Unit [Flight_Model_Str]
    Flight_Model_Str: Flight_Model_Str = field(default_factory=Flight_Model_Str)
    # The parts of Drag_Std: the bootstrap sd over flights and the half-range of the sensitivity rows.
    #   Unit [float, 1/m]
    Drag_Bootstrap_Sd: float = 0.0
    Drag_Half_Range: float = 0.0
    # The identified row first, then the sensitivity rows.
    #   Unit [List<Variant_Str>]
    Variants: tp.List[Variant_Str] = field(default_factory=list)


def Flight_Name(flight: Flight_Str) -> str:
    """
    Description:
        A flight's recording, as written in messages and outputs.

    Args:
        (1) flight [Flight_Str(object)]: The flight.

    Returns:
        (1) parameter [string]: '<folder>/Experiment_<n>.txt'.
    """

    return '%s/Experiment_%d.txt' % (flight.Folder, flight.Experiment)


def Session_Name(recording: Recording_Str) -> str:
    """
    Description:
        A recording's session: its folder and the recorder's fixed sample count (SPEC §9.2), e.g. the
        200 Hz OFF files of 2000 samples (Experiments 1-5).

    Args:
        (1) recording [Recording_Str(object)]: The recording.

    Returns:
        (1) parameter [string]: '<folder>, <n> samples'.
    """

    return '%s, %d samples' % (recording.Folder, len(recording.Frame))


def Trim_Flight(flight: Flight_Str, duration: float) -> Flight_Str:
    """
    Description:
        The flight without the samples before `duration` from its first: the frames re-based to the
        first kept sample, the first line moved. Only the identification uses it; F2's segment is
        unchanged.

    Args:
        (1) flight [Flight_Str(object)]: The flight.
        (2) duration [float]: The time dropped from the start [s], >= 0.

    Returns:
        (1) parameter [Flight_Str(object)]: The trimmed flight (a copy).
    """

    keep = np.where(flight.Time >= duration - 1e-12)[0]
    frame = flight.Frame[keep]
    return replace(flight, First_Line=flight.First_Line + int(keep[0]), Frame=frame - frame[0], Position=flight.Position[keep].copy())


def Fit_Initial_States(flights: tp.List[Flight_Str], *, Flight_Model_Str: Flight_Model_Str,
                       X_0: tp.Optional[np.ndarray] = None) -> tp.List[Fitted_Flight_Str]:
    """
    Description:
        Each flight's initial state under a given flight model, by Flight.Core.Fit_Initial_States (Gauss-Newton
        on its positions, least squares, all flights in one batch). No model parameter is fitted, so any
        flight may be passed (the validation flights included, SPEC §2.13).

    Args:
        (1) flights [List<Flight_Str>]: The flights.
        (2) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
        (3) X_0 [Matrix<float> Fx6]: The start of each fit [m, m/s]; None: the vacuum fit.

    Returns:
        (1) parameter [List<Fitted_Flight_Str>]: The fitted flights, in order.
    """

    x, residuals = Flight_Fit_Initial_States([flight.Time for flight in flights], [flight.Position for flight in flights],
                                             Flight_Model_Str=Flight_Model_Str, X_0=X_0)
    return [Fitted_Flight_Str(Flight=flight, X_0=x[i].copy(), Residual=residuals[i]) for i, flight in enumerate(flights)]


def Identify_Drag(flights: tp.List[Flight_Str], *, gravity: float,
                  trim: float = CONST_IDENTIFICATION_TRIM) -> tp.Tuple[float, np.ndarray, tp.List[Fitted_Flight_Str]]:
    """
    Description:
        The joint k of the flights, by profiling: each flight's sum of squared residuals is fitted as a
        quadratic in k over 5 values around the centre (its initial state refitted at each), and the
        centre moves to the minimum of their sum, at each spacing of CONST_PROFILE_STEPS in turn, until
        it moves less than a hundredth of the spacing (the finest: less than CONST_PROFILE_TOLERANCE).
        Refuses any flight that is not of the identification set (SPEC §7.3), naming each, before any
        computation.

    Args:
        (1) flights [List<Flight_Str>]: The identification flights (untrimmed).
        (2) gravity [float]: The magnitude of gravity, along -z [m/s^2].
        (3) trim [float]: The time dropped from the start of each flight [s].

    Returns:
        (1) parameter 1 [float]: The joint k [1/m].
        (2) parameter 2 [Matrix<float> Fx3]: Per flight, the quadratic (a, b, c) of its sum of squared
                                             residuals in (k - centre) at the coarsest spacing, about the
                                             centre it converged to [m^4, m^3, m^2] (Bootstrap_Drag_Sd).
        (3) parameter 3 [List<Fitted_Flight_Str>]: The trimmed flights fitted at the joint k.
    """

    refused = ['%s (%s)' % (Flight_Name(flight), flight.Set or 'no set') for flight in flights if flight.Set != CONST_SET_IDENTIFICATION]
    if refused:
        raise Recording_Error_Cls('The flight model is identified on the identification set only (SPEC §7.3), but %d flight(s) '
                                  'are not of it: %s. Pass only flights whose Set is %s.'
                                  % (len(refused), ', '.join(refused), CONST_SET_IDENTIFICATION))
    if not flights:
        raise Recording_Error_Cls('No flight was given to identify the flight model. Pass the identification flights.')
    trimmed = [Trim_Flight(flight, trim) for flight in flights]
    g = np.array([0.0, 0.0, -gravity])
    centre, x, coarse = CONST_PROFILE_START, None, np.zeros((len(flights), 3))
    for spacing in CONST_PROFILE_STEPS:
        offsets = spacing * np.arange(-2.0, 3.0)
        for _ in range(20):
            sse = []
            for k in centre + offsets:
                fitted = Fit_Initial_States(trimmed, Flight_Model_Str=Flight_Model_Str(Gravity=g, Drag=float(k)), X_0=x)
                x = np.array([f.X_0 for f in fitted])
                sse.append([float(np.sum(f.Residual ** 2)) for f in fitted])
            coefficients = np.array([np.polyfit(offsets, column, 2) for column in np.array(sse).T])
            if spacing == CONST_PROFILE_STEPS[0]:
                coarse = coefficients
            step = -np.sum(coefficients[:, 1]) / (2.0 * np.sum(coefficients[:, 0]))
            centre += step
            if abs(step) < (CONST_PROFILE_TOLERANCE if spacing == CONST_PROFILE_STEPS[-1] else 0.01 * spacing):
                break
    fitted = Fit_Initial_States(trimmed, Flight_Model_Str=Flight_Model_Str(Gravity=g, Drag=float(centre)), X_0=x)
    return float(centre), coarse, fitted


def Bootstrap_Drag_Sd(coefficients: np.ndarray, *, seed: int = CONST_BOOTSTRAP_SEED,
                      resamples: int = CONST_BOOTSTRAP_RESAMPLES) -> float:
    """
    Description:
        The standard deviation of the joint k over resamples of the flights with replacement: each
        resample's k is the minimum of its weighted sum of the per-flight quadratics (Identify_Drag),
        so no flight is refitted. The residuals are correlated within a flight, so the flight is the
        unit resampled.

    Args:
        (1) coefficients [Matrix<float> Fx3]: The per-flight quadratics in k about the profile's centre.
        (2) seed [int]: The seed of the resampling.
        (3) resamples [int]: The number of resamples.

    Returns:
        (1) parameter [float]: The standard deviation of k [1/m].
    """

    rng = np.random.default_rng(seed)
    F = len(coefficients)
    k = []
    for _ in range(resamples):
        w = np.bincount(rng.integers(0, F, F), minlength=F).astype(np.float64)
        k.append(-np.sum(w * coefficients[:, 1]) / (2.0 * np.sum(w * coefficients[:, 0])))
    return float(np.std(k))


def Estimated_Floor(time: np.ndarray, residual: np.ndarray) -> float:
    """
    Description:
        The estimated floor of a flight: the RMS of the residual's norm left
        after a per-axis Legendre polynomial of degree CONST_FLOOR_DEGREE in time is removed - the part
        no smooth force of plausible size explains. A heuristic, not a bound no predictor can beat
        (KNOWN_ISSUES).

    Args:
        (1) time [Vector<float> 1xM]: The sample times from the first [s].
        (2) residual [Matrix<float> Mx3]: The residual [m].

    Returns:
        (1) parameter [float]: The estimated floor [m].
    """

    u = 2.0 * (time - time[0]) / (time[-1] - time[0]) - 1.0
    V = legendre.legvander(u, CONST_FLOOR_DEGREE)
    c, *_ = np.linalg.lstsq(V, residual, rcond=None)
    left = residual - V @ c
    return float(np.sqrt(np.mean(np.sum(left ** 2, axis=1))))


def Edge_Interior_RMS(frame: np.ndarray, residual: np.ndarray) -> tp.Tuple[float, float]:
    """
    Description:
        The residual's RMS on the samples bordering a dropout (the frame before or after, inside the
        flight, is missing) and on the others (F3 Part A: 9.4 against 5.6 mm on the identification
        flights; an observation-model candidate for F6).

    Args:
        (1) frame [Vector<int> 1xM]: The frame indices, increasing.
        (2) residual [Matrix<float> Mx3]: The residual [m].

    Returns:
        (1) parameter 1 [float]: The RMS at the edges [m] (NaN without an edge sample).
        (2) parameter 2 [float]: The RMS elsewhere [m] (NaN without such a sample).
    """

    gap = np.diff(frame) > 1
    edge = np.zeros(len(frame), dtype=bool)
    edge[:-1] |= gap
    edge[1:] |= gap
    norm2 = np.sum(residual ** 2, axis=1)
    return (float(np.sqrt(np.mean(norm2[edge]))) if edge.any() else float('nan'),
            float(np.sqrt(np.mean(norm2[~edge]))) if (~edge).any() else float('nan'))


def _Variant(name: str, tolerance: float, trim: float, segmented: tp.List[tp.Tuple[str, tp.Optional[Flight_Str], str]],
             gravity: float, without_adjacent: bool = False) -> tp.Tuple[Variant_Str, np.ndarray]:
    """
    Description:
        One identification of k on the identification flights of one segmentation.

    Args:
        (1) name [string]: The variant's name.
        (2) tolerance [float]: The segmentation's tolerance [m].
        (3) trim [float]: The trim [s].
        (4) segmented [List<Tuple>]: Segment_Flights' result at that tolerance.
        (5) gravity [float]: The magnitude of gravity [m/s^2].
        (6) without_adjacent [bool]: Leave out the flights bordering an ambiguous step.

    Returns:
        (1) parameter 1 [Variant_Str(object)]: The variant.
        (2) parameter 2 [Matrix<float> Fx3]: The per-flight quadratics in k (Identify_Drag).
    """

    flights = [flight for set_name, flight, _ in segmented if flight is not None and set_name == CONST_SET_IDENTIFICATION]
    without = [Flight_Name(flight) for flight in flights if without_adjacent and flight.Adjacent_Ambiguous]
    flights = [flight for flight in flights if Flight_Name(flight) not in without]
    drag, coefficients, fitted = Identify_Drag(flights, gravity=gravity, trim=trim)
    return Variant_Str(Name=name, Tolerance=tolerance, Trim=trim, Without=without,
                       Flights_Segmented=sum(1 for _, flight, _ in segmented if flight is not None), Flights=len(flights),
                       Samples=sum(len(f.Flight.Frame) for f in fitted), Drag=drag, Fitted=fitted), coefficients


def Identify_Flight_Model(recordings: tp.List[Recording_Str], *, gravity: float) -> Identification_Str:
    """
    Description:
        The identified flight model (SPEC §2.3, §7.3): the joint k on the
        identification flights of F2's segmentation (CONST_SEGMENT_TOLERANCE) with CONST_IDENTIFICATION_TRIM
        dropped; the sensitivity rows (the tolerances of CONST_SENSITIVITY_TOLERANCES at that trim, the
        trims of CONST_SENSITIVITY_TRIMS at F2's tolerance, without the flights bordering an ambiguous
        step); Drag_Std = the bootstrap sd and the half-range of all rows' k in quadrature. Only
        identification flights are fitted.

    Args:
        (1) recordings [List<Recording_Str>]: The 90 recordings (Recording.Core.Load_Recordings).
        (2) gravity [float]: The magnitude of gravity, along -z [m/s^2].

    Returns:
        (1) parameter [Identification_Str(object)]: The identified model and its rows.
    """

    by_tolerance = {tolerance: Segment_Flights(recordings, tolerance=tolerance) for tolerance in CONST_SENSITIVITY_TOLERANCES}
    main, coefficients = _Variant('identified', CONST_SEGMENT_TOLERANCE, CONST_IDENTIFICATION_TRIM, by_tolerance[CONST_SEGMENT_TOLERANCE], gravity)
    variants = [main]
    for tolerance in CONST_SENSITIVITY_TOLERANCES:
        if tolerance != CONST_SEGMENT_TOLERANCE:
            variants.append(_Variant('tolerance %.0f mm' % (tolerance * 1e3), tolerance, CONST_IDENTIFICATION_TRIM,
                                     by_tolerance[tolerance], gravity)[0])
    for trim in CONST_SENSITIVITY_TRIMS:
        if trim != CONST_IDENTIFICATION_TRIM:
            variants.append(_Variant('trim %.0f ms' % (trim * 1e3), CONST_SEGMENT_TOLERANCE, trim, by_tolerance[CONST_SEGMENT_TOLERANCE], gravity)[0])
    variants.append(_Variant('without adjacent', CONST_SEGMENT_TOLERANCE, CONST_IDENTIFICATION_TRIM,
                             by_tolerance[CONST_SEGMENT_TOLERANCE], gravity, without_adjacent=True)[0])
    sd = Bootstrap_Drag_Sd(coefficients)
    drags = [variant.Drag for variant in variants]
    half_range = 0.5 * (max(drags) - min(drags))
    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -gravity]), Drag=main.Drag, Drag_Std=float(np.hypot(sd, half_range)))
    return Identification_Str(Flight_Model_Str=model, Drag_Bootstrap_Sd=sd, Drag_Half_Range=half_range, Variants=variants)


def Validate_Flight_Model(flights: tp.List[Flight_Str], *, Flight_Model_Str: Flight_Model_Str, trim: float) -> tp.List[Fitted_Flight_Str]:
    """
    Description:
        The identified model on other flights: each flight trimmed and given its initial state, the
        model unchanged (SPEC §2.13: the validation flights fit no model parameter;
        their results never feed back into the model, the trim or the segmentation).

    Args:
        (1) flights [List<Flight_Str>]: The flights (the validation set).
        (2) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the identified model.
        (3) trim [float]: The time dropped from the start of each flight [s].

    Returns:
        (1) parameter [List<Fitted_Flight_Str>]: The fitted flights.
    """

    return Fit_Initial_States([Trim_Flight(flight, trim) for flight in flights], Flight_Model_Str=Flight_Model_Str)
