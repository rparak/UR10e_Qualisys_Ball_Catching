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
# File Name: ../Recording/Segmentation.py
## =========================================================================== ##

# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Recording/Core - the recordings and the flight split
from Recording.Core import Recording_Str, Assign_Sets

"""
Description:
    Initialization of constants.

    The segmentation of the main free flight of a recording (SPEC §7.3): in
    the piece of time holding the highest sample, a window bounded by the wind-up before and the
    bounce after; a constant-acceleration fit whose worse end sample is dropped while its residual
    exceeds the tolerance; then the fit grown back sample by sample. Interior samples are never
    dropped: those off the fit by more than the tolerance and the stalls are listed (Q7).

    The constants were chosen on all 90 recordings, the validation set included (F2 Part A,
    2026-11-09; SPEC §7.3). Pure computation: no file, no configuration, no printing.
"""
# The residual tolerance of the fit [m]. Measured with the window bound below (F2 Part B, 2026-11-10):
# 90 of 90 flights segment for 15 to 50 mm, 89 for 10 mm (200 Hz OFF Experiment 36 is cut short by one
# later sample; F2 Part A measured 90 at 10 mm with a 0.05 m window bound); the start moves by up to
# 80 ms between 15 and 20 mm (KNOWN_ISSUES). F3 re-segments with 10 and 50 mm through tolerance.
#   Unit [float, m]
CONST_SEGMENT_TOLERANCE = 0.02
# The window around the highest sample: at most 1.0 s before it (release to the highest sample; a
# dropout may hide the apex) and 1.0 s after it (apex to floor: sqrt(2 * 2.9 m / g) = 0.77 s plus
# margin) [s].
#   Unit [float, s]
CONST_WINDOW_BEFORE = 1.0
CONST_WINDOW_AFTER = 1.0
# The window ends where z rises more than this above its running minimum, walking away from the
# highest sample: the wind-up before the release, the bounce after the floor contact [m]. Measured:
# the same 90 segments for 50, 100 and 200 mm; 5 mm trips on noise at the apex (88 of 90), 30 mm moves
# one flight's end by 685 ms.
#   Unit [float, m]
CONST_WINDOW_RISE = 0.1
# A recording whose highest sample is lower than this holds no throw [m] (the apexes are 1.86-3.18 m).
#   Unit [float, m]
CONST_APEX_MIN = 1.5
# The fewest samples a fit is made on (3 parameters per axis, with redundancy).
#   Unit [int]
CONST_MIN_SAMPLES = 6
# The acceptance of a segmented flight: free fall in z, little horizontal
# acceleration (drag only), the apex inside, a real flight's duration. Measured: a_z -9.65 to
# -9.89 m/s^2, horizontal at most 0.34 m/s^2 (median 0.17), 1.02-1.42 s on the 90 flights.
#   Unit [Tuple<float> m/s^2], [float, m/s^2], [float, s]
CONST_ACCELERATION_Z = (-10.5, -9.0)
CONST_ACCELERATION_HORIZONTAL = 1.5
CONST_MIN_DURATION = 0.3
# A later sample still fits while the fit is above this height: the ball is still in the air, so the
# segment was cut short by an outlier [m] (the ball's radius is 0.030 m, its markers reach 0.046 m).
#   Unit [float, m]
CONST_FLOOR_CLEARANCE = 0.05
# A stall (Q7): a one-frame step shorter than this fraction of both neighbouring one-frame steps, each
# longer than CONST_STALL_STEP: the sample repeats (nearly) the previous position, and the next step
# makes up two frames of motion.
#   Unit [float, -], [float, m]
CONST_STALL_FRACTION = 0.2
CONST_STALL_STEP = 0.01


@dataclass
class Flight_Str:
    """
    Description:
        The main free flight of a recording, from the release to the last sample before the first
        floor contact (SPEC §7.3), in the QTM frame. The samples are irregular: a dropout is a missing
        frame index, nothing is interpolated or resampled (SPEC §2.7).
    """

    # The recording's folder and experiment number.
    #   Unit [string], [int]
    Folder: str = ''
    Experiment: int = 0
    # The nominal sample rate.
    #   Unit [float, Hz]
    Rate: float = 0.0
    # CONST_SET_IDENTIFICATION or CONST_SET_VALIDATION (Recording.Core.Assign_Sets).
    #   Unit [string]
    Set: str = ''
    # The file lines of the first and the last sample of the flight.
    #   Unit [int]
    First_Line: int = 0
    Last_Line: int = 0
    # The frame index of each sample, 0 for the first.
    #   Unit [Vector<int> 1xM]
    Frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The positions in the QTM frame.
    #   Unit [Matrix<float> Mx3, m]
    Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The acceleration of the segmentation's constant-acceleration fit (not the flight model, F3).
    #   Unit [Vector<float> 1x3, m/s^2]
    Acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # The RMS and the largest norm of the fit's residuals.
    #   Unit [float, m]
    Residual_RMS: float = 0.0
    Residual_Max: float = 0.0
    # The lines of the samples off the fit by more than the tolerance, and of the stalls (kept, Q7).
    #   Unit [List<int>]
    Outlier_Lines: tp.List[int] = field(default_factory=list)
    Stall_Lines: tp.List[int] = field(default_factory=list)
    # An ambiguous step is adjacent to the first or the last sample: the flight
    # is kept; F3 may drop it in its sensitivity analysis.
    #   Unit [bool]
    Adjacent_Ambiguous: bool = False

    @property
    def Time(self) -> np.ndarray:
        """
        Description:
            The time of each sample from the first, Frame / Rate, exact to the frame.

        Returns:
            (1) parameter [Vector<float> 1xM]: The time [s].
        """

        return self.Frame * (1.0 / self.Rate)


def _Fit(t: np.ndarray, p: np.ndarray, t_0: float) -> np.ndarray:
    """
    Description:
        The least-squares constant-acceleration fit p(t) = p_0 + v_0 s + (a / 2) s^2, s = t - t_0, per axis.

    Args:
        (1) t [Vector<float> 1xM]: The times [s].
        (2) p [Matrix<float> Mx3]: The positions [m].
        (3) t_0 [float]: The reference time [s].

    Returns:
        (1) parameter [Matrix<float> 3x3]: The rows p_0 [m], v_0 [m/s], a / 2 [m/s^2].
    """

    s = t - t_0
    A = np.stack([np.ones_like(s), s, s ** 2], axis=1)
    c, *_ = np.linalg.lstsq(A, p, rcond=None)
    return c


def _Predict(c: np.ndarray, t: float, t_0: float) -> np.ndarray:
    """
    Description:
        The fit's position at one time.

    Args:
        (1) c [Matrix<float> 3x3]: The fit (_Fit).
        (2) t [float]: The time [s].
        (3) t_0 [float]: The fit's reference time [s].

    Returns:
        (1) parameter [Vector<float> 1x3]: The position [m].
    """

    s = t - t_0
    return c[0] + s * c[1] + s * s * c[2]


def _Window(recording: Recording_Str, k_0: int) -> tp.Tuple[int, int]:
    """
    Description:
        The window around the highest sample: in its piece of time, at most CONST_WINDOW_BEFORE /
        CONST_WINDOW_AFTER away, and cut where z rises more than CONST_WINDOW_RISE above its running
        minimum, walking away from the highest sample on each side.

    Args:
        (1) recording [Recording_Str(object)]: The recording.
        (2) k_0 [int]: The index of the highest sample.

    Returns:
        (1) parameter [Tuple<int, int>]: The first and the last index of the window.
    """

    t, z, piece = recording.Time, recording.Position[:, 2], recording.Piece
    index = np.where((piece == piece[k_0]) & (t >= t[k_0] - CONST_WINDOW_BEFORE) & (t <= t[k_0] + CONST_WINDOW_AFTER))[0]
    z_min, R = z[k_0], k_0
    for j in range(k_0 + 1, index[-1] + 1):
        if z[j] > z_min + CONST_WINDOW_RISE:
            break
        z_min, R = min(z_min, z[j]), j
    z_min, L = z[k_0], k_0
    for j in range(k_0 - 1, index[0] - 1, -1):
        if z[j] > z_min + CONST_WINDOW_RISE:
            break
        z_min, L = min(z_min, z[j]), j
    return L, R


def _Stalls(recording: Recording_Str, L: int, R: int) -> tp.List[int]:
    """
    Description:
        The stalls inside [L, R]: a one-frame step shorter than CONST_STALL_FRACTION of both
        neighbouring one-frame steps, each longer than CONST_STALL_STEP.

    Args:
        (1 - 2) L, R [int]: The first and the last index of the flight.

    Returns:
        (1) parameter [List<int>]: The file line of the later sample of each stalled step.
    """

    n = np.diff(recording.Frame)
    dp = np.linalg.norm(np.diff(recording.Position, axis=0), axis=1)
    return [j + 3 for j in range(max(L, 1), R - 1)
            if n[j - 1] == n[j] == n[j + 1] == 1 and dp[j - 1] > CONST_STALL_STEP and dp[j + 1] > CONST_STALL_STEP
            and dp[j] < CONST_STALL_FRACTION * min(dp[j - 1], dp[j + 1])]


def Segment_Flight(recording: Recording_Str, set_name: str, *,
                   tolerance: float = CONST_SEGMENT_TOLERANCE) -> tp.Tuple[tp.Optional[Flight_Str], str]:
    """
    Description:
        Segment the main free flight of a recording (SPEC §7.3). A flight is accepted when its fit has
        a_z in CONST_ACCELERATION_Z, a horizontal acceleration of at most CONST_ACCELERATION_HORIZONTAL,
        its apex inside the segment, at least CONST_MIN_DURATION, does not reach the start or the end of
        the recording, and is not cut short (no later sample still fits while the ball is in the air).

        A flight never contains an ambiguous step: the window lies in one piece of time, and a piece
        ends at every ambiguous step. A flight whose first or last sample borders one is flagged
        (Adjacent_Ambiguous) and kept.

    Args:
        (1) recording [Recording_Str(object)]: The recording.
        (2) set_name [string]: The recording's set (Recording.Core.Assign_Sets).
        (3) tolerance [float]: The fit's residual tolerance [m]; F3 varies it (10, 20, 50 mm).

    Returns:
        (1) parameter 1 [Flight_Str(object)]: The flight, or None when no flight is accepted.
        (2) parameter 2 [string]: '' for a flight; otherwise why not, each failed criterion
                                  separated by '; '.
    """

    t, p, piece = recording.Time, recording.Position, recording.Piece
    N = len(t)
    k_0 = int(np.argmax(p[:, 2]))
    if p[k_0, 2] < CONST_APEX_MIN:
        return None, 'no sample above %.1f m (the highest is %.2f m)' % (CONST_APEX_MIN, p[k_0, 2])
    lo, hi = _Window(recording, k_0)
    L, R, t_0 = lo, hi, t[k_0]

    # 1\ Drop the worse end sample while it is off the fit by more than the tolerance.
    while R - L + 1 >= CONST_MIN_SAMPLES:
        c = _Fit(t[L:R + 1], p[L:R + 1], t_0)
        e_L, e_R = np.linalg.norm(_Predict(c, t[L], t_0) - p[L]), np.linalg.norm(_Predict(c, t[R], t_0) - p[R])
        if max(e_L, e_R) <= tolerance:
            break
        if e_L >= e_R and L < k_0:
            L += 1
        elif R > k_0:
            R -= 1
        elif L < k_0:
            L += 1
        else:
            break
    else:
        return None, 'fewer than %d samples are left after trimming' % CONST_MIN_SAMPLES

    # 2\ Grow back, one sample at a time, while the next sample fits.
    changed = True
    while changed:
        changed = False
        c = _Fit(t[L:R + 1], p[L:R + 1], t_0)
        if L - 1 >= lo and np.linalg.norm(_Predict(c, t[L - 1], t_0) - p[L - 1]) <= tolerance:
            L, changed = L - 1, True
        elif R + 1 <= hi and np.linalg.norm(_Predict(c, t[R + 1], t_0) - p[R + 1]) <= tolerance:
            R, changed = R + 1, True

    # 3\ The acceptance.
    c = _Fit(t[L:R + 1], p[L:R + 1], t_0)
    a = 2.0 * c[2]
    residual = np.linalg.norm(c[0] + np.outer(t[L:R + 1] - t_0, c[1]) + np.outer((t[L:R + 1] - t_0) ** 2, c[2]) - p[L:R + 1], axis=1)
    t_apex = t_0 - c[1][2] / a[2] if a[2] != 0.0 else np.inf
    later = [j for j in range(R + 1, hi + 1)
             if _Predict(c, t[j], t_0)[2] > CONST_FLOOR_CLEARANCE and np.linalg.norm(_Predict(c, t[j], t_0) - p[j]) <= tolerance]
    why: tp.List[str] = []
    if not (CONST_ACCELERATION_Z[0] <= a[2] <= CONST_ACCELERATION_Z[1]):
        why.append('a_z %.2f m/s^2 outside [%.1f, %.1f]' % (a[2], *CONST_ACCELERATION_Z))
    if np.hypot(a[0], a[1]) > CONST_ACCELERATION_HORIZONTAL:
        why.append('horizontal acceleration %.2f m/s^2 > %.1f' % (np.hypot(a[0], a[1]), CONST_ACCELERATION_HORIZONTAL))
    if not (t[L] < t_apex < t[R]):
        why.append('the apex is not inside the segment')
    if t[R] - t[L] < CONST_MIN_DURATION:
        why.append('duration %.3f s < %.1f s' % (t[R] - t[L], CONST_MIN_DURATION))
    if L == 0 or R == N - 1:
        why.append('reaches the start or the end of the recording')
    if later:
        why.append('cut short: %d later sample(s) still fit, the first on line %d' % (len(later), later[0] + 2))
    if why:
        return None, '; '.join(why)

    adjacent = bool((L > 0 and piece[L - 1] != piece[L]) or (R < N - 1 and piece[R + 1] != piece[R]))
    flight = Flight_Str(Folder=recording.Folder, Experiment=recording.Experiment, Rate=recording.Rate, Set=set_name,
                        First_Line=L + 2, Last_Line=R + 2, Frame=recording.Frame[L:R + 1] - recording.Frame[L],
                        Position=p[L:R + 1].copy(), Acceleration=a, Residual_RMS=float(np.sqrt(np.mean(residual ** 2))),
                        Residual_Max=float(residual.max()), Outlier_Lines=[L + 2 + i for i in np.where(residual > tolerance)[0]],
                        Stall_Lines=_Stalls(recording, L, R), Adjacent_Ambiguous=adjacent)
    return flight, ''


def Segment_Flights(recordings: tp.List[Recording_Str], *,
                    tolerance: float = CONST_SEGMENT_TOLERANCE) -> tp.List[tp.Tuple[str, tp.Optional[Flight_Str], str]]:
    """
    Description:
        The set and the segmented flight of every recording, in the recordings' order.

    Args:
        (1) recordings [List<Recording_Str>]: The recordings (Recording.Core.Load_Recordings).
        (2) tolerance [float]: The fit's residual tolerance [m].

    Returns:
        (1) parameter [List<Tuple<string, Flight_Str, string>>]: Per recording, its set, its flight (or
                                                                 None) and why not ('' for a flight).
    """

    sets = Assign_Sets([(recording.Folder, 'Experiment_%d.txt' % recording.Experiment) for recording in recordings])
    out = []
    for recording in recordings:
        set_name = sets[(recording.Folder, 'Experiment_%d.txt' % recording.Experiment)]
        flight, reason = Segment_Flight(recording, set_name, tolerance=tolerance)
        out.append((set_name, flight, reason))
    return out
