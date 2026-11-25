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
# File Name: ../test/Recording/test_segmentation.py
## =========================================================================== ##

# System (Default)
import os
# Collections (Container datatypes)
import collections
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the segmentation under test
#   ../Recording/Segmentation, ../Recording/Core
from Recording.Segmentation import Segment_Flight, Segment_Flights
from Recording.Core import Build_Recording, Load_Recordings, Assign_Sets

"""
Description:
    Initialization of constants.

    The synthetic throws are built here from their own physics (release, free fall, floor contact),
    so the expected first and last sample come from the construction. The real-data pins come from
    the F2 Part A scratch measurement of 2026-11-09 (an independent script, flights.tsv), never from
    the code under test.
"""
# The first and the last file line of every segmented flight, per folder and experiment.
#   Unit [Dict<string, Dict<int, Tuple<int, int>>>]
CONST_FLIGHT_LINES = {
    'Frequency_100_Hz_Active_Filtering_OFF': {
        1: (497, 555), 2: (466, 524), 3: (373, 411), 4: (292, 362), 5: (481, 514), 6: (436, 480),
        7: (328, 397), 8: (359, 428), 9: (322, 397), 10: (353, 409)},
    'Frequency_100_Hz_Active_Filtering_ON': {
        1: (434, 504), 2: (485, 555), 3: (227, 266), 4: (644, 712), 5: (622, 661), 6: (682, 711),
        7: (477, 547), 8: (507, 555), 9: (552, 606), 10: (514, 555)},
    'Frequency_200_Hz_Active_Filtering_OFF': {
        1: (1230, 1435), 2: (876, 1069), 3: (950, 1151), 4: (918, 1040), 5: (821, 1012), 6: (1056, 1184),
        7: (913, 1041), 8: (922, 1077), 9: (1040, 1135), 10: (949, 1011), 11: (784, 974), 12: (1009, 1063),
        13: (769, 855), 14: (1112, 1274), 15: (512, 636), 16: (826, 911), 17: (1035, 1103), 18: (960, 1104),
        19: (409, 500), 20: (1370, 1550), 21: (1185, 1323), 22: (1419, 1502), 23: (1193, 1398),
        24: (855, 966), 25: (970, 1053), 26: (1113, 1239), 27: (1026, 1119), 28: (954, 1087),
        29: (1246, 1317), 30: (1200, 1284), 31: (1225, 1311), 32: (1007, 1120), 33: (451, 566),
        34: (273, 376), 35: (966, 1049), 36: (1316, 1429), 37: (1018, 1092), 38: (927, 1051),
        39: (1510, 1610), 40: (1269, 1398), 41: (1059, 1176), 42: (618, 784), 43: (756, 866), 44: (474, 572),
        45: (1032, 1138), 46: (559, 607), 47: (729, 830), 48: (706, 898), 49: (585, 713), 50: (652, 785)},
    'Frequency_200_Hz_Active_Filtering_ON': {
        1: (1012, 1146), 2: (1150, 1304), 3: (1034, 1095), 4: (1029, 1134), 5: (1035, 1196), 6: (996, 1100),
        7: (950, 1026), 8: (1062, 1180), 9: (1077, 1200), 10: (1094, 1201)},
    'Frequency_300_Hz_Active_Filtering_OFF': {
        1: (714, 967), 2: (602, 840), 3: (1081, 1251), 4: (610, 719), 5: (1035, 1266), 6: (1154, 1306),
        7: (1265, 1489), 8: (1469, 1633), 9: (528, 766), 10: (1079, 1208)}}
# The stalls inside the flights (F2 Part A, 2026-11-09), by file line.
#   Unit [Dict<Tuple<string, int>, List<int>>]
CONST_STALLS = {('Frequency_200_Hz_Active_Filtering_OFF', 50): [673], ('Frequency_300_Hz_Active_Filtering_OFF', 2): [663],
                ('Frequency_300_Hz_Active_Filtering_OFF', 7): [1348]}


def _Throw(release_z=1.0, v=(0.0, -4.0, 6.0), g=9.81, a_x=0.0, s_contact=1.3605, after='bounce', gap_after=0.02,
           drop_ascent=False, stop_at=None, jump_at=None, rate=200.0):
    """
    Description:
        A synthetic throw at 200 Hz: 0.5 s at rest in the hand at (0, 0, release_z), released at t = 0.5 s
        with velocity v, free flight under (a_x, 0, -g) until the floor contact s_contact after the release
        (the floor is where the ball is then), then no sample for gap_after, then a bounce (5 m/s up) or
        rest, 0.5 s. Every third frame of the flight is dropped (irregular sampling). drop_ascent removes
        every sample from 0.45 s before the release to the apex + 0.05 s (the hand and the ascent out of
        view, so no sample lies within the window's 1.0 s before the highest one); stop_at ends the recording s after the release;
        jump_at moves every later position by +1 m in x (a hidden gap). Returns the recording and the
        indices of the release sample and of the last sample before the contact.
    """

    period = 1.0 / rate
    f_release = int(round(0.5 * rate))
    s_apex = v[2] / g
    p_release = np.array([0.0, 0.0, release_z])
    a = np.array([a_x, 0.0, -g])
    frames, positions = [], []
    first = last = None
    for f in range(int(round((0.5 + s_contact + gap_after + 0.5) * rate)) + 1):
        s = (f - f_release) * period
        if stop_at is not None and s > stop_at:
            break
        if s < 0.0:
            if drop_ascent and s >= -0.45:
                continue
            p = p_release
        elif s < s_contact:
            if f % 3 == 1 and f != f_release:
                continue
            if drop_ascent and s < s_apex + 0.05:
                continue
            p = p_release + np.array(v) * s + 0.5 * a * s ** 2
        elif s < s_contact + gap_after:
            continue
        else:
            p_contact = p_release + np.array(v) * s_contact + 0.5 * a * s_contact ** 2
            u = min(s - s_contact, 0.4)
            p = p_contact + (np.array([v[0], v[1], 5.0]) * u - np.array([0.0, 0.0, 0.5 * g * u ** 2]) if after == 'bounce' else 0.0)
        if jump_at is not None and s >= jump_at:
            p = p + np.array([1.0, 0.0, 0.0])
        if 0.0 <= s < s_contact and first is None and not (drop_ascent and s < s_apex + 0.05):
            first = len(frames)
        if 0.0 <= s < s_contact and (stop_at is None or s <= stop_at):
            last = len(frames)
        frames.append(f)
        positions.append(p)
    ns = np.mod(100000000 + np.array(frames, dtype=np.int64) * int(round(period * 1e9)), 1000000000)
    recording = Build_Recording('Frequency_200_Hz_Active_Filtering_OFF', 1, rate, False, ns, np.array(positions))
    return recording, first, last


def test_a_synthetic_throw_is_segmented_from_the_release_to_the_last_sample_before_the_floor():
    """
    Description:
        A throw with dropouts in flight is segmented from its release sample to its last sample before
        the floor contact (SPEC §7.3); the fit's acceleration is gravity; time comes from the frames.
    """

    recording, first, last = _Throw()
    flight, reason = Segment_Flight(recording, 'Identification')
    assert reason == '' and flight is not None
    assert (flight.First_Line, flight.Last_Line) == (first + 2, last + 2)
    assert flight.Acceleration == pytest.approx([0.0, 0.0, -9.81], abs=1e-6)
    assert flight.Set == 'Identification' and not flight.Adjacent_Ambiguous and flight.Outlier_Lines == []


def test_a_hidden_gap_in_a_flight_cuts_it_and_flags_it_adjacent():
    """
    Description:
        A hidden gap (a 1 m jump) in the descent starts a new piece: the flight ends at the sample before
        it, never across it, and is kept with Adjacent_Ambiguous set.
    """

    recording, first, _ = _Throw(jump_at=1.0)
    ambiguous = [step.Line for step in recording.Steps if step.Kind == 'ambiguous']
    flight, reason = Segment_Flight(recording, 'Validation')
    assert reason == '' and flight.Adjacent_Ambiguous
    assert len(ambiguous) == 1 and flight.Last_Line == ambiguous[0] - 1 and flight.First_Line == first + 2
    assert not any(flight.First_Line < line <= flight.Last_Line for line in ambiguous)


@pytest.mark.parametrize('throw, expected', [
    (dict(release_z=0.5, v=(0.0, -4.0, 3.0), s_contact=0.6825), 'no sample above 1.5 m'),
    (dict(g=12.0, s_contact=1.1105), 'a_z -12.00'),
    (dict(a_x=3.0), 'horizontal acceleration 3.00'),
    (dict(release_z=1.6, v=(0.0, -6.0, 0.5), s_contact=0.2305, after='rest', gap_after=0.05), 'duration'),
    (dict(drop_ascent=True), 'the apex is not inside the segment'),
    (dict(stop_at=1.0), 'reaches the start or the end of the recording')])
def test_each_acceptance_criterion_rejects_on_its_own(throw, expected):
    """
    Description:
        Each criterion of the acceptance has a throw that fails it and no other:
        no throw, a_z outside [-10.5, -9.0], a horizontal acceleration above 1.5 m/s^2, a duration below
        0.3 s, the apex outside the segment, the recording's end reached.
    """

    recording, _, _ = _Throw(**throw)
    flight, reason = Segment_Flight(recording, 'Identification')
    assert flight is None and expected in reason and '; ' not in reason


def test_too_few_samples_are_refused():
    """
    Description:
        A window of fewer than 6 samples holds no fit (3 parameters per axis need redundancy).
    """

    ns = 100000000 + np.arange(5, dtype=np.int64) * 5000000
    recording = Build_Recording('Frequency_200_Hz_Active_Filtering_OFF', 1, 200.0, False, ns, np.tile([0.0, 0.0, 1.6], (5, 1)))
    flight, reason = Segment_Flight(recording, 'Identification')
    assert flight is None and reason == 'fewer than 6 samples are left after trimming'


def test_the_90_flights_have_the_pinned_lines(recordings, segmented):
    """
    Description:
        All 90 recordings segment a flight at the shipped tolerance (acceptance >= 85), each with the
        first and last line of the F2 Part A measurement.
    """

    found = collections.defaultdict(dict)
    for recording, (_, flight, reason) in zip(recordings, segmented):
        assert flight is not None, (recording.File, reason)
        found[recording.Folder][recording.Experiment] = (flight.First_Line, flight.Last_Line)
    assert dict(found) == CONST_FLIGHT_LINES


def test_only_100_hz_on_experiment_10_is_adjacent_to_an_ambiguous_step(segmented):
    """
    Description:
        The cited hidden gap at line 556 follows 100 Hz ON Experiment 10's last sample (line 555): that
        flight, and no other, is flagged and kept.
    """

    adjacent = [(flight.Folder, flight.Experiment, flight.Last_Line) for _, flight, _ in segmented if flight.Adjacent_Ambiguous]
    assert adjacent == [('Frequency_100_Hz_Active_Filtering_ON', 10, 555)]


def test_every_flight_carries_the_set_of_its_recording(recordings, segmented):
    """
    Description:
        Each flight's Set is its recording's set from the split, so F3 can refuse validation flights.
    """

    sets = Assign_Sets([(recording.Folder, 'Experiment_%d.txt' % recording.Experiment) for recording in recordings])
    for recording, (set_name, flight, _) in zip(recordings, segmented):
        assert flight.Set == set_name == sets[(recording.Folder, 'Experiment_%d.txt' % recording.Experiment)]


def test_flight_time_comes_from_the_frame_counts(recordings, segmented):
    """
    Description:
        Every flight's samples carry their own frame index from 0, strictly increasing (a dropout is a
        missing frame), and Time = Frame / Rate exactly: irregular sampling, nothing resampled.
    """

    for recording, (_, flight, _) in zip(recordings, segmented):
        L = flight.First_Line - 2
        assert flight.Frame[0] == 0 and np.all(np.diff(flight.Frame) >= 1)
        assert np.array_equal(flight.Frame, recording.Frame[L:L + len(flight.Frame)] - recording.Frame[L])
        assert np.array_equal(flight.Time, flight.Frame * (1.0 / recording.Rate))


def test_the_tolerance_resegments_and_a_flight_cut_short_is_rejected(recordings):
    """
    Description:
        F3 re-segments through tolerance: 90 flights at 50 mm; at 10 mm 89, and
        200 Hz OFF Experiment 36 fails only the cut-short criterion (one later sample on line 1396 still
        fits while the ball is in the air; measured in F2 Part B).
    """

    assert sum(flight is not None for _, flight, _ in Segment_Flights(recordings, tolerance=0.05)) == 90
    tight = Segment_Flights(recordings, tolerance=0.01)
    rejected = [(recording.Folder, recording.Experiment, reason) for recording, (_, flight, reason) in zip(recordings, tight) if flight is None]
    assert rejected == [('Frequency_200_Hz_Active_Filtering_OFF', 36, 'cut short: 1 later sample(s) still fit, the first on line 1396')]


def test_the_stalls_and_outliers_inside_the_flights_are_listed(segmented):
    """
    Description:
        The 3 stalls inside flights are listed at their lines, and the 229 samples off the fit by more
        than 20 mm (in 43 flights) are listed, not removed.
    """

    stalls = {(flight.Folder, flight.Experiment): flight.Stall_Lines for _, flight, _ in segmented if flight.Stall_Lines}
    assert stalls == CONST_STALLS
    assert sum(len(flight.Outlier_Lines) for _, flight, _ in segmented) == 229
    assert sum(1 for _, flight, _ in segmented if flight.Outlier_Lines) == 43
