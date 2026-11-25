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
# File Name: ../test/Recording/test_core.py
## =========================================================================== ##

# System (Default)
import os
# Collections (Container datatypes)
import collections
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the recording reader under test
#   ../Recording/Core
import Recording.Core as Core
from Recording.Core import Build_Recording, Load_Recording, Load_Recordings, Parse_Recording_Name, Recording_Error_Cls
from Recording.Core import Assign_Sets, Transform_To_Robot
#   ../Configuration/Core - the transform structure
from Configuration.Core import Transform_Str

"""
Description:
    Initialization of constants.

    Synthetic cases are built from the rule of SPEC §7.3 (the expected frames, gaps and pieces are
    written out here); the real-data pins come from the F2 Part A scratch measurement of 2026-11-09
    (an independent script), never from the code under test.
"""
# A 200 Hz folder for synthetic recordings: 1 period = 5 000 000 ns.
#   Unit [string], [int]
CONST_FOLDER = 'Frequency_200_Hz_Active_Filtering_OFF'
CONST_PERIOD_NS = 5000000
# The 18 ambiguous steps of the 90 recordings (F2 Part A and addendum, 2026-11-09): (file, line, reason).
#   Unit [Set<Tuple<string, int, string>>]
CONST_AMBIGUOUS = {('Frequency_100_Hz_Active_Filtering_OFF/Experiment_5.txt', 596, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_OFF/Experiment_7.txt', 688, 'zero step'),
                   ('Frequency_100_Hz_Active_Filtering_OFF/Experiment_7.txt', 964, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_OFF/Experiment_10.txt', 412, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_ON/Experiment_2.txt', 558, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_ON/Experiment_3.txt', 448, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_ON/Experiment_5.txt', 665, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_ON/Experiment_6.txt', 721, 'continuity'),
                   ('Frequency_100_Hz_Active_Filtering_ON/Experiment_7.txt', 558, 'cited'),
                   ('Frequency_100_Hz_Active_Filtering_ON/Experiment_10.txt', 556, 'cited'),
                   ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_13.txt', 609, 'continuity'),
                   ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_13.txt', 1365, 'zero step'),
                   ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_23.txt', 1987, 'continuity'),
                   ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_25.txt', 2035, 'cited'),
                   ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_34.txt', 533, 'continuity'),
                   ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_46.txt', 1237, 'continuity'),
                   ('Frequency_200_Hz_Active_Filtering_ON/Experiment_7.txt', 1074, 'continuity'),
                   ('Frequency_300_Hz_Active_Filtering_OFF/Experiment_3.txt', 1307, 'continuity')}
# The identification set, per folder, as the F2 addendum computed it (2026-11-09, scratch s17_split.py,
# seed 20261109, before the owner's answer); every other recording is validation.
#   Unit [Dict<string, Tuple<int>>]
CONST_IDENTIFICATION = {'Frequency_100_Hz_Active_Filtering_OFF': (2, 3, 5, 8, 10),
                        'Frequency_100_Hz_Active_Filtering_ON': (1, 2, 6, 8, 10),
                        'Frequency_200_Hz_Active_Filtering_OFF': (3, 8, 11, 12, 13, 14, 19, 20, 22, 26, 27, 29, 31, 32, 33, 34, 35,
                                                                  38, 39, 41, 42, 44, 48, 49, 50),
                        'Frequency_200_Hz_Active_Filtering_ON': (1, 5, 6, 7, 8),
                        'Frequency_300_Hz_Active_Filtering_OFF': (1, 3, 7, 8, 9)}


def _Build(steps_ns, jumps=None, start_ns=100000000, rate=200.0):
    """
    Description:
        A synthetic recording: the Time column from start_ns and the given steps [ns] (wrapped at 1e9),
        positions along x by the given jumps [m] (default 1 mm per step).
    """

    ns = np.mod(start_ns + np.concatenate(([0], np.cumsum(steps_ns))), 1000000000).astype(np.int64)
    jumps = [0.001] * len(steps_ns) if jumps is None else jumps
    x = np.concatenate(([0.0], np.cumsum(jumps)))
    position = np.stack([x, np.zeros_like(x), np.full_like(x, 1.0)], axis=1)
    return Build_Recording(CONST_FOLDER, 1, rate, False, ns, position)


def _Kinds(recording):
    return [(step.Line, step.Kind) for step in recording.Steps]


def _Write(folder, name, text):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, 'w', encoding='ascii', newline='') as file:
        file.write(text)
    return path


@pytest.mark.parametrize('rate, period_ns', [(100.0, 10000000), (200.0, 5000000), (300.0, 3333333)])
def test_a_wrap_of_the_second_is_one_period(rate, period_ns):
    """
    Description:
        A step across the end of a second (the Time column falls back below the previous value) is one
        period, not a negative step or a gap, at each rate of the recordings (SPEC §4.2).
    """

    recording = _Build([period_ns], start_ns=999000000, rate=rate)
    assert recording.Wraps == 1
    assert list(recording.Frame) == [0, 1] and recording.Steps == []
    assert recording.Time[1] == pytest.approx(1.0 / rate, abs=1e-12)


@pytest.mark.parametrize('periods, is_gap', [(1.49, False), (1.50, True)])
def test_a_step_of_1_50_periods_is_a_gap_and_1_49_is_not(periods, is_gap):
    """
    Description:
        The gap bound of SPEC §7.3 on both sides: a step of exactly 1.5 nominal
        periods is a gap, one of 1.49 is not. The bound is applied to the raw ratio.
    """

    recording = _Build([int(round(periods * CONST_PERIOD_NS))])
    assert ((3, 'gap') in _Kinds(recording)) is is_gap


def test_time_rounds_half_periods_up():
    """
    Description:
        A step of 2.5 periods counts 3 frames (half up), not 2 as Python's and
        NumPy's round-half-to-even would give.
    """

    recording = _Build([int(2.5 * CONST_PERIOD_NS)])
    assert list(recording.Frame) == [0, 3]


@pytest.mark.parametrize('periods, is_irregular', [(1.04, False), (1.06, True)])
def test_a_step_more_than_5_percent_off_a_whole_period_is_irregular(periods, is_irregular):
    """
    Description:
        The irregular-step bound of SPEC §7.3 on both sides: 0.04 periods off a whole multiple is
        regular, 0.06 is irregular and reported.
    """

    recording = _Build([int(round(periods * CONST_PERIOD_NS))])
    assert ((3, 'irregular') in _Kinds(recording)) is is_irregular


def test_a_zero_step_is_ambiguous_and_starts_a_new_piece():
    """
    Description:
        A step of 0 frames is a hidden gap of whole seconds: reported as a zero
        step and as ambiguous, and the samples after it are a new piece of time.
    """

    recording = _Build([CONST_PERIOD_NS, 0, CONST_PERIOD_NS])
    assert (4, 'zero') in _Kinds(recording)
    assert [(step.Line, step.Reason) for step in recording.Steps if step.Kind == 'ambiguous'] == [(4, 'zero step')]
    assert list(recording.Frame) == [0, 1, 1, 2] and list(recording.Piece) == [0, 0, 1, 1]


@pytest.mark.parametrize('jump, is_ambiguous', [(0.069, False), (0.071, True)])
def test_the_continuity_rule_is_tested_on_both_sides(jump, is_ambiguous):
    """
    Description:
        One period (5 ms) allows 10 m/s * 0.005 s + 0.02 m = 0.070 m: a jump of
        0.069 m is motion, one of 0.071 m an ambiguous step that starts a new piece.
    """

    recording = _Build([CONST_PERIOD_NS], jumps=[jump])
    assert ((3, 'ambiguous') in _Kinds(recording)) is is_ambiguous
    assert list(recording.Piece) == ([0, 1] if is_ambiguous else [0, 0])


def test_the_names_give_rate_filtering_and_experiment():
    """
    Description:
        The rate and QTM's active filtering come from the folder name, the experiment from the file
        name (SPEC §9.2); another name is refused with a sentence naming it.
    """

    assert Parse_Recording_Name('Frequency_300_Hz_Active_Filtering_OFF', 'Experiment_10.txt') == (300.0, False, 10)
    assert Parse_Recording_Name('Frequency_100_Hz_Active_Filtering_ON', 'Experiment_7.txt') == (100.0, True, 7)
    with pytest.raises(Recording_Error_Cls) as error:
        Parse_Recording_Name('Frequency_100_Hz_Active_Filtering_ON', 'Experiment_7.csv')
    assert 'Experiment_7.csv' in str(error.value)


@pytest.mark.parametrize('text, expected', [('Time,X,Y,Z\n100,0.0,0.0,1.0\n', 'line 1: the header'),
                                            ('Time,Data_X,Data_Y,Data_Z\n100,0.0,1.0\n', 'line 2: 3 field(s)'),
                                            ('Time,Data_X,Data_Y,Data_Z\n100,0.0,abc,1.0\n', 'line 2:'),
                                            ('Time,Data_X,Data_Y,Data_Z\n1000000000.0,0.0,0.0,1.0\n', 'line 2: Time'),
                                            ('', 'the file is empty')])
def test_invalid_input_is_refused_naming_the_file_and_line(tmp_path, text, expected):
    """
    Description:
        A wrong header, a row without 4 numbers, a non-number, a Time outside [0, 1e9) ns and an empty
        file are each refused with the file and the line (SPEC §7.3).
    """

    path = _Write(str(tmp_path / CONST_FOLDER), 'Experiment_1.txt', text)
    with pytest.raises(Recording_Error_Cls) as error:
        Load_Recording(path)
    assert '%s/Experiment_1.txt' % CONST_FOLDER in str(error.value) and expected in str(error.value)


def test_every_problem_of_every_file_is_listed_at_once(tmp_path):
    """
    Description:
        Two broken files and a stray folder fail one load that names all three, and nothing is
        returned (SPEC §7.3: the run is refused).
    """

    folder = str(tmp_path / CONST_FOLDER)
    _Write(folder, 'Experiment_1.txt', 'Time,Data_X,Data_Y,Data_Z\n100,0.0,1.0\n')
    _Write(folder, 'Experiment_2.txt', 'Time,X,Y,Z\n100,0.0,0.0,1.0\n')
    os.makedirs(str(tmp_path / 'Frequency_x'))
    with pytest.raises(Recording_Error_Cls) as error:
        Load_Recordings(str(tmp_path))
    message = str(error.value)
    assert '3 problem(s)' in message and 'Experiment_1.txt: line 2' in message
    assert 'Experiment_2.txt: line 1' in message and 'Frequency_x' in message


def test_a_missing_folder_is_refused_naming_it(tmp_path):
    """
    Description:
        A recordings folder that does not exist is refused with a sentence naming it.
    """

    with pytest.raises(Recording_Error_Cls) as error:
        Load_Recordings(str(tmp_path / 'absent'))
    assert 'absent' in str(error.value) and 'does not exist' in str(error.value)


def test_a_cited_line_outside_its_file_is_refused(monkeypatch):
    """
    Description:
        A cited hidden gap that is not a step of its file is refused, so a mistyped entry of the cited
        list cannot pass unnoticed.
    """

    monkeypatch.setattr(Core, 'CONST_CITED_HIDDEN_GAPS', {(CONST_FOLDER, 1, 99): 'test'})
    with pytest.raises(Recording_Error_Cls) as error:
        _Build([CONST_PERIOD_NS, CONST_PERIOD_NS])
    assert 'line 99' in str(error.value)


def test_the_90_recordings_load_with_the_measured_counts(recordings):
    """
    Description:
        All 90 files load (10 / 10 / 50 / 10 / 10 per folder) with the counts measured in F2 Part A
        (SPEC §4.2): 184 000 samples, 1 409 wraps, 7 305 gaps, 2 zero steps, no irregular step,
        18 ambiguous steps.
    """

    per_folder = collections.Counter(recording.Folder for recording in recordings)
    assert per_folder == {'Frequency_100_Hz_Active_Filtering_OFF': 10, 'Frequency_100_Hz_Active_Filtering_ON': 10,
                          'Frequency_200_Hz_Active_Filtering_OFF': 50, 'Frequency_200_Hz_Active_Filtering_ON': 10,
                          'Frequency_300_Hz_Active_Filtering_OFF': 10}
    kinds = collections.Counter(step.Kind for recording in recordings for step in recording.Steps)
    assert sum(len(recording.Frame) for recording in recordings) == 184000
    assert sum(recording.Wraps for recording in recordings) == 1409
    assert kinds == {'gap': 7305, 'zero': 2, 'ambiguous': 18}


def test_the_ambiguous_steps_are_the_18_listed_with_their_reasons(recordings):
    """
    Description:
        The ambiguous steps are exactly the 13 of the continuity rule, the 2 zero steps and the 3 cited
        hidden gaps of the F2 addendum, each at its file and line; every cited
        entry is matched.
    """

    found = {(recording.File, step.Line, step.Reason.split(':')[0])
             for recording in recordings for step in recording.Steps if step.Kind == 'ambiguous'}
    assert found == CONST_AMBIGUOUS



def _Names(recordings):
    return [(recording.Folder, 'Experiment_%d.txt' % recording.Experiment) for recording in recordings]


def test_the_split_is_45_45_by_folder_without_overlap(recordings):
    """
    Description:
        Every recording is in exactly one set, and each folder is split in halves: 5 / 5, 5 / 5,
        25 / 25, 5 / 5, 5 / 5, 45 / 45 in all.
    """

    sets = Assign_Sets(_Names(recordings))
    assert set(sets) == set(_Names(recordings)) and set(sets.values()) == {'Identification', 'Validation'}
    counts = collections.Counter((folder, set_name) for (folder, _), set_name in sets.items())
    for folder, size in (('Frequency_100_Hz_Active_Filtering_OFF', 5), ('Frequency_100_Hz_Active_Filtering_ON', 5),
                         ('Frequency_200_Hz_Active_Filtering_OFF', 25), ('Frequency_200_Hz_Active_Filtering_ON', 5),
                         ('Frequency_300_Hz_Active_Filtering_OFF', 5)):
        assert counts[(folder, 'Identification')] == size and counts[(folder, 'Validation')] == size, folder


def test_the_identification_set_is_the_pinned_list(recordings):
    """
    Description:
        The identification set equals the 45 names computed in the F2 addendum before the owner's
        answer; a change of the seed or the algorithm fails here, so a re-split after F3 has seen
        the flights cannot pass unnoticed.
    """

    sets = Assign_Sets(_Names(recordings))
    found = {folder: tuple(sorted(int(name[11:-4]) for (name_folder, name), set_name in sets.items()
                                  if name_folder == folder and set_name == 'Identification'))
             for folder in CONST_IDENTIFICATION}
    assert found == CONST_IDENTIFICATION


def test_the_split_depends_on_the_names_only(tmp_path):
    """
    Description:
        Recordings with the real names and other contents are assigned the same sets: the split reads
        no quantity of the data.
    """

    folder = 'Frequency_100_Hz_Active_Filtering_OFF'
    for n in range(1, 11):
        _Write(str(tmp_path / folder), 'Experiment_%d.txt' % n, 'Time,Data_X,Data_Y,Data_Z\n%d,%d.0,0.0,1.0\n' % (n * 1000, n))
    sets = Assign_Sets(_Names(Load_Recordings(str(tmp_path))))
    assert sorted(int(name[11:-4]) for (_, name), set_name in sets.items() if set_name == 'Identification') == list(CONST_IDENTIFICATION[folder])


def test_the_transform_maps_the_measured_descent_point():
    """
    Description:
        The assumed transform of SPEC §7.3 maps the measured descent point (0.827, -4.170) m at
        z = 0.8607 m onto (0, -0.5, 0.8607) m in the robot frame.
    """

    transform = Transform_Str(Orientation=[0.0, 0.0, np.pi], Position=[0.827, -4.670, 0.0])
    assert Transform_To_Robot(np.array([0.827, -4.170, 0.8607]), Transform_Str=transform) == pytest.approx(np.array([[0.0, -0.5, 0.8607]]), abs=1e-12)


def test_the_transform_rotates_before_it_translates():
    """
    Description:
        p_robot = R p + Position: (1, 0, 0) under Rz(180 deg) and (1, 2, 3) m is (0, 2, 3) m; translating
        first would give (-2, -2, 3) m.
    """

    transform = Transform_Str(Orientation=[0.0, 0.0, np.pi], Position=[1.0, 2.0, 3.0])
    assert Transform_To_Robot(np.array([[1.0, 0.0, 0.0]]), Transform_Str=transform) == pytest.approx(np.array([[0.0, 2.0, 3.0]]), abs=1e-12)
