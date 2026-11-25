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
# File Name: ../test/Recording/test_report.py
## =========================================================================== ##

# System (Default)
import os
# CSV (CSV file reading and writing)
import csv
# Collections (Container datatypes)
import collections
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the writers under test
#   ../Recording/Report, Core, Segmentation; ../Configuration/Core
from Recording.Report import (Write_Recording_Summary, Write_Recording_Steps, Summary_Lines, Write_Identification_Summary,
                              Write_Identification_Flights, Append_Ledger, Count_Ledger)
from Recording.Identification import Identification_Str, Variant_Str, Fitted_Flight_Str
from Flight.Core import Flight_Model_Str
from Recording.Segmentation import Flight_Str
from Configuration.Core import Transform_Str

"""
Description:
    Initialization of constants.

    The columns are written out from SPEC §9.4; the counts come from the F2 Part A measurement.
"""
# The summary's header and the steps' header (SPEC §9.4).
#   Unit [string]
CONST_SUMMARY_HEADER = ('file,rate_hz,filtering,samples,wraps,gaps,zero_steps,irregular_steps,ambiguous_steps,set,segmented,'
                        'reason,first_line,last_line,flight_samples,duration_s,coverage,a_x,a_y,a_z,residual_rms_m,residual_max_m,'
                        'outliers,stalls,adjacent_ambiguous,first_x_robot_m,first_y_robot_m,first_z_robot_m,last_x_robot_m,'
                        'last_y_robot_m,last_z_robot_m')
CONST_STEPS_HEADER = 'file,line,kind,reason,ratio,frames,jump_m'
# The identification's headers (SPEC §9.8).
#   Unit [string]
CONST_IDENTIFICATION_HEADER = ('kind,variant,session,tolerance_m,trim_s,without,flights_segmented,flights,samples,gravity_m_s2,'
                               'drag_per_m,drag_bootstrap_sd_per_m,drag_half_range_per_m,drag_std_per_m,rms_median_m,max_median_m,'
                               'floor_median_m,edge_rms_median_m,interior_rms_median_m,evaluation')
CONST_IDENTIFICATION_FLIGHTS_HEADER = ('kind,trim_s,file,session,set,first_line,last_line,samples,x_m,y_m,z_m,v_x_m_s,v_y_m_s,'
                                       'v_z_m_s,rms_m,max_m,floor_m,edge_rms_m,interior_rms_m')
CONST_LEDGER_HEADER = 'evaluation,commit,dirty,drag_per_m,drag_std_per_m'
# The assumed transform of SPEC §7.3, in [rad] and [m].
#   Unit [Transform_Str]
CONST_TRANSFORM = Transform_Str(Orientation=[0.0, 0.0, 3.141592653589793], Position=[0.827, -4.670, 0.0])


@pytest.fixture(scope='module')
def data(recordings, segmented):
    """
    Description:
        The 90 recordings and their flights (the session fixtures of test/conftest.py).
    """

    return recordings, segmented


def _Rows(path):
    with open(path, 'r', encoding='ascii', newline='') as file:
        return list(csv.reader(file))


def test_the_summary_has_one_row_per_recording_in_order(data, tmp_path):
    """
    Description:
        The summary holds the header of SPEC §9.4 and 90 rows, ordered by folder and experiment number,
        each with its set and its flight's lines.
    """

    recordings, segmented = data
    path = str(tmp_path / 'summary.csv')
    Write_Recording_Summary(path, recordings, segmented, Transform_Str=CONST_TRANSFORM)
    rows = _Rows(path)
    assert ','.join(rows[0]) == CONST_SUMMARY_HEADER and len(rows) == 91
    assert [row[0] for row in rows[1:3]] == ['Frequency_100_Hz_Active_Filtering_OFF/Experiment_1.txt',
                                            'Frequency_100_Hz_Active_Filtering_OFF/Experiment_2.txt']
    assert rows[-1][0] == 'Frequency_300_Hz_Active_Filtering_OFF/Experiment_10.txt'
    assert collections.Counter(row[9] for row in rows[1:]) == {'Identification': 45, 'Validation': 45}
    assert all(row[10] == 'true' for row in rows[1:])


def test_the_summary_places_the_flight_ends_in_the_robot_frame(data, tmp_path):
    """
    Description:
        The first sample of a flight appears in the robot frame by SPEC §7.3's formula, written out
        here: Rz(180 deg) maps (x, y, z) to (-x, -y, z), then (0.827, -4.670, 0) m is added.
    """

    recordings, segmented = data
    path = str(tmp_path / 'summary.csv')
    Write_Recording_Summary(path, recordings, segmented, Transform_Str=CONST_TRANSFORM)
    row = _Rows(path)[1]
    x, y, z = recordings[0].Position[int(row[12]) - 2]
    assert [float(v) for v in row[25:28]] == pytest.approx([-x + 0.827, -y - 4.670, z], abs=1e-6)


def test_the_steps_file_lists_every_reported_step_with_its_line(data, tmp_path):
    """
    Description:
        The steps file lists the 7 305 gaps, 2 zero steps and 18 ambiguous steps with file and line;
        a cited hidden gap carries its reason.
    """

    recordings, _ = data
    path = str(tmp_path / 'steps.csv')
    Write_Recording_Steps(path, recordings)
    rows = _Rows(path)
    assert ','.join(rows[0]) == CONST_STEPS_HEADER
    assert collections.Counter(row[2] for row in rows[1:]) == {'gap': 7305, 'zero': 2, 'ambiguous': 18}
    cited = [row for row in rows if row[:3] == ['Frequency_200_Hz_Active_Filtering_OFF/Experiment_25.txt', '2035', 'ambiguous']]
    assert len(cited) == 1 and cited[0][3].startswith('cited: at rest for 443 steps')


def test_the_writers_are_deterministic(data, tmp_path):
    """
    Description:
        Writing the same data twice gives the same bytes (no timestamp, no absolute path, fixed formats).
    """

    recordings, segmented = data
    for name in ('a', 'b'):
        Write_Recording_Summary(str(tmp_path / ('%s_summary.csv' % name)), recordings, segmented, Transform_Str=CONST_TRANSFORM)
        Write_Recording_Steps(str(tmp_path / ('%s_steps.csv' % name)), recordings)
    for kind in ('summary', 'steps'):
        assert (tmp_path / ('a_%s.csv' % kind)).read_bytes() == (tmp_path / ('b_%s.csv' % kind)).read_bytes()


def test_the_printed_summary_counts_steps_and_flights(data):
    """
    Description:
        The printed summary gives the counts of the acceptance: 90 files, 7 305 gaps, 2 zero steps,
        18 ambiguous steps by reason, 90 flights, none excluded, one adjacent-flagged, 45 / 45.
    """

    lines = Summary_Lines(*data)
    for expected in ('files loaded: 90 (184000 samples)', 'gaps: 7305', 'zero steps: 2', 'irregular steps: 0',
                     'ambiguous steps: 18 (rule 13, zero 2, cited 3)', 'flights segmented: 90 of 90', 'not segmented: 0',
                     'excluded (an ambiguous step inside): 0', 'adjacent-flagged: 1',
                     '  Frequency_100_Hz_Active_Filtering_ON/Experiment_10.txt',
                     'Identification: 45 recordings, 45 flights', 'Validation: 45 recordings, 45 flights'):
        assert expected in lines, expected


def _Fitted(experiment, set_name, scale):
    """
    Description:
        A constructed fitted flight: frames 0-3 and 6-9 at 100 Hz, a residual of known size.
    """

    frame = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    flight = Flight_Str(Folder='Synthetic', Experiment=experiment, Rate=100.0, Set=set_name, First_Line=10, Last_Line=20, Frame=frame,
                        Position=np.zeros((8, 3)))
    residual = scale * np.column_stack((np.sin(frame), np.cos(frame), 0.5 * np.ones(8)))
    return Fitted_Flight_Str(Flight=flight, X_0=np.arange(6.0), Residual=residual)


def test_the_identification_files_have_their_rows_and_are_deterministic(tmp_path):
    """
    Description:
        The summary holds the identified row, the sensitivity rows and per trim the validation rows of
        all flights and of each session, each with the evaluation number; the flights file one row per
        fitted flight. Written twice, the bytes are equal.
    """

    fitted = [_Fitted(1, 'Identification', 0.001), _Fitted(2, 'Identification', 0.002)]
    validation = [_Fitted(3, 'Validation', 0.003), _Fitted(4, 'Validation', 0.004)]
    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0093, Drag_Std=0.00025)
    identification = Identification_Str(Flight_Model_Str=model, Drag_Bootstrap_Sd=0.00015, Drag_Half_Range=0.0002,
                                        Variants=[Variant_Str(Name='identified', Tolerance=0.02, Trim=0.04, Flights_Segmented=90, Flights=2,
                                                              Samples=16, Drag=0.0093, Fitted=fitted),
                                                  Variant_Str(Name='trim 0 ms', Tolerance=0.02, Trim=0.0, Flights_Segmented=90, Flights=2,
                                                              Samples=16, Drag=0.0091, Fitted=fitted)])
    sessions = {'Synthetic/Experiment_%d.txt' % i: 'Synthetic, %d samples' % (1000 if i % 2 else 2000) for i in (1, 2, 3, 4)}
    for name in ('a', 'b'):
        Write_Identification_Summary(str(tmp_path / ('%s_summary.csv' % name)), identification, [(0.04, validation), (0.0, validation)],
                                     sessions, 7)
        Write_Identification_Flights(str(tmp_path / ('%s_flights.csv' % name)),
                                     [('identification', 0.04, fitted), ('validation', 0.04, validation)], sessions)
    for kind in ('summary', 'flights'):
        assert (tmp_path / ('a_%s.csv' % kind)).read_bytes() == (tmp_path / ('b_%s.csv' % kind)).read_bytes()
    rows = _Rows(str(tmp_path / 'a_summary.csv'))
    assert ','.join(rows[0]) == CONST_IDENTIFICATION_HEADER
    assert [row[:3] for row in rows[1:]] == [['identified', 'identified', ''], ['sensitivity', 'trim 0 ms', ''],
                                             ['validation', 'trim 40 ms', 'all'], ['validation', 'trim 40 ms', 'Synthetic, 1000 samples'],
                                             ['validation', 'trim 40 ms', 'Synthetic, 2000 samples'], ['validation', 'trim 0 ms', 'all'],
                                             ['validation', 'trim 0 ms', 'Synthetic, 1000 samples'],
                                             ['validation', 'trim 0 ms', 'Synthetic, 2000 samples']]
    assert rows[1][10:14] == ['0.0093', '0.00015', '0.0002', '0.00025'] and rows[2][13] == ''
    assert [row[-1] for row in rows[3:]] == ['7'] * 6 and rows[4][7] == '1'
    # Flight 3: |residual|^2 = 0.003^2 (1 + 0.25) at every sample, so its RMS is 0.003 sqrt(1.25) m (all: flights 3, 4; row 4: session 1000).
    assert float(rows[4][14]) == pytest.approx(0.003 * np.sqrt(1.25), abs=1e-11)  # 9 significant digits written
    flights = _Rows(str(tmp_path / 'a_flights.csv'))
    assert ','.join(flights[0]) == CONST_IDENTIFICATION_FLIGHTS_HEADER and len(flights) == 5
    assert flights[3][:5] == ['validation', '0.04', 'Synthetic/Experiment_3.txt', 'Synthetic, 1000 samples', 'Validation']


def test_the_ledger_counts_every_run(tmp_path):
    """
    Description:
        The ledger starts with its header; each run appends one numbered row, and the count is the
        number of rows.
    """

    path = str(tmp_path / 'ledger.csv')
    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0093, Drag_Std=0.00025)
    assert Count_Ledger(path) == 0
    assert Append_Ledger(path, 'abc123', False, model) == 1 and Append_Ledger(path, 'abc124', True, model) == 2
    assert Count_Ledger(path) == 2
    rows = _Rows(path)
    assert ','.join(rows[0]) == CONST_LEDGER_HEADER and rows[2] == ['2', 'abc124', 'true', '0.0093', '0.00025']
