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
# File Name: ../test/Legacy/test_reference.py
## =========================================================================== ##

# System (Default)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the 2022 test reference (test/Legacy/, put on sys.path by test/conftest.py)
#   ../test/Legacy/Reference
from Reference import CONST_LEGACY_BASELINE_DIR, CONST_LEGACY_INPUT_DIR, Generate_Legacy_Trajectory, Legacy_Error_Cls, \
    Read_Baseline_File, Read_Legacy_Inputs, Run_Legacy_Throw

"""
Description:
    Initialization of constants.

    F1's acceptance (SPEC §2.1, §10): each of the 12 files of Data/Legacy/Baseline/ reproduced throw
    by throw, through src/Geometry and src/Flight, within 1e-9 m of its %.10e text.
    The values compared are the baseline's (measured on the 0e59db2 code, SPEC §4.1),
    never the reference's own.
"""
# The methods of the 2022 inputs and the key states, as the baseline file names spell them.
#   Unit [Tuple<string>]
CONST_METHODS = ('Midpoint', 'Heun', 'Second-Runge-Kutta', 'Fourth-Runge-Kutta')
CONST_KINDS = ('Last', 'Mean', 'Mean_Opt')
# The tolerance per throw (SPEC §10).
#   Unit [m]
CONST_TOLERANCE = 1e-9
# The throws of the per-commit tier: the first 50 of each file (the full 1000 are marked slow).
#   Unit [int]
CONST_FAST_THROWS = 50
# The failures a message lists.
#   Unit [int]
CONST_LISTED = 10

CONST_CASES = [(m, k) for m in CONST_METHODS for k in CONST_KINDS]
CONST_IDS = ['%s_%s' % case for case in CONST_CASES]


def _Check_Baseline(method, kind, n_throws):
    name = '%s_%s.txt' % (method, kind)
    input_name = 'Experiment_1_%s_DRAG_False.txt' % method
    inputs = Read_Legacy_Inputs(os.path.join(CONST_LEGACY_INPUT_DIR, input_name))
    baseline = Read_Baseline_File(os.path.join(CONST_LEGACY_BASELINE_DIR, name))
    failures = []
    for throw in range(n_throws):
        try:
            value = Run_Legacy_Throw(inputs, throw, method, kind).Error
        except Legacy_Error_Cls as error:
            failures.append('throw %d (row %d of %s): %s' % (throw, throw + 2, input_name, error))
            continue
        delta = abs(value - baseline[throw])
        if not delta <= CONST_TOLERANCE:
            failures.append('throw %d (row %d of %s): reference %.10e m, baseline %.10e m, |d| %.1e m'
                            % (throw, throw + 2, input_name, value, baseline[throw], delta))
    assert not failures, '%s: %d of %d throws over %.0e m; first %d: %s' % (name, len(failures), n_throws, CONST_TOLERANCE,
                                                                          CONST_LISTED, '; '.join(failures[:CONST_LISTED]))


@pytest.mark.parametrize('method, kind', CONST_CASES, ids=CONST_IDS)
def test_the_first_50_throws_of_each_baseline_file_are_reproduced_within_1e9_m(method, kind):
    """
    Description:
        The per-commit tier of the acceptance: the first 50 throws of each file, all three key
        states and all four integrators.
    """

    _Check_Baseline(method, kind, CONST_FAST_THROWS)


@pytest.mark.slow
@pytest.mark.parametrize('method, kind', CONST_CASES, ids=CONST_IDS)
def test_each_baseline_file_is_reproduced_within_1e9_m_per_throw(method, kind):
    """
    Description:
        F1's acceptance: all 1000 throws of each of the 12 files.
    """

    _Check_Baseline(method, kind, 1000)


def test_the_reader_reads_each_input_as_1000_float32_throws_with_t_f_equal_to_r_n_z():
    """
    Description:
        The four inputs read as they are (SPEC §9.3): 1000 throws in float32, the first throw of
        the Midpoint file as written in it, and the t_f column equal to r_n_Z (the known defect).
    """

    for method in CONST_METHODS:
        inputs = Read_Legacy_Inputs(os.path.join(CONST_LEGACY_INPUT_DIR, 'Experiment_1_%s_DRAG_False.txt' % method))
        assert inputs.R_0.shape == inputs.R_N.shape == (1000, 3) and inputs.Flight_Time_Input.shape == (1000,)
        assert inputs.R_0.dtype == inputs.R_N.dtype == inputs.Flight_Time_Input.dtype == np.float32
        assert np.array_equal(inputs.Flight_Time_Input, inputs.R_N[:, 2])
    inputs = Read_Legacy_Inputs(os.path.join(CONST_LEGACY_INPUT_DIR, 'Experiment_1_Midpoint_DRAG_False.txt'))
    assert np.array_equal(inputs.R_0[0], np.float32([-1.6610495, 6.1329746, 0.8721319]))
    assert np.array_equal(inputs.R_N[0], np.float32([-0.54781073, 0.48584467, 0.4566355]))


@pytest.mark.parametrize('change, message', [('header', 'the header is'), ('row', '999 rows, not 1000'),
                                             ('t_f', 't_f 0.5 is not r_n_Z 0.4566355')],
                         ids=['wrong-header', 'missing-row', 't_f-not-r_n_Z'])
def test_the_reader_refuses_an_input_that_differs_from_spec_9_3(tmp_path, change, message):
    """
    Description:
        A changed header, a missing row and a t_f that is not r_n_Z are each refused, naming the
        file and the problem: the reference must not run on inputs the baseline was not made from.
    """

    with open(os.path.join(CONST_LEGACY_INPUT_DIR, 'Experiment_1_Midpoint_DRAG_False.txt'), newline='') as handle:
        lines = handle.read().split('\n')
    if change == 'header':
        lines[0] = lines[0].replace('t_f', 'T_f')
    elif change == 'row':
        del lines[500]
    else:
        lines[1] = lines[1].rsplit(',', 1)[0] + ',0.5'
    path = tmp_path / 'Experiment_1_Midpoint_DRAG_False.txt'
    path.write_text('\n'.join(lines))
    with pytest.raises(Legacy_Error_Cls, match=message) as error:
        Read_Legacy_Inputs(str(path))
    assert str(path) in str(error.value)


def test_the_frame_count_is_rounded_from_float64_as_numpy_1_did():
    """
    Description:
        t_f = 0.9824324 (float32) is 363.4999973 frames of 370 Hz in float64, which rounds to 363,
        but rounds to 364 if t_f / T is computed in float32, as NumPy 2 does (NEP 50). None of the
        4000 legacy inputs is that close to a tie (the closest is 1.25e-5 frames away, measured
        2026-11-09), so the baseline cannot show this cast; this case does. The expectation is
        Python's float64 arithmetic, the promotion of NumPy 1.x.
    """

    t_f = np.float32(0.9824324)
    expected = int(round(float(t_f) / (1 / 370)))
    assert expected == 363
    trajectory = Generate_Legacy_Trajectory(np.float32([0.0, 6.0, 1.0]), np.float32([0.0, 0.5, 0.9]), t_f, 'RK4')
    assert trajectory.shape == (expected, 6)
