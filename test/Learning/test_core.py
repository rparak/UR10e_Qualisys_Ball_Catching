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
# File Name: ../test/Learning/test_core.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the inputs of a learned predictor and the normalisation under test
#   ../Learning/Core
from Learning.Core import (Prefix_Features, Normalisation_Str, Write_Normalisation, Read_Normalisation, Features_Error_Cls,
                           Chebyshev_Fit, Chebyshev_Series, CONST_SAMPLE_FEATURES, CONST_BASIS_DEGREE, CONST_HORIZON)
#   ../Flight/Core, ../Dataset/Core - the simulated truth and its floor time
from Flight.Core import Flight_Model_Str, Simulate
from Dataset.Core import Floor_Time

"""
Description:
    Initialization of constants.

    The inputs of a prefix (SPEC §7.8): expectations are written out from the
    definitions, never taken from Prefix_Features.
"""
# An irregular 200 Hz prefix of 5 samples: one frame lost before the third, three before the fifth.
#   Unit [Vector<float> 1x5, s], [Matrix<float> 5x3, m]
CONST_TIME = np.array([0, 1, 3, 4, 8]) / 200.0
CONST_POSITION = np.array([[0.00, -5.00, 1.00], [0.01, -4.98, 1.03], [0.03, -4.94, 1.08], [0.04, -4.92, 1.10], [0.08, -4.84, 1.17]])
# A normalisation with distinct values per axis.
#   Unit [Normalisation_Str]
CONST_NORMALISATION = Normalisation_Str(Relative_Std=np.array([0.5, 2.0, 0.25]), Velocity_Mean=np.array([0.0, 4.0, 5.0]),
                                        Velocity_Std=np.array([1.0, 0.5, 2.0]), Output_Mean=np.zeros(3), Output_Scale=np.array([1.0, 2.5, 1.0]))


# The flight model: g = 9.81 m/s^2 along -z and F3's k (SPEC §4.2); four release states across the throws'
# range (5.9-7.5 m/s, SPEC §8.2) and three last observed times.
#   Unit [Flight_Model_Str], [Matrix<float> 4x6, m, m/s], [Tuple<float>, s]
CONST_MODEL = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.00930496257)
CONST_RELEASES = np.array([[-0.15, -4.78, 1.17, 0.1, 4.3, 5.2], [1.0, -5.6, 1.4, -0.6, 4.9, 5.6], [-1.2, -4.3, 1.05, 0.5, 3.6, 4.9],
                           [0.3, -6.0, 1.5, -0.1, 5.6, 4.6]])
CONST_LAST_TIMES = (0.03, 0.2, 0.5)


def _One(time, position, rate=200.0):
    return Prefix_Features(time[None, :], position[None, :, :], np.array([len(time)]), np.array([rate]), Normalisation_Str=CONST_NORMALISATION)


def test_a_sample_enters_at_its_own_time_relative_to_the_last_sample():
    """
    Description:
        SPEC §2.7, §7.8: each sample of an irregular prefix enters as observed. The fifth sample (t_0) has time 0
        and position 0; the third is 25 ms before t_0 (in units of 0.1 s), 0.05 / 0.10 / 0.09 m before the last
        position in units of the per-axis scale, one frame was lost before it (log 2), its step is 10 ms (in
        units of 0.01 s) and its finite-difference velocity is (2, 4, 5) m/s, normalised; the frame period is
        5 ms in units of 0.01 s.
    """

    sequence, context, last = _One(CONST_TIME, CONST_POSITION)
    assert sequence.shape == (1, 5, CONST_SAMPLE_FEATURES) and np.array_equal(last, CONST_POSITION[None, 4])
    assert np.allclose(sequence[0, 4, :4], 0.0, atol=1e-15) and context.shape == (1, 1) and context[0, 0] == pytest.approx(0.5)
    assert sequence[0, 2, 0] == pytest.approx(-0.25) and sequence[0, 0, 0] == pytest.approx(-0.4)
    assert np.allclose(sequence[0, 2, 1:4], [-0.05 / 0.5, -0.10 / 2.0, -0.09 / 0.25])
    assert np.allclose(sequence[0, 2, 4:7], [(2.0 - 0.0) / 1.0, (4.0 - 4.0) / 0.5, (5.0 - 5.0) / 2.0])
    assert np.allclose(sequence[0, :, 7], np.log1p([0.0, 0.0, 1.0, 0.0, 3.0])) and np.allclose(sequence[0, :, 8], [0.0, 0.5, 1.0, 0.5, 2.0])
    assert np.array_equal(sequence[0, 0, 4:7], sequence[0, 1, 4:7])


def test_the_inputs_hold_no_absolute_position():
    """
    Description:
        A predictor is given no absolute position. A prefix moved by (3, -2, 0.5) m has the
        same inputs (within the rounding of the subtraction); only the returned last position moves.
    """

    shift = np.array([3.0, -2.0, 0.5])
    sequence, context, last = _One(CONST_TIME, CONST_POSITION)
    moved_sequence, moved_context, moved_last = _One(CONST_TIME, CONST_POSITION + shift)
    assert np.allclose(moved_sequence, sequence, rtol=0.0, atol=1e-12) and np.array_equal(moved_context, context)
    assert np.allclose(moved_last - last, shift)


def test_padding_after_a_prefix_does_not_change_its_inputs():
    """
    Description:
        A batch pads the shorter prefix after its samples: its inputs equal those of the prefix alone bit for bit
        and the padded entries are 0, whatever values the padding holds (SPEC §2.13.4: never resampled or filled).
    """

    alone, _, _ = _One(CONST_TIME[:3], CONST_POSITION[:3])
    time = np.stack((CONST_TIME, np.concatenate((CONST_TIME[:3], [9.0, 9.5]))))
    position = np.stack((CONST_POSITION, np.concatenate((CONST_POSITION[:3], np.full((2, 3), 7.0)))))
    both, _, last = Prefix_Features(time, position, np.array([5, 3]), np.array([200.0, 200.0]), Normalisation_Str=CONST_NORMALISATION)
    assert np.array_equal(both[1, :3], alone[0]) and np.array_equal(both[1, 3:], np.zeros((2, CONST_SAMPLE_FEATURES)))
    assert np.array_equal(last[1], CONST_POSITION[2])


def test_a_prefix_of_one_sample_is_refused():
    """
    Description:
        A velocity needs two samples: a prefix of 1 is refused, naming the count; 2 samples are the bound and load.
    """

    with pytest.raises(Features_Error_Cls, match='1 sample'):
        Prefix_Features(CONST_TIME[None, :], CONST_POSITION[None, :, :], np.array([1]), np.array([200.0]), Normalisation_Str=CONST_NORMALISATION)
    sequence, _, _ = Prefix_Features(CONST_TIME[None, :], CONST_POSITION[None, :, :], np.array([2]), np.array([200.0]),
                                     Normalisation_Str=CONST_NORMALISATION)
    assert np.array_equal(sequence[0, 2:], np.zeros((3, CONST_SAMPLE_FEATURES)))


def test_the_normalisation_file_round_trips_and_is_deterministic(tmp_path):
    """
    Description:
        normalisation.json (SPEC §9.6) holds every array exactly: read back, each equals the written one bit for
        bit, a 3 x 7 output array included, and a second write gives the same bytes.
    """

    normalisation = Normalisation_Str(Relative_Std=np.array([0.1, 1.0 / 3.0, 0.7]), Velocity_Mean=np.array([0.1, -4.2, 5.0]),
                                      Velocity_Std=np.array([1.1, 0.5, 2.0]), Output_Mean=np.arange(21.0).reshape(3, 7) / 7.0,
                                      Output_Scale=np.full((3, 7), 1e-3))
    Write_Normalisation(str(tmp_path / 'a.json'), normalisation)
    Write_Normalisation(str(tmp_path / 'b.json'), normalisation)
    assert (tmp_path / 'a.json').read_bytes() == (tmp_path / 'b.json').read_bytes()
    read = Read_Normalisation(str(tmp_path / 'a.json'))
    for name in ('Relative_Std', 'Velocity_Mean', 'Velocity_Std', 'Output_Mean', 'Output_Scale'):
        assert np.array_equal(getattr(read, name), getattr(normalisation, name)), name


@pytest.mark.parametrize('old, new, reason', [('"format_version": 1', '"format_version": 2', 'format version 1'),
                                              ('"velocity_std"', '"velocity_sd"', 'lacks velocity_std'),
                                              ('"relative_std": [\n    0.5,', '"relative_std": [\n    0.0,', 'three positive numbers')])
def test_a_normalisation_file_of_another_version_or_shape_is_refused(tmp_path, old, new, reason):
    """
    Description:
        The reader refuses another format version, a missing entry and an input scale of 0, naming the file and
        the reason (a structured format carries a version).
    """

    path = tmp_path / 'normalisation.json'
    Write_Normalisation(str(path), CONST_NORMALISATION)
    text = path.read_text(encoding='ascii')
    assert old in text
    path.write_text(text.replace(old, new), encoding='ascii')
    with pytest.raises(Features_Error_Cls, match=reason):
        Read_Normalisation(str(path))


def test_the_basis_of_degree_6_holds_the_simulated_truth_within_half_a_millimetre():
    """
    Description:
        The basis decoder's degree is the smallest whose own approximation error is below
        0.5 mm. On the simulated truth of four throws from three last observed times each (Simulate, every
        1/600 s over the 2 s span), the least-squares series of degree CONST_BASIS_DEGREE = 6 is within 0.5 mm of
        the truth over the remaining flight (to the floor) on every one, and a series of degree 5 (NumPy's fit)
        is not (F8 Part B measured 0.20 and 0.97 mm on 4 377 validation prefixes).
    """

    assert CONST_BASIS_DEGREE == 6
    tau = np.arange(0.0, CONST_HORIZON + 1e-9, 1.0 / 600.0)
    errors, lower = [], []
    for release in CONST_RELEASES:
        floor = Floor_Time(release[None], np.array([1.0]), Flight_Model_Str=CONST_MODEL)[0]
        for t_0 in CONST_LAST_TIMES:
            truth = Simulate(release, t_0 + tau, Flight_Model_Str=CONST_MODEL)[:, :3]
            values, remaining = truth - truth[0], tau <= floor - t_0
            series, _ = Chebyshev_Series(Chebyshev_Fit(tau, values), tau)
            errors.append(np.linalg.norm(series - values, axis=1)[remaining].max())
            s = 2.0 * tau / CONST_HORIZON - 1.0
            fifth = np.polynomial.chebyshev.chebval(s, np.polynomial.chebyshev.chebfit(s, values, 5)).T
            lower.append(np.linalg.norm(fifth - values, axis=1)[remaining].max())
    assert max(errors) < 0.5e-3 and max(lower) > 0.5e-3, (max(errors), max(lower))


def test_a_seeds_epochs_are_read_back_and_another_file_is_refused(tmp_path):
    """
    Description:
        epochs.csv as Learning.Training writes it (the columns of CONST_EPOCH_COLUMNS, '%.9f') read back for the figures;
        a file with another header is refused.
    """

    from Learning.Core import Read_Epochs, CONST_EPOCH_COLUMNS
    assert CONST_EPOCH_COLUMNS == ('epoch', 'train_loss_m', 'validation_loss_m', 'validation_position_m')
    (tmp_path / 'epochs.csv').write_text('epoch,train_loss_m,validation_loss_m,validation_position_m\n1,0.259228576,0.071323588,0.071323588\n'
                                         '2,0.068435826,0.057255267,0.057255267\n', encoding='ascii')
    assert np.allclose(Read_Epochs(str(tmp_path / 'epochs.csv')), [[1, 0.259228576, 0.071323588, 0.071323588], [2, 0.068435826, 0.057255267,
                                                                                                                0.057255267]])
    (tmp_path / 'other.csv').write_text('epoch,loss\n1,0.2\n', encoding='ascii')
    with pytest.raises(Features_Error_Cls, match='does not have the columns epoch, train_loss_m'):
        Read_Epochs(str(tmp_path / 'other.csv'))
