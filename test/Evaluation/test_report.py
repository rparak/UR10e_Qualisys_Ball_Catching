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
# File Name: ../test/Evaluation/test_report.py
## =========================================================================== ##

# CSV (CSV file reading and writing)
import csv
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the writers under test
#   ../Evaluation/Report
from Evaluation.Report import (Append_Test_Ledger, Count_Test_Ledger, Prediction_Record_Str, Read_Coverage, Read_Per_Flight, Read_Per_Horizon,
                               Read_Predictions, Report_Error_Cls, Write_Comparison_Table, Write_Coverage, Write_Per_Flight,
                               Write_Per_Horizon, Write_Predictions, Write_Summary)
#   ../Evaluation/Core - the results written
from Evaluation.Core import Flight_Result_Str

"""
Description:
    Initialization of constants.

    The formats of SPEC §9.7 on constructed results: the columns and units written out here from the SPEC,
    never read from the writer.
"""
# The header of per_flight.csv (SPEC §9.7).
#   Unit [List<string>]
CONST_PER_FLIGHT_HEADER = ['flight', 'partition', 'dataset', 'n', 'prefix_duration_s', 'rate_hz', 'remainder_rms_mm', 'end_mismatch_ms',
                           'catch_point_error_mm', 'time_to_catch_error_ms', 'crossing', 'inference_ms', 'refusal', 'seed']


def _Read(path):
    with open(path, encoding='ascii', newline='') as file:
        return list(csv.reader(file))


def _Simulated():
    r = Flight_Result_Str(Flight='17', Partition='validation', Dataset='Default', N=20, Prefix_Duration=0.105, Rate=200.0, Remainder_RMS=0.0123,
                          End_Mismatch=-0.0042, Catch_Point_Error=0.0301, Time_To_Catch_Error=0.0015, Crossing='both', Inference_Time=0.055)
    r.Fixed = {'0.05': 0.011, 'end': 0.05}
    r.Ranges, r.Ranges_Observed = [(1, 0.014, 3), (6, 0.07, 2)], [(1, 0.013, 3)]
    return r


def _Real():
    r = Flight_Result_Str(Flight='Frequency_200_Hz_Active_Filtering_OFF/Experiment_2.txt', Partition='real', N=10, Prefix_Duration=0.045,
                          Rate=200.0, Remainder_RMS=0.02, End_Mismatch=0.01, Inference_Time=0.05)
    r.Ranges = [(0, 0.006, 4)]
    return r


def test_per_flight_writes_mm_and_ms_and_empty_catch_fields_on_real_flights(tmp_path):
    """
    Description:
        per_flight.csv: the SPEC §9.7 header; a simulated flight's values in mm and ms with '%.6f'; a real
        flight's catch-point, time-to-catch and crossing fields empty (SPEC §2.14); a refused prediction's metrics
        empty, its crossing 'refused' and its reason written.
    """

    path = tmp_path / 'per_flight.csv'
    Write_Per_Flight(str(path), [_Simulated(), _Real()])
    rows = _Read(path)
    assert rows[0] == CONST_PER_FLIGHT_HEADER and (path.read_bytes().count(b'\r'), len(rows)) == (0, 3)
    assert rows[1] == ['17', 'validation', 'Default', '20', '0.105000', '200', '12.300000', '-4.200000', '30.100000', '1.500000', 'both',
                       '55.000000', '', '']
    assert rows[2][8:11] == ['', '', ''] and rows[2][2] == ''
    refused = Flight_Result_Str(Flight='9', Partition='test', Dataset='Seed_1', N=40, Prefix_Duration=1.2, Rate=100.0, Crossing='refused',
                                Refusal='The predicted trajectory is below the floor height')
    Write_Per_Flight(str(path), [refused])
    assert _Read(path)[1][6:] == ['', '', '', '', 'refused', '', 'The predicted trajectory is below the floor height', '']


def test_per_horizon_writes_fixed_horizons_and_ranges_with_their_truth(tmp_path):
    """
    Description:
        per_horizon.csv: a simulated flight's fixed horizons against the exact truth without a sample count, its
        ranges against the exact truth and the observed positions with their counts, the last range closed
        ('[1.2, 2]'); a real flight's ranges against the recording.
    """

    path = tmp_path / 'per_horizon.csv'
    Write_Per_Horizon(str(path), [_Simulated(), _Real()])
    rows = [r[4:8] for r in _Read(path)[1:]]
    assert rows == [['0.05', 'exact', '11.000000', ''], ['end', 'exact', '50.000000', ''], ['[0.1, 0.2)', 'exact', '14.000000', '3'],
                    ['[1.2, 2]', 'exact', '70.000000', '2'], ['[0.1, 0.2)', 'observed', '13.000000', '3'], ['[0, 0.1)', 'recorded', '6.000000', '4']]


def test_per_horizon_writes_the_velocity_errors_and_the_training_seed(tmp_path):
    """
    Description:
        SPEC §7.12, §9.7: per_horizon.csv's columns F7 wrote keep their order, and after them
        come the velocity's direction error [deg] and speed error [m/s] and the training seed. A simulated
        flight's fixed horizon and its catch-surface row ('catch', with the catch-point error) carry them; the
        truth's end and the ranges do not; a real flight's rows have none (a recording holds positions only).
    """

    simulated = _Simulated()
    simulated.Seed = '3'
    simulated.Direction, simulated.Speed = {'0.05': np.radians(1.5), 'catch': np.radians(0.25)}, {'0.05': -0.02, 'catch': 0.004}
    path = tmp_path / 'per_horizon.csv'
    Write_Per_Horizon(str(path), [simulated, _Real()])
    rows = _Read(path)
    assert rows[0] == ['flight', 'partition', 'dataset', 'n', 'horizon', 'truth', 'position_error_mm', 'samples', 'direction_error_deg',
                       'speed_error_m_s', 'seed']
    assert [r[4:] for r in rows[1:4]] == [['0.05', 'exact', '11.000000', '', '1.500000', '-0.020000', '3'],
                                          ['end', 'exact', '50.000000', '', '', '', '3'],
                                          ['catch', 'exact', '30.100000', '', '0.250000', '0.004000', '3']]
    assert rows[4][4:] == ['[0.1, 0.2)', 'exact', '14.000000', '3', '', '', '3'] and rows[-1][8:] == ['', '', '']


def test_summary_writes_a_missing_statistic_as_an_empty_field(tmp_path):
    """
    Description:
        summary.csv: a range below the minimum of flights (NaN statistics) is written with its count and empty
        statistics; a spread row's count is a number, over the datasets and over the training seeds alike; the
        training seed follows the dataset.
    """

    path = tmp_path / 'summary.csv'
    nan = float('nan')
    Write_Summary(str(path), [{'partition': 'real', 'dataset': '', 'seed': '', 'n': 20, 'group': 'all', 'value': '', 'metric': 'range_1.2_2_mm',
                               'count': 9, 'mean': nan, 'median': nan, 'p90': nan, 'max': nan},
                              {'partition': 'test', 'dataset': 'spread_sd', 'seed': '', 'n': 20, 'group': 'all', 'value': '',
                               'metric': 'remainder_rms_mm', 'count': 1.5, 'mean': 0.25, 'median': 0.3, 'p90': 0.5, 'max': 2.0},
                              {'partition': 'validation', 'dataset': 'Default', 'seed': 'spread_max', 'n': 20, 'group': 'all', 'value': '',
                               'metric': 'remainder_rms_mm', 'count': 1497.0, 'mean': 0.25, 'median': 0.3, 'p90': 0.5, 'max': 2.0}])
    rows = _Read(path)
    assert rows[0][:4] == ['partition', 'dataset', 'seed', 'n']
    assert rows[1][7:] == ['9', '', '', '', ''] and rows[2][7:] == ['1.500000', '0.250000', '0.300000', '0.500000', '2.000000']
    assert rows[3][1:3] == ['Default', 'spread_max'] and rows[3][7] == '1497.000000'


def test_predictions_are_read_back_as_written_and_another_version_is_refused(tmp_path):
    """
    Description:
        predictions.npz round trip (format version 2, with the training seed): every field of two records (the first
        without queries, the second of seed 4) back bit for bit; a file of another format version is refused naming it.
    """

    rng = np.random.default_rng(0)
    records = [Prediction_Record_Str(Flight='real/Experiment_1.txt', Partition='real', N=40, T_0=0.3, T_End=1.1, X_First=rng.normal(size=6)),
               Prediction_Record_Str(Flight='3', Partition='test', Dataset='Seed_1', Seed='4', N=10, Grid_Start=0.0, T_0=0.05, T_End=1.2,
                                     X_First=rng.normal(size=6), Query_Time=np.array([0.1, 0.4]), Query_State=rng.normal(size=(2, 6)))]
    path = tmp_path / 'predictions.npz'
    Write_Predictions(str(path), records)
    back = Read_Predictions(str(path))
    for a, b in zip(records, back):
        assert (a.Flight, a.Partition, a.Dataset, a.Seed, a.N, a.Grid_Start, a.T_0, a.T_End) == (b.Flight, b.Partition, b.Dataset, b.Seed, b.N,
                                                                                                   b.Grid_Start, b.T_0, b.T_End)
        assert np.array_equal(a.X_First, b.X_First) and np.array_equal(a.Query_Time, b.Query_Time)
        assert np.array_equal(a.Query_State, b.Query_State)
    np.savez(str(tmp_path / 'old.npz'), format_version=np.int64(0))
    with pytest.raises(Report_Error_Cls, match='format version 2'):
        Read_Predictions(str(tmp_path / 'old.npz'))


def test_the_test_ledger_numbers_every_evaluation(tmp_path):
    """
    Description:
        The test ledger: none counted before the first; two appends number 1 and 2 below one header, with the
        run, the predictor, the hashes (datasets joined by ';'), the commit and the dirty flag.
    """

    path = str(tmp_path / 'test_ledger.csv')
    assert Count_Test_Ledger(path) == 0
    assert Append_Test_Ledger(path, 'QT_BASIS', 'QT_BASIS', 'a' * 64, ['b' * 64, 'c' * 64], 'd' * 40, False) == 1
    assert Append_Test_Ledger(path, 'GLS_M4', 'GLS_M4', 'e' * 64, ['b' * 64], 'f' * 40, True) == 2
    rows = _Read(path)
    assert rows[0] == ['evaluation', 'run', 'model', 'configuration_sha256', 'dataset_sha256', 'commit', 'dirty']
    assert rows[1] == ['1', 'QT_BASIS', 'QT_BASIS', 'a' * 64, 'b' * 64 + ';' + 'c' * 64, 'd' * 40, 'false']
    assert rows[2][0] == '2' and rows[2][6] == 'true'
    assert Count_Test_Ledger(path) == 2


def test_coverage_lists_every_flight_and_why_it_is_outside(tmp_path):
    """
    Description:
        coverage.csv (SPEC §2.13.6): one row per flight, 'true' with an empty reason inside, 'false' with its
        reasons outside.
    """

    path = tmp_path / 'coverage.csv'
    Write_Coverage(str(path), [('a/Experiment_1.txt', ''), ('a/Experiment_2.txt', 'release outside Areas.Throw; not in the target area')])
    assert _Read(path) == [['flight', 'inside', 'reason'], ['a/Experiment_1.txt', 'true', ''],
                           ['a/Experiment_2.txt', 'false', 'release outside Areas.Throw; not in the target area']]


def test_per_flight_and_per_horizon_are_read_back_as_written(tmp_path):
    """
    Description:
        The readers of a stored evaluation (SPEC §9.7; scripts/catch.py compare): per_flight.csv gives every row as
        text by column, an empty field for a value that does not exist; per_horizon.csv gives the rows of the
        partition asked for only, in the file's order.
    """

    Write_Per_Flight(str(tmp_path / 'per_flight.csv'), [_Simulated(), _Real()])
    rows = Read_Per_Flight(str(tmp_path / 'per_flight.csv'))
    assert [list(row) for row in rows] == [CONST_PER_FLIGHT_HEADER] * 2
    assert [(r['flight'], r['partition'], r['dataset'], r['n'], r['remainder_rms_mm'], r['crossing']) for r in rows] == [
        ('17', 'validation', 'Default', '20', '12.300000', 'both'),
        ('Frequency_200_Hz_Active_Filtering_OFF/Experiment_2.txt', 'real', '', '10', '20.000000', '')]
    assert (rows[1]['catch_point_error_mm'], rows[1]['end_mismatch_ms'], rows[1]['seed']) == ('', '10.000000', '')
    Write_Per_Horizon(str(tmp_path / 'per_horizon.csv'), [_Simulated(), _Real()])
    real = Read_Per_Horizon(str(tmp_path / 'per_horizon.csv'), 'real')
    assert real and {r['partition'] for r in real} == {'real'} and {r['truth'] for r in real} == {'recorded'}
    assert len(real) == len(_Real().Ranges)
    simulated = Read_Per_Horizon(str(tmp_path / 'per_horizon.csv'), 'validation')
    assert [(r['horizon'], r['truth'], r['position_error_mm']) for r in simulated][:2] == [('0.05', 'exact', '11.000000'),
                                                                                           ('end', 'exact', '50.000000')]
    assert Read_Per_Horizon(str(tmp_path / 'per_horizon.csv'), 'test') == []


def test_a_stored_file_with_other_columns_is_refused(tmp_path):
    """
    Description:
        A CSV file whose header is not the one this version writes (the files of the two counted evaluations of
        F7 end before the training seed) is refused, naming the file: its columns are never guessed.
    """

    path = tmp_path / 'per_flight.csv'
    path.write_text(','.join(CONST_PER_FLIGHT_HEADER[:-1]) + '\n17,validation,Default,20,0.1,200,12.3,-4.2,,,both,55.0,\n', encoding='ascii')
    with pytest.raises(Report_Error_Cls, match='per_flight.csv does not have the columns'):
        Read_Per_Flight(str(path))
    (tmp_path / 'empty.csv').write_text('', encoding='ascii')
    with pytest.raises(Report_Error_Cls, match='empty.csv does not have the columns'):
        Read_Per_Horizon(str(tmp_path / 'empty.csv'), 'real')


def test_coverage_is_read_back_and_another_value_is_refused(tmp_path):
    """
    Description:
        coverage.csv read back: per flight whether it is inside the support, in the file's order; a value other
        than true or false is refused, naming the flight.
    """

    path = tmp_path / 'coverage.csv'
    Write_Coverage(str(path), [('A/Experiment_1.txt', ''), ('B/Experiment_2.txt', 'release outside Areas.Throw'), ('C/Experiment_3.txt', '')])
    coverage = Read_Coverage(str(path))
    assert list(coverage.items()) == [('A/Experiment_1.txt', True), ('B/Experiment_2.txt', False), ('C/Experiment_3.txt', True)]
    path.write_text(path.read_text(encoding='ascii').replace('false', 'no'), encoding='ascii')
    with pytest.raises(Report_Error_Cls, match='"no" in the column inside of the flight B/Experiment_2.txt'):
        Read_Coverage(str(path))


def test_a_comparison_table_writes_text_whole_numbers_numbers_and_listed_values(tmp_path):
    """
    Description:
        A table of a comparison (SPEC §9.7): the columns given, in order; a text as it is, a whole number without
        a fraction, a number with six decimals, a value that does not exist as an empty field, and the listed
        per-flight values joined by ';'.
    """

    columns = ('run', 'n', 'flights', 'median_mm', 'low_mm', 'values')
    rows = [{'run': 'QT_BASIS', 'n': 20, 'flights': np.int64(45), 'median_mm': 1.25, 'low_mm': float('nan'), 'values': [], 'unused': 1},
            {'run': 'LS_F14', 'n': 40, 'flights': 9, 'median_mm': float('nan'), 'low_mm': -0.5, 'values': [1.0, -2.5]}]
    Write_Comparison_Table(str(tmp_path / 'paired.csv'), columns, rows)
    assert _Read(tmp_path / 'paired.csv') == [list(columns), ['QT_BASIS', '20', '45', '1.250000', '', ''],
                                             ['LS_F14', '40', '9', '', '-0.500000', '1.000000;-2.500000']]


def test_a_comparison_table_and_an_evaluations_summary_are_read_back_as_written(tmp_path):
    """
    Description:
        The readers the figures use (the visualisation, delivery 2): a comparison's table by its columns and an evaluation's
        summary.csv, every field as the writer wrote it; a table read with another table's columns is refused.
    """

    from Evaluation.Report import (Write_Comparison_Table, Read_Comparison_Table, Write_Summary, Read_Summary, CONST_COMPARISON_PAIRED_COLUMNS,
                                   CONST_COMPARISON_SUMMARY_COLUMNS)
    row = {'run': 'QT_BASIS', 'against': 'GLS_M4_F14', 'partition': 'real', 'dataset': '', 'support': 'all', 'n': 20, 'flights': 45,
           'left_out': 0, 'median_mm': 0.45, 'low_mm': float('nan'), 'high_mm': 1.08, 'values': [1.5, -2.25]}
    Write_Comparison_Table(str(tmp_path / 'paired.csv'), CONST_COMPARISON_PAIRED_COLUMNS, [row])
    assert Read_Comparison_Table(str(tmp_path / 'paired.csv'), CONST_COMPARISON_PAIRED_COLUMNS) == [
        dict(row, n='20', flights='45', left_out='0', median_mm='0.450000', low_mm='', high_mm='1.080000', values='1.500000;-2.250000')]
    with pytest.raises(Report_Error_Cls, match='does not have the columns run, partition'):
        Read_Comparison_Table(str(tmp_path / 'paired.csv'), CONST_COMPARISON_SUMMARY_COLUMNS)
    summary = {'partition': 'validation', 'dataset': 'Default', 'seed': '0', 'n': 10, 'group': 'all', 'value': '', 'metric': 'fixed_0.1_mm',
               'count': 1499, 'mean': 15.5, 'median': 14.3, 'p90': 30.0, 'max': 80.0}
    Write_Summary(str(tmp_path / 'summary.csv'), [summary])
    read = Read_Summary(str(tmp_path / 'summary.csv'))
    assert len(read) == 1 and read[0]['metric'] == 'fixed_0.1_mm' and float(read[0]['median']) == 14.3 and read[0]['seed'] == '0'


def test_the_predictions_of_one_flight_are_read_alone(tmp_path):
    """
    Description:
        Read_Predictions with a dataset and a flight gives that flight's records only, each with its own query rows, in
        the file's order; without them every record.
    """

    from Evaluation.Report import Write_Predictions, Read_Predictions
    records = [Prediction_Record_Str(Flight=flight, Partition='validation', Dataset=dataset, Seed='0', N=n, Grid_Start=0.0, T_0=0.1, T_End=1.0,
                                     X_First=np.full(6, np.nan), Query_Time=np.array([0.2, 0.3]) + n, Query_State=np.full((2, 6), float(n)))
               for dataset in ('Default', 'Noise_0') for flight in ('7', '8') for n in (10, 20)]
    Write_Predictions(str(tmp_path / 'predictions.npz'), records)
    read = Read_Predictions(str(tmp_path / 'predictions.npz'), dataset='Noise_0', flight='7')
    assert [(r.Dataset, r.Flight, r.N) for r in read] == [('Noise_0', '7', 10), ('Noise_0', '7', 20)]
    assert np.allclose(read[1].Query_Time, [20.2, 20.3]) and np.allclose(read[1].Query_State, 20.0)
    assert len(Read_Predictions(str(tmp_path / 'predictions.npz'))) == 8
