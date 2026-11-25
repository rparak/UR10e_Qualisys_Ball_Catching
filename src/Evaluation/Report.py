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
# File Name: ../Evaluation/Report.py
## =========================================================================== ##

# CSV (CSV file reading and writing)
import csv
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Evaluation/Core - the per-flight results, the ranges and the summary rows
from Evaluation.Core import Flight_Result_Str, CONST_HORIZON_RANGES, CONST_CATCH

"""
Description:
    Initialization of constants.

    The one writer of an evaluation's outputs (SPEC §9.7): per_flight.csv, per_horizon.csv and coverage.csv (each
    with its one reader, for a comparison of stored evaluations), summary.csv, predictions.npz (with its one
    reader), the one writer and reader of the test ledger, and the one writer of a comparison's tables. CSV files
    are comma-separated with a header line, LF, a fixed column and row order, '%.6f' floats and an empty field
    for a value that does not exist (a NaN: the catch and velocity metrics of a real flight, SPEC §2.14); no
    timestamp, no absolute path. Lengths in [mm], times in [ms] unless a column says [s]; a direction error in
    [deg], a speed error in [m/s].
"""
# The columns of per_flight.csv, per_horizon.csv and summary.csv (SPEC §9.7), in order. The training seed
# ('' for a numerical predictor) and the velocity's errors (F8) follow the columns F7
# wrote, whose order is unchanged.
#   Unit [Tuple<string>]
CONST_PER_FLIGHT_COLUMNS = ('flight', 'partition', 'dataset', 'n', 'prefix_duration_s', 'rate_hz', 'remainder_rms_mm', 'end_mismatch_ms',
                            'catch_point_error_mm', 'time_to_catch_error_ms', 'crossing', 'inference_ms', 'refusal', 'seed')
CONST_PER_HORIZON_COLUMNS = ('flight', 'partition', 'dataset', 'n', 'horizon', 'truth', 'position_error_mm', 'samples',
                             'direction_error_deg', 'speed_error_m_s', 'seed')
CONST_SUMMARY_COLUMNS = ('partition', 'dataset', 'seed', 'n', 'group', 'value', 'metric', 'count', 'mean', 'median', 'p90', 'max')
# The columns of coverage.csv ('test' only, SPEC §2.13.6): each real validation flight, whether its release state lies
# in the support of the generated throws, and why not.
#   Unit [Tuple<string>]
CONST_COVERAGE_COLUMNS = ('flight', 'inside', 'reason')
# The columns of the test ledger (SPEC §9.7): the evaluation's number, the run, its predictor, the SHA-256
# of its training configuration, of its datasets (joined by ';'), the commit and whether the tree was dirty.
#   Unit [Tuple<string>]
CONST_TEST_LEDGER_COLUMNS = ('evaluation', 'run', 'model', 'configuration_sha256', 'dataset_sha256', 'commit', 'dirty')
# The columns of a comparison's three tables (SPEC §9.7; scripts/catch.py compare): the
# summary of each run, the paired differences of two runs, and a run's real-flight error against its simulated one.
# 'values' lists the per-flight values of a group too small for statistics, joined by ';'.
#   Unit [Tuple<string>]
CONST_COMPARISON_SUMMARY_COLUMNS = ('run', 'partition', 'dataset', 'support', 'n', 'metric', 'flights', 'value', 'seed_min', 'seed_max',
                                    'values')
CONST_COMPARISON_PAIRED_COLUMNS = ('run', 'against', 'partition', 'dataset', 'support', 'n', 'flights', 'left_out', 'median_mm', 'low_mm',
                                   'high_mm', 'values')
CONST_COMPARISON_SIM_TO_REAL_COLUMNS = ('run', 'n', 'dataset', 'real_flights', 'real_median_mm', 'real_p90_mm', 'simulated_flights',
                                        'simulated_median_mm', 'simulated_p90_mm', 'difference_mm', 'low_mm', 'high_mm', 'ratio')
# The number format of the CSV files.
#   Unit [string]
CONST_FLOAT_FORMAT = '%.6f'
# predictions.npz's format version (2: with the training seed); a reader refuses another.
#   Unit [int]
CONST_PREDICTIONS_VERSION = 2


class Report_Error_Cls(ValueError):
    """
    Description:
        A stored file cannot be read: a predictions file of another format version or with a missing array, a CSV
        file whose header is not the one this version writes. The message names the file and what is expected.
    """


@dataclass
class Prediction_Record_Str:
    """
    Description:
        One prediction as predictions.npz holds it (the cache of SPEC §4.5): its states at the query times (the
        samples after the prefix) are stored. A trajectory integrated under the flight model (the numerical
        predictor's) is rebuilt from the state at its first grid time
        (Prediction.Core.Trajectory_From_State); a query-time model's is not a state's trajectory, and X_First
        is NaN (it is rebuilt from the run's weights and the prefix).
    """

    # The flight, its partition and dataset, the predictor's training seed ('' for a numerical predictor), and N.
    #   Unit [string], [string], [string], [string], [int]
    Flight: str = ''
    Partition: str = ''
    Dataset: str = ''
    Seed: str = ''
    N: int = 0
    # The time of the trajectory's first state (or of the prefix's first sample), t_0 and t_end, and the state there.
    #   Unit [float, s], [Vector<float> 1x6, m, m/s]
    Grid_Start: float = 0.0
    T_0: float = 0.0
    T_End: float = 0.0
    X_First: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # The query times and the predicted states there (held at t_end after it).
    #   Unit [Vector<float> 1xQ, s], [Matrix<float> Qx6, m, m/s]
    Query_Time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Query_State: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))


def _Number(value: float, scale: float = 1.0) -> str:
    """
    Description:
        A number in CONST_FLOAT_FORMAT after scaling, or '' when it does not exist (NaN).

    Args:
        (1) value [float]: The value.
        (2) scale [float]: The factor to the column's unit (1e3: [m] -> [mm], [s] -> [ms]).

    Returns:
        (1) parameter [string]: The field.
    """

    return '' if not np.isfinite(value) else CONST_FLOAT_FORMAT % (scale * value)


def _Write(path: str, columns: tp.Sequence[str], rows: tp.Iterable[tp.Sequence[str]]) -> None:
    """
    Description:
        A CSV file with its header.

    Args:
        (1) path [string]: The file.
        (2) columns [Sequence<string>]: The header.
        (3) rows [Iterable<Sequence<string>>]: The rows, as text.
    """

    with open(path, 'w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(rows)


def _Read(path: str, columns: tp.Sequence[str]) -> tp.Iterator[tp.Dict[str, str]]:
    """
    Description:
        The rows of a CSV file written by this module, as text by column; a file with another header is refused.

    Args:
        (1) path [string]: The file.
        (2) columns [Sequence<string>]: The header it must have.

    Returns:
        (1) parameter [Iterator<Dict<string, string>>]: The rows.
    """

    with open(path, 'r', encoding='ascii', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None or tuple(header) != tuple(columns):
            raise Report_Error_Cls('%s does not have the columns %s (SPEC §9.7); it was written by another version or is not that file. '
                                   'Evaluate the run again.' % (path, ', '.join(columns)))
        for row in reader:
            yield dict(zip(columns, row))


def Write_Per_Flight(path: str, results: tp.List[Flight_Result_Str]) -> None:
    """
    Description:
        per_flight.csv (SPEC §9.7): one row per flight, N and predictor, in the order given; a refused prediction has
        the crossing 'refused', its reason in 'refusal' and every metric empty.

    Args:
        (1) path [string]: The file.
        (2) results [List<Flight_Result_Str>]: The results.
    """

    _Write(path, CONST_PER_FLIGHT_COLUMNS, ([r.Flight, r.Partition, r.Dataset, '%d' % r.N, _Number(r.Prefix_Duration), '%g' % r.Rate,
                                             _Number(r.Remainder_RMS, 1e3), _Number(r.End_Mismatch, 1e3), _Number(r.Catch_Point_Error, 1e3),
                                             _Number(r.Time_To_Catch_Error, 1e3), r.Crossing, _Number(r.Inference_Time, 1e3), r.Refusal,
                                             r.Seed] for r in results))


def Read_Per_Flight(path: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        The rows of a per_flight.csv (SPEC §9.7), as text by column, in the file's order.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return list(_Read(path, CONST_PER_FLIGHT_COLUMNS))


def Write_Per_Horizon(path: str, results: tp.List[Flight_Result_Str]) -> None:
    """
    Description:
        per_horizon.csv (SPEC §9.7): per flight, N and predictor, each fixed horizon reached (truth 'exact', no
        sample count) with the velocity's direction error [deg] and speed error [m/s] there, the catch surface
        ('catch': the catch-point error and the velocity's errors at the crossing, when both trajectories cross),
        and each horizon range reached, one value per flight (the median of its samples) against the exact truth
        ('exact') and the observed positions ('observed') of a simulated flight, or the recording ('recorded') of
        a real one, with its sample count. A real flight has no velocity error.

    Args:
        (1) path [string]: The file.
        (2) results [List<Flight_Result_Str>]: The results.
    """

    def Rows() -> tp.Iterator[tp.List[str]]:
        """
        Description:
            The rows, flight by flight.

        Returns:
            (1) parameter [Iterator<List<string>>]: The rows.
        """

        nan = float('nan')
        for r in results:
            head = [r.Flight, r.Partition, r.Dataset, '%d' % r.N]
            for horizon, error in r.Fixed.items():
                yield head + [horizon, 'exact', _Number(error, 1e3), '', _Number(float(np.degrees(r.Direction.get(horizon, nan)))),
                              _Number(r.Speed.get(horizon, nan)), r.Seed]
            if CONST_CATCH in r.Speed:
                yield head + [CONST_CATCH, 'exact', _Number(r.Catch_Point_Error, 1e3), '',
                              _Number(float(np.degrees(r.Direction[CONST_CATCH]))), _Number(r.Speed[CONST_CATCH]), r.Seed]
            truth = 'recorded' if r.Partition == 'real' else 'exact'
            for name, ranges in ((truth, r.Ranges), ('observed', r.Ranges_Observed)):
                for index, error, samples in ranges:
                    low, high = CONST_HORIZON_RANGES[index]
                    closing = ']' if index == len(CONST_HORIZON_RANGES) - 1 else ')'
                    yield head + ['[%g, %g%s' % (low, high, closing), name, _Number(error, 1e3), '%d' % samples, '', '', r.Seed]

    _Write(path, CONST_PER_HORIZON_COLUMNS, Rows())


def Read_Per_Horizon(path: str, partition: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        The rows of one partition of a per_horizon.csv (SPEC §9.7), as text by column, in the file's order. The
        file of a simulated partition holds millions of rows, so the partition is selected while reading.

    Args:
        (1) path [string]: The file.
        (2) partition [string]: The partition whose rows are returned ('validation', 'test' or 'real').

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return [row for row in _Read(path, CONST_PER_HORIZON_COLUMNS) if row['partition'] == partition]


def Write_Summary(path: str, rows: tp.List[tp.Dict[str, tp.Any]]) -> None:
    """
    Description:
        summary.csv (SPEC §7.12, §9.7): the rows of Evaluation.Core.Summary_Rows in their order; a statistic that
        does not exist (a range below the minimum of flights) is an empty field.

    Args:
        (1) path [string]: The file.
        (2) rows [List<Dict<string, Any>>]: The summary rows.
    """

    _Write(path, CONST_SUMMARY_COLUMNS, ([r['partition'], r['dataset'], r['seed'], '%d' % r['n'], r['group'], r['value'], r['metric'],
                                          _Number(float(r['count'])) if 'spread' in (r['dataset'][:6], r['seed'][:6]) else '%d' % r['count']]
                                         + [_Number(r[k]) for k in ('mean', 'median', 'p90', 'max')] for r in rows))


def Read_Summary(path: str) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        The rows of an evaluation's summary.csv (SPEC §9.7; Write_Summary), as text by column, in the file's order; read
        by scripts/plots.py (the visualisation, delivery 2).

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return list(_Read(path, CONST_SUMMARY_COLUMNS))


def Write_Predictions(path: str, records: tp.List[Prediction_Record_Str]) -> None:
    """
    Description:
        predictions.npz (SPEC §4.5, §9.7), np.savez uncompressed: per prediction its identity (with the training
        seed), the trajectory's times and first state; ragged with offsets (prediction i is query_offset[i]:query_offset[i + 1]) the
        query times and states.

    Args:
        (1) path [string]: The file.
        (2) records [List<Prediction_Record_Str>]: The predictions.
    """

    offset = np.concatenate(([0], np.cumsum([len(r.Query_Time) for r in records]))).astype(np.int64)
    np.savez(path, format_version=np.int64(CONST_PREDICTIONS_VERSION), flight=np.array([r.Flight for r in records], dtype=str),
             partition=np.array([r.Partition for r in records], dtype=str), dataset=np.array([r.Dataset for r in records], dtype=str),
             seed=np.array([r.Seed for r in records], dtype=str), n=np.array([r.N for r in records], dtype=np.int64),
             grid_start=np.array([r.Grid_Start for r in records], dtype=np.float64),
             t_0=np.array([r.T_0 for r in records], dtype=np.float64), t_end=np.array([r.T_End for r in records], dtype=np.float64),
             x_first=np.array([r.X_First for r in records], dtype=np.float64).reshape(-1, 6), query_offset=offset,
             query_time=np.concatenate([r.Query_Time for r in records] + [np.zeros(0)]),
             query_state=np.concatenate([r.Query_State for r in records] + [np.zeros((0, 6))]))


def Read_Predictions(path: str, dataset: tp.Optional[str] = None, flight: tp.Optional[str] = None) -> tp.List[Prediction_Record_Str]:
    """
    Description:
        The predictions of a predictions.npz of format version CONST_PREDICTIONS_VERSION; with `dataset` and `flight`,
        only that flight's (the query arrays of the others are sliced, never copied).

    Args:
        (1) path [string]: The file.
        (2) dataset [string]: Only this dataset's predictions, or None for all.
        (3) flight [string]: Only this flight's predictions, or None for all.

    Returns:
        (1) parameter [List<Prediction_Record_Str>]: The predictions, in the file's order.
    """

    with np.load(path, allow_pickle=False) as data:
        if 'format_version' not in data or int(data['format_version']) != CONST_PREDICTIONS_VERSION:
            raise Report_Error_Cls('%s is not a predictions file of format version %d (SPEC §9.7). Evaluate the run again.'
                                   % (path, CONST_PREDICTIONS_VERSION))
        a = {key: data[key] for key in data.files}
    o = a['query_offset']
    keep = np.ones(len(a['n']), dtype=bool)
    if dataset is not None:
        keep &= a['dataset'] == dataset
    if flight is not None:
        keep &= a['flight'] == flight
    return [Prediction_Record_Str(Flight=str(a['flight'][i]), Partition=str(a['partition'][i]), Dataset=str(a['dataset'][i]),
                                  Seed=str(a['seed'][i]), N=int(a['n'][i]),
                                  Grid_Start=float(a['grid_start'][i]), T_0=float(a['t_0'][i]), T_End=float(a['t_end'][i]),
                                  X_First=a['x_first'][i].copy(), Query_Time=a['query_time'][o[i]:o[i + 1]].copy(),
                                  Query_State=a['query_state'][o[i]:o[i + 1]].copy()) for i in np.flatnonzero(keep)]


def Write_Coverage(path: str, coverage: tp.List[tp.Tuple[str, str]]) -> None:
    """
    Description:
        coverage.csv (SPEC §2.13.6, §9.7): one row per real validation flight, 'true' when its release state lies
        in the support of the generated throws, else 'false' and every reason.

    Args:
        (1) path [string]: The file.
        (2) coverage [List<Tuple<string, string>>]: (flight, reasons; '' when inside), Dataset.Calibration.Coverage.
    """

    _Write(path, CONST_COVERAGE_COLUMNS, ([name, 'true' if reason == '' else 'false', reason] for name, reason in coverage))


def Read_Coverage(path: str) -> tp.Dict[str, bool]:
    """
    Description:
        A coverage.csv (SPEC §9.7): per real flight whether its release state lies in the support of the generated
        throws. A value other than 'true' or 'false' is refused.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Dict<string, bool>]: flight -> inside, in the file's order.
    """

    coverage = {}
    for row in _Read(path, CONST_COVERAGE_COLUMNS):
        if row['inside'] not in ('true', 'false'):
            raise Report_Error_Cls('%s has the value "%s" in the column inside of the flight %s; it is true or false (SPEC §9.7). '
                                   'Evaluate the run again.' % (path, row['inside'], row['flight']))
        coverage[row['flight']] = row['inside'] == 'true'
    return coverage


def Write_Comparison_Table(path: str, columns: tp.Sequence[str], rows: tp.List[tp.Dict[str, tp.Any]]) -> None:
    """
    Description:
        One table of a comparison (SPEC §9.7; Evaluation.Comparison): the rows' values by column, a text as it is,
        a whole number as '%d', a number in CONST_FLOAT_FORMAT (empty when it does not exist), and a list of
        numbers joined by ';'.

    Args:
        (1) path [string]: The file.
        (2) columns [Sequence<string>]: One of the CONST_COMPARISON_*_COLUMNS.
        (3) rows [List<Dict<string, Any>>]: The rows.
    """

    def Field(value: tp.Any) -> str:
        """
        Description:
            One value as text.

        Args:
            (1) value [Any]: A text, a whole number, a number or a list of numbers.

        Returns:
            (1) parameter [string]: The field.
        """

        if isinstance(value, str):
            return value
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return '%d' % value
        if isinstance(value, (list, tuple)):
            return ';'.join(_Number(float(v)) for v in value)
        return _Number(float(value))

    _Write(path, columns, ([Field(row[column]) for column in columns] for row in rows))


def Read_Comparison_Table(path: str, columns: tp.Sequence[str]) -> tp.List[tp.Dict[str, str]]:
    """
    Description:
        The rows of one table of a comparison (SPEC §9.7; Write_Comparison_Table), as text by column, in the file's
        order; read by scripts/plots.py (the visualisation, delivery 2).

    Args:
        (1) path [string]: The file.
        (2) columns [Sequence<string>]: One of the CONST_COMPARISON_*_COLUMNS.

    Returns:
        (1) parameter [List<Dict<string, string>>]: The rows.
    """

    return list(_Read(path, columns))


def Count_Test_Ledger(path: str) -> int:
    """
    Description:
        The number of test evaluations recorded in the ledger (0 when it does not exist yet).

    Args:
        (1) path [string]: The ledger.

    Returns:
        (1) parameter [int]: Its rows below the header.
    """

    try:
        with open(path, 'r', encoding='ascii', newline='') as file:
            rows = list(csv.reader(file))
    except FileNotFoundError:
        return 0
    return max(len(rows) - 1, 0)


def Append_Test_Ledger(path: str, run: str, model: str, configuration_sha256: str, dataset_sha256: tp.List[str], commit: str, dirty: bool) -> int:
    """
    Description:
        Record one 'test' evaluation (SPEC §2.13.9, §9.7) before any result is written, so a run that fails
        afterwards is still counted: its number, the run, the predictor, the hashes, the commit and the dirty flag.

    Args:
        (1) path [string]: The ledger.
        (2) run [string]: The run's name.
        (3) model [string]: The predictor.
        (4) configuration_sha256 [string]: The SHA-256 of the run's training configuration.
        (5) dataset_sha256 [List<string>]: The SHA-256 of each dataset's flights.npz.
        (6) commit [string]: The git commit of the code.
        (7) dirty [bool]: Whether the tree had uncommitted changes.

    Returns:
        (1) parameter [int]: This evaluation's number (1 for the first).
    """

    evaluation = Count_Test_Ledger(path) + 1
    with open(path, 'a', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        if evaluation == 1:
            writer.writerow(CONST_TEST_LEDGER_COLUMNS)
        writer.writerow(['%d' % evaluation, run, model, configuration_sha256, ';'.join(dataset_sha256), commit, 'true' if dirty else 'false'])
    return evaluation
