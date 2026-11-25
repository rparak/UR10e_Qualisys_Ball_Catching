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
# File Name: ../Recording/Report.py
## =========================================================================== ##

# CSV (CSV file reading and writing)
import csv
# Collections (Container datatypes)
import collections
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the assumed transform
from Configuration.Core import Transform_Str
#   ../Recording/Core - the recordings, their steps and the transform
from Recording.Core import Recording_Str, Transform_To_Robot, CONST_KIND_AMBIGUOUS, CONST_KIND_GAP, CONST_KIND_IRREGULAR, CONST_KIND_ZERO
#   ../Recording/Segmentation - the flights
from Recording.Segmentation import Flight_Str
#   ../Recording/Identification - the identified flight model and the fitted flights
from Recording.Identification import (Identification_Str, Fitted_Flight_Str, Variant_Str, Estimated_Floor, Edge_Interior_RMS,
                                      Flight_Name)

"""
Description:
    Initialization of constants.

    The one writer of the recording loader's two outputs (SPEC §9.4): the summary, one row per
    recording, and the steps, one row per reported step; and the one writer (and reader) of the
    identification's three outputs (SPEC §9.8): the summary, the flights and the ledger. All are
    deterministic: a fixed column and row order, fixed number formats, no timestamp, no absolute path.
    Nothing reads the reports back; the ledger is read only to count its rows.
"""
# The summary's columns (SPEC §9.4), in order.
#   Unit [Tuple<string>]
CONST_SUMMARY_COLUMNS = ('file', 'rate_hz', 'filtering', 'samples', 'wraps', 'gaps', 'zero_steps', 'irregular_steps',
                         'ambiguous_steps', 'set', 'segmented', 'reason', 'first_line', 'last_line', 'flight_samples',
                         'duration_s', 'coverage', 'a_x', 'a_y', 'a_z', 'residual_rms_m', 'residual_max_m', 'outliers', 'stalls',
                         'adjacent_ambiguous', 'first_x_robot_m', 'first_y_robot_m', 'first_z_robot_m', 'last_x_robot_m',
                         'last_y_robot_m', 'last_z_robot_m')
# The steps' columns (SPEC §9.4), in order.
#   Unit [Tuple<string>]
CONST_STEP_COLUMNS = ('file', 'line', 'kind', 'reason', 'ratio', 'frames', 'jump_m')
# The number format of every float column.
#   Unit [string]
CONST_FLOAT_FORMAT = '%.6f'
# The identification summary's columns (SPEC §9.8), in order.
#   Unit [Tuple<string>]
CONST_IDENTIFICATION_COLUMNS = ('kind', 'variant', 'session', 'tolerance_m', 'trim_s', 'without', 'flights_segmented', 'flights',
                                'samples', 'gravity_m_s2', 'drag_per_m', 'drag_bootstrap_sd_per_m', 'drag_half_range_per_m',
                                'drag_std_per_m', 'rms_median_m', 'max_median_m', 'floor_median_m', 'edge_rms_median_m',
                                'interior_rms_median_m', 'evaluation')
# The identification flights' columns (SPEC §9.8), in order.
#   Unit [Tuple<string>]
CONST_IDENTIFICATION_FLIGHT_COLUMNS = ('kind', 'trim_s', 'file', 'session', 'set', 'first_line', 'last_line', 'samples', 'x_m', 'y_m',
                                       'z_m', 'v_x_m_s', 'v_y_m_s', 'v_z_m_s', 'rms_m', 'max_m', 'floor_m', 'edge_rms_m', 'interior_rms_m')
# The ledger's columns (SPEC §9.8): one row per identify run that read the validation flights.
#   Unit [Tuple<string>]
CONST_LEDGER_COLUMNS = ('evaluation', 'commit', 'dirty', 'drag_per_m', 'drag_std_per_m')
# The number format of the identification's floats: 9 significant digits (k to about 1e-11 1/m).
#   Unit [string]
CONST_IDENTIFICATION_FORMAT = '%.9g'


def _Bool(value: bool) -> str:
    """
    Description:
        A boolean as the reports write it.

    Args:
        (1) value [bool]: The value.

    Returns:
        (1) parameter [string]: 'true' or 'false'.
    """

    return 'true' if value else 'false'


def _Summary_Row(recording: Recording_Str, set_name: str, flight: tp.Optional[Flight_Str], reason: str, Transform_Str: Transform_Str) -> tp.List[str]:
    """
    Description:
        One summary row: the recording's counts, its set, and its flight (empty cells without one), with
        the flight's first and last sample in the robot frame by the assumed transform (SPEC §7.3).

    Args:
        (1) recording [Recording_Str(object)]: The recording.
        (2) set_name [string]: Its set.
        (3) flight [Flight_Str(object)]: Its flight, or None.
        (4) reason [string]: Why no flight ('' for a flight).
        (5) Transform_Str [Transform_Str(object)]: The assumed transform.

    Returns:
        (1) parameter [List<string>]: The cells, in CONST_SUMMARY_COLUMNS order.
    """

    kinds = collections.Counter(step.Kind for step in recording.Steps)
    row = [recording.File, '%d' % recording.Rate, _Bool(recording.Filtering), '%d' % len(recording.Frame), '%d' % recording.Wraps,
           '%d' % kinds[CONST_KIND_GAP], '%d' % kinds[CONST_KIND_ZERO], '%d' % kinds[CONST_KIND_IRREGULAR],
           '%d' % kinds[CONST_KIND_AMBIGUOUS], set_name, _Bool(flight is not None), reason]
    if flight is None:
        return row + [''] * (len(CONST_SUMMARY_COLUMNS) - len(row))
    ends = Transform_To_Robot(flight.Position[[0, -1]], Transform_Str=Transform_Str)
    row += ['%d' % flight.First_Line, '%d' % flight.Last_Line, '%d' % len(flight.Frame), CONST_FLOAT_FORMAT % flight.Time[-1],
            CONST_FLOAT_FORMAT % (len(flight.Frame) / (flight.Frame[-1] + 1)), *[CONST_FLOAT_FORMAT % v for v in flight.Acceleration],
            CONST_FLOAT_FORMAT % flight.Residual_RMS, CONST_FLOAT_FORMAT % flight.Residual_Max, '%d' % len(flight.Outlier_Lines),
            '%d' % len(flight.Stall_Lines), _Bool(flight.Adjacent_Ambiguous), *[CONST_FLOAT_FORMAT % v for v in np.ravel(ends)]]
    return row


def Write_Recording_Summary(path: str, recordings: tp.List[Recording_Str], segmented: tp.List[tp.Tuple[str, tp.Optional[Flight_Str], str]],
                            *, Transform_Str: Transform_Str) -> None:
    """
    Description:
        Write the summary (SPEC §9.4): a header of CONST_SUMMARY_COLUMNS, then one row per recording in
        the recordings' order.

    Args:
        (1) path [string]: The file to write.
        (2) recordings [List<Recording_Str>]: The recordings.
        (3) segmented [List<Tuple<string, Flight_Str, string>>]: Their sets and flights
                                                                 (Recording.Segmentation.Segment_Flights).
        (4) Transform_Str [Transform_Str(object)]: REQUIRED, keyword-only: the assumed transform.
    """

    with open(path, 'w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_SUMMARY_COLUMNS)
        for recording, (set_name, flight, reason) in zip(recordings, segmented):
            writer.writerow(_Summary_Row(recording, set_name, flight, reason, Transform_Str))


def Write_Recording_Steps(path: str, recordings: tp.List[Recording_Str]) -> None:
    """
    Description:
        Write the steps (SPEC §9.4): a header of CONST_STEP_COLUMNS, then every reported step (gap, zero,
        irregular, ambiguous) with its file and line, in the recordings' order and then line order.

    Args:
        (1) path [string]: The file to write.
        (2) recordings [List<Recording_Str>]: The recordings.
    """

    with open(path, 'w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_STEP_COLUMNS)
        for recording in recordings:
            for step in recording.Steps:
                writer.writerow([recording.File, '%d' % step.Line, step.Kind, step.Reason, CONST_FLOAT_FORMAT % step.Ratio,
                                 '%d' % step.Frames, CONST_FLOAT_FORMAT % step.Jump])


def Summary_Lines(recordings: tp.List[Recording_Str], segmented: tp.List[tp.Tuple[str, tp.Optional[Flight_Str], str]]) -> tp.List[str]:
    """
    Description:
        The summary a script prints: files, steps by kind, ambiguous steps by reason, flights segmented,
        excluded and adjacent-flagged, and the counts per set.

    Args:
        (1) recordings [List<Recording_Str>]: The recordings.
        (2) segmented [List<Tuple<string, Flight_Str, string>>]: Their sets and flights.

    Returns:
        (1) parameter [List<string>]: The lines.
    """

    steps = [step for recording in recordings for step in recording.Steps]
    kinds = collections.Counter(step.Kind for step in steps)
    reasons = collections.Counter(step.Reason.split(':')[0] for step in steps if step.Kind == CONST_KIND_AMBIGUOUS)
    flights = [(recording, set_name, flight, reason) for recording, (set_name, flight, reason) in zip(recordings, segmented)]
    lines = ['files loaded: %d (%d samples)' % (len(recordings), sum(len(recording.Frame) for recording in recordings)),
             'gaps: %d' % kinds[CONST_KIND_GAP], 'zero steps: %d' % kinds[CONST_KIND_ZERO],
             'irregular steps: %d' % kinds[CONST_KIND_IRREGULAR],
             'ambiguous steps: %d (rule %d, zero %d, cited %d)' % (kinds[CONST_KIND_AMBIGUOUS], reasons['continuity'],
                                                                   reasons['zero step'], reasons['cited']),
             'flights segmented: %d of %d' % (sum(flight is not None for _, _, flight, _ in flights), len(flights))]
    lines.append('not segmented: %d' % sum(flight is None for _, _, flight, _ in flights))
    lines += ['  %s: %s' % (recording.File, reason) for recording, _, flight, reason in flights if flight is None]
    inside = [recording.File for recording, _, flight, _ in flights if flight is not None
              and any(step.Kind == CONST_KIND_AMBIGUOUS and flight.First_Line < step.Line <= flight.Last_Line for step in recording.Steps)]
    lines.append('excluded (an ambiguous step inside): %d' % len(inside))
    lines += ['  %s' % name for name in inside]
    adjacent = [recording.File for recording, _, flight, _ in flights if flight is not None and flight.Adjacent_Ambiguous]
    lines.append('adjacent-flagged: %d' % len(adjacent))
    lines += ['  %s' % name for name in adjacent]
    for set_name in sorted({set_name for _, set_name, _, _ in flights}):
        members = [(recording, flight) for recording, name, flight, _ in flights if name == set_name]
        lines.append('%s: %d recordings, %d flights' % (set_name, len(members), sum(flight is not None for _, flight in members)))
    return lines


def _Number(value: float) -> str:
    """
    Description:
        A float as the identification's files write it ('nan' for a missing value).

    Args:
        (1) value [float]: The value.

    Returns:
        (1) parameter [string]: The text.
    """

    return CONST_IDENTIFICATION_FORMAT % value


def _Statistics(fitted: tp.List[Fitted_Flight_Str]) -> tp.List[str]:
    """
    Description:
        The medians over flights of the RMS, the max, the estimated floor and the edge and interior RMS.

    Args:
        (1) fitted [List<Fitted_Flight_Str>]: The fitted flights.

    Returns:
        (1) parameter [List<string>]: The five medians [m], as written.
    """

    split = np.array([Edge_Interior_RMS(f.Flight.Frame, f.Residual) for f in fitted])
    return [_Number(float(np.median([f.RMS for f in fitted]))), _Number(float(np.median([f.Max for f in fitted]))),
            _Number(float(np.median([Estimated_Floor(f.Flight.Time, f.Residual) for f in fitted]))),
            _Number(float(np.nanmedian(split[:, 0]))), _Number(float(np.nanmedian(split[:, 1])))]


def Validation_Groups(fitted: tp.List[Fitted_Flight_Str], sessions: tp.Dict[str, str]) -> tp.List[tp.Tuple[str, tp.List[Fitted_Flight_Str]]]:
    """
    Description:
        The validation rows' groups: all flights, then each session in name order.

    Args:
        (1) fitted [List<Fitted_Flight_Str>]: The fitted validation flights.
        (2) sessions [Dict<string, string>]: The session of each flight, by Flight_Name.

    Returns:
        (1) parameter [List<Tuple<string, List<Fitted_Flight_Str>>>]: ('all' or the session, its flights).
    """

    names = sorted({sessions[Flight_Name(f.Flight)] for f in fitted})
    return [('all', fitted)] + [(name, [f for f in fitted if sessions[Flight_Name(f.Flight)] == name]) for name in names]


def Write_Identification_Summary(path: str, identification: Identification_Str, validation: tp.List[tp.Tuple[float, tp.List[Fitted_Flight_Str]]],
                                 sessions: tp.Dict[str, str], evaluation: int) -> None:
    """
    Description:
        Write the identification summary (SPEC §9.8): a header of CONST_IDENTIFICATION_COLUMNS; the
        identified row; the sensitivity rows; per validation trim, the rows of all validation flights
        and of each session, each carrying the evaluation's number in the ledger.

    Args:
        (1) path [string]: The file to write.
        (2) identification [Identification_Str(object)]: The identified model and its rows.
        (3) validation [List<Tuple<float, List<Fitted_Flight_Str>>>]: Per trim [s], the fitted validation flights.
        (4) sessions [Dict<string, string>]: The session of each flight, by Flight_Name.
        (5) evaluation [int]: This run's row in the ledger.
    """

    model = identification.Flight_Model_Str

    def Variant_Row(kind: str, variant: Variant_Str) -> tp.List[str]:
        """
        Description:
            One identified or sensitivity row.

        Args:
            (1) kind [string]: 'identified' or 'sensitivity'.
            (2) variant [Variant_Str(object)]: The variant.

        Returns:
            (1) parameter [List<string>]: The row.
        """

        main = kind == 'identified'
        return [kind, variant.Name, '', _Number(variant.Tolerance), _Number(variant.Trim), ' '.join(variant.Without),
                '%d' % variant.Flights_Segmented, '%d' % variant.Flights, '%d' % variant.Samples, _Number(-model.Gravity[2]),
                _Number(variant.Drag), _Number(identification.Drag_Bootstrap_Sd) if main else '',
                _Number(identification.Drag_Half_Range) if main else '', _Number(model.Drag_Std) if main else '',
                *_Statistics(variant.Fitted), '']

    with open(path, 'w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_IDENTIFICATION_COLUMNS)
        writer.writerow(Variant_Row('identified', identification.Variants[0]))
        for variant in identification.Variants[1:]:
            writer.writerow(Variant_Row('sensitivity', variant))
        for trim, fitted in validation:
            for name, group in Validation_Groups(fitted, sessions):
                writer.writerow(['validation', 'trim %.0f ms' % (trim * 1e3), name, '', _Number(trim), '', '', '%d' % len(group),
                                 '%d' % sum(len(f.Flight.Frame) for f in group), _Number(-model.Gravity[2]), _Number(model.Drag),
                                 '', '', '', *_Statistics(group), '%d' % evaluation])


def Write_Identification_Flights(path: str, groups: tp.List[tp.Tuple[str, float, tp.List[Fitted_Flight_Str]]], sessions: tp.Dict[str, str]) -> None:
    """
    Description:
        Write one row per fitted flight (SPEC §9.8): the identification flights of the identified row,
        then the validation flights per trim, each in the given order.

    Args:
        (1) path [string]: The file to write.
        (2) groups [List<Tuple<string, float, List<Fitted_Flight_Str>>>]: (kind, trim [s], fitted flights).
        (3) sessions [Dict<string, string>]: The session of each flight, by Flight_Name.
    """

    with open(path, 'w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_IDENTIFICATION_FLIGHT_COLUMNS)
        for kind, trim, fitted in groups:
            for f in fitted:
                edge, interior = Edge_Interior_RMS(f.Flight.Frame, f.Residual)
                writer.writerow([kind, _Number(trim), Flight_Name(f.Flight), sessions[Flight_Name(f.Flight)], f.Flight.Set,
                                 '%d' % f.Flight.First_Line, '%d' % f.Flight.Last_Line, '%d' % len(f.Flight.Frame),
                                 *[_Number(v) for v in f.X_0], _Number(f.RMS), _Number(f.Max),
                                 _Number(Estimated_Floor(f.Flight.Time, f.Residual)), _Number(edge), _Number(interior)])


def Count_Ledger(path: str) -> int:
    """
    Description:
        The number of identify runs recorded in the ledger (0 when it does not exist yet).

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


def Append_Ledger(path: str, commit: str, dirty: bool, Flight_Model_Str: tp.Any) -> int:
    """
    Description:
        Record one identify run that read the validation flights: its
        number, the commit, whether the tree was dirty and the identified k. Written before the
        results, so a run that fails afterwards is still counted.

    Args:
        (1) path [string]: The ledger.
        (2) commit [string]: The git commit of the code.
        (3) dirty [bool]: Whether the tree had uncommitted changes.
        (4) Flight_Model_Str [Flight_Model_Str(object)]: The identified model.

    Returns:
        (1) parameter [int]: This run's number (1 for the first).
    """

    evaluation = Count_Ledger(path) + 1
    new = evaluation == 1
    with open(path, 'a', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        if new:
            writer.writerow(CONST_LEDGER_COLUMNS)
        writer.writerow(['%d' % evaluation, commit, _Bool(dirty), _Number(Flight_Model_Str.Drag), _Number(Flight_Model_Str.Drag_Std)])
    return evaluation


def Identification_Lines(identification: Identification_Str, validation: tp.List[tp.Tuple[float, tp.List[Fitted_Flight_Str]]],
                         sessions: tp.Dict[str, str], evaluation: int) -> tp.List[str]:
    """
    Description:
        The report identify prints: the identified values, the sensitivity rows and the validation
        rows per trim and session, in mm where a length.

    Args:
        (1) identification [Identification_Str(object)]: The identified model and its rows.
        (2) validation [List<Tuple<float, List<Fitted_Flight_Str>>>]: Per trim [s], the fitted validation flights.
        (3) sessions [Dict<string, string>]: The session of each flight, by Flight_Name.
        (4) evaluation [int]: This run's number in the ledger.

    Returns:
        (1) parameter [List<string>]: The lines.
    """

    model = identification.Flight_Model_Str
    lines = ['identified: k = %.6f 1/m, Drag_Std = %.6f 1/m (bootstrap sd %.6f, half-range %.6f), g = %.4f m/s^2'
             % (model.Drag, model.Drag_Std, identification.Drag_Bootstrap_Sd, identification.Drag_Half_Range, -model.Gravity[2]),
             'rows (k [1/m], flights segmented / identification flights, samples, RMS / max / floor median [mm]):']
    for variant in identification.Variants:
        rms, mx, floor = (1e3 * float(np.median(values)) for values in ([f.RMS for f in variant.Fitted], [f.Max for f in variant.Fitted],
                                                                      [Estimated_Floor(f.Flight.Time, f.Residual) for f in variant.Fitted]))
        lines.append('  %-18s tau %2.0f mm, trim %2.0f ms: k %.6f, %d / %d, %d, %.2f / %.1f / %.2f'
                     % (variant.Name, variant.Tolerance * 1e3, variant.Trim * 1e3, variant.Drag, variant.Flights_Segmented,
                        variant.Flights, variant.Samples, rms, mx, floor))
    lines.append('validation (evaluation %d; the model unchanged, each flight its initial state; RMS / max / floor median [mm]):' % evaluation)
    for trim, fitted in validation:
        for name, group in Validation_Groups(fitted, sessions):
            values = [float(v) * 1e3 for v in _Statistics(group)[:3]]
            lines.append('  trim %2.0f ms  %-48s %2d flights: %.2f / %.1f / %.2f' % (trim * 1e3, name, len(group), *values))
    return lines
