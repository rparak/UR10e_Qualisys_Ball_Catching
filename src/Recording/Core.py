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
# File Name: ../Recording/Core.py
## =========================================================================== ##

# System (Default)
import os
# Math (Mathematical functions)
import math
# Hashlib (Secure hashes)
import hashlib
# Regular expressions (Default)
import re
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the assumed transform to the robot base frame
from Configuration.Core import Transform_Str
#   ../Geometry/Core - the rotation matrices of the areas
from Geometry.Core import Rotation_Matrix_X, Rotation_Matrix_Y, Rotation_Matrix_Z

"""
Description:
    Initialization of constants.

    The only reader of the recordings (SPEC §7.3, §9.2): Data/Recordings/Frequency_<rate>_Hz_Active_
    Filtering_<ON|OFF>/Experiment_<n>.txt. It rebuilds the time from the frame count (the Time column
    holds only the nanoseconds within the second), reports every gap, zero step, irregular step and
    ambiguous step, and splits a recording into pieces of continuous time at every ambiguous step.

    Lines: a sample j (0-based) is on the file's line j + 2 (the header is line 1); a step is
    reported at the line of its later sample. It never prints and never writes.

    It also assigns each recording to the identification or the validation set (SPEC §7.3), from
    its folder and file name and a fixed seed only, and maps positions into the robot base frame by
    the assumed transform.
"""
# The header of every recording (SPEC §9.2).
#   Unit [string]
CONST_HEADER = 'Time,Data_X,Data_Y,Data_Z'
# The folder and file names of the recordings (SPEC §9.2): the rate and the QTM active-filtering
# setting come from the folder name, the experiment number from the file name.
#   Unit [re.Pattern]
CONST_FOLDER_PATTERN = re.compile(r'^Frequency_(\d+)_Hz_Active_Filtering_(ON|OFF)$')
CONST_FILE_PATTERN = re.compile(r'^Experiment_(\d+)\.txt$')
# Nanoseconds in a second: the Time column wraps at this value (SPEC §4.2).
#   Unit [int]
CONST_NS_PER_SECOND = 1000000000
# A step of at least this many nominal periods is a gap (SPEC §7.3). The bound is
# applied to the raw ratio r, never to the rounded frame count.
#   Unit [float]
CONST_GAP_PERIODS = 1.5
# A step whose ratio r is further than this fraction of a period from a whole multiple is irregular
# (SPEC §7.3). Measured: the worst step of the 90 recordings is 0.00029 periods off (SPEC §4.2).
#   Unit [float]
CONST_IRREGULAR_FRACTION = 0.05
# The continuity rule: a step whose jump |dp| exceeds SPEED * dt + ALLOWANCE
# cannot be travelled by the ball in the rebuilt time dt, so a hidden gap of >= 1 s is assumed.
# Measured on the 90 recordings (F2 Part A, 2026-11-09): the fastest sustained real motion is
# 8.96 m/s (300 Hz OFF Experiment 10, landing); the largest excess of a real step over 10 m/s * dt is
# 0.008 m (a stale sample catching up), 0.012 m below the allowance; the smallest flagged excess is
# 0.024 m (100 Hz OFF Experiment 5 line 596). Hidden gaps implying less than 10 m/s are not found by
# the rule: see CONST_CITED_HIDDEN_GAPS.
#   Unit [float, m/s], [float, m]
CONST_CONTINUITY_SPEED = 10.0
CONST_CONTINUITY_ALLOWANCE = 0.02
# Hidden gaps found by inspection that the rule misses (source: F2;
# no measured criterion separates hidden gaps from real motion on
# positions alone, so these are listed by file and line, each with its reason). The list is not
# provably complete.
#   Unit [Dict<Tuple<string, int, int>, string>] (folder, experiment, line) -> reason
CONST_CITED_HIDDEN_GAPS = {
    ('Frequency_200_Hz_Active_Filtering_OFF', 25, 2035): 'at rest for 443 steps, then 0.68 m higher 85 ms later '
                                                          '(about 190 m/s^2 from a standstill)',
    ('Frequency_100_Hz_Active_Filtering_ON', 7, 558): '0.92 m horizontally in 0.1 s while moving about 1 m/s '
                                                       'horizontally before and after',
    ('Frequency_100_Hz_Active_Filtering_ON', 10, 556): 'after the floor contact, 2.4 m horizontally in 0.3 s at '
                                                        '3.6 m/s horizontal before (a bounce adds no horizontal speed)'}
# The kinds of a reported step, and the reasons of an ambiguous one.
#   Unit [string]
CONST_KIND_GAP = 'gap'
CONST_KIND_ZERO = 'zero'
CONST_KIND_IRREGULAR = 'irregular'
CONST_KIND_AMBIGUOUS = 'ambiguous'
CONST_REASON_ZERO = 'zero step'
CONST_REASON_CONTINUITY = 'continuity'
CONST_REASON_CITED = 'cited'
# The flight split: per folder, the files are ranked by the SHA-256
# hex digest of '<seed>/<folder>/<file name>' and the first half (len // 2) is the identification
# set, the rest validation. The seed was fixed in the F2 addendum (2026-11-09) before any answer and
# is never changed: a new seed after seeing the split would be a seed search.
#   Unit [int]
CONST_SPLIT_SEED = 20261109
# The two sets.
#   Unit [string]
CONST_SET_IDENTIFICATION = 'Identification'
CONST_SET_VALIDATION = 'Validation'


class Recording_Error_Cls(ValueError):
    """
    Description:
        The recordings cannot be used. The message lists every problem found, each with its file and
        line, and says what is expected; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Step_Str:
    """
    Description:
        One reported step between two consecutive samples of a recording (SPEC §7.3).
    """

    # The file line of the later sample of the step (the header is line 1).
    #   Unit [int]
    Line: int = 0
    # One of CONST_KIND_GAP, CONST_KIND_ZERO, CONST_KIND_IRREGULAR, CONST_KIND_AMBIGUOUS.
    #   Unit [string]
    Kind: str = ''
    # For an ambiguous step: CONST_REASON_ZERO, CONST_REASON_CONTINUITY, or CONST_REASON_CITED followed
    # by ': ' and the cited reason; '' for the other kinds.
    #   Unit [string]
    Reason: str = ''
    # The step in nominal periods, from the Time column: ((ns_k - ns_(k-1)) mod 1e9) / (1e9 T).
    #   Unit [float, -]
    Ratio: float = 0.0
    # The step in frames, floor(Ratio + 0.5) (half up).
    #   Unit [int]
    Frames: int = 0
    # The distance between the two positions.
    #   Unit [float, m]
    Jump: float = 0.0


@dataclass
class Recording_Str:
    """
    Description:
        One recording with its time rebuilt from the frame count (SPEC §7.3). Time is meaningful only
        within one piece: across an ambiguous step the true time is unknown (at least 1 s more than
        the frame count says), so a time difference is never taken across a piece boundary.
    """

    # The folder, e.g. 'Frequency_200_Hz_Active_Filtering_OFF'.
    #   Unit [string]
    Folder: str = ''
    # The experiment number n of Experiment_<n>.txt.
    #   Unit [int]
    Experiment: int = 0
    # The nominal sample rate set in QTM.
    #   Unit [float, Hz]
    Rate: float = 0.0
    # QTM's active filtering (the folder's ON / OFF).
    #   Unit [bool]
    Filtering: bool = False
    # The frame index of each sample: 0 for the first, then the sum of the steps' frames.
    #   Unit [Vector<int> 1xN]
    Frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The piece of continuous time of each sample: 0 for the first, one more after each ambiguous step.
    #   Unit [Vector<int> 1xN]
    Piece: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # The position of the 6DOF body "Ball" in the QTM frame.
    #   Unit [Matrix<float> Nx3, m]
    Position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # The reported steps (gaps, zero, irregular and ambiguous steps), in line order.
    #   Unit [List<Step_Str>]
    Steps: tp.List[Step_Str] = field(default_factory=list)
    # How often the Time column wraps from the end of a second to the next.
    #   Unit [int]
    Wraps: int = 0

    @property
    def Period(self) -> float:
        """
        Description:
            The nominal period 1 / Rate.

        Returns:
            (1) parameter [float]: The period [s].
        """

        return 1.0 / self.Rate

    @property
    def Time(self) -> np.ndarray:
        """
        Description:
            The rebuilt time of each sample, Frame * Period, from the first sample.

        Returns:
            (1) parameter [Vector<float> 1xN]: The time [s].
        """

        return self.Frame * self.Period

    @property
    def File(self) -> str:
        """
        Description:
            The recording's path below Data/Recordings/, as the reports name it.

        Returns:
            (1) parameter [string]: '<folder>/Experiment_<n>.txt'.
        """

        return '%s/Experiment_%d.txt' % (self.Folder, self.Experiment)


def Parse_Recording_Name(folder: str, file_name: str) -> tp.Tuple[float, bool, int]:
    """
    Description:
        The rate, the active-filtering setting and the experiment number from a recording's folder and
        file names (SPEC §9.2).

    Args:
        (1) folder [string]: The folder name, e.g. 'Frequency_200_Hz_Active_Filtering_OFF'.
        (2) file_name [string]: The file name, e.g. 'Experiment_23.txt'.

    Returns:
        (1) parameter [Tuple<float, bool, int>]: The rate [Hz], the filtering, the experiment number.
    """

    folder_match = CONST_FOLDER_PATTERN.match(folder)
    file_match = CONST_FILE_PATTERN.match(file_name)
    if folder_match is None or file_match is None:
        raise Recording_Error_Cls('%s/%s: the name is not Frequency_<rate>_Hz_Active_Filtering_<ON|OFF>/Experiment_<n>.txt '
                                  '(SPEC §9.2). Rename it or move it out of the recordings folder.' % (folder, file_name))
    return float(folder_match.group(1)), folder_match.group(2) == 'ON', int(file_match.group(1))


def _Read_File(path: str) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[str]]:
    """
    Description:
        Read one recording's text: the header, then rows of 4 numbers, the first an integer number of
        nanoseconds in [0, 1e9). Every problem is collected with its line.

    Args:
        (1) path [string]: The recording file.

    Returns:
        (1) parameter 1 [Vector<int> 1xN]: The Time column [ns].
        (2) parameter 2 [Matrix<float> Nx3]: The positions [m].
        (3) parameter 3 [List<string>]: The problems, each a sentence naming the line.
    """

    try:
        with open(path, 'rb') as file:
            text = file.read().decode('ascii')
    except (OSError, UnicodeDecodeError) as error:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3)), ['the file cannot be read as ASCII text (%s).' % error]
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    if not lines:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3)), ['the file is empty; expected the header %s and samples.' % CONST_HEADER]
    problems: tp.List[str] = []
    if lines[0].rstrip('\r') != CONST_HEADER:
        problems.append('line 1: the header is %r; expected %s.' % (lines[0], CONST_HEADER))
    if len(lines) == 1:
        problems.append('the file holds no samples after the header.')
    ns: tp.List[int] = []
    position: tp.List[tp.List[float]] = []
    for index, line in enumerate(lines[1:]):
        number = index + 2
        fields = line.split(',')
        if len(fields) != 4:
            problems.append('line %d: %d field(s); expected 4 numbers (Time, Data_X, Data_Y, Data_Z).' % (number, len(fields)))
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            problems.append('line %d: %r is not 4 numbers.' % (number, line))
            continue
        if not all(math.isfinite(value) for value in values):
            problems.append('line %d: %r holds a value that is not finite.' % (number, line))
            continue
        if values[0] != math.floor(values[0]) or not (0 <= values[0] < CONST_NS_PER_SECOND):
            problems.append('line %d: Time %r is not a whole number of nanoseconds in [0, 1e9).' % (number, fields[0]))
            continue
        ns.append(int(values[0]))
        position.append(values[1:])
    return np.array(ns, dtype=np.int64), np.array(position, dtype=float).reshape(-1, 3), problems


def Build_Recording(folder: str, experiment: int, rate: float, filtering: bool, ns: np.ndarray, position: np.ndarray) -> Recording_Str:
    """
    Description:
        Rebuild the time of a recording from its Time column and find its steps (SPEC §7.3):
            r_k = ((ns_k - ns_(k-1)) mod 1e9) / (1e9 T),  n_k = floor(r_k + 0.5),  t_k = t_(k-1) + n_k T.
        A gap is r_k >= 1.5; a zero step n_k = 0; an irregular step |r_k - n_k| > 0.05; an ambiguous step
        a zero step, a step breaking the continuity rule, or a cited hidden gap. Each ambiguous step
        starts a new piece of time.

    Args:
        (1) folder [string]: The folder name (it names the recording in the cited list).
        (2) experiment [int]: The experiment number.
        (3) rate [float]: The nominal rate [Hz].
        (4) filtering [bool]: QTM's active filtering.
        (5) ns [Vector<int> 1xN]: The Time column [ns].
        (6) position [Matrix<float> Nx3]: The positions [m].

    Returns:
        (1) parameter [Recording_Str(object)]: The recording.
    """

    period = 1.0 / rate
    d = np.mod(np.diff(ns.astype(np.int64)), CONST_NS_PER_SECOND)
    ratio = d / (CONST_NS_PER_SECOND * period)
    frames = np.floor(ratio + 0.5).astype(np.int64)
    jump = np.linalg.norm(np.diff(position, axis=0), axis=1)
    cited = {line: reason for (cited_folder, cited_experiment, line), reason in CONST_CITED_HIDDEN_GAPS.items()
             if cited_folder == folder and cited_experiment == experiment}
    for line in sorted(cited):
        if not (3 <= line <= len(ns) + 1):
            raise Recording_Error_Cls('%s/Experiment_%d.txt: the cited hidden gap at line %d is not a step of the file '
                                      '(lines 3 to %d). Correct CONST_CITED_HIDDEN_GAPS.' % (folder, experiment, line, len(ns) + 1))

    steps: tp.List[Step_Str] = []
    ambiguous = np.zeros(len(d), dtype=bool)
    for k in range(len(d)):
        line = k + 3
        kinds: tp.List[tp.Tuple[str, str]] = []
        if ratio[k] >= CONST_GAP_PERIODS:
            kinds.append((CONST_KIND_GAP, ''))
        if frames[k] == 0:
            kinds.append((CONST_KIND_ZERO, ''))
        if abs(ratio[k] - frames[k]) > CONST_IRREGULAR_FRACTION:
            kinds.append((CONST_KIND_IRREGULAR, ''))
        # An ambiguous step's reason, in this order: a zero step, the continuity rule, the cited list.
        if frames[k] == 0:
            reason = CONST_REASON_ZERO
        elif jump[k] > CONST_CONTINUITY_SPEED * frames[k] * period + CONST_CONTINUITY_ALLOWANCE:
            reason = CONST_REASON_CONTINUITY
        elif line in cited:
            reason = '%s: %s' % (CONST_REASON_CITED, cited[line])
        else:
            reason = ''
        if reason:
            ambiguous[k] = True
            kinds.append((CONST_KIND_AMBIGUOUS, reason))
        for kind, why in kinds:
            steps.append(Step_Str(Line=line, Kind=kind, Reason=why, Ratio=float(ratio[k]), Frames=int(frames[k]), Jump=float(jump[k])))

    return Recording_Str(Folder=folder, Experiment=experiment, Rate=float(rate), Filtering=filtering,
                         Frame=np.concatenate(([0], np.cumsum(frames))).astype(np.int64),
                         Piece=np.concatenate(([0], np.cumsum(ambiguous))).astype(np.int64),
                         Position=np.asarray(position, dtype=float), Steps=steps, Wraps=int(np.sum(np.diff(ns) < 0)))


def _Load(path: str) -> tp.Tuple[tp.Optional[Recording_Str], tp.List[str]]:
    """
    Description:
        Load one recording, collecting its problems instead of raising them.

    Args:
        (1) path [string]: The recording file, inside its rate folder.

    Returns:
        (1) parameter 1 [Recording_Str(object)]: The recording, or None when it has a problem.
        (2) parameter 2 [List<string>]: The problems, each prefixed with '<folder>/<file>: '.
    """

    folder, file_name = os.path.basename(os.path.dirname(path)), os.path.basename(path)
    name = '%s/%s' % (folder, file_name)
    try:
        rate, filtering, experiment = Parse_Recording_Name(folder, file_name)
    except Recording_Error_Cls as error:
        return None, [str(error)]
    ns, position, problems = _Read_File(path)
    if problems:
        return None, ['%s: %s' % (name, problem) for problem in problems]
    try:
        return Build_Recording(folder, experiment, rate, filtering, ns, position), []
    except Recording_Error_Cls as error:
        return None, [str(error)]


def Load_Recording(path: str) -> Recording_Str:
    """
    Description:
        Load one recording file (SPEC §9.2); every problem of the file is listed in one error.

    Args:
        (1) path [string]: The recording file, inside its rate folder.

    Returns:
        (1) parameter [Recording_Str(object)]: The recording.
    """

    recording, problems = _Load(path)
    if problems:
        raise Recording_Error_Cls('The recording cannot be used (%d problem(s)):\n  %s' % (len(problems), '\n  '.join(problems)))
    return recording


def Load_Recordings(folder: str) -> tp.List[Recording_Str]:
    """
    Description:
        Load every recording below the recordings folder (SPEC §9.2), ordered by folder name, then by
        experiment number. Every problem of every file (and every entry that is not a recording) is
        collected first; then the load fails once, listing all of them, and nothing is returned.

    Args:
        (1) folder [string]: The recordings folder, e.g. Data/Recordings.

    Returns:
        (1) parameter [List<Recording_Str>]: The recordings.
    """

    if not os.path.isdir(folder):
        raise Recording_Error_Cls('The recordings folder "%s" does not exist. Check the path (SPEC §9.2).' % folder)
    problems: tp.List[str] = []
    recordings: tp.List[Recording_Str] = []
    for rate_folder in sorted(os.listdir(folder)):
        if rate_folder.startswith('.'):
            continue
        path = os.path.join(folder, rate_folder)
        if not os.path.isdir(path) or CONST_FOLDER_PATTERN.match(rate_folder) is None:
            problems.append('%s: not a folder named Frequency_<rate>_Hz_Active_Filtering_<ON|OFF> (SPEC §9.2). '
                            'Move it out of the recordings folder.' % rate_folder)
            continue
        names = [name for name in os.listdir(path) if not name.startswith('.')]
        for name in sorted(names, key=lambda n: (0, int(CONST_FILE_PATTERN.match(n).group(1))) if CONST_FILE_PATTERN.match(n) else (1, n)):
            recording, file_problems = _Load(os.path.join(path, name))
            problems.extend(file_problems)
            if recording is not None:
                recordings.append(recording)
    if problems:
        raise Recording_Error_Cls('The recordings cannot be used (%d problem(s)):\n  %s' % (len(problems), '\n  '.join(problems)))
    if not recordings:
        raise Recording_Error_Cls('The recordings folder "%s" holds no recording (SPEC §9.2).' % folder)
    return recordings


def Assign_Sets(names: tp.List[tp.Tuple[str, str]]) -> tp.Dict[tp.Tuple[str, str], str]:
    """
    Description:
        The flight split (SPEC §7.3): per folder, rank the files by the SHA-256 hex
        digest of '<CONST_SPLIT_SEED>/<folder>/<file name>'; the first half (len // 2) is the
        identification set, the rest the validation set. Only the names are used: no file is opened,
        no quantity of the data enters.

    Args:
        (1) names [List<Tuple<string, string>>]: (folder, file name) of every recording.

    Returns:
        (1) parameter [Dict<Tuple<string, string>, string>]: (folder, file name) -> CONST_SET_IDENTIFICATION
                                                             or CONST_SET_VALIDATION.
    """

    sets: tp.Dict[tp.Tuple[str, str], str] = {}
    for folder in sorted({folder for folder, _ in names}):
        files = sorted(file_name for name_folder, file_name in names if name_folder == folder)
        ranked = sorted(files, key=lambda file_name: hashlib.sha256(('%d/%s/%s' % (CONST_SPLIT_SEED, folder, file_name))
                                                                     .encode('ascii')).hexdigest())
        for rank, file_name in enumerate(ranked):
            sets[(folder, file_name)] = CONST_SET_IDENTIFICATION if rank < len(ranked) // 2 else CONST_SET_VALIDATION
    return sets


def Transform_To_Robot(position: np.ndarray, *, Transform_Str: Transform_Str) -> np.ndarray:
    """
    Description:
        Positions from the recordings' frame (QTM) into the robot base frame by the assumed transform
        of SPEC §7.3: p_robot = R p_QTM + Position, R = R_x R_y R_z of the orientation (as the areas,
        SPEC §7.2). Every result on real flights rests on this assumption.

    Args:
        (1) position [Matrix<float> Nx3]: Positions in the QTM frame [m].
        (2) Transform_Str [Transform_Str(object)]: REQUIRED, keyword-only: the assumed transform.

    Returns:
        (1) parameter [Matrix<float> Nx3]: The positions in the robot base frame [m].
    """

    alpha, beta, gamma = Transform_Str.Orientation
    R = Rotation_Matrix_X(alpha) @ Rotation_Matrix_Y(beta) @ Rotation_Matrix_Z(gamma)
    return np.atleast_2d(position) @ R.T + np.asarray(Transform_Str.Position, dtype=float)
