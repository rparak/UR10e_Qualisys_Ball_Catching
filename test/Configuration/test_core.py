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
# File Name: ../test/Configuration/test_core.py
## =========================================================================== ##

# System (Default)
import os
# Math (Mathematical functions)
import math
# PyYAML (YAML parser and emitter) [pip3 install pyyaml]
import yaml
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the configuration loader under test (its field table changed only to stand for a later version of the file)
#   ../Configuration/Core
from Configuration.Core import Load_Configuration, Load_Dataset_Configuration, Load_Training_Configuration, Configuration_Error_Cls, CONST_FIELDS

"""
Description:
    Initialization of constants.

    The loader is tested on mutated copies of the shipped file.
    The required fields are written out here, from SPEC §8, never read from the loader.
"""
# Every REQUIRED field of SPEC §8, by its full key path.
#   Unit [Tuple<string>]
CONST_REQUIRED = ('Areas.Throw.Centre', 'Areas.Throw.Size', 'Areas.Flight.Centre', 'Areas.Flight.Size',
                  'Areas.Catch.Centre', 'Areas.Catch.Size', 'Areas.Robot.Shape', 'Areas.Robot.Centre',
                  'Areas.Robot.Orientation', 'Areas.Robot.Radius', 'Transform.Orientation', 'Transform.Position',
                  'Recording.Summary_File', 'Recording.Steps_File', 'Flight.Gravity', 'Flight.Drag', 'Flight.Drag_Std',
                  'Identification.Summary_File', 'Identification.Flights_File', 'Identification.Ledger_File',
                  'Observation.White_Std', 'Observation.White_Dof', 'Observation.Drift_Std', 'Observation.Drift_Time',
                  'Observation.Offset_Std', 'Observation.Offset_Time', 'Observation.Offset_Dof', 'Observation.Offset_Bound',
                  'Observation.White_Bound', 'Observation.Summary_File',
                  'Robot.Joint_Speed', 'Robot.Joint_Accelerations', 'Robot.Ready_Pose', 'Catch.Joint_Margin', 'Catch.Clearance',
                  'Catch.Speed_Margin', 'Catch.Arrival_Elevation', 'Catch.Arrival_Azimuth', 'Catch.Workspace_Step', 'Workspace.Dataset',
                  'Workspace.Map_File', 'Workspace.Summary_File', 'Decision.Datasets', 'Decision.Calibration', 'Decision.Quantile',
                  'Decision.Stages', 'Decision.Output_Folder', 'Interception.Stages', 'Interception.Calibration',
                  'Interception.Latency', 'Interception.Compute_Budget', 'Interception.Guard', 'Interception.Path_Step',
                  'Interception.Angle_Tolerance', 'Interception.Catch_Mode', 'Interception.Stroke_Scales',
                  'Interception.Stroke_Compute_Budget', 'Interception.Output_Folder',
                  'Contact.Step', 'Contact.Time_Constant', 'Contact.Damping_Ratio', 'Contact.Friction', 'Contact.Settle_Time',
                  'Contact.Rest_Speed', 'Contact.Rest_Time', 'Contact.Levels', 'Contact.Sweep_Scales', 'Contact.Sweep_Throws',
                  'Contact.Output_Folder', 'Contact.Ledger_File')
# Every REQUIRED field of the dataset configuration, SPEC §8.2.
#   Unit [Tuple<string>]
CONST_DATASET_REQUIRED = ('Dataset.Name', 'Dataset.Seed', 'Dataset.Flights', 'Dataset.System_Configuration', 'Partitions.Train',
                          'Partitions.Validation', 'Partitions.Test', 'Throw.Release_Mean', 'Throw.Release_Std', 'Throw.Apex_Mean',
                          'Throw.Apex_Std', 'Throw.Flight_Time', 'Throw.Release_Speed', 'Throw.Target_Area.Centre',
                          'Throw.Target_Area.Orientation', 'Throw.Target_Area.Radius', 'Observation.Rates', 'Observation.Noise.Scale',
                          'Observation.Dropout.Pattern', 'Observation.Dropout.Donors', 'Output.Folder')

# Every REQUIRED field of the training configuration of F7, SPEC §8.3 (the learning fields join with F8).
#   Unit [Tuple<string>]
CONST_TRAINING_REQUIRED = ('Run.Name', 'Run.Datasets', 'Input.Observed_Samples', 'Model.Name', 'Output.Folder')
# The default dataset's SHA-256 (scripts/catch.py dataset at 6be743d, 2026-11-16: 10 000 flights with the bounded error
# model and the targets of Throw.Target_Area; before, at a30fad8, e39b4bc7...) and a well-formed other hash.
#   Unit [string, hex]
CONST_DEFAULT_HASH = '916ea9fcdefb246d5da94119b73dac6446792bd6a81a214c1c53f9edf6b8b74a'
CONST_OTHER_HASH = '0' * 64


def _Shipped(root):
    return os.path.join(root, 'Config', 'ball_catching.yaml')


def _Shipped_Dataset(root):
    return os.path.join(root, 'Config', 'dataset.yaml')


def _Write_Mutated(root, tmp_path, mutate, shipped=None):
    """
    Description:
        Load the shipped file raw (the system file unless another is named), apply mutate(dict), write
        the copy and return its path.
    """

    with open(shipped or _Shipped(root), 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    mutate(data)
    path = tmp_path / 'mutated.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return str(path)


def _Set(data, key_path, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        data = data[key]
    data[keys[-1]] = value


def _Delete(data, key_path):
    keys = key_path.split('.')
    for key in keys[:-1]:
        data = data[key]
    del data[keys[-1]]


def test_the_shipped_configuration_loads_in_canonical_units(root):
    """
    Description:
        The shipped file loads without notes; the owner's catch hemisphere (SPEC §4.4) comes back with
        its orientation converted from [deg] to [rad].
    """

    configuration = Load_Configuration(_Shipped(root))
    robot = configuration.Areas.Robot
    assert configuration.Notes == []
    assert robot.Shape == 'hemisphere' and robot.Radius == 1.0 and robot.Centre == [0.0, 0.0, 0.3607]
    assert robot.Orientation == pytest.approx([0.0, 0.0, -math.pi / 2.0], abs=1e-15)
    # The areas of the frame decision (SPEC §8.1): the Throw cuboid from the
    # identification releases, the Flight cuboid the 2022 Blender area at -y.
    assert configuration.Areas.Throw.Centre == [-0.125, -5.25, 1.275] and configuration.Areas.Throw.Size == [3.67, 2.42, 0.91]
    assert configuration.Areas.Flight.Centre == [0.0, -3.0, 2.25] and configuration.Areas.Flight.Size == [4.0, 4.0, 3.5]
    # The assumed transform of SPEC §7.3: Rz(180 deg), then (0.827, -4.670, 0.000) m.
    assert configuration.Transform.Orientation == pytest.approx([0.0, 0.0, math.pi], abs=1e-15)
    assert configuration.Transform.Position == [0.827, -4.670, 0.0]
    assert configuration.Recording.Summary_File == 'Data/Generated/Recordings/recording_summary.csv'
    assert configuration.Recording.Steps_File == 'Data/Generated/Recordings/recording_steps.csv'
    assert configuration.Flight.Gravity == 9.81
    assert configuration.Identification.Summary_File == 'Data/Generated/Identification/identification_summary.csv'
    assert configuration.Identification.Flights_File == 'Data/Generated/Identification/identification_flights.csv'
    assert configuration.Identification.Ledger_File == 'Data/Generated/Identification/identification_ledger.csv'
    assert configuration.Observation.Summary_File == 'Data/Generated/Observation/observation_summary.csv'
    assert len(configuration.Observation.White_Std) == 3 and configuration.Observation.White_Dof > 2.0
    # F4: the joint speeds of the datasheet and the accelerations from [deg] to [rad]; the catch's margins.
    assert configuration.Robot.Joint_Speed == pytest.approx([math.radians(v) for v in (120, 120, 180, 180, 180, 180)], abs=1e-15)
    assert configuration.Robot.Joint_Accelerations == pytest.approx([math.radians(800.0), math.radians(400.0)], abs=1e-15)
    assert len(configuration.Robot.Ready_Pose) == 6 and configuration.Robot.Ready_Pose[5] == 0.0
    assert configuration.Catch.Joint_Margin == pytest.approx(math.radians(10.0), abs=1e-15)
    assert configuration.Catch.Clearance == 0.05 and configuration.Catch.Speed_Margin == 0.5 and configuration.Catch.Workspace_Step == 0.05
    assert configuration.Catch.Arrival_Elevation == pytest.approx([math.radians(44.0), math.radians(68.0)], abs=1e-15)
    assert configuration.Catch.Arrival_Azimuth == pytest.approx([math.radians(-25.0), math.radians(27.0)], abs=1e-15)
    assert configuration.Workspace.Dataset == 'Data/Generated/Datasets/Default'
    assert configuration.Workspace.Map_File == 'Data/Generated/Workspace/workspace_map.npz'


@pytest.mark.parametrize('key_path', CONST_REQUIRED)
def test_a_missing_required_field_fails_naming_it(root, tmp_path, key_path):
    """
    Description:
        Every field is REQUIRED: removing one fails the load with its full key path in the message.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Delete(data, key_path))
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(path)
    assert '%s is missing' % key_path in str(error.value)


def test_every_problem_is_listed_at_once(root, tmp_path):
    """
    Description:
        Two invalid fields fail one load that names both, rather than the first only.
    """

    def Mutate(data):
        _Set(data, 'Areas.Throw.Size', [4.0, 0.0, 1.5])
        _Set(data, 'Areas.Robot.Radius', -1.0)
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, Mutate))
    message = str(error.value)
    assert '2 problem(s)' in message and 'Areas.Throw.Size' in message and 'Areas.Robot.Radius' in message


@pytest.mark.parametrize('key_path, value', [('Areas.Robot.Radius', 0.0), ('Areas.Robot.Radius', -1.0),
                                             ('Areas.Catch.Size', [4.0, 2.0, 0.0]), ('Areas.Flight.Centre', [0.0, 3.0]),
                                             ('Areas.Throw.Centre', 'origin'), ('Areas.Robot.Shape', 'cube'),
                                             ('Areas.Robot.Orientation', [0.0, 0.0, True]),
                                             ('Transform.Position', [0.827, -4.670]), ('Transform.Orientation', 'half turn'),
                                             ('Recording.Summary_File', ''), ('Recording.Steps_File', '/tmp/steps.csv'),
                                             ('Flight.Gravity', 0.0), ('Flight.Drag', -0.001), ('Flight.Drag_Std', 'small'),
                                             ('Identification.Ledger_File', ''), ('Observation.White_Dof', 2.0),
                                             ('Observation.Offset_Std', [0.0, -1e-3, 0.0]), ('Observation.Drift_Time', 0.0),
                                             ('Robot.Joint_Speed', [120.0] * 5), ('Robot.Joint_Speed', [120.0, 0.0, 180.0, 180.0, 180.0, 180.0]),
                                             ('Robot.Joint_Accelerations', [800.0]), ('Robot.Joint_Accelerations', [800.0, -400.0]),
                                             ('Robot.Ready_Pose', [0.0, 0.0, 361.0, 0.0, 0.0, 0.0]), ('Robot.Ready_Pose', 'home'),
                                             ('Catch.Joint_Margin', 0.0), ('Catch.Clearance', -0.05), ('Catch.Speed_Margin', 0.0),
                                             ('Catch.Arrival_Elevation', [68.0, 44.0]), ('Catch.Arrival_Elevation', [-1.0, 44.0]),
                                             ('Catch.Arrival_Elevation', [44.0, 91.0]), ('Catch.Arrival_Azimuth', [-181.0, 27.0]),
                                             ('Catch.Arrival_Azimuth', [27.0, 27.0]), ('Catch.Workspace_Step', 0.0),
                                             ('Workspace.Dataset', ''), ('Workspace.Map_File', '/tmp/map.npz')])
def test_an_invalid_value_fails_and_never_falls_back(root, tmp_path, key_path, value):
    """
    Description:
        A present but invalid value fails the load naming the field; no default replaces it.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, value)))
    assert key_path in str(error.value)


def test_a_positive_bound_is_tested_on_both_sides(root, tmp_path):
    """
    Description:
        Radius must be > 0: 1e-9 m is accepted, 0 m is refused.
    """

    assert Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Areas.Robot.Radius', 1e-9))).Areas.Robot.Radius == 1e-9
    with pytest.raises(Configuration_Error_Cls):
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Areas.Robot.Radius', 0)))


def test_the_drag_bound_is_tested_on_both_sides(root, tmp_path):
    """
    Description:
        Flight.Drag must be >= 0 (0 is the vacuum model): 0 is accepted, -1e-12 1/m is refused naming
        the field.
    """

    assert Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Flight.Drag', 0.0))).Flight.Drag == 0.0
    with pytest.raises(Configuration_Error_Cls, match='Flight.Drag must be a number >= 0'):
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Flight.Drag', -1e-12)))


def test_a_repeated_key_is_refused_naming_both_lines(root, tmp_path):
    """
    Description:
        A key written twice in one mapping is refused with both lines (PyYAML alone keeps the last).
    """

    text = open(_Shipped(root), 'r', encoding='utf-8').read()
    text = text.replace('    Radius: 1.0\n', '    Radius: 1.0\n    Radius: 2.0\n')
    path = tmp_path / 'repeated.yaml'
    path.write_text(text, encoding='utf-8')
    radius_line = text.splitlines().index('    Radius: 1.0') + 1
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(str(path))
    assert 'line %d: the key Areas.Robot.Radius appears again (first on line %d)' % (radius_line + 1, radius_line) in str(error.value)


@pytest.mark.parametrize('word', ['yes', 'on', 'ON', 'no', 'off'])
def test_only_true_and_false_are_booleans(root, tmp_path, word):
    """
    Description:
        A word PyYAML would turn into a boolean stays text: as a shape it is reported as the text
        found, not as True or False.
    """

    text = open(_Shipped(root), 'r', encoding='utf-8').read().replace('Shape: hemisphere', 'Shape: %s' % word)
    path = tmp_path / 'word.yaml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(str(path))
    assert "found '%s'" % word in str(error.value)


def test_an_unknown_key_is_a_warning_not_a_failure(root, tmp_path):
    """
    Description:
        An unknown key (a likely misspelling) loads, with one WARNING note naming it.
    """

    configuration = Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Areas.Robot.Radios', 1.0)))
    assert len(configuration.Notes) == 1
    level, message = configuration.Notes[0]
    assert level == 'WARNING' and 'Areas.Robot.Radios is unknown' in message


def test_an_unreadable_file_fails_with_a_sentence(tmp_path):
    """
    Description:
        A path that does not exist fails with a sentence naming it and what to check.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(str(tmp_path / 'absent.yaml'))
    assert 'absent.yaml' in str(error.value) and '--config' in str(error.value)


def test_the_dof_bound_is_tested_on_both_sides(root, tmp_path):
    """
    Description:
        A Student-t's degrees of freedom must exceed 2 (it has a variance): 2.000001 is accepted, 2 is
        refused naming the field.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Observation.Offset_Dof', 2.000001))
    assert Load_Configuration(path).Observation.Offset_Dof == 2.000001
    with pytest.raises(Configuration_Error_Cls, match='Observation.Offset_Dof must be a number > 2'):
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Observation.Offset_Dof', 2)))


def test_the_error_bounds_are_tested_on_both_sides(root, tmp_path):
    """
    Description:
        Observation.Offset_Bound and White_Bound must be > 0: 1e-9 m is accepted, 0 m is
        refused naming the field (a bound of 0 could be met by no draw). The shipped values are the owner's:
        0.070 and 0.040 m.
    """

    shipped = Load_Configuration(_Shipped(root)).Observation
    assert (shipped.Offset_Bound, shipped.White_Bound) == (0.070, 0.040)
    for key_path in ('Observation.Offset_Bound', 'Observation.White_Bound'):
        path = _Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, 1e-9))
        assert getattr(Load_Configuration(path).Observation, key_path.split('.')[1]) == 1e-9
        with pytest.raises(Configuration_Error_Cls, match='%s must be a number > 0' % key_path):
            Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, 0)))


def test_a_stored_copy_loads_its_four_sections_after_a_later_required_field(root, tmp_path, monkeypatch):
    """
    Description:
        A dataset's stored copy written before a REQUIRED field was added (SPEC §9.5): the copy
        holds the shipped file's bytes; then a field joins the loader's table, in a new section and in Robot (the
        table is changed here only to stand for a later version of the file). The full load refuses the older copy,
        naming both: the reader of the stored copies before this change. The stored copy's load gives the same areas,
        transform, flight model and error model as the full load before the field joined, without a note.
    """

    copy = tmp_path / 'ball_catching.yaml'
    copy.write_bytes(open(_Shipped(root), 'rb').read())
    before = Load_Configuration(str(copy))
    monkeypatch.setitem(CONST_FIELDS, 'Later.Speed', 'positive')
    monkeypatch.setitem(CONST_FIELDS, 'Robot.Later_Limit', 'positive')
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(str(copy))
    assert 'Later.Speed is missing' in str(error.value) and 'Robot.Later_Limit is missing' in str(error.value)
    stored = Load_Configuration(str(copy), stored_copy=True)
    assert (stored.Areas, stored.Transform, stored.Flight, stored.Observation) == (before.Areas, before.Transform, before.Flight,
                                                                                   before.Observation)
    assert stored.Notes == []


@pytest.mark.parametrize('key_path, value, reason', [('Observation.White_Dof', None, 'Observation.White_Dof is missing'),
                                                     ('Flight.Drag', -1.0, 'Flight.Drag must be'),
                                                     ('Areas.Robot.Shape', 'cube', 'Areas.Robot.Shape must be'),
                                                     ('Transform.Position', [0.0, 0.0], 'Transform.Position must be')])
def test_a_stored_copy_keeps_every_check_of_its_four_sections(root, tmp_path, key_path, value, reason):
    """
    Description:
        A stored copy is read only for Areas, Transform, Flight and Observation, but with every check of theirs: a missing
        or an invalid field of one of them refuses the copy, naming it (no default, no fallback).
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Delete(data, key_path) if value is None else _Set(data, key_path, value))
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(path, stored_copy=True)
    assert reason in str(error.value)


def test_a_stored_copy_refuses_a_later_field_of_its_sections_and_a_repeated_key_anywhere(root, tmp_path, monkeypatch):
    """
    Description:
        The limit of the stored copies' reader (SPEC §8): a REQUIRED field added to one of the four sections refuses an
        older copy, since a dataset depends on it; and the whole file is still read as YAML, so a key written twice in
        a section the copy is not read for (Robot) refuses it.
    """

    text = open(_Shipped(root), 'r', encoding='utf-8').read()
    assert text.count('  Ready_Pose: ') == 1
    repeated = tmp_path / 'repeated.yaml'
    repeated.write_text(text.replace('  Ready_Pose: ', '  Ready_Pose: [0, 0, 0, 0, 0, 0]\n  Ready_Pose: '), encoding='utf-8')
    with pytest.raises(Configuration_Error_Cls, match='the key Robot.Ready_Pose appears again'):
        Load_Configuration(str(repeated), stored_copy=True)
    monkeypatch.setitem(CONST_FIELDS, 'Flight.Later', 'positive')
    with pytest.raises(Configuration_Error_Cls, match='Flight.Later is missing'):
        Load_Configuration(_Shipped(root), stored_copy=True)


def test_a_stored_copy_reports_the_unknown_keys_of_its_four_sections_only(root, tmp_path):
    """
    Description:
        A stored copy's unknown key in one of the four sections is a WARNING, as in a full load; a key of a section the copy
        is not read for, known or not, is neither checked nor reported (a later version of the file may have dropped it).
    """

    def Mutate(data):
        _Set(data, 'Flight.Dreg', 0.1)
        _Set(data, 'Recording.Old_File', 'old.csv')
        _Set(data, 'Catch.Clearance', -1.0)
    configuration = Load_Configuration(_Write_Mutated(root, tmp_path, Mutate), stored_copy=True)
    assert len(configuration.Notes) == 1
    level, message = configuration.Notes[0]
    assert level == 'WARNING' and 'Flight.Dreg is unknown' in message


def test_a_copy_like_the_twenty_stored_datasets_loads_with_its_own_catch_area_and_no_later_section(root, tmp_path):
    """
    Description:
        The copy of the 20 datasets stored before F4 (measured 2026-11-21): no Robot, Catch, Workspace or Decision section,
        and Areas.Catch before F4 (centre (0, 0, 1.25) m, size (4, 2, 1.5) m). The full load refuses it with 17 problems,
        one per field of those sections (18 when measured, with Decision.Ledger_File, removed 2026-11-23; the defect of
        KNOWN_ISSUES before that change); the stored copy's load keeps its own catch area, not the shipped
        one, and gives None for every section it is not read for, so a reader of one fails instead of taking a default.
    """

    def Mutate(data):
        for section in ('Robot', 'Catch', 'Workspace', 'Decision'):
            del data[section]
        data['Areas']['Catch'] = {'Centre': [0.0, 0.0, 1.25], 'Size': [4.0, 2.0, 1.5]}
    path = _Write_Mutated(root, tmp_path, Mutate)
    with pytest.raises(Configuration_Error_Cls, match=r'17 problem\(s\)'):
        Load_Configuration(path)
    stored = Load_Configuration(path, stored_copy=True)
    assert (stored.Areas.Catch.Centre, stored.Areas.Catch.Size) == ([0.0, 0.0, 1.25], [4.0, 2.0, 1.5])
    assert (stored.Recording, stored.Identification, stored.Robot, stored.Catch, stored.Workspace, stored.Decision) == (None,) * 6
    with pytest.raises(AttributeError):
        stored.Robot.Joint_Speed


def test_the_shipped_dataset_configuration_loads(root):
    """
    Description:
        The shipped dataset file (SPEC §8.2) loads without notes: 10 000 flights from seed 20261111,
        0.70 / 0.15 / 0.15, the default rates 100, 200, 300 Hz (370 Hz
        optional), composed dropouts from all donors, the flight time [0.95, 1.40] s and the release speed
        [5.5, 8.0] m/s (F6 Q9), and the targets' own area: centred like the robot
        hemisphere at (0, 0, 0.3607) m, r 1.5 m, its orientation [0, 0, -90] deg in [rad].
    """

    configuration = Load_Dataset_Configuration(_Shipped_Dataset(root))
    assert configuration.Notes == []
    assert (configuration.Name, configuration.Seed, configuration.Flights) == ('Default', 20261111, 10000)
    assert configuration.System_Configuration == 'Config/ball_catching.yaml'
    assert (configuration.Partitions.Train, configuration.Partitions.Validation, configuration.Partitions.Test) == (0.70, 0.15, 0.15)
    assert configuration.Throw.Flight_Time == [0.95, 1.40] and configuration.Throw.Release_Speed == [5.5, 8.0]
    target = configuration.Throw.Target_Area
    assert (target.Centre, target.Radius) == ([0.0, 0.0, 0.3607], 1.5)
    assert target.Orientation == pytest.approx([0.0, 0.0, -math.pi / 2.0], abs=1e-15)
    assert configuration.Observation.Rates == [100.0, 200.0, 300.0] and configuration.Observation.Noise_Scale == 1.0
    assert (configuration.Observation.Pattern, configuration.Observation.Donors) == ('composed', 'all')
    assert configuration.Output_Folder == 'Data/Generated/Datasets'


@pytest.mark.parametrize('key_path', CONST_DATASET_REQUIRED)
def test_a_missing_dataset_field_fails_naming_it(root, tmp_path, key_path):
    """
    Description:
        Every field of the dataset file is REQUIRED: removing one fails the load with its full key path.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Delete(data, key_path), _Shipped_Dataset(root))
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Dataset_Configuration(path)
    assert '%s is missing' % key_path in str(error.value)


@pytest.mark.parametrize('key_path, value', [('Dataset.Name', 'two words'), ('Dataset.Seed', -1), ('Dataset.Seed', 1.5), ('Dataset.Seed', True),
                                             ('Dataset.Flights', 0), ('Throw.Flight_Time', [1.40, 0.95]), ('Throw.Release_Speed', [0.0, 8.0]),
                                             ('Throw.Release_Std', [0.8, 0.0, 0.1]), ('Throw.Apex_Std', 0.0), ('Observation.Rates', []),
                                             ('Observation.Rates', [200, 200]), ('Observation.Rates', [100, -1]),
                                             ('Observation.Dropout.Pattern', 'shuffled'), ('Observation.Dropout.Donors', 'C'),
                                             ('Output.Folder', '/tmp/datasets'), ('Throw.Target_Area.Radius', 0.0),
                                             ('Throw.Target_Area.Radius', -1.5), ('Throw.Target_Area.Centre', [0.0, 0.0]),
                                             ('Throw.Target_Area.Orientation', [0.0, 'west', -90.0])])
def test_an_invalid_dataset_value_fails_and_never_falls_back(root, tmp_path, key_path, value):
    """
    Description:
        A present but invalid value of the dataset file fails the load naming the field.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, value), _Shipped_Dataset(root)))
    assert key_path in str(error.value)


def test_the_partitions_sum_to_one_tested_on_both_sides(root, tmp_path):
    """
    Description:
        The partition ratios sum to 1 within 1e-9: a Train ratio 5e-10 too large is accepted, 2e-9 too
        large is refused with the rule named.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Partitions.Train', 0.70 + 5e-10), _Shipped_Dataset(root))
    assert Load_Dataset_Configuration(path).Partitions.Train == 0.70 + 5e-10
    with pytest.raises(Configuration_Error_Cls, match='must sum to 1 within 1e-09'):
        Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Partitions.Train', 0.70 + 2e-9),
                                                  _Shipped_Dataset(root)))


def test_an_unknown_dataset_key_is_a_warning_not_a_failure(root, tmp_path):
    """
    Description:
        An unknown key of the dataset file loads, with one WARNING note naming it.
    """

    configuration = Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Dataset.Seeds', 1),
                                                              _Shipped_Dataset(root)))
    assert len(configuration.Notes) == 1 and 'Dataset.Seeds is unknown' in configuration.Notes[0][1]


def test_the_shipped_decision_section_loads(root):
    """
    Description:
        The shipped Decision section (SPEC §8.1): the datasets Catch_In, Catch_Near
        and Default with their hashes, the first two calibrating, the quantile 0.95, the stages N = 20 then 40, the
        outputs under Data/Generated/Decisions. The section has no ledger field: a counted 'catch' is one row in the
        catch's ledger.
    """

    decision = Load_Configuration(_Shipped(root)).Decision
    assert [d.Path for d in decision.Datasets] == ['Data/Generated/Datasets/Catch_In', 'Data/Generated/Datasets/Catch_Near',
                                                   'Data/Generated/Datasets/Default']
    assert decision.Datasets[2].Hash == CONST_DEFAULT_HASH and all(len(d.Hash) == 64 for d in decision.Datasets)
    assert decision.Calibration == ['Data/Generated/Datasets/Catch_In', 'Data/Generated/Datasets/Catch_Near']
    assert (decision.Quantile, decision.Stages) == (0.95, [20, 40])
    assert decision.Output_Folder == 'Data/Generated/Decisions' and not hasattr(decision, 'Ledger_File')


@pytest.mark.parametrize('key_path, value', [('Decision.Quantile', 0.0), ('Decision.Quantile', 1.0), ('Decision.Quantile', 'high'),
                                             ('Decision.Stages', [40, 20]), ('Decision.Stages', [20, 20]), ('Decision.Stages', [1, 20]),
                                             ('Decision.Stages', []), ('Decision.Calibration', []),
                                             ('Decision.Calibration', ['Data/Generated/Datasets/Catch_In'] * 2), ('Decision.Datasets', [])])
def test_an_invalid_decision_value_fails_naming_it(root, tmp_path, key_path, value):
    """
    Description:
        The Decision section's values are checked (SPEC §8.1): the quantile in (0, 1), the stages increasing integers
        >= 2, the calibration a non-empty list of distinct paths, the datasets a non-empty list; each invalid value fails
        the load naming its field and its rule ('<field> must ...').
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, value)))
    assert '%s must' % key_path in str(error.value)


def test_a_calibration_dataset_the_decision_does_not_evaluate_fails(root, tmp_path):
    """
    Description:
        The margins are calibrated on datasets the decision evaluates (F9 Q7): Decision.Calibration naming a path absent
        from Decision.Datasets fails, naming it.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Decision.Calibration', ['Data/Generated/Datasets/Seed_1']))
    with pytest.raises(Configuration_Error_Cls, match='Decision.Calibration names .*Seed_1.* not among Decision.Datasets'):
        Load_Configuration(path)


def test_the_shipped_interception_section_loads(root):
    """
    Description:
        The shipped Interception section (SPEC §8.1): the ladder N = 20, 25, 30,
        35, 40, 50, 60, 70, 80, 100, 120, 150 calibrated on the train partitions of Catch_In and Catch_Near, the latency 4 ms,
        the compute budget 0.05 s, the guard 0.03 s, the path step 2.5 mm, the angle tolerance 10 deg in [rad], the outputs
        under Data/Generated/Interceptions; no ledger field. F15: the
        catch mode moving, the stroke's scales 1, 0.75, 0.5, 0.25 and its compute budget 0.10 s.
    """

    interception = Load_Configuration(_Shipped(root)).Interception
    assert interception.Stages == [20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150]
    assert interception.Calibration == ['Data/Generated/Datasets/Catch_In', 'Data/Generated/Datasets/Catch_Near']
    assert (interception.Latency, interception.Compute_Budget, interception.Guard, interception.Path_Step) == (0.004, 0.05, 0.03, 0.0025)
    assert interception.Angle_Tolerance == pytest.approx(math.radians(10.0), abs=1e-15)
    assert interception.Output_Folder == 'Data/Generated/Interceptions' and not hasattr(interception, 'Ledger_File')
    assert (interception.Catch_Mode, interception.Stroke_Scales, interception.Stroke_Compute_Budget) == ('moving', [1.0, 0.75, 0.5, 0.25], 0.1)


@pytest.mark.parametrize('key_path, value', [('Interception.Stages', [40, 20]), ('Interception.Stages', [1, 20]),
                                             ('Interception.Calibration', []), ('Interception.Latency', 0.0),
                                             ('Interception.Compute_Budget', -0.05), ('Interception.Guard', 0.0),
                                             ('Interception.Path_Step', 0.0), ('Interception.Angle_Tolerance', 'wide'),
                                             ('Interception.Catch_Mode', 'both'), ('Interception.Catch_Mode', True),
                                             ('Interception.Stroke_Scales', []), ('Interception.Stroke_Scales', [0.5, 1.0]),
                                             ('Interception.Stroke_Scales', [1.0, 0.0]), ('Interception.Stroke_Scales', [1.5, 1.0]),
                                             ('Interception.Stroke_Compute_Budget', 0.0)])
def test_an_invalid_interception_value_fails_naming_it(root, tmp_path, key_path, value):
    """
    Description:
        The Interception section's values are checked (SPEC §8.1): the ladder increasing integers >= 2, the calibration a
        non-empty list of paths, every time, the step and the angle > 0; the catch mode static or moving and nothing
        else; the stroke's scales a non-empty list of decreasing numbers in (0, 1] (an empty one, an increasing one, a
        scale of 0 and one above 1 fail); each invalid value fails naming its field.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, value)))
    assert '%s must' % key_path in str(error.value)


def test_the_ladder_holds_the_decisions_stages_and_calibrates_on_its_datasets(root, tmp_path):
    """
    Description:
        F10 Q3: Interception.Calibration naming a dataset absent from Decision.Datasets fails naming it, and a ladder
        without one of Decision.Stages (40, F9's baseline) fails naming the stage; both at once are both listed.
    """

    def Mutate(data):
        _Set(data, 'Interception.Calibration', ['Data/Generated/Datasets/Seed_1'])
        _Set(data, 'Interception.Stages', [20, 25, 30, 50])
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, Mutate))
    message = str(error.value)
    assert '2 problem(s)' in message and 'Interception.Calibration names' in message and 'Seed_1' in message
    assert 'does not hold Decision.Stages [40]' in message


def test_the_shipped_contact_section_loads(root):
    """
    Description:
        The shipped Contact section (SPEC §8.1): the step
        0.00025 s, the nominal contact (0.004 s, 0.7), the friction 0.5 / 0.005 / 0.0001, the settle window 2.5 s, at rest
        below 0.05 m/s for 0.2 s, the five sensitivity levels with the stiffer pair first (restitutions of 0.12, 0.29,
        0.35, 0.45 and 0.56; the third and the fifth since F15), the sweep's six scales on the
        first 1 500 validation throws of each dataset, the catches under Data/Generated/Catches and the ledger
        Data/Generated/Evaluations/catch_ledger.csv.
    """

    contact = Load_Configuration(_Shipped(root)).Contact
    assert (contact.Step, contact.Time_Constant, contact.Damping_Ratio) == (0.00025, 0.004, 0.7)
    assert contact.Friction == [0.5, 0.005, 0.0001]
    assert (contact.Settle_Time, contact.Rest_Speed, contact.Rest_Time) == (2.5, 0.05, 0.2)
    assert contact.Levels == [(0.00157, 0.9), (0.004, 0.5), (0.004, 0.4), (0.004, 0.3), (0.004, 0.2)]
    assert contact.Sweep_Scales == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] and contact.Sweep_Throws == 1500
    assert (contact.Output_Folder, contact.Ledger_File) == ('Data/Generated/Catches', 'Data/Generated/Evaluations/catch_ledger.csv')


@pytest.mark.parametrize('key_path, value', [('Contact.Step', 0.0), ('Contact.Time_Constant', -0.004), ('Contact.Damping_Ratio', 0.0),
                                             ('Contact.Friction', [0.5, 0.005]), ('Contact.Friction', [0.5, 0.0, 0.0001]),
                                             ('Contact.Settle_Time', 0.0), ('Contact.Rest_Speed', 'slow'), ('Contact.Rest_Time', 0.0),
                                             ('Contact.Levels', [[0.004]]), ('Contact.Levels', [[0.004, 0.0]]), ('Contact.Levels', 0.3),
                                             ('Contact.Sweep_Scales', []), ('Contact.Sweep_Scales', [1.0, 0.5]),
                                             ('Contact.Sweep_Scales', [-1.0, 1.0]), ('Contact.Sweep_Throws', 0),
                                             ('Contact.Sweep_Throws', 1.5), ('Contact.Output_Folder', '/tmp/Catches')])
def test_an_invalid_contact_value_fails_naming_it(root, tmp_path, key_path, value):
    """
    Description:
        The Contact section's values are checked (SPEC §8.1): the step, the contact's two numbers, the window, the rest
        speed and time > 0, three friction values > 0, the levels pairs of numbers > 0, the scales a non-empty increasing
        list >= 0, the sweep's throws an integer >= 1, the folder repository-relative; each invalid value fails naming
        its field.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, value)))
    assert '%s must' % key_path in str(error.value)


def test_a_time_constant_below_two_steps_is_refused_for_the_nominal_contact_and_for_a_level(root, tmp_path):
    """
    Description:
        MuJoCo raises a contact's time constant below two steps to two steps, silently, so the loader refuses one: the
        nominal 0.0004 s and the second level's 0.0003 s at the step 0.00025 s are both named with the step, and a value
        of exactly two steps (0.0005 s) loads.
    """

    def Mutate(data):
        _Set(data, 'Contact.Time_Constant', 0.0004)
        _Set(data, 'Contact.Levels', [[0.004, 0.5], [0.0003, 0.9]])
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Configuration(_Write_Mutated(root, tmp_path, Mutate))
    message = str(error.value)
    assert '2 problem(s)' in message and 'Contact.Time_Constant 0.0004 s is below two steps (Contact.Step 0.00025 s)' in message
    assert 'Contact.Levels[1] 0.0003 s is below two steps' in message
    contact = Load_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Contact.Time_Constant', 0.0005))).Contact
    assert contact.Time_Constant == 0.0005 and contact.Levels[0] == (0.00157, 0.9)


def _Catch_Target(data, margin):
    del data['Throw']['Target_Area']
    data['Throw']['Target_Catch_Area'] = {'Margin': margin}


def test_a_dataset_file_with_the_catch_target_loads_without_the_quarter_sphere(root, tmp_path):
    """
    Description:
        A catch dataset aims its throws into Areas.Catch grown by a margin. A file that gives
        Throw.Target_Catch_Area (Margin 0.25 m) instead of Throw.Target_Area loads without notes as the shape
        'catch_area' with that margin and no quarter sphere; Margin 0 (the area itself) is accepted. The shipped file
        is the quarter sphere, margin 0.
    """

    configuration = Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Catch_Target(data, 0.25), _Shipped_Dataset(root)))
    assert configuration.Notes == []
    assert (configuration.Throw.Target_Shape, configuration.Throw.Target_Margin) == ('catch_area', 0.25)
    assert (configuration.Throw.Target_Area.Centre, configuration.Throw.Target_Area.Radius) == ([], 0.0)
    zero = Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Catch_Target(data, 0), _Shipped_Dataset(root)))
    assert (zero.Throw.Target_Shape, zero.Throw.Target_Margin) == ('catch_area', 0.0)
    shipped = Load_Dataset_Configuration(_Shipped_Dataset(root))
    assert (shipped.Throw.Target_Shape, shipped.Throw.Target_Margin) == ('quarter_sphere', 0.0)


def test_both_target_areas_or_a_bad_catch_margin_fail_naming_them(root, tmp_path):
    """
    Description:
        A dataset has one target area (SPEC §8.2): a file giving both Throw.Target_Area and Throw.Target_Catch_Area fails
        naming both; a negative margin, one that is not a number and a missing one fail naming
        Throw.Target_Catch_Area.Margin.
    """

    def Both(data):
        data['Throw']['Target_Catch_Area'] = {'Margin': 0.25}

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, Both, _Shipped_Dataset(root)))
    assert 'Throw.Target_Area is given together with Throw.Target_Catch_Area' in str(error.value)
    for margin in (-0.05, 'wide'):
        with pytest.raises(Configuration_Error_Cls, match='Throw.Target_Catch_Area.Margin must be a number >= 0'):
            Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Catch_Target(data, margin), _Shipped_Dataset(root)))

    def Missing(data):
        del data['Throw']['Target_Area']
        data['Throw']['Target_Catch_Area'] = {}

    with pytest.raises(Configuration_Error_Cls, match='Throw.Target_Catch_Area.Margin is missing'):
        Load_Dataset_Configuration(_Write_Mutated(root, tmp_path, Missing, _Shipped_Dataset(root)))


def _Shipped_Training(root):
    return os.path.join(root, 'Config', 'training.yaml')


def test_the_shipped_training_configuration_loads(root):
    """
    Description:
        The shipped training file (SPEC §8.3; F7, F8, F14) loads without notes: the final predictor, the query-time
        model with the basis decoder, QT_BASIS, scored on the Default dataset
        and the four further noise scales of its throws (0, 0.25, 0.5, 1.5) and trained on the mix of the five,
        pooled in the order 0, 0.25, 0.5, 1.0, 1.5, each with the hash of its generation (the bounded error model,
        the targets' own area); all 7 000 training flights of each, 5 seeds on the CPU,
        N = 10, 20, 40, an encoder state of 128, the position loss, 16 epochs (the optimiser steps of 80 epochs
        of one dataset) and a patience of 16, equal to the epochs, so that no training stops early; the runs
        under Data/Generated/Models.
    """

    configuration = Load_Training_Configuration(_Shipped_Training(root))
    assert configuration.Notes == []
    assert (configuration.Name, configuration.Model_Name, configuration.Observed_Samples) == ('QT_BASIS', 'QT_BASIS', [10, 20, 40])
    hashes = {'Default': CONST_DEFAULT_HASH, 'Noise_0': '19a03de242871cf0a782da540849e9ffa1be08b20fa0e7cd952990f4f273d776',
              'Noise_0_25': 'e293dae46955adc8c56fa065dedb3d6a6059efa66d3b12bb0be9f32ba9437dc0',
              'Noise_0_5': '847b46c242b5bbd801963d010b1148f96984e990a83ce19193ae1ae1cf681c06',
              'Noise_1_5': 'f6c48e2acd7b4a9e88670b48a3f56f618bedc37b81b0cac4b9e6978f6b278d68'}
    listed = lambda names: [('Data/Generated/Datasets/%s' % name, hashes[name]) for name in names]
    assert [(d.Path, d.Hash) for d in configuration.Datasets] == listed(['Default', 'Noise_0', 'Noise_0_25', 'Noise_0_5', 'Noise_1_5'])
    assert [(d.Path, d.Hash) for d in configuration.Training_Datasets] == listed(['Noise_0', 'Noise_0_25', 'Noise_0_5', 'Default',
                                                                                 'Noise_1_5'])
    assert (configuration.Seeds, configuration.Device, configuration.Training_Flights, configuration.Hidden) == ([0, 1, 2, 3, 4], 'cpu', 7000, 128)
    assert (configuration.Velocity_Weight, configuration.Learning_Rate, configuration.Batch_Size, configuration.Max_Epochs) == (0.0, 0.002, 128, 16)
    assert (configuration.Weight_Decay, configuration.Patience, configuration.Min_Delta) == (0.0, 16, 0.0001)
    assert configuration.Patience == configuration.Max_Epochs
    assert configuration.Output_Folder == 'Data/Generated/Models'


@pytest.mark.parametrize('key_path', CONST_TRAINING_REQUIRED)
def test_a_missing_training_field_fails_naming_it(root, tmp_path, key_path):
    """
    Description:
        Every field of the training file is REQUIRED: removing one fails the load with its full key path.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: _Delete(data, key_path), _Shipped_Training(root))
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Training_Configuration(path)
    assert '%s is missing' % key_path in str(error.value)


@pytest.mark.parametrize('key_path, value, reason', [
    ('Run.Name', 'a/b', 'letters, digits'), ('Run.Datasets', [], 'non-empty list'),
    ('Run.Datasets', [{'Path': 'Data/A', 'Hash': CONST_DEFAULT_HASH, 'Seed': 1}], 'exactly the keys Path and Hash'),
    ('Run.Datasets', [{'Path': '/abs/A', 'Hash': CONST_DEFAULT_HASH}], 'repository-relative'),
    ('Run.Datasets', [{'Path': 'Data/A', 'Hash': CONST_DEFAULT_HASH[:63]}], '64 lower-case hex'),
    ('Run.Datasets', [{'Path': 'Data/A', 'Hash': CONST_DEFAULT_HASH.upper()}], '64 lower-case hex'),
    ('Run.Datasets', [{'Path': 'Data/A', 'Hash': CONST_DEFAULT_HASH}, {'Path': 'Data/A', 'Hash': CONST_OTHER_HASH}], 'each dataset once'),
    ('Input.Observed_Samples', [1, 20], 'integers >= 2'), ('Input.Observed_Samples', [20, 20], 'distinct'),
    ('Input.Observed_Samples', [10.0], 'integers >= 2'), ('Input.Observed_Samples', [], 'non-empty'),
    ('Model.Name', 'Kalman', 'one of GLS_M4, QT_BASIS, found'), ('Model.Name', 'LS', 'one of GLS_M4, QT_BASIS, found'),
    ('Model.Name', 'QT_MLP', 'one of GLS_M4, QT_BASIS, found'), ('Model.Name', 'HYBRID', 'one of GLS_M4, QT_BASIS, found'),
    ('Output.Folder', '', 'non-empty')])
def test_an_invalid_training_value_fails_and_never_falls_back(root, tmp_path, key_path, value, reason):
    """
    Description:
        A present but invalid value of the training file fails the load naming the field and the rule; N = 2
        (the bound, 6 unknowns from 2 samples) and one dataset load. The three removed predictors (LS, QT_MLP,
        HYBRID) are refused like an unknown one, the message naming the two that remain.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Training_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, key_path, value), _Shipped_Training(root)))
    assert key_path in str(error.value) and reason in str(error.value)
    path = _Write_Mutated(root, tmp_path, lambda data: (_Set(data, 'Input.Observed_Samples', [2]),
                                                        _Set(data, 'Run.Datasets', [{'Path': 'Data/A', 'Hash': CONST_OTHER_HASH}])),
                          _Shipped_Training(root))
    loaded = Load_Training_Configuration(path)
    assert loaded.Observed_Samples == [2] and loaded.Datasets[0].Hash == CONST_OTHER_HASH


def test_an_unknown_training_key_is_a_warning_not_a_failure(root, tmp_path):
    """
    Description:
        An unknown key of the training file (a misspelt Run.Seed) loads, with one WARNING note naming it.
    """

    configuration = Load_Training_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Set(data, 'Run.Seed', [0, 1, 2, 3, 4]),
                                                               _Shipped_Training(root)))
    assert len(configuration.Notes) == 1 and 'Run.Seed is unknown' in configuration.Notes[0][1]


def _Learned(data, model='QT_BASIS'):
    """
    Description:
        The shipped training document turned into a learned predictor's (SPEC §8.3; F8): the model and every
        learning field, written out here.
    """

    _Set(data, 'Model.Name', model)
    _Set(data, 'Model.Hidden', 128)
    _Set(data, 'Run.Seeds', [0, 1, 2, 3, 4])
    _Set(data, 'Run.Device', 'cpu')
    data['Training'] = {'Datasets': [{'Path': 'Data/Generated/Datasets/Default', 'Hash': CONST_DEFAULT_HASH}], 'Flights': 7000}
    data['Loss'] = {'Velocity_Weight': 0.0}
    data['Optimiser'] = {'Learning_Rate': 0.002, 'Batch_Size': 128, 'Max_Epochs': 80, 'Weight_Decay': 0.0}
    data['Early_Stopping'] = {'Patience': 15, 'Min_Delta': 0.0001}


def test_a_learned_predictors_training_configuration_loads(root, tmp_path):
    """
    Description:
        A training file naming a learned predictor (SPEC §8.3; F8) loads without notes, every
        learning field read: the 5 seeds, the device, the datasets trained on with the number of training
        flights, the encoder's size, the velocity weight [s], the optimiser and the early stopping [epochs], [m].
    """

    c = Load_Training_Configuration(_Write_Mutated(root, tmp_path, lambda data: _Learned(data, 'QT_BASIS'), _Shipped_Training(root)))
    assert c.Notes == [] and c.Model_Name == 'QT_BASIS'
    assert (c.Seeds, c.Device, c.Training_Flights, c.Hidden, c.Velocity_Weight) == ([0, 1, 2, 3, 4], 'cpu', 7000, 128, 0.0)
    assert [(d.Path, d.Hash) for d in c.Training_Datasets] == [('Data/Generated/Datasets/Default', CONST_DEFAULT_HASH)]
    assert (c.Learning_Rate, c.Batch_Size, c.Max_Epochs, c.Weight_Decay, c.Patience, c.Min_Delta) == (0.002, 128, 80, 0.0, 15, 0.0001)


@pytest.mark.parametrize('key_path', ['Run.Seeds', 'Run.Device', 'Training.Datasets', 'Training.Flights', 'Model.Hidden',
                                      'Loss.Velocity_Weight', 'Optimiser.Learning_Rate', 'Optimiser.Batch_Size', 'Optimiser.Max_Epochs',
                                      'Optimiser.Weight_Decay', 'Early_Stopping.Patience', 'Early_Stopping.Min_Delta'])
def test_a_missing_learning_field_of_a_learned_predictor_fails_naming_it(root, tmp_path, key_path):
    """
    Description:
        Every learning field is REQUIRED for a learned predictor: removing one fails the load with its full key
        path.
    """

    path = _Write_Mutated(root, tmp_path, lambda data: (_Learned(data), _Delete(data, key_path)), _Shipped_Training(root))
    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Training_Configuration(path)
    assert '%s is missing' % key_path in str(error.value)


@pytest.mark.parametrize('key_path, value, reason', [
    ('Run.Seeds', [0, 1, 2, 3], 'at least 5 distinct'), ('Run.Seeds', [0, 1, 2, 3, 3], 'at least 5 distinct'),
    ('Run.Seeds', [0, 1, 2, 3, -1], 'integers >= 0'), ('Run.Device', 'cuda', 'one of cpu, mps'),
    ('Training.Datasets', [], 'non-empty list'), ('Training.Flights', 0, 'integer >= 1'), ('Model.Hidden', 0, 'integer >= 1'),
    ('Loss.Velocity_Weight', -0.1, '>= 0'), ('Optimiser.Learning_Rate', 0.0, '> 0'), ('Optimiser.Batch_Size', 0, 'integer >= 1'),
    ('Optimiser.Max_Epochs', 1.5, 'integer >= 1'), ('Optimiser.Weight_Decay', -1e-5, '>= 0'), ('Early_Stopping.Patience', 0, 'integer >= 1'),
    ('Early_Stopping.Min_Delta', -0.001, '>= 0')])
def test_an_invalid_learning_value_fails_and_never_falls_back(root, tmp_path, key_path, value, reason):
    """
    Description:
        A present but invalid learning field fails the load naming the field and the rule; exactly 5 seeds (the
        bound of SPEC §2.13.7) and the device mps load.
    """

    with pytest.raises(Configuration_Error_Cls) as error:
        Load_Training_Configuration(_Write_Mutated(root, tmp_path, lambda data: (_Learned(data), _Set(data, key_path, value)),
                                                   _Shipped_Training(root)))
    assert key_path in str(error.value) and reason in str(error.value)
    loaded = Load_Training_Configuration(_Write_Mutated(root, tmp_path, lambda data: (_Learned(data), _Set(data, 'Run.Seeds', [7, 3, 5, 1, 9]),
                                                                                    _Set(data, 'Run.Device', 'mps')), _Shipped_Training(root)))
    assert loaded.Seeds == [7, 3, 5, 1, 9] and loaded.Device == 'mps'


def test_the_learning_fields_of_a_numerical_predictor_are_named_in_one_warning(root, tmp_path):
    """
    Description:
        A learning field cannot take effect for a numerical predictor (a field without effect is noted): the
        shipped numerical file with every learning field loads, with one WARNING note naming them, and none of
        them is read.
    """

    configuration = Load_Training_Configuration(_Write_Mutated(root, tmp_path, lambda data: (_Learned(data), _Set(data, 'Model.Name', 'GLS_M4')),
                                                               _Shipped_Training(root)))
    assert len(configuration.Notes) == 1 and 'have no effect for the numerical predictor GLS_M4' in configuration.Notes[0][1]
    assert all(key in configuration.Notes[0][1] for key in ('Run.Seeds', 'Training.Datasets', 'Early_Stopping.Min_Delta'))
    assert (configuration.Seeds, configuration.Device, configuration.Training_Datasets, configuration.Max_Epochs) == ([], '', [], 0)


def _Flat(data, prefix=''):
    """
    Description:
        A parsed configuration as {dotted key: value}.
    """

    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(_Flat(value, prefix + key + '.'))
        else:
            out[prefix + key] = value
    return out


def test_the_shipped_dataset_configurations_are_those_of_the_datasets_the_shipped_configurations_name(root):
    """
    Description:
        A clean clone can generate every dataset the shipped configurations name: beside
        Config/dataset.yaml (Default) the repository ships Config/Datasets/<Name>.yaml for the six further ones, each
        named after its Dataset.Name and loading without notes. Together they are exactly the datasets of
        Config/training.yaml (Run.Datasets, Training.Datasets) and of the system configuration (Decision.Datasets,
        Workspace.Dataset). Each differs from Config/dataset.yaml in the fields that define it and in nothing else: a
        noise dataset in its name and Observation.Noise.Scale (the same throws and dropouts, the error scaled); a catch
        dataset in its name, seed, partitions (0.10 / 0.45 / 0.45), its throws aimed into Areas.Catch grown by its margin
        (0 and 0.25 m), and Catch_Near in its 20 000 flights.
    """

    folder = os.path.join(root, 'Config', 'Datasets')
    assert sorted(os.listdir(folder)) == ['Catch_In.yaml', 'Catch_Near.yaml', 'Noise_0.yaml', 'Noise_0_25.yaml', 'Noise_0_5.yaml',
                                          'Noise_1_5.yaml']
    with open(_Shipped_Dataset(root), encoding='utf-8') as file:
        default = _Flat(yaml.safe_load(file))
    target = {'Throw.Target_Area.Centre', 'Throw.Target_Area.Orientation', 'Throw.Target_Area.Radius', 'Throw.Target_Catch_Area.Margin'}
    partitions = {'Partitions.Train': 0.10, 'Partitions.Validation': 0.45, 'Partitions.Test': 0.45}
    expected = {'Noise_0': {'Observation.Noise.Scale': 0.0}, 'Noise_0_25': {'Observation.Noise.Scale': 0.25},
                'Noise_0_5': {'Observation.Noise.Scale': 0.5}, 'Noise_1_5': {'Observation.Noise.Scale': 1.5},
                'Catch_In': dict(partitions, **{'Dataset.Seed': 20261121, 'Throw.Target_Catch_Area.Margin': 0.0}),
                'Catch_Near': dict(partitions, **{'Dataset.Seed': 20261122, 'Dataset.Flights': 20000, 'Throw.Target_Catch_Area.Margin': 0.25})}
    for name, values in expected.items():
        path = os.path.join(folder, name + '.yaml')
        configuration = Load_Dataset_Configuration(path)
        assert configuration.Notes == [] and configuration.Name == name
        with open(path, encoding='utf-8') as file:
            own = _Flat(yaml.safe_load(file))
        differing = {key for key in set(default) | set(own) if default.get(key) != own.get(key)}
        assert differing - target == set(values) - target | {'Dataset.Name'}, name
        assert all(own[key] == value for key, value in values.items()), name
        assert (differing & target == target) == name.startswith('Catch_'), name
    training, system = Load_Training_Configuration(_Shipped_Training(root)), Load_Configuration(_Shipped(root))
    named = {d.Path for d in training.Datasets + training.Training_Datasets + system.Decision.Datasets} | {system.Workspace.Dataset}
    assert named == {'Data/Generated/Datasets/%s' % name for name in list(expected) + ['Default']}


def test_both_catch_modes_load_and_the_scales_keep_their_order(root, tmp_path):
    """
    Description:
        F15: both catch modes are legal: the shipped file with Catch_Mode static and the
        scales 0.8, 0.4 loads with them, in the order given; with Catch_Mode moving and a single scale of 1 too.
    """

    def Change(data):
        _Set(data, 'Interception.Catch_Mode', 'static')
        _Set(data, 'Interception.Stroke_Scales', [0.8, 0.4])

    interception = Load_Configuration(_Write_Mutated(root, tmp_path, Change)).Interception
    assert (interception.Catch_Mode, interception.Stroke_Scales) == ('static', [0.8, 0.4])

    def Single(data):
        _Set(data, 'Interception.Catch_Mode', 'moving')
        _Set(data, 'Interception.Stroke_Scales', [1.0])

    single = Load_Configuration(_Write_Mutated(root, tmp_path, Single)).Interception
    assert single.Stroke_Scales == [1.0] and single.Catch_Mode == 'moving'
