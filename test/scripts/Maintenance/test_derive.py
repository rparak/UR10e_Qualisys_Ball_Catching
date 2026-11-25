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
# File Name: ../test/scripts/Maintenance/test_derive.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Regular expressions (Default; a configuration line replaced in a copy)
import re
# Subprocess (Subprocess management)
import subprocess
# CSV (CSV file reading and writing)
import csv
# JSON (JSON encoder and decoder)
import json
# Hashlib (Secure hashes)
import hashlib
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the scripts' shared test helpers
#   ../Script_Helpers
from Script_Helpers import (CONST_SCRIPT, CONST_DERIVE_SCRIPT, _Run, _Run_Derive, _Run_Guarded, _Dataset_Config, _Stored_Folders, _Load_Derive,
                            _Forbid_Leaks, _Trained_Run, _Stored_Rows, _Decision_Config, _Sweep_Args, _Stored_Errors)

"""
Description:
    Initialization of constants.

    scripts/Maintenance/derive.py, the project's derivations, is run the way a person runs it: a
    fresh interpreter, from an unrelated working directory. Its tests
    were those of the same sub-commands of scripts/catch.py until 2026-11-23, moved with them, their assertions
    unchanged.
"""


# The identification measured by 'identify' at commit e72d93d, 2026-11-10 (the ledger's evaluation 1;
# F3 Part B): per row (kind, variant, segmented flights or session, identification flights, k [1/m],
# the median residual RMS [m]); the validation rows (kind, variant, session, flights, k, RMS). A
# regression pin: the identification is deterministic on one platform, so the bands are narrow.
#   Unit [Tuple]
CONST_PINNED_ROWS = (
    ('identified', 'identified', 90, 45, 0.00930496257, 0.00542350358),
    ('sensitivity', 'tolerance 10 mm', 89, 45, 0.00928885766, 0.00536889811),
    ('sensitivity', 'tolerance 50 mm', 90, 45, 0.00942616117, 0.00550503219),
    ('sensitivity', 'trim 0 ms', 90, 45, 0.00905245742, 0.00588652011),
    ('sensitivity', 'trim 80 ms', 90, 45, 0.00938857304, 0.00542157709),
    ('sensitivity', 'without adjacent', 90, 44, 0.00930921974, 0.00543136701),
    ('validation', 'trim 40 ms', 'all', 45, 0.00930496257, 0.00662326361),
    ('validation', 'trim 40 ms', 'Frequency_100_Hz_Active_Filtering_OFF, 1000 samples', 4, 0.00930496257, 0.00494413657),
    ('validation', 'trim 40 ms', 'Frequency_100_Hz_Active_Filtering_OFF, 2500 samples', 1, 0.00930496257, 0.0100309225),
    ('validation', 'trim 40 ms', 'Frequency_100_Hz_Active_Filtering_ON, 750 samples', 5, 0.00930496257, 0.00514248174),
    ('validation', 'trim 40 ms', 'Frequency_200_Hz_Active_Filtering_OFF, 2000 samples', 4, 0.00930496257, 0.00732204925),
    ('validation', 'trim 40 ms', 'Frequency_200_Hz_Active_Filtering_OFF, 2500 samples', 20, 0.00930496257, 0.00596747285),
    ('validation', 'trim 40 ms', 'Frequency_200_Hz_Active_Filtering_OFF, 5000 samples', 1, 0.00930496257, 0.00513069153),
    ('validation', 'trim 40 ms', 'Frequency_200_Hz_Active_Filtering_ON, 1500 samples', 5, 0.00930496257, 0.00740353939),
    ('validation', 'trim 40 ms', 'Frequency_300_Hz_Active_Filtering_OFF, 2500 samples', 5, 0.00930496257, 0.00948954698),
    ('validation', 'trim 0 ms', 'all', 45, 0.00930496257, 0.00681607268),
    ('validation', 'trim 0 ms', 'Frequency_100_Hz_Active_Filtering_OFF, 1000 samples', 4, 0.00930496257, 0.00545446231),
    ('validation', 'trim 0 ms', 'Frequency_100_Hz_Active_Filtering_OFF, 2500 samples', 1, 0.00930496257, 0.0101594056),
    ('validation', 'trim 0 ms', 'Frequency_100_Hz_Active_Filtering_ON, 750 samples', 5, 0.00930496257, 0.00627273407),
    ('validation', 'trim 0 ms', 'Frequency_200_Hz_Active_Filtering_OFF, 2000 samples', 4, 0.00930496257, 0.00777813257),
    ('validation', 'trim 0 ms', 'Frequency_200_Hz_Active_Filtering_OFF, 2500 samples', 20, 0.00930496257, 0.00608586609),
    ('validation', 'trim 0 ms', 'Frequency_200_Hz_Active_Filtering_OFF, 5000 samples', 1, 0.00930496257, 0.00573345277),
    ('validation', 'trim 0 ms', 'Frequency_200_Hz_Active_Filtering_ON, 1500 samples', 5, 0.00930496257, 0.00772254158),
    ('validation', 'trim 0 ms', 'Frequency_300_Hz_Active_Filtering_OFF, 2500 samples', 5, 0.00930496257, 0.00943947333))


# The statistics 'observation' finds outside their bootstrap bounds with the configured models, the error
# statistics as the median over 5 synthetic sets (each named in KNOWN_ISSUES, no bound
# widened): (section, statistic, rate). Re-captured 2026-11-16 with the error model's bounds and
# the targets' own area: against the F6 closure's set, rms_p90 (10.700 against its bound's
# 10.794 mm; 11.005 before) and offset_run_1 (8.022 against 8.735 mm; 9.033 before) are inside, both within
# their range over the sets, and of the throw shape floor_time_p50 is outside where floor_time_p10 was.
#   Unit [Set<Tuple<string, string, string>>]
CONST_PINNED_OUTSIDE = {('error', 'rms_median', ''), ('error', 'offset_share', ''),
                        ('error', 'shuffled_offset_share', ''), ('error', 'within_5ms', ''), ('error', 'within_10ms', ''),
                        ('error', 'within_20ms', ''), ('error', 'within_40ms', ''), ('error', 'jump_1', ''), ('error', 'interior_rms', ''),
                        ('error', 'within_5ms_mean', ''), ('error_diagnostic', 'offset_run_2_4', ''),
                        ('error_diagnostic', 'offset_run_5_14', ''), ('error_diagnostic', 'offset_run_15', ''),
                        ('pattern', 'long_share', '300/300'), ('pattern', 'long_share', '370/300'), ('shape', 'floor_time_p50', '')}


# The section 'bound' of the observation summary: the largest values of the identification
# residuals [m] (Part A, 2026-11-15: 55.9, 55.2, 89.4 and 29.5 mm) and the bounds derived from the first and the
# last (x 1.25, rounded up to 0.005 m), which are the configured Observation.Offset_Bound and White_Bound.
#   Unit [Dict<string, string>]
CONST_PINNED_BOUND = {'residual_max': '0.055879131', 'run_mean_max': '0.0552318287', 'jump_max': '0.0894396341', 'step_max': '0.0294685159',
                      'Offset_Bound': '0.07', 'White_Bound': '0.04'}


# The identification flights outside the support of the generated throws (SPEC §2.13.6): three with the targets
# of Throw.Target_Area (six with F6's targets).
#   Unit [Set<string>]
CONST_PINNED_UNCOVERED = {'Frequency_100_Hz_Active_Filtering_OFF/Experiment_5.txt', 'Frequency_100_Hz_Active_Filtering_ON/Experiment_6.txt',
                          'Frequency_200_Hz_Active_Filtering_OFF/Experiment_27.txt'}


# The identified row's uncertainty: the bootstrap sd, the half-range and Drag_Std.
#   Unit [Tuple<float>, 1/m]
CONST_PINNED_UNCERTAINTY = (0.000157489219, 0.000186851872, 0.000244369548)


# The stored datasets 'dataset-check' regenerates in the slow tier: a prediction dataset and
# the two catch datasets; the full check of all 22 is run at a change's acceptance.
#   Unit [Tuple<string>]
CONST_CHECKED_DATASETS = ('Default', 'Catch_In', 'Catch_Near')


def test_recordings_writes_both_files_and_prints_the_summary(root, tmp_path):
    """
    Description:
        'recordings --out DIR' writes the configured file names into DIR (91 summary lines, 7 326 step
        lines with the headers) and prints the summary on stdout, the [INFO] lines on stderr.
    """

    out = tmp_path / 'out'
    result = _Run_Derive(root, tmp_path, 'recordings', '--out', str(out))
    assert result.returncode == 0, result.stderr
    assert sorted(os.listdir(str(out))) == ['recording_steps.csv', 'recording_summary.csv']
    assert len((out / 'recording_summary.csv').read_text().splitlines()) == 91
    assert len((out / 'recording_steps.csv').read_text().splitlines()) == 7326
    assert 'flights segmented: 90 of 90' in result.stdout.splitlines() and '[INFO]' not in result.stdout
    assert 'nothing is run on, or validated on, the real UR10e' in result.stderr


def test_an_invalid_configuration_is_refused_and_nothing_is_written(root, tmp_path):
    """
    Description:
        A configuration without Recording.Steps_File is refused with exit 1 and an [ERROR] naming the
        field; no file is written (SPEC §9.1).
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), 'r', encoding='utf-8').read()
    config = tmp_path / 'broken.yaml'
    config.write_text('\n'.join(line for line in text.splitlines() if 'Steps_File:' not in line) + '\n', encoding='utf-8')
    out = tmp_path / 'out'
    result = _Run_Derive(root, tmp_path, 'recordings', '--config', str(config), '--out', str(out))
    assert result.returncode == 1
    assert '[ERROR]' in result.stderr and 'Recording.Steps_File is missing' in result.stderr
    assert not out.exists()


def test_identify_refuses_an_invalid_configuration_and_writes_nothing(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        'identify' with a configuration without Flight.Drag is refused with exit 1 and an [ERROR] naming
        the field, before any identification; no file is written, the ledger included.
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), 'r', encoding='utf-8').read()
    config = tmp_path / 'broken.yaml'
    config.write_text('\n'.join(line for line in text.splitlines() if not line.startswith('  Drag:')) + '\n', encoding='utf-8')
    out = tmp_path / 'out'
    result = _Run_Guarded(root, tmp_path, monkeypatch, capsys, 'identify', '--config', str(config), '--out', str(out), derive=True)
    assert result.returncode == 1
    assert '[ERROR]' in result.stderr and 'Flight.Drag is missing' in result.stderr
    assert not out.exists()


@pytest.mark.slow
def test_identify_reproduces_the_pinned_identification_and_the_configured_drag(root, tmp_path):
    """
    Description:
        'identify --out DIR' on the 90 recordings (about 70 s) writes the three files and reproduces the
        pinned rows: the identified k and its uncertainty, the sensitivity rows (tau 10 / 50 mm with 89 /
        90 segmented flights, trims 0 / 80 ms, without 100 Hz ON Experiment 10) and the validation rows
        per trim and session; its ledger counts this one run. The configured Flight.Drag and Drag_Std
        are the identified ones (no WARNING), so the configuration cannot drift from the identification.
    """

    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, os.path.join(root, CONST_DERIVE_SCRIPT), 'identify', '--out', str(out)], cwd=str(tmp_path),
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert sorted(os.listdir(str(out))) == ['identification_flights.csv', 'identification_ledger.csv', 'identification_summary.csv']
    assert '[WARNING]' not in result.stderr and 'nothing is run on, or validated on, the real UR10e' in result.stderr
    with open(str(out / 'identification_summary.csv'), 'r', encoding='ascii', newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(CONST_PINNED_ROWS)
    for row, pinned in zip(rows, CONST_PINNED_ROWS):
        if pinned[0] == 'validation':
            assert (row['kind'], row['variant'], row['session'], int(row['flights'])) == pinned[:4]
            assert row['evaluation'] == '1'
        else:
            assert (row['kind'], row['variant'], int(row['flights_segmented']), int(row['flights'])) == pinned[:4]
        assert float(row['drag_per_m']) == pytest.approx(pinned[4], abs=1e-10)
        assert float(row['rms_median_m']) == pytest.approx(pinned[5], abs=1e-9)
    identified = [float(rows[0][key]) for key in ('drag_bootstrap_sd_per_m', 'drag_half_range_per_m', 'drag_std_per_m')]
    assert identified == pytest.approx(list(CONST_PINNED_UNCERTAINTY), abs=1e-10)
    with open(str(out / 'identification_flights.csv'), 'r', encoding='ascii', newline='') as file:
        kinds = [row['kind'] for row in csv.DictReader(file)]
    assert kinds.count('identification') == 45 and kinds.count('validation') == 90
    with open(os.path.join(root, 'Config', 'ball_catching.yaml'), 'r', encoding='utf-8') as file:
        text = file.read()
    assert '  Drag: %.9g\n' % CONST_PINNED_ROWS[0][4] in text and '  Drag_Std: %.9g\n' % CONST_PINNED_UNCERTAINTY[2] in text


def test_observation_refuses_an_invalid_dataset_configuration_and_writes_nothing(root, tmp_path):
    """
    Description:
        'observation' with a dataset configuration that misses a field exits 1 naming it and writes nothing.
    """

    text = open(os.path.join(root, 'Config', 'dataset.yaml'), encoding='utf-8').read().replace('  Apex_Std: ', '  Apex_Sd: ')
    bad = tmp_path / 'dataset.yaml'
    bad.write_text(text, encoding='utf-8')
    out = tmp_path / 'out'
    result = _Run_Derive(root, tmp_path, 'observation', '--dataset-config', str(bad), '--out', str(out))
    assert result.returncode == 1 and 'Throw.Apex_Std is missing' in result.stderr
    assert not out.exists()


@pytest.mark.slow
def test_observation_reproduces_the_pinned_calibration_and_the_configured_values(root, tmp_path):
    """
    Description:
        'observation' (F6; SPEC §9.9) on the identification flights: no WARNING, so Observation.* and
        Throw.Release_* / Apex_* equal their estimates to the 9 digits written and the two bounds the derived
        ones (the pin of the configured values, as identify's for Flight.Drag); the largest residuals and the
        bounds as pinned; 44 releases with the late start left out; the statistics
        outside their bootstrap bounds are exactly the pinned ones; 42 of 45 identification flights inside
        the support, the three outside named; the validation flights not read.
    """

    out = tmp_path / 'out'
    result = _Run_Derive(root, tmp_path, 'observation', '--out', str(out))
    assert result.returncode == 0, result.stderr
    assert '[WARNING]' not in result.stderr and 'The validation flights were not read' in result.stderr
    with open(out / 'observation_summary.csv', encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    outside = {(r['section'], r['statistic'], r['rate']) for r in rows if r['inside'] == 'false'}
    assert outside == CONST_PINNED_OUTSIDE
    assert {r['statistic']: r['real'] for r in rows if r['section'] == 'release'}['releases'] == '44'
    uncovered = {r['statistic'] for r in rows if r['section'] == 'coverage' and r['inside'] != 'true'}
    assert uncovered == CONST_PINNED_UNCOVERED and sum(r['section'] == 'coverage' for r in rows) == 45
    assert {r['statistic']: r['real'] for r in rows if r['section'] == 'bound'} == CONST_PINNED_BOUND


@pytest.mark.slow
def test_observation_warns_of_a_bound_that_differs_from_the_derived_one(root, tmp_path):
    """
    Description:
        The configured bounds are pinned by 'observation' printing no WARNING (the test above); the case that
        check exists for: with Observation.Offset_Bound 0.060 m in a copy of the system configuration, the run
        still succeeds and warns, naming the field, the configured value and the derived 0.07 m, and it does not
        warn about White_Bound.
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    text, count = re.subn(r'(?m)^  Offset_Bound: .*$', '  Offset_Bound: 0.060', text)
    assert count == 1
    (tmp_path / 'system.yaml').write_text(text, encoding='utf-8')
    result = _Run_Derive(root, tmp_path, 'observation', '--config', str(tmp_path / 'system.yaml'), '--out', str(tmp_path / 'out'))
    assert result.returncode == 0, result.stderr
    assert '[WARNING] The configured Observation.Offset_Bound 0.06 m differs from the bound derived from the identification flights, 0.07 m' \
        in result.stderr
    assert 'Observation.White_Bound' not in result.stderr


def test_dataset_check_cannot_check_a_folder_without_a_manifest(root, tmp_path):
    """
    Description:
        'dataset-check' on a folder without manifest.json exits 2 (could not check), never 0.
    """

    result = _Run_Derive(root, tmp_path, 'dataset-check', str(tmp_path))
    assert result.returncode == 2 and 'cannot be checked' in result.stderr


@pytest.mark.slow
def test_dataset_check_exits_0_on_a_regenerated_dataset_and_1_on_a_changed_one(root, tmp_path):
    """
    Description:
        SPEC §2.13.2: 'dataset-check DIR' regenerates a 300-flight dataset from its stored configuration and
        exits 0 (identical); with one byte of flights.npz changed it exits 1, and with the stored seed changed
        it exits 1 again, each difference named.
    """

    out = tmp_path / 'datasets'
    assert _Run(root, tmp_path, 'dataset', '--config', str(_Dataset_Config(root, tmp_path, 300)), '--out', str(out)).returncode == 0
    folder = out / 'Small'
    result = _Run_Derive(root, tmp_path, 'dataset-check', str(folder))
    assert result.returncode == 0 and 'identical' in result.stdout, result.stderr
    data = bytearray((folder / 'flights.npz').read_bytes())
    data[-100] ^= 0xFF
    (folder / 'flights.npz').write_bytes(bytes(data))
    result = _Run_Derive(root, tmp_path, 'dataset-check', str(folder))
    assert result.returncode == 1 and 'differs from the stored one' in result.stderr
    (folder / 'flights.npz').write_bytes(bytes(data[:-100]) + bytes([data[-100] ^ 0xFF]) + bytes(data[-99:]))
    text = (folder / 'dataset.yaml').read_text(encoding='utf-8').replace('  Seed: 20261111', '  Seed: 20261112')
    (folder / 'dataset.yaml').write_text(text, encoding='utf-8')
    result = _Run_Derive(root, tmp_path, 'dataset-check', str(folder))
    assert result.returncode == 1 and 'the regenerated flights.npz' in result.stderr


@pytest.mark.parametrize('change, reason', [('system copy', 'the stored ball_catching.yaml'), ('dataset copy', 'the stored dataset.yaml'),
                                            ('model hash', "differ from the manifest's model_sha256")])
def test_dataset_check_exits_1_on_a_stored_copy_or_model_hash_that_differs_from_the_manifest(root, tmp_path, small_dataset, change, reason):
    """
    Description:
        'dataset-check' compares each stored copy and the model hash with the manifest too: a copy
        of the 30-flight dataset with a comment added to its system or dataset copy (the flights regenerate identical) or
        with another model_sha256 in its manifest exits 1, naming the difference.
    """

    folder = tmp_path / 'Small'
    folder.mkdir()
    for name in ('dataset.yaml', 'ball_catching.yaml', 'flights.npz', 'manifest.json'):
        (folder / name).write_bytes((small_dataset[0] / name).read_bytes())
    if change == 'model hash':
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        manifest['model_sha256'] = '0' * 64
        (folder / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    else:
        name = 'ball_catching.yaml' if change == 'system copy' else 'dataset.yaml'
        (folder / name).write_bytes((folder / name).read_bytes() + b'# edited\n')
    result = _Run_Derive(root, tmp_path, 'dataset-check', str(folder))
    assert result.returncode == 1 and reason in result.stderr and 'the regenerated flights.npz' not in result.stderr, result.stderr


@pytest.mark.slow
@pytest.mark.parametrize('name', CONST_CHECKED_DATASETS)
def test_dataset_check_reproduces_a_stored_dataset(root, tmp_path, name):
    """
    Description:
        'dataset-check' on a stored dataset: Default (its copy written before F4's and F9's
        sections) and the two catch datasets (written before F9's Decision) are regenerated from their stored copies and
        exit 0, identical byte for byte. Data/Generated/ is not in a clean clone: skipped there.
    """

    assert name in _Stored_Folders(root, 'Datasets')
    result = subprocess.run([sys.executable, os.path.join(root, CONST_DERIVE_SCRIPT), 'dataset-check',
                             os.path.join(root, 'Data', 'Generated', 'Datasets', name)], cwd=str(tmp_path), capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0 and 'identical' in result.stdout, result.stderr


def test_scene_prints_the_summary_and_matches_the_fingerprint_from_another_directory(root, tmp_path):
    """
    Description:
        'scene' builds the MuJoCo scene, prints its summary on stdout (the mounting face at 0.180 m, the
        masses, the four areas) with the fingerprint's verdict, the [INFO] lines with the safety sentence
        on stderr, and exits 0 for the shipped scene. It writes nothing, not even MuJoCo's log file, into
        the working directory.
    """

    result = _Run_Derive(root, tmp_path, 'scene')
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == 'Scene: 10 bodies, 7 joints (13 positions, 12 velocities), 6 actuators, 108 geoms (65 collide), 24 meshes'
    assert 'Robot: mounting face at z 0.180 m; joints, range [deg]:' in lines
    assert 'Masses: robot 32.700 kg, cup 0.476 kg, ball 0.122 kg' in lines
    assert sum(line.startswith('Area ') and line.endswith('drawn, visual only') for line in lines) == 4
    assert lines[-1] == 'Fingerprint: identical to the stored one' and '[INFO]' not in result.stdout
    assert all(line.startswith('[INFO] ') for line in result.stderr.splitlines())
    assert 'nothing is run on, or validated on, the real UR10e' in result.stderr
    assert os.listdir(str(tmp_path)) == []


def test_scene_names_each_difference_with_exit_1_and_exits_2_when_it_cannot_check(root, tmp_path):
    """
    Description:
        Against a stored fingerprint in which the floor's colour and the ball's mass were changed, 'scene'
        exits 1 and names exactly those two fields with both values; a fingerprint file that does not
        exist and a configuration without its areas are exit 2 (could not check), each with an [ERROR].
    """

    with open(os.path.join(root, 'models', 'scene_fingerprint.json'), 'r', encoding='utf-8') as file:
        document = json.load(file)
    document['model']['geom']['floor']['rgba'][0] = 0.25
    document['model']['body']['ball']['mass'] = 0.2
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(document), encoding='utf-8')
    result = _Run_Derive(root, tmp_path, 'scene', '--fingerprint', str(changed))
    assert result.returncode == 1, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-3:] == ['Fingerprint: DIFFERS from the stored one in 2 fields',
                          '    body/ball/mass: stored 0.2, the model has 0.122',
                          '    geom/floor/rgba: stored [0.25, 0.57, 0.6, 1.0], the model has [0.55, 0.57, 0.6, 1.0]']
    assert '[ERROR] The compiled scene is not the one that was pinned.' in result.stderr

    result = _Run_Derive(root, tmp_path, 'scene', '--fingerprint', str(tmp_path / 'absent.json'))
    assert result.returncode == 2 and '[ERROR] The fingerprint' in result.stderr and 'cannot be read' in result.stderr
    assert result.stdout == ''
    with open(os.path.join(root, 'Config', 'ball_catching.yaml'), 'r', encoding='utf-8') as file:
        text = file.read()
    broken = tmp_path / 'broken.yaml'
    broken.write_text(text.replace('    Radius: 1.0\n', ''), encoding='utf-8')
    result = _Run_Derive(root, tmp_path, 'scene', '--config', str(broken))
    assert result.returncode == 2 and '[ERROR]' in result.stderr and 'Areas.Robot.Radius' in result.stderr


def _Workspace_Config(root, tmp_path, dataset=None, step=None):
    """
    Description:
        A copy of the shipped system configuration whose workspace reads another dataset (a folder, given as
        a path relative to the repository, as the field requires) and has another lattice step.
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    if dataset is not None:
        assert text.count('  Dataset: Data/Generated/Datasets/Default\n') == 1
        text = text.replace('  Dataset: Data/Generated/Datasets/Default\n', '  Dataset: %s\n' % os.path.relpath(str(dataset), root))
    if step is not None:
        assert text.count('  Workspace_Step: 0.05\n') == 1
        text = text.replace('  Workspace_Step: 0.05\n', '  Workspace_Step: %s\n' % step)
    path = tmp_path / 'ball_catching.yaml'
    path.write_text(text, encoding='utf-8')
    return path


def test_workspace_refuses_a_missing_dataset_and_an_invalid_configuration_and_writes_nothing(root, tmp_path):
    """
    Description:
        'workspace' with a dataset folder that does not exist, and with a configuration whose clearance is 0,
        exits 1 with an [ERROR] naming the cause before any point is checked, and writes nothing.
    """

    out = tmp_path / 'out'
    result = _Run_Derive(root, tmp_path, 'workspace', '--config', str(_Workspace_Config(root, tmp_path, dataset=tmp_path / 'absent')), '--out',
                         str(out))
    assert result.returncode == 1 and '[ERROR]' in result.stderr and 'absent' in result.stderr and 'Checking' not in result.stderr
    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read().replace('  Clearance: 0.05\n', '  Clearance: 0.0\n')
    bad = tmp_path / 'bad.yaml'
    bad.write_text(text, encoding='utf-8')
    result = _Run_Derive(root, tmp_path, 'workspace', '--config', str(bad), '--out', str(out))
    assert result.returncode == 1 and 'Catch.Clearance' in result.stderr
    assert not out.exists()


@pytest.mark.slow
def test_workspace_writes_its_files_from_the_train_partition_only(root, tmp_path, small_dataset, monkeypatch, capsys):
    """
    Description:
        'workspace' on a 0.1 m lattice and the 30-flight dataset, run in-process with the selectors of the
        validation and test partitions and every recording reader replaced by functions that raise: it exits 0
        and writes the map, the summary and the manifest; the summary names the 21 train flights, the 25
        directions, the largest cuboid of the qualifying points of F4 Part A at 0.1 m (x -0.7...0.7,
        y -0.9...-0.6, z 0.5...1.2 m), the reference time 0.7 s and the search's 800 deg/s^2, and the area
        (the rule B800): the map's largest cuboid shrunk by one step, every one of its points
        qualifying and reached from the ready pose within 0.7 s at 800 deg/s^2, no point of the layer beyond
        a face of the cuboid both, and the catchable shares in it of all throws and of those through it at
        both accelerations; the manifest records the train partition and the dataset's hash; no
        MUJOCO_LOG.TXT is left in the working directory.
    """

    from Catch.Workspace import Workspace_Str, Reach_Times
    folder, digest = small_dataset
    config = _Workspace_Config(root, tmp_path, dataset=folder, step=0.1)
    module = _Load_Derive(root)
    _Forbid_Leaks(module, monkeypatch)
    selector = module.Partition_Indices

    def Train_Only(data, code):
        if code != 0:
            raise AssertionError('the partition %d was selected; the workspace reads the train partition only' % code)
        return selector(data, code)

    monkeypatch.setattr(module, 'Partition_Indices', Train_Only)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['derive.py', 'workspace', '--config', str(config), '--out', str(out)])
    assert module.main() == 0
    captured = capsys.readouterr()
    assert sorted(os.listdir(str(out))) == ['manifest.json', 'workspace_map.npz', 'workspace_summary.csv']
    with open(str(out / 'workspace_summary.csv'), newline='', encoding='ascii') as file:
        rows = {row['quantity']: row['value'] for row in csv.DictReader(file)}
    assert rows['train_flights'] == '21' and rows['arrival_directions'] == '25' and rows['lattice_step'] == '0.100'
    assert rows['qualifying_cuboid_lower'] == '-0.700 -0.900 0.500' and rows['qualifying_cuboid_upper'] == '0.700 -0.600 1.200'
    assert rows['reach_time'] == '0.700' and rows['search_acceleration'] == '800' and rows['ready_pose_tcp_in_area'] in ('0', '1')
    for label in ('800', '400'):
        for name in ('volume', 'flights_through', 'catchable', 'catchable_in_area', 'catchable_in_area_of_through'):
            assert 'reach_%s_%s' % (label, name) in rows
    assert float(rows['reach_400_volume']) <= float(rows['reach_800_volume'])
    through = float(rows['flights_through_catch_area_cells'])
    if through > 0.0:
        assert float(rows['reach_800_catchable_in_area_of_through']) == pytest.approx(float(rows['reach_800_catchable_in_area']) / through, abs=1e-3)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['kind'] == 'workspace' and manifest['partition'] == 'train' and manifest['sha256']['dataset'] == digest
    assert 'Only the train partition' in captured.err and not os.path.exists(str(tmp_path / 'MUJOCO_LOG.TXT'))
    with np.load(str(out / 'workspace_map.npz')) as data:
        assert int(data['format_version']) == 1 and data['qualifies'].sum() == int(rows['qualifying_points'])
        workspace = Workspace_Str(Step=float(data['step']), Points=data['points'], Index=data['index'], Axes=data['axes'],
                                  Qualifies=data['qualifies'], Failed_Axis=data['failed_axis'], Failed_Criterion=data['failed_criterion'],
                                  Rows=data['rows'], Poses=data['poses'])
        lower, upper = data['largest_cuboid']
        centre, size = data['catch_area']
        ready = data['ready_pose'][:5]
    assert np.allclose(centre, 0.1 * (lower + upper) / 2.0, atol=1e-9) and np.allclose(size, 0.1 * (upper - lower - 2), atol=1e-9)
    assert rows['catch_area_centre'] == ' '.join('%.3f' % v for v in centre)
    reach = Reach_Times(workspace, ready, np.radians([120.0, 120.0, 180.0, 180.0, 180.0]), np.radians(800.0),
                        np.array([True, True, False, True, True]))
    good = workspace.Qualifies & (reach <= 0.7)
    cuboid = np.all((workspace.Index >= lower) & (workspace.Index <= upper), axis=1)
    assert np.all(good[cuboid]) and cuboid.sum() == np.prod(upper - lower + 1)
    for axis in range(3):
        for layer in (lower[axis] - 1, upper[axis] + 1):
            face = np.all((workspace.Index >= lower) & (workspace.Index <= upper), axis=1, where=np.arange(3) != axis) \
                & (workspace.Index[:, axis] == layer)
            assert not np.all(good[face]) or face.sum() < np.prod(np.delete(upper - lower + 1, axis)), (axis, layer)


def test_the_stored_real_flight_errors_are_read_by_compare_and_by_the_sweep_only(root):
    """
    Description:
        F11's guard, read from the code. In scripts/catch.py the reader of a stored
        evaluation's per-horizon errors is named by one function only, 'compare''s reader of a stored evaluation, and no
        function names the sweep's error tables or the injected predictor: the user's commands cannot reach them. In
        scripts/Maintenance/derive.py the reader is named by the sweep's error tables only, and those tables and the
        injected predictor by 'sweep' only. In src no module outside Evaluation names the reader, and no module imports
        Catch.Robustness, which itself imports nothing of the decision, the interception, the planner or a learned
        predictor.
    """

    import ast

    def Naming(script, name):
        tree = ast.parse(open(os.path.join(root, script), encoding='utf-8').read())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        return {function.name for function in functions
                if any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(function))
                or any(isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names) for node in ast.walk(function))}

    assert Naming(CONST_SCRIPT, 'Read_Per_Horizon') == {'_Stored_Evaluation'}
    assert all(Naming(CONST_SCRIPT, name) == set() for name in ('_Sweep_Errors', 'Injected_Predictor', 'Error_Table'))
    assert Naming(CONST_DERIVE_SCRIPT, 'Read_Per_Horizon') == {'_Sweep_Errors'}
    assert Naming(CONST_DERIVE_SCRIPT, '_Sweep_Errors') == {'_Sweep'} and Naming(CONST_DERIVE_SCRIPT, 'Injected_Predictor') == {'_Sweep'}
    assert Naming(CONST_DERIVE_SCRIPT, 'Error_Table') == {'_Sweep_Errors'}
    sources = {}
    for folder, _, names in os.walk(os.path.join(root, 'src')):
        for name in names:
            if name.endswith('.py'):
                path = os.path.join(folder, name)
                sources[os.path.relpath(path, os.path.join(root, 'src'))] = open(path, encoding='utf-8').read()
    assert {name for name, text in sources.items() if 'Read_Per_Horizon' in text} == {os.path.join('Evaluation', 'Report.py'),
                                                                                      os.path.join('Evaluation', 'Comparison.py')}
    assert not [name for name, text in sources.items() if 'from Catch.Robustness' in text or 'import Catch.Robustness' in text]
    robustness = ast.parse(sources[os.path.join('Catch', 'Robustness.py')])
    imported = {node.module for node in ast.walk(robustness) if isinstance(node, ast.ImportFrom)}
    assert imported == {'dataclasses', 'Flight.Core', 'Prediction.Core', 'Catch.Core'}


def test_sweep_refuses_no_worker_and_a_missing_interception_before_any_read(root, tmp_path, small_dataset, catch_dataset, monkeypatch, capsys):
    """
    Description:
        F11 (SPEC §9.14): 'sweep' refuses, each with exit 1 and its message, before any dataset's data or stored error is
        read and with nothing written: --workers 0, and a stored validation interception that is not there (naming the
        command that writes it and --interceptions); the test-partition selector and every recording reader are guarded
        in this script and in the user script it loads, and the reader of the stored errors is replaced by a failure.
    """

    module = _Load_Derive(root)
    _Forbid_Leaks(module, monkeypatch)
    for name in ('Read_Dataset', 'Read_Per_Horizon', '_Sweep_Errors'):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail('the sweep read data before its refusal'))
    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'catches'
    config = _Decision_Config(root, tmp_path, catch_dataset)
    assert module._Sweep(_Sweep_Args(run, config, out, tmp_path / 'none', tmp_path / 'evaluations', workers=0)) == 1
    assert '--workers must be an integer >= 1' in capsys.readouterr().err
    assert module._Sweep(_Sweep_Args(run, config, out, tmp_path / 'none', tmp_path / 'evaluations')) == 1
    error = capsys.readouterr().err
    assert 'Run scripts/catch.py catch on the validation partition first' in error and '--interceptions' in error
    assert not os.path.exists(out)


@pytest.mark.slow
def test_the_sweep_injects_the_stored_errors_with_the_stored_margins_and_writes_none(root, tmp_path, stored_chain, monkeypatch):
    """
    Description:
        F11 Q10 and the guard: 'sweep' on the first 6 validation throws with the first seed's stored margins (the
        validation chain written by scripts/catch.py catch), from constructed stored evaluations (12 real flights whose
        error grows with the horizon, the simulated ones half of it): the run's own predictor, the real errors at the
        scales 0 and 2, and the simulated errors at 1, each 6 throws x 2 accelerations, the rows beginning with the
        error's source and scale; the run's rows are the stored interception's decisions; at scale 0 the prediction is
        the true trajectory, so no go is false (with this module's quickly trained run's margins no throw goes at all:
        the sweep's goes are the stored run's test's). The manifest names the sweep and both stored files' hashes; the
        stored margins are byte for byte unchanged and the sweep writes none; the test partition and every recording
        stay guarded.
    """

    module = _Load_Derive(root)
    _Forbid_Leaks(module, monkeypatch)
    run, config, chain = stored_chain
    interceptions = chain / 'Interceptions'
    evaluations = tmp_path / 'evaluations'
    _Stored_Errors(evaluations / 'test', 'real', 12, lambda flight, n, index: (4.0 + 6.0 * index) * (1.0 + 0.1 * flight) * (1.0 if n == 20 else 0.8))
    _Stored_Errors(evaluations / 'validation', 'validation', 12,
                   lambda flight, n, index: (2.0 + 3.0 * index) * (1.0 + 0.1 * flight) * (1.0 if n == 20 else 0.8))
    margins = interceptions / 'Learned' / 'validation' / 'margins.csv'
    before = margins.read_bytes()
    assert module._Sweep(_Sweep_Args(run, config, tmp_path / 'out', interceptions, evaluations)) == 0
    folder = tmp_path / 'out' / 'Learned' / 'sweep'
    rows = _Stored_Rows(folder / 'per_throw.csv')
    cases = [(row['error_source'], row['error_scale']) for row in rows]
    assert sorted(set(cases)) == [('real', '0.0'), ('real', '2.0'), ('run', 'nan'), ('simulated', '1.0')]
    assert all(cases.count(case) == 12 for case in set(cases)) and {row['seed'] for row in rows} == {'0'}
    stored = {(row['flight'], row['acceleration_deg_s2']): row['decision']
              for row in _Stored_Rows(interceptions / 'Learned' / 'validation' / 'per_throw.csv') if row['seed'] == '0'}
    for row in rows:
        if row['error_source'] == 'run':
            assert row['decision'] == stored[(row['flight'], row['acceleration_deg_s2'])]
        if (row['error_source'], row['error_scale']) == ('real', '0.0'):
            assert not (row['decision'] == 'go' and row['catchable'] == 'false')
        if row['decision'] == 'go':
            assert (row['class'] == 'caught') == (row['retained'] == 'true') and row['robot_contact'] == '' and row['first_contact'] != ''
    summary = _Stored_Rows(folder / 'summary.csv')
    assert len(summary) == 8 and all(row['throws'] == '6' for row in summary)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['sweep_scales'], manifest['sweep_throws'], manifest['seeds']) == ([0.0, 2.0], 6, ['0'])
    assert manifest['r2_one_class_and_no_robot_contact'] is True and 'r1_interception_rows_identical' not in manifest
    assert manifest['sweep_evaluations'] == {name: hashlib.sha256((evaluations / name / 'per_horizon.csv').read_bytes()).hexdigest()
                                             for name in ('test', 'validation')}
    assert manifest['sha256']['interception_margins'] == hashlib.sha256(before).hexdigest()
    assert margins.read_bytes() == before and sorted(os.listdir(str(folder))) == ['manifest.json', 'per_throw.csv', 'summary.csv']


def test_an_abbreviated_option_and_a_user_sub_command_are_usage_errors(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        Scripts/Maintenance/derive.py has seven sub-commands (recordings, identify,
        observation, workspace, scene, dataset-check, sweep); a sub-command of the user script (dataset, train, catch) and
        an abbreviated option (--con for --config, --o for --out) are argparse usage errors, exit 2, before anything
        runs: the recordings' reader and the identification ledger's writer are replaced by failures, so 'identify --o
        DIR' never counts a read.
    """

    module = _Load_Derive(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(module, 'Append_Ledger', lambda *args, **kwargs: pytest.fail('the identification ledger was written'))

    def Main(*args):
        monkeypatch.setattr(sys, 'argv', [CONST_DERIVE_SCRIPT, *args])
        capsys.readouterr()
        try:
            code = module.main()
        except SystemExit as stop:
            code = stop.code
        return code, capsys.readouterr().err

    for command in ('dataset', 'train', 'catch'):
        code, error = Main(command)
        assert code == 2 and "invalid choice: '%s'" % command in error, command
        assert '{recordings,identify,observation,workspace,scene,dataset-check,sweep}' in error
    for arguments, option in ((('identify', '--o', str(tmp_path / 'out')), '--o'), (('identify', '--con', 'x.yaml'), '--con'),
                              (('recordings', '--o', str(tmp_path / 'out')), '--o')):
        code, error = Main(*arguments)
        assert code == 2 and 'unrecognized arguments: %s' % option in error, arguments
    assert not os.path.exists(tmp_path / 'out')
