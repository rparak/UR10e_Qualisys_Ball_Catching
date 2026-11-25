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
# File Name: ../test/scripts/test_catch.py
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
# PyYAML (YAML parser and emitter; a stored copy written before later sections) [pip3 install pyyaml]
import yaml
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: a run's datasets named for the check of the stored datasets
#   ../Configuration/Core
from Configuration.Core import Training_Configuration_Str, Run_Dataset_Str
#   ./Script_Helpers - the scripts' shared test helpers
from Script_Helpers import (CONST_SCRIPT, _Run, _Dataset_Config, _Stored_Folders, _Training_Config, _Load_Script, _Load_Derive, _Forbid_Leaks,
                            _Run_Guarded, _Trained_Run, _Learned_Config, _Stored_Rows, _Decision_Config, _Catch_Args, _Sweep_Args,
                            _Forbid_Stored_Errors, _Stored_Errors, _Moving_Config)

"""
Description:
    Initialization of constants.

    scripts/catch.py, the user's entry point (dataset, train, evaluate, compare, catch, view), is run
    the way a person runs it: a fresh interpreter, from an unrelated working directory. The
    derivations' tests are test/scripts/Maintenance/test_derive.py.
"""


# The real-flight results of the counted 'test' evaluation of GLS_M4 (test ledger 1, commit 8eecc80, 2026-11-15;
# F7 Part B), split from its stored outputs into two parts. The prediction part: the SHA-256 of
# the real rows of per_flight.csv (inference_ms blanked: a timing) and of per_horizon.csv; the predicted flights per
# N; the median remainder RMS per N [mm]. A regression pin: a rerun that reproduces it byte for byte is not a counted
# evaluation. The pin of LS (test ledger 2) left with the predictor; its
# record is SPEC §4.7. GLS_M4's digest is unchanged and never re-captured.
#   Unit [Dict<string, Tuple>]
CONST_PINNED_PREDICTIONS = {'GLS_M4': ('83289273ce2f1a22382e8840257d7ec10572d21a4613db9a1689791c9f0a0e5d', (45, 45, 43),
                                       (52.134033, 37.549502, 20.568150))}


# The coverage part: the SHA-256 of coverage.csv and the flights inside the support of
# SPEC §2.13.6, for F7's dataset configuration (the targets in the half of Areas.Robot facing the thrower, r 1.0 m).
# A record, superseded by Throw.Target_Area (r 1.5 m): the validation flights' coverage
# with the new targets is not computed before F14's counted 'test'.
#   Unit [Tuple<string, int>]
CONST_F7_COVERAGE = ('86394cd0ed3e949edba1839704cf61b509424c7d89f3cba4b51ea7ca174e628e', 34)


# The columns of per_flight.csv and per_horizon.csv as the two counted evaluations wrote them (F7, 2026-11-15): the
# pinned digests are taken over these; later columns follow them.
#   Unit [Tuple<string>]
CONST_F7_PER_FLIGHT = ('flight', 'partition', 'dataset', 'n', 'prefix_duration_s', 'rate_hz', 'remainder_rms_mm', 'end_mismatch_ms',
                       'catch_point_error_mm', 'time_to_catch_error_ms', 'crossing', 'inference_ms', 'refusal')


CONST_F7_PER_HORIZON = ('flight', 'partition', 'dataset', 'n', 'horizon', 'truth', 'position_error_mm', 'samples')


def test_a_missing_sub_command_is_a_usage_error(root, tmp_path):
    """
    Description:
        Without a sub-command the script stops with argparse's usage error (exit 2) and writes nothing.
    """

    result = _Run(root, tmp_path)
    assert result.returncode == 2 and 'usage' in result.stderr


def _Main(module, monkeypatch, capsys, *args):
    """
    Description:
        The script's main() in process with the arguments: (exit code, stderr). argparse's usage error leaves through
        SystemExit, whose code is returned.
    """

    monkeypatch.setattr(sys, 'argv', [os.path.join('scripts', 'catch.py'), *args])
    capsys.readouterr()
    try:
        code = module.main()
    except SystemExit as stop:
        code = stop.code
    return code, capsys.readouterr().err


def test_the_removed_sub_commands_and_options_are_usage_errors(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        The user's command line: scripts/catch.py has six sub-commands. The merged
        ones (validate, test, decide, intercept), the derivations moved to scripts/Maintenance/derive.py (recordings,
        identify, observation, scene, workspace) and the moved or removed options (dataset --check, catch --sweep, catch
        --interceptions) are argparse usage errors, exit 2, each naming what was given; nothing runs: every reader of a
        recording, of a dataset and of a run is replaced by a failure. The six that remain are named by the usage line.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    for name in ('Read_Dataset', '_Load_Run_Copy', '_Real_Flights', 'Append_Test_Ledger'):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail('a removed sub-command ran'))
    for command in ('validate', 'test', 'decide', 'intercept', 'recordings', 'identify', 'observation', 'scene', 'workspace'):
        code, error = _Main(module, monkeypatch, capsys, command, '--run', str(tmp_path))
        assert code == 2 and "invalid choice: '%s'" % command in error, command
        assert '{dataset,train,evaluate,compare,catch,view}' in error
    for arguments, option in ((('dataset', '--check', str(tmp_path)), '--check'),
                              (('catch', '--run', str(tmp_path), '--sweep', str(tmp_path)), '--sweep'),
                              (('catch', '--run', str(tmp_path), '--interceptions', str(tmp_path)), '--interceptions')):
        code, error = _Main(module, monkeypatch, capsys, *arguments)
        assert code == 2 and 'unrecognized arguments: %s' % option in error, option
    assert os.listdir(str(tmp_path)) == []


def test_an_abbreviated_option_is_a_usage_error_and_never_a_counted_read(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        An option is given by its full name. argparse would otherwise read '--part test' as
        '--partition test', a spelling that a rule matching the counted forms would not match. With
        the abbreviations --part, --p (also as --p=test) and --partitio, 'evaluate' and 'catch' stop with a usage error,
        exit 2, before the run is opened: the run's reader, the real flights and the ledger's writer are replaced by
        failures. The full name is accepted (the run folder given is then refused as not a run, exit 1).
    """

    module = _Load_Script(root)
    for name in ('_Load_Run', '_Load_Run_Copy', '_Real_Flights', 'Append_Test_Ledger'):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail('an abbreviated option reached the command'))
    for command in ('evaluate', 'catch'):
        for abbreviation in (('--part', 'test'), ('--p', 'test'), ('--p=test',), ('--partitio', 'test')):
            code, error = _Main(module, monkeypatch, capsys, command, '--run', str(tmp_path), *abbreviation)
            assert code == 2 and 'unrecognized arguments: %s' % abbreviation[0] in error, (command, abbreviation)
        code, error = _Main(module, monkeypatch, capsys, command, '--ru', str(tmp_path))
        assert code == 2 and 'the following arguments are required: --run' in error, command
    monkeypatch.undo()
    module = _Load_Script(root)
    monkeypatch.setattr(module, 'Append_Test_Ledger', lambda *args, **kwargs: pytest.fail('a ledger was written'))
    for command in ('evaluate', 'catch'):
        code, error = _Main(module, monkeypatch, capsys, command, '--run', str(tmp_path), '--partition', 'test', '--out', str(tmp_path / 'out'))
        assert code == 1 and 'is not a run folder' in error, command
    assert not os.path.exists(tmp_path / 'out')


def test_a_run_of_a_removed_predictor_is_refused_by_evaluate_catch_and_view(root, tmp_path, small_dataset, monkeypatch, capsys):
    """
    Description:
        No command runs LS, QT_MLP or HYBRID, on new data or old. A stored run of one of them (here a
        run folder whose training.yaml copy names the predictor, its manifest agreeing with the copy, as 'train' wrote
        such runs before 2026-11-23) is refused by 'evaluate' (both partitions), 'catch' (both partitions) and 'view'
        (as the driving run, given the module's own small dataset, so that the test needs nothing of the git-ignored
        Data/Generated/; as a --compare run: the slow test of 'view', which has a driving run with its stored
        interception), exit 1, the message naming the two predictors that remain; no dataset's data is read, no ledger is
        written and nothing is written at all.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    for name in ('Read_Dataset', '_Real_Flights', 'Append_Test_Ledger'):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail('a removed predictor\'s run was used'))
    good = _Trained_Run(root, tmp_path, [small_dataset])
    for model in ('LS', 'QT_MLP', 'HYBRID'):
        run = tmp_path / 'models' / model
        run.mkdir()
        text = (good / 'training.yaml').read_text(encoding='utf-8').replace('  Name: GLS_M4\n', '  Name: %s\n' % model)
        assert '  Name: %s\n' % model in text
        (run / 'training.yaml').write_text(text, encoding='utf-8')
        manifest = json.loads((good / 'manifest.json').read_text(encoding='utf-8'))
        manifest['model'], manifest['sha256']['configuration'] = model, hashlib.sha256(text.encode('utf-8')).hexdigest()
        (run / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        out = tmp_path / 'out'
        calls = [('evaluate', '--run', str(run), '--partition', partition, '--out', str(out)) for partition in ('validation', 'test')]
        calls += [('catch', '--run', str(run), '--partition', partition, '--out', str(out)) for partition in ('validation', 'test')]
        calls += [('view', '--run', str(run), '--dataset', str(small_dataset[0]), '--index', '0', '--frames', str(out))]
        for arguments in calls:
            code, error = _Main(module, monkeypatch, capsys, *arguments)
            assert code == 1 and 'Model.Name must be one of GLS_M4, QT_BASIS, found \'%s\'' % model in error, (model, arguments, error)
        assert not os.path.exists(out)


def test_dataset_writes_its_folder_and_never_overwrites_it(root, tmp_path):
    """
    Description:
        'dataset' (SPEC §9.5) writes <out>/<Dataset.Name>/ with flights.npz, the two configuration copies (byte
        for byte the files read) and manifest.json (the hashes of the copies and the data, the seed, the
        largest-remainder sizes 14 / 3 / 3 of 20 flights, the git state, the versions); no temporary folder is
        left. A second run is refused (exit 1) and leaves the dataset as it was.
    """

    config = _Dataset_Config(root, tmp_path, 20)
    out = tmp_path / 'datasets'
    result = _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(out))
    assert result.returncode == 0, result.stderr
    folder = out / 'Small'
    assert sorted(os.listdir(out)) == ['Small']
    assert sorted(os.listdir(folder)) == ['ball_catching.yaml', 'dataset.yaml', 'flights.npz', 'manifest.json']
    assert (folder / 'dataset.yaml').read_bytes() == config.read_bytes()
    assert (folder / 'ball_catching.yaml').read_bytes() == open(os.path.join(root, 'Config', 'ball_catching.yaml'), 'rb').read()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['sha256'] == {'configuration': hashlib.sha256(config.read_bytes()).hexdigest(),
                                  'system_configuration': hashlib.sha256((folder / 'ball_catching.yaml').read_bytes()).hexdigest(),
                                  'data': hashlib.sha256((folder / 'flights.npz').read_bytes()).hexdigest()}
    assert (manifest['kind'], manifest['seed'], manifest['flights'], manifest['partition_sizes']) == ('dataset', 20261111, 20, [14, 3, 3])
    assert len(manifest['git_commit']) == 40 and isinstance(manifest['git_dirty'], bool) and manifest['numpy'] == '2.4.6'
    before = (folder / 'flights.npz').read_bytes()
    again = _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(out))
    assert again.returncode == 1 and 'is never overwritten' in again.stderr
    assert (folder / 'flights.npz').read_bytes() == before and sorted(os.listdir(out)) == ['Small']


def test_dataset_refuses_a_rate_without_donors_and_leaves_nothing(root, tmp_path):
    """
    Description:
        A rate without donors (150 Hz) is refused (exit 1) naming it, and nothing is left in the output folder.
    """

    out = tmp_path / 'datasets'
    result = _Run(root, tmp_path, 'dataset', '--config', str(_Dataset_Config(root, tmp_path, 5, '[200, 150]')), '--out', str(out))
    assert result.returncode == 1 and '150 Hz has no donors' in result.stderr
    assert not out.exists() or os.listdir(out) == []


def test_dataset_generates_with_the_bounds_of_its_system_configuration(root, tmp_path):
    """
    Description:
        'dataset' builds the error model with the configured bounds: from a system
        configuration with the offset alone (white noise and drift 0), an offset of 50 mm per axis and
        Observation.Offset_Bound 0.010 m, every observed position of 20 flights lies within 10 mm of the truth of
        its frame, and some lie beyond 5 mm (the offset is there; without the bound it would be 87 mm RMS).
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    for field, value in (('White_Std', '[0.0, 0.0, 0.0]'), ('Drift_Std', '[0.0, 0.0, 0.0]'), ('Offset_Std', '[0.05, 0.05, 0.05]'),
                         ('Offset_Bound', '0.010')):
        text, count = re.subn(r'(?m)^  %s: .*$' % field, '  %s: %s' % (field, value), text)
        assert count == 1, field
    system = tmp_path / 'system.yaml'
    system.write_text(text, encoding='utf-8')
    config = _Dataset_Config(root, tmp_path, 20)
    config.write_text(config.read_text(encoding='utf-8').replace('  System_Configuration: Config/ball_catching.yaml\n',
                                                                 '  System_Configuration: %s\n' % os.path.relpath(str(system), root)),
                      encoding='utf-8')
    out = tmp_path / 'datasets'
    result = _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(out))
    assert result.returncode == 0, result.stderr
    with np.load(out / 'Small' / 'flights.npz') as data:
        frames = np.concatenate([data['truth_offset'][i] + data['observed_frame'][data['observed_offset'][i]:data['observed_offset'][i + 1]]
                                 for i in range(20)])
        error = np.linalg.norm(data['observed_position'] - data['truth_position'][frames], axis=1)
    assert error.max() <= 0.010 + 1e-12 and error.max() > 0.005


# The model hash of the shipped Flight and Observation sections: the manifests' model_sha256 of the 22 stored datasets
# and the 31 stored runs (measured 2026-11-21).
#   Unit [string, hex]
CONST_MODEL_HASH = '9796a4fc374e8bacc0f125125b8faaccce19a420ab9ed30e326a9fa3c6541591'


# The stored datasets and runs of Data/Generated/: every one must load, and an unlisted or a
# missing one fails when Data/Generated/Datasets/ exists; a new or an archived one is added or removed here.
#   Unit [Tuple<string>]
CONST_STORED_DATASETS = ('Catch_In', 'Catch_Near', 'Default', 'Large', 'Noise_0', 'Noise_0_25', 'Noise_0_5', 'Noise_1_5', 'Pool_A',
                         'Pool_B', 'Seed_1', 'Seed_1_Noise_0', 'Seed_1_Noise_0_25', 'Seed_1_Noise_0_5', 'Seed_1_Noise_1_5', 'Seed_2',
                         'Seed_2_Noise_0', 'Seed_2_Noise_0_25', 'Seed_2_Noise_0_5', 'Seed_2_Noise_1_5', 'Seed_3', 'Seed_4')


CONST_STORED_RUNS = ('GLS_M4', 'GLS_M4_F14', 'GLS_M4_F8', 'GLS_M4_Noise_0_5', 'GLS_M4_Noise_1_5', 'QT_BASIS',
                     'QT_BASIS_F1000', 'QT_BASIS_F2000', 'QT_BASIS_F21000', 'QT_BASIS_F4000', 'QT_BASIS_Mix3', 'QT_BASIS_Mix5',
                     'QT_BASIS_Noise_0_5', 'QT_BASIS_Noise_1_0', 'QT_BASIS_Noise_1_5', 'QT_BASIS_Pool_A', 'QT_BASIS_Seed_1',
                     'QT_BASIS_Seed_2', 'QT_BASIS_Velocity')


def _Stored_Dataset(root, folder, without=()):
    """
    Description:
        A stored dataset's folder as 'dataset' writes it, with a stand-in flights.npz ('train' of a numerical predictor
        reads no flight): the shipped dataset configuration's bytes, the shipped system configuration without the
        sections named (written before them), and a manifest with their hashes, the data's and CONST_MODEL_HASH:
        (folder, data hash).
    """

    folder.mkdir(parents=True)
    dataset_bytes = open(os.path.join(root, 'Config', 'dataset.yaml'), 'rb').read()
    with open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8') as file:
        system = yaml.safe_load(file)
    for section in without:
        del system[section]
    system_bytes, data = yaml.safe_dump(system, sort_keys=False).encode('utf-8'), b'a stand-in for the flights'
    for name, content in (('dataset.yaml', dataset_bytes), ('ball_catching.yaml', system_bytes), ('flights.npz', data)):
        (folder / name).write_bytes(content)
    digest = lambda content: hashlib.sha256(content).hexdigest()
    manifest = {'kind': 'dataset', 'model_sha256': CONST_MODEL_HASH,
                'sha256': {'configuration': digest(dataset_bytes), 'system_configuration': digest(system_bytes), 'data': digest(data)}}
    (folder / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return folder, digest(data)


def test_train_reads_a_stored_copy_written_before_the_later_sections(root, tmp_path):
    """
    Description:
        KNOWN_ISSUES until 2026-11-21: a dataset whose stored ball_catching.yaml was written before F4's and F9's REQUIRED
        sections (Robot, Catch, Workspace, Decision) stopped 'train', 'validate' and 'test' with 18 problems. The copy is
        now read for the sections a dataset depends on only: 'train' of a numerical predictor
        writes its run, with the dataset's model hash.
    """

    dataset = _Stored_Dataset(root, tmp_path / 'datasets' / 'Before', without=('Robot', 'Catch', 'Workspace', 'Decision'))
    result = _Run(root, tmp_path, 'train', '--config', str(_Training_Config(root, tmp_path, [dataset])))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / 'models' / 'Run' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['model_sha256'] == CONST_MODEL_HASH


@pytest.mark.parametrize('change, reason', [('system copy', 'stored copies that differ from its manifest (ball_catching.yaml'),
                                            ('dataset copy', 'stored copies that differ from its manifest (dataset.yaml'),
                                            ('model hash', "not its manifest's model_sha256")])
def test_train_refuses_a_stored_copy_or_model_hash_that_differs_from_the_manifest_and_writes_nothing(root, tmp_path, change, reason):
    """
    Description:
        A stored copy is a record: a system or dataset copy edited after 'dataset' (a comment added:
        it still loads, with the same models) or a manifest whose model hash is not that of the copy's models is refused,
        naming it, and no run is written.
    """

    folder, digest = _Stored_Dataset(root, tmp_path / 'datasets' / 'Stored')
    if change == 'model hash':
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        manifest['model_sha256'] = '0' * 64
        (folder / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    else:
        name = 'ball_catching.yaml' if change == 'system copy' else 'dataset.yaml'
        (folder / name).write_bytes((folder / name).read_bytes() + b'# edited\n')
    result = _Run(root, tmp_path, 'train', '--config', str(_Training_Config(root, tmp_path, [(folder, digest)])))
    assert result.returncode == 1 and reason in result.stderr, result.stderr
    assert not (tmp_path / 'models').exists()


def test_dataset_refuses_an_incomplete_shipped_system_configuration_and_writes_nothing(root, tmp_path):
    """
    Description:
        A new dataset's system configuration is loaded in full: one without the Robot section is
        refused before any flight is drawn, naming its fields, and nothing is left in the output folder. Only a stored
        dataset's copy is read for the four sections a dataset depends on.
    """

    with open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8') as file:
        system = yaml.safe_load(file)
    del system['Robot']
    (tmp_path / 'system.yaml').write_text(yaml.safe_dump(system, sort_keys=False), encoding='utf-8')
    config = _Dataset_Config(root, tmp_path, 30)
    text = config.read_text(encoding='utf-8')
    assert text.count('  System_Configuration: Config/ball_catching.yaml\n') == 1
    config.write_text(text.replace('  System_Configuration: Config/ball_catching.yaml\n', '  System_Configuration: %s\n'
                                   % os.path.relpath(str(tmp_path / 'system.yaml'), root)), encoding='utf-8')
    out = tmp_path / 'datasets'
    result = _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(out))
    assert result.returncode == 1 and 'Robot.Joint_Speed is missing' in result.stderr, result.stderr
    assert os.listdir(str(out)) == []


@pytest.mark.slow
def test_every_stored_dataset_and_run_loads(root):
    """
    Description:
        Every stored dataset of CONST_STORED_DATASETS passes the checks of a run's datasets (its data hash, its two copies'
        hashes and its model hash against its manifest, its copies loaded), and every stored run
        of CONST_STORED_RUNS loads as 'validate' and 'test' load it, a learned run's datasets trained on included; no
        flight is read. An unlisted or a missing folder fails. Data/Generated/ is not in a clean clone: skipped there.
    """

    assert _Stored_Folders(root, 'Datasets') == sorted(CONST_STORED_DATASETS)
    assert _Stored_Folders(root, 'Models') == sorted(CONST_STORED_RUNS)
    module, problems = _Load_Script(root), []
    for name in CONST_STORED_DATASETS:
        path = os.path.join('Data', 'Generated', 'Datasets', name)
        with open(os.path.join(root, path, 'manifest.json'), encoding='utf-8') as file:
            digest = json.load(file)['sha256']['data']
        try:
            module._Run_Datasets(Training_Configuration_Str(Datasets=[Run_Dataset_Str(Path=path, Hash=digest)]))
        except (module.Configuration_Error_Cls, module.Run_Error_Cls) as error:
            problems.append('%s: %s' % (name, error))
    for name in CONST_STORED_RUNS:
        try:
            training = module._Load_Run(os.path.join(root, 'Data', 'Generated', 'Models', name))[0]
            if training.Model_Name in module.CONST_LEARNED_MODEL_NAMES:
                module._Run_Datasets(training, True)
        except (module.Configuration_Error_Cls, module.Run_Error_Cls) as error:
            problems.append('%s: %s' % (name, error))
    assert problems == []


@pytest.mark.slow
def test_the_shipped_dataset_configurations_are_the_stored_datasets_own(root):
    """
    Description:
        The seven dataset configurations the repository ships (Config/dataset.yaml and Config/Datasets/) hold exactly
        the values of the stored datasets' own configuration copies (the comments apart), so
        'dataset' with a shipped file generates the dataset whose hash the shipped configurations name. Data/Generated/ is
        not in a clean clone: skipped there.
    """

    assert set(_Stored_Folders(root, 'Datasets')) >= {'Default', 'Noise_0', 'Noise_0_25', 'Noise_0_5', 'Noise_1_5', 'Catch_In', 'Catch_Near'}
    shipped = {'Default': os.path.join(root, 'Config', 'dataset.yaml')}
    shipped.update({name[:-5]: os.path.join(root, 'Config', 'Datasets', name) for name in os.listdir(os.path.join(root, 'Config', 'Datasets'))})
    assert len(shipped) == 7
    for name, path in shipped.items():
        with open(path, encoding='utf-8') as file:
            own = yaml.safe_load(file)
        with open(os.path.join(root, 'Data', 'Generated', 'Datasets', name, 'dataset.yaml'), encoding='utf-8') as file:
            assert yaml.safe_load(file) == own, name


def test_train_writes_the_run_folder_of_a_numerical_predictor_and_never_overwrites_it(root, tmp_path, small_dataset):
    """
    Description:
        'train' (SPEC §9.6; F7) writes <Output.Folder>/<Run.Name>/ with the training configuration's copy (byte for
        byte) and manifest.json only (kind run, the predictor, N, the dataset with its hash, the model hash, the
        git state): nothing is fitted. A second run of the same name is refused (exit 1) and leaves it as it was.
    """

    config = _Training_Config(root, tmp_path, [small_dataset])
    result = _Run(root, tmp_path, 'train', '--config', str(config))
    assert result.returncode == 0, result.stderr
    folder = tmp_path / 'models' / 'Run'
    assert sorted(os.listdir(folder)) == ['manifest.json', 'training.yaml'] and (folder / 'training.yaml').read_bytes() == config.read_bytes()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['kind'], manifest['model'], manifest['observed_samples']) == ('run', 'GLS_M4', [10, 20])
    assert manifest['datasets'] == [{'path': os.path.relpath(str(small_dataset[0]), root), 'sha256': small_dataset[1]}]
    assert len(manifest['model_sha256']) == 64 and isinstance(manifest['git_dirty'], bool)
    again = _Run(root, tmp_path, 'train', '--config', str(config))
    assert again.returncode == 1 and 'is never overwritten' in again.stderr and sorted(os.listdir(folder)) == ['manifest.json', 'training.yaml']


def test_train_refuses_a_dataset_whose_hash_differs_and_writes_nothing(root, tmp_path, small_dataset):
    """
    Description:
        A run naming the dataset with another hash is refused (exit 1) naming the dataset and both hashes; no run
        folder is written (SPEC §8.3).
    """

    config = _Training_Config(root, tmp_path, [(small_dataset[0], 'a' * 64)])
    result = _Run(root, tmp_path, 'train', '--config', str(config))
    assert result.returncode == 1 and small_dataset[1] in result.stderr and 'a' * 64 in result.stderr
    assert not (tmp_path / 'models').exists()


def test_train_refuses_datasets_generated_with_different_models_and_writes_nothing(root, tmp_path, small_dataset):
    """
    Description:
        SPEC §2.5 (one flight model drives the datasets and the predictor): a run naming the 30-flight dataset and a
        10-flight one generated with another drag (0.0095 1/m) is refused (exit 1, naming both datasets); no run
        folder is written. So no run that 'test' could read mixes models (each refusal before a
        counted read fails a test when removed).
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    assert text.count('  Drag: 0.00930496257\n') == 1
    system = tmp_path / 'system.yaml'
    system.write_text(text.replace('  Drag: 0.00930496257\n', '  Drag: 0.0095\n'), encoding='utf-8')
    config = _Dataset_Config(root, tmp_path, 10)
    config.write_text(config.read_text(encoding='utf-8').replace('  System_Configuration: Config/ball_catching.yaml\n',
                                                                 '  System_Configuration: %s\n' % os.path.relpath(str(system), root)),
                      encoding='utf-8')
    result = _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(tmp_path / 'datasets'))
    assert result.returncode == 0, result.stderr
    other = tmp_path / 'datasets' / 'Small'
    result = _Run(root, tmp_path, 'train', '--config', str(_Training_Config(root, tmp_path, [small_dataset, (other, hashlib.sha256(
        (other / 'flights.npz').read_bytes()).hexdigest())])))
    assert result.returncode == 1 and 'generated with different flight or observation models' in result.stderr
    assert os.path.relpath(str(other), root) in result.stderr and os.path.relpath(str(small_dataset[0]), root) in result.stderr
    assert not (tmp_path / 'models').exists()


def test_validate_scores_the_validation_partition_and_never_overwrites(root, tmp_path, small_dataset):
    """
    Description:
        'validate --run DIR --out DIR' (SPEC §2.13.8, §9.7) writes <out>/<Run.Name>/validation/ with the five files:
        per_flight.csv has the 5 validation flights (codes 21-25 of 30) for each N, all 'validation', each with
        a crossing category and the inference time; summary.csv and per_horizon.csv are not empty; the report
        is printed. A second call is refused (exit 1).
    """

    assert _Run(root, tmp_path, 'train', '--config', str(_Training_Config(root, tmp_path, [small_dataset]))).returncode == 0
    run, out = tmp_path / 'models' / 'Run', tmp_path / 'evaluations'
    result = _Run(root, tmp_path, 'evaluate', '--run', str(run), '--out', str(out))
    assert result.returncode == 0, result.stderr
    folder = out / 'Run' / 'validation'
    assert sorted(os.listdir(folder)) == ['manifest.json', 'per_flight.csv', 'per_horizon.csv', 'predictions.npz', 'summary.csv']
    with open(folder / 'per_flight.csv', encoding='ascii', newline='') as file:
        rows = list(csv.DictReader(file))
    assert {r['flight'] for r in rows} == {'21', '22', '23', '24', '25'} and {r['partition'] for r in rows} == {'validation'}
    assert {r['n'] for r in rows} == {'10', '20'} and all(r['crossing'] in ('both', 'disagree', 'passed', 'neither') for r in rows)
    assert all(float(r['inference_ms']) > 0.0 for r in rows)
    assert len((folder / 'summary.csv').read_text().splitlines()) > 20 and len((folder / 'per_horizon.csv').read_text().splitlines()) > 20
    assert 'validation Small N=10' in result.stdout and 'The test partition and the recordings were not read' in result.stderr
    again = _Run(root, tmp_path, 'evaluate', '--run', str(run), '--out', str(out))
    assert again.returncode == 1 and 'never overwritten' in again.stderr


def test_train_and_validate_read_neither_the_test_partition_nor_a_recording(root, tmp_path, small_dataset, monkeypatch):
    """
    Description:
        The leakage test of SPEC §2.13.7-13.8: with the test-partition selector and every recording reader of
        scripts/catch.py replaced by functions that raise, 'train' and 'validate' both succeed (exit 0).
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    import argparse
    config = _Training_Config(root, tmp_path, [small_dataset])
    assert module._Train(argparse.Namespace(config=str(config), out=None)) == 0
    assert module._Evaluate(argparse.Namespace(partition='validation', run=str(tmp_path / 'models' / 'Run'), out=str(tmp_path / 'evaluations'))) == 0


def test_test_refuses_a_dirty_run_and_appends_nothing(root, tmp_path, small_dataset, monkeypatch, capsys):
    """
    Description:
        SPEC §2.13.9: 'test' refuses a run made from a tree with uncommitted changes (exit 1, naming git_dirty);
        neither the ledger nor an evaluation folder is written.
    """

    run, out = _Trained_Run(root, tmp_path, [small_dataset], dirty=True), tmp_path / 'evaluations'
    result = _Run_Guarded(root, tmp_path, monkeypatch, capsys, 'evaluate', '--partition', 'test', '--run', str(run), '--out', str(out))
    assert result.returncode == 1 and 'uncommitted changes' in result.stderr and not out.exists()


def test_test_refuses_a_dataset_whose_hash_differs_and_appends_nothing(root, tmp_path, small_dataset, monkeypatch, capsys):
    """
    Description:
        SPEC §2.13.9: a dataset changed after its run was made (one byte of flights.npz) is refused (exit 1, both
        hashes named); neither the ledger nor an evaluation folder is written.
    """

    import shutil
    copy = tmp_path / 'Copy'
    shutil.copytree(str(small_dataset[0]), str(copy))
    run, out = _Trained_Run(root, tmp_path, [(copy, small_dataset[1])]), tmp_path / 'evaluations'
    data = bytearray((copy / 'flights.npz').read_bytes())
    data[-100] ^= 0xFF
    (copy / 'flights.npz').write_bytes(bytes(data))
    result = _Run_Guarded(root, tmp_path, monkeypatch, capsys, 'evaluate', '--partition', 'test', '--run', str(run), '--out', str(out))
    assert result.returncode == 1 and small_dataset[1] in result.stderr and hashlib.sha256(bytes(data)).hexdigest() in result.stderr
    assert not out.exists()


def test_test_refuses_an_existing_folder_and_appends_nothing(root, tmp_path, small_dataset, monkeypatch, capsys):
    """
    Description:
        An existing <out>/<Run.Name>/test/ is refused (exit 1) before the ledger is touched.
    """

    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'evaluations'
    (out / 'Run' / 'test').mkdir(parents=True)
    result = _Run_Guarded(root, tmp_path, monkeypatch, capsys, 'evaluate', '--partition', 'test', '--run', str(run), '--out', str(out))
    assert result.returncode == 1 and 'never overwritten' in result.stderr and not (out / 'test_ledger.csv').exists()


def test_test_refuses_a_run_whose_configuration_copy_differs_from_its_manifest_and_appends_nothing(root, tmp_path, small_dataset,
                                                                                                    monkeypatch, capsys):
    """
    Description:
        A run whose training.yaml was changed after 'train' (one comment line appended) no longer matches its
        manifest's hash and is refused by 'test' (exit 1, naming the copy) before the ledger; run in process with the
        test partition and every recording reader guarded, so a refusal removed cannot reach a recording (each refusal
        before a counted read fails a test when removed).
    """

    import argparse
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(module, '_Real_Flights', lambda: pytest.fail('a real flight was read'))
    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'evaluations'
    (run / 'training.yaml').write_bytes((run / 'training.yaml').read_bytes() + b'# edited\n')
    assert module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(out))) == 1
    assert 'training.yaml differs from its manifest' in capsys.readouterr().err and not out.exists()


def test_test_scores_the_test_partition_and_real_flights_counted_in_the_ledger(root, tmp_path, small_dataset, segmented, monkeypatch):
    """
    Description:
        'test' end to end (SPEC §2.13.9, §2.14, §9.7), in-process with the real validation flights replaced by three
        identification flights (the per-commit tier reads no validation flight): the ledger gets row 1 before the
        results; per_flight.csv has the 4 test flights (codes 26-29) and the 3 real flights, the real rows with
        empty catch fields; a real prefix is the flight with its first 40 ms trimmed, in the robot frame by the assumed
        transform (its prediction equals Predict on that prefix bit for bit); coverage.csv lists the 3; the manifest
        names evaluation 1. A second call is refused and the ledger keeps 1 row.
    """

    import argparse
    module = _Load_Script(root)
    stand_ins = [flight for set_name, flight, _ in segmented if set_name == 'Identification' and flight is not None][:3]
    monkeypatch.setattr(module, '_Real_Flights', lambda: stand_ins)
    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'evaluations'
    assert module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(out))) == 0
    folder = out / 'Run' / 'test'
    assert sorted(os.listdir(folder)) == ['coverage.csv', 'manifest.json', 'per_flight.csv', 'per_horizon.csv', 'predictions.npz', 'summary.csv']
    with open(folder / 'per_flight.csv', encoding='ascii', newline='') as file:
        rows = list(csv.DictReader(file))
    assert {r['flight'] for r in rows if r['partition'] == 'test'} == {'26', '27', '28', '29'}
    real = [r for r in rows if r['partition'] == 'real']
    assert {r['flight'] for r in real} == {'%s/Experiment_%d.txt' % (f.Folder, f.Experiment) for f in stand_ins}
    assert all((r['catch_point_error_mm'], r['time_to_catch_error_ms'], r['crossing']) == ('', '', '') for r in real)
    assert len((folder / 'coverage.csv').read_text().splitlines()) == 4
    assert json.loads((folder / 'manifest.json').read_text())['test_evaluation'] == 1
    from Recording.Identification import Trim_Flight
    from Recording.Core import Transform_To_Robot
    from Configuration.Core import Load_Configuration
    from Evaluation.Report import Read_Predictions
    from Prediction.Core import Predict
    from Observation.Core import Observation_Model_Str
    from Flight.Core import Flight_Model_Str
    configuration = Load_Configuration(os.path.join(root, 'Config', 'ball_catching.yaml'))
    o = configuration.Observation
    trimmed = Trim_Flight(stand_ins[0], 0.040)
    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag)
    m4 = Observation_Model_Str(White_Std=np.array(o.White_Std), White_Dof=o.White_Dof, Drift_Std=np.array(o.Drift_Std), Drift_Time=o.Drift_Time,
                               Offset_Std=np.array(o.Offset_Std), Offset_Time=o.Offset_Time, Offset_Dof=o.Offset_Dof)
    position = Transform_To_Robot(trimmed.Position[:10], Transform_Str=configuration.Transform)
    expected = Predict('GLS_M4', trimmed.Time[:10], position, stand_ins[0].Rate, Flight_Model_Str=model, Observation_Model_Str=m4)
    record = [r for r in Read_Predictions(str(folder / 'predictions.npz')) if r.Partition == 'real' and r.N == 10][0]
    assert np.array_equal(record.X_First, expected.Grid_State[0]) and record.T_0 == expected.T_0
    with open(out / 'test_ledger.csv', encoding='ascii', newline='') as file:
        ledger = list(csv.DictReader(file))
    assert [(r['evaluation'], r['run'], r['model'], r['dataset_sha256']) for r in ledger] == [('1', 'Run', 'GLS_M4', small_dataset[1])]
    assert module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(out))) == 1
    assert len((out / 'test_ledger.csv').read_text().splitlines()) == 2


def test_the_leakage_guard_replaces_every_reader_of_the_recordings_in_both_scripts(root, monkeypatch):
    """
    Description:
        The guard itself (the guards intact): of the functions that read the recordings
        (Recording.Core.Load_Recordings, Recording.Segmentation.Segment_Flights, Dataset.Core.Load_Donors and
        Identification_Flights), scripts/catch.py imports the first three and scripts/Maintenance/derive.py the first two
        and the last, written out here and read from each script's imports; with _Forbid_Leaks every one of them, called
        through its script, raises the guard's error, and so do the real validation flights of the user script, in the
        module a test loads and in the one the derivations' script loaded. A reader dropped from the guard's list, or
        imported under a name the list does not hold, fails here.
    """

    import ast
    readers = {'Load_Recordings', 'Segment_Flights', 'Load_Donors', 'Identification_Flights'}
    expected = {'catch.py': {'Load_Recordings', 'Segment_Flights', 'Load_Donors'},
                'derive.py': {'Load_Recordings', 'Segment_Flights', 'Identification_Flights'}}
    catch, derive = _Load_Script(root), _Load_Derive(root)
    for module in (catch, derive):
        name = os.path.basename(module.__file__)
        tree = ast.parse(open(module.__file__, encoding='utf-8').read())
        imported = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert imported & readers == expected[name], name
    _Forbid_Leaks(catch, monkeypatch)
    _Forbid_Leaks(derive, monkeypatch)
    for module in (catch, derive, derive.CONST_CATCH_SCRIPT):
        for reader in expected[os.path.basename(module.__file__)]:
            with pytest.raises(AssertionError, match='a recording was read'):
                getattr(module, reader)()
    for module in (catch, derive.CONST_CATCH_SCRIPT):
        with pytest.raises(AssertionError, match='a recording was read'):
            module._Real_Flights()


def test_the_leakage_guard_catches_a_read_of_the_test_partition(root, tmp_path, small_dataset, monkeypatch):
    """
    Description:
        The forced failure of the leakage test: with the same guards, 'test' (which must read the test partition)
        is stopped by the selector's guard, so the guard that 'train' and 'validate' pass is live.
    """

    import argparse
    module = _Load_Script(root)
    run = _Trained_Run(root, tmp_path, [small_dataset])
    _Forbid_Leaks(module, monkeypatch)
    with pytest.raises(AssertionError, match='test partition'):
        module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(tmp_path / 'evaluations')))


def test_test_lists_a_refused_real_flight_and_goes_on(root, tmp_path, small_dataset, segmented, monkeypatch, capsys):
    """
    Description:
        A real flight the predictor refuses (a stand-in identification flight moved 3 m down, so its
        prefix lies below the floor) does not stop 'test' (exit 0): it is listed by name with its reason (information,
        not a bound), its per_flight rows have the crossing 'refused' and the reason, and the other stand-in is scored.
    """

    import argparse
    from dataclasses import replace
    module = _Load_Script(root)
    flights = [flight for set_name, flight, _ in segmented if set_name == 'Identification' and flight is not None][:2]
    sunk = replace(flights[0], Position=flights[0].Position - np.array([0.0, 0.0, 3.0]))
    monkeypatch.setattr(module, '_Real_Flights', lambda: [sunk, flights[1]])
    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'evaluations'
    assert module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(out))) == 0
    name = '%s/Experiment_%d.txt' % (sunk.Folder, sunk.Experiment)
    printed = capsys.readouterr().out
    assert 'refused (information, not a bound): %s N=10: The predicted trajectory is below the floor height' % name in printed
    with open(out / 'Run' / 'test' / 'per_flight.csv', encoding='ascii', newline='') as file:
        real = [r for r in csv.DictReader(file) if r['partition'] == 'real']
    assert {r['crossing'] for r in real if r['flight'] == name} == {'refused'} and all(r['refusal'] for r in real if r['flight'] == name)
    assert all(r['crossing'] == '' and r['remainder_rms_mm'] for r in real if r['flight'] != name) and len(real) == 4


def _Real_Digest(folder):
    """
    Description:
        The real-flight predictions of a 'test' evaluation as CONST_PINNED_PREDICTIONS holds them: the SHA-256, the
        rows per N, the median remainder RMS per N [mm]. The digest is taken over the columns the two counted
        evaluations wrote (CONST_F7_PER_FLIGHT, CONST_F7_PER_HORIZON), which still open each file in that order;
        the columns F8 added after them (the training seed, the velocity's errors) are empty on a real flight
        of a numerical predictor, and that is asserted here. The pinned values are those of 2026-11-15, unchanged.
    """

    def Rows(name):
        with open(folder / name, encoding='ascii', newline='') as file:
            return list(csv.reader(file))

    per_flight, per_horizon = Rows('per_flight.csv'), Rows('per_horizon.csv')
    assert tuple(per_flight[0][:len(CONST_F7_PER_FLIGHT)]) == CONST_F7_PER_FLIGHT
    assert tuple(per_horizon[0][:len(CONST_F7_PER_HORIZON)]) == CONST_F7_PER_HORIZON
    assert all(r[len(CONST_F7_PER_FLIGHT):] == [''] for r in per_flight[1:] if r[1] == 'real')
    assert all(r[len(CONST_F7_PER_HORIZON):] == ['', '', ''] for r in per_horizon[1:] if r[1] == 'real')
    k = per_flight[0].index('inference_ms')
    real = [r[:k] + [''] + r[k + 1:len(CONST_F7_PER_FLIGHT)] for r in per_flight[1:] if r[1] == 'real']
    horizon = [r[:len(CONST_F7_PER_HORIZON)] for r in per_horizon[1:] if r[1] == 'real']
    text = ('\n'.join(','.join(r) for r in real) + '\n#\n' + '\n'.join(','.join(r) for r in horizon)).encode() + b'\n#\n'
    counts = tuple(sum(1 for r in real if r[3] == n) for n in ('10', '20', '40'))
    medians = tuple(float('%.6f' % np.median([float(r[6]) for r in real if r[3] == n and r[6]])) for n in ('10', '20', '40'))
    return hashlib.sha256(text).hexdigest(), counts, medians


def _Coverage_Digest(folder):
    """
    Description:
        The coverage of a 'test' evaluation as CONST_F7_COVERAGE holds it: the SHA-256 of coverage.csv and the flights
        inside the support.
    """

    with open(folder / 'coverage.csv', encoding='ascii', newline='') as file:
        inside = sum(1 for r in list(csv.reader(file))[1:] if r[1] == 'true')
    return hashlib.sha256((folder / 'coverage.csv').read_bytes()).hexdigest(), inside


@pytest.mark.slow
def test_test_reproduces_the_pinned_real_flight_results(root, tmp_path):
    """
    Description:
        'test' on the 45 real validation flights (F7 Part B; SPEC §2.13.9, §2.14) for GLS_M4, run on a
        small dataset into a temporary folder with its own ledger. The prediction part: the real-flight rows of
        per_flight.csv (timing blanked) and per_horizon.csv reproduce the counted evaluation byte for byte
        (45 / 45 / 43 flights at N = 10 / 20 / 40); they do not depend on the dataset, only on its stored system
        configuration, the shipped one (the error model's bounds do not enter the predictors).

        The coverage part is the record of F7's configuration, superseded by Throw.Target_Area. 'test' computes the
        coverage of the flights it reads from its dataset's configuration,
        so this run's dataset has F7's targets (the radius of the target area 1.0 m, the half of Areas.Robot):
        coverage.csv then reproduces F7's byte for byte (34 of 45 inside), which shows that the validation
        flights are not evaluated against the new target area here. Their coverage with it is first computed
        by F14's counted 'test'.
    """

    config, datasets = _Dataset_Config(root, tmp_path, 30, target_radius='1.0'), tmp_path / 'datasets'
    assert _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(datasets)).returncode == 0
    small_dataset = (datasets / 'Small', hashlib.sha256((datasets / 'Small' / 'flights.npz').read_bytes()).hexdigest())
    for model, pinned in sorted(CONST_PINNED_PREDICTIONS.items()):
        run, out = _Trained_Run(root, tmp_path, [small_dataset], name=model, model=model, samples='[10, 20, 40]'), tmp_path / 'evaluations'
        result = subprocess.run([sys.executable, os.path.join(root, CONST_SCRIPT), 'evaluate', '--partition', 'test', '--run', str(run), '--out',
                                 str(out)], cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        assert _Real_Digest(out / model / 'test') == pinned
        assert _Coverage_Digest(out / model / 'test') == CONST_F7_COVERAGE


def test_train_trains_one_model_per_seed_and_writes_its_files(root, small_dataset, learned_run):
    """
    Description:
        SPEC §2.13.7, §9.6 (F8): 'train' of a learned predictor writes the configuration's copy (byte for byte),
        manifest.json and one folder per seed of Run.Seeds with weights.npz, normalisation.json and epochs.csv
        (2 epochs). The manifest names the seeds, the device cpu with 1 thread, the torch version, the dataset
        trained on, 21 training and 5 validation flights, and per seed the SHA-256 of its weights file; the
        normalisation is the same for every seed, the weights are not. A second run of the name is refused.
    """

    run, config, tmp = learned_run
    assert sorted(os.listdir(run)) == ['manifest.json'] + ['seed_%d' % s for s in range(5)] + ['training.yaml']
    assert (run / 'training.yaml').read_bytes() == config.read_bytes()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['kind'], manifest['model'], manifest['seeds'], manifest['device'], manifest['threads']) == ('run', 'QT_BASIS', [0, 1, 2, 3, 4],
                                                                                                             'cpu', 1)
    assert manifest['torch'] == '2.15.0' and (manifest['training_flights'], manifest['validation_flights']) == (21, 5)
    assert manifest['training_datasets'] == [{'path': os.path.relpath(str(small_dataset[0]), root), 'sha256': small_dataset[1]}]
    weights = set()
    for seed in range(5):
        folder = run / ('seed_%d' % seed)
        assert sorted(os.listdir(folder)) == ['epochs.csv', 'normalisation.json', 'weights.npz']
        assert len((folder / 'epochs.csv').read_text().splitlines()) == 3
        assert (folder / 'normalisation.json').read_bytes() == (run / 'seed_0' / 'normalisation.json').read_bytes()
        digest = hashlib.sha256((folder / 'weights.npz').read_bytes()).hexdigest()
        assert manifest['trained'][str(seed)]['weights_sha256'] == digest and manifest['trained'][str(seed)]['epochs'] == 2
        weights.add(digest)
    assert len(weights) == 5
    again = _Run(root, tmp, 'train', '--config', str(config))
    assert again.returncode == 1 and 'is never overwritten' in again.stderr


def test_train_refuses_more_training_flights_than_the_train_partition_holds(root, tmp_path, small_dataset):
    """
    Description:
        Training.Flights is tested on both sides of the train partition's 21 flights (SPEC §8.3): 22 is refused
        (exit 1) naming both numbers, and no run folder is left; 21 trains (the fixture of this module).
    """

    result = _Run(root, tmp_path, 'train', '--config', str(_Learned_Config(root, tmp_path, small_dataset, flights=22)))
    assert result.returncode == 1 and 'Training.Flights is 22' in result.stderr and 'holds 21 flights' in result.stderr
    assert not (tmp_path / 'models' / 'Learned').exists() and os.listdir(tmp_path / 'models') == []


def test_validate_scores_every_seed_of_a_learned_run(root, tmp_path, learned_run):
    """
    Description:
        SPEC §2.13.8, §2.13.11 (F8): 'validate' of a learned run scores the 5 validation flights with every seed's
        model: per_flight.csv has the seeds 0-4 on the flights 21-25 for each N, each row with a crossing category
        or 'refused'; summary.csv has the spread rows over the seeds; predictions.npz is read back with its seeds;
        the report names each seed. Neither the training's runtime nor a recording is needed for it.
    """

    run, _, _ = learned_run
    out = tmp_path / 'evaluations'
    result = _Run(root, tmp_path, 'evaluate', '--run', str(run), '--out', str(out))
    assert result.returncode == 0, result.stderr
    folder = out / 'Learned' / 'validation'
    with open(folder / 'per_flight.csv', encoding='ascii', newline='') as file:
        rows = list(csv.DictReader(file))
    assert {r['seed'] for r in rows} == {'0', '1', '2', '3', '4'} and {r['flight'] for r in rows} == {'21', '22', '23', '24', '25'}
    assert len(rows) == 5 * 5 * 2 and all(r['crossing'] in ('both', 'disagree', 'passed', 'neither', 'refused') for r in rows)
    with open(folder / 'summary.csv', encoding='ascii', newline='') as file:
        summary = list(csv.DictReader(file))
    assert {'spread_min', 'spread_max', 'spread_sd'} <= {r['seed'] for r in summary if r['metric'] == 'refused_share'}
    from Evaluation.Report import Read_Predictions
    assert {r.Seed for r in Read_Predictions(str(folder / 'predictions.npz'))} <= {'0', '1', '2', '3', '4'}
    assert 'validation Small seed 4 N=20' in result.stdout and json.loads((folder / 'manifest.json').read_text())['seeds'] == [0, 1, 2, 3, 4]


def test_training_a_learned_predictor_reads_neither_the_test_partition_nor_a_recording(root, tmp_path, small_dataset, monkeypatch):
    """
    Description:
        The training leakage test of SPEC §2.13.7: with the test-partition selector and every recording reader of
        scripts/catch.py replaced by functions that raise, 'train' of the learned predictor and
        'validate' of its run both succeed (exit 0): the training reads the train partition and, for early
        stopping, the validation partition only.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    import argparse
    config = _Learned_Config(root, tmp_path, small_dataset)
    assert module._Train(argparse.Namespace(config=str(config), out=None)) == 0
    assert module._Evaluate(argparse.Namespace(partition='validation', run=str(tmp_path / 'models' / 'Learned'),
                                               out=str(tmp_path / 'evaluations'))) == 0


def test_the_leakage_guard_catches_a_training_that_selects_the_test_partition(root, tmp_path, small_dataset, monkeypatch):
    """
    Description:
        The forced failure of the training leakage test: a 'train' made to take its training flights from the
        test partition (the partition code it selects replaced by 2) is stopped by the selector's guard, and no
        run folder is left; so the guard that the training passes is live on the path the training takes.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(module, 'CONST_PARTITION_TRAIN', 2)
    import argparse
    with pytest.raises(AssertionError, match='test partition'):
        module._Train(argparse.Namespace(config=str(_Learned_Config(root, tmp_path, small_dataset)), out=None))
    assert os.listdir(tmp_path / 'models') == []


def test_the_normalisation_comes_from_the_train_partition_alone(root, tmp_path, small_dataset, monkeypatch):
    """
    Description:
        SPEC §7.8, §10 (leakage): the normalisation is fitted on the training flights only. A second training in
        which the validation partition is cut to its first 2 flights gives the same normalisation.json byte for
        byte, while its epochs.csv differs (the validation loss is taken on other flights): the validation
        flights reach the early stopping and nothing else.
    """

    module = _Load_Script(root)
    import argparse
    assert module._Train(argparse.Namespace(config=str(_Learned_Config(root, tmp_path, small_dataset, name='Whole')), out=None)) == 0
    selector = module.Partition_Indices
    monkeypatch.setattr(module, 'Partition_Indices', lambda data, code: selector(data, code)[:2] if code == 1 else selector(data, code))
    assert module._Train(argparse.Namespace(config=str(_Learned_Config(root, tmp_path, small_dataset, name='Cut')), out=None)) == 0
    whole, cut = tmp_path / 'models' / 'Whole' / 'seed_0', tmp_path / 'models' / 'Cut' / 'seed_0'
    assert (whole / 'normalisation.json').read_bytes() == (cut / 'normalisation.json').read_bytes()
    assert (whole / 'epochs.csv').read_bytes() != (cut / 'epochs.csv').read_bytes()
    assert json.loads((tmp_path / 'models' / 'Cut' / 'manifest.json').read_text())['validation_flights'] == 2


def test_test_refuses_a_run_trained_on_mps_and_appends_nothing(root, tmp_path, learned_run, monkeypatch, capsys):
    """
    Description:
        SPEC §7.8 (every reported number comes from the CPU): 'test' refuses a run whose
        manifest names the device mps (exit 1, naming the device) before the ledger gets a row and before any
        flight is read; the same run with the device cpu is not refused for its device (it reaches the reading of
        the test partition, which the guard of this test stops).
    """

    import argparse
    import shutil
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(module, '_Real_Flights', lambda: pytest.fail('a real flight was read'))
    run, out = tmp_path / 'Learned', tmp_path / 'evaluations'
    shutil.copytree(str(learned_run[0]), str(run))
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    manifest['git_dirty'], manifest['device'] = False, 'mps'
    (run / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    assert module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(out))) == 1
    assert 'was trained on the device mps' in capsys.readouterr().err and not (out / 'test_ledger.csv').exists()
    manifest['device'] = 'cpu'
    (run / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    with pytest.raises(AssertionError, match='test partition'):
        module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(out)))


@pytest.mark.slow
def test_a_training_run_is_reproduced_byte_for_byte_and_lowers_the_validation_loss(root, tmp_path):
    """
    Description:
        SPEC §2.13.7, §7.8 at a size beyond the per-commit tier (F8): on a 300-flight dataset (210 training
        flights), the query-time model with the basis decoder (a state of 32, 12 epochs, 5 seeds on the CPU) is
        trained twice by 'train' under two run names: every seed's weights.npz and epochs.csv are the same byte
        for byte, and on every seed the validation loss of the last epoch is below half of the first epoch's.
    """

    config, datasets = _Dataset_Config(root, tmp_path, 300), tmp_path / 'datasets'
    assert _Run(root, tmp_path, 'dataset', '--config', str(config), '--out', str(datasets)).returncode == 0
    dataset = (datasets / 'Small', hashlib.sha256((datasets / 'Small' / 'flights.npz').read_bytes()).hexdigest())
    for name in ('First', 'Second'):
        path = _Learned_Config(root, tmp_path, dataset, name=name, flights=210, epochs=12, hidden=32, samples='[10, 20, 40]')
        result = subprocess.run([sys.executable, os.path.join(root, CONST_SCRIPT), 'train', '--config', str(path)], cwd=str(tmp_path),
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
    for seed in range(5):
        first, second = tmp_path / 'models' / 'First' / ('seed_%d' % seed), tmp_path / 'models' / 'Second' / ('seed_%d' % seed)
        assert (first / 'weights.npz').read_bytes() == (second / 'weights.npz').read_bytes()
        assert (first / 'epochs.csv').read_bytes() == (second / 'epochs.csv').read_bytes()
        with open(first / 'epochs.csv', encoding='ascii', newline='') as file:
            epochs = list(csv.DictReader(file))
        assert len(epochs) == 12 and float(epochs[-1]['validation_loss_m']) < 0.5 * float(epochs[0]['validation_loss_m'])


@pytest.fixture(scope='module')
def stored_evaluations(root, tmp_path_factory, segmented):
    """
    Description:
        Two stored 'test' evaluations of GLS_M4 at N = 10, the runs GLS_M4 and GLS_M4_B, written once per module and
        worker, each on its own 80-flight dataset (12 test flights) with 12 identification flights standing in for the
        real validation flights (no validation flight is read): the folder that holds <run>/test and the test ledger.
        The dataset of GLS_M4_B has a system configuration whose M4 offset is 2 mm per axis, so its predictor weights
        the samples differently and the two runs' predictions differ (the removed LS was the second run before). It
        takes about 5 s per worker, so the four tests that use it are in the slow tier.
    """

    import argparse
    tmp = tmp_path_factory.mktemp('stored')
    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    text, count = re.subn(r'(?m)^  Offset_Std: .*$', '  Offset_Std: [0.002, 0.002, 0.002]', text)
    assert count == 1
    (tmp / 'system.yaml').write_text(text, encoding='utf-8')
    module = _Load_Script(root)
    module._Real_Flights = lambda: [flight for set_name, flight, _ in segmented if set_name == 'Identification' and flight is not None][:12]
    for name, system in (('GLS_M4_B', tmp / 'system.yaml'), ('GLS_M4', None)):
        (tmp / name).mkdir()
        config = _Dataset_Config(root, tmp / name, 80)
        if system is not None:
            config.write_text(config.read_text(encoding='utf-8').replace('  System_Configuration: Config/ball_catching.yaml\n',
                                                                         '  System_Configuration: %s\n' % os.path.relpath(str(system), root)),
                              encoding='utf-8')
        result = _Run(root, tmp, 'dataset', '--config', str(config), '--out', str(tmp / name / 'datasets'))
        assert result.returncode == 0, result.stderr
        folder = tmp / name / 'datasets' / 'Small'
        dataset = (folder, hashlib.sha256((folder / 'flights.npz').read_bytes()).hexdigest())
        run = _Trained_Run(root, tmp, [dataset], name=name, model='GLS_M4', samples='[10]')
        assert module._Evaluate(argparse.Namespace(partition='test', run=str(run), out=str(tmp / 'evaluations'))) == 0
    return tmp / 'evaluations'


@pytest.mark.slow
def test_compare_writes_its_tables_from_stored_evaluations_and_never_overwrites(root, tmp_path, stored_evaluations):
    """
    Description:
        'compare' end to end (SPEC §9.7; F14), in a fresh interpreter: from the two stored
        evaluations it writes summary.csv, paired.csv, sim_to_real.csv and manifest.json and prints the report. The
        paired median of GLS_M4 minus GLS_M4_B on the real flights is the one computed here from the two per_flight.csv;
        the supports hold the flights coverage.csv puts inside and outside; the simulated side of the sim-to-real
        rows is the test partition of the run's dataset; the manifest names the bootstrap and the inputs' hashes;
        the test ledger is unchanged (a comparison is not a test evaluation). A second call is refused.
    """

    gls, ls, out = stored_evaluations / 'GLS_M4' / 'test', stored_evaluations / 'GLS_M4_B' / 'test', tmp_path / 'comparisons'
    ledger = (stored_evaluations / 'test_ledger.csv').read_bytes()
    arguments = ('compare', '--evaluations', str(gls), str(ls), '--against', 'GLS_M4_B', '--name', 'Check', '--out', str(out))
    result = _Run(root, tmp_path, *arguments)
    assert result.returncode == 0, result.stderr
    folder = out / 'Check'
    assert sorted(os.listdir(folder)) == ['manifest.json', 'paired.csv', 'sim_to_real.csv', 'summary.csv']
    real = lambda path: {r['flight']: float(r['remainder_rms_mm']) for r in _Stored_Rows(path / 'per_flight.csv')
                         if r['partition'] == 'real' and r['n'] == '10'}
    a, b = real(gls), real(ls)
    expected = float(np.median([a[flight] - b[flight] for flight in a]))
    paired = _Stored_Rows(folder / 'paired.csv')
    assert {(r['run'], r['against']) for r in paired} == {('GLS_M4', 'GLS_M4_B')}
    row = [r for r in paired if (r['partition'], r['support'], r['n']) == ('real', 'all', '10')][0]
    assert (row['flights'], row['left_out']) == ('12', '0') and float(row['median_mm']) == pytest.approx(expected, abs=1e-6)
    assert float(row['low_mm']) <= expected <= float(row['high_mm'])
    inside = sum(1 for r in _Stored_Rows(ls / 'coverage.csv') if r['inside'] == 'true')
    summary = _Stored_Rows(folder / 'summary.csv')
    counts = {r['support']: int(r['flights']) for r in summary
              if (r['run'], r['partition'], r['n'], r['metric']) == ('GLS_M4_B', 'real', '10', 'refused')}
    assert counts == {'all': 12, 'inside': inside, 'outside': 12 - inside}
    assert {r['support'] for r in summary if r['partition'] == 'test'} == {'all'}
    median = [r for r in summary if (r['run'], r['partition'], r['support'], r['n'], r['metric']) == ('GLS_M4_B', 'real', 'all', '10',
                                                                                                    'remainder_rms_median_mm')][0]
    assert float(median['value']) == pytest.approx(float(np.median(list(b.values()))), abs=1e-6)
    sim_to_real = _Stored_Rows(folder / 'sim_to_real.csv')
    assert [(r['run'], r['n'], r['dataset'], r['real_flights']) for r in sim_to_real] == [('GLS_M4', '10', 'Small', '12'),
                                                                                         ('GLS_M4_B', '10', 'Small', '12')]
    assert all(10 <= int(r['simulated_flights']) <= 12 for r in sim_to_real)
    assert 'paired GLS_M4 - GLS_M4_B, real (real) all N=10: ' in result.stdout and 'sim-to-real GLS_M4_B N=10: real ' in result.stdout
    assert 'coverage (SPEC §2.13.6): %d of 12 real validation flights inside' % inside in result.stdout
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['kind'], manifest['runs'], manifest['against'], manifest['name']) == ('comparison', ['GLS_M4', 'GLS_M4_B'], ['GLS_M4_B'],
                                                                                             'Check')
    assert (manifest['bootstrap_seed'], manifest['bootstrap_resamples'], manifest['real_flights'], manifest['coverage_inside']) == (
        20261118, 10000, 12, inside)
    assert manifest['sha256']['GLS_M4_B/manifest.json'] == hashlib.sha256((ls / 'manifest.json').read_bytes()).hexdigest()
    assert manifest['sha256']['paired.csv'] == hashlib.sha256((folder / 'paired.csv').read_bytes()).hexdigest()
    assert (stored_evaluations / 'test_ledger.csv').read_bytes() == ledger
    again = _Run(root, tmp_path, *arguments)
    assert again.returncode == 1 and 'is never overwritten' in again.stderr


@pytest.mark.slow
def test_compare_reads_no_recording_no_dataset_and_no_run(root, tmp_path, stored_evaluations, monkeypatch):
    """
    Description:
        The condition of F14: 'compare' reads stored evaluation files only. With every reader
        of a recording, of a dataset, of a run and of its weights, the real flights and the test ledger's writer
        replaced in scripts/catch.py by functions that raise, it still succeeds.
    """

    import argparse
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)

    def Refuse(*args, **kwargs):
        raise AssertionError('compare read something else than a stored evaluation')

    for name in ('_Real_Flights', 'Read_Dataset', '_Load_Run', '_Run_Datasets', 'Run_Predictor', 'Append_Test_Ledger', 'Load_Configuration'):
        monkeypatch.setattr(module, name, Refuse)
    arguments = argparse.Namespace(evaluations=[str(stored_evaluations / 'GLS_M4' / 'test'), str(stored_evaluations / 'GLS_M4_B' / 'test')],
                                   against=['GLS_M4_B'], name='Leak', out=str(tmp_path))
    assert module._Compare(arguments) == 0
    assert (tmp_path / 'Leak' / 'paired.csv').is_file()


@pytest.mark.slow
def test_compare_refuses_what_it_cannot_compare_and_writes_nothing(root, tmp_path, stored_evaluations):
    """
    Description:
        Each refusal of 'compare' exits 1, names the problem and writes nothing: an evaluation file changed after
        its evaluation (its SHA-256 no longer the manifest's), a reference that is not among the runs, a run given
        twice, a name that is not a folder name, a run folder in place of an evaluation, and a folder without a
        manifest.
    """

    import shutil
    gls, ls, out = stored_evaluations / 'GLS_M4' / 'test', stored_evaluations / 'GLS_M4_B' / 'test', tmp_path / 'comparisons'
    changed = tmp_path / 'changed' / 'GLS_M4_B' / 'test'
    shutil.copytree(str(ls), str(changed))
    with open(changed / 'per_flight.csv', 'a', encoding='ascii') as file:
        file.write('99,test,Small,10,0.1,100,1.000000,0.000000,,,neither,1.000000,,\n')
    (tmp_path / 'empty').mkdir()
    cases = [((str(gls), str(changed)), ('GLS_M4_B',), 'Check', 'per_flight.csv of the evaluation'),
             ((str(gls), str(ls)), ('Unknown',), 'Check', '--against names Unknown, which is not among the evaluations given'),
             ((str(ls), str(ls)), (), 'Check', 'The run GLS_M4_B is given twice'),
             ((str(gls), str(ls)), ('GLS_M4_B',), 'a/b', 'letters, digits, _ and - only'),
             ((str(stored_evaluations.parent / 'models' / 'GLS_M4_B'),), (), 'Check', 'holds the manifest of a run, not of an evaluation'),
             ((str(tmp_path / 'empty'),), (), 'Check', 'is not an evaluation folder')]
    for evaluations, against, name, message in cases:
        arguments = ['compare', '--evaluations', *evaluations, '--name', name, '--out', str(out)] + (['--against', *against] if against else [])
        result = _Run(root, tmp_path, *arguments)
        assert result.returncode == 1 and message in result.stderr, (message, result.stderr)
        assert not out.exists()


@pytest.mark.slow
def test_compare_refuses_evaluations_whose_coverage_differs(root, tmp_path, stored_evaluations):
    """
    Description:
        The support split needs one support (SPEC §2.13.6): two evaluations whose coverage.csv put a real flight
        on different sides are refused, naming both runs, and nothing is written.
    """

    import shutil
    gls, other, out = stored_evaluations / 'GLS_M4' / 'test', tmp_path / 'Other' / 'test', tmp_path / 'comparisons'
    shutil.copytree(str(stored_evaluations / 'GLS_M4_B' / 'test'), str(other))
    lines = (other / 'coverage.csv').read_text(encoding='ascii').splitlines()
    flight, inside, reason = lines[1].split(',', 2)
    lines[1] = ','.join([flight, 'false' if inside == 'true' else 'true', '' if inside != 'true' else 'moved for the test'])
    (other / 'coverage.csv').write_text('\n'.join(lines) + '\n', encoding='ascii')
    manifest = json.loads((other / 'manifest.json').read_text(encoding='utf-8'))
    manifest['run_name'], manifest['sha256']['coverage.csv'] = 'Other', hashlib.sha256((other / 'coverage.csv').read_bytes()).hexdigest()
    (other / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    result = _Run(root, tmp_path, 'compare', '--evaluations', str(gls), str(other), '--name', 'Check', '--out', str(out))
    assert result.returncode == 1 and 'The coverage of the real flights differs between GLS_M4 and Other' in result.stderr
    assert not out.exists()


def test_compare_refuses_an_evaluation_of_a_removed_predictor_before_its_files(root, tmp_path, capsys):
    """
    Description:
        The program has two predictors: a stored evaluation of LS, QT_MLP or HYBRID, given by its
        path, is refused by 'compare' from its manifest alone, naming the predictor and the two that remain; the
        folder holds no other file, so the refusal comes before any of them is read; nothing is written. An
        evaluation of one of the two passes this check and is refused later for its missing file.
    """

    import argparse
    module = _Load_Script(root)
    for model in ('LS', 'QT_MLP', 'HYBRID', 'GLS_M4'):
        folder = tmp_path / model / 'validation'
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text(json.dumps({'kind': 'evaluation', 'model': model, 'run_name': model, 'partition': 'validation',
                                                          'datasets': [{'path': 'Data/A', 'sha256': '0' * 64}], 'sha256': {}}), encoding='utf-8')
        assert module._Compare(argparse.Namespace(evaluations=[str(folder)], against=[], name='Check', out=str(tmp_path / 'out'))) == 1
        error = capsys.readouterr().err
        if model == 'GLS_M4':
            assert 'has no per_flight.csv' in error and 'is of the predictor' not in error
        else:
            assert 'is of the predictor %s; the program compares evaluations of GLS_M4 and QT_BASIS only' % model in error
        assert not (tmp_path / 'out').exists()


def _Decision_Files(module, run, config, folder, partition='validation', margins=None):
    """
    Description:
        The decision stage of 'catch' alone, prepared as the command prepares it (the configuration, the run's copy, the
        decision's datasets, the scene's context, the reach of the area's map), its rows written by the decision's own
        writers into `folder`; the stage's manifest fields are returned. The numerical predictor decides at N = 20 and 40
        in seconds, where the whole chain predicts at every sample (about 150 ms each).
    """

    import time
    from Scene.Core import Build_Scene, Scene_Robot
    from Catch.Core import Catch_Context
    from Catch.Decision import Write_Per_Throw, Write_Summary, Write_Margins
    configuration = module.Load_Configuration(str(config))
    training, _, _ = module._Load_Run_Copy(str(run))
    datasets = module._Decision_Datasets(configuration)
    model = Build_Scene(Areas_Str=configuration.Areas)
    context = Catch_Context(model, Scene_Robot(model), Criteria_Str=module._Catch_Criteria(configuration))
    setup, _, _ = module._Interception_Setup(configuration, training, str(run), context)
    per_throw, summary, margin_rows, fields = module._Decision_Stage(configuration, training, str(run), partition, datasets, context,
                                                                     setup.Grids, margins or {}, time.perf_counter())
    os.makedirs(str(folder))
    Write_Per_Throw(os.path.join(str(folder), 'per_throw.csv'), per_throw)
    Write_Summary(os.path.join(str(folder), 'summary.csv'), summary)
    if margin_rows:
        Write_Margins(os.path.join(str(folder), 'margins.csv'), margin_rows)
    return fields


def test_the_decision_stage_gives_its_files_from_the_validation_partition_only(root, tmp_path, small_dataset, catch_dataset, monkeypatch):
    """
    Description:
        F9 (SPEC §7.11, §9.12), the first stage of 'catch': with the test-partition selector and every recording reader
        replaced by functions that raise, the decision on the validation partition (GLS_M4, stages 20 and 40, both
        accelerations) gives per_throw.csv (27 throws x 2 accelerations), summary.csv (2 rows) and margins.csv (2 stages
        x 40 bins), and the throws and the truth samples its manifest names (the manifest itself: the slow test of the
        whole chain).
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    run = _Trained_Run(root, tmp_path, [small_dataset])
    config = _Decision_Config(root, tmp_path, catch_dataset)
    folder = tmp_path / 'decisions' / 'Run' / 'validation'
    fields = _Decision_Files(module, run, config, folder)
    per_throw = _Stored_Rows(folder / 'per_throw.csv')
    summary = _Stored_Rows(folder / 'summary.csv')
    margins = _Stored_Rows(folder / 'margins.csv')
    assert len(per_throw) == 54 and {row['acceleration_deg_s2'] for row in per_throw} == {'800', '400'}
    assert {row['outcome'] for row in per_throw} <= {'true_go', 'false_go', 'true_no_go', 'false_no_go'}
    assert [(row['dataset'], row['acceleration_deg_s2'], row['throws']) for row in summary] == [('Catch_Small', '800', '27'),
                                                                                                 ('Catch_Small', '400', '27')]
    assert len(margins) == 80 and {row['n'] for row in margins} == {'20', '40'}
    assert fields['throws'] == [27] and fields['seeds'] == ['']
    # The truth area (F9 Q1 amended): the 0.1 m cube grown by one 0.05 m step; its samples are the ones the decision counted.
    sys.path.insert(0, os.path.join(root, 'src'))
    from Dataset.Core import Read_Dataset, Partition_Indices
    from Catch.Decision import Truth_Samples
    data = Read_Dataset(str(catch_dataset[0] / 'flights.npz'))
    indices = Partition_Indices(data, 1)
    model = module._Flight_Model(module.Load_Configuration(str(config)))
    grown = Truth_Samples(data.Release_State[indices], data.Floor_Time[indices], [0.7, -0.175, 0.9], [0.2, 0.2, 0.2], Flight_Model_Str=model)
    alone = Truth_Samples(data.Release_State[indices], data.Floor_Time[indices], [0.7, -0.175, 0.9], [0.1, 0.1, 0.1], Flight_Model_Str=model)
    assert fields['truth_samples'] == [len(grown[0])] and len(grown[0]) > len(alone[0])


def test_catch_refuses_a_dataset_whose_hash_or_models_differ_and_writes_nothing(root, tmp_path, small_dataset, catch_dataset):
    """
    Description:
        F9: catch never reads a dataset's stored configuration copies (KNOWN_ISSUES); it refuses a dataset whose
        flights.npz differs from Decision.Datasets' hash, and one generated with other models than the configuration
        given (another drag), each named, exit 1, nothing written.
    """

    run = _Trained_Run(root, tmp_path, [small_dataset])
    out = tmp_path / 'chain'
    wrong = _Run(root, tmp_path, 'catch', '--run', str(run), '--config', str(_Decision_Config(root, tmp_path, catch_dataset, digest='a' * 64)),
                 '--out', str(out))
    assert wrong.returncode == 1 and 'not Decision.Datasets\' %s' % ('a' * 64) in wrong.stderr
    other = _Run(root, tmp_path, 'catch', '--run', str(run), '--config', str(_Decision_Config(root, tmp_path, catch_dataset, drag=0.0095)),
                 '--out', str(out))
    assert other.returncode == 1 and 'was generated with the models' in other.stderr
    assert not os.path.exists(out)


def test_catch_refuses_an_area_with_a_point_that_does_not_qualify(root, tmp_path, small_dataset, catch_dataset):
    """
    Description:
        F9: every lattice point of Areas.Catch must qualify (F4's area); a 0.1 m cube moved beside the stand (centre
        (0.2, 0.0, 0.3) m) holds points that do not, and catch refuses, naming the count and the derivation that gives the
        area, exit 1, nothing written.
    """

    run = _Trained_Run(root, tmp_path, [small_dataset])
    config = _Decision_Config(root, tmp_path, catch_dataset, centre='[0.2, 0.0, 0.3]')
    result = _Run(root, tmp_path, 'catch', '--run', str(run), '--config', str(config), '--out', str(tmp_path / 'chain'))
    assert result.returncode == 1 and 'lattice points of Areas.Catch do not qualify' in result.stderr
    assert 'scripts/Maintenance/derive.py workspace' in result.stderr and not os.path.exists(tmp_path / 'chain')


def test_catch_refuses_no_worker_a_dataset_hash_and_a_run_on_mps_before_any_read(root, tmp_path, small_dataset, catch_dataset, monkeypatch,
                                                                                 capsys):
    """
    Description:
        F9, F10, F11 (SPEC §7.8, §9.12-9.14): 'catch' refuses, each with exit 1 and its message,
        before any dataset's data is read and with nothing written: --workers 0, a dataset whose flights.npz differs from
        Decision.Datasets' hash, and a run trained on the device mps (on either partition); the test-partition selector,
        every recording reader and the reader of the stored errors are guarded, and no ledger is started (each refusal
        before a counted read fails a test when removed).
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    _Forbid_Stored_Errors(module, monkeypatch)
    monkeypatch.setattr(module, 'Read_Dataset', lambda *args: pytest.fail('a dataset was read'))
    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'chain'
    config = _Decision_Config(root, tmp_path, catch_dataset)
    assert module._Catch(_Catch_Args(run, config, out, workers=0)) == 1
    assert '--workers must be an integer >= 1' in capsys.readouterr().err
    wrong = _Decision_Config(root, tmp_path, catch_dataset, digest='a' * 64)
    assert module._Catch(_Catch_Args(run, wrong, out)) == 1
    assert 'not Decision.Datasets\' %s' % ('a' * 64) in capsys.readouterr().err
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    (run / 'manifest.json').write_text(json.dumps(dict(manifest, device='mps'), sort_keys=True, indent=2) + '\n', encoding='utf-8')
    for partition in ('validation', 'test'):
        assert module._Catch(_Catch_Args(run, config, out, partition)) == 1
        assert 'was trained on the device mps' in capsys.readouterr().err, partition
    assert not os.path.exists(out)


def test_catch_refuses_an_existing_folder_of_any_stage_and_on_test_a_dirty_tree_and_a_missing_validation_before_its_ledger(
        root, tmp_path, small_dataset, catch_dataset, monkeypatch, capsys):
    """
    Description:
        The merged 'catch' refuses before anything is computed, read or counted, each
        with exit 1 and its message: an existing folder of any of its three stages under --out (Decisions/,
        Interceptions/, Catches/), naming it; and on the test partition a dirty tree, a run of a dirty tree and, with
        both committed, the validation result that is not stored, naming the stage and what to run. No dataset is read,
        the scene is not built (Build_Scene would fail the test), and no ledger exists afterwards: neither the catch's
        nor a decision's or an interception's of its own.
    """

    import Scene.Core
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(module, 'Read_Dataset', lambda *args: pytest.fail('a dataset was read'))
    monkeypatch.setattr(Scene.Core, 'Build_Scene', lambda *args, **kwargs: pytest.fail('the scene was built'))
    run, out = _Trained_Run(root, tmp_path, [small_dataset]), tmp_path / 'chain'
    config = _Decision_Config(root, tmp_path, catch_dataset)
    for stage in ('Decisions', 'Interceptions', 'Catches'):
        folder = out / stage / 'Run' / 'validation'
        folder.mkdir(parents=True)
        assert module._Catch(_Catch_Args(run, config, out)) == 1
        error = capsys.readouterr().err
        assert 'The folder %s exists; a decision, an interception and a catch are never overwritten' % folder in error
        folder.rmdir()
    monkeypatch.setattr(module, '_Git_State', lambda: ('f' * 40, True))
    assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1
    assert '(git_dirty: the tree True, the run False)' in capsys.readouterr().err
    monkeypatch.setattr(module, '_Git_State', lambda: ('f' * 40, False))
    committed = (run / 'manifest.json').read_bytes()
    (run / 'manifest.json').write_text(json.dumps(dict(json.loads(committed), git_dirty=True), sort_keys=True, indent=2) + '\n', encoding='utf-8')
    assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1
    assert '(git_dirty: the tree False, the run True)' in capsys.readouterr().err
    (run / 'manifest.json').write_bytes(committed)
    assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1
    error = capsys.readouterr().err
    assert 'the validation decision %s cannot be read' % (out / 'Decisions' / 'Run' / 'validation') in error
    assert 'Run catch on the validation partition first' in error
    assert not [name for _, _, names in os.walk(str(out)) for name in names]


def test_a_moving_catch_has_its_own_folders_and_planner_fields_and_takes_no_static_validation(root, tmp_path, small_dataset, catch_dataset,
                                                                                               monkeypatch, capsys):
    """
    Description:
        F15: with Interception.Catch_Mode static the three stages' folders are
        <parent>/<Run.Name>, as before; with moving they are <parent>/<Run.Name>/moving, under --out and under the
        configured folders alike, and the ledger is the same file for both. A moving catch's planner fields are the
        static ones and the mode, the stroke's scales and its compute budget; the static ones hold none of the three, so
        a stored static result is compared as before. Refused before anything is read or built, with exit 1: a moving
        run into an existing <...>/<Run.Name>/moving/validation, naming it; and a moving read of the test partition
        when only the static validation results are stored, naming the moving validation it needs. No ledger is
        written.
    """

    import Scene.Core
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    static_file = _Decision_Config(root, tmp_path, catch_dataset)
    moving_file = _Moving_Config(static_file, tmp_path)
    static, moving = module.Load_Configuration(str(static_file)), module.Load_Configuration(str(moving_file))
    out = tmp_path / 'chain'
    stages = {'decision': 'Decisions', 'interception': 'Interceptions', 'catch': 'Catches'}
    folders, ledger = module._Catch_Targets(static, 'Run', str(out))
    assert folders == {stage: str(out / folder / 'Run') for stage, folder in stages.items()} and ledger == str(out / 'catch_ledger.csv')
    assert module._Catch_Targets(moving, 'Run', str(out)) == ({stage: str(out / folder / 'Run' / 'moving') for stage, folder in stages.items()},
                                                             ledger)
    generated = os.path.join(root, 'Data', 'Generated')
    assert module._Catch_Targets(static, 'Run', None)[0] == {stage: os.path.join(generated, folder, 'Run') for stage, folder in stages.items()}
    assert module._Catch_Targets(moving, 'Run', None)[0] == {stage: os.path.join(generated, folder, 'Run', 'moving')
                                                            for stage, folder in stages.items()}
    assert module._Catch_Targets(moving, 'Run', None)[1] == module._Catch_Targets(static, 'Run', None)[1]
    datasets = module._Decision_Datasets(static)
    plain, stroked = module._Planner_Fields(static, b'run', datasets), module._Planner_Fields(moving, b'run', datasets)
    assert len(plain) == 12 and {key: stroked[key] for key in plain} == plain
    assert {key: stroked[key] for key in set(stroked) - set(plain)} == {'catch_mode': 'moving', 'stroke_scales': [1.0, 0.75, 0.5, 0.25],
                                                                        'stroke_compute_budget_s': 0.1}
    monkeypatch.setattr(module, 'Read_Dataset', lambda *args: pytest.fail('a dataset was read'))
    monkeypatch.setattr(Scene.Core, 'Build_Scene', lambda *args, **kwargs: pytest.fail('the scene was built'))
    run = _Trained_Run(root, tmp_path, [small_dataset])
    folder = out / 'Catches' / 'Run' / 'moving' / 'validation'
    folder.mkdir(parents=True)
    assert module._Catch(_Catch_Args(run, moving_file, out)) == 1
    assert 'The folder %s exists; a decision, an interception and a catch are never overwritten' % folder in capsys.readouterr().err
    folder.rmdir()
    monkeypatch.setattr(module, '_Git_State', lambda: ('f' * 40, False))
    for name in stages.values():
        (out / name / 'Run' / 'validation').mkdir(parents=True)
        (out / name / 'Run' / 'validation' / 'manifest.json').write_text('{}\n', encoding='utf-8')
    assert module._Catch(_Catch_Args(run, moving_file, out, 'test')) == 1
    error = capsys.readouterr().err
    assert 'the validation decision %s cannot be read' % (out / 'Decisions' / 'Run' / 'moving' / 'validation') in error
    assert not os.path.exists(out / 'catch_ledger.csv')


def test_r1_compares_the_decisions_columns_with_the_static_catchs_stored_lines(root, tmp_path):
    """
    Description:
        F15: R1 holds when the first 13 fields of every row of a moving catch's interception
        (the dataset to the catch time) are those of the lines of the static catch's stored per_throw.csv, in the same
        order, whatever the later fields hold (the static file has no moving columns, and its plan's values differ). It
        fails when one of those fields differs in one row (the decision, the catch time), when a row is missing and
        when the rows are in another order; without the stored file it is not applied (None).
    """

    from Catch.Interception import Write_Per_Throw, CONST_PER_THROW_COLUMNS, CONST_MOVING_COLUMNS
    module = _Load_Script(root)

    def Row(flight, decision, catch_time, plans):
        row = dict.fromkeys(CONST_PER_THROW_COLUMNS, '')
        row.update(dataset='Set', flight='%d' % flight, seed='0', acceleration_deg_s2='800', catchable='true', decision=decision,
                   go_n='20' if decision == 'go' else '', go_time_s=0.19 if decision == 'go' else '', outcome='true_go', stages_decision='go',
                   stages_go_n='20', stages_outcome='true_go', catch_time_s=catch_time, plans=plans)
        return tuple(row[name] for name in CONST_PER_THROW_COLUMNS)

    static = [Row(1, 'go', 1.27, 7), Row(2, 'no-go', '', ''), Row(3, 'go', 1.1500000000000001, 6)]
    Write_Per_Throw(str(tmp_path / 'per_throw.csv'), static)
    moving = [Row(1, 'go', 1.27, 5) + ('moving', 1.0, 1.07, 1.52, 1.3, 5.7, 0.6, 'true'), Row(2, 'no-go', '', '') + ('',) * len(CONST_MOVING_COLUMNS),
              Row(3, 'go', 1.1500000000000001, 4) + ('static', 0.0, '', '', 0.0, 7.1, '', '')]
    path = str(tmp_path / 'per_throw.csv')
    assert module._Decision_Columns_Equal(path, moving) is True
    other_decision = [moving[0], Row(2, 'go', 1.2, 3) + moving[1][len(CONST_PER_THROW_COLUMNS):], moving[2]]
    other_time = [moving[0], moving[1], Row(3, 'go', 1.15, 4) + moving[2][len(CONST_PER_THROW_COLUMNS):]]
    assert module._Decision_Columns_Equal(path, other_decision) is False and module._Decision_Columns_Equal(path, other_time) is False
    assert module._Decision_Columns_Equal(path, moving[:2]) is False and module._Decision_Columns_Equal(path, moving[::-1]) is False
    assert module._Decision_Columns_Equal(str(tmp_path / 'absent.csv'), moving) is None


def _Forbid_Recordings(module, monkeypatch):
    """
    Description:
        Replace every reader of the recordings in the script by a function that raises; the test partition stays
        readable (a counted read on generated data).
    """

    for name in ('Load_Recordings', 'Load_Donors', 'Segment_Flights'):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('a recording was read')))


@pytest.mark.slow
def test_catch_writes_the_files_of_its_three_stages_from_the_validation_partition_only_and_the_same_with_two_workers(
        root, tmp_path, stored_chain, catch_dataset, monkeypatch, capsys):
    """
    Description:
        The whole chain (F9, F10, F11; SPEC §7.11, §9.12-9.14): 'catch' on the
        validation partition (the module's learned run, 5 seeds, both accelerations; once per worker with one process,
        the fixture) writes under --out the three stages' folders. The decision: per_throw.csv (27 throws x 5 seeds x 2
        accelerations), summary.csv, margins.csv (5 seeds x 2 stages x 40 bins) and a manifest naming the partition, the
        stages, the quantile, the datasets and the truth area. The interception: per_throw.csv (270 rows, the outcomes
        F9's), summary.csv (10 rows), margins.csv (5 seeds x 12 stages x 40 bins) and a manifest naming the ladder, the
        calibration and the planner's values; every go with a plan has its motion certified, within the limits and at
        rest before a crossing (R2 of F10). The catch: per_throw.csv (the 270 rows with the nominal contact and the
        first seed's again with the one sensitivity level), summary.csv (12 rows) and a manifest with the contact, R2
        holding and the SHA-256 of the interception's rows and margins written beside it; every row has one of the nine
        classes, a no-go one of the two no-go classes and nothing simulated, a go retained exactly when it is caught,
        the robot touching nothing. No ledger. R1 as a property: the interception's rows that the
        one pass wrote equal, byte for byte, the rows Catch.Interception.Intercept_Chunk computes on its own with the
        stored margins. With --workers 2, the test-partition selector, every recording reader and the reader of the
        stored errors replaced by functions that raise, the eight data files are byte for byte the same; a second call
        into the same folder is refused.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    _Forbid_Stored_Errors(module, monkeypatch)
    run, config, out = stored_chain
    decision, interception, catch = (out / stage / 'Learned' / 'validation' for stage in ('Decisions', 'Interceptions', 'Catches'))
    # 1\ The decision's files.
    decided = _Stored_Rows(decision / 'per_throw.csv')
    assert len(decided) == 270 and {row['acceleration_deg_s2'] for row in decided} == {'800', '400'}
    assert {row['outcome'] for row in decided} <= {'true_go', 'false_go', 'true_no_go', 'false_no_go'}
    margins = _Stored_Rows(decision / 'margins.csv')
    assert len(margins) == 5 * 2 * 40 and {row['n'] for row in margins} == {'20', '40'}
    manifest = json.loads((decision / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['kind'], manifest['partition'], manifest['stages'], manifest['quantile']) == ('decision', 'validation', [20, 40], 0.95)
    assert manifest['datasets'] == [catch_dataset[1]] and manifest['throws'] == [27]
    assert manifest['truth_area'][0] == [0.7, -0.175, 0.9] and manifest['truth_area'][1] == pytest.approx([0.2, 0.2, 0.2], abs=1e-12)
    # 2\ The interception's files.
    per_throw, summary = _Stored_Rows(interception / 'per_throw.csv'), _Stored_Rows(interception / 'summary.csv')
    margins = _Stored_Rows(interception / 'margins.csv')
    assert len(per_throw) == 270 and {row['acceleration_deg_s2'] for row in per_throw} == {'800', '400'}
    assert {row['seed'] for row in per_throw} == {'0', '1', '2', '3', '4'}
    assert {row['outcome'] for row in per_throw} <= {'true_go', 'false_go', 'true_no_go', 'false_no_go'}
    assert [(row['dataset'], row['seed'], row['acceleration_deg_s2'], row['throws']) for row in summary] == \
        [('Catch_Small', seed, acceleration, '27') for seed in '01234' for acceleration in ('800', '400')]
    assert len(margins) == 5 * 12 * 40 and sorted({int(row['n']) for row in margins}) == [20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150]
    for row in per_throw:
        if row['decision'] == 'go' and row['plans'] not in ('', '0'):
            assert (row['certified'], row['within_limits']) == ('true', 'true')
            assert row['at_rest'] == 'true' or row['crossing_s'] == 'nan'
            assert float(row['clear_environment_m']) >= 0.05 and float(row['clear_self_m']) >= 0.05 and float(row['clear_exempt_m']) > 0.0
    manifest = json.loads((interception / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['kind'], manifest['partition'], manifest['quantile'], manifest['decision_stages']) == ('interception', 'validation', 0.95,
                                                                                                         [20, 40])
    assert manifest['stages'] == [20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150] and manifest['throws'] == [27]
    assert (manifest['latency_s'], manifest['compute_budget_s'], manifest['guard_s'], manifest['path_step_m']) == (0.004, 0.05, 0.03, 0.0025)
    assert manifest['calibration'] == [os.path.relpath(str(catch_dataset[0]), root)] and 'workers' not in manifest
    # 3\ The catch's files.
    rows, summary = _Stored_Rows(catch / 'per_throw.csv'), _Stored_Rows(catch / 'summary.csv')
    nominal = [row for row in rows if row['damping_ratio'] == '0.7']
    level = [row for row in rows if row['damping_ratio'] == '0.3']
    assert len(nominal) == 270 and len(level) == 54 and {row['seed'] for row in level} == {'0'} and len(summary) == 12
    assert [(row['dataset'], row['flight'], row['seed'], row['acceleration_deg_s2'], row['decision']) for row in nominal] == \
        [(row['dataset'], row['flight'], row['seed'], row['acceleration_deg_s2'], row['decision']) for row in per_throw]
    for row in rows:
        if row['decision'] == 'go':
            assert row['class'] in ('caught', 'bounced_out', 'rim', 'missed', 'hit_robot', 'hit_stand', 'late')
            assert (row['class'] == 'caught') == (row['retained'] == 'true') and row['robot_contact'] == ''
        else:
            assert row['class'] == ('no_go_catchable' if row['catchable'] == 'true' else 'no_go')
            assert (row['retained'], row['first_contact'], row['end_speed_m_s']) == ('false', '', '')
    assert sum(int(row['retained']) for row in summary) == sum(row['retained'] == 'true' for row in rows)
    manifest = json.loads((catch / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['kind'], manifest['partition'], manifest['step_s'], manifest['time_constant_s'], manifest['damping_ratio']) == \
        ('catch', 'validation', 0.00025, 0.004, 0.7)
    assert (manifest['settle_time_s'], manifest['friction'], manifest['levels']) == (0.6, [0.5, 0.005, 0.0001], [[0.004, 0.3]])
    assert manifest['r2_one_class_and_no_robot_contact'] is True and 'r1_interception_rows_identical' not in manifest
    assert manifest['sha256']['interception_per_throw'] == hashlib.sha256((interception / 'per_throw.csv').read_bytes()).hexdigest()
    assert manifest['sha256']['interception_margins'] == hashlib.sha256((interception / 'margins.csv').read_bytes()).hexdigest()
    assert not [name for name in os.listdir(str(out)) if name.endswith('.csv')] and 'workers' not in manifest
    # 4\ R1 as a property: the interception's rows of the one pass are those of Intercept_Chunk on its own.
    from Scene.Core import Build_Scene, Scene_Robot
    from Catch.Core import Catch_Context
    from Catch.Decision import Read_Margins
    from Catch.Interception import Intercept_Chunk, Write_Per_Throw
    from Dataset.Core import Read_Dataset, Partition_Indices
    configuration = module.Load_Configuration(str(config))
    training, _, _ = module._Load_Run_Copy(str(run))
    model = Build_Scene(Areas_Str=configuration.Areas)
    context = Catch_Context(model, Scene_Robot(model), Criteria_Str=module._Catch_Criteria(configuration))
    setup, seeds, _ = module._Interception_Setup(configuration, training, str(run), context)
    setup.Margins = Read_Margins(str(interception / 'margins.csv'))
    data_file = str(catch_dataset[0] / 'flights.npz')
    indices = Partition_Indices(Read_Dataset(data_file), 1)
    alone = [row for seed in seeds for k in range(0, len(indices), module.CONST_CHUNK_THROWS)
             for row in Intercept_Chunk((setup, 'Catch_Small', data_file, indices[k:k + module.CONST_CHUNK_THROWS], seed))]
    Write_Per_Throw(str(tmp_path / 'alone.csv'), alone)
    assert (tmp_path / 'alone.csv').read_bytes() == (interception / 'per_throw.csv').read_bytes()
    # 5\ A second call is refused; two workers, guarded, give the same eight files.
    assert module._Catch(_Catch_Args(run, config, out)) == 1
    capsys.readouterr()
    assert module._Catch(_Catch_Args(run, config, tmp_path / 'two', workers=2)) == 0
    assert 'R2 (every throw has one class; the robot touches nothing but the ball): holds.' in capsys.readouterr().err
    for stage, names in (('Decisions', ('per_throw.csv', 'summary.csv', 'margins.csv')),
                         ('Interceptions', ('per_throw.csv', 'summary.csv', 'margins.csv')), ('Catches', ('per_throw.csv', 'summary.csv'))):
        for name in names:
            assert (tmp_path / 'two' / stage / 'Learned' / 'validation' / name).read_bytes() == \
                (out / stage / 'Learned' / 'validation' / name).read_bytes(), (stage, name)


@pytest.mark.slow
def test_catch_test_needs_a_committed_tree_and_the_stored_validation_and_counts_every_call(root, tmp_path, stored_chain, catch_dataset,
                                                                                           monkeypatch, capsys):
    """
    Description:
        'catch --partition test' (one counted read of the whole chain):
        with the validation chain stored it refuses, each before it counts and with its own message, a dirty tree alone,
        a run of a dirty tree alone, a call with another system configuration than the validation's (another settle
        window: refused at the first stage's check, naming system_configuration), and a stored validation stage whose
        manifest differs from this call (the interception's ladder, the catch's settle window, each changed in the
        stored manifest and named with its stage and its field). With a committed
        tree and the stored validation it appends row 1 to the catch ledger (the test ledger's columns) and to no other
        ledger, decides, intercepts and catches the test partition with the validation's margins (no margins.csv of its
        own; the nominal contact only, no sensitivity level) and reads no recording; a second call (its folders exist)
        is refused, uncounted, and the stored validation margins are byte for byte unchanged.
    """

    import shutil
    module = _Load_Script(root)
    _Forbid_Recordings(module, monkeypatch)
    _Forbid_Stored_Errors(module, monkeypatch)
    run, config, stored = stored_chain
    out = tmp_path / 'chain'
    shutil.copytree(str(stored), str(out))
    ledger = out / 'catch_ledger.csv'
    kept = {stage: (out / stage / 'Learned' / 'validation' / 'margins.csv').read_bytes() for stage in ('Decisions', 'Interceptions')}
    monkeypatch.setattr(module, '_Git_State', lambda: ('f' * 40, True))
    assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1 and not os.path.exists(ledger)
    assert '(git_dirty: the tree True, the run False)' in capsys.readouterr().err
    monkeypatch.setattr(module, '_Git_State', lambda: ('f' * 40, False))
    committed = (run / 'manifest.json').read_bytes()
    (run / 'manifest.json').write_text(json.dumps(dict(json.loads(committed), git_dirty=True), sort_keys=True, indent=2) + '\n', encoding='utf-8')
    try:
        assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1 and not os.path.exists(ledger)
        assert '(git_dirty: the tree False, the run True)' in capsys.readouterr().err
    finally:
        (run / 'manifest.json').write_bytes(committed)
    text = open(str(config), encoding='utf-8').read()
    assert text.count('  Settle_Time: 0.6\n') == 1
    other = tmp_path / 'other_window.yaml'
    other.write_text(text.replace('  Settle_Time: 0.6\n', '  Settle_Time: 0.7\n'), encoding='utf-8')
    assert module._Catch(_Catch_Args(run, other, out, 'test')) == 1 and not os.path.exists(ledger)
    error = capsys.readouterr().err
    assert 'The stored validation decision' in error and 'differs from this call in system_configuration' in error
    for stage, folder, field, value in (('interception', 'Interceptions', 'stages', [20, 30, 40]), ('catch', 'Catches', 'settle_time_s', 0.7)):
        path = out / folder / 'Learned' / 'validation' / 'manifest.json'
        stored_manifest = path.read_bytes()
        path.write_text(json.dumps(dict(json.loads(stored_manifest), **{field: value}), sort_keys=True, indent=2) + '\n', encoding='utf-8')
        assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1 and not os.path.exists(ledger)
        error = capsys.readouterr().err
        assert 'The stored validation %s' % stage in error and 'differs from this call in %s;' % field in error, stage
        path.write_bytes(stored_manifest)
    assert module._Catch(_Catch_Args(run, config, out, 'test')) == 0
    rows = _Stored_Rows(ledger)
    assert [(row['evaluation'], row['run'], row['model'], row['dirty']) for row in rows] == [('1', 'Learned', 'QT_BASIS', 'false')]
    assert rows[0]['dataset_sha256'] == catch_dataset[1]
    assert sorted(name for name in os.listdir(str(out)) if name.endswith('.csv')) == ['catch_ledger.csv']
    decision, interception, catch = (out / stage / 'Learned' / 'test' for stage in ('Decisions', 'Interceptions', 'Catches'))
    assert len(_Stored_Rows(decision / 'per_throw.csv')) == 270 and not os.path.exists(decision / 'margins.csv')
    assert len(_Stored_Rows(interception / 'per_throw.csv')) == 270 and not os.path.exists(interception / 'margins.csv')
    caught = _Stored_Rows(catch / 'per_throw.csv')
    assert len(caught) == 270 and {row['damping_ratio'] for row in caught} == {'0.7'}
    for folder in (decision, interception, catch):
        assert json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))['partition'] == 'test'
    manifest = json.loads((catch / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['levels'], manifest['r2_one_class_and_no_robot_contact']) == ([], True)
    assert manifest['sha256']['interception_margins'] == hashlib.sha256(kept['Interceptions']).hexdigest()
    assert module._Catch(_Catch_Args(run, config, out, 'test')) == 1 and len(_Stored_Rows(ledger)) == 1
    assert all((out / stage / 'Learned' / 'validation' / 'margins.csv').read_bytes() == kept[stage] for stage in kept)


def _Go_Config(root, tmp_path, levels):
    """
    Description:
        The stored QT_BASIS run (5 seeds; the test is skipped in a clean clone, which has none) with a 30-flight catch
        dataset aimed into the shipped Areas.Catch (21 train, 6 validation and 3 test throws), generated into
        tmp_path/datasets, and tmp_path/system.yaml: the shipped system configuration with that dataset as the
        decision's and the calibration's, the static catch whatever mode is shipped, the sensitivity levels given, and
        the sweep at the scale 0 on the 6 validation throws. Returns the run's folder.
    """

    run = os.path.join(root, 'Data', 'Generated', 'Models', 'QT_BASIS')
    if not os.path.isdir(run):
        pytest.skip('Data/Generated/Models/QT_BASIS is absent: a clean clone holds no stored run (Data/Generated/ is never committed)')
    text = open(os.path.join(root, 'Config', 'dataset.yaml'), encoding='utf-8').read()
    text = text.replace('  Flights: 10000\n', '  Flights: 30\n').replace('  Name: Default\n', '  Name: Catch_Go\n')
    text = text.replace('  Validation: 0.15\n', '  Validation: 0.20\n').replace('  Test: 0.15\n', '  Test: 0.10\n')
    start, end = text.index('  Target_Area:\n'), text.index('\n# The observation model')
    (tmp_path / 'catch.yaml').write_text(text[:start] + '  Target_Catch_Area:\n    Margin: 0.0\n' + text[end:], encoding='utf-8')
    assert _Run(root, tmp_path, 'dataset', '--config', str(tmp_path / 'catch.yaml'), '--out', str(tmp_path / 'datasets')).returncode == 0
    folder = tmp_path / 'datasets' / 'Catch_Go'
    system = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    start, end = system.index('  Datasets:\n    - Path: Data/Generated/Datasets/Catch_In'), system.index('  # The datasets, among Datasets')
    relative = os.path.relpath(str(folder), root)
    system = system[:start] + '  Datasets:\n    - Path: %s\n      Hash: %s\n' % (relative, hashlib.sha256((folder / 'flights.npz').read_bytes())
                                                                             .hexdigest()) + system[end:]
    system = re.sub(r'  Calibration: \[.*\]\n', '  Calibration: [%s]\n' % relative, system)
    system, modes = re.subn(r'\n  Catch_Mode: (static|moving)\n', '\n  Catch_Mode: static\n', system)
    system, changed = re.subn(r'\n  Levels: \[\[.*\]\]\n', '\n  Levels: %s\n' % levels, system)
    assert (modes, changed) == (1, 1)
    for old, new in (('  Sweep_Scales: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]\n', '  Sweep_Scales: [0.0]\n'),
                     ('  Sweep_Throws: 1500\n', '  Sweep_Throws: 6\n')):
        assert system.count(old) == 1, old
        system = system.replace(old, new)
    (tmp_path / 'system.yaml').write_text(system, encoding='utf-8')
    return run


@pytest.mark.slow
def test_catch_with_the_stored_predictor_plans_goes_that_meet_r2_retains_them_and_a_bouncy_ball_leaves(root, tmp_path, monkeypatch):
    """
    Description:
        The chain end to end with goes (F10's R2; F11): the stored
        QT_BASIS run (5 seeds; skipped in a clean clone) on 6 validation throws of a 30-flight catch dataset aimed into
        the shipped Areas.Catch (the ladder calibrated on its 21 train throws), with the shipped contact and settle
        window and one sensitivity level (0.004 s, 0.3: a restitution of 0.45), the test partition, every recording and
        the stored errors guarded. The interception: some throws go and are planned; every planned go's motion is
        certified (>= 0.05 m to the environment and between the parts, the exempt pair apart), within the limits and at
        rest before its crossing; an intercepted go has its lateral miss <= 32.5 mm, its angle <= 10 deg and a margin
        >= 0; the tracked cup stays within 2 mm of the plan's at the crossing, no servo at a limit. The catch: R2 holds;
        at least 10 goes are simulated; with the nominal contact at least 80 % of them are retained (Part A: 97-98 %),
        each caught ball met the cup first and came to rest within the window; with the bouncy level at most 20 % of the
        first seed's goes are retained (Part A: 1.5-2.3 %); no robot contact, no servo at its limit, the joints within
        5 mrad of the plan. Then the sweep of scripts/Maintenance/derive.py on the same 6 throws from constructed stored
        errors (the scale 0): with the true trajectory as the prediction (scale 0) no go is false, at least 4 throws go
        and all but at most one are retained; the stored margins are unchanged.
    """

    run = _Go_Config(root, tmp_path, '[[0.004, 0.3]]')
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    with monkeypatch.context() as guarded:
        _Forbid_Stored_Errors(module, guarded)
        assert module._Catch(_Catch_Args(run, tmp_path / 'system.yaml', tmp_path / 'chain', workers=2)) == 0
    # 1\ The interception's goes (F10).
    rows = _Stored_Rows(tmp_path / 'chain' / 'Interceptions' / 'QT_BASIS' / 'validation' / 'per_throw.csv')
    planned = [row for row in rows if row['decision'] == 'go' and row['plans'] not in ('', '0')]
    assert len(rows) == 6 * 5 * 2 and len(planned) >= 10
    for row in planned:
        assert (row['certified'], row['within_limits'], row['at_rest']) == ('true', 'true', 'true')
        assert float(row['clear_environment_m']) >= 0.05 and float(row['clear_self_m']) >= 0.05 and float(row['clear_exempt_m']) > 0.0
        assert float(row['tracking_m']) < 0.002 and (row['force_limit_ticks'], row['control_limit_ticks']) == ('0', '0')
        if row['intercepted'] == 'true':
            assert float(row['lateral_m']) <= 0.0325 and float(row['angle_deg']) <= 10.0 and float(row['margin_s']) >= 0.0
    assert any(row['intercepted'] == 'true' for row in planned)
    # 2\ The catch of the goes (F11).
    target = tmp_path / 'chain' / 'Catches' / 'QT_BASIS' / 'validation'
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['r2_one_class_and_no_robot_contact'] is True
    rows = _Stored_Rows(target / 'per_throw.csv')
    nominal = [row for row in rows if row['decision'] == 'go' and row['damping_ratio'] == '0.7']
    bouncy = [row for row in rows if row['decision'] == 'go' and row['damping_ratio'] == '0.3']
    assert len(rows) == 6 * 2 * 5 + 6 * 2 and len(nominal) >= 10 and len(bouncy) >= 2 and {row['seed'] for row in bouncy} == {'0'}
    assert sum(row['retained'] == 'true' for row in nominal) >= 0.8 * len(nominal)
    assert sum(row['retained'] == 'true' for row in bouncy) <= 0.2 * len(bouncy)
    for row in nominal + bouncy:
        assert row['robot_contact'] == '' and row['force_limit_ticks'] == '0' and float(row['tracking_joint_rad']) < 0.005
        assert (row['class'] == 'caught') == (row['retained'] == 'true')
        if row['retained'] == 'true':
            assert row['first_contact'].startswith('cup_') and row['at_rest'] == 'true' and 0.0 < float(row['rest_s']) < 2.5
    # 3\ The sweep (scripts/Maintenance/derive.py) on the same throws.
    evaluations = tmp_path / 'evaluations'
    _Stored_Errors(evaluations / 'test', 'real', 12, lambda flight, n, index: (4.0 + 6.0 * index) * (1.0 + 0.1 * flight) * (1.0 if n == 20 else 0.8))
    _Stored_Errors(evaluations / 'validation', 'validation', 12,
                   lambda flight, n, index: (2.0 + 3.0 * index) * (1.0 + 0.1 * flight) * (1.0 if n == 20 else 0.8))
    margins = tmp_path / 'chain' / 'Interceptions' / 'QT_BASIS' / 'validation' / 'margins.csv'
    before = margins.read_bytes()
    derive = _Load_Derive(root)
    _Forbid_Leaks(derive, monkeypatch)
    assert derive._Sweep(_Sweep_Args(run, tmp_path / 'system.yaml', tmp_path / 'chain' / 'Catches', tmp_path / 'chain' / 'Interceptions',
                                     evaluations, workers=2)) == 0
    swept = _Stored_Rows(tmp_path / 'chain' / 'Catches' / 'QT_BASIS' / 'sweep' / 'per_throw.csv')
    exact = [row for row in swept if (row['error_source'], row['error_scale']) == ('real', '0.0')]
    assert len(exact) == 12 and not any(row['decision'] == 'go' and row['catchable'] == 'false' for row in exact)
    gone = [row for row in exact if row['decision'] == 'go']
    assert len(gone) >= 4 and sum(row['retained'] == 'true' for row in gone) >= len(gone) - 1
    assert all(row['first_contact'].startswith('cup_') for row in gone if row['retained'] == 'true') and margins.read_bytes() == before


@pytest.mark.slow
def test_a_moving_catch_with_the_stored_predictor_meets_r1_and_the_new_r2_and_holds_a_ball_the_static_catch_loses(root, tmp_path, monkeypatch,
                                                                                                                  capsys):
    """
    Description:
        F15 end to end (the stored QT_BASIS run on 6 validation throws of a 30-flight catch
        dataset, 5 seeds, one sensitivity level of restitution 0.29; skipped in a clean clone), the test partition, every
        recording and the stored errors guarded. The static catch first, into one folder; then the moving catch
        (Catch_Mode moving) into the same folder: its three stages are written under <Run.Name>/moving/validation and
        the static ones are byte for byte what they were. The decision's files are the static catch's. R1 holds (the
        manifest says so): the decision's columns of the 60 rows equal the static rows'. The interception's rows carry
        the eight further columns; at least 80 % of the planned goes have a stroke, and each of them is certified
        (>= 0.05 m), within the limits, at rest at the stroke's start and crossed inside the stroke (the new R2: no
        violation in the summary and the manifest), its scale one of the ladder's, the cup's speed along the ball
        between 0.1 and 2.6 m/s, the relative speed below the ball's 8.2 m/s and above 3.5 m/s, the cup's velocity
        within 10 deg of the ball's, the tracked cup within 3 mm of the plan's at the crossing, no servo at a limit.
        The catch: R2 holds, no robot contact, the joints within 5 mrad of the plan; with the nominal contact at least
        80 % of the goes are retained; at 0.29 the moving catch retains at least 80 % of the first seed's goes and the
        static catch at most 60 % of the same goes; the strokes' contact speeds lie between 3.5 and 8.2 m/s, their
        median below 7.0 m/s (the ball's own speed at the cup is 7.2 m/s at the median, Part A). The markers variant
        (Q12): the first seed's 12 rows of each contact once more with the markers, 24 rows after the others, the same
        throws and decisions, at 0.29 at least 80 % of the goes retained, no robot contact; the static run has none. Then
        the same moving catch beside static rows of which one decision is changed (a copy): R1 is violated, named, the
        exit code is 1, the files are written and the manifest records it. Then the sweep of
        scripts/Maintenance/derive.py in moving mode: it takes the stored moving interception and writes
        <Run.Name>/moving/sweep with the moving columns last.
    """

    import shutil
    run = _Go_Config(root, tmp_path, '[[0.004, 0.5]]')
    static_file, moving_file = tmp_path / 'system.yaml', _Moving_Config(tmp_path / 'system.yaml', tmp_path)
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    out = tmp_path / 'chain'
    with monkeypatch.context() as guarded:
        _Forbid_Stored_Errors(module, guarded)
        assert module._Catch(_Catch_Args(run, static_file, out, workers=2)) == 0
        names = [os.path.join(stage, 'QT_BASIS', 'validation', name) for stage in ('Decisions', 'Interceptions', 'Catches')
                 for name in ('per_throw.csv', 'summary.csv', 'manifest.json')]
        before = {name: (out / name).read_bytes() for name in names}
        capsys.readouterr()
        assert module._Catch(_Catch_Args(run, moving_file, out, workers=2)) == 0
        printed = capsys.readouterr().err
    assert 'R1 (the decision\'s columns equal the static catch\'s stored rows' in printed and '): holds.' in printed
    assert {name: (out / name).read_bytes() for name in names} == before
    static, moving = ({stage: out / stage / 'QT_BASIS' / tail for stage in ('Decisions', 'Interceptions', 'Catches')}
                      for tail in ('validation', os.path.join('moving', 'validation')))
    # 1\ The decision and R1.
    for name in ('per_throw.csv', 'summary.csv', 'margins.csv'):
        assert (moving['Decisions'] / name).read_bytes() == (static['Decisions'] / name).read_bytes()
    manifest = json.loads((moving['Interceptions'] / 'manifest.json').read_text(encoding='utf-8'))
    assert (manifest['catch_mode'], manifest['stroke_scales'], manifest['stroke_compute_budget_s']) == ('moving', [1.0, 0.75, 0.5, 0.25], 0.1)
    assert (manifest['r1_decision_columns_equal_static'], manifest['r2_violations']) == (True, 0)
    assert 'catch_mode' not in json.loads((static['Interceptions'] / 'manifest.json').read_text(encoding='utf-8'))
    rows, plain = _Stored_Rows(moving['Interceptions'] / 'per_throw.csv'), _Stored_Rows(static['Interceptions'] / 'per_throw.csv')
    decided = ('dataset', 'flight', 'seed', 'acceleration_deg_s2', 'catchable', 'decision', 'go_n', 'go_time_s', 'outcome', 'stages_decision',
               'stages_go_n', 'stages_outcome', 'catch_time_s')
    assert len(rows) == len(plain) == 60 and [[row[name] for name in decided] for row in rows] == [[row[name] for name in decided] for row in plain]
    assert list(rows[0])[-8:] == ['plan_mode', 'stroke_scale', 'stroke_start_s', 'stroke_end_s', 'cup_speed_m_s', 'relative_speed_m_s',
                                  'direction_deg', 'in_stroke'] and 'plan_mode' not in plain[0]
    # 2\ The strokes and the new R2.
    planned = [row for row in rows if row['decision'] == 'go' and row['plans'] not in ('', '0')]
    strokes = [row for row in planned if row['plan_mode'] == 'moving']
    assert len(planned) >= 10 and len(strokes) >= 0.8 * len(planned) and {row['plan_mode'] for row in planned} <= {'moving', 'static'}
    for row in strokes:
        assert (row['certified'], row['within_limits'], row['at_rest'], row['in_stroke']) == ('true', 'true', 'true', 'true')
        assert float(row['clear_environment_m']) >= 0.05 and float(row['clear_self_m']) >= 0.05 and float(row['clear_exempt_m']) > 0.0
        assert float(row['stroke_scale']) in (1.0, 0.75, 0.5, 0.25) and float(row['arrival_s']) <= float(row['stroke_start_s'])
        assert float(row['stroke_start_s']) < float(row['crossing_s']) < float(row['stroke_end_s'])
        assert 0.1 < float(row['cup_speed_m_s']) < 2.6 and 3.5 < float(row['relative_speed_m_s']) < 8.2 and float(row['direction_deg']) < 10.0
        assert float(row['tracking_m']) < 0.003 and (row['force_limit_ticks'], row['control_limit_ticks']) == ('0', '0')
    assert all(row['plan_mode'] == '' for row in rows if row['decision'] != 'go')
    summary = _Stored_Rows(moving['Interceptions'] / 'summary.csv')
    assert sum(int(row['r2_violations']) for row in summary) == 0 and sum(int(row['strokes']) for row in summary) == len(strokes)
    # 3\ The catch: the moving catch holds at 0.29 what the static one loses.
    manifest = json.loads((moving['Catches'] / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['r2_one_class_and_no_robot_contact'] is True and manifest['catch_mode'] == 'moving'
    caught, held = _Stored_Rows(moving['Catches'] / 'per_throw.csv'), _Stored_Rows(static['Catches'] / 'per_throw.csv')
    assert list(caught[0])[-4:-1] == ['plan_mode', 'stroke_scale', 'contact_speed_m_s'] and 'contact_speed_m_s' not in held[0]
    assert list(caught[0])[-1] == 'markers' and len(held) == 60 + 12 and len(caught) == 60 + 12 + 24 and manifest['markers_variant'] is True
    variant = [row for row in caught if row['markers'] == 'true']
    caught = [row for row in caught if row['markers'] == 'false']
    assert len(variant) == 24 and {row['seed'] for row in variant} == {'0'} and {row['damping_ratio'] for row in variant} == {'0.7', '0.5'}
    assert [[row[name] for name in decided[:6]] for row in variant[:12]] == [[row[name] for name in decided[:6]] for row in caught[:12]]
    gone = (lambda table, ratio: [row for row in table if row['decision'] == 'go' and row['damping_ratio'] == ratio])
    retained = (lambda table: sum(row['retained'] == 'true' for row in table))
    assert len(gone(variant, '0.5')) == len(gone(caught, '0.5')) and retained(gone(variant, '0.5')) >= 0.8 * len(gone(variant, '0.5'))
    assert all(row['robot_contact'] == '' for row in variant)
    assert retained(gone(caught, '0.7')) >= 0.8 * len(gone(caught, '0.7')) and len(gone(caught, '0.5')) == len(gone(held, '0.5')) >= 2
    assert retained(gone(caught, '0.5')) >= 0.8 * len(gone(caught, '0.5')) and retained(gone(held, '0.5')) <= 0.6 * len(gone(held, '0.5'))
    for row in gone(caught, '0.7') + gone(caught, '0.5'):
        assert row['robot_contact'] == '' and float(row['tracking_joint_rad']) < 0.005 and (row['class'] == 'caught') == (row['retained'] == 'true')
    stroked = [float(row['contact_speed_m_s']) for row in gone(caught, '0.7') if row['plan_mode'] == 'moving' and row['contact_speed_m_s']]
    assert len(stroked) >= 8 and np.median(stroked) < 7.0 and 3.5 < min(stroked) and max(stroked) < 8.2
    # 4\ R1 violated: one decision of the static rows changed, in a copy.
    other = tmp_path / 'other'
    shutil.copytree(str(static['Interceptions']), str(other / 'Interceptions' / 'QT_BASIS' / 'validation'))
    path = other / 'Interceptions' / 'QT_BASIS' / 'validation' / 'per_throw.csv'
    lines = path.read_text(encoding='ascii').splitlines()
    fields = lines[1].split(',')
    fields[12] = fields[12] + '1' if fields[12] else '1.0'
    path.write_text('\n'.join([lines[0], ','.join(fields)] + lines[2:]) + '\n', encoding='ascii')
    with monkeypatch.context() as guarded:
        _Forbid_Stored_Errors(module, guarded)
        assert module._Catch(_Catch_Args(run, moving_file, other, workers=2)) == 1
    assert 'R1 is violated' in capsys.readouterr().err
    violated = other / 'Interceptions' / 'QT_BASIS' / 'moving' / 'validation'
    assert json.loads((violated / 'manifest.json').read_text(encoding='utf-8'))['r1_decision_columns_equal_static'] is False
    assert (violated / 'per_throw.csv').read_bytes() == (moving['Interceptions'] / 'per_throw.csv').read_bytes()
    # 5\ The sweep in moving mode.
    evaluations = tmp_path / 'evaluations'
    _Stored_Errors(evaluations / 'test', 'real', 12, lambda flight, n, index: (4.0 + 6.0 * index) * (1.0 + 0.1 * flight) * (1.0 if n == 20 else 0.8))
    _Stored_Errors(evaluations / 'validation', 'validation', 12,
                   lambda flight, n, index: (2.0 + 3.0 * index) * (1.0 + 0.1 * flight) * (1.0 if n == 20 else 0.8))
    derive = _Load_Derive(root)
    _Forbid_Leaks(derive, monkeypatch)
    assert derive._Sweep(_Sweep_Args(run, moving_file, out / 'Catches', out / 'Interceptions', evaluations, workers=2)) == 0
    swept = _Stored_Rows(out / 'Catches' / 'QT_BASIS' / 'moving' / 'sweep' / 'per_throw.csv')
    assert list(swept[0])[:2] == ['error_source', 'error_scale'] and list(swept[0])[-4:] == ['plan_mode', 'stroke_scale', 'contact_speed_m_s',
                                                                                              'markers']
    assert {row['markers'] for row in swept} == {'false'}
    exact = [row for row in swept if (row['error_source'], row['error_scale']) == ('real', '0.0') and row['decision'] == 'go']
    assert len(exact) >= 4 and all(row['plan_mode'] in ('moving', 'static') for row in exact)
    assert not os.path.exists(out / 'Catches' / 'QT_BASIS' / 'sweep')


def _View_Main(module, monkeypatch, capsys, *args):
    """
    Description:
        'view' through the script's main() in process, with the arguments; its result has _Run's returncode, stdout and
        stderr.
    """

    import types
    monkeypatch.setattr(sys, 'argv', [os.path.join('scripts', 'catch.py'), 'view', *args])
    capsys.readouterr()
    code = module.main()
    captured = capsys.readouterr()
    return types.SimpleNamespace(returncode=code, stdout=captured.out, stderr=captured.err)


def test_view_refuses_a_test_throw_before_the_dataset_file_and_the_run_are_opened(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        'view' and the test partitions (SPEC §2.14): for a dataset whose manifest gives the partition
        sizes 3 / 2 / 5, the throw 5 (the first test throw) is refused, exit 1, naming the test partition and the allowed
        throws 0 to 4, although the dataset folder holds no flights.npz and the run does not exist: the refusal comes
        from the manifest alone, before either is opened, and with the test partition's selector and every recording
        reader of the script replaced by failures. The throw 4 (the last validation throw) passes that check and stops
        at the missing file; a random pick with any seed is never refused as a test throw; a folder without a manifest
        is refused as no dataset. No frames folder is written.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    dataset, out = tmp_path / 'Small', tmp_path / 'frames'
    dataset.mkdir()
    (dataset / 'manifest.json').write_text(json.dumps({'partition_sizes': [3, 2, 5], 'flights': 10}), encoding='utf-8')
    common = ('--run', str(tmp_path / 'no_run'), '--dataset', str(dataset), '--frames', str(out))
    refused = _View_Main(module, monkeypatch, capsys, *common, '--index', '5')
    assert refused.returncode == 1 and 'The throw 5 is in the dataset\'s test partition (the flights 5 to 9)' in refused.stderr
    assert 'throw (0 to 4)' in refused.stderr and 'flights.npz' not in refused.stderr and 'no_run' not in refused.stderr
    allowed = _View_Main(module, monkeypatch, capsys, *common, '--index', '4')
    assert allowed.returncode == 1 and 'test partition' not in allowed.stderr and 'flights.npz' in allowed.stderr
    for seed in range(20):
        picked = _View_Main(module, monkeypatch, capsys, *common, '--random', '--pick-seed', '%d' % seed)
        assert picked.returncode == 1 and 'test partition' not in picked.stderr and 'flights.npz' in picked.stderr
    beyond = _View_Main(module, monkeypatch, capsys, *common, '--index', '10')
    assert beyond.returncode == 1 and 'The throw 10 is not in the dataset: it holds the flights 0 to 9' in beyond.stderr
    empty = tmp_path / 'Empty'
    empty.mkdir()
    nothing = _View_Main(module, monkeypatch, capsys, '--run', str(tmp_path / 'no_run'), '--dataset', str(empty), '--frames', str(out),
                         '--index', '0')
    assert nothing.returncode == 1 and 'is not a dataset folder' in nothing.stderr and not out.exists()


def test_view_refuses_options_that_do_not_name_one_throw_the_wrong_launcher_and_an_existing_frames_folder(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        'view' refuses, exit 1 with the reason, before anything is read (the run and the dataset named do not exist): no
        throw named, two named (--index with --new, --random with --index), --pick-seed without --random, --new with
        --dataset, a frame size beyond 3840 x 2160 or not WIDTHxHEIGHT, an unknown theme or camera, a speed of 0, and
        an existing --frames folder, which stays as it was. Without --frames on macOS outside MuJoCo's launcher it names
        .venv/bin/mjpython; under the launcher (its environment variable set) that check passes.
    """

    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.delenv('MJPYTHON_BIN', raising=False)
    out = tmp_path / 'frames'
    base = ('--run', str(tmp_path / 'no_run'), '--dataset', str(tmp_path / 'no_dataset'), '--frames', str(out))
    cases = [((), 'exactly one of --index'), (('--index', '1', '--random'), 'exactly one of --index'),
             (('--index', '1', '--pick-seed', '3'), '--pick-seed picks a stored throw with --random'),
             (('--index', '1', '--size', '3841x2160'), '--size is WIDTHxHEIGHT'), (('--index', '1', '--size', 'large'), '--size is WIDTHxHEIGHT'),
             (('--index', '1', '--theme', 'pink'), '--theme is one of dark, light'), (('--index', '1', '--camera', 'top'), '--camera one of side'),
             (('--index', '1', '--speed', '0'), '--speed and --fps must be above 0')]
    for extra, reason in cases:
        result = _View_Main(module, monkeypatch, capsys, *base, *extra)
        assert result.returncode == 1 and reason in result.stderr and 'no_dataset' not in result.stderr.replace(str(out), ''), extra
    both = _View_Main(module, monkeypatch, capsys, '--run', str(tmp_path / 'no_run'), '--frames', str(out), '--index', '1', '--new', '7')
    assert both.returncode == 1 and 'exactly one of --index' in both.stderr
    mixed = _View_Main(module, monkeypatch, capsys, *base, '--new', '7')
    assert mixed.returncode == 1 and '--new generates a throw from --dataset-config' in mixed.stderr
    window = _View_Main(module, monkeypatch, capsys, '--run', str(tmp_path / 'no_run'), '--dataset', str(tmp_path / 'no_dataset'), '--index', '1')
    assert window.returncode == 1 and '.venv/bin/mjpython scripts/catch.py view' in window.stderr and 'no_dataset' not in window.stderr
    monkeypatch.setenv('MJPYTHON_BIN', 'set by the launcher')
    launched = _View_Main(module, monkeypatch, capsys, '--run', str(tmp_path / 'no_run'), '--dataset', str(tmp_path / 'no_dataset'), '--index', '1')
    assert launched.returncode == 1 and 'mjpython' not in launched.stderr and 'is not a dataset folder' in launched.stderr
    out.mkdir()
    (out / 'kept.txt').write_text('kept', encoding='utf-8')
    existing = _View_Main(module, monkeypatch, capsys, *base, '--index', '1')
    assert existing.returncode == 1 and 'exists; frames are never overwritten' in existing.stderr and os.listdir(str(out)) == ['kept.txt']


def test_a_new_throw_is_the_same_for_the_same_seed_aimed_as_asked_and_never_a_stored_one(root, tmp_path, monkeypatch):
    """
    Description:
        A new throw of 'view': one flight generated as 'dataset' generates. The same seed gives the
        same flights.npz, byte for byte; another seed another throw. Aimed into the catch area its target lies inside
        Areas.Catch (centre (0.7, -0.175, 0.9) m, size 0.3 x 0.75 x 0.5 m); aimed anywhere, from the same seed, it is
        another throw whose target lies in the quarter sphere of r 1.5 m about (0, 0, 0.3607) m. The observations are
        as a dataset's: fewer samples than frames (dropouts) and off the truth (the error model). Only identification
        recordings are opened (the reader fails for a validation file), 45 of them per throw. A seed of a stored
        dataset is refused naming the dataset; --aim anywhere with a configuration that gives the catch target is
        refused.
    """

    import Dataset.Core as Dataset_Core
    from Recording.Core import Assign_Sets, CONST_SET_VALIDATION
    module = _Load_Script(root)
    folder = os.path.join(root, 'Data', 'Recordings')
    sets = Assign_Sets([(sub, name) for sub in sorted(os.listdir(folder)) for name in sorted(os.listdir(os.path.join(folder, sub)))])
    opened, original = [], Dataset_Core.Load_Recording

    def Guarded(path):
        key = (os.path.basename(os.path.dirname(path)), os.path.basename(path))
        if sets[key] == CONST_SET_VALIDATION:
            raise AssertionError('a validation flight was opened: %s' % (key,))
        opened.append(key)
        return original(path)

    monkeypatch.setattr(Dataset_Core, 'Load_Recording', Guarded)
    monkeypatch.setattr(module, '_Stored_Seeds', lambda: {20261121: 'Catch_In'})
    configuration = module.Load_Configuration(module.CONST_CONFIG_FILE)
    shipped = os.path.join(root, 'Config', 'dataset.yaml')
    files = {}
    for name, seed, aim in (('a', 7, 'catch'), ('b', 7, 'catch'), ('c', 8, 'catch'), ('d', 7, 'anywhere')):
        (tmp_path / name).mkdir()
        files[name] = module._New_Throw(configuration, shipped, seed, aim, str(tmp_path / name))
        assert files[name] == str(tmp_path / name / 'flights.npz')
    digest = {name: hashlib.sha256(open(path, 'rb').read()).hexdigest() for name, path in files.items()}
    assert digest['a'] == digest['b'] and len({digest['a'], digest['c'], digest['d']}) == 3
    assert len(opened) == 4 * 45 and len(set(opened)) == 45
    catch, anywhere = module.Read_Dataset(files['a']), module.Read_Dataset(files['d'])
    assert catch.Flights == 1 and anywhere.Flights == 1
    assert np.all(np.abs(catch.Target[0] - np.array([0.7, -0.175, 0.9])) <= 0.5 * np.array([0.3, 0.75, 0.5]) + 1e-12)
    offset = anywhere.Target[0] - np.array([0.0, 0.0, 0.3607])
    assert np.linalg.norm(offset) <= 1.5 + 1e-9 and offset[2] >= -1e-9 and offset[1] <= 1e-9
    assert np.any(np.abs(anywhere.Target[0] - np.array([0.7, -0.175, 0.9])) > 0.5 * np.array([0.3, 0.75, 0.5]))
    times, observed = catch.Observations(0)
    frames, truth = catch.Truth(0)
    assert 2 <= len(times) < len(frames)
    error = np.linalg.norm(observed - truth[np.round(times * catch.Rate[0]).astype(int)], axis=1)
    assert 1e-5 < error.max() < 0.11
    with pytest.raises(module.Run_Error_Cls) as refused:
        module._New_Throw(configuration, shipped, 20261121, 'catch', str(tmp_path / 'a'))
    assert 'the seed of the stored dataset Catch_In' in str(refused.value) and 'flight 0' in str(refused.value)
    text = open(shipped, encoding='utf-8').read()
    start, end = text.index('  Target_Area:\n'), text.index('\n# The observation model')
    (tmp_path / 'catch.yaml').write_text(text[:start] + '  Target_Catch_Area:\n    Margin: 0.0\n' + text[end:], encoding='utf-8')
    with pytest.raises(module.Run_Error_Cls, match='--aim anywhere needs a dataset configuration with Throw.Target_Area'):
        module._New_Throw(configuration, str(tmp_path / 'catch.yaml'), 9, 'anywhere', str(tmp_path / 'c'))


def test_the_stored_seeds_are_those_of_the_stored_datasets_manifests(root, tmp_path, monkeypatch):
    """
    Description:
        The seeds a new throw may not take are read from the manifests under Data/Generated/Datasets: with two dataset
        folders there (seeds 5 and 20261121), a folder without a manifest and a manifest without a seed, exactly the two
        seeds are found, each with its folder's name; without the folder, none.
    """

    module = _Load_Script(root)
    monkeypatch.setattr(module, 'CONST_REPO_ROOT', str(tmp_path))
    assert module._Stored_Seeds() == {}
    parent = tmp_path / 'Data' / 'Generated' / 'Datasets'
    for name, manifest in (('Large', {'seed': 5}), ('Catch_In', {'seed': 20261121}), ('Broken', None), ('Seedless', {'flights': 3})):
        (parent / name).mkdir(parents=True)
        if manifest is not None:
            (parent / name / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert module._Stored_Seeds() == {5: 'Large', 20261121: 'Catch_In'}


@pytest.mark.slow
def test_view_of_a_stored_throw_equals_the_stored_catch_and_writes_its_frames(root, tmp_path, monkeypatch, capsys):
    """
    Description:
        'view' on the stored results (skipped in a clean clone, which has no stored run): with the
        test partition's selector and every recording reader of the script replaced by failures, (1) the validation
        throws 1000 (a go, caught) and 1002 (a no-go) of Catch_In are recorded with the catch's own code (the stored
        QT_BASIS run, seed 0, 800 deg/s^2, the stored validation margins) and each one's interception row and catch row
        equal the stored validation rows of 'intercept' and 'catch' byte for byte; the go's record holds the scene's
        positions at the start and after every step of 0.00025 s to its catch time plus the settle window, the ball at
        its release state first and within the cup's 0.19 m of the mouth's point at the end, 7 plans, and per state
        whether the ball is in the cup (not before the catch time less 0.05 s, entering within 0.05 s of it, in it for
        the last second); the no-go's holds
        the robot at the ready pose and the ball on its true path; the predictions asked by the pipeline are a subset of
        those shown, one per observed sample from N 20 to the catch time. (2) A stored row with one field changed (in a
        copy) is reported as different, naming the file and the field; a stored file without the throw is not compared.
        (3) The command end to end: it reports the same decision and class, prints that the stored check holds, and
        writes two frames of 320 x 180 px with frames.csv; every stored file it read has the bytes it had. (4) A run of
        a removed predictor given as a comparison (--compare; a copy of the stored run's files with Model.Name LS) is
        refused, naming the two predictors that remain, and no frame is written. The stored results of
        (1) to (4) are the static catch's, so the configuration is the shipped one with Catch_Mode static. (5) F15,
        when the moving catch's validation results are stored
        (Data/Generated/Interceptions/QT_BASIS/moving/validation): with the shipped configuration (moving) the same go is
        recorded with a stroke as its last plan, its two rows equal the stored moving rows byte for byte, the ghost of
        every plan is a catch pose (the mouth within 0.1 m of the predicted catch point, not the stroke's end 0.1 m and
        more further on), the cup's speed and the relative speed are those of its interception row, and the command says
        so. Every stored file read has the bytes it had.
    """

    run = os.path.join(root, 'Data', 'Generated', 'Models', 'QT_BASIS')
    stored_interception = os.path.join(root, 'Data', 'Generated', 'Interceptions', 'QT_BASIS', 'validation')
    stored_catch = os.path.join(root, 'Data', 'Generated', 'Catches', 'QT_BASIS', 'validation')
    dataset = os.path.join(root, 'Data', 'Generated', 'Datasets', 'Catch_In', 'flights.npz')
    if not all(os.path.exists(path) for path in (run, stored_interception, stored_catch, dataset)):
        pytest.skip('Data/Generated holds no stored QT_BASIS run with its validation interception and catch: a clean clone has none '
                    '(Data/Generated/ is never committed)')
    watched = [os.path.join(stored_interception, name) for name in ('per_throw.csv', 'margins.csv', 'manifest.json')] + \
        [os.path.join(stored_catch, 'per_throw.csv'), dataset]
    before = {path: hashlib.sha256(open(path, 'rb').read()).hexdigest() for path in watched}
    module = _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    from Scene.Core import Build_Scene, Scene_Robot
    from Catch.Core import Catch_Context
    from Catch.Decision import Read_Margins
    from Catch.Physical import Contact_Str
    from View.Core import Record_Throw
    static_file = _Moving_Config(module.CONST_CONFIG_FILE, tmp_path, 'static.yaml', 'static')
    configuration = module.Load_Configuration(str(static_file))
    training, _, _ = module._Load_Run_Copy(run)
    model = Build_Scene(Areas_Str=configuration.Areas)
    context = Catch_Context(model, Scene_Robot(model), Criteria_Str=module._Catch_Criteria(configuration))
    setup, _, _ = module._Interception_Setup(configuration, training, run, context)
    setup.Margins = Read_Margins(os.path.join(stored_interception, 'margins.csv'))
    contact = configuration.Contact
    settings = Contact_Str(Step=contact.Step, Friction=list(contact.Friction), Settle_Time=contact.Settle_Time, Rest_Speed=contact.Rest_Speed,
                           Rest_Time=contact.Rest_Time, Levels=[(contact.Time_Constant, contact.Damping_Ratio)])
    # 1\ A go and a no-go, recorded and compared with the stored rows.
    go = Record_Throw(setup, 'Catch_In', dataset, 1000, '0', 0, settings)
    no_go = Record_Throw(setup, 'Catch_In', dataset, 1002, '0', 0, settings)
    assert module._Stored_Check(go, stored_interception, stored_catch)[0] is True
    assert module._Stored_Check(no_go, stored_interception, stored_catch)[0] is True
    assert (go.Decision, go.Class, go.Go_N, len(go.Plan_Time)) == ('go', 'caught', 20, 7) and go.Catch_Time == 1.27
    steps = int(round((go.Catch_Time + contact.Settle_Time - go.State_Time[0]) / contact.Step))
    assert len(go.State_Time) == steps + 1 and np.allclose(np.diff(go.State_Time), contact.Step, rtol=0.0, atol=1e-9)
    data = module.Read_Dataset(dataset)
    from Flight.Core import Simulate
    start = Simulate(data.Release_State[1000], np.array([go.State_Time[0]]), Flight_Model_Str=setup.Flight)[0, :3]
    assert go.State_Time[0] <= 0.002 and np.allclose(go.State_Position[0, go.Ball_Address:go.Ball_Address + 3], start, rtol=0.0, atol=1e-9)
    assert np.allclose(go.State_Position[0, go.Robot_Address], setup.Ready, rtol=0.0, atol=2e-3)
    assert np.allclose(go.State_Position[-1, go.Robot_Address], go.Plan_Goal[-1], rtol=0.0, atol=5e-3)
    end = go.State_Position[-1, go.Ball_Address:go.Ball_Address + 3]
    centre, size = np.array(configuration.Areas.Catch.Centre), np.array(configuration.Areas.Catch.Size)
    assert np.all(np.abs(end - centre) <= 0.5 * size + 0.19)
    assert go.State_In_Cup.shape == go.State_Time.shape and go.State_In_Cup.dtype == bool and no_go.State_In_Cup.size == 0
    assert not go.State_In_Cup[go.State_Time < go.Catch_Time - 0.05].any() and go.State_In_Cup[go.State_Time > go.State_Time[-1] - 1.0].all()
    assert abs(float(go.State_Time[np.argmax(go.State_In_Cup)]) - go.Catch_Time) < 0.05
    shown = set(int(n) for n in go.Sample_N)
    assert shown == set(range(20, int(np.searchsorted(go.Observed_Time, go.Catch_Time)) + 1)) and go.Used.sum() >= 20 and go.Methods == ['QT_BASIS']
    assert go.Paths['QT_BASIS'].shape == (len(shown), 40, 3) and go.Catch_Point['QT_BASIS'].all()
    assert np.linalg.norm(go.Paths['QT_BASIS'][-1, -1] - np.array([0.732, -0.172, 0.928])) < 0.02
    assert (no_go.Decision, no_go.Class, len(no_go.Plan_Time)) == ('no-go', 'no_go_catchable', 0) and not no_go.Catch_Point['QT_BASIS'].any()
    assert np.allclose(no_go.State_Position[:, no_go.Robot_Address], setup.Ready)
    assert np.allclose(no_go.State_Position[:, no_go.Ball_Address:no_go.Ball_Address + 3], no_go.Truth_Position)
    assert abs(no_go.Truth_Time[-1] - float(data.Floor_Time[1002])) < 1e-12 and abs(no_go.Truth_Position[-1, 2] - 0.030) < 1e-6
    # 2\ A changed stored row is found; a stored file without the throw is not compared.
    for name, folder, field in (('interception', stored_interception, 13), ('catch', stored_catch, 9)):
        copies = {'interception': tmp_path / name / 'i', 'catch': tmp_path / name / 'c'}
        for key, source in (('interception', stored_interception), ('catch', stored_catch)):
            copies[key].mkdir(parents=True)
            prefix = 'Catch_In,1000,0,800,'
            lines = [line for line in open(os.path.join(source, 'per_throw.csv'), encoding='ascii') if line.startswith(('dataset,', prefix))]
            if key == name:
                parts = lines[1].split(',')
                parts[field] = parts[field] + '1'
                lines[1] = ','.join(parts)
            (copies[key] / 'per_throw.csv').write_text(''.join(lines), encoding='ascii')
        equal, found = module._Stored_Check(go, str(copies['interception']), str(copies['catch']))
        assert equal is False and 'the %s\'s row differs' % name in found and '[%d]' % field in found and str(copies[name]) in found
        assert module._Stored_Check(no_go, str(copies['interception']), str(copies['catch']))[0] is None
    # 3\ The command, end to end, into frames.
    out = tmp_path / 'frames'
    result = _View_Main(module, monkeypatch, capsys, '--run', run, '--config', str(static_file), '--index', '1000', '--frames', str(out),
                        '--size', '320x180', '--fps', '100', '--from', '0.30', '--to', '0.31', '--overlay')
    assert result.returncode == 0, result.stderr
    assert 'Decision:   go at N 20 (0.190 s), the catch time 1.270 s; catchable: true' in result.stdout
    assert 'Class:      caught (caught)' in result.stdout
    assert 'Throw:      Catch_In 1000 (validation)' in result.stdout and '[INFO]' not in result.stdout
    assert 'Stored check: the interception\'s row and the catch\'s row of this throw equal the stored validation rows byte for byte' in result.stderr
    assert 'nothing is run on, or validated on, the real UR10e' in result.stderr
    assert sorted(os.listdir(str(out))) == ['frame_000000.png', 'frame_000001.png', 'frames.csv']
    rows = _Stored_Rows(out / 'frames.csv')
    assert [(row['time_s'], row['phase'], row['method']) for row in rows] == [('0.300000', 'GO', 'QT_BASIS'), ('0.310000', 'GO', 'QT_BASIS')]
    header = (out / 'frame_000000.png').read_bytes()[:24]
    assert header[:8] == bytes([137, 80, 78, 71, 13, 10, 26, 10]) and header[16:24] == (320).to_bytes(4, 'big') + (180).to_bytes(4, 'big')
    # 4\ A removed predictor's run as a comparison is refused.
    removed = tmp_path / 'LS'
    removed.mkdir()
    text = open(os.path.join(run, 'training.yaml'), encoding='utf-8').read()
    text, count = re.subn(r'(?s)(\nModel:\n.*?\n  Name: )QT_BASIS\n', r'\1LS\n', text, count=1)
    assert count == 1
    (removed / 'training.yaml').write_text(text, encoding='utf-8')
    manifest = json.loads(open(os.path.join(run, 'manifest.json'), encoding='utf-8').read())
    manifest['model'], manifest['sha256']['configuration'] = 'LS', hashlib.sha256(text.encode('utf-8')).hexdigest()
    (removed / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    result = _View_Main(module, monkeypatch, capsys, '--run', run, '--config', str(static_file), '--compare', str(removed), '--index', '1000',
                        '--frames', str(tmp_path / 'refused'))
    assert result.returncode == 1 and 'Model.Name must be one of GLS_M4, QT_BASIS, found \'LS\'' in result.stderr
    assert not os.path.exists(tmp_path / 'refused')
    # 5\ The moving catch's stored results, when they exist.
    moving_interception = os.path.join(root, 'Data', 'Generated', 'Interceptions', 'QT_BASIS', 'moving', 'validation')
    moving_catch = os.path.join(root, 'Data', 'Generated', 'Catches', 'QT_BASIS', 'moving', 'validation')
    if os.path.isdir(moving_interception) and os.path.isdir(moving_catch):
        kept = [os.path.join(moving_interception, name) for name in ('per_throw.csv', 'margins.csv', 'manifest.json')] + \
            [os.path.join(moving_catch, 'per_throw.csv')]
        hashes = {path: hashlib.sha256(open(path, 'rb').read()).hexdigest() for path in kept}
        shipped = module.Load_Configuration(module.CONST_CONFIG_FILE)
        assert shipped.Interception.Catch_Mode == 'moving'
        moving_setup, _, _ = module._Interception_Setup(shipped, training, run, context)
        moving_setup.Margins = Read_Margins(os.path.join(moving_interception, 'margins.csv'))
        stroke = Record_Throw(moving_setup, 'Catch_In', dataset, 1000, '0', 0, settings)
        assert module._Stored_Check(stroke, moving_interception, moving_catch)[0] is True
        assert stroke.Moving and stroke.Stroke and (stroke.Decision, stroke.Go_N, stroke.Catch_Time) == (go.Decision, go.Go_N, go.Catch_Time)
        assert stroke.Stroke_Scale in (1.0, 0.75, 0.5, 0.25) and stroke.Plan_Time[-1] == stroke.Stroke_Time < stroke.Catch_Time
        assert len(stroke.Plan_Goal) == len(stroke.Plan_Time) and 0.1 < stroke.Cup_Speed < 2.6 and stroke.Relative_Speed < 8.2
        from Robot.Core import Forward_Kinematics
        predicted = stroke.Paths['QT_BASIS'][-1, -1]
        for pose in stroke.Plan_Goal:
            assert np.linalg.norm(Forward_Kinematics(model, context.Data, pose, Robot_Str=context.Robot)[0] - predicted) < 0.1
        assert np.max(np.abs(stroke.State_Position[-1, stroke.Robot_Address] - stroke.Plan_Goal[-1])) > 0.05
        shown = _View_Main(module, monkeypatch, capsys, '--run', run, '--index', '1000', '--frames', str(tmp_path / 'moving_frames'), '--size',
                           '320x180', '--fps', '100', '--from', '1.20', '--to', '1.21', '--overlay')
        assert shown.returncode == 0, shown.stderr
        assert 'Stored check: the interception\'s row and the catch\'s row of this throw equal the stored validation rows byte for byte' \
            in shown.stderr and 'Class:      %s' % stroke.Class in shown.stdout
        assert {path: hashlib.sha256(open(path, 'rb').read()).hexdigest() for path in kept} == hashes
    assert {path: hashlib.sha256(open(path, 'rb').read()).hexdigest() for path in watched} == before


def _Png_Pixels(path):
    """
    Description:
        The pixels of an 8-bit RGB PNG whose rows are unfiltered and whose image data is one chunk (the view's frames),
        read by the PNG specification: height x width x 3.
    """

    import struct
    import zlib
    raw = open(path, 'rb').read()
    width, height = struct.unpack('>II', raw[16:24])
    start = raw.index(b'IDAT')
    length = struct.unpack('>I', raw[start - 4:start])[0]
    rows = np.frombuffer(zlib.decompress(raw[start + 4:start + 4 + length]), dtype=np.uint8).reshape(height, 1 + 3 * width)
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].reshape(height, width, 3)


def test_view_frames_draw_the_ball_as_its_collision_sphere_and_with_the_markers_when_asked(root, tmp_path, scene, configuration):
    """
    Description:
        The frames of 'view' and the ball: for a constructed record that holds the ball still at the
        point the close-up camera looks at (the robot at its zero pose, away from it), one frame of 640 x 360 px without
        zones is written without and with --markers. Without them the ball is its collision sphere: an orange disc
        whose width and height are equal within 2 px, 15 to 18 px (R 30 mm at 2.3 m, a field of view of 32 degrees high
        over 360 px: 16.4 px). With them the marked mesh is drawn: the orange region is at least 5 px wider and higher
        and at most 26 px (the markers reach 45.9 mm from the centre: 25.1 px across when seen side-on, less from this
        camera, which looks along none of their axes). Each folder holds the frame and frames.csv with one row.
    """

    import argparse
    import mujoco
    from View.Core import Throw_Record_Str
    module = _Load_Script(root)
    ball = int(scene.jnt_qposadr[mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_JOINT, 'ball')])
    state = np.array(scene.qpos0)
    state[ball:ball + 3], state[ball + 3:ball + 7] = (0.7, -0.175, 0.95), (1.0, 0.0, 0.0, 0.0)
    record = Throw_Record_Str(Dataset='Made', Flight=0, Method='QT_BASIS', Methods=['QT_BASIS'], First_N=20, Decision='no-go', Class='no_go',
                              Truth_Time=np.array([0.0, 0.01]), Truth_Position=np.array([[0.7, -0.175, 0.95], [0.7, -0.175, 0.95]]),
                              State_Time=np.array([0.0, 0.01]), State_Position=np.array([state, state]),
                              Paths={'QT_BASIS': np.zeros((0, 40, 3))}, Catch_Point={'QT_BASIS': np.zeros(0, dtype=bool)},
                              Inference={'QT_BASIS': np.zeros(0)})
    extent = {}
    for name, markers in (('sphere', False), ('marked', True)):
        args = argparse.Namespace(theme='dark', camera='cup', fps=60.0, speed=1.0, start=0.0, end=0.0, no_zones=True, no_labels=True,
                                  ghost=False, overlay=False, markers=markers, frames=str(tmp_path / name))
        assert module._View_Frames(record, configuration, args, (640, 360)) == 1
        assert sorted(os.listdir(str(tmp_path / name))) == ['frame_000000.png', 'frames.csv']
        assert len((tmp_path / name / 'frames.csv').read_text(encoding='ascii').splitlines()) == 2
        image = _Png_Pixels(str(tmp_path / name / 'frame_000000.png')).astype(np.int64)
        orange = (image[..., 0] > 100) & (image[..., 1] > 0.3 * image[..., 0]) & (image[..., 1] < 0.7 * image[..., 0]) & \
            (image[..., 2] < 0.3 * image[..., 0])
        rows, columns = np.flatnonzero(orange.any(axis=1)), np.flatnonzero(orange.any(axis=0))
        extent[name] = (int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1))
    assert all(15 <= side <= 18 for side in extent['sphere']) and abs(extent['sphere'][0] - extent['sphere'][1]) <= 2
    assert all(extent['sphere'][k] + 5 <= extent['marked'][k] <= 26 for k in (0, 1))
