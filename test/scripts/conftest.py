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
# File Name: ../test/scripts/conftest.py
## =========================================================================== ##

# System (Default)
import os
# Hashlib (Secure hashes)
import hashlib
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the scripts' shared test helpers
#   ./Script_Helpers
from Script_Helpers import _Run, _Dataset_Config, _Learned_Config, _Load_Script, _Committed_Copy, _Catch_Config, _Catch_Args

"""
Description:
    The fixtures the tests of the two scripts share (test/scripts/test_catch.py, test/scripts/Maintenance/test_derive.py):
    small datasets, a quickly trained run and its stored validation chain, each made once per test session (per worker
    with a parallel runner) by the scripts themselves.
"""


@pytest.fixture(scope='session')
def small_dataset(root, tmp_path_factory):
    """
    Description:
        A 30-flight dataset (21 / 5 / 4 by the largest-remainder rule) generated once per worker by
        'dataset' from the shipped configuration: (folder, SHA-256 of its flights.npz).
    """

    tmp = tmp_path_factory.mktemp('small')
    config = _Dataset_Config(root, tmp, 30)
    out = tmp / 'datasets'
    result = _Run(root, tmp, 'dataset', '--config', str(config), '--out', str(out))
    assert result.returncode == 0, result.stderr
    folder = out / 'Small'
    return folder, hashlib.sha256((folder / 'flights.npz').read_bytes()).hexdigest()


@pytest.fixture(scope='session')
def learned_run(root, tmp_path_factory, small_dataset):
    """
    Description:
        A learned run trained once per worker by 'train' on the 30-flight dataset (21 training flights,
        2 epochs, a state of 8): (run folder, its configuration, the working folder).
    """

    tmp = tmp_path_factory.mktemp('learned')
    config = _Learned_Config(root, tmp, small_dataset)
    result = _Run(root, tmp, 'train', '--config', str(config))
    assert result.returncode == 0, result.stderr
    return tmp / 'models' / 'Learned', config, tmp


@pytest.fixture(scope='session')
def catch_dataset(root, tmp_path_factory):
    """
    Description:
        A 60-flight catch dataset (the throws aimed into Areas.Catch, margin 0; the partitions
        0.10 / 0.45 / 0.45, so 6 / 27 / 27) generated once per worker by 'dataset': (folder, SHA-256).
    """

    tmp = tmp_path_factory.mktemp('catch')
    text = open(os.path.join(root, 'Config', 'dataset.yaml'), encoding='utf-8').read()
    text = text.replace('  Flights: 10000\n', '  Flights: 60\n').replace('  Name: Default\n', '  Name: Catch_Small\n')
    text = text.replace('  Train: 0.70\n', '  Train: 0.10\n').replace('  Validation: 0.15\n', '  Validation: 0.45\n')
    text = text.replace('  Test: 0.15\n', '  Test: 0.45\n')
    start, end = text.index('  Target_Area:\n'), text.index('\n# The observation model')
    text = text[:start] + '  Target_Catch_Area:\n    Margin: 0.0\n' + text[end:]
    config = tmp / 'catch.yaml'
    config.write_text(text, encoding='utf-8')
    result = _Run(root, tmp, 'dataset', '--config', str(config), '--out', str(tmp / 'datasets'))
    assert result.returncode == 0, result.stderr
    folder = tmp / 'datasets' / 'Catch_Small'
    return folder, hashlib.sha256((folder / 'flights.npz').read_bytes()).hexdigest()


@pytest.fixture(scope='session')
def stored_chain(root, tmp_path_factory, learned_run, catch_dataset):
    """
    Description:
        The validation chain of the learned run on the 60-flight catch dataset, stored once per worker by 'catch' with
        one process (the decision, the interception and the catch; about 30 s): (the committed run copy, the
        configuration, the folder holding Decisions/, Interceptions/ and Catches/).
    """

    tmp = tmp_path_factory.mktemp('stored_chain')
    module = _Load_Script(root)
    run, config = _Committed_Copy(learned_run[0], tmp), _Catch_Config(root, tmp, catch_dataset)
    assert module._Catch(_Catch_Args(run, config, tmp / 'out')) == 0
    return run, config, tmp / 'out'
