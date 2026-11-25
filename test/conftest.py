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
# File Name: ../test/conftest.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Pytest (Testing framework) [pip3 install pytest]
import pytest

"""
Description:
    Initialization of constants.

    Shared fixtures. src/ is put on sys.path here (there are no __init__.py files), and so are
    test/Legacy/, the folder of the 2022 test reference (Reference.py; SPEC §2.1), and test/scripts/,
    the folder of the scripts' shared test helpers (Script_Helpers.py), which only tests import.
"""
# The repository root.
#   Unit [string]
CONST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _path in (os.path.join(CONST_ROOT, 'src'), os.path.join(CONST_ROOT, 'test', 'Legacy'), os.path.join(CONST_ROOT, 'test', 'scripts')):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope='session')
def root():
    """
    Description:
        The repository root.
    """

    return CONST_ROOT


@pytest.fixture(scope='session')
def recordings(root):
    """
    Description:
        The 90 recordings, loaded once per test session (per worker with a parallel runner) and shared by
        the Recording tests, which only read them.
    """

    from Recording.Core import Load_Recordings
    return Load_Recordings(os.path.join(root, 'Data', 'Recordings'))


@pytest.fixture(scope='session')
def segmented(recordings):
    """
    Description:
        (set, flight, reason) of every recording at the shipped tolerance, once per test session.
    """

    from Recording.Segmentation import Segment_Flights
    return Segment_Flights(recordings)


@pytest.fixture(scope='session')
def configuration(root):
    """
    Description:
        The shipped system configuration, loaded once per test session; tests that change it work on a
        copy.
    """

    from Configuration.Core import Load_Configuration
    return Load_Configuration(os.path.join(root, 'Config', 'ball_catching.yaml'))


@pytest.fixture(scope='session')
def scene(configuration):
    """
    Description:
        The shipped MuJoCo scene, compiled once per test session (0.3 s per worker) and shared by the
        Scene and Robot tests, which only read it; a test that changes a model changes a copy.
    """

    from Scene.Core import Build_Scene
    return Build_Scene(Areas_Str=configuration.Areas)
