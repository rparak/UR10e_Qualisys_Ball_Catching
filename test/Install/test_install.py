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
# File Name: ../test/Install/test_install.py
## =========================================================================== ##

# System (Default)
import os
# Subprocess (Subprocess management)
import subprocess
# Regular expressions (Default)
import re
# Hashlib (Secure hashes; the list of a stand-in vendored folder)
import hashlib
# Pytest (Testing framework) [pip3 install pytest]
import pytest

"""
Description:
    Initialization of constants.

    install.sh is sourced in a fresh /bin/bash with a hermetic PATH: stubs for the commands that
    describe the machine (uname, sw_vers, the versioned python, python3) ahead of the system
    directories. What ran for real: the script's own functions and date. What ran only against
    stubs: platform detection (each platform of CONST_PLATFORMS and an unsupported OS) and the
    interpreter lookup. The check of the vendored robot runs for real, with the system's shasum: on the
    repository's own folder and on a stand-in folder with a list the test writes. pip and venv
    creation are not run here; install.sh itself was run for real with F4 (MacBook M2 Pro, macOS
    26.6.2, 2026-11-20: mujoco 3.16.0 installed, the 26 vendored files matched).
"""
# The Python version install.sh requires; written here, not read from the script under test.
#   Unit [string]
CONST_PYTHON_VERSION = '3.11'
# One entry per platform block of install.sh: uname -s, uname -m, the PLATFORM it sets, the word
# its INFO line names, and how the interpreter is found ('absolute': CONST_PYTHON_{PLATFORM} is a path;
# 'path': the versioned name is looked up on PATH).
#   Unit [List<Tuple<string>>]
CONST_PLATFORMS = [('Darwin', 'arm64', 'macos', 'macOS', 'absolute')]
# The number of files of the vendored UR10e folder (MuJoCo Menagerie at the pinned commit, git ls-tree -r).
#   Unit [int]
CONST_VENDORED_FILES = 26
# One log line: 'YYYY-MM-DD HH:MM:SS [LEVEL  ] message', the level padded to 7.
#   Unit [Pattern]
CONST_LOG_LINE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(INFO   |WARNING|ERROR  )\] \S.*$')


def _Stub(directory, name, body):
    path = directory / name
    path.write_text('#!/bin/sh\n' + body + '\n', encoding='utf-8')
    path.chmod(0o755)
    return path


def _Run(root, stubs, script):
    """
    Description:
        Source install.sh and run a snippet; returns (exit code, stdout lines, stderr).
    """

    command = 'source "%s"; %s' % (os.path.join(root, 'install.sh'), script)
    result = subprocess.run(['/bin/bash', '-c', command], capture_output=True, text=True, timeout=60,
                            env={'PATH': '%s:/usr/bin:/bin' % stubs, 'HOME': str(stubs)})
    lines = [line for line in result.stdout.splitlines() if line]
    return result.returncode, lines, result.stderr


@pytest.fixture
def stubs(tmp_path):
    """
    Description:
        A stub directory; python3 records that it was called (it must never be).
    """

    directory = tmp_path / 'bin'
    directory.mkdir()
    _Stub(directory, 'python3', 'touch "%s/python3_was_called"; exit 1' % tmp_path)
    _Stub(directory, 'sw_vers', 'echo 26.6.2')
    return directory


def test_the_script_parses(root):
    """
    Description:
        bash -n accepts the script.
    """

    assert subprocess.run(['/bin/bash', '-n', os.path.join(root, 'install.sh')]).returncode == 0


def _Python(stubs, platform, lookup, version):
    """
    Description:
        A python stub reporting `version`, placed where install.sh looks for it on `platform`;
        returns (the shell snippet that selects the platform, the path install.sh must find).
    """

    # macOS is the only platform: the interpreter is found by its absolute Homebrew path.
    python = _Stub(stubs, 'brew_python', 'echo ' + version)
    return 'PLATFORM=%s; CONST_PYTHON_%s="%s";' % (platform, platform.upper(), python), python


@pytest.mark.parametrize('system, arch, platform, word, lookup', CONST_PLATFORMS)
def test_the_platform_is_detected(root, stubs, system, arch, platform, word, lookup):
    """
    Description:
        uname decides the platform, and the INFO line naming it is printed in the log format.
    """

    _Stub(stubs, 'uname', '[ "$1" = "-s" ] && echo %s || echo %s' % (system, arch))
    code, lines, _ = _Run(root, stubs, 'Detect_Platform; echo "PLATFORM=$PLATFORM"')
    assert code == 0 and lines[-1] == 'PLATFORM=%s' % platform
    assert any('[INFO   ] Detected %s' % word in line for line in lines[:-1])
    assert all(CONST_LOG_LINE.match(line) for line in lines[:-1])


def test_an_unsupported_system_fails_with_error_lines(root, stubs):
    """
    Description:
        An operating system without a platform block is refused with ERROR lines, the last naming
        the supported platforms, and a non-zero exit.
    """

    _Stub(stubs, 'uname', '[ "$1" = "-s" ] && echo FreeBSD || echo amd64')
    code, lines, _ = _Run(root, stubs, 'Detect_Platform')
    assert code == 1 and all(CONST_LOG_LINE.match(line) for line in lines)
    assert '[ERROR  ]' in lines[-1] and all(word in lines[-1] for _, _, _, word, _ in CONST_PLATFORMS)


@pytest.mark.parametrize('system, arch, platform, word, lookup', CONST_PLATFORMS)
def test_python_is_found_by_its_explicit_name_never_python3(root, stubs, tmp_path, system, arch, platform, word,
                                                            lookup):
    """
    Description:
        The interpreter is found where the platform's block looks for it; the bare python3 is
        never called.
    """

    select, python = _Python(stubs, platform, lookup, CONST_PYTHON_VERSION)
    code, lines, _ = _Run(root, stubs, '%s Find_Python; echo "PYTHON=$PYTHON"' % select)
    assert code == 0 and lines[-1] == 'PYTHON=%s' % python
    assert all(CONST_LOG_LINE.match(line) for line in lines[:-1])
    assert not (tmp_path / 'python3_was_called').exists()


@pytest.mark.parametrize('system, arch, platform, word, lookup', CONST_PLATFORMS)
def test_a_missing_or_wrong_python_is_refused_with_its_install_command(root, stubs, system, arch, platform,
                                                                       word, lookup):
    """
    Description:
        A missing interpreter, or one of another version, fails with the platform's install command
        (shown, never run: a prerequisite is checked by its presence only).
    """

    select, _ = _Python(stubs, platform, lookup, '2.7')
    code, lines, _ = _Run(root, stubs, '%s Find_Python' % select)
    assert code == 1 and 'Install Python ' + CONST_PYTHON_VERSION in lines[-1]
    missing = 'PLATFORM=%s; CONST_PYTHON_%s="%s";' % (platform, platform.upper(), stubs / 'absent')
    os.remove(stubs / 'brew_python')
    code, lines, _ = _Run(root, stubs, '%s Find_Python' % missing)
    assert code == 1 and 'Install it with: ' in lines[-1]


def _Vendored(tmp_path):
    """
    Description:
        A stand-in root: models/universal_robots_ur10e with two files and models/SHA256SUMS naming
        their SHA-256, in the form the build script writes; returns the root and the folder.
    """

    folder = tmp_path / 'root' / 'models' / 'universal_robots_ur10e'
    (folder / 'assets').mkdir(parents=True)
    files = {'universal_robots_ur10e/assets/base.obj': b'v 0 0 0\n', 'universal_robots_ur10e/ur10e.xml': b'<mujoco/>\n'}
    for name, data in files.items():
        (tmp_path / 'root' / 'models' / name).write_bytes(data)
    (tmp_path / 'root' / 'models' / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (hashlib.sha256(data).hexdigest(), name)
                                                                    for name, data in sorted(files.items())), encoding='utf-8')
    return tmp_path / 'root', folder


def test_the_vendored_robot_of_the_repository_matches_its_hash_list(root, stubs):
    """
    Description:
        The repository's vendored UR10e passes the installer's check: all 26 files match
        models/SHA256SUMS, reported in one INFO line in the log format.
    """

    code, lines, stderr = _Run(root, stubs, 'Verify_Vendored')
    assert code == 0, (lines, stderr)
    assert len(lines) == 1 and CONST_LOG_LINE.match(lines[0])
    assert lines[0].endswith('[INFO   ] The vendored UR10e matches models/SHA256SUMS (%d files).' % CONST_VENDORED_FILES)


def test_a_changed_or_missing_vendored_file_is_refused_naming_it(root, stubs, tmp_path):
    """
    Description:
        In a stand-in root the unchanged folder passes (2 files); one changed byte fails with an ERROR
        line naming that file and no other, the last line giving the command that restores the folder; a
        removed file is named the same way. Nothing is repaired or deleted by the check.
    """

    stand_in, folder = _Vendored(tmp_path)
    select = 'CONST_ROOT="%s"; Verify_Vendored' % stand_in
    code, lines, _ = _Run(root, stubs, select)
    assert code == 0 and lines[-1].endswith('The vendored UR10e matches models/SHA256SUMS (2 files).')

    (folder / 'assets' / 'base.obj').write_bytes(b'v 0 0 1\n')
    code, lines, _ = _Run(root, stubs, select)
    assert code == 1 and all(CONST_LOG_LINE.match(line) for line in lines)
    assert any('[ERROR  ] shasum: universal_robots_ur10e/assets/base.obj: FAILED' in line for line in lines)
    assert not any('ur10e.xml' in line for line in lines)
    assert lines[-1].endswith('Restore it with: git checkout -- models/universal_robots_ur10e')
    assert (folder / 'assets' / 'base.obj').read_bytes() == b'v 0 0 1\n'

    (folder / 'assets' / 'base.obj').write_bytes(b'v 0 0 0\n')
    os.remove(folder / 'ur10e.xml')
    code, lines, _ = _Run(root, stubs, select)
    assert code == 1 and any('universal_robots_ur10e/ur10e.xml' in line and 'FAILED' in line for line in lines)


def test_a_missing_hash_list_is_refused_with_the_command_that_restores_it(root, stubs, tmp_path):
    """
    Description:
        Without models/SHA256SUMS nothing can be checked: the installer fails with ERROR lines, the last
        naming the command that restores the list, instead of passing an unchecked folder.
    """

    stand_in, _ = _Vendored(tmp_path)
    os.remove(stand_in / 'models' / 'SHA256SUMS')
    code, lines, _ = _Run(root, stubs, 'CONST_ROOT="%s"; Verify_Vendored' % stand_in)
    assert code == 1 and all(CONST_LOG_LINE.match(line) for line in lines)
    assert '[ERROR  ] The SHA-256 list models/SHA256SUMS is missing.' in lines[0]
    assert lines[-1].endswith('Restore it with: git checkout -- models/SHA256SUMS')


def test_the_installer_checks_the_vendored_robot_after_the_requirements(root, stubs):
    """
    Description:
        Main runs the check of the vendored robot as its last step, after the requirements are installed,
        and --detect-only names it in the plan without running it: a check that exists but is never
        called would pass every test of the function itself.
    """

    with open(os.path.join(root, 'install.sh'), 'r', encoding='utf-8') as file:
        text = file.read()
    main = text[text.index('\nMain() {'):]
    assert re.search(r'\n    Ensure_Venv\n    Install_Requirements\n    Verify_Vendored\n    Log INFO "Done\.', main)
    _Stub(stubs, 'uname', '[ "$1" = "-s" ] && echo Darwin || echo arm64')
    python = _Stub(stubs, 'brew_python', 'echo ' + CONST_PYTHON_VERSION)
    code, lines, _ = _Run(root, stubs, 'CONST_PYTHON_MACOS="%s"; Main --detect-only' % python)
    assert code == 0 and any('Plan: ' in line and 'the hash check of the vendored UR10e' in line for line in lines)
    assert not any('matches models/SHA256SUMS' in line for line in lines)
