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
# File Name: ../Dataset/Manifest.py
## =========================================================================== ##

# Hashlib (Secure hashes)
import hashlib
# JSON (JSON encoder and decoder)
import json
# Platform (Access to underlying platform's identifying data)
import platform
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Flight/Core, ../Observation/Core - the models whose parameters are hashed
from Flight.Core import Flight_Model_Str
from Observation.Core import Observation_Model_Str

"""
Description:
    Initialization of constants.

    The one writer of manifest.json (SPEC §2.13.10, §9.5): datasets (F6), runs and evaluations (from
    F7). Sorted keys, no wall-clock time: a manifest depends on its inputs only. The git state is given
    by the caller (the script); this module runs nothing.
"""
# The version of the manifest's fields.
#   Unit [int]
CONST_MANIFEST_VERSION = 1


def Sha256_Bytes(data: bytes) -> str:
    """
    Description:
        The SHA-256 hex digest of bytes.

    Args:
        (1) data [bytes]: The bytes.

    Returns:
        (1) parameter [string]: The digest.
    """

    return hashlib.sha256(data).hexdigest()


def Sha256_File(path: str) -> str:
    """
    Description:
        The SHA-256 hex digest of a file, read in blocks.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [string]: The digest.
    """

    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def Model_Hash(*, Flight_Model_Str: Flight_Model_Str, Observation_Model_Str: tp.Optional[Observation_Model_Str] = None) -> str:
    """
    Description:
        The SHA-256 of the models' parameters as canonical JSON (every float by repr, 17 significant
        digits): the identified model's hash of the manifest (SPEC §2.13.10).

    Args:
        (1) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
        (2) Observation_Model_Str [Observation_Model_Str(object)]: The observation error model, or None.

    Returns:
        (1) parameter [string]: The digest.
    """

    def Numbers(value: tp.Any) -> tp.Any:
        """
        Description:
            A parameter as JSON-ready numbers.

        Args:
            (1) value [Any]: A float or an array.

        Returns:
            (1) parameter [Any]: A float or a list of floats.
        """

        return [float(v) for v in np.ravel(value)] if np.ndim(value) else float(value)

    flight = Flight_Model_Str
    content: tp.Dict[str, tp.Any] = {'flight': {'Gravity': Numbers(flight.Gravity), 'Drag': Numbers(flight.Drag),
                                                'Drag_Std': Numbers(flight.Drag_Std)}}
    if Observation_Model_Str is not None:
        content['observation'] = {name: Numbers(getattr(Observation_Model_Str, name))
                                  for name in ('White_Std', 'White_Dof', 'Drift_Std', 'Drift_Time', 'Offset_Std', 'Offset_Time', 'Offset_Dof',
                                               'Offset_Bound', 'White_Bound')}
    return Sha256_Bytes(json.dumps(content, sort_keys=True).encode('utf-8'))


def Build_Manifest(kind: str, files: tp.Dict[str, str], fields: tp.Dict[str, tp.Any], commit: str, dirty: bool) -> tp.Dict[str, tp.Any]:
    """
    Description:
        The manifest of a dataset, a run or an evaluation (SPEC §2.13.10): its kind, the SHA-256 of each
        input and output file given (the configuration copies, the data file), the caller's fields (the
        seed or seeds, the model hash, the counts), the git commit and dirty flag, the platform, and the
        Python and NumPy versions (a learned predictor's run adds the torch version among its fields).

    Args:
        (1) kind [string]: 'dataset', 'run' or 'evaluation'.
        (2) files [Dict<string, string>]: Role -> SHA-256 of a file (e.g. 'configuration', 'data').
        (3) fields [Dict<string, Any>]: Further JSON-ready fields.
        (4) commit [string]: The git commit of the code.
        (5) dirty [bool]: Whether the tree had uncommitted changes.

    Returns:
        (1) parameter [Dict<string, Any>]: The manifest.
    """

    manifest = {'manifest_version': CONST_MANIFEST_VERSION, 'kind': kind, 'sha256': dict(files), 'git_commit': commit, 'git_dirty': bool(dirty),
                'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
                'numpy': np.__version__}
    manifest.update(fields)
    return manifest


def Write_Manifest(path: str, manifest: tp.Dict[str, tp.Any]) -> None:
    """
    Description:
        Write manifest.json: sorted keys, two-space indent, a final newline.

    Args:
        (1) path [string]: The file.
        (2) manifest [Dict<string, Any>]: The manifest (Build_Manifest).
    """

    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')


def Read_Manifest(path: str) -> tp.Dict[str, tp.Any]:
    """
    Description:
        Read manifest.json.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Dict<string, Any>]: The manifest.
    """

    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
