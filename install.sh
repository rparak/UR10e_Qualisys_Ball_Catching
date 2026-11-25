#!/usr/bin/env bash
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
# File Name: ../install.sh
## =========================================================================== ##

# Description:
#   Install the Robotic Ball Catching (MuJoCo) environment: a .venv with the pinned requirements, and the
#   check that the vendored robot model is the one that was vendored.
#
#   Steps, cheapest check first:
#     1. Detect the platform (SPEC §5: macOS on Apple silicon, verified; no other platform).
#     2. Find Python 3.11. It is always called by its explicit name, never as a bare python3,
#        which resolves to whatever interpreter is first on PATH.
#     3. Create .venv if it is absent. A broken .venv is reported with the command to remove it;
#        it is never deleted by this script.
#     4. pip install -r requirements.txt (every version pinned).
#     5. Check every file of the vendored UR10e (models/universal_robots_ur10e, MuJoCo Menagerie,
#        never edited) against its SHA-256 in models/SHA256SUMS. Nothing is downloaded.
#
#   Usage:
#     ./install.sh                  run all steps
#     ./install.sh --detect-only    report the platform, the interpreter and the plan, then stop
#
#   Output: every line is 'YYYY-MM-DD HH:MM:SS [LEVEL  ] message', the level padded to 7.

# Description:
#   Initialization of constants.

# The repository root: the directory holding this script, so it works from any directory.
#   Unit [string]
CONST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The Python version every pinned requirement was verified with.
#   Unit [string]
CONST_PYTHON_VERSION="3.11"
# The platforms: one block per platform of SPEC §5 (macOS only), here and in Detect_Platform and
# Find_Python.
# ----- platform: macOS -----
# The interpreter on macOS: Homebrew's, by absolute path, so PATH order cannot pick another one.
#   Unit [string]
CONST_PYTHON_MACOS="/opt/homebrew/bin/python3.11"
# The prerequisite's install command on macOS (shown, never run).
#   Unit [string]
CONST_INSTALL_MACOS="brew install python@3.11"
# ----- end macOS -----
# The supported platforms, as the failure message names them.
#   Unit [string]
CONST_PLATFORMS_TEXT="macOS (Apple silicon)"
# The vendored robot's folder and its SHA-256 list, relative to the root; the list names each file
# relative to the folder that holds it (written by scripts/Maintenance/build_scene.py; SPEC §11).
#   Unit [string]
CONST_VENDORED_DIR="models/universal_robots_ur10e"
CONST_VENDORED_SUMS="models/SHA256SUMS"
# Width the level is padded to, so the messages line up.
#   Unit [int]
CONST_LEVEL_WIDTH=7

# Description:
#   Print one log line in the project's format.
# Args:
#   (1) level [string]: INFO, WARNING or ERROR.
#   (2) message [string]: The message.
# Returns:
#   (1) parameter [string]: The line, on stdout.
Log() {
    local level="$1"; shift
    printf '%s [%-*s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CONST_LEVEL_WIDTH" "$level" "$*"
}

# Description:
#   Report one or more ERROR lines and exit non-zero. The last line says what to do.
# Args:
#   (1 - n) message [string]: One ERROR line each.
# Returns:
#   (1) parameter [int]: Exit status 1.
Die() {
    local line
    for line in "$@"; do Log ERROR "$line"; done
    exit 1
}

# Description:
#   Name the line and command of anything unanticipated, instead of a bare trace.
# Args:
#   (1) line [int]: The line number.
#   (2) command [string]: The failing command.
# Returns:
#   None
On_Unexpected_Error() {
    Log ERROR "Unexpected failure at install.sh line $1: $2"
    Log ERROR "Nothing was deleted. Fix the cause above and run ./install.sh again."
    exit 1
}

# Description:
#   Interrupted: say that re-running is safe.
# Args:
#   None
# Returns:
#   None
On_Interrupt() {
    Log WARNING "Interrupted. Every step is re-runnable; run ./install.sh again."
    exit 130
}

# Description:
#   Detect the operating system and architecture.
# Args:
#   None
# Returns:
#   (1) parameter [string]: Sets PLATFORM to the name of the matching platform block; fails on
#       anything else.
Detect_Platform() {
    local os arch
    os="$(uname -s)"; arch="$(uname -m)"
    case "$os" in
        # ----- platform: macOS -----
        Darwin)
            PLATFORM="macos"
            if [ "$arch" = "arm64" ]; then
                Log INFO "Detected macOS $(sw_vers -productVersion 2>/dev/null) on Apple silicon ($arch)."
            else
                Log WARNING "Detected macOS on $arch; only Apple silicon has been verified."
            fi ;;
        # ----- end macOS -----
        *)
            Die "Unsupported operating system '$os'." \
                "This project installs on $CONST_PLATFORMS_TEXT only." ;;
    esac
}

# Description:
#   Find the project's Python interpreter by its explicit name and check its version.
# Args:
#   None
# Returns:
#   (1) parameter [string]: Sets PYTHON to the interpreter path; fails with the install command.
Find_Python() {
    local candidate where install version
    case "$PLATFORM" in
        # ----- platform: macOS -----
        macos) candidate="$CONST_PYTHON_MACOS"; where="at $CONST_PYTHON_MACOS"; install="$CONST_INSTALL_MACOS" ;;
        # ----- end macOS -----
    esac
    # The presence check of the prerequisite: the install command is shown, never run.
    if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then
        Die "Python $CONST_PYTHON_VERSION was not found $where." "Install it with: $install"
    fi
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [ "$version" != "$CONST_PYTHON_VERSION" ]; then
        Die "$candidate is Python $version, not $CONST_PYTHON_VERSION." \
            "Install Python $CONST_PYTHON_VERSION ($install); the pinned requirements were verified with it only."
    fi
    PYTHON="$candidate"
    Log INFO "Using Python $version at $PYTHON."
}

# Description:
#   Create .venv if absent; report a broken one with the command to remove it.
# Args:
#   None
# Returns:
#   None
Ensure_Venv() {
    local venv="$CONST_ROOT/.venv"
    if [ -d "$venv" ]; then
        if [ "$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = "$CONST_PYTHON_VERSION" ]; then
            Log INFO ".venv already present and working (Python $CONST_PYTHON_VERSION)."
            return 0
        fi
        Die ".venv exists but its interpreter is missing or not Python $CONST_PYTHON_VERSION." \
            "Remove it with: rm -rf \"$venv\" and run ./install.sh again."
    fi
    "$PYTHON" -m venv "$venv" || Die "Creating .venv failed." "Check that the venv module is installed for $PYTHON."
    Log INFO "Created .venv with $PYTHON."
}

# Description:
#   Install the pinned requirements into .venv; pip's output is kept and only its tail is shown
#   on failure.
# Args:
#   None
# Returns:
#   None
Install_Requirements() {
    local out line
    out="$(mktemp)"
    if "$CONST_ROOT/.venv/bin/python" -m pip install --disable-pip-version-check -r "$CONST_ROOT/requirements.txt" > "$out" 2>&1; then
        if grep -q '^Successfully installed' "$out"; then
            Log INFO "Installed: $(grep '^Successfully installed' "$out" | sed 's/^Successfully installed //')"
        else
            Log INFO "pip changed nothing; every pinned requirement is already installed."
        fi
        rm -f "$out"
    else
        tail -n 15 "$out" | while IFS= read -r line; do Log ERROR "pip: $line"; done
        rm -f "$out"
        Die "pip install -r requirements.txt failed (output above)." "Fix the cause and run ./install.sh again."
    fi
}

# Description:
#   Check the vendored UR10e against its SHA-256 list; every file that differs or is missing is named.
#   shasum is part of macOS; it is checked by its presence only.
# Args:
#   None
# Returns:
#   None
Verify_Vendored() {
    local sums="$CONST_ROOT/$CONST_VENDORED_SUMS" out line count
    command -v shasum > /dev/null 2>&1 || Die "shasum was not found." "It is part of macOS (/usr/bin/shasum); check PATH and run ./install.sh again."
    if [ ! -f "$sums" ]; then
        Die "The SHA-256 list $CONST_VENDORED_SUMS is missing." "Restore it with: git checkout -- $CONST_VENDORED_SUMS"
    fi
    out="$(mktemp)"
    if (cd "$(dirname "$sums")" && shasum -a 256 -c "$(basename "$sums")") > "$out" 2>&1; then
        count="$(grep -c ': OK$' "$out" || true)"
        rm -f "$out"
        Log INFO "The vendored UR10e matches $CONST_VENDORED_SUMS ($count files)."
    else
        grep -v ': OK$' "$out" | while IFS= read -r line; do Log ERROR "shasum: $line"; done
        rm -f "$out"
        Die "The vendored UR10e in $CONST_VENDORED_DIR differs from $CONST_VENDORED_SUMS (lines above)." \
            "It is never edited. Restore it with: git checkout -- $CONST_VENDORED_DIR"
    fi
}

# Description:
#   Run the steps in order.
# Args:
#   (1) option [string]: Optional --detect-only.
# Returns:
#   None
Main() {
    trap 'On_Unexpected_Error $LINENO "$BASH_COMMAND"' ERR
    trap On_Interrupt INT TERM
    case "${1:-}" in
        ""|--detect-only) ;;
        *) Die "Unknown option '$1'." "Usage: ./install.sh [--detect-only]" ;;
    esac
    Detect_Platform
    Find_Python
    Log INFO "Plan: .venv with Python $CONST_PYTHON_VERSION, the pinned requirements.txt, the hash check of the vendored UR10e."
    if [ "${1:-}" = "--detect-only" ]; then Log INFO "Detect only: stopping before any change."; return 0; fi
    Ensure_Venv
    Install_Requirements
    Verify_Vendored
    Log INFO "Done. Run the tests with: .venv/bin/python -m pytest"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then Main "$@"; fi
