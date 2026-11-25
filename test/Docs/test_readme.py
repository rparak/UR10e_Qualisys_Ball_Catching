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
# File Name: ../test/Docs/test_readme.py
## =========================================================================== ##

# System (Default)
import os
# Regular expressions (Default)
import re
# Subprocess (Subprocess management)
import subprocess
# System-specific parameters (Default)
import sys
# Pytest (Testing framework) [pip3 install pytest]
import pytest

"""
Description:
    Initialization of constants.

    The README is the public page (F13): its title, the safety alert, the video's link, the images it shows, the
    commands it names, its numbers and the acknowledgement. The expectations are written out here, never read from
    the README; the numbers' sources are the sections of docs/SPEC.md.
"""
# The title and the safety statement (every output is a simulation result).
#   Unit [string]
CONST_TITLE = '# UR10e Robot Arm Catches a Thrown Ball with AI Prediction in MuJoCo'
CONST_SAFETY = 'Every output is a simulation result. Nothing is run on, or validated on, the real UR10e.'
# The acknowledgement and the author, exactly as decided for v1.0.
#   Unit [string]
CONST_ACKNOWLEDGEMENT = ('This project was carried out at the Institute of Robotics, Johannes Kepler University Linz, as part of an Erasmus+\n'
                         'research internship.')
CONST_AUTHOR = 'Author: **Roman Parak, Ph.D.**'
# The README's image budgets: each image shown and all of them together (decided for v1.0, not raised).
#   Unit [int, byte], [int, byte]
CONST_MAX_IMAGE = 400 * 1000
CONST_MAX_MEDIA = 6 * 1000 * 1000
# The scripts whose sub-commands the README names.
#   Unit [Tuple<string>]
CONST_SCRIPTS = ('scripts/catch.py', 'scripts/plots.py', 'scripts/Maintenance/derive.py')
# The three highlights: each one's statement in the README, the cell of docs/SPEC.md it states, and that section.
#   Unit [List<Tuple<string, string, string>>]
CONST_HIGHLIGHTS = [('**Same accuracy, about 58× faster.**', 'about 58× faster (58.4', '4.9'),
                    ('**99.3 % of catch attempts caught**, the moving catch on the `Catch_In` test throws at 800 °/s²',
                     '| test `Catch_In` | 800 / 400 | 95.7–97.5 / 95.0–97.4 % | 99.3 / 98.4–98.6 % |', '4.16'),
                    ('**0 false go** on the 4 681 uncatchable `Catch_Near` test throws at 800 °/s²',
                     '| `Catch_Near` | 800 / 400 | 4 681 / 4 718 | **0 [0, 0.079 %]', '4.11')]
# The README's numbers and the section of docs/SPEC.md each is stated in (§4: the measured facts).
#   Unit [List<Tuple<string, string>>]
CONST_NUMBERS = [('22.4', '4.9'), ('21.1', '4.9'), ('22.6', '4.9'), ('20.6', '4.9'), ('−0.29', '4.9'), ('[−0.72, +0.26]', '4.9'),
                 ('0.42', '4.9'), ('0.83', '4.9'), ('4 681', '4.11'), ('4 718', '4.11'), ('98.4–98.6', '4.16'),
                 ('95.7–97.5', '4.16'), ('95.0–97.4', '4.16'), ('87.7', '4.16'), ('80.0', '4.16'), ('99.3', '4.16'), ('58×', '4.9'),
                 ('0.009305', '4.2')]


def _Readme(root):
    """
    Description:
        The README's text.
    """

    with open(os.path.join(root, 'README.md'), encoding='utf-8') as handle:
        return handle.read()


def _Section(root, number):
    """
    Description:
        The text of one section of docs/SPEC.md, from its heading to the next heading of the same or a higher level.
    """

    with open(os.path.join(root, 'docs', 'SPEC.md'), encoding='utf-8') as handle:
        text = handle.read()
    match = re.search(r'^(#+) %s[ .]' % re.escape(number), text, re.M)
    assert match, 'docs/SPEC.md has no section %s' % number
    end = re.search(r'^#{1,%d} ' % len(match.group(1)), text[match.end():], re.M)
    return text[match.start():match.end() + (end.start() if end else len(text))]


def test_the_readme_opens_with_its_title_the_video_and_the_safety_alert(root):
    """
    Description:
        The first line is the title; the first image links the video; a warning alert with the safety statement comes
        before the first section.
    """

    text = _Readme(root)
    assert text.splitlines()[0] == CONST_TITLE
    head = text.split('\n## ', 1)[0]
    assert re.search(r'\[!\[[^\]]+\]\(Media/readme_catch\.gif\)\]\(', head)
    assert '> [!WARNING]' in head and CONST_SAFETY in head


def test_the_readme_shows_no_ip_address_no_local_path_and_no_e_mail_address(root):
    """
    Description:
        No IPv4 address, no absolute local path of a user and no e-mail address anywhere in the README.
    """

    text = _Readme(root)
    assert not re.search(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)', text)
    assert not re.search(r'/Users/|/home/|[A-Z]:\\\\', text)
    assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)


def test_every_image_the_readme_shows_is_in_media_within_the_budgets_and_each_picture_has_both_themes(root):
    """
    Description:
        Every image is a file of Media/, at most 400 kB each and 6 MB together; every <picture> pairs a dark source
        with a light image, the same figure in both themes.
    """

    text = _Readme(root)
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text) + re.findall(r'(?:src|srcset)="([^"]+)"', text)
    assert images and all(path.startswith('Media/') for path in images), images
    sizes = {path: os.path.getsize(os.path.join(root, path)) for path in set(images)}
    assert max(sizes.values()) <= CONST_MAX_IMAGE, sizes
    assert sum(sizes.values()) <= CONST_MAX_MEDIA
    pictures = re.findall(r'<picture>(.*?)</picture>', text, re.S)
    assert len(pictures) == 8
    for picture in pictures:
        dark = re.search(r'media="\(prefers-color-scheme: dark\)" srcset="Media/(readme_\w+)_dark\.png"', picture)
        light = re.search(r'<img src="Media/(readme_\w+)_light\.png"', picture)
        assert dark and light and dark.group(1) == light.group(1), picture


def test_every_picture_serves_its_light_png_by_default_and_its_dark_png_to_dark_mode_readers(root):
    """
    Description:
        Every <picture> holds one <source> for prefers-color-scheme: dark, whose srcset is a dark PNG of Media/, and one
        <img> whose src is the light PNG of the same figure (the default, what a light-mode reader gets); both files
        exist and differ.
    """

    pictures = re.findall(r'<picture>(.*?)</picture>', _Readme(root), re.S)
    assert pictures
    for picture in pictures:
        sources = re.findall(r'<source ([^>]*)>', picture)
        images = re.findall(r'<img ([^>]*)>', picture)
        assert len(sources) == 1 and len(images) == 1, picture
        assert 'media="(prefers-color-scheme: dark)"' in sources[0], picture
        dark = re.search(r'srcset="(Media/readme_\w+)_dark\.png"', sources[0])
        light = re.search(r'src="(Media/readme_\w+)_light\.png"', images[0])
        assert dark and light and dark.group(1) == light.group(1), picture
        paths = [os.path.join(root, dark.group(1) + '_dark.png'), os.path.join(root, light.group(1) + '_light.png')]
        assert all(os.path.isfile(path) for path in paths), paths
        with open(paths[0], 'rb') as first, open(paths[1], 'rb') as second:
            assert first.read() != second.read(), paths


def test_every_sub_command_the_readme_names_is_one_its_script_has(root):
    """
    Description:
        Each 'scripts/<script> <sub-command>' in the README's commands is a sub-command of that script, by the
        script's own --help (run in a fresh interpreter).
    """

    text = _Readme(root)
    for script in CONST_SCRIPTS:
        result = subprocess.run([sys.executable, os.path.join(root, script), '--help'], cwd=root, capture_output=True, text=True,
                                timeout=60)
        assert result.returncode == 0, result.stderr
        commands = set(re.search(r'\{([a-z,-]+)\}', result.stdout).group(1).split(','))
        named = set(re.findall(r'%s ([a-z][a-z-]+)' % re.escape(script), text))
        assert named and named <= commands, (script, named - commands)


@pytest.mark.parametrize('number, section', CONST_NUMBERS)
def test_each_number_of_the_readme_is_stated_in_its_spec_section(root, number, section):
    """
    Description:
        Each number the README states is in the README and in the section of docs/SPEC.md it cites.
    """

    assert number in _Readme(root)
    assert number in _Section(root, section)


def test_each_highlight_states_only_numbers_of_the_spec_section_it_cites(root):
    """
    Description:
        The three highlights under the opening paragraph: each makes its statement (CONST_HIGHLIGHTS) and cites one
        section of docs/SPEC.md, the section holds the cell the statement reads, and every number in the highlight's
        text (its link left out) is stated in that section.
    """

    text = _Readme(root)
    head = text.split('\n## ', 1)[0]
    block = '- **' + head.split('\n\n- **', 1)[1].split('\n\n', 1)[0]
    highlights = re.split(r'\n(?=- \*\*)', block)
    assert len(highlights) == 3, highlights
    for highlight in highlights:
        section = re.search(r'§(\d+(?:\.\d+)+)\]\(docs/SPEC\.md\)', highlight)
        assert section, highlight
        body = re.sub(r'\(\[[^\]]*\]\([^)]*\)\)', '', highlight)
        numbers = re.findall(r'\d+(?:[.,]\d+)?(?:[ –]\d+(?:[.,]\d+)?)?', body)
        assert numbers, highlight
        spec = _Section(root, section.group(1))
        assert all(number in spec for number in numbers), (section.group(1), [n for n in numbers if n not in spec])
    for (statement, cell, number), highlight in zip(CONST_HIGHLIGHTS, highlights):
        assert statement in highlight and '§%s](docs/SPEC.md)' % number in highlight, highlight
        assert cell in _Section(root, number), cell


def test_the_acknowledgement_the_author_and_the_licences_are_stated(root):
    """
    Description:
        The acknowledgement and the author exactly as decided; the MIT licence and the vendored UR10e's own licence,
        both files present.
    """

    text = _Readme(root)
    assert CONST_ACKNOWLEDGEMENT in text and CONST_AUTHOR in text
    assert '[MIT](LICENSE)' in text and '(models/universal_robots_ur10e/LICENSE)' in text
    assert os.path.isfile(os.path.join(root, 'LICENSE')) and os.path.isfile(os.path.join(root, 'models', 'universal_robots_ur10e', 'LICENSE'))
