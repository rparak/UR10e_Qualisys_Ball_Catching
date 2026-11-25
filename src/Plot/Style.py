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
# File Name: ../Plot/Style.py
## =========================================================================== ##

# OS (Operating system interfaces; a figure written through a temporary file)
import os
# Shutil (LaTeX found on the PATH)
import shutil
# Contextlib (the style of one figure as a context)
import contextlib
# Dataclasses (Data classes)
from dataclasses import dataclass
# Matplotlib (Figures) [pip3 install matplotlib]; the Agg backend before anything imports pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
# Warnings (SciencePlots 2.2.2 calls two functions matplotlib 3.11 deprecates: silenced at its import only; KNOWN_ISSUES)
import warnings
# SciencePlots (the 'science' style registered in matplotlib's style library) [pip3 install SciencePlots]
with warnings.catch_warnings():
    warnings.simplefilter('ignore', matplotlib.MatplotlibDeprecationWarning)
    import scienceplots  # noqa: F401
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    The one style of the figures of scripts/plots.py (the visualisation, delivery 2,
    with the dark theme): SciencePlots' 'science' style, with LaTeX when it is installed (usetex) and matplotlib's
    own mathtext otherwise ('no-latex'), and a theme per background as data: dark (the default, GitHub's dark page) and
    light. Every figure takes its colours, line styles, markers and sizes from here; no figure sets its own. Each
    entity (a method, a catch mode, a class) has one colour per theme, and every colour that carries meaning is held at
    a WCAG contrast of 4.5:1 against its theme's background (a test). The colours are Okabe-Ito's hues, a lighter tint
    on the dark background and a darker shade on the light one; the values were chosen to meet the floor, not
    measured otherwise.
"""
# The themes, the default first (dark by default, the light one selectable).
#   Unit [Tuple<string>]
CONST_THEMES = ('dark', 'light')
# Each theme's background, text and axes, and its muted colour (the observed samples, a no-go, a grid at low alpha).
#   Unit [Dict<string, string: #RRGGBB>]
CONST_BACKGROUND = {'dark': '#0D1117', 'light': '#FFFFFF'}
CONST_FOREGROUND = {'dark': '#E6EDF3', 'light': '#1F2328'}
CONST_MUTED = {'dark': '#8B949E', 'light': '#6E7781'}
# One colour per entity and theme (dark, light): the two predictors, the two catch modes and the markers variant, the
# catch's classes, the true path (the theme's text colour) and the six joints (each also by its line style). In both
# themes 'no go' is the lighter grey and 'no go, catchable' the darker one, so that the outcomes that matter carry the
# weight; in the light theme 'no go' is the muted colour, the lightest grey at the contrast floor (4.55:1 on white).
#   Unit [Dict<string, Tuple<string, string>: #RRGGBB>]
CONST_COLOURS = {'QT_BASIS': ('#56B4E9', '#0072B2'), 'GLS_M4': ('#F0884A', '#B84A00'),
                 'static': ('#B1BAC4', '#57606A'), 'moving': ('#3FCB97', '#00785A'), 'markers': ('#E19BC4', '#A0427A'),
                 'caught': ('#3FCB97', '#00785A'), 'bounced_out': ('#F0C34A', '#8A6300'), 'rim': ('#E19BC4', '#A0427A'),
                 'missed': ('#F0884A', '#B84A00'), 'hit_robot': ('#F0E442', '#7A6F00'), 'hit_stand': ('#F0E442', '#7A6F00'),
                 'late': ('#F0E442', '#7A6F00'), 'no_go_catchable': (CONST_MUTED['dark'], '#57606A'), 'no_go': ('#B1BAC4', CONST_MUTED['light']),
                 'truth': (CONST_FOREGROUND['dark'], CONST_FOREGROUND['light']), 'observed': (CONST_MUTED['dark'], CONST_MUTED['light']),
                 'joint_1': ('#56B4E9', '#0072B2'), 'joint_2': ('#F0884A', '#B84A00'), 'joint_3': ('#3FCB97', '#00785A'),
                 'joint_4': ('#E19BC4', '#A0427A'), 'joint_5': ('#F0C34A', '#8A6300'), 'joint_6': ('#B1BAC4', '#57606A')}
# The contrast floor of a colour that carries meaning against its background (WCAG 2.1).
#   Unit [float]
CONST_CONTRAST_FLOOR = 4.5
# Meaning never by colour alone: a marker per entity and a line style per joint acceleration and per comparison set
# (Plot.Core.CONST_COMPARISON_SETS).
#   Unit [Dict<string, string>], [Dict<int, string>], [Dict<string, string>]
CONST_MARKERS = {'QT_BASIS': 'o', 'GLS_M4': 's', 'static': 'o', 'moving': '^', 'markers': 'D'}
CONST_ACCELERATION_STYLES = {800: '-', 400: '--'}
CONST_SET_STYLES = {'validation': ':', 'held_out': '--', 'real': '-'}
# The line styles of successive prefixes (the smallest N first) and of the six joints.
#   Unit [Tuple<string>]
CONST_N_STYLES = (':', '--', '-')
CONST_JOINT_STYLES = ('-', '--', '-.', ':', '-', '--')
# The line styles of the training seeds (five distinct ones).
#   Unit [Tuple<Any>]
CONST_SEED_STYLES = ('-', '--', '-.', ':', (0, (6, 2, 1, 2, 1, 2)))
# A joint acceleration's unit in a label, set by both text engines.
#   Unit [string]
CONST_DEGREES_S2 = '$^{\\circ}$/s$^2$'
# The resolution of every image, and the sizes of the three kinds of output: a figure of the full output and of the
# README (a page width), and a video figure of 3840 x 2160 px at that resolution with type readable on a screen.
#   Unit [int, dpi], [Dict<string, Tuple<float, float>: in>], [Dict<string, float: pt>]
CONST_DPI = 300
CONST_SIZES = {'page': (7.0, 2.8), 'tall': (7.0, 4.2), 'video': (12.8, 7.2)}
CONST_FONT = {'page': 8.0, 'video': 22.0}
CONST_LINE = {'page': 1.0, 'video': 3.0}
# The programs LaTeX needs for a figure with usetex: latex itself and dvipng for the PNG (matplotlib's usetex).
#   Unit [Tuple<string>]
CONST_LATEX_PROGRAMS = ('latex', 'dvipng')
# The characters LaTeX gives a meaning in text, escaped in a label with usetex.
#   Unit [Dict<string, string>]
CONST_LATEX_ESCAPES = {'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#'}


class Style_Error_Cls(ValueError):
    """
    Description:
        A figure cannot be styled or saved: an unknown theme or entity, or a file that cannot be written. The message
        names the value and what it must be; the script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Look_Str:
    """
    Description:
        The look of one figure (Style_Context): its theme, whether LaTeX sets its text, and its kind of output.
    """

    # The theme, LaTeX on or off, the kind ('page' or 'video') and whether it is the key set's variant (the README's or
    # the video's: a figure may leave a series out there, its full output keeps it).
    #   Unit [string], [bool], [string], [bool]
    Theme: str = CONST_THEMES[0]
    Usetex: bool = False
    Kind: str = 'page'
    Key: bool = False

    def Words(self, page: str, video: str) -> str:
        """
        Description:
            A label in few words for the video (large type on a screen), in full otherwise.

        Args:
            (1) page [string]: The label of a page.
            (2) video [string]: The label of the video.

        Returns:
            (1) parameter [string]: The label, through Text.
        """

        return self.Text(video if self.Kind == 'video' else page)

    def Colour(self, entity: str) -> str:
        """
        Description:
            The entity's colour in this theme.

        Args:
            (1) entity [string]: A key of CONST_COLOURS.

        Returns:
            (1) parameter [string]: The colour, #RRGGBB.
        """

        if entity not in CONST_COLOURS:
            raise Style_Error_Cls('No colour is defined for "%s"; the entities are %s.' % (entity, ', '.join(sorted(CONST_COLOURS))))
        return CONST_COLOURS[entity][CONST_THEMES.index(self.Theme)]

    def Text(self, text: str) -> str:
        """
        Description:
            A label as the figure's text engine takes it: with usetex the characters LaTeX gives a meaning are escaped
            outside $...$ (a name with an underscore, a percent sign); mathtext takes the text as it is.

        Args:
            (1) text [string]: The label.

        Returns:
            (1) parameter [string]: The label for the figure.
        """

        if not self.Usetex:
            return text
        parts = text.split('$')
        for k in range(0, len(parts), 2):
            parts[k] = ''.join(CONST_LATEX_ESCAPES.get(c, c) for c in parts[k])
        return '$'.join(parts)

    def Figure(self, size: str = 'page') -> matplotlib.figure.Figure:
        """
        Description:
            A new figure of this look's size (a video figure is always the video's size), with the constrained layout.

        Args:
            (1) size [string]: 'page' or 'tall'.

        Returns:
            (1) parameter [matplotlib.figure.Figure(object)]: The figure.
        """

        return matplotlib.figure.Figure(figsize=CONST_SIZES['video' if self.Kind == 'video' else size], layout='constrained')

    def Panel(self, axes: tp.Any, letter: str) -> None:
        """
        Description:
            The panel letter (a), (b), ... at the axes' top left, outside the data.

        Args:
            (1) axes [matplotlib.axes.Axes(object)]: The panel.
            (2) letter [string]: Its letter.
        """

        axes.set_title('(%s)' % letter, loc='left', fontsize=matplotlib.rcParams['font.size'], pad=3)


def Contrast_Ratio(first: str, second: str) -> float:
    """
    Description:
        The WCAG 2.1 contrast ratio of two colours: (L1 + 0.05) / (L2 + 0.05) of their relative luminances, the lighter
        first.

    Args:
        (1) first [string]: A colour, #RRGGBB.
        (2) second [string]: Another.

    Returns:
        (1) parameter [float]: The ratio, 1 to 21.
    """

    def Luminance(colour: str) -> float:
        """
        Description:
            The relative luminance of an sRGB colour.

        Args:
            (1) colour [string]: #RRGGBB.

        Returns:
            (1) parameter [float]: The luminance, 0 to 1.
        """

        channels = [int(colour.lstrip('#')[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((Luminance(first), Luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def Latex_Available() -> bool:
    """
    Description:
        Whether LaTeX can set the figures' text: latex and dvipng on the PATH (matplotlib's usetex needs both for a PNG).

    Returns:
        (1) parameter [bool]: Both found.
    """

    return all(shutil.which(program) is not None for program in CONST_LATEX_PROGRAMS)


def Style_Parameters(theme: str, usetex: bool, kind: str = 'page') -> tp.Dict[str, tp.Any]:
    """
    Description:
        matplotlib's parameters of one look: SciencePlots' 'science' (with its 'no-latex' without LaTeX), then the theme's
        colours, the kind's type and line sizes, and what makes a saved file the same on every run (no date in a PDF).

    Args:
        (1) theme [string]: One of CONST_THEMES.
        (2) usetex [bool]: LaTeX sets the text.
        (3) kind [string]: 'page' or 'video'.

    Returns:
        (1) parameter [Dict<string, Any>]: The parameters.
    """

    if theme not in CONST_THEMES or kind not in CONST_FONT:
        raise Style_Error_Cls('The theme is one of %s and the kind one of %s, found %r and %r.' % (', '.join(CONST_THEMES), ', '.join(CONST_FONT),
                                                                                                theme, kind))
    library = matplotlib.style.library
    parameters = dict(library['science'])
    if not usetex:
        parameters.update(library['no-latex'])
    background, foreground, muted = CONST_BACKGROUND[theme], CONST_FOREGROUND[theme], CONST_MUTED[theme]
    font, line = CONST_FONT[kind], CONST_LINE[kind]
    parameters.update({'figure.facecolor': background, 'axes.facecolor': background, 'savefig.facecolor': background,
                       'savefig.edgecolor': background, 'text.color': foreground, 'axes.labelcolor': foreground,
                       'axes.edgecolor': foreground, 'xtick.color': foreground, 'ytick.color': foreground, 'grid.color': muted,
                       'legend.labelcolor': foreground, 'font.size': font, 'axes.labelsize': font, 'legend.fontsize': 0.9 * font,
                       'xtick.labelsize': 0.9 * font, 'ytick.labelsize': 0.9 * font, 'lines.linewidth': line, 'lines.markersize': 3.0 * line,
                       'axes.linewidth': 0.5 * line, 'errorbar.capsize': 1.5 * line, 'xtick.major.width': 0.5 * line, 'ytick.major.width': 0.5 * line,
                       'xtick.minor.width': 0.5 * line, 'ytick.minor.width': 0.5 * line, 'pdf.fonttype': 42,
                       'savefig.bbox': None if kind == 'video' else 'tight', 'figure.dpi': CONST_DPI, 'savefig.dpi': CONST_DPI})
    return parameters


@contextlib.contextmanager
def Style_Context(theme: str, usetex: bool, kind: str = 'page', key: bool = False) -> tp.Iterator[Look_Str]:
    """
    Description:
        The context in which one figure is drawn and saved: matplotlib's parameters of the look (Style_Parameters) are
        in force inside it and restored after it, so that no figure changes another's style.

    Args:
        (1) theme [string]: One of CONST_THEMES.
        (2) usetex [bool]: LaTeX sets the text.
        (3) kind [string]: 'page' or 'video'.
        (4) key [bool]: The key set's variant.

    Returns:
        (1) parameter [Iterator<Look_Str>]: The look, once.
    """

    with matplotlib.rc_context(Style_Parameters(theme, usetex, kind)):
        yield Look_Str(Theme=theme, Usetex=usetex, Kind=kind, Key=key)


def Save_Figure(figure: matplotlib.figure.Figure, path: str, formats: tp.Sequence[str]) -> tp.List[str]:
    """
    Description:
        Save a figure as each format (pdf, png) at CONST_DPI, inside its Style_Context, each through a temporary file
        beside it renamed at the end (an existing file of that name is replaced, a failed write leaves none); a PDF
        carries no creation date, so the same figure gives the same file.

    Args:
        (1) figure [matplotlib.figure.Figure(object)]: The figure.
        (2) path [string]: The file's path without its extension.
        (3) formats [Sequence<string>]: The formats, 'pdf' and 'png'.

    Returns:
        (1) parameter [List<string>]: The files written.
    """

    written = []
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for extension in formats:
        if extension not in ('pdf', 'png'):
            raise Style_Error_Cls('A figure is saved as pdf or png, found %r.' % extension)
        target = '%s.%s' % (path, extension)
        temporary = '%s.part.%s' % (path, extension)
        metadata = {'CreationDate': None} if extension == 'pdf' else {}
        try:
            figure.savefig(temporary, format=extension, dpi=CONST_DPI, metadata=metadata)
            os.replace(temporary, target)
        except BaseException:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise
        written.append(target)
    return written
