"""One matplotlib theme, matched to the site palette.

Every generated figure was drawn on matplotlib's default white. On a dark page
that is four glaring white slabs, and it is the single most obvious sign that the
figures were produced by a script rather than designed alongside the page.

Import and call `apply()` before plotting. Colours here mirror
frontend/src/styles/tokens.css; if that file changes, change this with it.
"""
import matplotlib as mpl

BG = '#16232f'          # --bg-panel
FG = '#e4e8ec'          # --text-body
MUTED = '#a3b3c0'       # --text-muted
FAINT = '#8496a5'       # --text-faint
GRID = '#25384a'
AMBER = '#f0a868'
BLUE = '#66c0f4'
VIOLET = '#9b7fe8'
TEAL = '#4ecdc4'
GREEN = '#a1cd44'
PINK = '#e089c4'
SLATE = '#8496a5'

# Same order and meaning as the --c-* tokens, so a model keeps its colour
# whether it is drawn in Python or in the browser.
SERIES = [AMBER, BLUE, VIOLET, TEAL, GREEN, SLATE, PINK]


def apply():
    mpl.rcParams.update({
        'figure.facecolor': BG,
        'savefig.facecolor': BG,
        'axes.facecolor': BG,
        'savefig.bbox': 'tight',
        'savefig.dpi': 140,

        'text.color': FG,
        'axes.labelcolor': MUTED,
        'axes.titlecolor': FG,
        'xtick.color': FAINT,
        'ytick.color': FAINT,

        'axes.edgecolor': GRID,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.alpha': 0.55,
        'grid.linewidth': 0.7,

        # Chartjunk off: a boxed plot reads as a spreadsheet, not a figure.
        'axes.spines.top': False,
        'axes.spines.right': False,

        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'DejaVu Sans'],
        'font.size': 10.5,
        'axes.titlesize': 12.5,
        'axes.titleweight': 'medium',
        'axes.labelsize': 10.5,
        'legend.frameon': False,
        'legend.labelcolor': MUTED,

        'axes.prop_cycle': mpl.cycler(color=SERIES),
        'patch.edgecolor': BG,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
    })
