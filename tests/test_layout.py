"""Layout invariants that are arithmetic, not taste.

The contents rail is `position: fixed` at a computed left offset, and the reading
column is pushed right by a padding. Those two numbers are set in different rules
tens of lines apart in base.css, so nothing stops them drifting into each other.

They did: the rail was 190px wide starting at 24px, ending at exactly 214px, and
`#report-body` had `padding-left: 214px`. A zero gutter, the rail touching the
text. It looked like a spacing opinion and was actually two constants that had
silently met.

This asserts the gap rather than the constants, so either may be tuned freely and
only an actual collision fails.
"""
import os
import re

import pytest

CSS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'frontend', 'src', 'styles', 'base.css')

# The wrap's horizontal padding is --s5, which is where the rail's left offset
# comes from in placeRail(). Kept as a named constant so the arithmetic below
# reads the same way the layout does.
RAIL_LEFT = 24

MIN_GUTTER = 32


@pytest.fixture(scope='module')
def css():
    with open(CSS, encoding='utf-8') as f:
        return f.read()


def _desktop_block(css):
    """The @media (min-width: 1180px) block that owns both numbers."""
    i = css.index('@media (min-width: 1180px)')
    depth, j = 0, css.index('{', i)
    for k in range(j, len(css)):
        if css[k] == '{':
            depth += 1
        elif css[k] == '}':
            depth -= 1
            if depth == 0:
                return css[i:k + 1]
    raise AssertionError('unterminated media block')


def test_rail_does_not_touch_the_reading_column(css):
    block = _desktop_block(css)

    width = re.search(r'\.toc\s*\{[^}]*?width:\s*(\d+)px', block, re.S)
    padding = re.search(r'#report-body\s*\{[^}]*?padding-left:\s*(\d+)px', block, re.S)
    assert width and padding, 'rail width or column padding not found in base.css'

    rail_right = RAIL_LEFT + int(width.group(1))
    column_left = int(padding.group(1))
    gutter = column_left - rail_right

    assert gutter >= MIN_GUTTER, (
        f'contents rail ends at {rail_right}px and the column starts at '
        f'{column_left}px, leaving a {gutter}px gutter. This is the bug that '
        f'made the rail read as glued to the text; keep at least {MIN_GUTTER}px.'
    )


def test_gutter_hairline_sits_inside_the_gutter(css):
    """The divider is offset from the rail's right edge and must not land on
    either the rail or the text."""
    block = _desktop_block(css)

    width = int(re.search(r'\.toc\s*\{[^}]*?width:\s*(\d+)px', block, re.S).group(1))
    padding = int(re.search(r'#report-body\s*\{[^}]*?padding-left:\s*(\d+)px',
                            block, re.S).group(1))
    offset = re.search(r'\.toc::after\s*\{[^}]*?margin-left:\s*(\d+)px', block, re.S)
    assert offset, 'gutter hairline not found'

    rail_right = RAIL_LEFT + width
    hairline = rail_right + int(offset.group(1))

    assert rail_right < hairline < padding, (
        f'hairline at {hairline}px is not strictly between the rail edge '
        f'({rail_right}px) and the column ({padding}px)'
    )
