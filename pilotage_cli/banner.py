"""ANSI printing helpers and the CLI version label.

Pure display functions with no PilotageCLI state dependency.
"""
import logging

from pilotage_cli import __version__ as VERSION, __release_date__ as RELEASE_DATE

logger = logging.getLogger(__name__)


# =========================================================================
# ANSI building blocks for conversation display
# =========================================================================

_GOLD = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's renderer."""
    from prompt_toolkit import print_formatted_text as _pt_print
    from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
    try:
        _pt_print(_PT_ANSI(text))
    except Exception:
        # prompt_toolkit needs a real console. On Windows, a redirected or
        # absent stdout (pythonw.exe, CI, `pilotage ... > file`) raises
        # NoConsoleScreenBufferError from its Win32Output — display helpers
        # must never crash the caller over that, so degrade to plain print.
        print(text)


def format_banner_version_label() -> str:
    """Return the version label shown in the startup banner title."""
    return f"Pilotage Agent v{VERSION} ({RELEASE_DATE})"
