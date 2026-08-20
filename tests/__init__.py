"""Tests. Standard library only — `python -m unittest discover -s tests`.

The suite runs against a state directory of its own. Configuration is read
from a file now, and a suite that reads the operator's real one passes or
fails depending on whose machine it is running on.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_STATE_DIR = tempfile.mkdtemp(prefix="pilotage-tests-")
atexit.register(shutil.rmtree, _STATE_DIR, True)

os.environ["PILOTAGE_HOME"] = _STATE_DIR
# An operator's own override must not follow the suite in either.
os.environ.pop("PILOTAGE_CONFIG", None)
