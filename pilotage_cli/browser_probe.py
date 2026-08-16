"""Local browser readiness probes for setup/status surfaces.

Mirrors the local-CLI tail of :func:`tools.browser_tool.check_browser_requirements`
so the setup, status and picker surfaces can't diverge from what the browser
tools actually find at runtime.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def has_agent_browser() -> bool:
    """Return True when the ``agent-browser`` CLI resolves on this machine."""
    from pilotage_constants import agent_browser_runnable

    # agent-browser resolves lazily via npx for most installs, which a bare
    # PATH + node_modules probe can't see; validate=False keeps this a cheap
    # existence check with no subprocess spawn.
    try:
        from tools.browser_tool import (
            _find_agent_browser,
            _requires_real_termux_browser_install,
        )
    except Exception:
        # Fall back to binary presence when the runtime probe can't be
        # imported, rather than crashing the setup/status surface.
        if agent_browser_runnable(shutil.which("agent-browser")):
            return True

        # Pilotage-managed Node dirs are prepended to PATH at runtime but are
        # usually absent from the *probe* process's PATH.
        from pilotage_constants import with_pilotage_node_path

        managed_path = with_pilotage_node_path().get("PATH", "")
        if managed_path:
            managed_hit = shutil.which("agent-browser", path=managed_path)
            if managed_hit and agent_browser_runnable(managed_hit):
                return True

        # Local node_modules/.bin — resolve via PATHEXT-aware ``shutil.which``
        # so Windows picks the executable ``.cmd`` shim.
        local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
        if local_bin_dir.is_dir():
            local_which = shutil.which("agent-browser", path=str(local_bin_dir))
            if local_which and agent_browser_runnable(local_which):
                return True
        return False

    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False
    # On Termux the bare npx fallback is too fragile to advertise as ready.
    if _requires_real_termux_browser_install(browser_cmd):
        return False
    return True


def local_browser_runnable() -> bool:
    """Return True when the *local* browser backend would actually start.

    The CLI being present is necessary but not sufficient: agent-browser also
    needs a Chromium build on disk (without one it hangs on first use until the
    command timeout fires), unless the Lightpanda engine is selected — text-only
    navigation needs no Chromium. Cloud providers host their own Chromium and
    therefore gate on :func:`has_agent_browser` alone.
    """
    if not has_agent_browser():
        return False
    try:
        from tools.browser_tool import _chromium_installed, _using_lightpanda_engine
    except Exception:
        return True
    if _using_lightpanda_engine():
        return True
    return _chromium_installed()
