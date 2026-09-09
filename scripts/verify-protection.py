"""Exercise the installed Linux permission boundary using disposable probes.

Run as root after install-protection.sh. No real configuration is edited and
no service is started or stopped. Probe children run with the agent's UID.
"""

import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys
import tempfile

from pilotage.deployment import validate_policy_paths
from pilotage.env import read_env_values


def main():
    if sys.platform != "linux" or os.geteuid() != 0:
        raise SystemExit("Run this verification as root inside the Ubuntu LXC.")
    agent = pwd.getpwnam("agent")
    manifest = Path("/etc/pilotage-agent.json")
    info = manifest.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022
    operator = pwd.getpwnam(json.loads(manifest.read_text())["operator"])
    assert agent.pw_uid != operator.pw_uid and agent.pw_uid != 0
    assert os.getgrouplist("agent", agent.pw_gid) == [agent.pw_gid]
    subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).with_name("verify-agent-sudo.py"))],
        check=True,
    )

    def run(code, *arguments):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, *map(str, arguments)],
            user=agent.pw_uid, group=agent.pw_gid, extra_groups=[],
            cwd="/", capture_output=True, text=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)

    home = Path("/home/agent")
    state = home / ".pilotage-agent"
    assert home.stat().st_uid == 0 and not home.stat().st_mode & 0o022
    profile_root = state / "profiles"
    assert profile_root.stat().st_uid == operator.pw_uid
    assert profile_root.stat().st_gid == agent.pw_gid
    assert stat.S_IMODE(profile_root.stat().st_mode) == 0o2750
    profiles = [state, *sorted(profile_root.iterdir())]
    for profile in profiles:
        info = profile.lstat()
        assert stat.S_ISDIR(info.st_mode) and info.st_uid in {0, operator.pw_uid}
        assert info.st_gid == agent.pw_gid
        assert stat.S_IMODE(info.st_mode) == 0o3770
        for name in ("config.yaml", ".env", "SOUL.md", ".runtime.lock"):
            target = profile / name
            info = target.lstat()
            assert stat.S_ISREG(info.st_mode) and info.st_uid == operator.pw_uid and info.st_nlink == 1
            assert stat.S_IMODE(info.st_mode) == (0o660 if name == ".runtime.lock" else 0o640)
            assert info.st_gid == agent.pw_gid
        validate_policy_paths(profile, read_env_values(profile / ".env"))
        # Opening the shared lock must work even with fs.protected_regular=2.
        run("import os,sys; os.close(os.open(sys.argv[1], os.O_RDWR|os.O_NOFOLLOW))", profile / ".runtime.lock")
        fd, written = tempfile.mkstemp(prefix=".protection-probe-", dir=profile)
        probe = Path(written)
        os.fchown(fd, operator.pw_uid, agent.pw_gid)
        os.fchmod(fd, 0o640)
        os.close(fd)
        try:
            run('''
import os, pathlib, sys
p = pathlib.Path(sys.argv[1])
for action in (lambda: p.write_text("changed"), lambda: p.unlink(),
               lambda: p.rename(p.with_suffix(".renamed")),
               lambda: p.parent.chmod(0o777)):
    try:
        action()
    except PermissionError:
        continue
    raise AssertionError("Agent changed protected configuration or its parent")
assert p.read_text() == ""
''', probe)
        finally:
            probe.unlink(missing_ok=True)
            probe.with_suffix(".renamed").unlink(missing_ok=True)
        run('''
import pathlib, sqlite3, sys, tempfile
root = pathlib.Path(sys.argv[1])
for name in ("workspace", "memories", "skills", "cron"):
    folder = root / name
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".protection-check-", dir=folder) as temporary:
        p = pathlib.Path(temporary) / "probe"
        p.write_text("one")
        p.write_text("two")
        assert p.read_text() == "two"
        p.unlink()
with tempfile.TemporaryDirectory(prefix=".database-check-", dir=root) as temporary:
    with sqlite3.connect(str(pathlib.Path(temporary) / "probe.db")) as db:
        db.execute("pragma journal_mode=wal")
        db.execute("create table probe(value)")
        db.execute("insert into probe values (1)")
''', profile)

    run('''
import os, pathlib, sys
repo = pathlib.Path("/opt/pilotage-agent")
for p in (repo, repo / "pilotage", repo / "pilotage/agent.py", repo / ".venv",
          repo / "scripts/install.sh", pathlib.Path("/etc/systemd/system/pilotage-agent@.service"),
          pathlib.Path("/usr/local/bin/pilotage"), pathlib.Path("/etc/pilotage-agent.json")):
    assert not os.access(p, os.W_OK), f"Agent can modify {p}"
assert not os.access(repo / ".git", os.R_OK)
assert not os.access(sys.argv[1], os.R_OK | os.X_OK)
''', operator.pw_dir)
    print("Protection verified: runtime/settings protected; agent data remains writable.")


if __name__ == "__main__":
    main()
