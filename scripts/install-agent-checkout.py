"""Install one live asset checkout in the protected agent home.

Run with sudo after bootstrap, before configuring the agent. Git authentication
runs as the operator. Only versioned assets are installed; runtime state is kept.
"""

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from pilotage.deployment import CHECKOUT, STATE, load


STATE_FILES = {"config.yaml", "SOUL.md", ".env.example"}
ROOT_FILES = {"README.md", ".gitignore", ".gitattributes"}


def asset_paths(source: Path, home: Path) -> list[Path]:
    """Validate the entire overlay before replacing any bootstrap defaults."""
    if (home / ".git").exists() or (home / ".git").is_symlink():
        raise ValueError("A live checkout already exists; review it with git status.")
    paths = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative.parts[0] == ".git":
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise ValueError(f"Asset must be a regular file or directory: {relative}")
        parts = relative.parts
        allowed = (
            (len(parts) == 1 and parts[0] in ROOT_FILES and path.is_file())
            or parts[0] == ".resources"
            or (parts[0] == ".pilotage-agent" and (
                len(parts) == 1
                or parts[1] == "skills"
                or (len(parts) == 2 and parts[1] in STATE_FILES and path.is_file())
            ))
        )
        if not allowed:
            raise ValueError(f"Repository contains a non-asset path: {relative}")
        target = home / relative
        # Check every existing component, including dangling links.
        for parent in [target, *target.parents]:
            if parent == home:
                break
            if parent.is_symlink():
                raise ValueError(f"Destination cannot contain links: {relative}")
        if target.exists():
            if path.is_dir():
                if not target.is_dir():
                    raise ValueError(f"Destination is not a directory: {relative}")
            elif relative.as_posix() not in {
                ".pilotage-agent/config.yaml", ".pilotage-agent/SOUL.md"
            } or not target.is_file() or target.stat().st_nlink != 1:
                raise ValueError(f"Refusing to overwrite existing data: {relative}")
        paths.append(relative)
    for required in (".pilotage-agent/config.yaml", ".pilotage-agent/SOUL.md", ".gitignore"):
        if Path(required) not in paths or not (source / required).is_file():
            raise ValueError(f"Repository is missing {required}")
    return paths


def install_assets(source, home, paths, operator_uid, agent_gid):
    for relative in paths:
        origin, target = source / relative, home / relative
        if origin.is_dir():
            if target.exists():
                continue  # Preserve the protected state and existing data directories.
            target.mkdir()
            mode = 0o750
        else:
            if target.exists():
                # Linux protected_regular rejects O_CREAT on another user's
                # file in this sticky directory, even for root. Open the
                # validated bootstrap setting without recreating its inode.
                with origin.open("rb") as source_file, target.open("r+b") as target_file:
                    shutil.copyfileobj(source_file, target_file)
                    target_file.truncate()
            else:
                shutil.copyfile(origin, target)
            mode = 0o750 if origin.stat().st_mode & 0o111 else 0o640
        os.chown(target, operator_uid, agent_gid)
        target.chmod(mode)


def share_skills(skills, operator_uid, agent_uid):
    """Keep new settings read-only to agent; share new and existing skills."""
    # Git replaces files on pull. The state directory's setgid group is agent, so an
    # operator umask of 0002 alone would make replacement settings writable.
    subprocess.run([
        "setfacl", "-d", "--set", "u::rwx,g::r-x,o::---", str(skills.parent),
    ], check=True)
    skills.mkdir(exist_ok=True)
    access = f"u:{operator_uid}:rwX,u:{agent_uid}:rwX,m::rwX,o::---"
    subprocess.run(["setfacl", "-R", "-P", "-m", access, str(skills)], check=True)
    default = f"u::rwx,u:{operator_uid}:rwx,u:{agent_uid}:rwx,g::---,m::rwx,o::---"
    for path in [skills, *skills.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            subprocess.run(["setfacl", "-d", "-m", default, str(path)], check=True)


def main():
    if sys.platform != "linux" or os.geteuid() != 0:
        raise SystemExit("Run with sudo inside the protected Ubuntu container.")
    if len(sys.argv) != 2:
        raise SystemExit("Usage: install-agent-checkout.py HTTPS_REPOSITORY_URL")
    url = urlsplit(sys.argv[1])
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise SystemExit("Use an HTTPS repository URL without embedded credentials.")
    import pwd

    managed = load()
    if managed is None:
        raise SystemExit("Install the protected deployment first.")
    operator = pwd.getpwnam(managed.operator)
    agent = pwd.getpwnam("agent")
    home = STATE.parent
    if home.is_symlink() or home.stat().st_uid != operator.pw_uid:
        raise SystemExit("Agent home must belong to the operator; use the current bootstrap.")
    if shutil.which("setfacl") is None:
        raise SystemExit("Install the acl system dependency first.")
    # This also validates the agent state, settings and lack of agent sudo access.
    subprocess.run([sys.executable, "-I", "-B", str(CHECKOUT / "scripts/verify-protection.py")], check=True)
    result = subprocess.run(
        ["systemctl", "show", "pilotage-agent.service", "--property=ActiveState", "--value"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() not in {"inactive", "failed"}:
        raise SystemExit("Stop pilotage-agent.service first.")
    processes = subprocess.run(["pgrep", "-u", str(agent.pw_uid)], capture_output=True)
    if processes.returncode != 1:
        raise SystemExit("Stop agent processes and login sessions before installing assets.")
    if (home / ".git").exists() or (home / ".git").is_symlink():
        raise SystemExit("A live checkout already exists; review it with git status.")
    git = ["sudo", "-H", "-u", operator.pw_name, "--", "git"]
    # Temporary staging is removed after installation; /home/agent is the only
    # retained checkout. Never clone or execute Git hooks as root or as agent.
    with tempfile.TemporaryDirectory(prefix="pilotage-assets-", dir=operator.pw_dir) as staging:
        stage = Path(staging)
        os.chown(stage, operator.pw_uid, operator.pw_gid)
        source = stage / "checkout"
        subprocess.run([*git, "clone", "--", sys.argv[1], str(source)], cwd=operator.pw_dir, check=True)
        paths = asset_paths(source, home)
        install_assets(source, home, paths, operator.pw_uid, agent.pw_gid)
        share_skills(STATE / "skills", operator.pw_uid, agent.pw_uid)
        metadata = home / ".git"
        shutil.move(str(source / ".git"), metadata)
        metadata.chmod(0o700)
        # Protect local runtime files even if an agent repository omits ignores.
        exclude = metadata / "info/exclude"
        with exclude.open("a") as handle:
            handle.write("\n# Local container state\n/.bash*\n/.profile\n/.ssh/\n/.git-credentials\n"
                         "/.gitconfig\n/.cache/\n/.config/\n/.local/\n/.npm/\n/workspace/\n"
                         "/.pilotage-agent/*\n!/.pilotage-agent/config.yaml\n"
                         "!/.pilotage-agent/SOUL.md\n!/.pilotage-agent/.env.example\n"
                         "!/.pilotage-agent/skills/\n")
        subprocess.run([*git, "-C", str(home), "status", "--short"], cwd=operator.pw_dir, check=True)
    subprocess.run([sys.executable, "-I", "-B", str(CHECKOUT / "scripts/verify-protection.py")], check=True)
    print("Live agent checkout installed at /home/agent. Services remain stopped.")
    print("As operator: cd /home/agent; git status")


if __name__ == "__main__":
    main()
