#!/usr/bin/env bash
# Explicit migration/install gate. Run from the operator's trusted checkout.
# Existing data is kept; no credentials are copied from an old runtime checkout.
set -euo pipefail
shopt -s nullglob dotglob
fail() { echo "error: $*" >&2; exit 1; }
[ "$(id -u)" -eq 0 ] || fail "run as root"
operator="${1:-operator}"
[[ "$operator" =~ ^[a-z][a-z0-9_-]{0,30}$ ]] || fail "invalid operator account"
case "$operator" in root|agent) fail "operator must be separate from root and agent" ;; esac
[ "$#" -le 1 ] || fail "usage: $0 [OPERATOR]"
repo=/opt/pilotage-agent
state=/home/agent/.pilotage-agent
script_root="$(cd "$(dirname "$0")/.." && pwd -P)"
[ "$script_root" = "$repo" ] || fail "use a fresh operator-owned checkout at $repo"
operator_uid="$(id -u "$operator")"
agent_uid="$(id -u agent)"
[ "$operator_uid" -ne 0 ] && [ "$agent_uid" -ne 0 ] || fail "accounts cannot have UID 0"
[ "$operator_uid" -ne "$agent_uid" ] || fail "accounts must have different UIDs"
[ "$(stat -c %U "$repo")" = "$operator" ] || fail "checkout must belong to $operator"
[ ! -L "$repo" ] && [ ! -L /home/agent ] && [ ! -L "$state" ] || fail "deployment roots cannot be links"
for parent in / /opt /home /etc /usr /usr/local /usr/local/bin /etc/systemd /etc/systemd/system; do
  [ -d "$parent" ] && [ ! -L "$parent" ] && [ "$(stat -c %U "$parent")" = root ] \
    || fail "deployment parent must be a root-owned real directory: $parent"
  mode="$(stat -c %a "$parent")"
  (( (8#$mode & 0022) == 0 )) || fail "deployment parent is writable by other accounts: $parent"
done
for group in $(id -nG agent); do
  [ "$group" = agent ] || fail "agent must belong only to its unprivileged group"
done
if sudo -l -U agent >/dev/null 2>&1; then fail "agent still has sudo permissions"; fi
id -nG "$operator" | tr ' ' '\n' | grep -qx agent || fail "run setup-accounts.sh first"
id -nG "$operator" | tr ' ' '\n' | grep -qx sudo || fail "operator needs sudo"
[ -x "$repo/.venv/bin/python" ] || fail "install locked dependencies as the operator first"
[ -d "$repo/bridge/node_modules" ] || fail "install locked dependencies as the operator first"
[ ! -e "$repo/.env" ] && [ ! -L "$repo/.env" ] || fail "keep runtime secrets in the profile, not the checkout"
operator_home="$(getent passwd "$operator" | cut -d: -f6)"
[ "$operator_home" = "/home/$operator" ] && [ ! -L "$operator_home" ] \
  && [ "$(stat -c %U "$operator_home")" = "$operator" ] || fail "invalid operator home"
[ "$(stat -c %a "$operator_home")" = 700 ] || fail "operator home must be private (chmod 700)"
# Do not take ownership while an untrusted process can race the filesystem work.
if pgrep -u "$agent_uid" >/dev/null; then
  fail "stop the agent and terminate its login sessions first (loginctl terminate-user agent); then retry"
fi
for credential in /home/agent/.git-credentials /home/agent/.ssh; do
  if [ -e "$credential" ] || [ -L "$credential" ]; then
    fail "review and relocate old deployment credentials at $credential before protection"
  fi
done
for directory in .config .config/systemd .config/systemd/user .config/systemd/user/default.target.wants; do
  [ ! -L "/home/agent/$directory" ] || fail "legacy service path cannot contain links"
done
if [ -e /etc/pilotage-agent.json ]; then
  [ ! -L /etc/pilotage-agent.json ] || fail "manifest cannot be a link"
  existing_operator="$(python3 -c 'import json; print(json.load(open("/etc/pilotage-agent.json"))["operator"])')"
  [ "$existing_operator" = "$operator" ] || fail "existing deployment has a different operator"
fi

profiles=("$state")
if [ -e "$state/profiles" ] || [ -L "$state/profiles" ]; then
  [ -d "$state/profiles" ] && [ ! -L "$state/profiles" ] || fail "profiles root must be a real directory"
  for profile in "$state/profiles"/*; do
    [ -e "$profile" ] || [ -L "$profile" ] || continue
    [[ "$(basename "$profile")" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || fail "invalid profile directory"
    case "$(basename "$profile")" in default|pilotage|test|tmp|root|sudo) fail "reserved profile name" ;; esac
    [ -d "$profile" ] && [ ! -L "$profile" ] || fail "profile cannot be a link"
    profiles+=("$profile")
  done
fi
# Check every input before changing ownership, including files whose inode is
# retained (locks must never be replaced during a migration).
for profile in "${profiles[@]}"; do
  name=default
  [ "$profile" = "$state" ] || name="$(basename "$profile")"
  service_state="$(systemctl show "pilotage-agent@$name.service" --property=LoadState,ActiveState 2>/dev/null || true)"
  if ! grep -qx 'LoadState=not-found' <<< "$service_state"; then
    grep -qx 'LoadState=loaded' <<< "$service_state" \
      && grep -Eq '^ActiveState=(inactive|failed)$' <<< "$service_state" \
      || fail "stop pilotage-agent@$name.service before migration (or check systemd)"
  fi
  for name in config.yaml .env SOUL.md .runtime.lock; do
    file="$profile/$name"
    if [ -e "$file" ] || [ -L "$file" ]; then
      [ -f "$file" ] && [ ! -L "$file" ] && [ "$(stat -c %h "$file")" -eq 1 ] \
        || fail "protected file must be regular and have one link: $file"
    fi
  done
done
unexpected="$(find "$repo" -xdev ! -type l ! -user "$operator" -print -quit)"
[ -z "$unexpected" ] || fail "checkout contains files not owned by $operator: $unexpected"
checkout_status="$(runuser -u "$operator" -- git -C "$repo" status --porcelain --untracked-files=normal)"
[ -z "$checkout_status" ] || fail "checkout has local edits; use a clean trusted checkout"
systemd-analyze verify "$repo/scripts/pilotage-agent@.service"
for profile in "${profiles[@]}"; do
  "$repo/.venv/bin/python" -I -B -c '
import sys
from pathlib import Path
from pilotage.deployment import validate_policy_paths
from pilotage.env import read_env_values
profile = Path(sys.argv[1])
env_file = profile / ".env"
if env_file.exists():
    validate_policy_paths(profile, read_env_values(env_file))
' "$profile"
done

# A root-owned sticky, setgid directory lets the agent create database journals
# and working data, but it cannot unlink/rename operator-owned configuration.
# Protect /home/agent too, otherwise it could replace the entire state directory.
chown root:root /home/agent
chmod 755 /home/agent
for directory in .cache .local .config .npm workspace; do
  if [ ! -e "/home/agent/$directory" ]; then
    install -d -o agent -g agent -m 0700 "/home/agent/$directory"
  fi
done
for profile in "${profiles[@]}"; do
  install -d -o root -g agent -m 3770 "$profile"
  if [ ! -e "$profile/config.yaml" ]; then
    install -o "$operator" -g agent -m 0640 "$repo/config.yaml.example" "$profile/config.yaml"
  fi
  if [ ! -e "$profile/.env" ]; then
    install -o "$operator" -g agent -m 0640 "$repo/.env.example" "$profile/.env"
  fi
  for name in SOUL.md .runtime.lock; do
    if [ ! -e "$profile/$name" ]; then
      install -o "$operator" -g agent -m 0640 /dev/null "$profile/$name"
    fi
  done
  chown "$operator":agent "$profile/config.yaml" "$profile/.env" "$profile/SOUL.md" "$profile/.runtime.lock"
  chmod 640 "$profile/config.yaml" "$profile/.env" "$profile/SOUL.md"
  chmod 660 "$profile/.runtime.lock"
done
install -d -o "$operator" -g agent -m 2750 "$state/profiles"
runuser -u "$operator" -- "$repo/.venv/bin/python" -I -B -c '
from pilotage.deployment import CHECKOUT, make_runtime_readable
make_runtime_readable(CHECKOUT)
'
chmod 700 "$repo/.git"

runuser -u agent -- "$repo/.venv/bin/python" -I -B -c 'import pilotage.main'

umask 022
manifest="$(mktemp /etc/.pilotage-agent.XXXXXX)"
printf '{"operator":"%s"}\n' "$operator" > "$manifest"
chmod 644 "$manifest"
mv -T "$manifest" /etc/pilotage-agent.json
install -o root -g root -m 0644 "$repo/scripts/pilotage-agent@.service" /etc/systemd/system/pilotage-agent@.service
install -o root -g root -m 0755 "$repo/scripts/pilotage-launcher" /usr/local/bin/pilotage
# A legacy user unit must not restart alongside the system unit on next boot.
loginctl disable-linger agent
if [ -d /home/agent/.config/systemd/user/default.target.wants ]; then
  for unit in /home/agent/.config/systemd/user/default.target.wants/pilotage-agent@*.service; do
    [ -L "$unit" ] || continue
    unlink "$unit"
  done
fi
systemctl daemon-reload
for profile in "${profiles[@]}"; do
  name=default
  [ "$profile" = "$state" ] || name="$(basename "$profile")"
  systemctl enable "pilotage-agent@$name.service"
done
echo "Protection installed. Services remain stopped."
echo "As $operator: configure/login if needed, then pilotage restart and pilotage doctor."
