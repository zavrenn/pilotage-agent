#!/usr/bin/env bash
# Explicit protection install gate. Run from the operator's trusted checkout.
# Existing data is kept; no credentials are copied from an old runtime checkout.
set -euo pipefail
export PATH="/opt/pilotage-python/bin:$PATH"
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
python3 -I -B "$repo/scripts/verify-agent-sudo.py"
id -nG "$operator" | tr ' ' '\n' | grep -qx agent || fail "run setup-accounts.sh first"
id -nG "$operator" | tr ' ' '\n' | grep -qx sudo || fail "operator needs sudo"
[ -x "$repo/.venv/bin/python" ] || fail "install locked dependencies as the operator first"
[ -d "$repo/bridge/node_modules" ] || fail "install locked dependencies as the operator first"
[ ! -e "$repo/.env" ] && [ ! -L "$repo/.env" ] || fail "keep runtime secrets in the agent state, not the checkout"
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

# Check every input before changing ownership, including files whose inode is
# retained (locks must never be replaced during provisioning).
service_state="$(systemctl show "pilotage-agent.service" --property=LoadState,ActiveState 2>/dev/null || true)"
if ! grep -qx 'LoadState=not-found' <<< "$service_state"; then
  grep -qx 'LoadState=loaded' <<< "$service_state" \
    && grep -Eq '^ActiveState=(inactive|failed)$' <<< "$service_state" \
    || fail "stop pilotage-agent.service before protection (or check systemd)"
fi
for name in config.yaml .env SOUL.md .runtime.lock; do
  file="$state/$name"
  if [ -e "$file" ] || [ -L "$file" ]; then
    [ -f "$file" ] && [ ! -L "$file" ] && [ "$(stat -c %h "$file")" -eq 1 ] \
      || fail "protected file must be regular and have one link: $file"
  fi
done
unexpected="$(find "$repo" -xdev ! -type l ! -user "$operator" -print -quit)"
[ -z "$unexpected" ] || fail "checkout contains files not owned by $operator: $unexpected"
checkout_status="$(runuser -u "$operator" -- git -C "$repo" status --porcelain --untracked-files=normal)"
[ -z "$checkout_status" ] || fail "checkout has local edits; use a clean trusted checkout"
systemd-analyze verify "$repo/scripts/pilotage-agent.service"
"$repo/.venv/bin/python" -I -B -c '
import sys
from pathlib import Path
from pilotage.deployment import validate_policy_paths
from pilotage.env import read_env_values
state = Path(sys.argv[1])
env_file = state / ".env"
if env_file.exists():
    validate_policy_paths(state, read_env_values(env_file))
' "$state"

# A root-owned sticky, setgid directory lets the agent create database journals
# and working data, but it cannot unlink/rename operator-owned configuration.
# The operator owns /home/agent for the live asset checkout; agent cannot
# replace the state directory or Git metadata.
chown "$operator":agent /home/agent
chmod 755 /home/agent
for directory in .cache .local .config .npm workspace; do
  if [ ! -e "/home/agent/$directory" ]; then
    install -d -o agent -g agent -m 0700 "/home/agent/$directory"
  fi
done
install -d -o root -g agent -m 3770 "$state"
if [ ! -e "$state/config.yaml" ]; then
  install -o "$operator" -g agent -m 0640 "$repo/config.yaml.example" "$state/config.yaml"
fi
if [ ! -e "$state/.env" ]; then
  install -o "$operator" -g agent -m 0640 "$repo/.env.example" "$state/.env"
fi
for name in SOUL.md .runtime.lock; do
  if [ ! -e "$state/$name" ]; then
    install -o "$operator" -g agent -m 0640 /dev/null "$state/$name"
  fi
done
chown "$operator":agent "$state/config.yaml" "$state/.env" "$state/SOUL.md" "$state/.runtime.lock"
chmod 640 "$state/config.yaml" "$state/.env" "$state/SOUL.md"
chmod 660 "$state/.runtime.lock"
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
install -o root -g root -m 0644 "$repo/scripts/pilotage-agent.service" /etc/systemd/system/pilotage-agent.service
install -o root -g root -m 0755 "$repo/scripts/pilotage-launcher" /usr/local/bin/pilotage
# A legacy user unit must not restart alongside the system unit on next boot.
loginctl disable-linger agent
user_unit=/home/agent/.config/systemd/user/default.target.wants/pilotage-agent.service
if [ -L "$user_unit" ]; then
  unlink "$user_unit"
fi
systemctl daemon-reload
systemctl enable "pilotage-agent.service"
echo "Protection installed. Service remains stopped."
echo "As $operator: configure/login if needed, then pilotage restart and pilotage doctor."
