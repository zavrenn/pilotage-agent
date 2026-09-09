#!/usr/bin/env bash
# Run from a trusted checkout as root. Does not grant the agent sudo access.
set -euo pipefail
fail() { echo "error: $*" >&2; exit 1; }
[ "$(id -u)" -eq 0 ] || fail "run as root"
operator="${1:-operator}"
[[ "$operator" =~ ^[a-z][a-z0-9_-]{0,30}$ ]] || fail "invalid operator account"
case "$operator" in root|agent) fail "operator must be separate from root and agent" ;; esac
[ "$#" -le 1 ] || fail "usage: $0 [OPERATOR]"
command -v sudo >/dev/null || fail "install sudo first"
if ! id "$operator" >/dev/null 2>&1; then
  adduser --disabled-password --gecos "Pilotage operator" "$operator"
fi
if ! id agent >/dev/null 2>&1; then
  adduser --disabled-password --gecos "Pilotage runtime" agent
fi
[ "$(getent passwd agent | cut -d: -f6)" = /home/agent ] || fail "agent home must be /home/agent"
[ "$(id -u "$operator")" -ne 0 ] || fail "operator cannot have UID 0"
[ "$(id -u agent)" -ne 0 ] || fail "agent cannot have UID 0"
[ "$(id -u "$operator")" -ne "$(id -u agent)" ] || fail "accounts must have different UIDs"
# Refuse unexpected privileges instead of silently removing existing access.
for group in $(id -nG agent); do
  [ "$group" = agent ] || fail "remove agent from privileged/unexpected group '$group' first"
done
if sudo -l -U agent >/dev/null 2>&1; then
  fail "agent has sudo permissions; remove those grants first"
fi
usermod -aG sudo,agent "$operator"
passwd --lock agent >/dev/null
operator_home="$(getent passwd "$operator" | cut -d: -f6)"
[ "$operator_home" = "/home/$operator" ] || fail "operator must have its own /home directory"
[ ! -L "$operator_home" ] && [ ! -L /home/agent ] || fail "account homes cannot be links"
chmod 700 "$operator_home"
echo "Accounts ready: $operator administers; agent runs without sudo."
echo "Set the operator password with: passwd $operator (or configure your normal administrative login)."
