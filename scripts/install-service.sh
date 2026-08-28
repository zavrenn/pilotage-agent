#!/usr/bin/env bash
# Install one Pilotage profile as a long-lived user systemd service.

set -euo pipefail

fail() { echo "error: $*" >&2; exit 1; }

profile="default"
if [ "${1:-}" = "--profile" ]; then
  [ -n "${2:-}" ] || fail "--profile requires a name"
  profile="$2"
  shift 2
fi
[ "$#" -eq 0 ] || fail "usage: $0 [--profile NAME]"
case "$profile" in
  default|[a-z0-9]* ) ;;
  * ) fail "profile names use lowercase letters, digits, '_' or '-'" ;;
esac
case "$profile" in
  *[!a-z0-9_-]* ) fail "profile names use lowercase letters, digits, '_' or '-'" ;;
esac
[ "${#profile}" -le 64 ] || fail "profile name is longer than 64 characters"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pilotage_bin="$repo_root/.venv/bin/pilotage"
[ -x "$pilotage_bin" ] || fail "run scripts/install.sh first"
command -v systemctl >/dev/null 2>&1 || fail "systemctl is not installed"
command -v node >/dev/null 2>&1 || fail "node is not on PATH"

state_root="$(realpath -m -- "${PILOTAGE_HOME:-$HOME/.pilotage-agent}")"
if [ "$(basename "$(dirname "$state_root")")" = "profiles" ]; then
  state_root="$(dirname "$(dirname "$state_root")")"
fi
if [ "$profile" = "default" ]; then
  profile_root="$state_root"
else
  profile_root="$state_root/profiles/$profile"
  [ -d "$profile_root" ] || fail "create the profile first: $pilotage_bin profile create $profile"
fi
[ -d "$repo_root/bridge/node_modules" ] || fail "run scripts/install.sh first"
if ! PILOTAGE_HOME="$state_root" "$pilotage_bin" --profile "$profile" status >/dev/null; then
  fail "the selected profile failed its configuration/authentication health check"
fi

escape_unit_value() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//%/%%}"
  printf '%s' "$value"
}

unit_dir="$HOME/.config/systemd/user"
unit_path="$unit_dir/pilotage-agent@.service"
umask 077
mkdir -p "$unit_dir"
chmod 700 "$unit_dir" 2>/dev/null || true

escaped_repo="$(escape_unit_value "$repo_root")"
escaped_bin="$(escape_unit_value "$pilotage_bin")"
escaped_home="$(escape_unit_value "$HOME")"
escaped_state="$(escape_unit_value "$state_root")"
escaped_path="$(escape_unit_value "$repo_root/.venv/bin:$(dirname "$(command -v node)"):$PATH")"

# Full worst-case stop path: Telegram intake teardown runs before the shared
# drain deadline, channel cleanup runs after it, and the final reserve covers
# client close plus systemd/process scheduling overhead.
stop_intake_budget=20
shutdown_drain_budget=30
channel_cleanup_budget=10
shutdown_headroom=30
timeout_stop_sec=$((stop_intake_budget + shutdown_drain_budget + channel_cleanup_budget + shutdown_headroom))

temp_unit="$(mktemp "$unit_dir/.pilotage-agent.XXXXXX")"
trap 'rm -f "$temp_unit"' EXIT
cat >"$temp_unit" <<EOF
[Unit]
Description=Pilotage Agent (%i)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart="$escaped_bin" --profile %i run
# WorkingDirectory= consumes the path directly; unlike ExecStart=, wrapping it
# in quotes makes the quote part of the path and systemd rejects it as relative.
WorkingDirectory=$escaped_repo
Environment="HOME=$escaped_home"
Environment="PILOTAGE_HOME=$escaped_state"
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=$escaped_path"
Restart=always
RestartSec=5
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=$timeout_stop_sec
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF
mv "$temp_unit" "$unit_path"
trap - EXIT
chmod 600 "$unit_path"

systemctl --user daemon-reload
systemctl --user enable --now "pilotage-agent@$profile.service"

echo "Installed and started pilotage-agent@$profile.service"
echo "Status: systemctl --user status pilotage-agent@$profile.service"
echo "Logs:   journalctl --user -u pilotage-agent@$profile.service -f"

if command -v loginctl >/dev/null 2>&1; then
  linger="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
  if [ "$linger" != "yes" ]; then
    echo "warning: enable lingering so the user service starts at boot without a login:" >&2
    echo "  sudo loginctl enable-linger $USER" >&2
  fi
fi
