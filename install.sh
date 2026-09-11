#!/usr/bin/env bash
# Public bootstrap for the existing protected Ubuntu installation.
# Keep execution inside one parsed block: curl must deliver the complete body
# before any installation work starts. Prompts use the terminal, not the pipe.
{
set -Eeuo pipefail

pilotage_fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

pilotage_platform() {
  [[ "$(uname -s)" == Linux ]] || pilotage_fail "Ubuntu Server 24.04 amd64 is required."
  local release_file="${1:-/etc/os-release}"
  [[ -r "$release_file" ]] || pilotage_fail "Cannot identify this operating system."
  local ID="" VERSION_ID=""
  . "$release_file"
  [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] \
    || pilotage_fail "Ubuntu Server 24.04 amd64 is required."
  [[ "$(dpkg --print-architecture)" == amd64 ]] \
    || pilotage_fail "The installer requires amd64."
  command -v systemctl >/dev/null \
    && systemctl show --property=Version --value >/dev/null 2>&1 \
    || pilotage_fail "A running systemd installation is required."
}

pilotage_root() {
  /usr/bin/env -i HOME=/root PATH="$PATH" DEBIAN_FRONTEND=noninteractive "$@"
}

pilotage_operator() {
  runuser -u operator -- /usr/bin/env -i \
    HOME="$operator_home" USER=operator LOGNAME=operator PATH="$PATH" \
    GIT_TERMINAL_PROMPT=0 "$@"
}

pilotage_lock() {
  local lock_dir=/run/pilotage-installer
  [[ ! -L "$lock_dir" ]] || pilotage_fail "Installer lock directory cannot be a link."
  install -d -o root -g root -m 0700 "$lock_dir"
  exec 9>"$lock_dir/lock"
  flock -n 9 || pilotage_fail "Another Pilotage installation is running."
}

pilotage_preflight() {
  [[ ! -e "$manifest" && ! -L "$manifest" ]] \
    || pilotage_fail "Pilotage is already installed." \
      "Finish setup with sudo -iu operator, then pilotage login, pilotage model and pilotage whatsapp or pilotage telegram." \
      "Start with sudo systemctl enable --now pilotage-agent.service, then pilotage doctor." \
      "Use pilotage update for software updates."
  [[ ! -e "$state_dir" && ! -L "$state_dir" ]] \
    || pilotage_fail "Agent state already exists. Follow the README's protected deployment instructions; this installer does not migrate it."
  [[ ! -e "$launcher" && ! -L "$launcher" ]] \
    || pilotage_fail "$launcher already exists; review the existing installation."
  local service_state
  service_state="$(systemctl show pilotage-agent.service --property=LoadState --value)"
  [[ "$service_state" == not-found ]] \
    || pilotage_fail "pilotage-agent.service already exists; review it before installation."

  local path account expected_home
  for path in "$checkout" "$operator_home" "$agent_home"; do
    [[ ! -L "$path" ]] || pilotage_fail "Installation paths cannot be links: $path"
    if [[ -e "$path" ]]; then
      [[ -d "$path" ]] || pilotage_fail "Installation path is not a directory: $path"
      $resume || pilotage_fail "$path already exists. Use --resume only for an interrupted installation."
    fi
  done
  for account in operator agent; do
    expected_home="$operator_home"
    [[ "$account" != agent ]] || expected_home="$agent_home"
    if id "$account" >/dev/null 2>&1; then
      $resume || pilotage_fail "Account $account already exists. Use --resume only for an interrupted installation."
      [[ "$(id -u "$account")" != 0 ]] || pilotage_fail "$account cannot have UID 0."
      [[ "$(getent passwd "$account" | cut -d: -f6)" == "$expected_home" ]] \
        || pilotage_fail "Unexpected home directory for $account."
    elif [[ -e "$expected_home" ]]; then
      pilotage_fail "$expected_home exists without its account; inspect it before retrying."
    fi
  done
  if id agent >/dev/null 2>&1; then
    local group
    for group in $(id -nG agent); do
      [[ "$group" == agent ]] || pilotage_fail "Agent has unexpected group membership: $group"
    done
    if pgrep -u "$(id -u agent)" >/dev/null; then
      pilotage_fail "Agent processes are running. Stop them before retrying installation."
    else
      [[ "$?" == 1 ]] || pilotage_fail "Could not verify whether agent processes are running."
    fi
  fi
  for path in /opt /home; do
    [[ -d "$path" && ! -L "$path" && "$(stat -c %u "$path")" == 0 ]] \
      || pilotage_fail "$path must be a real root-owned directory."
    local mode
    mode="$(stat -c %a "$path")"
    (( (8#$mode & 0022) == 0 )) || pilotage_fail "$path must not be writable by other accounts."
  done
}

pilotage_checkout() {
  pilotage_root apt-get update
  pilotage_root apt-get install -y ca-certificates git sudo
  if ! id operator >/dev/null 2>&1; then
    getent group operator >/dev/null || addgroup operator
    adduser --disabled-password --ingroup operator --gecos 'Pilotage operator' operator
  fi
  [[ "$(stat -c %U "$operator_home")" == operator ]] \
    || pilotage_fail "The operator home must belong to operator."
  chmod 700 "$operator_home"
  if [[ ! -e "$checkout" ]]; then
    install -d -o operator -g operator -m 0755 "$checkout"
  fi
  [[ "$(stat -c %U "$checkout")" == operator ]] \
    || pilotage_fail "The runtime checkout must belong to operator."
  [[ ! -L "$checkout/.git" ]] || pilotage_fail "Git metadata cannot be a link."
  if [[ ! -d "$checkout/.git" ]]; then
    pilotage_operator git clone --branch main \
      https://github.com/zavrenn/pilotage-agent.git "$checkout"
  fi
  local unexpected script
  # Check ownership before invoking Git in a resumed checkout: Git can execute
  # configured helpers even for a status check.
  unexpected="$(find "$checkout" -xdev ! -type l \( ! -user operator -o -perm /022 \) -print -quit)"
  [[ -z "$unexpected" ]] || pilotage_fail "Checkout entry must be operator-owned and not writable by other accounts: $unexpected"
  unexpected="$(find "$checkout/.git" -type l -print -quit)"
  [[ -z "$unexpected" ]] || pilotage_fail "Git metadata cannot contain links: $unexpected"
  [[ "$(pilotage_operator git -C "$checkout" remote get-url origin)" == \
    https://github.com/zavrenn/pilotage-agent.git ]] \
    || pilotage_fail "The checkout has an unexpected origin."
  [[ "$(pilotage_operator git -C "$checkout" symbolic-ref --short HEAD)" == main ]] \
    || pilotage_fail "The installer expects the main branch."
  [[ "$(pilotage_operator git -C "$checkout" rev-parse HEAD)" == \
    "$(pilotage_operator git -C "$checkout" rev-parse refs/remotes/origin/main)" ]] \
    || pilotage_fail "The checkout contains local commits. Preserve them and inspect the checkout."
  [[ -z "$(pilotage_operator git -C "$checkout" status --porcelain --untracked-files=normal)" ]] \
    || pilotage_fail "The checkout has local changes. Preserve them and inspect the checkout."
  for script in install-system-dependencies.sh setup-accounts.sh install-protection.sh verify-protection.py; do
    [[ -f "$checkout/scripts/$script" && ! -L "$checkout/scripts/$script" ]] \
      || pilotage_fail "Required installation script is missing or linked: $script"
  done
}

pilotage_failed() {
  local status="$1"
  trap - ERR
  printf '\nInstallation stopped during %s. Existing files were preserved.\n' "$phase" >&2
  if [[ "$phase" == protection ]]; then
    printf 'Correct the error, then run as root:\n' >&2
    printf '  bash /opt/pilotage-agent/scripts/install-protection.sh operator &&\n' >&2
    printf '  systemctl disable pilotage-agent.service &&\n' >&2
    printf '  /opt/pilotage-agent/.venv/bin/python -I -B /opt/pilotage-agent/scripts/verify-protection.py\n' >&2
  else
    printf 'Correct the error, then retry with:\n' >&2
    printf '  curl -fsSL https://raw.githubusercontent.com/zavrenn/pilotage-agent/main/install.sh | bash -s -- --resume\n' >&2
  fi
  exit "$status"
}

pilotage_prompt() {
  local answer
  printf '%s' "$1" >&4
  IFS= read -r answer <&3 || return 1
  case "${answer:-$2}" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

pilotage_terminal() {
  { exec 3</dev/tty 4>/dev/tty; } 2>/dev/null
}

pilotage_setup() {
  printf '\nPilotage is installed; the service is stopped and disabled at boot.\n'
  printf 'Manage it from the operator account: sudo -iu operator (or su - operator as root).\n'
  printf 'Review /home/agent/.pilotage-agent/config.yaml, .env and SOUL.md.\n'
  printf 'Next: pilotage login, pilotage model, pilotage whatsapp and/or pilotage telegram.\n'
  printf 'Then: sudo systemctl enable --now pilotage-agent.service and pilotage doctor.\n'

  if $skip_setup || ! pilotage_terminal; then
    printf 'Complete account and channel setup from an interactive terminal.\n'
    printf 'If operator has no password, set it as root: passwd operator\n'
    return 0
  fi
  local password_status channel
  password_status="$(passwd --status operator | awk '{print $2}')"
  if [[ "$password_status" != P ]]; then
    printf '\nSet the operator password for login and sudo.\n' >&4
    passwd operator <&3 >&4 2>&4 || return 1
  fi
  pilotage_prompt 'Set up ChatGPT and messaging now? [Y/n] ' y || return 0
  pilotage_operator /usr/local/bin/pilotage login <&3 >&4 2>&4 || return 1
  pilotage_operator /usr/local/bin/pilotage model <&3 >&4 2>&4 || return 1
  while true; do
    printf '\nChannel: 1 WhatsApp, 2 Telegram, 3 both, 0 finish later [1]: ' >&4
    IFS= read -r channel <&3 || return 1
    case "${channel:-1}" in
      1|2|3) channel="${channel:-1}"; break ;;
      0) return 0 ;;
      *) printf 'Choose 0, 1, 2 or 3.\n' >&4 ;;
    esac
  done
  if [[ "$channel" == 1 || "$channel" == 3 ]]; then
    pilotage_operator /usr/local/bin/pilotage whatsapp <&3 >&4 2>&4 || return 1
  fi
  if [[ "$channel" == 2 || "$channel" == 3 ]]; then
    pilotage_operator /usr/local/bin/pilotage telegram <&3 >&4 2>&4 || return 1
  fi
  printf '\nText messaging is ready to start. Voice is optional; see config.yaml and .env to enable it.\n' >&4
  if pilotage_prompt 'Start now, enable startup at boot, and check readiness? [y/N] ' n; then
    pilotage_operator sudo systemctl enable --now pilotage-agent.service <&3 >&4 2>&4 || return 1
    pilotage_operator /usr/local/bin/pilotage doctor <&3 >&4 2>&4 || return 1
  fi
}

pilotage_main() {
  local resume=false skip_setup=false
  local argument
  for argument in "$@"; do
    case "$argument" in
      --resume) resume=true ;;
      --skip-setup) skip_setup=true ;;
      -h|--help)
        printf 'Install Pilotage on an existing Ubuntu Server 24.04 amd64 system.\n'
        printf 'Usage: install.sh [--resume] [--skip-setup]\n'
        printf 'Run as root or a user with sudo. LXC/LXD is not required.\n'
        printf '  --resume      Retry an interrupted installation before protection.\n'
        printf '  --skip-setup  Install without starting or enabling the service; finish setup later.\n'
        return 0 ;;
      *) pilotage_fail "Unknown option: $argument" ;;
    esac
  done
  export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  umask 022
  pilotage_platform
  if [[ "$(id -u)" != 0 ]]; then
    command -v sudo >/dev/null || pilotage_fail "Run as root, or install sudo for this administrator account."
    # Serialize only this installer's functions. No second download, inherited
    # shell setup, or caller-supplied command string runs with root privileges.
    local bootstrap
    bootstrap="$(declare -f pilotage_fail pilotage_platform pilotage_root pilotage_operator \
      pilotage_lock pilotage_preflight pilotage_checkout pilotage_failed \
      pilotage_prompt pilotage_terminal pilotage_setup pilotage_main)"
    sudo /usr/bin/env -i HOME=/root PATH="$PATH" /bin/bash -c \
      "set -Eeuo pipefail; $bootstrap; pilotage_main \"\$@\"" pilotage-installer "$@" </dev/null
    return $?
  fi

  local checkout=/opt/pilotage-agent operator_home=/home/operator agent_home=/home/agent
  local state_dir=/home/agent/.pilotage-agent manifest=/etc/pilotage-agent.json
  local launcher=/usr/local/bin/pilotage
  local phase=preflight
  cd /
  pilotage_lock
  pilotage_preflight
  trap 'pilotage_failed "$?"' ERR
  phase='accounts and checkout'
  printf 'Installing Pilotage with separate operator and agent accounts.\n'
  printf 'Do not interrupt package and account installation. Login and messaging setup can be finished later.\n'
  pilotage_checkout </dev/null
  phase='system dependencies'
  pilotage_root /bin/bash "$checkout/scripts/install-system-dependencies.sh" </dev/null
  pilotage_root /bin/bash "$checkout/scripts/setup-accounts.sh" operator </dev/null
  phase='runtime dependencies'
  pilotage_operator /bin/bash -c 'cd /opt/pilotage-agent && bash scripts/install.sh --dependencies-only' </dev/null
  phase=protection
  pilotage_root /bin/bash "$checkout/scripts/install-protection.sh" operator </dev/null
  pilotage_root systemctl disable pilotage-agent.service </dev/null
  pilotage_root "$checkout/.venv/bin/python" -I -B "$checkout/scripts/verify-protection.py" </dev/null
  trap - ERR
  if ! pilotage_setup; then
    printf '\nInstallation is complete, but setup or readiness checks failed. Continue as operator using the commands above.\n' >&2
    return 1
  fi
}

if [[ -z "${BASH_SOURCE[0]:-}" || "${BASH_SOURCE[0]}" == "$0" ]]; then
  pilotage_main "$@"
fi
}
