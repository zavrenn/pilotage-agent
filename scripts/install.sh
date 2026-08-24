#!/usr/bin/env bash
#
# Install the agent's own dependencies. Run once after cloning, and again after
# pulling changes.
#
# Python 3.11-3.13, uv, and Node 20+ are expected to be on the machine already — they
# belong to the container, not to the agent.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() { echo "error: $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || fail "python3 is not installed."
command -v uv >/dev/null 2>&1 || fail "uv is not installed."
command -v node >/dev/null 2>&1 || fail "node is not installed (Node 20 or newer)."
command -v npm >/dev/null 2>&1 || fail "npm is not installed."

python_version="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info < (3, 14)))' \
  || fail "Python 3.11 through 3.13 is required (found $python_version)."

node_major="$(node -p 'process.versions.node.split(".")[0]')"
[ "$node_major" -ge 20 ] || fail "Node 20 or newer is required (found $(node --version))."

sync_environment() {
  local environment="$1"
  shift
  local isolated_uv_config
  isolated_uv_config="$(mktemp -d)"
  local result=0
  (
    unset UV_CONFIG_FILE UV_NO_CONFIG UV_PROJECT_ENVIRONMENT UV_PYTHON
    export XDG_CONFIG_HOME="$isolated_uv_config"
    export XDG_CONFIG_DIRS="$isolated_uv_config"
    UV_PROJECT_ENVIRONMENT="$environment" uv sync \
      --locked \
      --python "$(command -v python3)" \
      --no-python-downloads \
      "$@"
  ) || result=$?
  rmdir "$isolated_uv_config" 2>/dev/null || true
  return "$result"
}

echo "==> Python environment (locked)"
sync_environment "$repo_root/.venv" --no-dev

echo "==> Prepared Python environments"
bash scripts/setup-runtime-environments.sh

echo "==> WhatsApp bridge"
(cd bridge && npm ci --silent --no-fund --no-audit)

state_dir="${PILOTAGE_HOME:-$HOME/.pilotage-agent}"
umask 077
mkdir -p "$state_dir" "$state_dir/workspace"
chmod 700 "$state_dir"
chmod 700 "$state_dir/workspace"
if [ ! -f "$state_dir/.env" ] && [ -f .env.example ]; then
  cp .env.example "$state_dir/.env"
  echo "==> Wrote $state_dir/.env — configure enabled channel credentials before starting."
fi
if [ -f "$state_dir/.env" ]; then
  chmod 600 "$state_dir/.env"
fi
if [ ! -f "$state_dir/config.yaml" ] && [ -f config.yaml.example ]; then
  cp config.yaml.example "$state_dir/config.yaml"
  echo "==> Wrote $state_dir/config.yaml — review the agent behavior before starting."
fi
if [ -f "$state_dir/config.yaml" ]; then
  chmod 600 "$state_dir/config.yaml"
fi

echo
echo "Installed. Next:"
echo "  1. edit $state_dir/.env and configure enabled channel credentials"
echo "  2. review $state_dir/config.yaml"
echo "  3. ./.venv/bin/pilotage login"
echo "  4. ./.venv/bin/pilotage run  # pair and test enabled channels"
echo "  5. bash scripts/install-service.sh # keep it running under systemd"
echo "  6. ./.venv/bin/pilotage doctor # prove the full deployment is ready"
