#!/usr/bin/env bash
# Build the four shared, deployment-time Python environments.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
env_root="$repo_root/.pilotage-envs"

fail() { echo "error: $*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || fail "python3 is not installed."
command -v uv >/dev/null 2>&1 || fail "uv is not installed."

install_environment() {
  local name="$1"
  local environment="$env_root/$name"
  local isolated_uv_config

  isolated_uv_config="$(mktemp -d)"
  local result=0
  (
    unset UV_CONFIG_FILE UV_NO_CONFIG UV_PROJECT_ENVIRONMENT UV_PYTHON
    export XDG_CONFIG_HOME="$isolated_uv_config"
    export XDG_CONFIG_DIRS="$isolated_uv_config"
    UV_PROJECT_ENVIRONMENT="$environment" uv sync \
      --locked \
      --only-group "$name" \
      --no-install-project \
      --python "$(command -v python3)" \
      --no-python-downloads
  ) || result=$?
  rmdir "$isolated_uv_config" 2>/dev/null || true
  return "$result"
}

mkdir -p "$env_root"
install_environment chart
install_environment docs
install_environment excel
install_environment pdf

echo "Prepared Python environments at $env_root"
