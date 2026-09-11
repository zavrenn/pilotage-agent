#!/usr/bin/env bash
set -euo pipefail

# Build the Ubuntu runtime before running scripts/install.sh as the
# unprivileged Pilotage service user. This script is intentionally root-only;
# the resident agent never installs packages.

NODE_MAJOR="${NODE_MAJOR:-22}"
UV_VERSION="0.12.0"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ "$(id -u)" -eq 0 ] || fail "run this script as root on Ubuntu"

. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || fail "Ubuntu is required"
command -v apt-get >/dev/null || fail "APT is required to install system dependencies"
# Chrome is supplied as a native amd64 package, not an Ubuntu snap wrapper.
[ "$(dpkg --print-architecture)" = amd64 ] \
  || fail "automatic headless Chrome installation currently supports amd64 only"

export DEBIAN_FRONTEND=noninteractive
umask 022

apt-get update
apt-get install -y \
  acl \
  build-essential \
  ca-certificates \
  catdoc \
  curl \
  ffmpeg \
  file \
  fontconfig \
  fonts-crosextra-caladea \
  fonts-crosextra-carlito \
  fonts-dejavu-core \
  fonts-freefont-ttf \
  fonts-ipafont-gothic \
  fonts-liberation \
  fonts-noto-cjk \
  fonts-noto-color-emoji \
  fonts-noto-core \
  fonts-tlwg-loma-otf \
  fonts-unifont \
  fonts-wqy-zenhei \
  ghostscript \
  git \
  gnupg \
  graphviz \
  imagemagick \
  jq \
  libffi-dev \
  libmagic1 \
  libreoffice-calc \
  libreoffice-impress \
  libreoffice-writer \
  p7zip-full \
  pandoc \
  poppler-utils \
  qpdf \
  ripgrep \
  shared-mime-info \
  sqlite3 \
  tesseract-ocr \
  tesseract-ocr-ara \
  tesseract-ocr-eng \
  tesseract-ocr-fra \
  unrtf \
  unzip \
  xfonts-cyrillic \
  xfonts-scalable \
  xvfb \
  xz-utils \
  zip

node_major_installed=0
if command -v node >/dev/null 2>&1; then
  node_major_installed="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
fi
if [ "$node_major_installed" -lt "$NODE_MAJOR" ]; then
  nodesource_setup="$(mktemp /tmp/pilotage-nodesource.XXXXXX.sh)"
  trap 'rm -f -- "$nodesource_setup"' EXIT
  curl -fsSLo "$nodesource_setup" \
    "https://deb.nodesource.com/setup_${NODE_MAJOR}.x"
  bash "$nodesource_setup"
  rm -f -- "$nodesource_setup"
  trap - EXIT
  apt-get install -y nodejs
fi

install -d -m 0755 /etc/apt/keyrings
chrome_key="$(mktemp /tmp/pilotage-google-linux-signing-key.XXXXXX.pub)"
trap 'rm -f -- "$chrome_key"' EXIT
curl -fsSLo "$chrome_key" \
  https://dl.google.com/linux/linux_signing_key.pub
gpg --dearmor --yes \
  --output /etc/apt/keyrings/google-chrome.gpg \
  "$chrome_key"
rm -f -- "$chrome_key"
trap - EXIT
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] https://dl.google.com/linux/chrome/deb/ stable main" \
  > /etc/apt/sources.list.d/google-chrome.list
apt-get update
apt-get install -y google-chrome-stable

# Keep Python independent of Ubuntu's system interpreter and accessible to both
# accounts. The resident agent cannot modify these root-owned installations.
uv_installer="$(mktemp /tmp/pilotage-uv.XXXXXX.sh)"
trap 'rm -f -- "$uv_installer"' EXIT
curl -fsSLo "$uv_installer" "https://astral.sh/uv/$UV_VERSION/install.sh"
UV_UNMANAGED_INSTALL=/opt/pilotage-uv/bin sh "$uv_installer"
rm -f -- "$uv_installer"
trap - EXIT
ln -sfn /opt/pilotage-uv/bin/uv /usr/local/bin/uv
install -d -m 0755 /opt/pilotage-python/bin
UV_PYTHON_INSTALL_DIR=/opt/pilotage-python \
  UV_PYTHON_BIN_DIR=/opt/pilotage-python/bin \
  /opt/pilotage-uv/bin/uv python install 3.13
ln -sfn python3.13 /opt/pilotage-python/bin/python3
/opt/pilotage-python/bin/python3 --version

install_sql_tools() {
  if ! dpkg-query -W -f='${Status}' packages-microsoft-prod 2>/dev/null \
    | grep -q '^install ok installed$'; then
    local microsoft_repo
    microsoft_repo="$(mktemp /tmp/pilotage-microsoft-prod.XXXXXX.deb)"
    if ! curl -fsSLo "$microsoft_repo" \
      "https://packages.microsoft.com/config/ubuntu/${VERSION_ID}/packages-microsoft-prod.deb"; then
      rm -f -- "$microsoft_repo"
      echo "Optional SQL tools skipped: Microsoft repository unavailable for Ubuntu ${VERSION_ID}."
      return 0
    fi
    dpkg -i "$microsoft_repo"
    rm -f -- "$microsoft_repo"
    apt-get update
  fi
  if ! apt-cache policy mssql-tools18 | grep -Eq 'Candidate: [0-9]'; then
    echo "Optional SQL tools skipped: mssql-tools18 is unavailable in the configured repositories."
    return 0
  fi
  ACCEPT_EULA=Y apt-get install -y mssql-tools18 unixodbc-dev
  if [ ! -e /usr/local/bin/sqlcmd ]; then
    ln -s /opt/mssql-tools18/bin/sqlcmd /usr/local/bin/sqlcmd
  fi
  if [ ! -e /usr/local/bin/bcp ]; then
    ln -s /opt/mssql-tools18/bin/bcp /usr/local/bin/bcp
  fi
}
install_sql_tools

fc-cache -f

echo "Pilotage system dependencies installed."
echo "Continue as the unprivileged service user with: bash scripts/install.sh"
