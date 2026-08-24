#!/usr/bin/env bash
set -euo pipefail

# Build the Ubuntu 24.04 container runtime before running install.sh as the
# unprivileged Pilotage service user. This script is intentionally root-only;
# the resident agent never installs packages.

NODE_MAJOR="${NODE_MAJOR:-22}"
UV_VERSION="0.12.0"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ "$(id -u)" -eq 0 ] || fail "run this script as root inside the container"

. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || fail "Ubuntu 24.04 is required"
[ "${VERSION_ID:-}" = "24.04" ] || fail "Ubuntu 24.04 is required"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
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
  python3-dev \
  python3-venv \
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

# Google publishes a native amd64 package. Other architectures must supply a
# working Chromium binary explicitly; failing here is safer than installing an
# Ubuntu snap wrapper that is unsuitable for the unprivileged LXC target.
architecture="$(dpkg --print-architecture)"
if [ "$architecture" != "amd64" ]; then
  fail "automatic headless Chrome installation currently supports amd64 only"
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

if ! dpkg-query -W -f='${Status}' packages-microsoft-prod 2>/dev/null \
  | grep -q '^install ok installed$'; then
  microsoft_repo="$(mktemp /tmp/pilotage-microsoft-prod.XXXXXX.deb)"
  trap 'rm -f -- "$microsoft_repo"' EXIT
  curl -fsSLo "$microsoft_repo" \
    https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb
  dpkg -i "$microsoft_repo"
  rm -f -- "$microsoft_repo"
  trap - EXIT
  apt-get update
fi

ACCEPT_EULA=Y apt-get install -y mssql-tools18 unixodbc-dev
if [ ! -e /usr/local/bin/sqlcmd ]; then
  ln -s /opt/mssql-tools18/bin/sqlcmd /usr/local/bin/sqlcmd
fi
if [ ! -e /usr/local/bin/bcp ]; then
  ln -s /opt/mssql-tools18/bin/bcp /usr/local/bin/bcp
fi

# Keep the resolver itself outside the agent runtime and pin it at image build.
python3 -m venv /opt/pilotage-uv
/opt/pilotage-uv/bin/pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  "uv==$UV_VERSION"
ln -sfn /opt/pilotage-uv/bin/uv /usr/local/bin/uv

fc-cache -f

echo "Pilotage system dependencies installed."
echo "Continue as the unprivileged service user with: bash scripts/install.sh"
