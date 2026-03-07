#!/usr/bin/env bash
set -euo pipefail

# Installs slangc into ~/.local/bin for Linux/WSL.
# Usage:
#   ./install-slangc.sh            # installs latest release
#   ./install-slangc.sh v2026.3.1  # installs specific tag

TAG="${1:-latest}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd tar
need_cmd python3

if [ "$TAG" = "latest" ]; then
  TAG="$(python3 - <<'PY'
import json, urllib.request
url = 'https://api.github.com/repos/shader-slang/slang/releases/latest'
with urllib.request.urlopen(url) as r:
    data = json.load(r)
print(data['tag_name'])
PY
)"
fi

VERSION="${TAG#v}"
ARCH="$(uname -m)"

case "$ARCH" in
  x86_64)
    CANDIDATES=(
      "slang-${VERSION}-linux-x86_64-glibc-2.27.tar.gz"
      "slang-${VERSION}-linux-x86_64.tar.gz"
    )
    ;;
  aarch64)
    CANDIDATES=(
      "slang-${VERSION}-linux-aarch64.tar.gz"
    )
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

BASE_URL="https://github.com/shader-slang/slang/releases/download/${TAG}"
TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/slang.tar.gz"
trap 'rm -rf "${TMP_DIR}"' EXIT

FOUND=""
for asset in "${CANDIDATES[@]}"; do
  url="${BASE_URL}/${asset}"
  if curl -fsI "$url" >/dev/null 2>&1; then
    FOUND="$url"
    break
  fi
done

if [ -z "$FOUND" ]; then
  echo "Could not find a matching release asset for ${TAG} (${ARCH})." >&2
  echo "Release page: https://github.com/shader-slang/slang/releases/tag/${TAG}" >&2
  exit 1
fi

echo "Downloading: $FOUND"
curl -fL "$FOUND" -o "$ARCHIVE_PATH"

INSTALL_ROOT="${HOME}/.local/slang"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "$INSTALL_ROOT" "$BIN_DIR"

rm -rf "$INSTALL_ROOT"
mkdir -p "$INSTALL_ROOT"

tar -xzf "$ARCHIVE_PATH" -C "$INSTALL_ROOT"

SLANGC_PATH=""
if [ -x "${INSTALL_ROOT}/bin/slangc" ]; then
  SLANGC_PATH="${INSTALL_ROOT}/bin/slangc"
elif [ -x "${INSTALL_ROOT}/slangc" ]; then
  SLANGC_PATH="${INSTALL_ROOT}/slangc"
else
  maybe_nested="$(find "$INSTALL_ROOT" -maxdepth 3 -type f -name slangc 2>/dev/null | python3 - <<'PY'
import sys
paths = [line.strip() for line in sys.stdin if line.strip()]
print(paths[0] if paths else "")
PY
)"
  if [ -n "$maybe_nested" ] && [ -x "$maybe_nested" ]; then
    SLANGC_PATH="$maybe_nested"
  fi
fi

if [ -z "$SLANGC_PATH" ]; then
  echo "Install failed: could not locate slangc after extraction" >&2
  exit 1
fi

ln -sf "$SLANGC_PATH" "${BIN_DIR}/slangc"

echo
SLANGC_VERSION="$("${BIN_DIR}/slangc" -version 2>/dev/null || true)"
if [ -n "$SLANGC_VERSION" ]; then
  echo "Installed slangc: ${SLANGC_VERSION}"
else
  echo "Installed slangc: ${BIN_DIR}/slangc"
fi
echo "Location: ${BIN_DIR}/slangc"

case ":${PATH}:" in
  *":${BIN_DIR}:"*)
    echo "${BIN_DIR} is already on PATH."
    ;;
  *)
    echo "Add this to your shell rc if needed:"
    echo "  export PATH=\"${BIN_DIR}:\$PATH\""
    ;;
esac
