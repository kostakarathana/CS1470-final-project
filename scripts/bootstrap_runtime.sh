#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLAYGROUND_DIR="${PLAYGROUND_DIR:-$ROOT_DIR/playground}"

"$PYTHON_BIN" -m pip install -e "$ROOT_DIR[dev,analysis]"

if [ ! -d "$PLAYGROUND_DIR/.git" ]; then
  git clone https://github.com/MultiAgentLearning/playground.git "$PLAYGROUND_DIR"
else
  git -C "$PLAYGROUND_DIR" pull --ff-only
fi

if ! "$PYTHON_BIN" -m pip install -U "$PLAYGROUND_DIR"; then
  echo "Pommerman playground install failed."
  echo "Known issue: the upstream dependency pin for python-rapidjson~=0.6.3 does not build cleanly on this macOS/Python 3.10 setup."
  echo "Recommended workaround: use a dedicated compatibility environment for the official playground, then rerun this script."
  exit 1
fi
echo "Runtime bootstrap complete."
