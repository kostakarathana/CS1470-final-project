#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$PYTHON_BIN" -m pip install -e "$ROOT_DIR[dev,analysis]"

"$PYTHON_BIN" - <<'PY'
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env

cfg = load_experiment_config("configs/shared_smoke.yaml")
env = build_env(cfg)
observations = env.reset(seed=0)
assert observations["agent_0"].shape == env.observation_shape
env.close()
print("Bundled Pommerman backend smoke check passed.")
PY

echo "Runtime bootstrap complete."
