from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from madreamer.trainers.common import ensure_dir


class JsonlLogger:
    def __init__(self, log_dir: str | Path, filename: str = "metrics.jsonl") -> None:
        directory = ensure_dir(log_dir)
        self.path = directory / filename

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
