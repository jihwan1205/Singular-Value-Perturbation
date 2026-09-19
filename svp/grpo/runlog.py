"""Per-iteration metrics: local jsonl always, wandb when a project is set."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class RunLogger:
    def __init__(self, run_dir: Path, config: dict, enabled: bool, project: str = "", entity: str = ""):
        self.enabled = enabled
        self._wandb = None
        if not enabled:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (run_dir / "log.jsonl").open("a")
        if not project:
            return
        try:
            import wandb
            id_file = run_dir / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else f"{run_dir.name}-{uuid.uuid4().hex[:8]}"
            id_file.write_text(run_id)
            self._wandb = wandb.init(project=project, entity=entity or None, name=run_dir.name,
                                     id=run_id, resume="allow", config=config, dir=str(run_dir))
        except Exception as exc:  # wandb must never kill a run
            logger.warning("wandb disabled: %s", exc)

    def log(self, step: int, metrics: dict) -> None:
        if not self.enabled:
            return
        self._jsonl.write(json.dumps({"iter": step, "ts": time.time(), **metrics}) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception as exc:
                logger.warning("wandb.log failed: %s", exc)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()
