from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


ISO = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _lease_expires_iso(minutes: int = 10) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime(ISO)


@dataclass
class OrchestrationState:
    iteration: int = 1
    expected_actor: str = "galph"  # "galph" | "ralph"
    status: str = "idle"  # "idle" | "running-galph" | "waiting-ralph" | "running-ralph" | "complete" | "failed"
    last_update: str = field(default_factory=_utc_now_iso)
    lease_expires_at: str = field(default_factory=_lease_expires_iso)
    galph_commit: Optional[str] = None
    ralph_commit: Optional[str] = None

    @staticmethod
    def read(path: str) -> "OrchestrationState":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return OrchestrationState()

        return OrchestrationState(
            iteration=int(data.get("iteration", 1)),
            expected_actor=str(data.get("expected_actor", "galph")),
            status=str(data.get("status", "idle")),
            last_update=str(data.get("last_update", _utc_now_iso())),
            lease_expires_at=str(data.get("lease_expires_at", _lease_expires_iso())),
            galph_commit=data.get("galph_commit"),
            ralph_commit=data.get("ralph_commit"),
        )

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="state.", suffix=".json", dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def stamp(self, *, expected_actor: Optional[str] = None, status: Optional[str] = None,
              increment: bool = False, galph_commit: Optional[str] = None,
              ralph_commit: Optional[str] = None) -> None:
        if expected_actor is not None:
            self.expected_actor = expected_actor
        if status is not None:
            self.status = status
        if increment:
            self.iteration += 1
        if galph_commit:
            self.galph_commit = galph_commit
        if ralph_commit:
            self.ralph_commit = ralph_commit
        self.last_update = _utc_now_iso()
        self.lease_expires_at = _lease_expires_iso()

