from __future__ import annotations

import io
import shutil
from importlib import util
from pathlib import Path


def _load_tail_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "orchestration" / "tail_interleave_logs.py"
    spec = util.spec_from_file_location("tail_interleave_logs", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_interleave_summaries(tmp_path, monkeypatch):
    tail = _load_tail_module()
    prefix = Path("pytest-tail-summaries")
    root = Path("logs") / prefix

    galph_dir = root / "galph-summaries"
    ralph_dir = root / "ralph-summaries"
    galph_dir.mkdir(parents=True, exist_ok=True)
    ralph_dir.mkdir(parents=True, exist_ok=True)

    try:
        galph_file = galph_dir / "iter-00001_20250101_010203-summary.md"
        ralph_file = ralph_dir / "iter-00001_20250101_010305-summary.md"
        galph_file.write_text("Galph summary iteration 1\n")
        ralph_file.write_text("Ralph summary iteration 1\n")

        monkeypatch.setattr(tail, "load_post_state_commits", lambda max_commits=2000: {})

        buf = io.StringIO()
        rc = tail.interleave_last(
            prefix,
            count=1,
            out=buf,
            include_ls=False,
            source="summaries",
        )

        assert rc == 0
        output = buf.getvalue()
        assert '<logs prefix="pytest-tail-summaries"' in output
        assert 'role="galph"' in output
        assert 'source="summary"' in output
        assert 'format="markdown"' in output
        assert 'iter="1"' in output
        assert str(galph_file) in output
        assert str(ralph_file) in output
    finally:
        shutil.rmtree(root)
