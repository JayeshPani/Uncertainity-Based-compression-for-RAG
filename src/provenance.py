"""Record exactly what produced a result, automatically.

Every number in the paper has to trace back to a commit, a config, a dataset and an
environment. Collecting that by hand is how a reproducibility section ends up describing
a run that no longer exists — the config drifts, a threshold moves, and the recorded
command stops producing the recorded number with nothing to show it changed.

So this is captured by the harness on every run and embedded in the result JSON itself.
Not a separate step anyone has to remember.

Nothing here may raise. A provenance failure must degrade to a recorded `null` and let
the experiment finish — losing an hour of compute to a missing `git` binary would be an
absurd trade for metadata.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - metadata must never break a run
        return None


# Paths whose contents determine a result. Anything else in the tree may be modified
# without affecting whether a run can be reproduced from its commit.
INPUT_PREFIXES = ("src/", "scripts/", "config/", "data/", "tests/", "requirements.txt")


def _is_input(status_line: str) -> bool:
    """Whether a `git status --porcelain` line names something a result depends on.

    The distinction is load-bearing. Runs write their JSON into `artifacts/`, so a
    sequence of experiments inevitably dirties the tree for every run after the first —
    and treating that as "not reproducible" would make the checker cry wolf on every real
    experiment session, which is how a safeguard gets ignored.
    """
    path = status_line[3:].strip().strip('"')
    # Renames appear as "old -> new"; the destination is what matters.
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.startswith(INPUT_PREFIXES)


def git_state() -> dict:
    """Commit, branch, and whether the tree was dirty when the run started.

    `dirty_inputs` matters more than the commit: a result produced from edited *code* is
    not reproducible from that commit, and saying so is the difference between a bundle a
    reviewer can trust and one they cannot. `dirty` covers the whole tree and is kept for
    completeness, but it is `dirty_inputs` that should gate certification.
    """
    status = _run(["git", "status", "--porcelain"])
    lines = status.splitlines() if status else []
    inputs = [line for line in lines if _is_input(line)]
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
        "dirty_inputs": bool(inputs) if status is not None else None,
        "dirty_files": lines,
        "dirty_input_files": inputs,
    }


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def hash_corpus(kb_path: str | Path) -> dict:
    """Hash a knowledge base as a manifest, not as a blob.

    A single hash over concatenated files would change if the filesystem returned them in
    a different order. Hashing `name:sha256` lines in sorted order makes the digest depend
    on content and naming only.
    """
    directory = Path(kb_path)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    try:
        files = sorted(p for p in directory.glob("*.txt") if p.is_file())
    except Exception:  # noqa: BLE001
        return {"path": str(kb_path), "n_documents": None, "manifest_sha256": None}

    if not files:
        # Do NOT hash the empty list. It would produce the sha256 of the empty string —
        # a real-looking digest, identical for every missing or mistyped path, which is
        # exactly the kind of plausible-but-wrong metadata provenance exists to prevent.
        return {"path": str(kb_path), "n_documents": 0, "manifest_sha256": None}

    lines = [f"{p.name}:{sha256_file(p)}" for p in files]
    manifest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {
        "path": str(kb_path),
        "n_documents": len(files),
        "manifest_sha256": manifest,
    }


def model_revision(model_id: str) -> str | None:
    """The HF snapshot commit a model id actually resolved to on this machine.

    Model ids are mutable — `Qwen/Qwen2.5-0.5B-Instruct` today is not necessarily the
    weights it named six months ago. The snapshot sha is what makes the run pinnable.
    Read from the local cache layout rather than the network, so this works offline.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        cache = Path.home() / ".cache" / "huggingface" / "hub"

    folder = cache / f"models--{model_id.replace('/', '--')}"
    for ref in ("main", "master"):
        candidate = folder / "refs" / ref
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            pass
    return None


def environment() -> dict:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "numpy", "yaml"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001
            versions[name] = None
    versions["platform"] = platform.platform()
    versions["machine"] = platform.machine()
    return versions


def collect(cfg, argv: list[str] | None = None, extra: dict | None = None) -> dict:
    """Full provenance record for one run.

    `cfg` is the *resolved* config — after every `--set` and flag override — because that
    is what the run actually used. Recording the on-disk defaults instead would document
    a run nobody performed.
    """
    from .config import resolve_device

    models = {}
    for role, path in (
        ("embedding", "models.embedding"),
        ("generation", "models.generation"),
    ):
        model_id = cfg.get_path(path)
        models[role] = {"id": model_id, "revision": model_revision(model_id or "")}

    scorer = cfg.get_path("evaluation.scorer", "relevance")
    scorer_id = cfg.get_path(f"evaluation.scorers.{scorer}.model")
    models["support"] = {
        "role": scorer,
        "id": scorer_id,
        "revision": model_revision(scorer_id or ""),
    }

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(argv or sys.argv),
        "git": git_state(),
        "models": models,
        "device": resolve_device(cfg.get_path("models.device", "auto")),
        "environment": environment(),
        # The resolved config, serialised through JSON so it is inspectable rather than
        # a repr of whatever object graph happened to be in memory.
        "config": json.loads(json.dumps(cfg, default=str)),
    }
    if extra:
        record.update(extra)
    return record


def dataset_record(questions_path: str | Path, kb_path: str | Path | None) -> dict:
    path = Path(questions_path)
    record = {
        "questions": {"path": str(questions_path), "sha256": sha256_file(path)},
    }
    if kb_path:
        record["kb"] = hash_corpus(kb_path)
    return record
