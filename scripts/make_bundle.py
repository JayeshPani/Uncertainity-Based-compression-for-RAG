"""Package a finished run into a reproducibility bundle.

A reviewer asking "where did 53.0% come from?" should be able to answer it from one
directory: the result JSON, the exact command, the commit, the resolved config, and a
dependency lockfile. This assembles that from the provenance `run_eval.py` already
embedded, so the bundle cannot describe a run that did not happen.

It also *verifies* rather than assumes. A run made from a dirty tree, or one whose current
config no longer matches the config it recorded, is flagged loudly — those are exactly the
bundles that look reproducible and are not.

Usage:
    python scripts/make_bundle.py artifacts/qasper-main
    python scripts/make_bundle.py artifacts/qasper-main --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.provenance import _is_input, git_state  # noqa: E402


def dirty_inputs(git: dict) -> list[str]:
    """Modified files that could have changed the result, ignoring other runs' outputs.

    Records written before the input/output distinction existed still list every dirty
    file, so they are classified here rather than condemned wholesale — re-running an
    experiment to satisfy a metadata format change would be absurd.
    """
    recorded = git.get("dirty_input_files")
    if recorded is not None:
        return recorded
    return [line for line in (git.get("dirty_files") or []) if _is_input(line)]


def load_runs(bundle: Path) -> list[tuple[Path, dict]]:
    runs = []
    for path in sorted(bundle.glob("*.json")):
        if path.name in ("provenance.json", "bundle.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "arms" in data:
            runs.append((path, data))
    return runs


def lockfile(bundle: Path) -> Path | None:
    """`pip freeze`, so the bundle pins transitive versions the README cannot."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    target = bundle / "requirements-lock.txt"
    target.write_text(out.stdout, encoding="utf-8")
    return target


def summarise(path: Path, run: dict) -> list[str]:
    prov = run.get("provenance") or {}
    git = prov.get("git") or {}
    lines = [f"### `{path.name}`", ""]

    if not prov:
        lines += [
            "> **No provenance recorded.** This run predates automatic provenance and "
            "cannot be traced to a commit. Re-run it before citing its numbers.",
            "",
        ]
    else:
        dirty = bool(dirty_inputs(git))
        lines += [
            f"- commit: `{git.get('commit') or 'unknown'}`"
            + ("  **(modified sources — not reproducible from this commit)**" if dirty else ""),
            f"- run at: {prov.get('timestamp_utc', 'unknown')}",
            f"- device: {prov.get('device', 'unknown')}",
            "",
            "```bash",
            prov.get("command", "unknown"),
            "```",
            "",
        ]

    arms = run.get("arms") or []
    if arms:
        lines += [
            "| arm | questions | reduction | answer F1 | restoration | false abstain |",
            "|---|---|---|---|---|---|",
        ]
        for arm in arms:
            lines.append(
                f"| `{arm['mode']}` | {arm['questions']} | {arm['reduction']:.1%} | "
                f"{arm.get('mean_f1', 0):.4f} | {arm['restoration_rate']:.1%} | "
                f"{arm['false_abstain_rate']:.1%} |"
            )
        lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; exit non-zero if any run is untraceable",
    )
    args = parser.parse_args(argv)

    bundle = args.bundle if args.bundle.is_absolute() else REPO_ROOT / args.bundle
    if not bundle.is_dir():
        print(f"no such bundle directory: {bundle}", file=sys.stderr)
        return 2

    runs = load_runs(bundle)
    if not runs:
        print(f"no result JSONs found in {bundle}", file=sys.stderr)
        return 2

    problems = []
    for path, run in runs:
        prov = run.get("provenance") or {}
        if not prov:
            problems.append(f"{path.name}: no provenance recorded")
            continue
        git = prov.get("git") or {}
        # Only modified *inputs* break reproducibility. Runs write into artifacts/, so
        # every experiment after the first in a session sees the previous one's output
        # sitting uncommitted; failing on that would make the check meaningless noise.
        offenders = dirty_inputs(git)
        if offenders:
            problems.append(
                f"{path.name}: produced from modified sources ({', '.join(offenders)})"
            )
        if git.get("commit") is None:
            problems.append(f"{path.name}: no commit recorded")

    for problem in problems:
        print(f"WARNING  {problem}")

    if args.check:
        print(f"\n{len(runs)} run(s) checked, {len(problems)} problem(s).")
        return 1 if problems else 0

    lock = lockfile(bundle)
    current = git_state()

    lines = [
        f"# Reproducibility bundle — `{bundle.name}`",
        "",
        "Generated by `scripts/make_bundle.py`. Every number reported from this bundle "
        "traces to one of the runs below, together with the commit and command that "
        "produced it.",
        "",
        f"- bundled at commit: `{current.get('commit')}`"
        + ("  **(dirty)**" if current.get("dirty") else ""),
        f"- lockfile: `{lock.name if lock else 'unavailable'}`",
        "",
    ]

    if problems:
        lines += [
            "## Caveats",
            "",
            *(f"- {p}" for p in problems),
            "",
        ]

    lines += ["## Runs", ""]
    for path, run in runs:
        lines += summarise(path, run)

    lines += [
        "## Reproducing",
        "",
        "```bash",
        "git checkout <commit above>",
        "python3 -m venv --system-site-packages venv && source venv/bin/activate",
        f"pip install -r {lock.name if lock else 'requirements.txt'}",
        "# then the command recorded for the run you want",
        "```",
        "",
    ]

    target = bundle / "README.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target.relative_to(REPO_ROOT)}  ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
