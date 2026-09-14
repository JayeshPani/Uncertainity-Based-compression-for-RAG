"""Provenance tests.

Provenance is metadata, which makes it tempting to treat as best-effort and untested. But
its whole job is to let a reviewer verify a number months later, and a provenance record
that is silently wrong is worse than none — it certifies a run that did not happen.

Two properties matter and are pinned here: it records what was actually used (the resolved
config, not the on-disk defaults), and it never raises, because losing an hour of compute
to a missing `git` binary would be an absurd trade for metadata.
"""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.provenance import (
    _is_input,
    collect,
    dataset_record,
    environment,
    git_state,
    hash_corpus,
    model_revision,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestNeverRaises:
    """A provenance failure must degrade to a recorded null, never kill the experiment."""

    def test_missing_file_hashes_to_none(self):
        assert sha256_file(Path("/no/such/file")) is None

    def test_missing_corpus_directory_is_reported_not_raised(self):
        record = hash_corpus("/no/such/directory")
        assert record["manifest_sha256"] is None

    def test_unknown_model_has_no_revision(self):
        assert model_revision("not-a-real-org/not-a-real-model") is None

    def test_empty_model_id_does_not_raise(self):
        assert model_revision("") is None


class TestRecordsWhatWasActuallyUsed:
    def test_config_is_the_resolved_one_not_the_file(self):
        """A run with an override must record the override. Recording the defaults would
        document a run nobody performed."""
        cfg = load_config()
        cfg["compression"] = dict(cfg["compression"], max_keep=0.33)
        record = collect(cfg, argv=["run_eval.py"])
        assert record["config"]["compression"]["max_keep"] == 0.33

    def test_command_is_recorded_verbatim(self):
        record = collect(load_config(), argv=["python", "scripts/run_eval.py", "--limit", "5"])
        assert record["command"] == "python scripts/run_eval.py --limit 5"

    def test_all_three_model_roles_are_recorded(self):
        """Embedding, support and generation each affect results independently."""
        models = collect(load_config(), argv=["x"])["models"]
        assert set(models) == {"embedding", "generation", "support"}
        assert all("id" in m and "revision" in m for m in models.values())

    def test_environment_names_the_libraries_that_move_numbers(self):
        env = environment()
        for key in ("python", "torch", "transformers", "numpy"):
            assert key in env


class TestDatasetIdentity:
    def test_corpus_manifest_is_order_independent(self):
        """Hashing concatenated bytes would change with filesystem ordering; hashing a
        sorted name:digest manifest depends only on content and naming."""
        first = hash_corpus(REPO_ROOT / "data" / "kb")
        second = hash_corpus(REPO_ROOT / "data" / "kb")
        assert first["manifest_sha256"] == second["manifest_sha256"]
        assert first["n_documents"] > 0

    def test_dataset_record_covers_questions_and_corpus(self):
        record = dataset_record(
            REPO_ROOT / "data" / "eval" / "questions.yaml", REPO_ROOT / "data" / "kb"
        )
        assert record["questions"]["sha256"]
        assert record["kb"]["manifest_sha256"]

    def test_kb_is_optional(self):
        record = dataset_record(REPO_ROOT / "data" / "eval" / "questions.yaml", None)
        assert "kb" not in record


class TestGitState:
    def test_reports_dirtiness_explicitly(self):
        """`dirty_inputs` matters more than the commit: a result from edited code is not
        reproducible from that commit, and the bundle has to say so."""
        state = git_state()
        assert set(state) >= {"commit", "branch", "dirty", "dirty_inputs", "dirty_files"}
        assert state["dirty"] in (True, False, None)


class TestInputClassification:
    """Runs write their JSON into artifacts/, so a sequence of experiments dirties the
    tree for every run after the first. Counting that as "not reproducible" would make the
    check fire on every real session — and a safeguard that always fires gets ignored."""

    def test_source_changes_count(self):
        for line in (
            " M src/pipeline.py",
            "?? scripts/new_experiment.py",
            " M config/default.yaml",
            " M data/qasper/questions.yaml",
            " M requirements.txt",
        ):
            assert _is_input(line), line

    def test_result_files_do_not_count(self):
        for line in (
            " M artifacts/qasper-main/tight_absolute.json",
            "?? artifacts/qasper-main/README.md",
            " M README.md",
            " M docs/PAPER-OUTLINE.md",
        ):
            assert not _is_input(line), line

    def test_rename_is_judged_by_its_destination(self):
        assert _is_input('R  docs/old.py -> src/pipeline.py')
        assert not _is_input('R  src/old.py -> artifacts/dead.json')

    def test_quoted_paths_with_spaces_are_handled(self):
        assert _is_input(' M "src/some file.py"')
