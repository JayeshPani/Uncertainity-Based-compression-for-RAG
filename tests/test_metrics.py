"""Tests for the scoring functions in scripts/run_eval.py.

These produce the project's headline numbers and its ablation verdict, and until now
nothing tested them — the one component whose bugs would be reported as findings was the
one component with no coverage. Two real errors motivate this file: a verdict declared
from a 3-keyword difference inside noise, and an F1 of 0.033 that turned out to be the
answer-length ceiling rather than a broken system.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    """`scripts/` is not a package, so load the module by path."""
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load_script("run_eval")
calibrate = _load_script("calibrate_threshold")


class TestAnswerF1:
    def test_exact_match_scores_one(self):
        assert run_eval.answer_f1("BERT", ["BERT"]) == pytest.approx(1.0)

    def test_normalisation_ignores_articles_case_and_punctuation(self):
        assert run_eval.answer_f1("The BERT.", ["bert"]) == pytest.approx(1.0)

    def test_best_gold_wins(self):
        """QASPER gives several annotator answers; matching any one of them counts."""
        assert run_eval.answer_f1("BERT", ["LSTM", "BERT"]) == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self):
        assert run_eval.answer_f1("LSTM", ["BERT"]) == 0.0

    def test_verbosity_is_penalised(self):
        """The mechanism behind the whole answer-style fix, stated as a test."""
        terse = run_eval.answer_f1("BERT", ["BERT"])
        padded = run_eval.answer_f1(
            "Based on the evidence provided, the model that was used is BERT.", ["BERT"]
        )
        assert padded < terse


class TestLengthCeiling:
    def test_equal_lengths_allow_a_perfect_score(self):
        assert run_eval.length_ceiling("BERT", ["BERT"]) == pytest.approx(1.0)

    def test_ceiling_holds_even_for_a_completely_wrong_answer(self):
        """It is a bound from length alone — correctness never enters."""
        assert run_eval.length_ceiling("LSTM", ["BERT"]) == pytest.approx(1.0)

    def test_long_answer_against_short_gold_is_capped(self):
        # 40 predicted words vs 7 gold: 2*7/47.
        prediction = " ".join(f"word{i}" for i in range(40))
        gold = " ".join(f"g{i}" for i in range(7))
        assert run_eval.length_ceiling(prediction, [gold]) == pytest.approx(2 * 7 / 47)

    def test_f1_never_exceeds_its_own_ceiling(self):
        """The property that makes the diagnostic trustworthy."""
        golds = ["BERT", "we use BERT base"]
        for prediction in (
            "BERT",
            "We use BERT base uncased throughout all of the reported experiments.",
            "The model is BERT.",
            "nothing relevant here",
        ):
            assert run_eval.answer_f1(prediction, golds) <= run_eval.length_ceiling(
                prediction, golds
            ) + 1e-9

    def test_empty_prediction_has_no_ceiling(self):
        assert run_eval.length_ceiling("", ["BERT"]) == 0.0


class TestSignificanceGuards:
    def test_small_difference_is_not_significant(self):
        """The guard added after a 3-of-30 keyword gap was reported as a verdict."""
        assert run_eval.two_proportion_p(18, 30, 15, 30) > 0.05

    def test_large_difference_is_significant(self):
        assert run_eval.two_proportion_p(280, 300, 150, 300) < 0.05

    def test_paired_test_detects_a_consistent_small_gain(self):
        """Pairing is the point: a gap far smaller than the spread still resolves."""
        a = [0.50, 0.62, 0.71, 0.44, 0.58, 0.66, 0.39, 0.75, 0.52, 0.61]
        b = [x - 0.05 for x in a]
        mean, p = run_eval.paired_p(a, b)
        assert mean == pytest.approx(0.05)
        assert p < 0.05

    def test_identical_arms_are_never_significant(self):
        scores = [0.1, 0.5, 0.9, 0.3]
        mean, p = run_eval.paired_p(scores, list(scores))
        assert mean == 0.0
        assert p == 1.0

    def test_too_few_samples_returns_no_verdict(self):
        assert run_eval.paired_p([0.5], [0.1]) == (0.0, 1.0)


class TestClusteredInference:
    """Questions from one paper are correlated. Treating them as independent counts
    information that is not there and reports a narrower interval than the data supports."""

    def test_clustered_interval_is_wider_than_the_naive_one(self):
        """The whole reason for clustering: 40 questions across 4 papers carry far less
        information than 40 independent questions, and the interval must show it."""
        a, b, papers = [], [], []
        for paper in range(4):
            # A per-paper offset — exactly the correlation clustering exists to handle.
            offset = 0.2 if paper % 2 else -0.2
            for _ in range(10):
                a.append(0.5 + offset)
                b.append(0.5)
                papers.append(f"paper_{paper}")

        _, naive_lo, naive_hi = (
            run_eval.paired_interval(a, b)[0],
            run_eval.paired_interval(a, b)[2],
            run_eval.paired_interval(a, b)[3],
        )
        _, clustered_lo, clustered_hi = run_eval.cluster_bootstrap(a, b, papers, n_boot=2000)
        assert (clustered_hi - clustered_lo) > (naive_hi - naive_lo)

    def test_bootstrap_brackets_the_observed_mean(self):
        a = [0.5, 0.6, 0.4, 0.7, 0.55, 0.65]
        b = [0.3, 0.4, 0.2, 0.5, 0.35, 0.45]
        papers = ["p1", "p1", "p2", "p2", "p3", "p3"]
        mean, lo, hi = run_eval.cluster_bootstrap(a, b, papers, n_boot=2000)
        assert mean == pytest.approx(0.2)
        assert lo <= mean <= hi

    def test_bootstrap_is_deterministic(self):
        a, b = [0.1, 0.9, 0.4, 0.6], [0.2, 0.3, 0.5, 0.1]
        papers = ["p1", "p1", "p2", "p2"]
        first = run_eval.cluster_bootstrap(a, b, papers, n_boot=500)
        assert first == run_eval.cluster_bootstrap(a, b, papers, n_boot=500)

    def test_permutation_p_is_high_when_arms_match(self):
        a = [0.5, 0.6, 0.4, 0.7]
        papers = ["p1", "p1", "p2", "p2"]
        assert run_eval.cluster_permutation_p(a, list(a), papers, n_perm=500) > 0.05

    def test_permutation_p_never_reports_zero(self):
        """The observed arrangement is itself one of the possibilities being counted."""
        a = [1.0] * 20
        b = [0.0] * 20
        papers = [f"p{i}" for i in range(20)]
        assert run_eval.cluster_permutation_p(a, b, papers, n_perm=200) > 0

    def test_single_cluster_yields_no_verdict(self):
        """One paper cannot support inference about papers in general."""
        a, b = [0.5, 0.6], [0.1, 0.2]
        assert run_eval.cluster_bootstrap(a, b, ["p1", "p1"]) == (0.0, 0.0, 0.0)


class TestConfigOverride:
    def test_sets_a_nested_value_with_the_right_type(self):
        from src.config import load_config

        cfg = load_config()
        run_eval.apply_override(cfg, "compression.max_keep=0.5")
        assert cfg["compression"]["max_keep"] == 0.5  # float, not the string "0.5"

    def test_unknown_key_fails_loudly(self):
        """A typo must not be silently ignored — the run would report a budget it never
        used."""
        from src.config import load_config

        with pytest.raises(KeyError):
            run_eval.apply_override(load_config(), "compression.max_kepe=0.5")

    def test_malformed_assignment_is_rejected(self):
        from src.config import load_config

        with pytest.raises(ValueError):
            run_eval.apply_override(load_config(), "compression.max_keep")

    def test_override_does_not_mutate_the_shared_default_config(self):
        """`default_config()` is lru_cached; mutating it would leak into other runs."""
        from src.config import default_config, load_config

        before = default_config()["compression"]["max_keep"]
        run_eval.apply_override(load_config(), "compression.max_keep=0.11")
        assert default_config()["compression"]["max_keep"] == before


class TestNullVersusUnderpowered:
    """A p-value alone conflates "no effect" with "not enough data". Those are opposite
    conclusions — one closes the question, the other says collect more — and the
    confidence interval is what tells them apart."""

    def test_a_tight_interval_around_zero_is_a_real_null(self):
        """Many questions, differences that scatter but average to ~nothing — the shape
        of the actual QASPER result at n=276."""
        a = [0.10 + 0.01 * (i % 11) for i in range(300)]
        b = [x + 0.01 * ((i % 7) - 3) / 3 for i, x in enumerate(a)]
        mean, p, lo, hi = run_eval.paired_interval(a, b)
        assert p > 0.05
        assert -0.02 < lo and hi < 0.02  # small enough to rule out a real effect

    def test_a_wide_interval_around_zero_is_merely_underpowered(self):
        a = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        b = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        mean, p, lo, hi = run_eval.paired_interval(a, b)
        assert p > 0.05
        assert lo < -0.02 or hi > 0.02  # cannot rule anything out

    def test_interval_brackets_the_mean_difference(self):
        a = [0.5, 0.6, 0.4, 0.55, 0.62]
        b = [0.3, 0.4, 0.2, 0.35, 0.42]
        mean, p, lo, hi = run_eval.paired_interval(a, b)
        assert lo < mean < hi
        assert mean == pytest.approx(0.2)


class TestThresholdCalibration:
    """The threshold that abstained on half of QASPER was hand-carried from another
    corpus. These pin the properties that make the replacement defensible."""

    def test_split_is_deterministic(self):
        assert calibrate.split_of("some_paper_id") == calibrate.split_of("some_paper_id")

    def test_split_is_by_paper_so_a_paper_never_straddles_it(self):
        """Questions about one paper share its vocabulary and retrieval behaviour;
        splitting by question would leak dev into test."""
        papers = [f"paper_{i}" for i in range(200)]
        for paper in papers:
            assert calibrate.split_of(paper) in ("dev", "test")
        dev = sum(calibrate.split_of(p) == "dev" for p in papers)
        assert 0.35 < dev / len(papers) < 0.65

    def test_sweep_reports_both_error_directions(self):
        records = [
            {"answerable": True, "weakest_claim_support": 0.30},
            {"answerable": True, "weakest_claim_support": 0.05},
            {"answerable": False, "weakest_claim_support": 0.02},
            {"answerable": False, "weakest_claim_support": 0.40},
        ]
        row = calibrate.sweep(records, [0.10])[0]
        assert row["false_abstain_rate"] == pytest.approx(0.5)   # 0.05 wrongly abstains
        assert row["correct_abstain_rate"] == pytest.approx(0.5)  # 0.02 rightly abstains

    def test_balanced_accuracy_is_not_gamed_by_never_abstaining(self):
        """The reason accuracy is the wrong objective: the set is 276 answerable to 11
        unanswerable, so a threshold of 0 scores 96% accuracy by answering everything."""
        records = [{"answerable": True, "weakest_claim_support": 0.5}] * 90
        records += [{"answerable": False, "weakest_claim_support": 0.5}] * 10
        row = calibrate.sweep(records, [0.0])[0]
        assert row["answer_rate_on_answerable"] == 1.0
        assert row["balanced_accuracy"] == pytest.approx(0.5)


class TestCeilingReporting:
    def test_arm_reports_quality_relative_to_what_was_achievable(self):
        arm = run_eval.ArmResult(mode="identity")
        arm.f1_scores = [0.15, 0.15]
        arm.ceilings = [0.30, 0.30]
        assert arm.mean_ceiling == pytest.approx(0.30)
        assert arm.f1_vs_ceiling == pytest.approx(0.5)

    def test_no_data_does_not_divide_by_zero(self):
        assert run_eval.ArmResult(mode="identity").f1_vs_ceiling == 0.0
