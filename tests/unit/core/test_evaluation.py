# mypy: ignore-errors
# flake8: noqa: F821

from types import SimpleNamespace

import pytest

from ae.core.evaluation import ExperimentMatcher, RecordMatcher
from ae.core.evaluation.matcher import (
    _SEMANTIC_JUDGE_EXAMPLE,
    _clean_judge_response,
    _extract_field_verdict,
    _extract_first_json,
    _parse_judge_json,
)


@pytest.fixture(autouse=True)
def _setup_experiment_model(nanozyme_task, request):
    """Setup experiment_model at module level for matcher tests."""
    request.module.experiment_model = nanozyme_task["experiment_model"]


@pytest.mark.unit
class TestEvaluationEngine:
    """Consolidated unit tests for the evaluation matcher and judges."""

    def test_string_normalization(self):
        """Test normalization variants (whitespace, dash characters, case-sensitive)."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)
        assert matcher._normalize_text("Fe3O4") == "Fe3O4"
        assert matcher._normalize_text("  Fe3O4  ") == "Fe3O4"
        assert matcher._normalize_text("Fe−3O4") == matcher._normalize_text("Fe-3O4")
        assert matcher._normalize_text("Fe 3 O4") == "Fe3O4"
        assert matcher._normalize_text(None) == ""
        assert matcher._normalize_text(123) == "123"

    @pytest.mark.parametrize("pred,gold,expected", [
        (0.05, 0.05, True),
        (0.05, 0.050000000001, True),
        (0.054, 0.05, False),
        (1e-10, 0.0, False),
    ])
    def test_float_comparison(self, pred, gold, expected):
        """Test float comparison using strict tolerance (rel_tol=1e-9)."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        assert matcher._compare_floats(pred, gold) is expected

    def test_general_value_matching(self):
        """Test general is_match logic for mixed types, None/NaN values and strings."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)
        assert matcher._is_match("Fe3O4", "Fe3O4") is True
        assert matcher._is_match("FE3O4", "fe3o4") is False
        assert matcher._is_match(None, None) is True
        assert matcher._is_match(float("nan"), None) is True
        assert matcher._is_match(None, float("nan")) is True
        assert matcher._is_match("Fe3O4", None) is False
        assert matcher._is_match("10.0", 10.0) is True
        # Distinct string representations in str-typed fields remain distinct
        assert matcher._is_match("1.1", "1.10") is False
        assert matcher._is_match("1,2", "1.2") is False
        assert matcher._is_match("None", None) is False

    def test_hungarian_pairs_alignment(self):
        """Test alignment of predictions and ground truths via Hungarian algorithm."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        # Empty
        assert matcher.align_pairs([], []) == []

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="Au", activity="catalase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="ZnO", activity="catalase"),
        ]

        pairs = matcher.align_pairs(preds, gts)
        assert len(pairs) == 2
        # Fe3O4 should be paired with Fe3O4
        matched = [p for p in pairs if p[0] and p[1] and p[0].formula == "Fe3O4"]
        assert len(matched) == 1

    def test_f1_computation(self):
        """Test detailed precision, recall, and F1 score reporting."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]

        report = matcher.get_detailed_report(preds, gts)
        assert report["precision"] == 1.0
        assert report["recall"] == 0.5
        assert report["f1"] == 2/3

        # Fields scores
        assert report["fields"]["formula"] == 0.5
        assert report["fields"]["activity"] == 0.5

    def test_matcher_validation_rules(self):
        """Test initialization constraints for comparison fields and tolerance."""
        with pytest.raises(ValueError, match="fields_to_compare"):
            ExperimentMatcher(fields_to_compare=[], float_tolerance=0.05)

        with pytest.raises(ValueError, match="float_tolerance"):
            ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=-0.1)

    def test_semantic_judge(self):
        """Test Semantic Judge workflow, prompts, mock verdicts, and exceptions."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )
        # Verify it works correctly with judge disabled
        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        report = matcher.get_detailed_report(preds, gts)
        assert report["f1"] == 1.0

        # Mock LLM that returns YES verdict
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                return ['{"formula": "YES"}']

        matcher_with_judge = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            student_llm=MockLLM(),
            enable_semantic_judge=True,
        )

        verdicts = matcher_with_judge._call_semantic_judge(
            task_name="test",
            gt_json={"formula": "Fe3O4"},
            pred_json={"formula": "iron oxide"},
            discrepancies=["formula"],
        )
        assert verdicts == {"formula": "YES"}

        # Exception fallback
        class FailLLM:
            def __call__(self, prompt, **kwargs):
                raise RuntimeError("LLM Failure")

        matcher_fail = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            student_llm=FailLLM(),
            enable_semantic_judge=True,
        )
        assert matcher_fail._call_semantic_judge("test", {"f": "A"}, {"f": "B"}, ["f"]) == {}


@pytest.mark.unit
class TestExtractFirstJson:
    """Tests for the brace-balancing JSON extractor utility."""

    def test_balanced_json_parsing(self):
        """Test extraction of clean, nested, and markdown-wrapped JSON."""
        assert _extract_first_json('{"a": 1}') == '{"a": 1}'
        assert _extract_first_json('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert _extract_first_json('{"outer": {"inner": 2}}') == '{"outer": {"inner": 2}}'
        assert _extract_first_json('{"a": 1} trailing text') == '{"a": 1}'
        assert _extract_first_json('just text') is None
        assert _extract_first_json('{"formula": "Fe{3}O4"}') == '{"formula": "Fe{3}O4"}'


@pytest.mark.unit
class TestSemanticJudgeAuditImprovements:
    """Tests for domain independence, cost optimization, caching, batching, and robustness."""

    def test_judge_prompt_domain_neutral_and_schema_compliant(self):
        """Judge prompt and examples must be domain-agnostic and enforce schema constraints."""
        for forbidden in ("km_value", "μM", "\\u03bcM", "mM", "Deductive Equivalence", "Implicit Nulls", "IRRELEVANT"):
            assert forbidden not in _SEMANTIC_JUDGE_EXAMPLE

        matcher = RecordMatcher(
            fields_to_compare=["endpoint", "code"],
            float_tolerance=0.05,
            field_descriptions={"endpoint": "Primary clinical endpoint", "code": "Record identifier"},
            field_constraints={
                "endpoint": {"choices": ["OS", "PFS"]},
                "code": {"pattern": r"^CT-\d+$"},
            },
        )
        prompt = matcher._build_judge_system_prompt("clinical_trials", is_batch=False)
        for forbidden in ("km_value", "μM", "mM", "Deductive Equivalence", "Implicit Nulls", "IRRELEVANT"):
            assert forbidden not in prompt
        assert "CRITICAL SCHEMA COMPLIANCE RULE" in prompt
        assert "allowed choices: ['OS', 'PFS']" in prompt
        assert "required regex pattern: ^CT-\\d+$" in prompt

        batch_prompt = matcher._build_judge_system_prompt("clinical_trials", is_batch=True)
        assert 'pair IDs ("0", "1", ...)' in batch_prompt

    def test_short_circuit_pure_miss_and_pure_addition(self):
        """Pure Miss (Pred is None/empty) and Pure Addition (GT is None) must never invoke LLM judge."""
        calls = []

        class SpyLLM:
            def __call__(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return ['{"formula": "YES", "activity": "YES"}']

        matcher = RecordMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            student_llm=SpyLLM(),
            enable_semantic_judge=True,
        )

        # Direct _call_semantic_judge short-circuit
        direct_verdicts = matcher._call_semantic_judge(
            task_name="materials",
            gt_json={"formula": None, "activity": "peroxidase"},
            pred_json={"formula": "extra", "activity": None},
            discrepancies=["formula", "activity"],
        )
        assert direct_verdicts == {"formula": "NO", "activity": "NO"}
        assert len(calls) == 0

        # Pure Miss (None and empty string) and Pure Addition inside get_detailed_report
        preds = [
            SimpleNamespace(formula="Fe3O4", activity=None),
            SimpleNamespace(formula="CuO", activity=""),
            SimpleNamespace(formula="ZnO", activity="hallucinated"),
        ]
        gts = [
            SimpleNamespace(formula="Fe3O4", activity="peroxidase"),
            SimpleNamespace(formula="CuO", activity="oxidase"),
            SimpleNamespace(formula="ZnO", activity=None),
        ]
        report = matcher.get_detailed_report(preds, gts, task_name="materials")
        assert len(calls) == 0
        # 3 strict formula matches (TP=3), 2 missed activities (FN=2, including "" mismatch FP+FN? Wait: "" is pred!=None, gt!=None -> FP=1,FN=1), 1 hallucinated activity (FP=1)
        assert report["precision"] < 1.0
        assert report["recall"] < 1.0

    def test_verdict_cache_deduplication_and_coupled_keys(self):
        """Identical discrepancies hit in-memory cache; differing coupled unit keys do not collide."""
        calls = []

        class CountingLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                return ['{"target_value": {"reasoning": "ok", "match": "YES"}, "target_units": {"reasoning": "ok", "match": "YES"}}']

        matcher = RecordMatcher(
            fields_to_compare=["compound_id", "target_value", "target_units"],
            float_tolerance=0.05,
            student_llm=CountingLLM(),
            enable_semantic_judge=True,
        )

        preds = [
            SimpleNamespace(compound_id="C1", target_value="0.091", target_units="mM"),
            SimpleNamespace(compound_id="C2", target_value="0.091", target_units="mM"),
        ]
        gts = [
            SimpleNamespace(compound_id="C1", target_value="91", target_units="uM"),
            SimpleNamespace(compound_id="C2", target_value="91", target_units="uM"),
        ]
        report1 = matcher.get_detailed_report(preds, gts, task_name="bioassay")
        assert report1["f1"] == 1.0
        assert len(calls) == 1  # Deduplicated across both rows in the document

        # Re-evaluating across trials hits cache
        report2 = matcher.get_detailed_report(preds, gts, task_name="bioassay")
        assert report2["f1"] == 1.0
        assert len(calls) == 1

        # Changing coupled unit field produces a distinct cache key and triggers a new call
        preds_diff_unit = [SimpleNamespace(compound_id="C1", target_value="0.091", target_units="nM")]
        gts_diff_unit = [SimpleNamespace(compound_id="C1", target_value="91", target_units="uM")]
        matcher.get_detailed_report(preds_diff_unit, gts_diff_unit, task_name="bioassay")
        assert len(calls) == 2

        matcher.clear_cache()
        matcher.get_detailed_report(preds[:1], gts[:1], task_name="bioassay")
        assert len(calls) == 3

    def test_batch_judge_and_chunking_in_compute_stats(self):
        """Multiple distinct discrepancies in one document are batched into minimal LLM calls."""
        calls = []

        class BatchLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                if len(calls) == 1:
                    return [
                        '{"0": {"name": {"reasoning": "abbr", "match": "YES"}}, '
                        '"1": {"name": {"reasoning": "diff", "match": "NO"}}}'
                    ]
                return ['{"name": {"reasoning": "bool", "match": true}}']

        matcher = RecordMatcher(
            fields_to_compare=["id", "name"],
            float_tolerance=0.05,
            student_llm=BatchLLM(),
            enable_semantic_judge=True,
            max_judge_batch_size=2,
        )

        preds = [
            SimpleNamespace(id="1", name="S. aureus"),
            SimpleNamespace(id="2", name="E. coli"),
            SimpleNamespace(id="3", name="B. subtilis"),
        ]
        gts = [
            SimpleNamespace(id="1", name="Staphylococcus aureus"),
            SimpleNamespace(id="2", name="Pseudomonas aeruginosa"),
            SimpleNamespace(id="3", name="Bacillus subtilis"),
        ]

        report = matcher.get_detailed_report(preds, gts, task_name="microbiology")
        # 3 unique pairs with batch_size=2 -> 1 batch call (pairs 0 & 1) + 1 single call (pair 2)
        assert len(calls) == 2
        assert '=== PAIR "0" ===' in calls[0][1]["content"]
        assert '=== PAIR "1" ===' in calls[0][1]["content"]
        # TP = 3 (ids) + 2 (names 0 and 2) = 5; FP = 1 (name 1); FN = 1 (name 1)
        assert report["precision"] == pytest.approx(5 / 6)
        assert report["recall"] == pytest.approx(5 / 6)

    def test_batch_judge_failure_no_retry_storm_or_cache_poisoning(self):
        """If a batched LLM call fails, matcher falls back cleanly without retrying N individual calls."""
        calls = []

        class BrokenBatchLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                raise RuntimeError("API error")

        matcher = RecordMatcher(
            fields_to_compare=["id", "name"],
            float_tolerance=0.05,
            student_llm=BrokenBatchLLM(),
            enable_semantic_judge=True,
        )
        preds = [
            SimpleNamespace(id="1", name="Alpha"),
            SimpleNamespace(id="2", name="Beta"),
        ]
        gts = [
            SimpleNamespace(id="1", name="A"),
            SimpleNamespace(id="2", name="B"),
        ]
        report = matcher.get_detailed_report(preds, gts, task_name="test")
        assert len(calls) == 1  # No individual fallback storm
        assert len(matcher._verdict_cache) == 0  # Failure not cached
        assert report["precision"] == pytest.approx(2 / 4)

    def test_response_cleaning_json_robustness_and_no_forced_kwargs(self):
        """Strip <think>/<thinking>/dangling </think>, handle booleans/trailing commas, avoid forced kwargs."""
        assert _clean_judge_response("<think>internal {brace}</think>\n{\"a\": \"YES\"}") == '{"a": "YES"}'
        assert _clean_judge_response("<THINKING>plan</THINKING> {\"a\": \"NO\"}") == '{"a": "NO"}'
        assert _clean_judge_response("reasoning with {fake: json}\n</think>\n{\"a\": true}") == '{"a": true}'

        parsed = _parse_judge_json('{"a": {"reasoning": "ok", "match": true,}, "b": {"match": false}}')
        assert parsed is not None
        assert _extract_field_verdict(parsed["a"]) == "YES"
        assert _extract_field_verdict(parsed["b"]) == "NO"
        assert _extract_field_verdict("true") == "YES"

        received_kwargs = []

        class KwargsSpyLLM:
            def __call__(self, messages, **kwargs):
                received_kwargs.append(kwargs)
                return ['internal {noise}</think>\n{"val": {"reasoning": "same", "match": true,}}']

        matcher = RecordMatcher(
            fields_to_compare=["val"],
            float_tolerance=0.05,
            student_llm=KwargsSpyLLM(),
            enable_semantic_judge=True,
        )
        verdicts = matcher._call_semantic_judge("test", {"val": "100"}, {"val": "1e2"}, ["val"])
        assert verdicts == {"val": "YES"}
        assert received_kwargs == [{}]  # No forced reasoning/enable_thinking/prompt_caching kwargs

    def test_schema_field_constraints_reject_without_llm(self):
        """Predictions violating choices/pattern/range constraints are rejected deterministically."""
        calls = []

        class SpyLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                return ['{"target_type": {"reasoning": "ok", "match": "YES"}}']

        matcher = RecordMatcher(
            fields_to_compare=["compound_id", "target_type"],
            float_tolerance=0.05,
            student_llm=SpyLLM(),
            field_constraints={
                "compound_id": {"pattern": r"^C\d+$"},
                "target_type": {"choices": ["MIC", "MBC"]},
            },
            enable_semantic_judge=True,
        )

        # Violates both pattern and choices -> 0 LLM calls
        verdicts = matcher._call_semantic_judge(
            task_name="assay",
            gt_json={"compound_id": "C1", "target_type": "MIC"},
            pred_json={"compound_id": "invalid_id", "target_type": "IC50"},
            discrepancies=["compound_id", "target_type"],
        )
        assert verdicts == {"compound_id": "NO", "target_type": "NO"}
        assert len(calls) == 0

        # Valid choice different from GT -> LLM IS called
        verdicts2 = matcher._call_semantic_judge(
            task_name="assay",
            gt_json={"compound_id": "C1", "target_type": "MIC"},
            pred_json={"compound_id": "C1", "target_type": "MBC"},
            discrepancies=["target_type"],
        )
        assert verdicts2 == {"target_type": "YES"}
        assert len(calls) == 1

    def test_align_pairs_soft_similarity_and_shared_null_tie_breaking(self):
        """Hungarian alignment must prioritize real matches over shared None and break ties cleanly."""
        matcher = RecordMatcher(
            fields_to_compare=["id", "value", "note1", "note2", "note3"],
            float_tolerance=0.05,
        )
        # Pred 0 matches GT 1 on real field 'id' and has scaling/case match on 'value';
        # Pred 0 shares 3 None fields with GT 0, which previously outweighed 1 real match!
        preds = [
            SimpleNamespace(id="A1", value="0.091", note1=None, note2=None, note3=None),
            SimpleNamespace(id="B2", value="50", note1="x", note2="y", note3="z"),
        ]
        gts = [
            SimpleNamespace(id="Z9", value="999", note1=None, note2=None, note3=None),
            SimpleNamespace(id="A1", value="91", note1="k1", note2="k2", note3="k3"),
        ]
        pairs = matcher.align_pairs(preds, gts)
        paired_by_pred = {p.id: g.id for p, g in pairs if p is not None and g is not None}
        assert paired_by_pred == {"A1": "A1"}
        # Unrelated B2 (FP) and Z9 (FN) with 0 similarity are split by the default cutoff
        assert any(p is not None and p.id == "B2" and g is None for p, g in pairs)
        assert any(p is None and g is not None and g.id == "Z9" for p, g in pairs)

        # Explicitly disabling cutoff (min_alignment_score=0.0) retains forced pairing
        matcher_nocut = RecordMatcher(
            fields_to_compare=["id", "value", "note1", "note2", "note3"],
            float_tolerance=0.05,
            min_alignment_score=0.0,
        )
        pairs_nocut = matcher_nocut.align_pairs(preds, gts)
        paired_nocut = {p.id: g.id for p, g in pairs_nocut if p is not None and g is not None}
        assert paired_nocut["A1"] == "A1"
        assert paired_nocut["B2"] == "Z9"

    def test_field_validators_canonical_equivalence_and_deterministic_reject(self):
        """SMILES validator canonicalization resolves equivalent forms and rejects mismatches with 0 judge calls."""
        pytest.importorskip("rdkit")
        from ae.core.schema.validators import FieldValidatorSpec

        calls = []

        class SpyLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                return ['{"smiles": {"reasoning": "ok", "match": "YES"}}']

        matcher = RecordMatcher(
            fields_to_compare=["compound_id", "smiles"],
            float_tolerance=0.05,
            student_llm=SpyLLM(),
            enable_semantic_judge=True,
            field_validators={"smiles": [FieldValidatorSpec(name="smiles", on_error="warn")]},
        )

        # Kekule vs aromatic SMILES for benzimidazole should match deterministically (0 judge calls)
        preds_eq = [SimpleNamespace(compound_id="1a", smiles="c1ccc2[nH]cnc2c1")]
        gts_eq = [SimpleNamespace(compound_id="1a", smiles="C1=CC=C2C(=C1)NC=N2")]
        rep_eq = matcher.get_detailed_report(preds_eq, gts_eq, task_name="chemistry")
        assert rep_eq["f1"] == 1.0
        assert len(calls) == 0

        # Non-isomorphic SMILES must fail deterministically without calling LLM judge
        preds_diff = [SimpleNamespace(compound_id="1a", smiles="c1ccccc1")]
        gts_diff = [SimpleNamespace(compound_id="1a", smiles="Cc1ccccc1")]
        rep_diff = matcher.get_detailed_report(preds_diff, gts_diff, task_name="chemistry")
        assert rep_diff["fields"]["smiles"] == 0.0
        assert len(calls) == 0

    def test_evaluate_pairs_returns_indices_and_respects_equivalence_fn(self):
        """evaluate_pairs returns pred/gt indices, mismatch reasons, and short-circuits via field_equivalence_fn."""
        calls = []

        class SpyLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                return ['{"bacteria": {"reasoning": "diff", "match": "NO"}}']

        matcher = RecordMatcher(
            fields_to_compare=["compound_id", "target_value", "bacteria"],
            float_tolerance=0.05,
            student_llm=SpyLLM(),
            enable_semantic_judge=True,
        )

        preds = [
            SimpleNamespace(compound_id="5a", target_value="0.030", bacteria="E. coli"),
            SimpleNamespace(compound_id="5b", target_value="10", bacteria="P. aeruginosa"),
            SimpleNamespace(compound_id="CTRL", target_value="1", bacteria="E. coli"),
        ]
        gts = [
            SimpleNamespace(compound_id="5a", target_value="0.03", bacteria="E. coli"),
            SimpleNamespace(compound_id="5b", target_value="10", bacteria="S. aureus"),
        ]

        def equiv_fn(field_name, p_val, g_val):
            return field_name == "target_value" and float(str(p_val)) == float(str(g_val))

        res = matcher.evaluate_pairs(
            preds, gts, task_name="bioassay", field_equivalence_fn=equiv_fn
        )
        statuses = {
            (r["pred_index"], r["gt_index"]): r["status"] for r in res["pair_results"]
        }
        assert statuses[(0, 0)] == "tp"
        assert statuses[(1, 1)] == "field_mismatch"
        assert statuses[(2, None)] == "fp"

        mismatch_row = next(r for r in res["pair_results"] if r["status"] == "field_mismatch")
        assert list(mismatch_row["mismatched_fields"].keys()) == ["bacteria"]
        assert mismatch_row["mismatched_fields"]["bacteria"]["reason"] == "value_mismatch"
        # Only 1 judge call for bacteria on (1, 1); target_value on (0, 0) was resolved by equiv_fn
        assert len(calls) == 1

        # Also verify FN index when len(gts) > len(preds)
        fn_res = matcher.evaluate_pairs(
            preds[:1],
            gts + [SimpleNamespace(compound_id="5c", target_value="25", bacteria="S. aureus")],
            task_name="bioassay",
            field_equivalence_fn=equiv_fn,
        )
        fn_statuses = {(r["pred_index"], r["gt_index"]): r["status"] for r in fn_res["pair_results"]}
        assert fn_statuses[(0, 0)] == "tp"
        assert any(k[0] is None and k[1] is not None and v == "fn" for k, v in fn_statuses.items())

    def test_smiles_unassigned_stereo_vs_conflict_reason(self):
        """Unassigned E/Z stereo matches as TP; explicit (E) vs (Z) conflict yields reason='stereo_diff'."""
        pytest.importorskip("rdkit")
        from ae.core.schema.validators import FieldValidatorSpec

        calls = []

        class SpyLLM:
            def __call__(self, messages, **kwargs):
                calls.append(messages)
                return ['{"smiles": {"reasoning": "ok", "match": "YES"}}']

        for judge_enabled in (True, False):
            matcher = RecordMatcher(
                fields_to_compare=["compound_id", "smiles"],
                float_tolerance=0.05,
                student_llm=SpyLLM() if judge_enabled else None,
                enable_semantic_judge=judge_enabled,
                field_validators={"smiles": [FieldValidatorSpec(name="smiles", on_error="warn")]},
            )

            # Flat SMILES vs E/Z GT -> TP
            res_tp = matcher.evaluate_pairs(
                [SimpleNamespace(compound_id="12", smiles="CC=CC")],
                [SimpleNamespace(compound_id="12", smiles="C/C=C/C")],
                task_name="chemistry",
            )
            assert res_tp["pair_results"][0]["status"] == "tp"
            assert res_tp["f1"] == 1.0

            # Explicit (E) vs (Z) conflict -> field_mismatch with reason="stereo_diff" and 0 judge calls
            res_diff = matcher.evaluate_pairs(
                [SimpleNamespace(compound_id="12", smiles="C/C=C/C")],
                [SimpleNamespace(compound_id="12", smiles="C/C=C\\C")],
                task_name="chemistry",
            )
            assert res_diff["pair_results"][0]["status"] == "field_mismatch"
            assert res_diff["pair_results"][0]["mismatched_fields"]["smiles"]["reason"] == "stereo_diff"
            assert len(calls) == 0


@pytest.mark.unit
class TestExp03DualMetricsCutoffAndPotDiagnostics:
    """Tests for EXP-03 dual strict/semantic F1, soft matches, cache pinning, cutoff, and POT."""

    def test_dual_strict_and_semantic_f1_soft_matches_and_judge_reasoning(self):
        """One evaluation pass yields strict_f1, semantic_f1, soft_matched_fields, and judge_reasoning."""
        pytest.importorskip("rdkit")
        pytest.importorskip("pint")
        from ae.core.schema.validators import FieldValidatorSpec

        class ReasonLLM:
            model = "judge-model-v1"

            def __call__(self, messages, **kwargs):
                return [
                    '{"bacteria": {"reasoning": "Punctuation artifact in genus name", "match": "YES"}}'
                ]

        matcher = RecordMatcher(
            fields_to_compare=["compound_id", "smiles", "target_units", "bacteria"],
            float_tolerance=0.05,
            student_llm=ReasonLLM(),
            enable_semantic_judge=True,
            field_validators={
                "smiles": [FieldValidatorSpec(name="smiles", on_error="warn")],
                "target_units": [FieldValidatorSpec(name="unit", on_error="warn")],
            },
        )

        preds = [
            SimpleNamespace(
                compound_id="1a",
                smiles="CC=CC",
                target_units="µg mL-1",
                bacteria="Staphylococcus. aureus",
            )
        ]
        gts = [
            SimpleNamespace(
                compound_id="1a",
                smiles="C/C=C/C",
                target_units="µg/mL",
                bacteria="Staphylococcus aureus",
            )
        ]

        rep = matcher.get_detailed_report(preds, gts, task_name="antibiotics")
        # 3 deterministic matches (compound_id, smiles via ignore_unassigned_stereo, target_units via pint)
        # + 1 bacteria mismatch (FP=1, FN=1) -> strict precision=3/4, recall=3/4, f1=0.75
        assert rep["strict_f1"] == pytest.approx(0.75)
        assert rep["semantic_f1"] == pytest.approx(1.0)
        assert rep["f1"] == pytest.approx(1.0)
        assert rep["soft_matched_fields_count"] == 3

        pair = rep["pair_results"][0]
        assert pair["strict_matched_fields"] == ["compound_id", "smiles", "target_units"]
        assert set(pair["soft_matched_fields"]) == {"smiles", "target_units", "bacteria"}
        assert pair["soft_match_details"]["smiles"]["reason"] == "ignore_unassigned_stereo"
        assert pair["soft_match_details"]["target_units"]["reason"] == "pint_unit_equivalence"
        assert pair["judge_reasoning"]["bacteria"] == "Punctuation artifact in genus name"
        assert len(rep["judge_reasonings"]) == 1

    def test_cache_key_pinned_to_judge_model_and_schema_constraints_hash(self):
        """Cache key includes (judge_model_id, schema_constraints_hash) preventing cross-config collisions."""
        class ModelA:
            model = "model-a"

            def __call__(self, messages, **kwargs):
                return ['{"bacteria": {"reasoning": "a", "match": "YES"}}']

        class ModelB:
            model = "model-b"

            def __call__(self, messages, **kwargs):
                return ['{"bacteria": {"reasoning": "b", "match": "NO"}}']

        m1 = RecordMatcher(
            fields_to_compare=["bacteria"],
            float_tolerance=0.05,
            student_llm=ModelA(),
            field_constraints={"bacteria": {"choices": ["S. aureus", "E. coli"]}},
        )
        m2 = RecordMatcher(
            fields_to_compare=["bacteria"],
            float_tolerance=0.05,
            student_llm=ModelB(),
            field_constraints={"bacteria": {"choices": ["S. aureus", "E. coli"]}},
        )
        m3 = RecordMatcher(
            fields_to_compare=["bacteria"],
            float_tolerance=0.05,
            student_llm=ModelA(),
            field_constraints={"bacteria": {"choices": ["S. aureus", "MRSA"]}},
        )

        k1 = m1._make_discrepancy_cache_key("t", {"bacteria": "S. aureus"}, {"bacteria": "S.a."}, ["bacteria"])
        k2 = m2._make_discrepancy_cache_key("t", {"bacteria": "S. aureus"}, {"bacteria": "S.a."}, ["bacteria"])
        k3 = m3._make_discrepancy_cache_key("t", {"bacteria": "S. aureus"}, {"bacteria": "S.a."}, ["bacteria"])
        assert k1[0] == "model-a"
        assert k2[0] == "model-b"
        assert k1 != k2
        assert k1[1] != k3[1]
        assert k1 != k3

    def test_hungarian_cutoff_splits_unrelated_fp_fn_sharing_constant_columns(self):
        """Records sharing only constant columns (MIC, =, µg/mL) are not glued into field_mismatch."""
        matcher = RecordMatcher(
            fields_to_compare=[
                "compound_id",
                "target_type",
                "target_relation",
                "target_value",
                "target_units",
                "bacteria",
            ],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )
        preds = [
            SimpleNamespace(
                compound_id="1a",
                target_type="MIC",
                target_relation="=",
                target_value="8",
                target_units="µg/mL",
                bacteria="S. aureus",
            ),
            SimpleNamespace(
                compound_id="1b",
                target_type="MIC",
                target_relation="=",
                target_value="16",
                target_units="µg/mL",
                bacteria="E. coli",
            ),
            SimpleNamespace(
                compound_id="Penicillin",
                target_type="MIC",
                target_relation="=",
                target_value="0.5",
                target_units="µg/mL",
                bacteria="B. subtilis",
            ),
        ]
        gts = [
            SimpleNamespace(
                compound_id="1a",
                target_type="MIC",
                target_relation="=",
                target_value="8",
                target_units="µg/mL",
                bacteria="S. aureus",
            ),
            SimpleNamespace(
                compound_id="1b",
                target_type="MIC",
                target_relation="=",
                target_value="16",
                target_units="µg/mL",
                bacteria="E. coli",
            ),
            SimpleNamespace(
                compound_id="9z",
                target_type="MIC",
                target_relation="=",
                target_value="128",
                target_units="µg/mL",
                bacteria="P. aeruginosa",
            ),
        ]
        res = matcher.evaluate_pairs(preds, gts, task_name="antibiotics")
        statuses = [r["status"] for r in res["pair_results"]]
        assert statuses.count("tp") == 2
        assert statuses.count("field_mismatch") == 0
        assert statuses.count("fp") == 1
        assert statuses.count("fn") == 1

    def test_pot_rapidfuzz_posthoc_tags_duplicate_merge_and_split(self):
        """POT unbalanced OT + RapidFuzz detects duplicate_of_tp, row_merge, and row_split."""
        matcher = RecordMatcher(
            fields_to_compare=["compound_id", "target_value", "bacteria"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )

        # 1. duplicate_of_tp
        preds_dup = [
            SimpleNamespace(compound_id="8b", target_value="8", bacteria="S. aureus ATCC 29213"),
            SimpleNamespace(compound_id="8b", target_value="8", bacteria="MSSA S. aureus ATCC 29213"),
        ]
        gts_dup = [
            SimpleNamespace(compound_id="8b", target_value="8", bacteria="S. aureus ATCC 29213"),
        ]
        res_dup = matcher.evaluate_pairs(preds_dup, gts_dup)
        assert len(res_dup["diagnostics"]["duplicate_of_tp"]) == 1
        assert res_dup["diagnostics"]["duplicate_of_tp"][0]["pred_index"] == 1
        assert res_dup["pair_results"][1]["diagnostic_tag"] == "duplicate_of_tp"

        # 2. row_merge (1 pred merges 2 GT rows)
        preds_merge = [
            SimpleNamespace(compound_id="1a", target_value="8", bacteria="S. aureus, E. coli"),
        ]
        gts_merge = [
            SimpleNamespace(compound_id="1a", target_value="8", bacteria="S. aureus"),
            SimpleNamespace(compound_id="1a", target_value="8", bacteria="E. coli"),
        ]
        res_merge = matcher.evaluate_pairs(preds_merge, gts_merge)
        assert len(res_merge["diagnostics"]["row_merge"]) == 1
        assert set(res_merge["diagnostics"]["row_merge"][0]["gt_indices"]) == {0, 1}
        assert any(r.get("diagnostic_tag") == "row_merge" for r in res_merge["pair_results"])

        # 3. row_split (2 preds split 1 GT row)
        preds_split = [
            SimpleNamespace(compound_id="1a", target_value="8", bacteria="S. aureus"),
            SimpleNamespace(compound_id="1a", target_value="8", bacteria="E. coli"),
        ]
        gts_split = [
            SimpleNamespace(compound_id="1a", target_value="8", bacteria="S. aureus, E. coli"),
        ]
        res_split = matcher.evaluate_pairs(preds_split, gts_split)
        assert len(res_split["diagnostics"]["row_split"]) == 1
        assert set(res_split["diagnostics"]["row_split"][0]["pred_indices"]) == {0, 1}
        assert any(r.get("diagnostic_tag") == "row_split" for r in res_split["pair_results"])
