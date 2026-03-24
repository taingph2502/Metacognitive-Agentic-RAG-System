"""Tests for the metacognitive evaluation module (Paper §4)."""

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring — satisfaction gate (Paper §4.1)
# ──────────────────────────────────────────────────────────────────────────────

def test_monitoring_passes_good_answer():
    from app.agent.metacognitive import is_answer_satisfactory

    assert is_answer_satisfactory(
        faithfulness=0.85,
        completeness=0.75,
        citation_precision=0.6,
    ) is True


def test_monitoring_flags_low_faithfulness():
    from app.agent.metacognitive import is_answer_satisfactory

    assert is_answer_satisfactory(
        faithfulness=0.4,
        completeness=0.8,
        citation_precision=0.7,
    ) is False


def test_monitoring_flags_low_completeness():
    from app.agent.metacognitive import is_answer_satisfactory

    assert is_answer_satisfactory(
        faithfulness=0.8,
        completeness=0.3,
        citation_precision=0.5,
    ) is False


def test_monitoring_flags_low_citation_precision():
    from app.agent.metacognitive import is_answer_satisfactory

    assert is_answer_satisfactory(
        faithfulness=0.8,
        completeness=0.7,
        citation_precision=0.2,
    ) is False


# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis — heuristic fallback (no LLM needed)
# ──────────────────────────────────────────────────────────────────────────────

def test_heuristic_diagnosis_insufficient():
    from app.agent.metacognitive import INSUFFICIENT, _heuristic_diagnosis

    diag = _heuristic_diagnosis("test query", faithfulness=0.2, completeness=0.2, citation_precision=0.1)
    assert diag.category == INSUFFICIENT
    assert diag.suggested_query is not None


def test_heuristic_diagnosis_external_only():
    from app.agent.metacognitive import EXTERNAL_ONLY, _heuristic_diagnosis

    diag = _heuristic_diagnosis("test query", faithfulness=0.3, completeness=0.6, citation_precision=0.2)
    assert diag.category == EXTERNAL_ONLY


def test_heuristic_diagnosis_reasoning_error_incomplete():
    from app.agent.metacognitive import ERROR_INCOMPLETE, REASONING_ERROR, _heuristic_diagnosis

    diag = _heuristic_diagnosis("test query", faithfulness=0.6, completeness=0.3, citation_precision=0.5)
    assert diag.category == REASONING_ERROR
    assert ERROR_INCOMPLETE in diag.error_types


# ──────────────────────────────────────────────────────────────────────────────
# Planning — writing directives (Paper §4.3)
# ──────────────────────────────────────────────────────────────────────────────

def test_directive_satisfactory_is_none():
    from app.agent.metacognitive import SATISFACTORY, Diagnosis, build_writing_directive

    diag = Diagnosis(category=SATISFACTORY)
    assert build_writing_directive(diag) is None


def test_directive_internal_only_discards_refs():
    from app.agent.metacognitive import INTERNAL_ONLY, Diagnosis, build_writing_directive

    diag = Diagnosis(category=INTERNAL_ONLY)
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "own knowledge" in directive.lower() or "intrinsic" in directive.lower() or "rely primarily" in directive.lower()


def test_directive_external_only_forces_refs():
    from app.agent.metacognitive import EXTERNAL_ONLY, Diagnosis, build_writing_directive

    diag = Diagnosis(category=EXTERNAL_ONLY)
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "only" in directive.lower()
    assert "source documents" in directive.lower() or "references" in directive.lower()


def test_directive_reasoning_error_incomplete():
    from app.agent.metacognitive import (
        ERROR_INCOMPLETE,
        REASONING_ERROR,
        Diagnosis,
        build_writing_directive,
    )

    diag = Diagnosis(category=REASONING_ERROR, error_types=[ERROR_INCOMPLETE])
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "incomplete" in directive.lower() or "chain" in directive.lower()


def test_directive_reasoning_error_redundance():
    from app.agent.metacognitive import (
        ERROR_REDUNDANCE,
        REASONING_ERROR,
        Diagnosis,
        build_writing_directive,
    )

    diag = Diagnosis(category=REASONING_ERROR, error_types=[ERROR_REDUNDANCE])
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "concise" in directive.lower() or "consolidate" in directive.lower()


def test_directive_reasoning_error_ambiguity():
    from app.agent.metacognitive import (
        ERROR_AMBIGUITY,
        REASONING_ERROR,
        Diagnosis,
        build_writing_directive,
    )

    diag = Diagnosis(category=REASONING_ERROR, error_types=[ERROR_AMBIGUITY])
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "question" in directive.lower() or "query" in directive.lower()


def test_directive_reasoning_error_with_custom_suggestion():
    from app.agent.metacognitive import REASONING_ERROR, Diagnosis, build_writing_directive

    diag = Diagnosis(
        category=REASONING_ERROR,
        error_types=[],
        suggestion="Focus on the main hypothesis only.",
    )
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "main hypothesis" in directive.lower()


def test_directive_insufficient_has_retrieval_nudge():
    from app.agent.metacognitive import INSUFFICIENT, Diagnosis, build_writing_directive

    diag = Diagnosis(category=INSUFFICIENT)
    directive = build_writing_directive(diag)
    assert directive is not None
    assert "evidence" in directive.lower() or "retrieved" in directive.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Convergence detection (Paper §6.5)
# ──────────────────────────────────────────────────────────────────────────────

def test_convergence_detects_identical_answers():
    from app.agent.metacognitive import check_convergence

    answer = "Transformers use self-attention for sequence modeling."
    assert check_convergence(answer, answer, threshold=0.85) is True


def test_convergence_detects_similar_answers():
    from app.agent.metacognitive import check_convergence

    a1 = "Transformers use self-attention for sequence modeling in NLP tasks."
    a2 = "Transformers use self-attention for sequence modeling in NLP applications."
    assert check_convergence(a1, a2, threshold=0.7) is True


def test_convergence_allows_different_answers():
    from app.agent.metacognitive import check_convergence

    a1 = "Transformers use self-attention for sequence modeling."
    a2 = "BERT is a bidirectional encoder trained with masked language modeling."
    assert check_convergence(a1, a2, threshold=0.85) is False


def test_convergence_empty_previous():
    from app.agent.metacognitive import check_convergence

    assert check_convergence("", "any answer", threshold=0.85) is False


# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis dataclass
# ──────────────────────────────────────────────────────────────────────────────

def test_diagnosis_defaults():
    from app.agent.metacognitive import SATISFACTORY, Diagnosis

    diag = Diagnosis()
    assert diag.category == SATISFACTORY
    assert diag.error_types == []
    assert diag.internal_sufficient is True
    assert diag.external_sufficient is True
    assert diag.suggested_query is None
    assert diag.suggestion is None


# ──────────────────────────────────────────────────────────────────────────────
# Graph integration — should_remediate edge function
# ──────────────────────────────────────────────────────────────────────────────

def test_should_remediate_returns_end_for_satisfactory():
    from langgraph.graph import END

    from app.agent.graph import should_remediate

    state = {
        "diagnosis": "satisfactory",
        "abstained": False,
        "metacognitive_round": 0,
        "previous_answer": "",
        "answer": "Some answer.",
    }
    assert should_remediate(state) == END


def test_should_remediate_triggers_for_reasoning_error():
    from app.agent.graph import should_remediate

    state = {
        "diagnosis": "reasoning_error",
        "abstained": False,
        "metacognitive_round": 0,
        "previous_answer": "",
        "answer": "Some answer.",
    }
    assert should_remediate(state) == "remediate"


def test_should_remediate_respects_max_rounds(monkeypatch):
    from langgraph.graph import END

    from app.agent.graph import should_remediate
    import app.config as cfg_mod

    monkeypatch.setattr(cfg_mod.settings, "max_metacognitive_rounds", 2)

    state = {
        "diagnosis": "reasoning_error",
        "abstained": False,
        "metacognitive_round": 2,
        "previous_answer": "",
        "answer": "Some answer.",
    }
    assert should_remediate(state) == END


def test_should_remediate_stops_on_convergence():
    from langgraph.graph import END

    from app.agent.graph import should_remediate

    answer = "Transformers use self-attention for sequence modeling."
    state = {
        "diagnosis": "reasoning_error",
        "abstained": False,
        "metacognitive_round": 1,
        "previous_answer": answer,
        "answer": answer,  # identical = converged
    }
    assert should_remediate(state) == END


def test_should_remediate_returns_end_for_abstained():
    from langgraph.graph import END

    from app.agent.graph import should_remediate

    state = {
        "diagnosis": "insufficient_knowledge",
        "abstained": True,
        "metacognitive_round": 0,
        "previous_answer": "",
        "answer": "Abstained.",
    }
    assert should_remediate(state) == END
