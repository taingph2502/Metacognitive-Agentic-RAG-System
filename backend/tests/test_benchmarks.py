"""Tests for the benchmark evaluation metrics (Paper §5.1)."""


def test_exact_match_identical():
    from benchmarks.metrics import exact_match

    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive():
    from benchmarks.metrics import exact_match

    assert exact_match("paris", "Paris") == 1.0


def test_exact_match_article_removal():
    from benchmarks.metrics import exact_match

    assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0


def test_exact_match_punctuation():
    from benchmarks.metrics import exact_match

    assert exact_match("hello, world!", "hello world") == 1.0


def test_exact_match_different():
    from benchmarks.metrics import exact_match

    assert exact_match("Paris", "London") == 0.0


def test_token_f1_perfect():
    from benchmarks.metrics import token_f1

    result = token_f1("the cat sat", "the cat sat")
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_token_f1_partial():
    from benchmarks.metrics import token_f1

    result = token_f1("the cat sat on the mat", "the cat sat")
    # prediction has extra tokens → precision < 1, recall = 1
    assert result["recall"] == 1.0
    assert result["precision"] < 1.0
    assert 0 < result["f1"] < 1.0


def test_token_f1_no_overlap():
    from benchmarks.metrics import token_f1

    result = token_f1("hello world", "foo bar")
    assert result["f1"] == 0.0


def test_compute_metrics_combined():
    from benchmarks.metrics import compute_metrics

    m = compute_metrics("Paris", "Paris")
    assert m["exact_match"] == 1.0
    assert m["f1"] == 1.0


def test_aggregate_metrics():
    from benchmarks.metrics import aggregate_metrics

    results = [
        {"exact_match": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0},
        {"exact_match": 0.0, "f1": 0.5, "precision": 0.5, "recall": 0.5},
    ]
    agg = aggregate_metrics(results)
    assert agg["exact_match"] == 0.5
    assert agg["f1"] == 0.75
    assert agg["num_examples"] == 2


def test_aggregate_metrics_empty():
    from benchmarks.metrics import aggregate_metrics

    agg = aggregate_metrics([])
    assert agg["exact_match"] == 0.0
    assert agg["f1"] == 0.0
