import pytest
from evaluation.metrics import compute_parallax_metrics


def test_compute_metrics_perfect():
    y_true = ["SUPPORTED_CURRENT", "OUTDATED", "CONTRADICTED", "UNSUPPORTED"]
    y_pred = ["SUPPORTED_CURRENT", "OUTDATED", "CONTRADICTED", "UNSUPPORTED"]
    metrics = compute_parallax_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["fsr_outdated"] == 0.0


def test_compute_metrics_false_support():
    # 2 Outdated claims falsely predicted as Supported-Current
    y_true = ["OUTDATED", "OUTDATED", "SUPPORTED_CURRENT"]
    y_pred = ["SUPPORTED_CURRENT", "SUPPORTED_CURRENT", "SUPPORTED_CURRENT"]
    metrics = compute_parallax_metrics(y_true, y_pred)
    assert metrics["total_outdated_count"] == 2
    assert metrics["false_supported_outdated_count"] == 2
    assert metrics["fsr_outdated"] == 1.0  # 100% false support rate!
