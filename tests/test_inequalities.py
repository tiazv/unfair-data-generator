import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)

from unfair_data_generator.unfair_classification import make_unfair_classification


def _train_and_predict(X, y, z):
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    return y_test, y_pred_test, z_test


def _group_metric_difference(metric_frame):
    # Return the range (max - min) across sensitive groups for a metric.
    values = metric_frame.by_group.values
    return float(np.max(values) - np.min(values))


def compute_group_metric_range_for_metric(y_true, y_pred, z, metric):
    # Compute the group range for a single metric function.
    mf = MetricFrame(
        metrics=metric, y_true=y_true, y_pred=y_pred, sensitive_features=z
    )
    return _group_metric_difference(mf)


def compute_all_metric_ranges(y_true, y_pred, z):
    #Compute range across groups for all metrics relevant to these tests.
    return {
        "accuracy": compute_group_metric_range_for_metric(
            y_true, y_pred, z, accuracy_score
        ),
        "selection_rate": compute_group_metric_range_for_metric(
            y_true, y_pred, z, selection_rate
        ),
        "tpr": compute_group_metric_range_for_metric(
            y_true, y_pred, z, true_positive_rate
        ),
        "fpr": compute_group_metric_range_for_metric(
            y_true, y_pred, z, false_positive_rate
        ),
    }


def test_equal_quality_violation():
    # Equal quality: expect accuracy to vary across sensitive groups.
    X, y, z = make_unfair_classification(
        n_samples=4000,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
        n_sensitive_groups=2,
        fairness_type="Equal quality",
    )

    y_test, y_pred, z_test = _train_and_predict(X, y, z)

    ranges = compute_all_metric_ranges(y_test, y_pred, z_test)

    assert ranges["accuracy"] > 0.05, f"ranges={ranges}"


def test_demographic_parity_violation():
    # Demographic parity: expect selection rate to vary across groups.
    X, y, z = make_unfair_classification(
        n_samples=4000,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
        n_sensitive_groups=3,
        fairness_type="Demographic parity",
    )

    y_test, y_pred, z_test = _train_and_predict(X, y, z)

    ranges = compute_all_metric_ranges(y_test, y_pred, z_test)

    assert ranges["selection_rate"] > 0.05, f"ranges={ranges}"


def test_equal_opportunity_violation():
    # Equal opportunity: expect TPR to vary across groups.
    X, y, z = make_unfair_classification(
        n_samples=4000,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
        n_sensitive_groups=3,
        fairness_type="Equal opportunity",
    )

    y_test, y_pred, z_test = _train_and_predict(X, y, z)

    ranges = compute_all_metric_ranges(y_test, y_pred, z_test)

    assert ranges["tpr"] > 0.05, f"ranges={ranges}"


def test_equalized_odds_violation():
    # Equalized odds: expect both TPR and FPR to vary across groups.
    X, y, z = make_unfair_classification(
        n_samples=4000,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
        n_sensitive_groups=4,
        fairness_type="Equalized odds",
    )

    y_test, y_pred, z_test = _train_and_predict(X, y, z)

    ranges = compute_all_metric_ranges(y_test, y_pred, z_test)

    # Expect both TPR and FPR to vary across groups
    assert ranges["tpr"] > 0.05, f"ranges={ranges}"
    assert ranges["fpr"] > 0.04, f"ranges={ranges}"
