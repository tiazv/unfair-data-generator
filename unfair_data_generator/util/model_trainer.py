from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

try:
    from .helpers import get_group_name
except ImportError:
    from unfair_data_generator.util.helpers import get_group_name

import numpy as np


def evaluate_fairness_by_group(y_true, y_pred, groups, sensitive_groups):
    """
    Evaluate fairness metrics for each sensitive group.

    This function calculates metrics such as accuracy, True Positive Rate (TPR), False Positive Rate (FPR),
    and confusion matrices for each sensitive group in the dataset.

    Parameters:
        y_true (ndarray): Ground truth (true) target values.
        y_pred (ndarray): Predicted labels from the classifier.
        groups (ndarray): Sensitive group assignments for each sample. Each element indicates which sensitive group the corresponding sample belongs to.
        sensitive_groups (list): List of unique sensitive group identifiers present in the dataset. For example, ["Sunny", "Cloudy", "Rainy"].

    Returns:
        dict: Dictionary containing fairness metrics for each sensitive group. Each key is formatted as "Group {group}" and maps to a dictionary containing:

            - 'Confusion Matrix' : list of lists
                2x2 confusion matrix as nested lists in format [[TN, FP], [FN, TP]].

            - 'Accuracy' : float
                Classification accuracy :math:`\\frac{TP + TN}{TP + TN + FP + FN}`.

            - 'True Positive Rate (TPR)' : float
                Sensitivity or recall, calculated as :math:`\\frac{TP}{TP + FN}`. Returns 0 if no positive samples exist in the group.

            - 'False Positive Rate (FPR)' : float
                Calculated as :math:`\\frac{FP}{FP + TN}`. Returns 0 if no negative samples exist in the group.
    """
    results = {}
    for group in sensitive_groups:
        mask = groups == group
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        tn, fp, fn, tp = cm.ravel()
        results[f"Group {group}"] = {
            "Confusion Matrix": cm.tolist(),
            "Accuracy": (tp + tn) / (tp + tn + fp + fn),
            "True Positive Rate (TPR)": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "False Positive Rate (FPR)": fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
    return results


def train_and_evaluate_model_with_classifier(X, y, Z):
    """
    Train a Random Forest classifier and evaluate performance and fairness across sensitive groups.

    This function separates sensitive features from the dataset, trains a Random Forest classifier,
    and calculates fairness metrics for each sensitive group.

    Parameters:
        X (ndarray): Training data feature matrix. Contains the input features used for classification, excluding sensitive attributes.
        y (ndarray): Target values for classification. Binary labels where 0 represents the negative class and 1 represents the positive class.
        Z (ndarray): Sensitive group information for each sample. Each element indicates which sensitive/protected group the corresponding sample belongs to.

    Returns:
        dict: Comprehensive fairness metrics organized by sensitive group. Each key represents a group name (determined by `get_group_name` function) and maps to a dictionary containing:

            - 'Accuracy' : float
                Classification accuracy for the group.

            - 'True Positive Rate (TPR)' : float
                Sensitivity/recall for the group, calculated as :math:`\\frac{TP}{TP+FN}`.

            - 'False Positive Rate (FPR)' : float
                Calculated as :math:`\\frac{FP}{FP+TN}` for the group.

            - 'Samples in Positive class' : int
                Number of samples predicted as positive class (TP + FP).

            - 'Samples in Negative class' : int
                Number of samples predicted as negative class (TN + FN).

            - 'Confusion Matrix' : dict
                Detailed breakdown with keys 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)'.
    """

    sensitive_groups_data = Z
    features = X

    # Split the data (sensitive_groups_data is aligned with the rows of X and y)
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        features, y, sensitive_groups_data, test_size=0.3, random_state=42, stratify=y
    )

    # Train the model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate fairness metrics for each group
    metrics = {}
    for group in np.unique(sensitive_test):
        mask = sensitive_test == group
        group_y_test = y_test[mask]
        group_y_pred = y_pred[mask]

        # Get group name using the sensitive data we already determined
        unique_groups = np.unique(sensitive_groups_data)
        group_name = get_group_name(unique_groups, group)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            group_y_test, group_y_pred, labels=[0, 1]).ravel()

        group_metrics = {
            "Accuracy": accuracy_score(group_y_test, group_y_pred),
            "True Positive Rate (TPR)": recall_score(group_y_test, group_y_pred, pos_label=1),
            "False Positive Rate (FPR)": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "Samples in Positive class": fp + tp,
            "Samples in Negative class": fn + tn,
            "Confusion Matrix": {
                "True Negative (TN)": tn,
                "False Positive (FP)": fp,
                "False Negative (FN)": fn,
                "True Positive (TP)": tp,
            }
        }
        metrics[group_name] = group_metrics

    return metrics
