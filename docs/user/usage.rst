Getting started
===============

This section demonstrates the usage of ``unfair-data-generator``.

Installation
------------

To install ``unfair-data-generator`` with pip, run the following command:

..  code:: bash

    pip install unfair-data-generator

Usage
-----

The following example demonstrates how to generate a biased dataset and evaluate fairness using ``unfair-data-generator``.

..  code:: python

    from unfair_data_generator.unfair_classification import make_unfair_classification
    from unfair_data_generator.util.helpers import get_params_for_certain_equality_type
    from unfair_data_generator.util.model_trainer import train_and_evaluate_model_with_classifier
    from unfair_data_generator.util.visualizer import (
        visualize_TPR_FPR_metrics, 
        visualize_accuracy, 
        visualize_groups_separately
    )

    # Configure dataset parameters
    fairness_type = "Demographic parity"
    n_sensitive_groups = 3

    # Generate group-specific parameters for fairness violation
    group_params = get_params_for_certain_equality_type(fairness_type, n_sensitive_groups)

    # Generate biased dataset
    X, y, Z, centroids = make_unfair_classification(
        n_samples=5000,
        n_features=10,
        n_informative=3,
        n_leaky=2,
        random_state=42
        group_params=group_params,
        return_sensitive_group_centroids=True,
    )

    # Visualize group-specific patterns
    visualize_groups_separately(X, y, Z)
    visualize_group_classes(X, y, Z, centroids)

    # Train model and evaluate fairness
    metrics = train_and_evaluate_model_with_classifier(X, y, Z)

    # Visualize fairness metrics
    title = f"{fairness_type} with {n_sensitive_groups} sensitive groups"
    visualize_TPR_FPR_metrics(metrics, title)
    visualize_accuracy(metrics, title)