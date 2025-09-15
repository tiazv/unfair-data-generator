import numpy as np

from unfair_data_generator.unfair_classification import make_unfair_classification


SEED = 42
TOLERANCE_NORMAL = 0.05
TOLERANCE_EXTREME = 0.08


def _get_group_proportions(y, z, n_groups):
    proportions = []
    for group_id in range(n_groups):
        group_mask = (z == group_id)
        if np.sum(group_mask) > 0:
            positive_prop = np.mean(y[group_mask])
            proportions.append(positive_prop)
        else:
            proportions.append(0.0)
    return proportions


def _assert_proportions_close(actual_props, target_props, tolerance):
    for i, (actual, target) in enumerate(zip(actual_props, target_props)):
        assert abs(actual - target) < tolerance, (
            f"Group {i} proportion {actual:.3f} too far from target {target}"
        )


def test_weights_uniform_distribution():
    X, y, z = make_unfair_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_repeated=2,
        group_params={
            "Sunny": {"weights": [0.7, 0.3], "class_sep": 1},
            "Cloudy": {"weights": [0.7, 0.3], "class_sep": 1}, 
            "Rainy": {"weights": [0.7, 0.3], "class_sep": 1}
        },
        random_state=SEED
    )

    actual_props = _get_group_proportions(y, z, 3)
    target_props = [0.3, 0.3, 0.3]
    
    _assert_proportions_close(actual_props, target_props, TOLERANCE_NORMAL)


def test_weights_different_distribution():
    X, y, z = make_unfair_classification(
        n_samples=1200,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        n_repeated=1,
        group_params={
            "Sunny": {"weights": [0.8, 0.2], "class_sep": 1.5},
            "Cloudy": {"weights": [0.5, 0.5], "class_sep": 1.0},
            "Rainy": {"weights": [0.3, 0.7], "class_sep": 0.8}
        },
        random_state=123
    )

    actual_props = _get_group_proportions(y, z, 3)
    target_props = [0.2, 0.5, 0.7]
    
    _assert_proportions_close(actual_props, target_props, TOLERANCE_EXTREME)


def test_weights_extreme_distribution():
    X, y, z = make_unfair_classification(
        n_samples=800,
        n_features=6,
        n_informative=3,
        n_redundant=2,
        n_repeated=1,
        group_params={
            "Sunny": {"weights": [0.9, 0.1], "class_sep": 2.0},
            "Cloudy": {"weights": [0.1, 0.9], "class_sep": 1.5}
        },
        random_state=456
    )

    actual_props = _get_group_proportions(y, z, 2)
    target_props = [0.1, 0.9]
    
    _assert_proportions_close(actual_props, target_props, 0.1)


def test_weights_balanced_distribution():
    X, y, z = make_unfair_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        n_redundant=4,
        n_repeated=2,
        group_params={
            "Sunny": {"weights": [0.5, 0.5], "class_sep": 1.0},
            "Cloudy": {"weights": [0.5, 0.5], "class_sep": 1.0},
            "Rainy": {"weights": [0.5, 0.5], "class_sep": 1.0},
            "Windy": {"weights": [0.5, 0.5], "class_sep": 1.0}
        },
        random_state=789
    )

    actual_props = _get_group_proportions(y, z, 4)
    target_props = [0.5, 0.5, 0.5, 0.5]
    
    _assert_proportions_close(actual_props, target_props, TOLERANCE_EXTREME)