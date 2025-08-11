import numpy as np
import numpy.testing as npt

from unfair_data_generator.unfair_classification import make_unfair_classification


SEED = 42
N_SAMPLES = 500
N_FEATURES = 10
N_GROUPS = 3
N_INFORMATIVE = 3


def _generate_dataset(append_sensitive, return_centroids):
    return make_unfair_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=2,
        n_leaky=0,
        random_state=SEED,
        n_sensitive_groups=N_GROUPS,
        append_sensitive_to_X=append_sensitive,
        return_sensitive_group_centroids=return_centroids,
    )


def test_returns_X_y_Z():
    # When append_sensitive_to_X=False: expect (X, y, Z)
    X, y, Z = _generate_dataset(
        append_sensitive=False, return_centroids=False
    )

    # Shapes: X has exactly N_FEATURES columns; y and Z are 1-D length N_SAMPLES
    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert y.shape == (N_SAMPLES,)
    assert Z.shape == (N_SAMPLES,)

    # Labels and group indicators are integers in valid ranges
    assert set(np.unique(y)).issubset({0, 1})
    assert np.min(Z) >= 0 and np.max(Z) < N_GROUPS
    assert np.issubdtype(y.dtype, np.integer)
    assert np.issubdtype(Z.dtype, np.integer)


def test_returns_X_y():
    # Appending Z to X should place Z as the last column
    
    # Without appending sensitive to X
    X, y, Z = _generate_dataset(
        append_sensitive=False, return_centroids=False
    )

    # With sensitive appended to the last column of X
    X_with_sensitive, y_with_sensitive = _generate_dataset(
        append_sensitive=True, return_centroids=False
    )

    # Shapes: appended X has one extra column for Z
    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert X_with_sensitive.shape == (N_SAMPLES, N_FEATURES + 1)
    assert y.shape == (N_SAMPLES,) and y_with_sensitive.shape == (N_SAMPLES,)

    # Consistency between configurations
    npt.assert_array_equal(y, y_with_sensitive)
    npt.assert_array_equal(X_with_sensitive[:, :-1], X)
    npt.assert_array_equal(X_with_sensitive[:, -1].astype(int), Z)
    # Last column should be valid group IDs
    assert np.min(X_with_sensitive[:, -1]) >= 0 and np.max(X_with_sensitive[:, -1]) < N_GROUPS


def test_returns_X_y_Z_centroids():
    # Requesting centroids changes the return values depending on append flag

    # With separate Z and centroids
    X, y, Z, centroids = _generate_dataset(append_sensitive=False, return_centroids=True)
    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert y.shape == (N_SAMPLES,)
    assert Z.shape == (N_SAMPLES,)
    assert isinstance(centroids, dict) and len(centroids) == N_GROUPS
    
    # Centroids should have correct shape (clusters x N_INFORMATIVE)
    for group, centroid_array in centroids.items():
        assert centroid_array.shape[1] == N_INFORMATIVE


def test_returns_X_y_centroids():
    # With sensitive appended and centroids

    X, y, centroids = _generate_dataset(append_sensitive=True, return_centroids=True)
    assert X.shape == (N_SAMPLES, N_FEATURES + 1)
    assert y.shape == (N_SAMPLES,)
    assert isinstance(centroids, dict) and len(centroids) == N_GROUPS
    
    # Centroids should have correct shape (clusters x N_INFORMATIVE)
    for group, centroid_array in centroids.items():
        assert centroid_array.shape[1] == N_INFORMATIVE
